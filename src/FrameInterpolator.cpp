#include "FrameInterpolator.h"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

static const float GAUSSIAN_5TAP[5] = {
    1.0f / 16.0f,
    4.0f / 16.0f,
    6.0f / 16.0f,
    4.0f / 16.0f,
    1.0f / 16.0f
};

Image FrameInterpolator::extractDetailLayer(const Image& img) {
    Image base = img.convolve5(GAUSSIAN_5TAP);
    Image detail(img.width, img.height, img.channels);
    size_t total = static_cast<size_t>(img.width) * img.height * img.channels;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(total); ++i) {
        detail.data[i] = img.data[i] - base.data[i];
    }
    return detail;
}

Image FrameInterpolator::interpolate(
    const Image& frame0,
    const Image& frame1,
    float t,
    FlowField* outForwardFlow,
    FlowField* outBackwardFlow
) const {
    if (frame0.empty() || frame1.empty()) {
        std::cerr << "Error: empty input frames provided to interpolator\n";
        return Image();
    }

    t = std::clamp(t, 0.0f, 1.0f);
    if (t <= 0.001f) return frame0;
    if (t >= 0.999f) return frame1;

    int W = frame0.width;
    int H = frame0.height;
    int C = frame0.channels;

    ZNCCMatcher matcher(params.matcherParams);

    // 1. Forward motion: frame0 -> frame1
    FlowField forwardFlow = matcher.estimateFlow(frame0, frame1);

    // 2. Backward motion: frame1 -> frame0
    FlowField backwardFlow;
    if (params.bidirectional) {
        backwardFlow = matcher.estimateFlow(frame1, frame0);
    } else {
        backwardFlow = FlowField(W, H);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const MotionVector& fv = forwardFlow.get(x, y);
                backwardFlow.set(x, y, MotionVector(-fv.vx, -fv.vy, fv.score));
            }
        }
    }

    if (outForwardFlow) *outForwardFlow = forwardFlow;
    if (outBackwardFlow) *outBackwardFlow = backwardFlow;

    // 3. Construct intermediate motion field at time t via splatting. Both
    // fields contribute in proportion to how close time t is to their own
    // frame; nothing else weights them.
    std::vector<float> accumVx(static_cast<size_t>(W) * H, 0.0f);
    std::vector<float> accumVy(static_cast<size_t>(W) * H, 0.0f);
    std::vector<float> accumWeight(static_cast<size_t>(W) * H, 0.0f);

    // Splat forward flow from frame0 into intermediate frame t
    for (int y0 = 0; y0 < H; ++y0) {
        for (int x0 = 0; x0 < W; ++x0) {
            const MotionVector& mv = forwardFlow.get(x0, y0);
            float xt = static_cast<float>(x0) + t * mv.vx;
            float yt = static_cast<float>(y0) + t * mv.vy;
            float baseW = 1.0f - t;

            int ix = static_cast<int>(std::floor(xt));
            int iy = static_cast<int>(std::floor(yt));
            float fx = xt - static_cast<float>(ix);
            float fy = yt - static_cast<float>(iy);

            for (int dy = 0; dy <= 1; ++dy) {
                int qy = iy + dy;
                if (qy < 0 || qy >= H) continue;
                float wy = dy ? fy : (1.0f - fy);

                for (int dx = 0; dx <= 1; ++dx) {
                    int qx = ix + dx;
                    if (qx < 0 || qx >= W) continue;
                    float wx = dx ? fx : (1.0f - fx);

                    float w = baseW * wx * wy;
                    size_t idx = static_cast<size_t>(qy) * W + qx;
                    accumVx[idx] += mv.vx * w;
                    accumVy[idx] += mv.vy * w;
                    accumWeight[idx] += w;
                }
            }
        }
    }

    // Splat backward flow from frame1 into intermediate frame t
    for (int y1 = 0; y1 < H; ++y1) {
        for (int x1 = 0; x1 < W; ++x1) {
            const MotionVector& mv = backwardFlow.get(x1, y1);
            float fvx = -mv.vx;
            float fvy = -mv.vy;
            float xt = static_cast<float>(x1) + (1.0f - t) * mv.vx;
            float yt = static_cast<float>(y1) + (1.0f - t) * mv.vy;
            float baseW = t;

            int ix = static_cast<int>(std::floor(xt));
            int iy = static_cast<int>(std::floor(yt));
            float fx = xt - static_cast<float>(ix);
            float fy = yt - static_cast<float>(iy);

            for (int dy = 0; dy <= 1; ++dy) {
                int qy = iy + dy;
                if (qy < 0 || qy >= H) continue;
                float wy = dy ? fy : (1.0f - fy);

                for (int dx = 0; dx <= 1; ++dx) {
                    int qx = ix + dx;
                    if (qx < 0 || qx >= W) continue;
                    float wx = dx ? fx : (1.0f - fx);

                    float w = baseW * wx * wy;
                    size_t idx = static_cast<size_t>(qy) * W + qx;
                    accumVx[idx] += fvx * w;
                    accumVy[idx] += fvy * w;
                    accumWeight[idx] += w;
                }
            }
        }
    }

    // Artifact Refinement: Extract high-frequency detail layers if enabled
    Image detail0, detail1;
    if (params.enableDetailRestoration) {
        detail0 = extractDetailLayer(frame0);
        detail1 = extractDetailLayer(frame1);
    }

    // 4. Synthesize target intermediate frame
    Image result(W, H, C);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            size_t idx = static_cast<size_t>(y) * W + x;
            float totalW = accumWeight[idx];

            float vx = 0.0f;
            float vy = 0.0f;

            if (totalW > 1e-4f) {
                vx = accumVx[idx] / totalW;
                vy = accumVy[idx] / totalW;
            } else {
                // Nothing splatted here: keep the frame0 vector.
                const MotionVector& fv = forwardFlow.get(x, y);
                vx = fv.vx;
                vy = fv.vy;
            }

            // Continuous sample locations in frame0 and frame1
            float src0X = static_cast<float>(x) - t * vx;
            float src0Y = static_cast<float>(y) - t * vy;

            float src1X = static_cast<float>(x) + (1.0f - t) * vx;
            float src1Y = static_cast<float>(y) + (1.0f - t) * vy;

            // Sample RGB values from both frames
            float val0[4] = { 0.0f };
            float val1[4] = { 0.0f };
            for (int c = 0; c < C; ++c) {
                val0[c] = frame0.sampleBilinear(src0X, src0Y, c);
                val1[c] = frame1.sampleBilinear(src1X, src1Y, c);
            }

            // Blend weights are the temporal position and nothing else.
            const float w0 = 1.0f - t;
            const float w1 = t;

            float blended[4] = { 0.0f };
            for (int c = 0; c < C; ++c) {
                blended[c] = w0 * val0[c] + w1 * val1[c];
            }

            // Artifact Refinement 1: High-Frequency Detail Restoration (Laplacian Detail Transfer)
            if (params.enableDetailRestoration) {
                for (int c = 0; c < C; ++c) {
                    float d0 = detail0.sampleBilinear(src0X, src0Y, c);
                    float d1 = detail1.sampleBilinear(src1X, src1Y, c);
                    blended[c] += params.detailStrength * (w0 * d0 + w1 * d1);
                }
            }

            // Artifact Refinement 2: Local Color Bounding Box Clamping
            if (params.enableColorClamping) {
                int s0x = static_cast<int>(std::round(src0X));
                int s0y = static_cast<int>(std::round(src0Y));
                int s1x = static_cast<int>(std::round(src1X));
                int s1y = static_cast<int>(std::round(src1Y));

                for (int c = 0; c < C; ++c) {
                    float minC = 1e30f;
                    float maxC = -1e30f;

                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            float p0 = frame0.get(s0x + dx, s0y + dy, c);
                            float p1 = frame1.get(s1x + dx, s1y + dy, c);
                            minC = std::min({ minC, p0, p1 });
                            maxC = std::max({ maxC, p0, p1 });
                        }
                    }

                    // Allow margin of +/- 3 intensity levels
                    minC = std::max(0.0f, minC - 3.0f);
                    maxC = std::min(255.0f, maxC + 3.0f);
                    blended[c] = std::clamp(blended[c], minC, maxC);
                }
            }

            for (int c = 0; c < C; ++c) {
                result.set(x, y, c, std::clamp(blended[c], 0.0f, 255.0f));
            }
        }
    }

    return result;
}
