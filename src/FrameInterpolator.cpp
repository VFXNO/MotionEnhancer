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

Image FrameInterpolator::computeOcclusionMask(
    const FlowField& forwardFlow,
    const FlowField& backwardFlow,
    float threshold
) {
    int W = forwardFlow.width;
    int H = forwardFlow.height;
    Image mask(W, H, 1, 1.0f);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const MotionVector& fv = forwardFlow.get(x, y);
            float targetX = static_cast<float>(x) + fv.vx;
            float targetY = static_cast<float>(y) + fv.vy;

            MotionVector bv = backwardFlow.sampleBilinear(targetX, targetY);

            // Forward-backward consistency error: |fv + bv|
            float errX = fv.vx + bv.vx;
            float errY = fv.vy + bv.vy;
            float consistencyErr = std::sqrt(errX * errX + errY * errY);

            if (consistencyErr <= threshold) {
                mask.set(x, y, 0, 1.0f);
            } else {
                float conf = std::exp(-(consistencyErr - threshold) / 2.0f);
                mask.set(x, y, 0, std::clamp(conf, 0.0f, 1.0f));
            }
        }
    }

    return mask;
}

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

void FrameInterpolator::inpaintMotionHoles(
    std::vector<float>& accumVx,
    std::vector<float>& accumVy,
    std::vector<float>& accumWeight,
    std::vector<float>& accumConf0,
    std::vector<float>& accumConf1,
    const FlowField& fallbackFlow,
    int W, int H,
    int searchRadius
) {
    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            size_t idx = static_cast<size_t>(y) * W + x;
            if (accumWeight[idx] >= 1e-3f) continue; // Not a hole

            float fillVx = 0.0f, fillVy = 0.0f;
            float fillC0 = 0.0f, fillC1 = 0.0f;
            float fillW = 0.0f;

            for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
                int ny = y + dy;
                if (ny < 0 || ny >= H) continue;

                for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= W) continue;

                    size_t nidx = static_cast<size_t>(ny) * W + nx;
                    float nw = accumWeight[nidx];
                    if (nw < 1e-3f) continue;

                    float distSq = static_cast<float>(dx * dx + dy * dy);
                    float spatW = std::exp(-distSq / 8.0f);

                    float nvx = accumVx[nidx] / nw;
                    float nvy = accumVy[nidx] / nw;
                    float mag = std::sqrt(nvx * nvx + nvy * nvy);

                    // Prefer background vectors (smaller motion) in disocclusion zones
                    float w = spatW * (1.0f / (1.0f + 0.08f * mag));
                    fillVx += nvx * w;
                    fillVy += nvy * w;
                    fillC0 += (accumConf0[nidx] / nw) * w;
                    fillC1 += (accumConf1[nidx] / nw) * w;
                    fillW += w;
                }
            }

            if (fillW > 1e-4f) {
                accumVx[idx] = fillVx / fillW;
                accumVy[idx] = fillVy / fillW;
                accumConf0[idx] = fillC0 / fillW;
                accumConf1[idx] = fillC1 / fillW;
                accumWeight[idx] = 1.0f;
            } else {
                const MotionVector& fv = fallbackFlow.get(x, y);
                accumVx[idx] = fv.vx;
                accumVy[idx] = fv.vy;
                accumConf0[idx] = 0.5f;
                accumConf1[idx] = 0.5f;
                accumWeight[idx] = 1.0f;
            }
        }
    }
}

Image FrameInterpolator::interpolate(
    const Image& frame0,
    const Image& frame1,
    float t,
    FlowField* outForwardFlow,
    FlowField* outBackwardFlow,
    Image* outOcclusionMask
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

    // 3. Occlusion masks via consistency checking
    Image occ0 = computeOcclusionMask(forwardFlow, backwardFlow, params.occlusionThreshold);
    Image occ1 = computeOcclusionMask(backwardFlow, forwardFlow, params.occlusionThreshold);

    if (outForwardFlow) *outForwardFlow = forwardFlow;
    if (outBackwardFlow) *outBackwardFlow = backwardFlow;
    if (outOcclusionMask) *outOcclusionMask = occ0;

    // 4. Construct intermediate motion field at time t via splatting
    std::vector<float> accumVx(static_cast<size_t>(W) * H, 0.0f);
    std::vector<float> accumVy(static_cast<size_t>(W) * H, 0.0f);
    std::vector<float> accumWeight(static_cast<size_t>(W) * H, 0.0f);
    std::vector<float> accumConf0(static_cast<size_t>(W) * H, 0.0f);
    std::vector<float> accumW0(static_cast<size_t>(W) * H, 0.0f);
    std::vector<float> accumConf1(static_cast<size_t>(W) * H, 0.0f);
    std::vector<float> accumW1(static_cast<size_t>(W) * H, 0.0f);

    // Splat forward flow from frame0 into intermediate frame t
    for (int y0 = 0; y0 < H; ++y0) {
        for (int x0 = 0; x0 < W; ++x0) {
            const MotionVector& mv = forwardFlow.get(x0, y0);
            float xt = static_cast<float>(x0) + t * mv.vx;
            float yt = static_cast<float>(y0) + t * mv.vy;
            float c0 = occ0.get(x0, y0, 0);
            float baseW = (1.0f - t) * (0.2f + 0.8f * c0);

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
                    accumConf0[idx] += c0 * w;
                    accumW0[idx] += w;
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
            float c1 = occ1.get(x1, y1, 0);
            float baseW = t * (0.2f + 0.8f * c1);

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
                    accumConf1[idx] += c1 * w;
                    accumW1[idx] += w;
                    accumWeight[idx] += w;
                }
            }
        }
    }

    // Artifact Refinement: Inpaint disocclusion holes in the motion field
    if (params.enableHoleInpainting) {
        inpaintMotionHoles(accumVx, accumVy, accumWeight, accumConf0, accumConf1, forwardFlow, W, H, 3);
    }

    // Artifact Refinement: Extract high-frequency detail layers if enabled
    Image detail0, detail1;
    if (params.enableDetailRestoration) {
        detail0 = extractDetailLayer(frame0);
        detail1 = extractDetailLayer(frame1);
    }

    // 5. Synthesize target intermediate frame with artifact refinement
    Image result(W, H, C);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            size_t idx = static_cast<size_t>(y) * W + x;
            float totalW = accumWeight[idx];

            float vx = 0.0f;
            float vy = 0.0f;
            float conf0 = 1.0f;
            float conf1 = 1.0f;

            if (totalW > 1e-4f) {
                vx = accumVx[idx] / totalW;
                vy = accumVy[idx] / totalW;
                conf0 = (accumW0[idx] > 1e-4f) ? (accumConf0[idx] / accumW0[idx]) : occ0.get(x, y, 0);
                conf1 = (accumW1[idx] > 1e-4f) ? (accumConf1[idx] / accumW1[idx]) : occ1.get(x, y, 0);
            } else {
                const MotionVector& fv = forwardFlow.get(x, y);
                vx = fv.vx;
                vy = fv.vy;
                conf0 = occ0.get(x, y, 0);
                conf1 = occ1.get(x, y, 0);
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

            // Initial blend weights based on temporal position and occlusion confidence
            float w0 = (1.0f - t) * (std::clamp(conf1, 0.02f, 1.0f));
            float w1 = t * (std::clamp(conf0, 0.02f, 1.0f));

            // Artifact Refinement 1: Photometric-Aware Occlusion Gating
            if (params.enablePhotometricGating) {
                float colorDistSq = 0.0f;
                for (int c = 0; c < C; ++c) {
                    float diff = val0[c] - val1[c];
                    colorDistSq += diff * diff;
                }
                float colorDist = std::sqrt(colorDistSq);

                if (colorDist > params.photoGateThreshold) {
                    float excess = colorDist - params.photoGateThreshold;
                    float invSigma = 1.0f / params.photoGateSigma;
                    float suppress = std::exp(-(excess * excess * invSigma * invSigma * 0.5f));

                    // Check which source is more confident:
                    if (conf0 > conf1 + 0.10f) {
                        // Frame 0 is valid; Frame 1 is occluded -> suppress Frame 1
                        w1 *= suppress;
                    } else if (conf1 > conf0 + 0.10f) {
                        // Frame 1 is valid; Frame 0 is occluded -> suppress Frame 0
                        w0 *= suppress;
                    }
                }
            }

            float normW = w0 + w1;
            if (normW < 1e-5f) normW = 1.0f;

            float blended[4] = { 0.0f };
            for (int c = 0; c < C; ++c) {
                blended[c] = (w0 * val0[c] + w1 * val1[c]) / normW;
            }

            // Artifact Refinement 2: High-Frequency Detail Restoration (Laplacian Detail Transfer)
            if (params.enableDetailRestoration) {
                for (int c = 0; c < C; ++c) {
                    float d0 = detail0.sampleBilinear(src0X, src0Y, c);
                    float d1 = detail1.sampleBilinear(src1X, src1Y, c);
                    float blendedDetail = (w0 * d0 + w1 * d1) / normW;
                    blended[c] += params.detailStrength * blendedDetail;
                }
            }

            // Artifact Refinement 3: Local Color Bounding Box Clamping
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
