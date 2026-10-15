#pragma once

#include "Image.h"
#include <vector>
#include <cmath>
#include <algorithm>

struct MotionVector {
    float vx = 0.0f;
    float vy = 0.0f;
    float score = -1.0f; // ZNCC score in [-1.0, 1.0]

    MotionVector() = default;
    MotionVector(float x, float y, float s = -1.0f) : vx(x), vy(y), score(s) {}

    float length() const {
        return std::sqrt(vx * vx + vy * vy);
    }
};

class FlowField {
public:
    int width = 0;
    int height = 0;
    std::vector<MotionVector> vectors;

    FlowField() = default;
    FlowField(int w, int h, MotionVector initVec = MotionVector(0.0f, 0.0f, 0.0f))
        : width(w), height(h), vectors(static_cast<size_t>(w) * h, initVec) {}

    bool empty() const { return vectors.empty() || width <= 0 || height <= 0; }

    inline int index(int x, int y) const {
        return y * width + x;
    }

    inline const MotionVector& get(int x, int y) const {
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        return vectors[index(x, y)];
    }

    inline MotionVector& at(int x, int y) {
        return vectors[index(x, y)];
    }

    inline void set(int x, int y, const MotionVector& mv) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            vectors[index(x, y)] = mv;
        }
    }

    // Bilinear interpolation of motion vectors at continuous coordinates (x, y)
    MotionVector sampleBilinear(float x, float y) const {
        if (width <= 0 || height <= 0) return MotionVector();

        x = std::clamp(x, 0.0f, static_cast<float>(width - 1));
        y = std::clamp(y, 0.0f, static_cast<float>(height - 1));

        int x0 = static_cast<int>(std::floor(x));
        int y0 = static_cast<int>(std::floor(y));
        int x1 = std::min(x0 + 1, width - 1);
        int y1 = std::min(y0 + 1, height - 1);

        float dx = x - static_cast<float>(x0);
        float dy = y - static_cast<float>(y0);

        const MotionVector& v00 = get(x0, y0);
        const MotionVector& v10 = get(x1, y0);
        const MotionVector& v01 = get(x0, y1);
        const MotionVector& v11 = get(x1, y1);

        float vx = (v00.vx * (1.0f - dx) + v10.vx * dx) * (1.0f - dy) +
                   (v01.vx * (1.0f - dx) + v11.vx * dx) * dy;
        float vy = (v00.vy * (1.0f - dx) + v10.vy * dx) * (1.0f - dy) +
                   (v01.vy * (1.0f - dx) + v11.vy * dx) * dy;
        float sc = (v00.score * (1.0f - dx) + v10.score * dx) * (1.0f - dy) +
                   (v01.score * (1.0f - dx) + v11.score * dx) * dy;

        return MotionVector(vx, vy, sc);
    }

    // Upsample flow field to target dimensions (with vector scaling factor)
    FlowField upsample(int targetW, int targetH) const {
        FlowField out(targetW, targetH);
        float scaleX = static_cast<float>(targetW) / static_cast<float>(width);
        float scaleY = static_cast<float>(targetH) / static_cast<float>(height);

        for (int y = 0; y < targetH; ++y) {
            float srcY = (static_cast<float>(y) + 0.5f) / scaleY - 0.5f;
            for (int x = 0; x < targetW; ++x) {
                float srcX = (static_cast<float>(x) + 0.5f) / scaleX - 0.5f;
                MotionVector mv = sampleBilinear(srcX, srcY);
                // Vector displacements scale with spatial dimensions
                mv.vx *= scaleX;
                mv.vy *= scaleY;
                out.set(x, y, mv);
            }
        }
        return out;
    }

    // 3x3 Vector Median Filter to eliminate motion outliers
    FlowField applyVectorMedianFilter(int radius = 1) const {
        FlowField filtered(width, height);

        #pragma omp parallel for schedule(dynamic, 16)
        for (int y = 0; y < height; ++y) {
            MotionVector neighborhood[49]; // Max radius 3
            for (int x = 0; x < width; ++x) {
                int count = 0;
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int nx = std::clamp(x + dx, 0, width - 1);
                        int ny = std::clamp(y + dy, 0, height - 1);
                        neighborhood[count++] = get(nx, ny);
                    }
                }

                // Find candidate vector with minimal sum of Euclidean distances to all other vectors
                float minTotalDist = 1e30f;
                int bestIdx = 0;

                for (int i = 0; i < count; ++i) {
                    float totalDist = 0.0f;
                    for (int j = 0; j < count; ++j) {
                        float dvx = neighborhood[i].vx - neighborhood[j].vx;
                        float dvy = neighborhood[i].vy - neighborhood[j].vy;
                        totalDist += std::sqrt(dvx * dvx + dvy * dvy);
                    }
                    if (totalDist < minTotalDist) {
                        minTotalDist = totalDist;
                        bestIdx = i;
                    }
                }

                filtered.set(x, y, neighborhood[bestIdx]);
            }
        }
        return filtered;
    }

    // Color-Guided Edge-Preserving Vector Median Filter
    FlowField applyColorGuidedMedianFilter(const Image& guideImage, int radius = 1, float sigmaColor = 25.0f) const {
        FlowField filtered(width, height);
        float invTwoSigmaSq = 1.0f / (2.0f * sigmaColor * sigmaColor);

        #pragma omp parallel for schedule(dynamic, 16)
        for (int y = 0; y < height; ++y) {
            MotionVector neighborhood[49];
            float weights[49];
            for (int x = 0; x < width; ++x) {
                int count = 0;
                float r0 = guideImage.get(x, y, 0);
                float g0 = guideImage.channels > 1 ? guideImage.get(x, y, 1) : r0;
                float b0 = guideImage.channels > 2 ? guideImage.get(x, y, 2) : r0;

                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int nx = std::clamp(x + dx, 0, width - 1);
                        int ny = std::clamp(y + dy, 0, height - 1);
                        neighborhood[count] = get(nx, ny);

                        float r1 = guideImage.get(nx, ny, 0);
                        float g1 = guideImage.channels > 1 ? guideImage.get(nx, ny, 1) : r1;
                        float b1 = guideImage.channels > 2 ? guideImage.get(nx, ny, 2) : r1;

                        float dr = r0 - r1;
                        float dg = g0 - g1;
                        float db = b0 - b1;
                        float colorDistSq = dr * dr + dg * dg + db * db;
                        weights[count] = std::exp(-colorDistSq * invTwoSigmaSq);
                        count++;
                    }
                }

                // Weighted vector median: candidate minimizing sum of weighted distances
                float minTotalDist = 1e30f;
                int bestIdx = 0;

                for (int i = 0; i < count; ++i) {
                    float totalDist = 0.0f;
                    for (int j = 0; j < count; ++j) {
                        float dvx = neighborhood[i].vx - neighborhood[j].vx;
                        float dvy = neighborhood[i].vy - neighborhood[j].vy;
                        totalDist += weights[j] * std::sqrt(dvx * dvx + dvy * dvy);
                    }
                    if (totalDist < minTotalDist) {
                        minTotalDist = totalDist;
                        bestIdx = i;
                    }
                }

                filtered.set(x, y, neighborhood[bestIdx]);
            }
        }
        return filtered;
    }

    // Visualize flow field as color-coded RGB image (Hue = Angle, Value = Magnitude)
    Image toColorImage(float maxFlow = -1.0f) const {
        Image img(width, height, 3);
        if (maxFlow <= 0.0f) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    maxFlow = std::max(maxFlow, get(x, y).length());
                }
            }
            if (maxFlow < 1.0f) maxFlow = 1.0f;
        }

        const float PI = 3.14159265358979323846f;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const MotionVector& mv = get(x, y);
                float rad = mv.length() / maxFlow;
                rad = std::clamp(rad, 0.0f, 1.0f);
                float angle = std::atan2(-mv.vy, -mv.vx) / PI; // [-1.0, 1.0]
                float fk = (angle + 1.0f) / 2.0f; // [0.0, 1.0]
                float hue = fk * 6.0f;
                int k0 = static_cast<int>(std::floor(hue)) % 6;
                float f = hue - std::floor(hue);

                float r = 0.0f, g = 0.0f, b = 0.0f;
                switch (k0) {
                    case 0: r = 1.0f; g = f; b = 0.0f; break;
                    case 1: r = 1.0f - f; g = 1.0f; b = 0.0f; break;
                    case 2: r = 0.0f; g = 1.0f; b = f; break;
                    case 3: r = 0.0f; g = 1.0f - f; b = 1.0f; break;
                    case 4: r = f; g = 0.0f; b = 1.0f; break;
                    default: r = 1.0f; g = 0.0f; b = 1.0f - f; break;
                }

                // Modulate by radius
                r = (1.0f - rad * (1.0f - r)) * 255.0f;
                g = (1.0f - rad * (1.0f - g)) * 255.0f;
                b = (1.0f - rad * (1.0f - b)) * 255.0f;

                img.set(x, y, 0, r);
                img.set(x, y, 1, g);
                img.set(x, y, 2, b);
            }
        }
        return img;
    }
};
