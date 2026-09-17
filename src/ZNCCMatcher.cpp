#include "ZNCCMatcher.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

void ZNCCMatcher::buildGaussianWeights(int halfBlock, std::vector<float>& weights, float& sumW) {
    int dim = 2 * halfBlock + 1;
    weights.resize(static_cast<size_t>(dim) * dim);
    sumW = 0.0f;
    float sigma = std::max(1.0f, static_cast<float>(halfBlock) * 0.65f);
    float invTwoSigmaSq = 1.0f / (2.0f * sigma * sigma);

    size_t idx = 0;
    for (int dy = -halfBlock; dy <= halfBlock; ++dy) {
        for (int dx = -halfBlock; dx <= halfBlock; ++dx) {
            float w = std::exp(-static_cast<float>(dx * dx + dy * dy) * invTwoSigmaSq);
            weights[idx++] = w;
            sumW += w;
        }
    }
}

float ZNCCMatcher::computeWeightedZNCC(
    const Image& gray0,
    const Image& gray1,
    int x, int y,
    float u, float v,
    int halfBlock,
    const float* weights,
    float sumW,
    float meanT,
    float stdT,
    bool isIntegerDisp
) {
    float sumC = 0.0f;
    float sumC2 = 0.0f;
    float sumTC = 0.0f;
    size_t idx = 0;

    int intU = static_cast<int>(std::round(u));
    int intV = static_cast<int>(std::round(v));

    for (int dy = -halfBlock; dy <= halfBlock; ++dy) {
        int py = y + dy;
        float qy = static_cast<float>(y) + v + static_cast<float>(dy);
        int intQy = y + intV + dy;

        for (int dx = -halfBlock; dx <= halfBlock; ++dx) {
            int px = x + dx;
            float qx = static_cast<float>(x) + u + static_cast<float>(dx);
            int intQx = x + intU + dx;

            float valT = gray0.get(px, py, 0);
            float valC = isIntegerDisp ?
                gray1.get(intQx, intQy, 0) :
                gray1.sampleBilinear(qx, qy, 0);

            float w = weights ? weights[idx++] : 1.0f;
            sumC += w * valC;
            sumC2 += w * valC * valC;
            sumTC += w * valT * valC;
        }
    }

    float meanC = sumC / sumW;
    float varC = sumC2 - sumW * meanC * meanC;
    if (varC < 0.0f) varC = 0.0f;
    float stdC = std::sqrt(varC);

    // Flat / low-contrast regions
    if (stdT < 0.5f && stdC < 0.5f) {
        float diff = std::abs(meanT - meanC);
        return (diff < 5.0f) ? (1.0f - diff * 0.05f) : -1.0f;
    }
    if (stdT < 0.5f || stdC < 0.5f) {
        return -1.0f;
    }

    float cov = sumTC - sumW * meanT * meanC;
    float zncc = cov / (stdT * stdC + 1e-6f);
    return std::clamp(zncc, -1.0f, 1.0f);
}

FlowField ZNCCMatcher::estimateLevel(
    const Image& gray0,
    const Image& gray1,
    const FlowField* predictor,
    int searchRadius,
    int levelIndex,
    int totalLevels
) const {
    int W = gray0.width;
    int H = gray0.height;
    int step = std::max(1, params.gridStep);

    int gridW = (W + step - 1) / step;
    int gridH = (H + step - 1) / step;

    // Determine level-adaptive block size
    int effectiveBlockSize = params.blockSize;
    if (params.adaptiveBlockSize && totalLevels > 1) {
        // levelIndex: 0 is finest, (totalLevels - 1) is coarsest
        float t = static_cast<float>(levelIndex) / static_cast<float>(totalLevels - 1);
        float nominal = static_cast<float>(params.minBlockSize) * (1.0f - t) +
                        static_cast<float>(params.maxBlockSize) * t;
        effectiveBlockSize = static_cast<int>(std::round(nominal));
    }

    int halfBlock = std::clamp(effectiveBlockSize / 2, 1, std::max(1, std::min(W, H) / 2));
    int blockPixelCount = (2 * halfBlock + 1) * (2 * halfBlock + 1);
    int effSearchRadius = std::min(searchRadius, std::max(2, std::max(W, H)));

    // Precompute 2D Gaussian block weights
    std::vector<float> gaussianWeights;
    float sumW = static_cast<float>(blockPixelCount);
    const float* weightsPtr = nullptr;
    if (params.useGaussianWeights) {
        buildGaussianWeights(halfBlock, gaussianWeights, sumW);
        weightsPtr = gaussianWeights.data();
    }

    FlowField gridFlow(gridW, gridH);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int gy = 0; gy < gridH; ++gy) {
        int y = std::min(gy * step, H - 1);

        for (int gx = 0; gx < gridW; ++gx) {
            int x = std::min(gx * step, W - 1);

            // Precompute template weighted mean and standard deviation
            float sumT = 0.0f;
            float sumT2 = 0.0f;
            size_t wIdx = 0;
            for (int dy = -halfBlock; dy <= halfBlock; ++dy) {
                int py = y + dy;
                for (int dx = -halfBlock; dx <= halfBlock; ++dx) {
                    int px = x + dx;
                    float valT = gray0.get(px, py, 0);
                    float w = weightsPtr ? weightsPtr[wIdx++] : 1.0f;
                    sumT += w * valT;
                    sumT2 += w * valT * valT;
                }
            }
            float meanT = sumT / sumW;
            float varT = sumT2 - sumW * meanT * meanT;
            if (varT < 0.0f) varT = 0.0f;
            float stdT = std::sqrt(varT);

            // Multi-Candidate Spatial / Hierarchical Predictor Selection
            int coarseU = 0;
            int coarseV = 0;
            std::vector<std::pair<int, int>> candidatePredictors;
            candidatePredictors.reserve(8);

            // 1. Hierarchical predictor from coarser level
            if (predictor && !predictor->empty()) {
                MotionVector mv = predictor->sampleBilinear(static_cast<float>(x), static_cast<float>(y));
                coarseU = static_cast<int>(std::round(mv.vx));
                coarseV = static_cast<int>(std::round(mv.vy));
                candidatePredictors.push_back({ coarseU, coarseV });

                if (params.useSpatialPredictors) {
                    // Sample neighborhood in predictor field
                    float off = static_cast<float>(step);
                    MotionVector mvLeft = predictor->sampleBilinear(static_cast<float>(x) - off, static_cast<float>(y));
                    MotionVector mvTop = predictor->sampleBilinear(static_cast<float>(x), static_cast<float>(y) - off);
                    candidatePredictors.push_back({ static_cast<int>(std::round(mvLeft.vx)), static_cast<int>(std::round(mvLeft.vy)) });
                    candidatePredictors.push_back({ static_cast<int>(std::round(mvTop.vx)), static_cast<int>(std::round(mvTop.vy)) });
                }
            } else {
                candidatePredictors.push_back({ 0, 0 });
            }

            int predU = coarseU;
            int predV = coarseV;
            float bestInitialScore = -1e30f;

            // Evaluate candidate predictors to select best initial search center
            for (const auto& cand : candidatePredictors) {
                float zncc = computeWeightedZNCC(
                    gray0, gray1, x, y,
                    static_cast<float>(cand.first), static_cast<float>(cand.second),
                    halfBlock, weightsPtr, sumW, meanT, stdT, true
                );
                int devU = cand.first - coarseU;
                int devV = cand.second - coarseV;
                float cost = zncc - params.smoothnessWeight * static_cast<float>(devU * devU + devV * devV);
                if (cost > bestInitialScore) {
                    bestInitialScore = cost;
                    predU = cand.first;
                    predV = cand.second;
                }
            }

            // Integer search in window [-effSearchRadius, effSearchRadius] around chosen predictor
            float bestCost = bestInitialScore;
            float bestZNCC = -1.0f;
            int bestU = predU;
            int bestV = predV;

            const int winSize = 2 * effSearchRadius + 1;
            float scoreMap[33 * 33]; // Stack-allocated for speed (max search radius 16)
            for (int k = 0; k < winSize * winSize; ++k) scoreMap[k] = -1.0f;

            for (int dv = -effSearchRadius; dv <= effSearchRadius; ++dv) {
                int candV = predV + dv;
                for (int du = -effSearchRadius; du <= effSearchRadius; ++du) {
                    int candU = predU + du;

                    float zncc = computeWeightedZNCC(
                        gray0, gray1, x, y,
                        static_cast<float>(candU), static_cast<float>(candV),
                        halfBlock, weightsPtr, sumW, meanT, stdT, true
                    );
                    scoreMap[(dv + effSearchRadius) * winSize + (du + effSearchRadius)] = zncc;

                    int devU = candU - coarseU;
                    int devV = candV - coarseV;
                    float dispPenalty = params.smoothnessWeight * static_cast<float>(devU * devU + devV * devV);
                    float cost = zncc - dispPenalty;

                    if (cost > bestCost) {
                        bestCost = cost;
                        bestZNCC = zncc;
                        bestU = candU;
                        bestV = candV;
                    }
                }
            }

            float finalU = static_cast<float>(bestU);
            float finalV = static_cast<float>(bestV);
            float finalZNCC = bestZNCC;

            // Sub-Pixel Refinement
            if (params.subpelMode == SUBPEL_PARABOLIC) {
                int bestDu = bestU - predU;
                int bestDv = bestV - predV;

                if (bestDu > -effSearchRadius && bestDu < effSearchRadius &&
                    bestDv > -effSearchRadius && bestDv < effSearchRadius) {

                    int idxCenter = (bestDv + effSearchRadius) * winSize + (bestDu + effSearchRadius);
                    float sC = scoreMap[idxCenter];
                    float sL = scoreMap[idxCenter - 1];
                    float sR = scoreMap[idxCenter + 1];
                    float sT = scoreMap[idxCenter - winSize];
                    float sB = scoreMap[idxCenter + winSize];

                    float curvX = sL - 2.0f * sC + sR;
                    if (curvX < -1e-5f) {
                        float deltaX = (sL - sR) / (2.0f * curvX);
                        finalU += std::clamp(deltaX, -0.5f, 0.5f);
                    }

                    float curvY = sT - 2.0f * sC + sB;
                    if (curvY < -1e-5f) {
                        float deltaY = (sT - sB) / (2.0f * curvY);
                        finalV += std::clamp(deltaY, -0.5f, 0.5f);
                    }
                }
            } else if (params.subpelMode >= SUBPEL_HALF) {
                // Direct Half-Pel Bilinear Search around best integer match
                float halfBestU = finalU;
                float halfBestV = finalV;
                float halfBestScore = bestCost;

                const float halfOffsets[3] = { -0.5f, 0.0f, 0.5f };
                for (int hvy = 0; hvy < 3; ++hvy) {
                    float candV = finalV + halfOffsets[hvy];
                    for (int hux = 0; hux < 3; ++hux) {
                        if (hvy == 1 && hux == 1) continue; // Skip (0, 0) since already computed
                        float candU = finalU + halfOffsets[hux];

                        float zncc = computeWeightedZNCC(
                            gray0, gray1, x, y,
                            candU, candV,
                            halfBlock, weightsPtr, sumW, meanT, stdT, false
                        );
                        float devU = candU - static_cast<float>(coarseU);
                        float devV = candV - static_cast<float>(coarseV);
                        float cost = zncc - params.smoothnessWeight * (devU * devU + devV * devV);

                        if (cost > halfBestScore) {
                            halfBestScore = cost;
                            halfBestU = candU;
                            halfBestV = candV;
                            finalZNCC = zncc;
                        }
                    }
                }

                finalU = halfBestU;
                finalV = halfBestV;

                // Direct Quarter-Pel Bilinear Search around best half-pel match
                if (params.subpelMode >= SUBPEL_QUARTER) {
                    float qtrBestU = finalU;
                    float qtrBestV = finalV;
                    float qtrBestScore = halfBestScore;

                    const float qtrOffsets[3] = { -0.25f, 0.0f, 0.25f };
                    for (int qvy = 0; qvy < 3; ++qvy) {
                        float candV = finalV + qtrOffsets[qvy];
                        for (int qux = 0; qux < 3; ++qux) {
                            if (qvy == 1 && qux == 1) continue;
                            float candU = finalU + qtrOffsets[qux];

                            float zncc = computeWeightedZNCC(
                                gray0, gray1, x, y,
                                candU, candV,
                                halfBlock, weightsPtr, sumW, meanT, stdT, false
                            );
                            float devU = candU - static_cast<float>(coarseU);
                            float devV = candV - static_cast<float>(coarseV);
                            float cost = zncc - params.smoothnessWeight * (devU * devU + devV * devV);

                            if (cost > qtrBestScore) {
                                qtrBestScore = cost;
                                qtrBestU = candU;
                                qtrBestV = candV;
                                finalZNCC = zncc;
                            }
                        }
                    }

                    finalU = qtrBestU;
                    finalV = qtrBestV;
                }
            }

            gridFlow.set(gx, gy, MotionVector(finalU, finalV, finalZNCC));
        }
    }

    // Regularize grid flow with vector median filter
    FlowField filteredGrid = gridFlow.applyVectorMedianFilter(1);

    // Upsample grid to full pixel resolution if grid step > 1
    if (step > 1) {
        FlowField denseFlow(W, H);
        float scaleX = static_cast<float>(gridW) / static_cast<float>(W);
        float scaleY = static_cast<float>(gridH) / static_cast<float>(H);

        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            float gy = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;
            for (int x = 0; x < W; ++x) {
                float gx = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
                MotionVector mv = filteredGrid.sampleBilinear(gx, gy);
                denseFlow.set(x, y, mv);
            }
        }
        return denseFlow.applyColorGuidedMedianFilter(gray0, 1, 25.0f);
    }

    return filteredGrid.applyColorGuidedMedianFilter(gray0, 1, 25.0f);
}

FlowField ZNCCMatcher::estimateFlow(const Image& img0, const Image& img1) const {
    Image gray0 = img0.toGrayscale();
    Image gray1 = img1.toGrayscale();

    int totalLevels = std::max(1, params.pyramidLevels);

    // Build 8-level image pyramids
    Pyramid pyr0(gray0, totalLevels);
    Pyramid pyr1(gray1, totalLevels);

    // Start at coarsest level (numLevels - 1, e.g. Level 7)
    int coarsestIdx = totalLevels - 1;
    FlowField currentFlow = estimateLevel(
        pyr0.getLevel(coarsestIdx),
        pyr1.getLevel(coarsestIdx),
        nullptr,
        params.searchRadiusCoarse,
        coarsestIdx,
        totalLevels
    );

    // Coarse-to-fine propagation down to Level 0
    for (int l = coarsestIdx - 1; l >= 0; --l) {
        const Image& curImg0 = pyr0.getLevel(l);
        const Image& curImg1 = pyr1.getLevel(l);

        // Upsample flow field from level l+1 to match level l dimensions and scale vectors
        FlowField predictor = currentFlow.upsample(curImg0.width, curImg0.height);

        // Refine at level l
        currentFlow = estimateLevel(
            curImg0,
            curImg1,
            &predictor,
            params.searchRadiusRefine,
            l,
            totalLevels
        );
    }

    return currentFlow;
}
