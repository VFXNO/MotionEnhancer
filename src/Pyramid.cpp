#include "Pyramid.h"
#include <iostream>

static const float GAUSSIAN_5TAP[5] = {
    1.0f / 16.0f,
    4.0f / 16.0f,
    6.0f / 16.0f,
    4.0f / 16.0f,
    1.0f / 16.0f
};

Pyramid::Pyramid(const Image& baseImage, int levels) {
    build(baseImage, levels);
}

void Pyramid::build(const Image& baseImage, int requestedLevels) {
    numLevels = std::max(1, requestedLevels);
    levels.clear();
    levels.reserve(numLevels);

    if (baseImage.empty()) {
        return;
    }

    // Level 0 is the original full-resolution image
    levels.push_back(baseImage);

    // Build successive coarsened levels
    for (int l = 1; l < numLevels; ++l) {
        if (levels.back().width == 1 && levels.back().height == 1) break;
        Image smoothed = levels[l - 1].convolve5(GAUSSIAN_5TAP);
        Image downsampled = smoothed.downsample2x();
        levels.push_back(downsampled);
    }
    numLevels = static_cast<int>(levels.size());
}
