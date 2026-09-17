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
    numLevels = requestedLevels;
    levels.clear();
    levels.reserve(numLevels);

    if (baseImage.empty()) {
        return;
    }

    // Level 0 is the original full-resolution image
    levels.push_back(baseImage);

    // Build successive coarsened levels
    for (int l = 1; l < numLevels; ++l) {
        Image smoothed = levels[l - 1].convolve5(GAUSSIAN_5TAP);
        Image downsampled = smoothed.downsample2x();
        levels.push_back(downsampled);
    }
}
