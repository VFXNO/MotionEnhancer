#pragma once

#include "Image.h"
#include <vector>

class Pyramid {
public:
    int numLevels = 8;
    std::vector<Image> levels; // levels[0] = finest (original), levels[numLevels - 1] = coarsest

    Pyramid() = default;
    explicit Pyramid(const Image& baseImage, int levels = 8);

    void build(const Image& baseImage, int requestedLevels = 8);

    const Image& getLevel(int level) const { return levels[level]; }
    Image& getLevel(int level) { return levels[level]; }
    int size() const { return static_cast<int>(levels.size()); }
};
