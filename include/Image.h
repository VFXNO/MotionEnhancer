#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

class Image {
public:
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<float> data;

    Image() = default;
    Image(int w, int h, int c = 3);
    Image(int w, int h, int c, float initVal);

    bool empty() const { return data.empty() || width <= 0 || height <= 0; }

    inline int index(int x, int y, int c = 0) const {
        return (y * width + x) * channels + c;
    }

    inline float get(int x, int y, int c = 0) const {
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        return data[index(x, y, c)];
    }

    inline void set(int x, int y, int c, float val) {
        if (x >= 0 && x < width && y >= 0 && y < height && c >= 0 && c < channels) {
            data[index(x, y, c)] = val;
        }
    }

    inline float& at(int x, int y, int c = 0) {
        return data[index(x, y, c)];
    }

    inline const float& at(int x, int y, int c = 0) const {
        return data[index(x, y, c)];
    }

    // Bilinear sampling with edge clamping
    float sampleBilinear(float x, float y, int c = 0) const;

    // Convert to 1-channel grayscale luminance
    Image toGrayscale() const;

    // Convolve with separable 1D kernel [k0, k1, k2, k1, k0]
    Image convolve5(const float kernel[5]) const;

    // Downsample by factor of 2 (with boundary safety)
    Image downsample2x() const;

    // Load from disk (PNG, JPG, BMP)
    static Image load(const std::string& path);

    // Save to disk as PNG (8-bit per channel)
    bool savePNG(const std::string& path) const;
};
