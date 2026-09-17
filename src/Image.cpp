#include "Image.h"
#include <iostream>
#include <algorithm>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image::Image(int w, int h, int c)
    : width(w), height(h), channels(c), data(static_cast<size_t>(w) * h * c, 0.0f) {}

Image::Image(int w, int h, int c, float initVal)
    : width(w), height(h), channels(c), data(static_cast<size_t>(w) * h * c, initVal) {}

float Image::sampleBilinear(float x, float y, int c) const {
    if (width <= 0 || height <= 0) return 0.0f;
    
    // Clamp sample coordinates inside valid frame bounds
    x = std::clamp(x, 0.0f, static_cast<float>(width - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(height - 1));

    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);

    float dx = x - static_cast<float>(x0);
    float dy = y - static_cast<float>(y0);

    float v00 = get(x0, y0, c);
    float v10 = get(x1, y0, c);
    float v01 = get(x0, y1, c);
    float v11 = get(x1, y1, c);

    float v0 = v00 * (1.0f - dx) + v10 * dx;
    float v1 = v01 * (1.0f - dx) + v11 * dx;

    return v0 * (1.0f - dy) + v1 * dy;
}

Image Image::toGrayscale() const {
    Image gray(width, height, 1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (channels >= 3) {
                float r = get(x, y, 0);
                float g = get(x, y, 1);
                float b = get(x, y, 2);
                gray.set(x, y, 0, 0.299f * r + 0.587f * g + 0.114f * b);
            } else if (channels == 1) {
                gray.set(x, y, 0, get(x, y, 0));
            } else {
                gray.set(x, y, 0, get(x, y, 0));
            }
        }
    }
    return gray;
}

Image Image::convolve5(const float kernel[5]) const {
    // Horizontal pass
    Image temp(width, height, channels);
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int k = -2; k <= 2; ++k) {
                    int nx = std::clamp(x + k, 0, width - 1);
                    sum += get(nx, y, c) * kernel[k + 2];
                }
                temp.set(x, y, c, sum);
            }
        }
    }

    // Vertical pass
    Image result(width, height, channels);
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int k = -2; k <= 2; ++k) {
                    int ny = std::clamp(y + k, 0, height - 1);
                    sum += temp.get(x, ny, c) * kernel[k + 2];
                }
                result.set(x, y, c, sum);
            }
        }
    }

    return result;
}

Image Image::downsample2x() const {
    int newW = std::max(1, width / 2);
    int newH = std::max(1, height / 2);
    Image out(newW, newH, channels);

    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < newH; ++y) {
            for (int x = 0; x < newW; ++x) {
                // Map coordinate to source image
                int srcX = x * 2;
                int srcY = y * 2;
                out.set(x, y, c, get(srcX, srcY, c));
            }
        }
    }
    return out;
}

Image Image::load(const std::string& path) {
    int w = 0, h = 0, c = 0;
    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &c, 0);
    if (!raw) {
        std::cerr << "Error: failed to load image from " << path << " (" << stbi_failure_reason() << ")\n";
        return Image();
    }

    Image img(w, h, c);
    size_t totalPixels = static_cast<size_t>(w) * h * c;
    for (size_t i = 0; i < totalPixels; ++i) {
        img.data[i] = static_cast<float>(raw[i]);
    }

    stbi_image_free(raw);
    return img;
}

bool Image::savePNG(const std::string& path) const {
    if (empty()) {
        std::cerr << "Error: cannot save empty image to " << path << "\n";
        return false;
    }

    std::vector<unsigned char> bytes(static_cast<size_t>(width) * height * channels);
    size_t total = static_cast<size_t>(width) * height * channels;
    for (size_t i = 0; i < total; ++i) {
        float val = std::clamp(data[i], 0.0f, 255.0f);
        bytes[i] = static_cast<unsigned char>(std::round(val));
    }

    int stride = width * channels;
    int ret = stbi_write_png(path.c_str(), width, height, channels, bytes.data(), stride);
    if (!ret) {
        std::cerr << "Error: failed to write PNG to " << path << "\n";
        return false;
    }
    return true;
}
