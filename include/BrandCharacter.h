#pragma once

#include "BrandLogo.h"
#include "BrandCharacterAtlas.h"

// Decorative, silent artwork only. Decode once at startup into premultiplied
// pixels; painting a frame never opens a file or invokes a video/GPU decoder.
class BrandCharacter {
public:
    BrandCharacter() = default;
    BrandCharacter(const BrandCharacter&) = delete;
    BrandCharacter& operator=(const BrandCharacter&) = delete;
    ~BrandCharacter() { reset(); }

    bool load(HINSTANCE module) {
        reset();
        HRSRC resource = FindResourceW(module, MAKEINTRESOURCEW(IDR_BRAND_CHARACTER), MAKEINTRESOURCEW(10));
        if (!resource) return false;
        const DWORD size = SizeofResource(module, resource);
        const void* data = LockResource(LoadResource(module, resource));
        if (!size || !data) return false;
        HGLOBAL memory = GlobalAlloc(GMEM_MOVEABLE, size);
        if (!memory) return false;
        void* buffer = GlobalLock(memory);
        if (!buffer) { GlobalFree(memory); return false; }
        std::memcpy(buffer, data, size);
        GlobalUnlock(memory);
        IStream* stream = nullptr;
        if (FAILED(CreateStreamOnHGlobal(memory, TRUE, &stream))) {
            GlobalFree(memory);
            return false;
        }
        Gdiplus::Bitmap* source = Gdiplus::Bitmap::FromStream(stream, FALSE);
        const int rows = (CharacterAtlas::Frames + CharacterAtlas::Columns - 1) / CharacterAtlas::Columns;
        const int width = CharacterAtlas::Columns * CharacterAtlas::CellWidth;
        const int height = rows * CharacterAtlas::CellHeight;
        bool ready = source && source->GetLastStatus() == Gdiplus::Ok &&
            source->GetWidth() == width && source->GetHeight() == height;
        if (ready) {
            atlas_ = new Gdiplus::Bitmap(width, height, PixelFormat32bppPARGB);
            ready = atlas_->GetLastStatus() == Gdiplus::Ok;
            if (ready) {
                Gdiplus::Graphics g(atlas_);
                g.SetCompositingMode(Gdiplus::CompositingModeSourceCopy);
                ready = g.DrawImage(source, 0, 0, width, height) == Gdiplus::Ok;
            }
        }
        delete source;
        stream->Release();
        if (!ready) reset(); // Missing/bad artwork must not prevent startup.
        return ready;
    }

    void reset() {
        delete atlas_;
        atlas_ = nullptr;
    }

    bool loaded() const { return atlas_ != nullptr; }

    void draw(Gdiplus::Graphics& g, const Gdiplus::RectF& area, float time, float opacity = 1.0f) const {
        if (!atlas_ || area.Width <= 0 || area.Height <= 0 || opacity <= 0) return;
        const float cycle = std::fmod(std::max(0.0f, time), CharacterAtlas::Period);
        const int index = std::min(CharacterAtlas::Frames - 1, static_cast<int>(cycle * CharacterAtlas::Fps));
        const int x = (index % CharacterAtlas::Columns) * CharacterAtlas::CellWidth + CharacterAtlas::Padding;
        const int y = (index / CharacterAtlas::Columns) * CharacterAtlas::CellHeight + CharacterAtlas::Padding;
        const auto state = g.Save();
        g.SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBilinear);
        Gdiplus::ColorMatrix fade = {{{1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0},
                                      {0, 0, 0, std::clamp(opacity, 0.0f, 1.0f), 0}, {0, 0, 0, 0, 1}}};
        Gdiplus::ImageAttributes attributes;
        attributes.SetColorMatrix(&fade);
        g.DrawImage(atlas_, area, static_cast<float>(x), static_cast<float>(y),
                    static_cast<float>(CharacterAtlas::Width), static_cast<float>(CharacterAtlas::Height),
                    Gdiplus::UnitPixel, &attributes);
        g.Restore(state);
    }

private:
    Gdiplus::Bitmap* atlas_ = nullptr;
};
