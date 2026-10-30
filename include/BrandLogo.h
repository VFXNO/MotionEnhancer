#pragma once

#include "AppResources.h"
#include <windows.h>
#include <objidl.h>
#include <algorithm>
#include <cmath>
#include <cstring>
namespace Gdiplus {
using std::max;
using std::min;
}
#include <gdiplus.h>

// The app and NSIS skin use the very same transparent artwork as the ICO.
// Keep the source stream alive for GDI+, and release it before GDI+ shuts down.
class BrandLogo {
public:
    static constexpr float AnimationPeriod = 12.0f;
    BrandLogo() = default;
    BrandLogo(const BrandLogo&) = delete;
    BrandLogo& operator=(const BrandLogo&) = delete;
    ~BrandLogo() { reset(); }

    bool load(HINSTANCE module) {
        reset();
        HRSRC resource = FindResourceW(module, MAKEINTRESOURCEW(IDR_BRAND_LOGO), MAKEINTRESOURCEW(10));
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
        if (FAILED(CreateStreamOnHGlobal(memory, TRUE, &stream_))) {
            GlobalFree(memory);
            return false;
        }
        image_ = Gdiplus::Bitmap::FromStream(stream_, FALSE);
        if (!image_ || image_->GetLastStatus() != Gdiplus::Ok) { reset(); return false; }
        return true;
    }

    void reset() {
        delete image_;
        image_ = nullptr;
        if (stream_) stream_->Release();
        stream_ = nullptr;
    }

    void draw(Gdiplus::Graphics& g, float cx, float cy, float size, float time,
              float reveal = 1.0f, bool assembling = false) const {
        if (!image_) return;
        reveal = std::clamp(reveal, 0.0f, 1.0f);
        const float extent = size * (0.92f + 0.08f * reveal);
        const auto state = g.Save();
        if (time < 0) {
            // Explicit still-image mode uses the same master as Explorer.
            Gdiplus::ColorMatrix fade = {{{1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0},
                                          {0, 0, 0, reveal, 0}, {0, 0, 0, 0, 1}}};
            Gdiplus::ImageAttributes attributes;
            attributes.SetColorMatrix(&fade);
            attributes.SetWrapMode(Gdiplus::WrapModeTileFlipXY);
            g.SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBicubic);
            g.DrawImage(image_, Gdiplus::RectF(cx - extent / 2, cy - extent / 2, extent, extent),
                        0.0f, 0.0f, static_cast<float>(image_->GetWidth()), static_cast<float>(image_->GetHeight()),
                        Gdiplus::UnitPixel, &attributes);
            g.Restore(state);
            return;
        }

        // Vector animation follows the SVG's 100-unit geometry. Moving parts
        // are drawn separately, so no stationary nodes/trails show underneath.
        g.TranslateTransform(cx, cy);
        g.ScaleTransform(extent / 100, extent / 100);
        g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
        constexpr float pi = 3.14159265358979323846f;
        auto phase = [](float t, float start, float length) {
            return std::clamp((t - start) / length, 0.0f, 1.0f);
        };
        // Zero velocity and acceleration at each reveal endpoint: no springy
        // overshoot or abrupt speed change as the assembled logo settles.
        auto ease = [](float p) { return std::clamp(p * p * p * (10 + p * (-15 + 6 * p)), 0.0f, 1.0f); };
        auto ink = [reveal](float alpha, BYTE value = 242) {
            return Gdiplus::Color(static_cast<BYTE>(std::clamp(alpha * reveal, 0.0f, 255.0f)), value, value,
                                  static_cast<BYTE>(std::min(255, value + 5)));
        };
        // All ambient components share a seamless 12-second clock: two spark
        // revolutions, one node spacing, two breaths and four trail cycles.
        const float cycle = std::fmod(time, AnimationPeriod) / AnimationPeriod;
        const float turn = 2 * pi * cycle;
        const float ring = assembling ? ease(phase(time, 0.12f, 0.9f)) : 1.0f;
        const float trails = assembling ? ease(phase(time, 0.9f, 0.6f)) : 1.0f;
        const float coreReveal = assembling ? ease(phase(time, 0.72f, 0.55f)) : 1.0f;
        const float pulse = assembling ? std::sin(pi * phase(time, 1.05f, 0.65f)) : 0.0f;
        const float breath = 0.5f - 0.5f * std::cos(2 * turn);
        auto softGlow = [&g, &ink](float x, float y, float radius, float alpha) {
            if (alpha <= 0) return;
            Gdiplus::GraphicsPath disc;
            disc.AddEllipse(x - radius, y - radius, radius * 2, radius * 2);
            Gdiplus::PathGradientBrush glow(&disc);
            glow.SetCenterPoint(Gdiplus::PointF(x, y));
            glow.SetCenterColor(ink(alpha));
            Gdiplus::Color edge = ink(0);
            int count = 1;
            glow.SetSurroundColors(&edge, &count);
            g.FillPath(&glow, &disc);
        };
        // A true radial fade instead of stacked translucent circles, which
        // produced visible concentric bands in the old animation.
        softGlow(0, 0, 63, (16 + 8 * breath) * ring);
        Gdiplus::SolidBrush backdrop(Gdiplus::Color(static_cast<BYTE>(255 * reveal * ring), 17, 17, 21));
        g.FillEllipse(&backdrop, -39.0f, -39.0f, 78.0f, 78.0f);
        Gdiplus::Pen inner(ink(205), 1.45f), outer(ink(120), 1.1f);
        inner.SetStartCap(Gdiplus::LineCapRound);
        inner.SetEndCap(Gdiplus::LineCapRound);
        outer.SetStartCap(Gdiplus::LineCapRound);
        outer.SetEndCap(Gdiplus::LineCapRound);
        if (ring >= 1) {
            g.DrawEllipse(&inner, -30.0f, -30.0f, 60.0f, 60.0f);
            g.DrawEllipse(&outer, -39.0f, -39.0f, 78.0f, 78.0f);
        } else if (ring > 0) {
            g.DrawArc(&inner, -30.0f, -30.0f, 60.0f, 60.0f, -90.0f, 360 * ring);
            g.DrawArc(&outer, -39.0f, -39.0f, 78.0f, 78.0f, -90.0f, -360 * ring);
        }

        // Constant angular velocity before/after assembly; the spark fades in
        // once the orbit is nearly drawn rather than snapping between clocks.
        const float head = -48 + cycle * 720;
        const float sparkReveal = assembling ? ease(phase(time, 0.68f, 0.45f)) : 1.0f;
        if (sparkReveal > 0) {
            const float angle = head * pi / 180;
            const float x = 30 * std::cos(angle), y = 30 * std::sin(angle);
            const float tailAngle = (head - 58) * pi / 180;
            Gdiplus::LinearGradientBrush comet(Gdiplus::PointF(30 * std::cos(tailAngle), 30 * std::sin(tailAngle)),
                                              Gdiplus::PointF(x, y), ink(0), ink(240 * sparkReveal));
            Gdiplus::Pen tail(&comet, 2.0f);
            tail.SetStartCap(Gdiplus::LineCapRound);
            tail.SetEndCap(Gdiplus::LineCapRound);
            g.DrawArc(&tail, -30.0f, -30.0f, 60.0f, 60.0f, head - 58, 58.0f);
            softGlow(x, y, 4.5f, 38 * sparkReveal);
            Gdiplus::SolidBrush spark(ink(255 * sparkReveal));
            g.FillEllipse(&spark, x - 1.55f, y - 1.55f, 3.1f, 3.1f);
        }

        Gdiplus::SolidBrush dot(ink(225));
        for (int k = 0; k < 6; ++k) {
            const float p = assembling ? ease(phase(time, 0.58f + 0.06f * k, 0.6f)) : 1.0f;
            if (p <= 0) continue;
            const float reach = 6 * p;
            const float angle = (k * 60 + cycle * 60) * pi / 180;
            const float dx = std::cos(angle), dy = std::sin(angle);
            g.DrawLine(&outer, 39 * dx, 39 * dy, (39 + reach) * dx, (39 + reach) * dy);
            const float radius = 1.75f * p;
            g.FillEllipse(&dot, (39 + reach) * dx - radius, (39 + reach) * dy - radius, radius * 2, radius * 2);
        }

        for (int i = 0; i < 3 && trails > 0; ++i) {
            const float sweep = assembling ? ease(phase(time, 0.88f + 0.08f * i, 0.5f)) : trails;
            if (sweep <= 0) continue;
            const float y = -7.5f + 7.5f * i, end = i == 1 ? 4.0f : 2.0f;
            const float flow = 0.5f + 0.5f * std::sin(4 * turn - i * 2 * pi / 3);
            const float length = (i == 1 ? 27.0f : 21.0f) * (0.97f + 0.03f * flow) * sweep;
            Gdiplus::LinearGradientBrush fade(Gdiplus::PointF(end - length - 3, y), Gdiplus::PointF(end, y),
                                              ink(0), ink((195 + 30 * flow) * sweep));
            Gdiplus::Pen trail(&fade, i == 1 ? 1.8f : 1.4f);
            trail.SetStartCap(Gdiplus::LineCapRound);
            trail.SetEndCap(Gdiplus::LineCapRound);
            g.DrawLine(&trail, end - length, y, end, y);
        }

        // A soft luminance bloom, not a multi-ring flash or a changing core size.
        softGlow(9, 0, 24, 38 * std::max(0.0f, pulse));
        const float core = 6 * coreReveal;
        if (core > 0) {
            Gdiplus::SolidBrush light(ink(255));
            g.FillEllipse(&light, 9 - core, -core, core * 2, core * 2);
        }
        Gdiplus::Pen halo(ink((100 + 14 * breath) * coreReveal), 1.0f);
        g.DrawEllipse(&halo, -0.8f, -9.8f, 19.6f, 19.6f);
        g.Restore(state);
    }

private:
    IStream* stream_ = nullptr;
    Gdiplus::Bitmap* image_ = nullptr;
};
