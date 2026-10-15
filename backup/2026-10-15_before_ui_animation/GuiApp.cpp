#include "GuiApp.h"
#include "Image.h"
#include "WindowHelper.h"

#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <dwmapi.h>
#include <objidl.h>
// GDI+ (anti-aliased shapes) uses min/max, which NOMINMAX removes.
namespace Gdiplus {
using std::max;
using std::min;
}
#include <gdiplus.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr int IDC_TAB_LIVE = 120;
constexpr int IDC_TAB_OFFLINE = 121;
constexpr int IDC_WINDOW_LIST = 101;
constexpr int IDC_REFRESH_WINDOWS = 102;
constexpr int IDC_START_CAPTURE = 103;
constexpr int IDC_LIVE_STATUS = 104;
constexpr int IDC_LIVE_SOURCE_FPS = 106;
constexpr int IDC_GPU_LEVELS = 107;
constexpr int IDC_GPU_MIN_REFINE = 108;
constexpr int IDC_GPU_COARSE_RADIUS = 109;
constexpr int IDC_GPU_REFINE_RADIUS = 110;
constexpr int IDC_GPU_SMOOTHNESS = 111;
constexpr int IDC_LIVE_MULTIPLIER = 112;
constexpr int IDC_LIVE_FLOW_ENGINE = 113;
constexpr int IDC_LIVE_ADAPTER = 114;
constexpr int IDC_AI_DEPTH_RATE = 115;
constexpr int IDC_INPUT0 = 201;
constexpr int IDC_BROWSE_INPUT0 = 202;
constexpr int IDC_INPUT1 = 203;
constexpr int IDC_BROWSE_INPUT1 = 204;
constexpr int IDC_OUTPUT = 205;
constexpr int IDC_BROWSE_OUTPUT = 206;
constexpr int IDC_TIME = 207;
constexpr int IDC_RUN_OFFLINE = 208;
constexpr int IDC_OFFLINE_STATUS = 209;
constexpr int IDC_OFFLINE_LEVELS = 210;
constexpr int IDC_OFFLINE_BLOCK_SIZE = 211;
constexpr int IDC_OFFLINE_MIN_BLOCK = 212;
constexpr int IDC_OFFLINE_MAX_BLOCK = 213;
constexpr int IDC_OFFLINE_COARSE_RADIUS = 214;
constexpr int IDC_OFFLINE_REFINE_RADIUS = 215;
constexpr int IDC_OFFLINE_GRID_STEP = 216;
constexpr int IDC_OFFLINE_SUBPEL = 217;
constexpr int IDC_OFFLINE_SMOOTHNESS = 218;
constexpr int IDC_OFFLINE_OCCLUSION = 219;
constexpr int IDC_OFFLINE_DETAIL_STRENGTH = 220;
constexpr int IDC_OFFLINE_PHOTO_THRESHOLD = 221;
constexpr int IDC_OFFLINE_PHOTO_SIGMA = 222;
constexpr int IDC_TRAIN_IMAGES = 301;
constexpr int IDC_TRAIN_MODEL = 302;
constexpr int IDC_SAVE_TRAIN_MODEL = 303;
constexpr int IDC_OPEN_TRAIN_MODEL = 304;
constexpr int IDC_RUN_TRAIN = 305;
constexpr int IDC_TRAIN_STATUS = 306;
constexpr UINT WM_PROCESS_COMPLETE = WM_APP + 1;
constexpr UINT WM_TRAIN_COMPLETE = WM_APP + 2;
constexpr UINT WM_TRAIN_PROGRESS = WM_APP + 3;

enum class JobType : LPARAM {
    Live = 1,
    Offline = 2,
    Train = 3
};

// ------------------------------------------------------------------
// Theme: monochrome, near-black surfaces, white accents, soft rounded
// cards, a navigation sidebar on the left.
// ------------------------------------------------------------------
constexpr COLORREF ColBg        = RGB(8, 8, 10);
constexpr COLORREF ColSidebar   = RGB(12, 12, 14);
constexpr COLORREF ColPanel     = RGB(17, 17, 20);
constexpr COLORREF ColPanelHi   = RGB(28, 28, 32);
constexpr COLORREF ColField     = RGB(13, 13, 15);
constexpr COLORREF ColBorder    = RGB(42, 42, 48);
constexpr COLORREF ColBorderHi  = RGB(74, 74, 82);
constexpr COLORREF ColAccent    = RGB(240, 240, 244);
constexpr COLORREF ColAccentDk  = RGB(178, 178, 186);
constexpr COLORREF ColText      = RGB(238, 238, 242);
constexpr COLORREF ColDim       = RGB(138, 138, 148);
constexpr COLORREF ColTextOn    = RGB(10, 10, 12);

// The sidebar's width; every page control is placed right of it
// (addControl adds g_offsetX).
constexpr int kSidebarW = 232;
constexpr RECT LivePanelRect = { kSidebarW + 36, 172, kSidebarW + 36 + 794, 172 + 206 };
constexpr RECT OfflinePanelRect = { kSidebarW + 36, 272, kSidebarW + 36 + 794, 272 + 330 };
constexpr RECT TrainPanelRect = { kSidebarW + 36, 514, kSidebarW + 36 + 794, 514 + 240 };

struct GuiState {
    HWND hwnd = nullptr;
    int activeTab = 0;
    HWND tabLive = nullptr;
    HWND tabOffline = nullptr;
    HWND windowList = nullptr;
    HWND refreshButton = nullptr;
    HWND captureButton = nullptr;
    HANDLE captureStopEvent = nullptr;
    HWND liveStatus = nullptr;
    HWND trainingModel = nullptr;
    HWND trainingImagesLabel = nullptr;
    HWND trainButton = nullptr;
    HWND trainingStatus = nullptr;
    HWND useTrainedModel = nullptr;
    std::vector<std::string> trainingImages;
    HWND liveSourceFps = nullptr;
    HWND liveMultiplier = nullptr;
    HWND liveFlowEngine = nullptr;
    HWND liveAdapter = nullptr;
    HWND gpuLevels = nullptr;
    HWND gpuMinRefine = nullptr;
    HWND gpuCoarseRadius = nullptr;
    HWND gpuRefineRadius = nullptr;
    HWND gpuSmoothness = nullptr;
    HWND uiMask = nullptr;
    HWND ffxRefine = nullptr;
    HWND occlusionFill = nullptr;
    HWND ffxOcclusion = nullptr;
    HWND aiDepth = nullptr;
    HWND aiDepthRate = nullptr;
    HWND input0 = nullptr;
    HWND input1 = nullptr;
    HWND output = nullptr;
    HWND time = nullptr;
    HWND offlineButton = nullptr;
    HWND offlineStatus = nullptr;
    HWND offlineLevels = nullptr;
    HWND offlineBlockSize = nullptr;
    HWND offlineMinBlock = nullptr;
    HWND offlineMaxBlock = nullptr;
    HWND offlineCoarseRadius = nullptr;
    HWND offlineRefineRadius = nullptr;
    HWND offlineGridStep = nullptr;
    HWND offlineSubpel = nullptr;
    HWND offlineSmoothness = nullptr;
    HWND offlineDetailStrength = nullptr;
    HWND adaptiveBlock = nullptr;
    HWND spatialPredictors = nullptr;
    HWND gaussianWeights = nullptr;
    HWND bidirectional = nullptr;
    HWND detailRestore = nullptr;
    HWND colorClamp = nullptr;
    std::vector<HWND> liveControls;
    std::vector<HWND> offlineControls;
};

GuiState g_state;
HWND g_hover = nullptr;

HFONT g_fontBody = nullptr;
HFONT g_fontLabel = nullptr;
HFONT g_fontButton = nullptr;
HFONT g_fontSection = nullptr;
HBRUSH g_brField = nullptr;
HBRUSH g_brBg = nullptr;
HBRUSH g_brPanel = nullptr;
HBRUSH g_brPanelHi = nullptr;
HBRUSH g_brSidebar = nullptr;
HFONT g_fontTitle = nullptr;
HFONT g_fontBrand = nullptr;
HFONT g_fontNav = nullptr;
ULONG_PTR g_gdiplusToken = 0;
int g_offsetX = 0;   // added to x by addControl: the page area starts right of the sidebar

HFONT createFont(const char* face, int height, int weight) {
    return CreateFontA(
        height, 0, 0, 0, weight, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, face);
}

void ensureTheme() {
    if (g_fontBody) return;
    g_fontBody = createFont("Segoe UI", -14, FW_NORMAL);
    g_fontLabel = createFont("Segoe UI", -13, FW_NORMAL);
    g_fontButton = createFont("Segoe UI", -14, FW_SEMIBOLD);
    g_fontSection = createFont("Segoe UI Semibold", -16, FW_SEMIBOLD);
    g_fontTitle = createFont("Segoe UI Semibold", -26, FW_SEMIBOLD);
    g_fontBrand = createFont("Segoe UI Semibold", -13, FW_SEMIBOLD);
    g_fontNav = createFont("Segoe UI", -15, FW_NORMAL);
    g_brField = CreateSolidBrush(ColField);
    g_brBg = CreateSolidBrush(ColBg);
    g_brPanel = CreateSolidBrush(ColPanel);
    g_brPanelHi = CreateSolidBrush(ColPanelHi);
    g_brSidebar = CreateSolidBrush(ColSidebar);
}

COLORREF lerpColor(COLORREF a, COLORREF b, double t) {
    t = std::clamp(t, 0.0, 1.0);
    int r = static_cast<int>(GetRValue(a) + (GetRValue(b) - GetRValue(a)) * t);
    int g = static_cast<int>(GetGValue(a) + (GetGValue(b) - GetGValue(a)) * t);
    int bl = static_cast<int>(GetBValue(a) + (GetBValue(b) - GetBValue(a)) * t);
    return RGB(r, g, bl);
}

// Anti-aliased shapes (GDI+).
Gdiplus::Color gpColor(COLORREF c, int alpha = 255) {
    return Gdiplus::Color(static_cast<BYTE>(alpha), GetRValue(c), GetGValue(c), GetBValue(c));
}

void roundPath(Gdiplus::GraphicsPath& path, float x, float y, float w, float h, float r) {
    r = std::min(r, std::min(w, h) / 2.0f);
    if (r <= 0.5f) {
        path.AddRectangle(Gdiplus::RectF(x, y, w, h));
        return;
    }
    const float d = 2.0f * r;
    path.AddArc(x, y, d, d, 180.0f, 90.0f);
    path.AddArc(x + w - d, y, d, d, 270.0f, 90.0f);
    path.AddArc(x + w - d, y + h - d, d, d, 0.0f, 90.0f);
    path.AddArc(x, y + h - d, d, d, 90.0f, 90.0f);
    path.CloseFigure();
}

void prepare(Gdiplus::Graphics& g) {
    g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
    g.SetPixelOffsetMode(Gdiplus::PixelOffsetModeHalf);
}

void fillRoundRect(HDC hdc, const RECT& rc, COLORREF color, int radius, int alpha = 255) {
    Gdiplus::Graphics g(hdc);
    prepare(g);
    Gdiplus::GraphicsPath path;
    roundPath(path, static_cast<float>(rc.left), static_cast<float>(rc.top),
              static_cast<float>(rc.right - rc.left), static_cast<float>(rc.bottom - rc.top), static_cast<float>(radius));
    Gdiplus::SolidBrush brush(gpColor(color, alpha));
    g.FillPath(&brush, &path);
}

void frameRoundRect(HDC hdc, const RECT& rc, COLORREF color, int radius, int alpha = 255, float width = 1.0f) {
    Gdiplus::Graphics g(hdc);
    prepare(g);
    Gdiplus::GraphicsPath path;
    roundPath(path, rc.left + 0.5f, rc.top + 0.5f, rc.right - rc.left - 1.0f, rc.bottom - rc.top - 1.0f,
              static_cast<float>(radius));
    Gdiplus::Pen pen(gpColor(color, alpha), width);
    g.DrawPath(&pen, &path);
}

// A card: solid fill and a border lit from above (a glass edge).
void paintCard(HDC hdc, const RECT& rc, COLORREF fill, int radius) {
    Gdiplus::Graphics g(hdc);
    prepare(g);
    const float x = static_cast<float>(rc.left), y = static_cast<float>(rc.top);
    const float w = static_cast<float>(rc.right - rc.left), h = static_cast<float>(rc.bottom - rc.top);
    Gdiplus::GraphicsPath path;
    roundPath(path, x, y, w, h, static_cast<float>(radius));
    Gdiplus::SolidBrush brush(gpColor(fill));
    g.FillPath(&brush, &path);
    Gdiplus::GraphicsPath edge;
    roundPath(edge, x + 0.5f, y + 0.5f, w - 1.0f, h - 1.0f, static_cast<float>(radius));
    // Light at the top, fading to the plain border within 140 px.
    Gdiplus::LinearGradientBrush light(Gdiplus::PointF(0.0f, y - 1.0f), Gdiplus::PointF(0.0f, y + h + 1.0f),
                                       gpColor(RGB(92, 92, 102)), gpColor(ColBorder));
    const Gdiplus::Color colors[3] = {gpColor(RGB(92, 92, 102)), gpColor(ColBorder), gpColor(ColBorder)};
    const Gdiplus::REAL positions[3] = {0.0f, std::min(1.0f, 140.0f / std::max(h, 1.0f)), 1.0f};
    light.SetInterpolationColors(colors, positions, 3);
    Gdiplus::Pen pen(&light, 1.0f);
    g.DrawPath(&pen, &edge);
}

// The surface a control sits on, for owner-drawn controls' corners.
COLORREF surfaceUnder(HWND control) {
    RECT rc = {};
    GetWindowRect(control, &rc);
    MapWindowPoints(HWND_DESKTOP, GetParent(control), reinterpret_cast<POINT*>(&rc), 2);
    const int cx = (rc.left + rc.right) / 2;
    const int cy = (rc.top + rc.bottom) / 2;
    if (cx < kSidebarW) return ColSidebar;
    auto within = [cx, cy](const RECT& panel) {
        return cx >= panel.left && cx < panel.right && cy >= panel.top && cy < panel.bottom;
    };
    const bool onPanel = g_state.activeTab == 0 ? within(LivePanelRect) || within(TrainPanelRect)
                                                : within(OfflinePanelRect);
    return onPanel ? ColPanel : ColBg;
}

void setControlFont(HWND control, HFONT font) {
    SendMessageA(control, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
}

HWND addControl(const char* className, const char* text, DWORD style,
                int x, int y, int w, int h, int id) {
    HWND control = CreateWindowExA(
        0, className, text, WS_CHILD | WS_VISIBLE | style,
        x + g_offsetX, y, w, h, g_state.hwnd,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
        GetModuleHandleA(nullptr), nullptr);
    return control;
}

// ------------------------------------------------------------------
// Hover tracking for owner-drawn controls
// ------------------------------------------------------------------
LRESULT CALLBACK styledControlProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam,
                                   UINT_PTR subclassId, DWORD_PTR refData) {
    // Checkboxes are BS_OWNERDRAW buttons (BS_AUTOCHECKBOX | BS_OWNERDRAW is
    // just BS_OWNERDRAW), and an owner-drawn button keeps no check state:
    // Windows ignores BM_SETCHECK on it and BM_GETCHECK returns 0. So every
    // checkbox read as unchecked - "Keep static UI sharp" launched every GUI
    // capture with --no-ui-mask - and a click changed nothing. The state is
    // kept here instead, and the main window toggles it on BN_CLICKED.
    if (GetPropA(hwnd, "MeCheckbox")) {
        switch (message) {
            case BM_GETCHECK:
                return GetPropA(hwnd, "MeChecked") ? BST_CHECKED : BST_UNCHECKED;
            case BM_SETCHECK:
                if (wParam == BST_CHECKED) SetPropA(hwnd, "MeChecked", reinterpret_cast<HANDLE>(static_cast<INT_PTR>(1)));
                else RemovePropA(hwnd, "MeChecked");
                InvalidateRect(hwnd, nullptr, FALSE);
                return 0;
            case WM_NCDESTROY:
                RemovePropA(hwnd, "MeChecked");
                RemovePropA(hwnd, "MeCheckbox");
                break;
        }
    }
    switch (message) {
        case WM_MOUSEMOVE:
            if (g_hover != hwnd) {
                g_hover = hwnd;
                InvalidateRect(hwnd, nullptr, FALSE);
                TRACKMOUSEEVENT track = {};
                track.cbSize = sizeof(track);
                track.dwFlags = TME_LEAVE;
                track.hwndTrack = hwnd;
                TrackMouseEvent(&track);
            }
            break;
        case WM_MOUSELEAVE:
            if (g_hover == hwnd) {
                g_hover = nullptr;
                InvalidateRect(hwnd, nullptr, FALSE);
            }
            break;
    }
    return DefSubclassProc(hwnd, message, wParam, lParam);
}

// A STATIC control with dynamically changing text must erase its entire
// rectangle before drawing the next message. Otherwise a shorter warning
// can be painted over the glyphs of the previous one.
LRESULT CALLBACK changingStaticProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam,
                                    UINT_PTR subclassId, DWORD_PTR refData) {
    LRESULT result = DefSubclassProc(hwnd, message, wParam, lParam);
    if (message == WM_SETTEXT) InvalidateRect(hwnd, nullptr, TRUE);
    return result;
}

LRESULT CALLBACK comboNcProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam,
                             UINT_PTR subclassId, DWORD_PTR refData) {
    LRESULT result = DefSubclassProc(hwnd, message, wParam, lParam);
    if (message == WM_NCPAINT) {
        HDC hdc = GetWindowDC(hwnd);
        if (hdc) {
            RECT rc = {};
            GetWindowRect(hwnd, &rc);
            OffsetRect(&rc, -rc.left, -rc.top);
            HPEN pen = CreatePen(PS_SOLID, 1, ColField);
            HPEN oldPen = static_cast<HPEN>(SelectObject(hdc, pen));
            HBRUSH oldBrush = static_cast<HBRUSH>(SelectObject(hdc, GetStockObject(NULL_BRUSH)));
            Rectangle(hdc, rc.left, rc.top, rc.right, rc.bottom);
            SelectObject(hdc, oldPen);
            SelectObject(hdc, oldBrush);
            DeleteObject(pen);
            ReleaseDC(hwnd, hdc);
        }
    } else if (message == WM_PAINT) {
        // Replace the system's light dropdown button with a themed one.
        HDC hdc = GetDC(hwnd);
        if (hdc) {
            RECT rc = {};
            GetClientRect(hwnd, &rc);
            // The system's square border is painted over: the window behind
            // draws a rounded frame around the combo (paintFieldFrames).
            HBRUSH field = CreateSolidBrush(ColField);
            for (int i = 0; i < 3; ++i) {
                RECT edge = rc;
                InflateRect(&edge, -i, -i);
                FrameRect(hdc, &edge, field);
            }
            DeleteObject(field);
            RECT button = { rc.right - 28, rc.top + 1, rc.right - 1, rc.bottom - 1 };
            FillRect(hdc, &button, g_brField);
            int cx = (button.left + button.right) / 2;
            int cy = (button.top + button.bottom) / 2;
            HPEN pen = CreatePen(PS_SOLID, 2, ColDim);
            HPEN oldPen = static_cast<HPEN>(SelectObject(hdc, pen));
            MoveToEx(hdc, cx - 5, cy - 2, nullptr);
            LineTo(hdc, cx, cy + 3);
            LineTo(hdc, cx + 5, cy - 2);
            SelectObject(hdc, oldPen);
            DeleteObject(pen);
            ReleaseDC(hwnd, hdc);
        }
    }
    return result;
}

void drawStyledButton(const DRAWITEMSTRUCT* draw) {
    HWND hwnd = draw->hwndItem;
    HDC hdc = draw->hDC;
    RECT rc = draw->rcItem;
    char text[96] = {};
    GetWindowTextA(hwnd, text, sizeof(text));

    // Base fill in the colour under the control, so rounded corners blend.
    {
        HBRUSH base = CreateSolidBrush(surfaceUnder(hwnd));
        FillRect(hdc, &rc, base);
        DeleteObject(base);
    }

    bool pressed = (draw->itemState & ODS_SELECTED) != 0;
    bool disabled = (draw->itemState & ODS_DISABLED) != 0;
    bool hovered = (g_hover == hwnd) && !disabled;
    int id = draw->CtlID;

    bool isTab = (id == IDC_TAB_LIVE || id == IDC_TAB_OFFLINE);
    bool tabActive = isTab && ((id == IDC_TAB_LIVE) == (g_state.activeTab == 0));
    bool primary = (id == IDC_START_CAPTURE || id == IDC_RUN_OFFLINE);
    // Style bits are ambiguous (BS_OWNERDRAW overlaps BS_CHECKBOX), so
    // checkboxes are tagged explicitly with a window property.
    bool isCheckbox = GetPropA(hwnd, "MeCheckbox") != nullptr;

    if (isCheckbox) {
        bool checked = SendMessageA(hwnd, BM_GETCHECK, 0, 0) == BST_CHECKED;
        if (hovered) fillRoundRect(hdc, rc, ColPanelHi, 6);
        RECT box = { rc.left + 2, rc.top + (rc.bottom - rc.top - 16) / 2,
                     rc.left + 18, rc.top + (rc.bottom - rc.top + 16) / 2 };
        fillRoundRect(hdc, box, checked ? ColAccent : ColField, 5);
        frameRoundRect(hdc, box, checked ? ColAccent : (hovered ? ColAccentDk : ColBorderHi), 5);
        if (checked) {
            HPEN pen = CreatePen(PS_SOLID, 2, ColTextOn);
            HPEN oldPen = static_cast<HPEN>(SelectObject(hdc, pen));
            MoveToEx(hdc, box.left + 4, box.top + 8, nullptr);
            LineTo(hdc, box.left + 7, box.top + 11);
            LineTo(hdc, box.left + 13, box.top + 4);
            SelectObject(hdc, oldPen);
            DeleteObject(pen);
        }
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, disabled ? ColDim : ColText);
        RECT textRect = { rc.left + 26, rc.top, rc.right, rc.bottom };
        DrawTextA(hdc, text, -1, &textRect, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        return;
    }

    if (isTab) {
        // A sidebar navigation item: icon and label, the active one on a
        // lit card.
        if (tabActive) {
            paintCard(hdc, rc, ColPanelHi, 12);
        } else if (hovered) {
            fillRoundRect(hdc, rc, RGB(20, 20, 23), 12);
        }
        const COLORREF ink = tabActive ? ColText : (hovered ? ColText : ColDim);
        {
            Gdiplus::Graphics g(hdc);
            prepare(g);
            Gdiplus::Pen pen(gpColor(ink), 1.7f);
            const float cx = rc.left + 30.0f, cy = (rc.top + rc.bottom) / 2.0f;
            if (id == IDC_TAB_LIVE) {
                // A live signal: a dot and two arcs on each side.
                Gdiplus::SolidBrush dot(gpColor(ink));
                g.FillEllipse(&dot, cx - 3.0f, cy - 3.0f, 6.0f, 6.0f);
                for (float r : {7.0f, 11.5f}) {
                    g.DrawArc(&pen, cx - r, cy - r, 2 * r, 2 * r, -50.0f, 100.0f);
                    g.DrawArc(&pen, cx - r, cy - r, 2 * r, 2 * r, 130.0f, 100.0f);
                }
            } else {
                // Two frames and the one made between them.
                g.DrawRectangle(&pen, cx - 10.0f, cy - 8.0f, 12.0f, 10.0f);
                g.DrawRectangle(&pen, cx - 2.0f, cy - 2.0f, 12.0f, 10.0f);
            }
        }
        HFONT oldFont = static_cast<HFONT>(SelectObject(hdc, g_fontNav));
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, ink);
        RECT label = { rc.left + 52, rc.top, rc.right - 8, rc.bottom };
        DrawTextA(hdc, text, -1, &label, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        SelectObject(hdc, oldFont);
        return;
    }

    int radius = (rc.bottom - rc.top) / 2;   // pills
    if (primary) {
        COLORREF fill = disabled ? ColPanelHi : pressed ? ColAccentDk : (hovered ? RGB(255, 255, 255) : ColAccent);
        fillRoundRect(hdc, rc, fill, radius);
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, disabled ? ColDim : ColTextOn);
    } else {
        fillRoundRect(hdc, rc, disabled ? ColPanel : (hovered ? RGB(34, 34, 39) : ColPanelHi), radius);
        frameRoundRect(hdc, rc, hovered ? ColBorderHi : ColBorder, radius);
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, disabled ? ColDim : ColText);
    }
    RECT textRect = rc;
    if (pressed) OffsetRect(&textRect, 1, 1);
    DrawTextA(hdc, text, -1, &textRect, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
}

void drawStyledComboItem(const DRAWITEMSTRUCT* draw) {
    HDC hdc = draw->hDC;
    RECT rc = draw->rcItem;

    // Always paint the full item region, including the empty (-1) state,
    // so no system-white background can bleed through.
    FillRect(hdc, &rc, g_brField);

    if (draw->itemID == static_cast<UINT>(-1)) {
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, ColDim);
        RECT emptyRect = rc;
        emptyRect.left += 10;
        DrawTextA(hdc, "", -1, &emptyRect, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        return;
    }

    const LRESULT length = SendMessageA(draw->hwndItem, CB_GETLBTEXTLEN, draw->itemID, 0);
    if (length == CB_ERR || length < 0 || length > 32767) return;
    std::string text(static_cast<size_t>(length) + 1, '\0');
    const LRESULT copied = SendMessageA(draw->hwndItem, CB_GETLBTEXT, draw->itemID,
                                        reinterpret_cast<LPARAM>(text.data()));
    if (copied == CB_ERR || copied < 0) return;

    bool isEditPart = (draw->itemState & ODS_COMBOBOXEDIT) != 0;
    bool highlighted = !isEditPart &&
        ((draw->itemState & ODS_SELECTED) != 0 || (draw->itemState & ODS_FOCUS) != 0);

    FillRect(hdc, &rc, highlighted ? g_brPanelHi : g_brField);
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, highlighted ? ColAccent : ColText);

    RECT textRect = rc;
    textRect.left += 10;
    // The dropdown caret is painted once by comboNcProc, not here too.
    // Leave room for its button so long item names cannot run underneath it.
    if (isEditPart) textRect.right -= 32;
    DrawTextA(hdc, text.c_str(), -1, &textRect,
              DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS | DT_NOPREFIX);
}

// ------------------------------------------------------------------
// Business logic (unchanged behavior)
// ------------------------------------------------------------------
std::string getWindowText(HWND control) {
    int length = GetWindowTextLengthA(control);
    std::string text(static_cast<size_t>(length) + 1, '\0');
    GetWindowTextA(control, text.data(), length + 1);
    text.resize(static_cast<size_t>(length));
    return text;
}

std::string lowercaseWindowText(HWND control) {
    std::string value = getWindowText(control);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

void populateSubpixelModes(HWND combo) {
    const char* modes[] = { "None", "Parabolic", "Half", "Quarter" };
    for (const char* mode : modes) {
        SendMessageA(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(mode));
    }
    SendMessageA(combo, CB_SETCURSEL, 3, 0);
}

void populateSourceRates(HWND combo) {
    const char* rates[] = { "Auto", "24", "30", "60" };
    for (const char* rate : rates) {
        SendMessageA(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(rate));
    }
    SendMessageA(combo, CB_SETCURSEL, 0, 0);
}

void populateMultipliers(HWND combo) {
    const char* multipliers[] = { "Max", "2x", "3x", "4x" };
    for (const char* multiplier : multipliers) {
        SendMessageA(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(multiplier));
    }
    SendMessageA(combo, CB_SETCURSEL, 0, 0);
}

std::string quoteArgument(const std::string& value) {
    if (value.find_first_of(" \t\"") == std::string::npos) return value;

    std::string result = "\"";
    size_t backslashes = 0;
    for (char ch : value) {
        if (ch == '\\') {
            ++backslashes;
        } else if (ch == '"') {
            result.append(backslashes * 2 + 1, '\\');
            result.push_back('"');
            backslashes = 0;
        } else {
            result.append(backslashes, '\\');
            backslashes = 0;
            result.push_back(ch);
        }
    }
    result.append(backslashes * 2, '\\');
    result.push_back('"');
    return result;
}

std::string trainingResult(DWORD exitCode, const std::string& output) {
    auto lineAfter = [&output](const char* marker) -> std::string {
        const size_t at = output.rfind(marker);
        if (at == std::string::npos) return {};
        const size_t end = output.find('\n', at);
        std::string line = output.substr(at, end == std::string::npos ? end : end - at);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        return line;
    };

    // The trainer writes the model only after validation improves, so the
    // presence of this line is the whole success condition.
    if (output.find("Model written: ") != std::string::npos) {
        const std::string summary = lineAfter("Neural training finished:");
        if (!summary.empty()) return "Model saved. " + summary.substr(std::strlen("Neural training finished: "));
        return "Model saved. Ready for neural capture.";
    }
    if (output.find("Refusing to train") != std::string::npos)
        return "Training aborted: the analytic gradient disagreed with the forward pass. "
               "This is a code defect, not a data problem - run --neural-gradient-check.";
    if (output.find("Model did not improve validation error") != std::string::npos)
        return "No model saved: training did not improve validation error. "
               "Try more or more varied footage.";
    if (output.find("Too few training examples") != std::string::npos)
        return "No model saved: too few usable examples. Use longer or larger sources.";
    if (output.find("Neural training needs colour sources") != std::string::npos)
        return "Training needs colour sources at least 256x256 pixels.";
    if (exitCode == 0) return "Training complete. Model ready for neural capture.";
    const size_t decode = output.rfind("Could not decode ");
    if (decode != std::string::npos) {
        const size_t end = output.find('\n', decode);
        const size_t reason = output.rfind(": ", end);
        if (reason != std::string::npos && reason > decode)
            return "Video decode failed: " + output.substr(reason + 2, end - (reason + 2));
        return "Video decode failed. Check its codec or try MP4.";
    }
    if (output.find("failed to load image") != std::string::npos)
        return "Could not open that picture; check its format or installed Windows image codec.";
    if (output.find("Training requires RGB images") != std::string::npos)
        return "Training needs colour images at least 96x96 pixels.";
    if (output.find("Insufficient or invalid training data") != std::string::npos)
        return "Too few usable training blocks; try a longer or more detailed video.";
    return "Training failed. Try a different video or image, or run the CLI for details.";
}

bool launchJob(const std::vector<std::string>& arguments, JobType jobType,
               bool configureNativeModel = false, const std::string& nativeModelPath = {}) {
    char executable[MAX_PATH] = {};
    if (GetModuleFileNameA(nullptr, executable, MAX_PATH) == 0) return false;

    std::string commandLine = quoteArgument(executable);
    for (const auto& argument : arguments) {
        commandLine += " " + quoteArgument(argument);
    }

    std::vector<char> mutableCommand(commandLine.begin(), commandLine.end());
    mutableCommand.push_back('\0');

    STARTUPINFOA startup = {};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process = {};
    // Only the neural capture child receives the GUI's model selection. Save
    // and restore the parent's environment so training/FFX jobs and later
    // launches cannot accidentally inherit a stale model. The GUI launches
    // jobs on its single UI thread, so no other launch interleaves this.
    char oldModel[MAX_PATH] = {};
    const DWORD oldLength = configureNativeModel
        ? GetEnvironmentVariableA("MOTION_ENHANCER_NEURAL_MODEL", oldModel, sizeof(oldModel)) : 0;
    if (configureNativeModel && oldLength >= sizeof(oldModel)) return false;
    if (configureNativeModel && !SetEnvironmentVariableA(
            "MOTION_ENHANCER_NEURAL_MODEL", nativeModelPath.empty() ? nullptr : nativeModelPath.c_str())) return false;
    HANDLE logRead = nullptr, logWrite = nullptr, inputNull = nullptr;
    if (jobType == JobType::Train) {
        SECURITY_ATTRIBUTES security = {sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        if (!CreatePipe(&logRead, &logWrite, &security, 0) ||
            !SetHandleInformation(logRead, HANDLE_FLAG_INHERIT, 0)) {
            if (logRead) CloseHandle(logRead);
            if (logWrite) CloseHandle(logWrite);
            return false;
        }
        inputNull = CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (inputNull == INVALID_HANDLE_VALUE) {
            CloseHandle(logRead);
            CloseHandle(logWrite);
            return false;
        }
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = inputNull;
        startup.hStdOutput = logWrite;
        startup.hStdError = logWrite;
    }
    BOOL started = CreateProcessA(
            executable,
            mutableCommand.data(),
            nullptr,
            nullptr,
            jobType == JobType::Train,
            CREATE_NO_WINDOW,
            nullptr,
            nullptr,
            &startup,
            &process);
    if (configureNativeModel) SetEnvironmentVariableA(
        "MOTION_ENHANCER_NEURAL_MODEL", oldLength ? oldModel : nullptr);
    if (logWrite) CloseHandle(logWrite);
    if (inputNull) CloseHandle(inputNull);
    if (!started) {
        if (logRead) CloseHandle(logRead);
        return false;
    }

    CloseHandle(process.hThread);
    HWND notifyWindow = g_state.hwnd;
    std::thread([handle = process.hProcess, notifyWindow, jobType, logRead]() {
        std::string output;
        if (logRead) {
            char buffer[1024];
            DWORD length = 0;
            std::string pending;
            while (ReadFile(logRead, buffer, sizeof(buffer), &length, nullptr) && length) {
                output.append(buffer, buffer + length);
                if (output.size() > 8192) output.erase(0, output.size() - 8192);
                pending.append(buffer, buffer + length);
                size_t end = 0;
                while ((end = pending.find('\n')) != std::string::npos) {
                    const std::string line = pending.substr(0, end);
                    if (line.rfind("Epoch ", 0) == 0 ||
                        line.rfind("Real-frame step ", 0) == 0 ||
                        line.rfind("Real-frame epoch ", 0) == 0) {
                        const size_t clip = line.find(", val accuracy");
                        auto* progress = new std::string(line.substr(0, clip));
                        if (!PostMessageA(notifyWindow, WM_TRAIN_PROGRESS, 0,
                                          reinterpret_cast<LPARAM>(progress))) delete progress;
                    }
                    pending.erase(0, end + 1);
                }
                if (pending.size() > 2048) pending.clear();
            }
            CloseHandle(logRead);
        }
        WaitForSingleObject(handle, INFINITE);
        DWORD exitCode = 1;
        GetExitCodeProcess(handle, &exitCode);
        CloseHandle(handle);
        if (jobType == JobType::Train) {
            auto* result = new std::string(trainingResult(exitCode, output));
            if (!PostMessageA(notifyWindow, WM_TRAIN_COMPLETE, exitCode, reinterpret_cast<LPARAM>(result)))
                delete result;
        } else {
            PostMessageA(notifyWindow, WM_PROCESS_COMPLETE, exitCode, static_cast<LPARAM>(jobType));
        }
    }).detach();
    return true;
}

void refreshWindowList() {
    SendMessageA(g_state.windowList, CB_RESETCONTENT, 0, 0);
    for (const auto& info : WindowHelper::enumerateWindows()) {
        SendMessageA(
            g_state.windowList, CB_ADDSTRING, 0,
            reinterpret_cast<LPARAM>(info.title.c_str()));
    }
}

void selectFile(HWND edit, bool saveDialog) {
    char path[MAX_PATH] = {};
    OPENFILENAMEA dialog = {};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = g_state.hwnd;
    dialog.lpstrFile = path;
    dialog.nMaxFile = MAX_PATH;
    dialog.lpstrFilter = saveDialog ? "PNG images (*.png)\0*.png\0All files (*.*)\0*.*\0"
        : "Images\0*.png;*.jpg;*.jpeg;*.jfif;*.bmp;*.gif;*.tif;*.tiff;*.webp;*.heic;*.heif;*.avif;*.ico;*.tga;*.psd;*.pnm;*.hdr;*.jxr;*.wdp\0All files (*.*)\0*.*\0";
    dialog.lpstrDefExt = "png";
    dialog.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | (saveDialog ? OFN_OVERWRITEPROMPT : OFN_FILEMUSTEXIST);

    BOOL selected = saveDialog ? GetSaveFileNameA(&dialog) : GetOpenFileNameA(&dialog);
    if (selected) SetWindowTextA(edit, path);
}

void selectTrainingImages() {
    // Explorer returns either a single full path or a directory followed
    // by NUL-separated filenames for multiselect.
    std::vector<char> buffer(32768, '\0');
    OPENFILENAMEA dialog = {};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = g_state.hwnd;
    dialog.lpstrFile = buffer.data();
    dialog.nMaxFile = static_cast<DWORD>(buffer.size());
    dialog.lpstrFilter = "Videos and images\0*.mp4;*.m4v;*.mov;*.mkv;*.avi;*.wmv;*.webm;*.mpg;*.mpeg;*.png;*.jpg;*.jpeg;*.jfif;*.bmp;*.gif;*.tif;*.tiff;*.webp;*.heic;*.heif;*.avif;*.ico;*.tga;*.psd;*.pnm;*.hdr;*.jxr;*.wdp\0"
                          "Videos\0*.mp4;*.m4v;*.mov;*.mkv;*.avi;*.wmv;*.webm;*.mpg;*.mpeg\0"
                          "Images\0*.png;*.jpg;*.jpeg;*.jfif;*.bmp;*.gif;*.tif;*.tiff;*.webp;*.heic;*.heif;*.avif;*.ico;*.tga;*.psd;*.pnm;*.hdr;*.jxr;*.wdp\0All files (*.*)\0*.*\0";
    dialog.Flags = OFN_EXPLORER | OFN_ALLOWMULTISELECT | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (!GetOpenFileNameA(&dialog)) return;
    std::vector<std::string> paths;
    const char* first = buffer.data();
    const char* name = first + std::strlen(first) + 1;
    if (!*name) {
        paths.emplace_back(first);
    } else {
        const std::string directory(first);
        while (*name) {
            paths.emplace_back(directory + "\\" + name);
            name += std::strlen(name) + 1;
        }
    }
    if (paths.size() > 16) {
        SetWindowTextA(g_state.trainingStatus, "Select up to 16 video or image files for one training run.");
        return;
    }
    g_state.trainingImages = std::move(paths);
    const std::string label = std::to_string(g_state.trainingImages.size()) + " file(s) selected";
    SetWindowTextA(g_state.trainingImagesLabel, label.c_str());
    SetWindowTextA(g_state.trainingStatus, "Ready to decode frames and train offline.");
}

void selectTrainingModel(bool saveDialog) {
    char path[MAX_PATH] = {};
    OPENFILENAMEA dialog = {};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = g_state.hwnd;
    dialog.lpstrFile = path;
    dialog.nMaxFile = MAX_PATH;
    dialog.lpstrFilter = "Neural motion model (*.bin)\0*.bin\0All files (*.*)\0*.*\0";
    dialog.lpstrDefExt = "bin";
    dialog.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST |
                   (saveDialog ? OFN_OVERWRITEPROMPT : OFN_FILEMUSTEXIST);
    if (saveDialog ? GetSaveFileNameA(&dialog) : GetOpenFileNameA(&dialog)) {
        SetWindowTextA(g_state.trainingModel, path);
        if (!saveDialog) {
            SendMessageA(g_state.useTrainedModel, BM_SETCHECK, BST_CHECKED, 0);
            SendMessageA(g_state.liveFlowEngine, CB_SETCURSEL, 2, 0);
            InvalidateRect(g_state.useTrainedModel, nullptr, FALSE);
            SetWindowTextA(g_state.trainingStatus, "Model selected for the next neural capture.");
        }
    }
}

void trainModel() {
    const std::string output = getWindowText(g_state.trainingModel);
    if (g_state.trainingImages.empty() || output.empty()) {
        SetWindowTextA(g_state.trainingStatus, "Choose a video or images and a model output path first.");
        return;
    }
    std::vector<std::string> args = {"--train-neural", output};
    args.insert(args.end(), g_state.trainingImages.begin(), g_state.trainingImages.end());
    if (!launchJob(args, JobType::Train)) {
        SetWindowTextA(g_state.trainingStatus, "Could not start model training.");
        return;
    }
    EnableWindow(g_state.trainButton, FALSE);
    EnableWindow(g_state.captureButton, FALSE);
    EnableWindow(g_state.offlineButton, FALSE);
    SetWindowTextA(g_state.trainingStatus, "Decoding video frames and training; window stays responsive...");
}

bool validateInteger(HWND control, int minimum, int maximum, const char* label, HWND status) {
    std::string value = getWindowText(control);
    char* end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (end != value.c_str() && *end == '\0' && parsed >= minimum && parsed <= maximum) {
        return true;
    }
    std::string message = std::string(label) + " must be " + std::to_string(minimum) + " to " +
                          std::to_string(maximum) + ".";
    SetWindowTextA(status, message.c_str());
    SetFocus(control);
    return false;
}

bool validateFloat(HWND control, float minimum, float maximum, const char* label, HWND status) {
    std::string value = getWindowText(control);
    char* end = nullptr;
    float parsed = std::strtof(value.c_str(), &end);
    if (end != value.c_str() && *end == '\0' && parsed >= minimum && parsed <= maximum) {
        return true;
    }
    std::string message = std::string(label) + " is outside its allowed range.";
    SetWindowTextA(status, message.c_str());
    SetFocus(control);
    return false;
}

std::string checkValue(HWND control) {
    return SendMessageA(control, BM_GETCHECK, 0, 0) == BST_CHECKED ? "1" : "0";
}

void startCapture() {
    if (g_state.captureStopEvent) {
        if (SetEvent(g_state.captureStopEvent)) {
            EnableWindow(g_state.captureButton, FALSE);
            SetWindowTextA(g_state.captureButton, "STOPPING...");
            SetWindowTextA(g_state.liveStatus, "Stopping capture and releasing GPU resources...");
        } else {
            SetWindowTextA(g_state.liveStatus, "Could not request capture stop; use Ctrl+Alt+Esc.");
        }
        return;
    }
    int selection = static_cast<int>(SendMessageA(g_state.windowList, CB_GETCURSEL, 0, 0));
    if (selection == CB_ERR) {
        SetWindowTextA(g_state.liveStatus, "Choose a target window first.");
        return;
    }

    int length = static_cast<int>(SendMessageA(g_state.windowList, CB_GETLBTEXTLEN, selection, 0));
    if (length == CB_ERR || length < 0 || length > 32767) {
        SetWindowTextA(g_state.liveStatus, "Window list changed; refresh and choose the target again.");
        return;
    }
    std::string title(static_cast<size_t>(length) + 1, '\0');
    const LRESULT copied = SendMessageA(g_state.windowList, CB_GETLBTEXT,
                                        selection, reinterpret_cast<LPARAM>(title.data()));
    if (copied == CB_ERR || copied < 0) {
        SetWindowTextA(g_state.liveStatus, "Window list changed; refresh and choose the target again.");
        return;
    }
    title.resize(static_cast<size_t>(copied));

    if (!validateInteger(g_state.gpuLevels, 1, 8, "Pyramid levels", g_state.liveStatus) ||
        !validateInteger(g_state.gpuMinRefine, 0, 7, "Finest searched level", g_state.liveStatus) ||
        !validateInteger(g_state.gpuCoarseRadius, 0, 8, "Coarse radius", g_state.liveStatus) ||
        !validateInteger(g_state.gpuRefineRadius, 0, 8, "Refine radius", g_state.liveStatus) ||
        !validateFloat(g_state.gpuSmoothness, 0.0f, 0.1f, "Smoothness", g_state.liveStatus)) {
        return;
    }
    if (std::atoi(getWindowText(g_state.gpuMinRefine).c_str()) >=
        std::atoi(getWindowText(g_state.gpuLevels).c_str())) {
        SetWindowTextA(g_state.liveStatus, "Finest searched level must be below pyramid levels.");
        SetFocus(g_state.gpuMinRefine);
        return;
    }

    int flowSelection = static_cast<int>(SendMessageA(g_state.liveFlowEngine, CB_GETCURSEL, 0, 0));
    const char* flowEngineArg = flowSelection == 3 ? "dis" :
                                flowSelection == 2 ? "neural" :
                                flowSelection == 1 ? "sad" : "ffx";
    int adapterSelection = static_cast<int>(SendMessageA(g_state.liveAdapter, CB_GETCURSEL, 0, 0));
    const char* adapterArg = adapterSelection == 1 ? "igpu" :
                             adapterSelection == 2 ? "dgpu" : "auto";

    bool useModel = SendMessageA(g_state.useTrainedModel, BM_GETCHECK, 0, 0) == BST_CHECKED;
    std::string modelPath = useModel ? getWindowText(g_state.trainingModel) : std::string();
    if (flowSelection == 2 && !useModel) {
        const std::string selectedPath = getWindowText(g_state.trainingModel);
        const DWORD attributes = selectedPath.empty() ? INVALID_FILE_ATTRIBUTES
                                                       : GetFileAttributesA(selectedPath.c_str());
        if (attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
            useModel = true;
            modelPath = selectedPath;
            SendMessageA(g_state.useTrainedModel, BM_SETCHECK, BST_CHECKED, 0);
            InvalidateRect(g_state.useTrainedModel, nullptr, FALSE);
        }
    }
    if (useModel) {
        const DWORD attributes = GetFileAttributesA(modelPath.c_str());
        if (modelPath.empty() || attributes == INVALID_FILE_ATTRIBUTES ||
            (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
            SetWindowTextA(g_state.liveStatus, "Select an existing .bin model or uncheck Use trained model.");
            return;
        }
    }
    if (flowSelection == 2 && !useModel) {
        SetWindowTextA(g_state.liveStatus, "Choose a trained .bin and check Use model for neural capture.");
        return;
    }

    static uint64_t captureSequence = 0;
    const std::string stopName = "Local\\MotionEnhancerStop-" + std::to_string(GetCurrentProcessId()) +
                                 "-" + std::to_string(++captureSequence);
    HANDLE stopEvent = CreateEventA(nullptr, TRUE, FALSE, stopName.c_str());
    if (!stopEvent) {
        SetWindowTextA(g_state.liveStatus, "Could not create capture stop event.");
        return;
    }
    std::vector<std::string> captureArgs = {
        "--capture-window", title,
        "--stop-event", stopName,
        "--source-fps", lowercaseWindowText(g_state.liveSourceFps),
        "--multiplier", lowercaseWindowText(g_state.liveMultiplier),
        "--gpu-levels", getWindowText(g_state.gpuLevels),
        "--gpu-min-refine", getWindowText(g_state.gpuMinRefine),
        "--gpu-coarse-radius", getWindowText(g_state.gpuCoarseRadius),
        "--gpu-refine-radius", getWindowText(g_state.gpuRefineRadius),
        "--gpu-smoothness", getWindowText(g_state.gpuSmoothness),
        "--flow-engine", flowEngineArg,
        "--adapter", adapterArg
    };
    if (SendMessageA(g_state.uiMask, BM_GETCHECK, 0, 0) != BST_CHECKED) captureArgs.push_back("--no-ui-mask");
    // Engine-specific, ignored by the others: the occlusion fill is DIS's,
    // the refinement and its occlusion handling FFX's (occlusion handling
    // is part of the refinement, so it needs the refinement on).
    if (SendMessageA(g_state.occlusionFill, BM_GETCHECK, 0, 0) != BST_CHECKED) captureArgs.push_back("--no-occlusion-fill");
    if (SendMessageA(g_state.ffxRefine, BM_GETCHECK, 0, 0) != BST_CHECKED) captureArgs.push_back("--no-ffx-refine");
    if (SendMessageA(g_state.ffxOcclusion, BM_GETCHECK, 0, 0) == BST_CHECKED) captureArgs.push_back("--ffx-occlusion");
    if (SendMessageA(g_state.aiDepth, BM_GETCHECK, 0, 0) == BST_CHECKED) {
        captureArgs.push_back("--ai-depth");
        const LRESULT rate = SendMessageA(g_state.aiDepthRate, CB_GETCURSEL, 0, 0);
        captureArgs.push_back("--ai-depth-every");
        captureArgs.push_back(std::to_string(rate > 0 ? rate + 1 : 1));
    }
    if (!launchJob(captureArgs, JobType::Live, true, modelPath)) {
        CloseHandle(stopEvent);
        SetWindowTextA(g_state.liveStatus, "Could not start the capture process.");
        return;
    }

    g_state.captureStopEvent = stopEvent;
    SetWindowTextA(g_state.captureButton, "STOP CAPTURE");
    EnableWindow(g_state.trainButton, FALSE);
    SetWindowTextA(g_state.liveStatus, "Capture running. Click Stop Capture to exit cleanly.");
}

void runOffline() {
    std::string frame0 = getWindowText(g_state.input0);
    std::string frame1 = getWindowText(g_state.input1);
    std::string output = getWindowText(g_state.output);
    std::string time = getWindowText(g_state.time);

    char* end = nullptr;
    float timeValue = std::strtof(time.c_str(), &end);
    if (frame0.empty() || frame1.empty() || output.empty()) {
        SetWindowTextA(g_state.offlineStatus, "Select both input frames and an output path.");
        return;
    }
    if (end == time.c_str() || *end != '\0' || timeValue <= 0.0f || timeValue >= 1.0f) {
        SetWindowTextA(g_state.offlineStatus, "Interpolation time must be between 0 and 1.");
        return;
    }
    if (!validateInteger(g_state.offlineLevels, 1, 8, "Pyramid levels", g_state.offlineStatus) ||
        !validateInteger(g_state.offlineBlockSize, 1, 64, "Block size", g_state.offlineStatus) ||
        !validateInteger(g_state.offlineMinBlock, 1, 64, "Minimum block", g_state.offlineStatus) ||
        !validateInteger(g_state.offlineMaxBlock, 1, 64, "Maximum block", g_state.offlineStatus) ||
        !validateInteger(g_state.offlineCoarseRadius, 0, 64, "Coarse radius", g_state.offlineStatus) ||
        !validateInteger(g_state.offlineRefineRadius, 0, 16, "Refine radius", g_state.offlineStatus) ||
        !validateInteger(g_state.offlineGridStep, 1, 8, "Grid step", g_state.offlineStatus) ||
        !validateFloat(g_state.offlineSmoothness, 0.0f, 0.1f, "Smoothness", g_state.offlineStatus) ||
        !validateFloat(g_state.offlineDetailStrength, 0.0f, 1.0f, "Detail strength", g_state.offlineStatus)) {
        return;
    }
    if (std::atoi(getWindowText(g_state.offlineMinBlock).c_str()) >
        std::atoi(getWindowText(g_state.offlineMaxBlock).c_str())) {
        SetWindowTextA(g_state.offlineStatus, "Minimum block cannot exceed maximum block.");
        SetFocus(g_state.offlineMinBlock);
        return;
    }

    if (!launchJob({
            frame0, frame1, output,
            "--time", time,
            "--levels", getWindowText(g_state.offlineLevels),
            "--block-size", getWindowText(g_state.offlineBlockSize),
            "--min-block-size", getWindowText(g_state.offlineMinBlock),
            "--max-block-size", getWindowText(g_state.offlineMaxBlock),
            "--coarse-radius", getWindowText(g_state.offlineCoarseRadius),
            "--search-radius", getWindowText(g_state.offlineRefineRadius),
            "--grid-step", getWindowText(g_state.offlineGridStep),
            "--subpel", lowercaseWindowText(g_state.offlineSubpel),
            "--smoothness", getWindowText(g_state.offlineSmoothness),
            "--detail-strength", getWindowText(g_state.offlineDetailStrength),
            "--adaptive-block", checkValue(g_state.adaptiveBlock),
            "--spatial-pred", checkValue(g_state.spatialPredictors),
            "--gaussian-weight", checkValue(g_state.gaussianWeights),
            "--bidirectional", checkValue(g_state.bidirectional),
            "--detail-restore", checkValue(g_state.detailRestore),
            "--color-clamp", checkValue(g_state.colorClamp)
        }, JobType::Offline)) {
        SetWindowTextA(g_state.offlineStatus, "Could not start offline processing.");
        return;
    }

    EnableWindow(g_state.offlineButton, FALSE);
    EnableWindow(g_state.trainButton, FALSE);
    SetWindowTextA(g_state.offlineStatus, "Processing frames...");
}

// ------------------------------------------------------------------
// Layout
// ------------------------------------------------------------------
void applyPageVisibility() {
    bool live = g_state.activeTab == 0;
    for (HWND control : g_state.liveControls) ShowWindow(control, live ? SW_SHOW : SW_HIDE);
    for (HWND control : g_state.offlineControls) ShowWindow(control, live ? SW_HIDE : SW_SHOW);
    InvalidateRect(g_state.tabLive, nullptr, FALSE);
    InvalidateRect(g_state.tabOffline, nullptr, FALSE);
    InvalidateRect(g_state.hwnd, nullptr, FALSE);
}

HWND makeCombo(int x, int y, int w, int id) {
    // HASSTRINGS is required for owner-draw combos: without it CB_ADDSTRING
    // stores the pointer as raw item data and CB_GETLBTEXT returns garbage.
    // No WS_VSCROLL: tall lists avoid the classic light scrollbar entirely.
    HWND combo = addControl(
        "COMBOBOX", "",
        CBS_DROPDOWNLIST | CBS_OWNERDRAWFIXED | CBS_HASSTRINGS,
        x, y, w, 620, id);
    SendMessageA(combo, CB_SETITEMHEIGHT, static_cast<WPARAM>(-1), 26);
    setControlFont(combo, g_fontBody);
    SetWindowSubclass(combo, comboNcProc, 2, 0);
    return combo;
}

HWND makeEdit(const char* text, int x, int y, int w, int id, bool numeric) {
    HWND edit = addControl(
        "EDIT", text,
        ES_AUTOHSCROLL | (numeric ? ES_NUMBER : 0),
        x, y, w, 26, id);
    setControlFont(edit, g_fontBody);
    return edit;
}

HWND makeLabel(const char* text, int x, int y, int w) {
    HWND label = addControl("STATIC", text, SS_LEFT, x, y, w, 20, 0);
    setControlFont(label, g_fontLabel);
    return label;
}

HWND makeButton(const char* text, int x, int y, int w, int h, int id) {
    HWND button = addControl(
        "BUTTON", text,
        BS_PUSHBUTTON | BS_OWNERDRAW,
        x, y, w, h, id);
    setControlFont(button, g_fontButton);
    SetWindowSubclass(button, styledControlProc, 1, 0);
    return button;
}

HWND makeCheckbox(const char* text, int x, int y, int w, bool checked) {
    HWND checkbox = addControl(
        "BUTTON", text,
        BS_AUTOCHECKBOX | BS_OWNERDRAW,
        x, y, w, 24, 0);
    // Tagged and subclassed before the initial state is set: the state lives
    // in styledControlProc (see there).
    SetPropA(checkbox, "MeCheckbox", reinterpret_cast<HANDLE>(static_cast<INT_PTR>(1)));
    setControlFont(checkbox, g_fontLabel);
    SetWindowSubclass(checkbox, styledControlProc, 1, 0);
    SendMessageA(checkbox, BM_SETCHECK, checked ? BST_CHECKED : BST_UNCHECKED, 0);
    return checkbox;
}

void createControls() {
    ensureTheme();

    // Sidebar navigation; the pages are placed right of the sidebar.
    g_offsetX = 0;
    g_state.tabLive = makeButton("Live Capture", 16, 214, kSidebarW - 32, 46, IDC_TAB_LIVE);
    g_state.tabOffline = makeButton("Offline Interpolation", 16, 266, kSidebarW - 32, 46, IDC_TAB_OFFLINE);
    g_offsetX = kSidebarW;

    // ---- Live page ----
    HWND liveHeading = makeLabel("TARGET WINDOW", 36, 68, 300);
    g_state.windowList = makeCombo(36, 92, 650, IDC_WINDOW_LIST);
    g_state.refreshButton = makeButton("Refresh", 700, 91, 130, 30, IDC_REFRESH_WINDOWS);
    HWND liveFpsLabel = makeLabel("SOURCE FPS", 36, 138, 100);
    g_state.liveSourceFps = makeCombo(140, 134, 130, IDC_LIVE_SOURCE_FPS);
    populateSourceRates(g_state.liveSourceFps);
    HWND multiplierLabel = makeLabel("OUTPUT MULTIPLIER", 330, 138, 130);
    g_state.liveMultiplier = makeCombo(460, 134, 130, IDC_LIVE_MULTIPLIER);
    populateMultipliers(g_state.liveMultiplier);
    HWND adapterLabel = makeLabel("GPU", 620, 138, 40);
    g_state.liveAdapter = makeCombo(665, 134, 165, IDC_LIVE_ADAPTER);
    SendMessageA(g_state.liveAdapter, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("Auto"));
    SendMessageA(g_state.liveAdapter, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("Integrated GPU"));
    SendMessageA(g_state.liveAdapter, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("Discrete GPU"));
    SendMessageA(g_state.liveAdapter, CB_SETCURSEL, 0, 0);

    HWND gpuLevelsLabel = makeLabel("Pyramid levels (1-8)", 56, 208, 160);
    g_state.gpuLevels = makeEdit("8", 224, 204, 80, IDC_GPU_LEVELS, true);
    HWND gpuMinRefineLabel = makeLabel("Finest searched level", 430, 208, 175);
    g_state.gpuMinRefine = makeEdit("0", 610, 204, 80, IDC_GPU_MIN_REFINE, true);
    HWND gpuCoarseLabel = makeLabel("Coarse radius (0-8)", 56, 248, 160);
    // Matching FlowSettings: a wide coarse window locks onto aliases on
    // periodic content (bar charts, text rows), and a radius of 7 or 8 at
    // the finest level doubles the matcher's inner loop for no measured
    // quality. See FlowSettings::coarseSearchRadius.
    g_state.gpuCoarseRadius = makeEdit("4", 224, 244, 80, IDC_GPU_COARSE_RADIUS, true);
    HWND gpuRefineLabel = makeLabel("Refine radius (0-8)", 430, 248, 175);
    g_state.gpuRefineRadius = makeEdit("6", 610, 244, 80, IDC_GPU_REFINE_RADIUS, true);
    HWND gpuSmoothnessLabel = makeLabel("Smoothness (0-0.1)", 56, 288, 160);
    g_state.gpuSmoothness = makeEdit("0", 224, 284, 80, IDC_GPU_SMOOTHNESS, false);
    HWND flowEngineLabel = makeLabel("FLOW ENGINE", 430, 288, 130);
    g_state.liveFlowEngine = makeCombo(610, 284, 180, IDC_LIVE_FLOW_ENGINE);
    SendMessageA(g_state.liveFlowEngine, CB_ADDSTRING, 0,
                 reinterpret_cast<LPARAM>("FFX Optical Flow"));
    SendMessageA(g_state.liveFlowEngine, CB_ADDSTRING, 0,
                 reinterpret_cast<LPARAM>("SAD block matching"));
    SendMessageA(g_state.liveFlowEngine, CB_ADDSTRING, 0,
                 reinterpret_cast<LPARAM>("Neural motion estimator"));
    SendMessageA(g_state.liveFlowEngine, CB_ADDSTRING, 0,
                 reinterpret_cast<LPARAM>("DIS optical flow"));
    SendMessageA(g_state.liveFlowEngine, CB_SETCURSEL, 3, 0);   // DIS
    // Static UI (HUD, subtitles, logos) is shown where it is instead of
    // being dragged along by the motion underneath it. See UiMaskCS.hlsl.
    g_state.uiMask = makeCheckbox("Keep static UI sharp (HUD, subtitles, logos)", 56, 318, 420, true);
    // DIS: one consistent motion for the band beside a moving object
    // (FlowSettings::occlusionFill, OcclusionFillCS.hlsl).
    g_state.occlusionFill = makeCheckbox("Fill occlusion bands (DIS)", 490, 318, 300, true);
    // FFX engine only (FlowSettings::ffxRefine / ffxOcclusion): refine FFX's
    // whole-pixel 8x8 field per pixel, and - experimental, off by default -
    // estimate both directions so occlusions are handled as with DIS.
    g_state.ffxRefine = makeCheckbox("FFX: refine motion", 56, 346, 220, true);
    g_state.ffxOcclusion = makeCheckbox("FFX: occlusion handling (experimental)", 290, 346, 290, false);
    // AI depth (FlowSettings::aiDepth): the nearer surface wins where two fit
    // equally well. Two-field modes only (DIS, FFX occlusion handling).
    g_state.aiDepth = makeCheckbox("AI depth (DIS, FFX occl.)", 590, 346, 200, true);

    g_state.captureButton = makeButton("START CAPTURE", 36, 388, 180, 40, IDC_START_CAPTURE);
    // How often the AI depth network runs (FlowSettings::aiDepthStride):
    // every frame, or every n-th with the frames between borrowing a
    // neighbour's depth (about 1.3-1.6 ms per run on an RX 9070 XT).
    HWND aiDepthRateLabel = makeLabel("AI DEPTH RATE", 430, 398, 175);
    g_state.aiDepthRate = makeCombo(610, 394, 180, IDC_AI_DEPTH_RATE);
    for (const char* rate : {"Every frame", "Every 2nd frame", "Every 3rd frame", "Every 4th frame"})
        SendMessageA(g_state.aiDepthRate, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(rate));
    SendMessageA(g_state.aiDepthRate, CB_SETCURSEL, 0, 0);
    HWND liveHelp = addControl(
        "STATIC",
        "Values remain fixed during capture. Click Stop Capture to exit; Ctrl+Alt+F1 toggles the overlay, Ctrl+Alt+F3 cycles debug views.",
        SS_LEFT, 36, 436, 760, 40, 0);
    setControlFont(liveHelp, g_fontLabel);
    g_state.liveStatus = addControl("STATIC", "", SS_LEFTNOWORDWRAP | SS_ENDELLIPSIS,
                                     36, 476, 760, 24, IDC_LIVE_STATUS);
    setControlFont(g_state.liveStatus, g_fontBody);
    SetWindowSubclass(g_state.liveStatus, changingStaticProc, 3, 0);

    HWND trainHelp = makeLabel("Choose a video or images. Frames are extracted automatically, with no PNG export.", 56, 552, 740);
    HWND selectImages = makeButton("Choose video/images...", 56, 578, 190, 30, IDC_TRAIN_IMAGES);
    g_state.trainingImagesLabel = makeLabel("No files selected", 260, 583, 530);
    SetWindowSubclass(g_state.trainingImagesLabel, changingStaticProc, 3, 0);
    HWND modelLabel = makeLabel("MODEL FILE", 56, 620, 110);
    g_state.trainingModel = makeEdit("", 170, 615, 410, IDC_TRAIN_MODEL, false);
    HWND saveModel = makeButton("Save as...", 590, 615, 105, 30, IDC_SAVE_TRAIN_MODEL);
    HWND openModel = makeButton("Load...", 705, 615, 105, 30, IDC_OPEN_TRAIN_MODEL);
    g_state.trainButton = makeButton("TRAIN NEURAL MODEL", 56, 665, 180, 40, IDC_RUN_TRAIN);
    g_state.useTrainedModel = makeCheckbox("Use model on next capture", 260, 672, 240, false);
    g_state.trainingStatus = addControl("STATIC", "Choose video or images and save a model to begin.",
                                        SS_LEFTNOWORDWRAP | SS_ENDELLIPSIS, 56, 720, 740, 24, IDC_TRAIN_STATUS);
    setControlFont(g_state.trainingStatus, g_fontBody);
    SetWindowSubclass(g_state.trainingStatus, changingStaticProc, 3, 0);

    g_state.liveControls = {
        liveHeading, g_state.windowList, g_state.refreshButton,
        liveFpsLabel, g_state.liveSourceFps, multiplierLabel, g_state.liveMultiplier,
        adapterLabel, g_state.liveAdapter,
        gpuLevelsLabel, g_state.gpuLevels, gpuMinRefineLabel, g_state.gpuMinRefine,
        gpuCoarseLabel, g_state.gpuCoarseRadius, gpuRefineLabel, g_state.gpuRefineRadius,
        gpuSmoothnessLabel, g_state.gpuSmoothness, flowEngineLabel, g_state.liveFlowEngine,
        g_state.uiMask, g_state.occlusionFill, g_state.ffxRefine, g_state.ffxOcclusion, g_state.aiDepth, aiDepthRateLabel, g_state.aiDepthRate, g_state.captureButton, liveHelp, g_state.liveStatus,
        trainHelp, selectImages, g_state.trainingImagesLabel, modelLabel,
        g_state.trainingModel, saveModel, openModel, g_state.trainButton,
        g_state.useTrainedModel, g_state.trainingStatus
    };

    // ---- Offline page ----
    HWND input0Label = makeLabel("FIRST FRAME", 36, 66, 300);
    g_state.input0 = makeEdit("", 36, 90, 650, IDC_INPUT0, false);
    HWND input0Browse = makeButton("Browse", 700, 88, 130, 30, IDC_BROWSE_INPUT0);
    HWND input1Label = makeLabel("SECOND FRAME", 36, 126, 300);
    g_state.input1 = makeEdit("", 36, 150, 650, IDC_INPUT1, false);
    HWND input1Browse = makeButton("Browse", 700, 148, 130, 30, IDC_BROWSE_INPUT1);
    HWND outputLabel = makeLabel("OUTPUT PNG", 36, 186, 300);
    g_state.output = makeEdit("", 36, 210, 650, IDC_OUTPUT, false);
    HWND outputBrowse = makeButton("Browse", 700, 208, 130, 30, IDC_BROWSE_OUTPUT);
    HWND timeLabel = makeLabel("INTERPOLATION TIME (0-1)", 36, 246, 200);
    g_state.time = makeEdit("0.5", 240, 242, 80, IDC_TIME, false);

    HWND levelsLabel = makeLabel("Pyramid levels", 56, 312, 145);
    g_state.offlineLevels = makeEdit("8", 205, 308, 70, IDC_OFFLINE_LEVELS, true);
    HWND blockLabel = makeLabel("Block size", 56, 344, 145);
    g_state.offlineBlockSize = makeEdit("8", 205, 340, 70, IDC_OFFLINE_BLOCK_SIZE, true);
    HWND minBlockLabel = makeLabel("Minimum block", 56, 376, 145);
    g_state.offlineMinBlock = makeEdit("6", 205, 372, 70, IDC_OFFLINE_MIN_BLOCK, true);
    HWND maxBlockLabel = makeLabel("Maximum block", 56, 408, 145);
    g_state.offlineMaxBlock = makeEdit("14", 205, 404, 70, IDC_OFFLINE_MAX_BLOCK, true);
    HWND coarseLabel = makeLabel("Coarse radius", 56, 440, 145);
    g_state.offlineCoarseRadius = makeEdit("8", 205, 436, 70, IDC_OFFLINE_COARSE_RADIUS, true);
    HWND refineLabel = makeLabel("Refine radius", 56, 472, 145);
    g_state.offlineRefineRadius = makeEdit("4", 205, 468, 70, IDC_OFFLINE_REFINE_RADIUS, true);
    HWND gridLabel = makeLabel("Grid step", 56, 504, 145);
    g_state.offlineGridStep = makeEdit("2", 205, 500, 70, IDC_OFFLINE_GRID_STEP, true);
    HWND subpelLabel = makeLabel("Subpixel mode", 56, 536, 145);
    g_state.offlineSubpel = makeCombo(205, 532, 120, IDC_OFFLINE_SUBPEL);
    populateSubpixelModes(g_state.offlineSubpel);

    HWND smoothnessLabel = makeLabel("Smoothness", 430, 312, 150);
    g_state.offlineSmoothness = makeEdit("0.0005", 580, 308, 80, IDC_OFFLINE_SMOOTHNESS, false);
    HWND detailLabel = makeLabel("Detail strength", 430, 344, 150);
    g_state.offlineDetailStrength = makeEdit("0.65", 580, 340, 80, IDC_OFFLINE_DETAIL_STRENGTH, false);

    g_state.adaptiveBlock = makeCheckbox("Adaptive blocks", 430, 472, 150, true);
    g_state.spatialPredictors = makeCheckbox("Spatial predictors", 580, 472, 170, true);
    g_state.gaussianWeights = makeCheckbox("Gaussian weights", 430, 498, 150, true);
    g_state.bidirectional = makeCheckbox("Bidirectional", 580, 498, 170, true);
    g_state.detailRestore = makeCheckbox("Detail restoration", 430, 524, 150, true);
    g_state.colorClamp = makeCheckbox("Color clamp", 580, 524, 170, true);

    g_state.offlineButton = makeButton("INTERPOLATE", 36, 620, 180, 40, IDC_RUN_OFFLINE);
    g_state.offlineStatus = addControl("STATIC", "Select two PNG frames to begin.",
                                       SS_LEFTNOWORDWRAP | SS_ENDELLIPSIS, 232, 628, 590, 24, IDC_OFFLINE_STATUS);
    setControlFont(g_state.offlineStatus, g_fontBody);
    SetWindowSubclass(g_state.offlineStatus, changingStaticProc, 3, 0);

    g_state.offlineControls = {
        input0Label, g_state.input0, input0Browse, input1Label, g_state.input1, input1Browse,
        outputLabel, g_state.output, outputBrowse, timeLabel, g_state.time,
        levelsLabel, g_state.offlineLevels, blockLabel, g_state.offlineBlockSize,
        minBlockLabel, g_state.offlineMinBlock, maxBlockLabel, g_state.offlineMaxBlock,
        coarseLabel, g_state.offlineCoarseRadius, refineLabel, g_state.offlineRefineRadius,
        gridLabel, g_state.offlineGridStep, subpelLabel, g_state.offlineSubpel,
        smoothnessLabel, g_state.offlineSmoothness,
        detailLabel, g_state.offlineDetailStrength,
        g_state.adaptiveBlock, g_state.spatialPredictors,
        g_state.gaussianWeights, g_state.bidirectional,
        g_state.detailRestore, g_state.colorClamp,
        g_state.offlineButton, g_state.offlineStatus
    };

    g_offsetX = 0;
    applyPageVisibility();
    refreshWindowList();
}

// ------------------------------------------------------------------
// Custom painting
// ------------------------------------------------------------------
void paintSectionPanel(HDC hdc, const RECT& rc, const char* title) {
    paintCard(hdc, rc, ColPanel, 16);
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, ColText);
    RECT titleRect = { rc.left + 20, rc.top + 8, rc.right - 16, rc.top + 30 };
    HFONT oldFont = static_cast<HFONT>(SelectObject(hdc, g_fontSection));
    DrawTextA(hdc, title, -1, &titleRect, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, oldFont);
}

// The emblem: a ring with circuit nodes around a frame in motion - a bright
// core and the trails it leaves - with a soft glow behind.
void paintEmblem(Gdiplus::Graphics& g, float cx, float cy) {
    for (int i = 6; i >= 1; --i) {
        const float r = 34.0f + 5.0f * i;
        Gdiplus::SolidBrush glow(gpColor(RGB(255, 255, 255), 5));
        g.FillEllipse(&glow, cx - r, cy - r, 2 * r, 2 * r);
    }
    Gdiplus::Pen ring(gpColor(RGB(255, 255, 255), 215), 1.6f);
    g.DrawEllipse(&ring, cx - 30.0f, cy - 30.0f, 60.0f, 60.0f);
    Gdiplus::Pen thin(gpColor(RGB(255, 255, 255), 90), 1.0f);
    g.DrawEllipse(&thin, cx - 38.0f, cy - 38.0f, 76.0f, 76.0f);
    // Circuit nodes on the outer ring, with short traces outward.
    Gdiplus::SolidBrush node(gpColor(RGB(255, 255, 255), 230));
    for (float a : {-140.0f, -40.0f, 40.0f, 140.0f, -90.0f, 90.0f}) {
        const float rad = a * 3.14159265f / 180.0f;
        const float x = cx + 38.0f * std::cos(rad), y = cy + 38.0f * std::sin(rad);
        const float ox = cx + 48.0f * std::cos(rad), oy = cy + 48.0f * std::sin(rad);
        g.DrawLine(&thin, x, y, ox, oy);
        g.FillEllipse(&node, ox - 2.5f, oy - 2.5f, 5.0f, 5.0f);
    }
    // Motion trails left of the core, fading.
    for (int i = 0; i < 3; ++i) {
        const float y = cy - 9.0f + 9.0f * i;
        const float len = i == 1 ? 26.0f : 18.0f;
        Gdiplus::LinearGradientBrush fade(Gdiplus::PointF(cx - len - 4.0f, y), Gdiplus::PointF(cx, y),
                                          gpColor(RGB(255, 255, 255), 0), gpColor(RGB(255, 255, 255), 220));
        Gdiplus::Pen trail(&fade, i == 1 ? 2.4f : 1.6f);
        trail.SetStartCap(Gdiplus::LineCapRound);
        trail.SetEndCap(Gdiplus::LineCapRound);
        g.DrawLine(&trail, cx - len - 4.0f, y, cx + 2.0f, y);
    }
    Gdiplus::SolidBrush core(gpColor(RGB(255, 255, 255)));
    g.FillEllipse(&core, cx + 1.0f, cy - 7.0f, 14.0f, 14.0f);
    Gdiplus::Pen halo(gpColor(RGB(255, 255, 255), 120), 1.2f);
    g.DrawEllipse(&halo, cx - 3.0f, cy - 11.0f, 22.0f, 22.0f);
}

void paintSidebar(HDC hdc, const RECT& client) {
    RECT side = { 0, 0, kSidebarW, client.bottom };
    FillRect(hdc, &side, g_brSidebar);
    RECT edge = { kSidebarW - 1, 0, kSidebarW, client.bottom };
    HBRUSH line = CreateSolidBrush(RGB(30, 30, 34));
    FillRect(hdc, &edge, line);
    DeleteObject(line);

    Gdiplus::Graphics g(hdc);
    prepare(g);
    const float cx = kSidebarW / 2.0f;
    paintEmblem(g, cx, 84.0f);

    // Flowing lines and a few points of light, below the navigation.
    const float top = 340.0f, bottom = static_cast<float>(client.bottom) - 120.0f;
    if (bottom > top + 40.0f) {
        for (int i = 0; i < 4; ++i) {
            const float o = 18.0f * i;
            Gdiplus::Pen wave(gpColor(RGB(255, 255, 255), 34 - 6 * i), 1.0f);
            g.DrawBezier(&wave, -10.0f, bottom - o, 70.0f, bottom - 120.0f - o, 140.0f, top + 60.0f + o,
                         static_cast<float>(kSidebarW) + 10.0f, top + o);
        }
        Gdiplus::SolidBrush star(gpColor(RGB(255, 255, 255), 110));
        const float stars[][2] = {{38, 380}, {176, 410}, {96, 470}, {200, 520}, {54, 560}, {150, 600}};
        for (const auto& s : stars)
            if (s[1] < bottom) g.FillEllipse(&star, s[0], s[1], 2.0f, 2.0f);
    }

    SetBkMode(hdc, TRANSPARENT);
    HFONT oldFont = static_cast<HFONT>(SelectObject(hdc, g_fontBrand));
    SetTextColor(hdc, ColText);
    RECT brand = { 0, 144, kSidebarW, 164 };
    DrawTextA(hdc, "M O T I O N   E N H A N C E R", -1, &brand, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, g_fontLabel);
    SetTextColor(hdc, ColDim);
    RECT tagline = { 0, 166, kSidebarW, 184 };
    DrawTextA(hdc, "Real-time frame generation", -1, &tagline, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    // Status card: what the program is doing now.
    const bool capturing = g_state.captureStopEvent != nullptr;
    const bool busy = !capturing && g_state.trainButton && !IsWindowEnabled(g_state.trainButton);
    RECT card = { 16, client.bottom - 92, kSidebarW - 16, client.bottom - 16 };
    paintCard(hdc, card, ColPanel, 14);
    {
        const float dx = card.left + 26.0f, dy = (card.top + card.bottom) / 2.0f;
        const COLORREF dotColor = capturing || busy ? RGB(255, 255, 255) : RGB(90, 90, 98);
        if (capturing || busy)
            for (int i = 3; i >= 1; --i) {
                Gdiplus::SolidBrush glow(gpColor(RGB(255, 255, 255), 22));
                const float r = 5.0f + 3.0f * i;
                g.FillEllipse(&glow, dx - r, dy - r, 2 * r, 2 * r);
            }
        Gdiplus::SolidBrush dot(gpColor(dotColor));
        g.FillEllipse(&dot, dx - 5.0f, dy - 5.0f, 10.0f, 10.0f);
    }
    SelectObject(hdc, g_fontButton);
    SetTextColor(hdc, ColText);
    RECT state = { card.left + 46, card.top + 16, card.right - 10, card.top + 38 };
    DrawTextA(hdc, capturing ? "Capturing" : (busy ? "Working..." : "Idle"), -1, &state,
              DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, g_fontLabel);
    SetTextColor(hdc, ColDim);
    RECT hint = { card.left + 46, card.top + 38, card.right - 10, card.top + 58 };
    DrawTextA(hdc, capturing ? "Ctrl+Alt+F1 overlay" : "Ready", -1, &hint, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, oldFont);
}

void paintPageHeader(HDC hdc) {
    const bool live = g_state.activeTab == 0;
    SetBkMode(hdc, TRANSPARENT);
    HFONT oldFont = static_cast<HFONT>(SelectObject(hdc, g_fontTitle));
    SetTextColor(hdc, ColText);
    RECT title = { kSidebarW + 36, 10, kSidebarW + 830, 44 };
    DrawTextA(hdc, live ? "Live Capture" : "Offline Interpolation", -1, &title, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, g_fontLabel);
    SetTextColor(hdc, ColDim);
    RECT subtitle = { kSidebarW + 36, 42, kSidebarW + 830, 60 };
    DrawTextA(hdc, live ? "Generate frames for any window in real time on the GPU."
                        : "Interpolate a frame between two images.",
              -1, &subtitle, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, oldFont);
}

// Rounded frames around the visible edit fields, drawn on the window
// behind them (they have no border of their own).
void paintFieldFrames(HDC hdc, HWND hwnd) {
    for (HWND child = GetWindow(hwnd, GW_CHILD); child; child = GetWindow(child, GW_HWNDNEXT)) {
        if (!IsWindowVisible(child)) continue;
        char cls[16] = {};
        GetClassNameA(child, cls, sizeof(cls));
        const bool combo = _stricmp(cls, "ComboBox") == 0;
        if (!combo && _stricmp(cls, "Edit") != 0) continue;
        RECT rc = {};
        GetWindowRect(child, &rc);
        MapWindowPoints(HWND_DESKTOP, hwnd, reinterpret_cast<POINT*>(&rc), 2);
        InflateRect(&rc, combo ? 2 : 6, combo ? 2 : 3);
        fillRoundRect(hdc, rc, ColField, 8);
        frameRoundRect(hdc, rc, GetFocus() == child ? ColAccentDk : ColBorder, 8);
    }
}

void paintClient(HWND hwnd) {
    PAINTSTRUCT paint = {};
    HDC screen = BeginPaint(hwnd, &paint);
    RECT client = {};
    GetClientRect(hwnd, &client);
    // Drawn off screen, then copied: no flicker.
    HDC hdc = CreateCompatibleDC(screen);
    HBITMAP bitmap = CreateCompatibleBitmap(screen, client.right, client.bottom);
    HGDIOBJ oldBitmap = SelectObject(hdc, bitmap);

    FillRect(hdc, &client, g_brBg);
    paintSidebar(hdc, client);
    paintPageHeader(hdc);
    if (g_state.activeTab == 0) {
        paintSectionPanel(hdc, LivePanelRect, "Advanced GPU flow");
        paintSectionPanel(hdc, TrainPanelRect, "Train neural flow");
    } else {
        paintSectionPanel(hdc, OfflinePanelRect, "Advanced interpolation");
    }
    paintFieldFrames(hdc, hwnd);

    BitBlt(screen, 0, 0, client.right, client.bottom, hdc, 0, 0, SRCCOPY);
    SelectObject(hdc, oldBitmap);
    DeleteObject(bitmap);
    DeleteDC(hdc);
    EndPaint(hwnd, &paint);
}

// ------------------------------------------------------------------
// Window procedure
// ------------------------------------------------------------------
LRESULT CALLBACK guiWindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_CREATE:
            g_state.hwnd = hwnd;
            createControls();
            return 0;

        case WM_PAINT:
            paintClient(hwnd);
            return 0;

        case WM_ERASEBKGND:
            return 1;

        case WM_DRAWITEM: {
            const DRAWITEMSTRUCT* draw = reinterpret_cast<const DRAWITEMSTRUCT*>(lParam);
            if (!draw) break;
            if (draw->CtlType == ODT_BUTTON) {
                drawStyledButton(draw);
                return TRUE;
            }
            if (draw->CtlType == ODT_COMBOBOX) {
                drawStyledComboItem(draw);
                return TRUE;
            }
            break;
        }

        case WM_MEASUREITEM: {
            MEASUREITEMSTRUCT* measure = reinterpret_cast<MEASUREITEMSTRUCT*>(lParam);
            if (measure && measure->CtlType == ODT_COMBOBOX) {
                measure->itemHeight = 26;
                return TRUE;
            }
            break;
        }

        case WM_CTLCOLORSTATIC: {
            HDC hdc = reinterpret_cast<HDC>(wParam);
            HWND control = reinterpret_cast<HWND>(lParam);
            int controlId = GetDlgCtrlID(control);
            RECT rc = {};
            GetWindowRect(control, &rc);
            MapWindowPoints(HWND_DESKTOP, hwnd, reinterpret_cast<POINT*>(&rc), 2);
            const int cx = (rc.left + rc.right) / 2;
            const int cy = (rc.top + rc.bottom) / 2;
            auto within = [cx, cy](const RECT& panel) {
                return cx >= panel.left && cx < panel.right && cy >= panel.top && cy < panel.bottom;
            };
            const bool onPanel = g_state.activeTab == 0
                ? within(LivePanelRect) || within(TrainPanelRect) : within(OfflinePanelRect);
            SetBkMode(hdc, OPAQUE);
            SetBkColor(hdc, onPanel ? ColPanel : ColBg);
            if (controlId == IDC_LIVE_STATUS || controlId == IDC_OFFLINE_STATUS || controlId == IDC_TRAIN_STATUS) {
                SetTextColor(hdc, ColAccent);
            } else {
                SetTextColor(hdc, ColDim);
            }
            return reinterpret_cast<LRESULT>(onPanel ? g_brPanel : g_brBg);
        }

        case WM_CTLCOLOREDIT: {
            HDC hdc = reinterpret_cast<HDC>(wParam);
            SetBkColor(hdc, ColField);
            SetTextColor(hdc, ColText);
            return reinterpret_cast<LRESULT>(g_brField);
        }

        case WM_CTLCOLORLISTBOX: {
            HDC hdc = reinterpret_cast<HDC>(wParam);
            SetBkColor(hdc, ColPanel);
            SetTextColor(hdc, ColText);
            return reinterpret_cast<LRESULT>(g_brPanel);
        }

        case WM_COMMAND:
            // A click (or Space) on a checkbox flips its state; they keep
            // it themselves (styledControlProc).
            // The sidebar's status card follows the jobs.
            if (HIWORD(wParam) == BN_CLICKED) {
                RECT side = { 0, 0, kSidebarW, 4000 };
                InvalidateRect(hwnd, &side, FALSE);
            }
            // Edit focus changes repaint their frames.
            if (HIWORD(wParam) == EN_SETFOCUS || HIWORD(wParam) == EN_KILLFOCUS) {
                RECT rc = {};
                GetWindowRect(reinterpret_cast<HWND>(lParam), &rc);
                MapWindowPoints(HWND_DESKTOP, hwnd, reinterpret_cast<POINT*>(&rc), 2);
                InflateRect(&rc, 8, 5);
                InvalidateRect(hwnd, &rc, FALSE);
            }
            if (HIWORD(wParam) == BN_CLICKED && lParam &&
                GetPropA(reinterpret_cast<HWND>(lParam), "MeCheckbox")) {
                HWND box = reinterpret_cast<HWND>(lParam);
                const bool checked = SendMessageA(box, BM_GETCHECK, 0, 0) == BST_CHECKED;
                SendMessageA(box, BM_SETCHECK, checked ? BST_UNCHECKED : BST_CHECKED, 0);
                return 0;
            }
            switch (LOWORD(wParam)) {
                case IDC_TAB_LIVE:
                    if (g_state.activeTab != 0) {
                        g_state.activeTab = 0;
                        applyPageVisibility();
                    }
                    return 0;
                case IDC_TAB_OFFLINE:
                    if (g_state.activeTab != 1) {
                        g_state.activeTab = 1;
                        applyPageVisibility();
                    }
                    return 0;
                case IDC_REFRESH_WINDOWS: refreshWindowList(); return 0;
                case IDC_START_CAPTURE: startCapture(); return 0;
                case IDC_TRAIN_IMAGES: selectTrainingImages(); return 0;
                case IDC_SAVE_TRAIN_MODEL: selectTrainingModel(true); return 0;
                case IDC_OPEN_TRAIN_MODEL: selectTrainingModel(false); return 0;
                case IDC_RUN_TRAIN: trainModel(); return 0;
                case IDC_BROWSE_INPUT0: selectFile(g_state.input0, false); return 0;
                case IDC_BROWSE_INPUT1: selectFile(g_state.input1, false); return 0;
                case IDC_BROWSE_OUTPUT: selectFile(g_state.output, true); return 0;
                case IDC_RUN_OFFLINE: runOffline(); return 0;
            }
            break;

        case WM_PROCESS_COMPLETE: {
            InvalidateRect(hwnd, nullptr, FALSE);
            DWORD exitCode = static_cast<DWORD>(wParam);
            JobType jobType = static_cast<JobType>(lParam);
            if (jobType == JobType::Live) {
                if (g_state.captureStopEvent) {
                    CloseHandle(g_state.captureStopEvent);
                    g_state.captureStopEvent = nullptr;
                }
                EnableWindow(g_state.captureButton, TRUE);
                SetWindowTextA(g_state.captureButton, "START CAPTURE");
                EnableWindow(g_state.trainButton, TRUE);
                SetWindowTextA(g_state.liveStatus, exitCode == 0 ? "Capture stopped." : "Capture ended with an error.");
            } else if (jobType == JobType::Offline) {
                EnableWindow(g_state.offlineButton, TRUE);
                EnableWindow(g_state.trainButton, TRUE);
                SetWindowTextA(g_state.offlineStatus, exitCode == 0 ? "Interpolation complete." : "Interpolation failed.");
            }
            return 0;
        }

        case WM_TRAIN_PROGRESS: {
            std::string* progress = reinterpret_cast<std::string*>(lParam);
            if (progress) {
                SetWindowTextA(g_state.trainingStatus, progress->c_str());
                delete progress;
            }
            return 0;
        }

        case WM_TRAIN_COMPLETE: {
            InvalidateRect(hwnd, nullptr, FALSE);
            const DWORD exitCode = static_cast<DWORD>(wParam);
            std::string* result = reinterpret_cast<std::string*>(lParam);
            EnableWindow(g_state.trainButton, TRUE);
            EnableWindow(g_state.captureButton, TRUE);
            EnableWindow(g_state.offlineButton, TRUE);
            const bool modelSaved = result && result->rfind("Model saved", 0) == 0;
            if (exitCode == 0 || modelSaved) {
                SendMessageA(g_state.useTrainedModel, BM_SETCHECK, BST_CHECKED, 0);
                SendMessageA(g_state.liveFlowEngine, CB_SETCURSEL, 2, 0);
                InvalidateRect(g_state.useTrainedModel, nullptr, FALSE);
            }
            if (result) {
                SetWindowTextA(g_state.trainingStatus, result->c_str());
                delete result;
            }
            return 0;
        }

        case WM_DESTROY: {
            MSG queued = {};
            while (PeekMessageA(&queued, hwnd, WM_TRAIN_COMPLETE, WM_TRAIN_COMPLETE, PM_REMOVE))
                delete reinterpret_cast<std::string*>(queued.lParam);
            while (PeekMessageA(&queued, hwnd, WM_TRAIN_PROGRESS, WM_TRAIN_PROGRESS, PM_REMOVE))
                delete reinterpret_cast<std::string*>(queued.lParam);
            if (g_state.captureStopEvent) {
                SetEvent(g_state.captureStopEvent);
                CloseHandle(g_state.captureStopEvent);
                g_state.captureStopEvent = nullptr;
            }
            g_state.hwnd = nullptr;
            PostQuitMessage(0);
            return 0;
        }
    }
    return DefWindowProcA(hwnd, message, wParam, lParam);
}

} // namespace

int runGuiApplication() {
    SetProcessDPIAware();
    HWND console = GetConsoleWindow();
    if (console) ShowWindow(console, SW_HIDE);

    INITCOMMONCONTROLSEX controls = {};
    controls.dwSize = sizeof(controls);
    controls.dwICC = ICC_STANDARD_CLASSES;
    InitCommonControlsEx(&controls);

    Gdiplus::GdiplusStartupInput gdiplusInput;
    Gdiplus::GdiplusStartup(&g_gdiplusToken, &gdiplusInput, nullptr);
    ensureTheme();

    HINSTANCE instance = GetModuleHandleA(nullptr);
    WNDCLASSEXA windowClass = {};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = guiWindowProc;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursor(nullptr, IDC_ARROW);
    windowClass.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    windowClass.hbrBackground = nullptr;
    windowClass.lpszClassName = "MotionEnhancerGuiClass";
    RegisterClassExA(&windowClass);

    HWND window = CreateWindowExA(
        0,
        windowClass.lpszClassName,
        "Motion Enhancer",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_CLIPCHILDREN,
        CW_USEDEFAULT, CW_USEDEFAULT, 900 + kSidebarW, 820,
        nullptr, nullptr, instance, nullptr
    );
    if (!window) return 1;

    // Dark immersive title bar where supported.
    BOOL dark = TRUE;
    DwmSetWindowAttribute(window, 20, &dark, sizeof(dark));
    DwmSetWindowAttribute(window, 19, &dark, sizeof(dark));
    COLORREF captionColor = ColBg;
    DwmSetWindowAttribute(window, 1024, &captionColor, sizeof(captionColor));

    // Test hook for offline UI checks: MOTION_ENHANCER_GUI_SNAPSHOT=<png>
    // renders this window off screen into a PNG and exits (page 1 when
    // MOTION_ENHANCER_GUI_PAGE=1). Nothing on the screen is read.
    char snapshot[MAX_PATH] = {};
    if (GetEnvironmentVariableA("MOTION_ENHANCER_GUI_SNAPSHOT", snapshot, sizeof(snapshot))) {
        char page[4] = {};
        if (GetEnvironmentVariableA("MOTION_ENHANCER_GUI_PAGE", page, sizeof(page)) && page[0] == '1') {
            g_state.activeTab = 1;
            applyPageVisibility();
        }
        SetWindowPos(window, nullptr, -12000, -12000, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
        ShowWindow(window, SW_SHOWNOACTIVATE);
        UpdateWindow(window);
        MSG pending;
        for (int i = 0; i < 50; ++i) {
            while (PeekMessageA(&pending, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&pending);
                DispatchMessageA(&pending);
            }
            Sleep(10);
        }
        RECT client = {};
        GetClientRect(window, &client);
        HDC screen = GetDC(window);
        HDC memory = CreateCompatibleDC(screen);
        HBITMAP bitmap = CreateCompatibleBitmap(screen, client.right, client.bottom);
        HGDIOBJ old = SelectObject(memory, bitmap);
        PrintWindow(window, memory, PW_CLIENTONLY | 2 /* PW_RENDERFULLCONTENT */);
        BITMAPINFO info = {};
        info.bmiHeader.biSize = sizeof(info.bmiHeader);
        info.bmiHeader.biWidth = client.right;
        info.bmiHeader.biHeight = -client.bottom;
        info.bmiHeader.biPlanes = 1;
        info.bmiHeader.biBitCount = 32;
        std::vector<uint8_t> pixels(static_cast<size_t>(client.right) * client.bottom * 4);
        SelectObject(memory, old);
        GetDIBits(memory, bitmap, 0, client.bottom, pixels.data(), &info, DIB_RGB_COLORS);
        Image image(client.right, client.bottom, 3);
        for (int y = 0; y < client.bottom; ++y)
            for (int x = 0; x < client.right; ++x)
                for (int c = 0; c < 3; ++c)
                    image.set(x, y, c, pixels[(static_cast<size_t>(y) * client.right + x) * 4 + (2 - c)]);
        DeleteObject(bitmap);
        DeleteDC(memory);
        ReleaseDC(window, screen);
        const bool saved = image.savePNG(snapshot);
        DestroyWindow(window);
        Gdiplus::GdiplusShutdown(g_gdiplusToken);
        return saved ? 0 : 1;
    }

    ShowWindow(window, SW_SHOW);
    UpdateWindow(window);

    MSG message;
    while (GetMessageA(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageA(&message);
    }
    Gdiplus::GdiplusShutdown(g_gdiplusToken);
    return static_cast<int>(message.wParam);
}
