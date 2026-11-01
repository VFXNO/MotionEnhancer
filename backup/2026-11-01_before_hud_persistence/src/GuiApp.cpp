#include "GuiApp.h"
#include "AppResources.h"
#include "FlowEngine.h"
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
#include "BrandLogo.h"
#include "BrandCharacter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <fstream>
#include <filesystem>
#include <regex>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

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
constexpr int IDC_OPEN_TRAINER = 116;
constexpr int IDC_USE_AI_ENGINE = 117;
constexpr int IDC_AI_REFRESH = 118;
constexpr int IDC_AI_STATUS = 119;
constexpr int IDC_AI_REFINEMENT = 120;
constexpr int IDC_AI_UI_REGIONS = 121;
constexpr int IDC_UI_MASK = 122;
constexpr int IDC_UI_DETECTOR = 123;
constexpr int IDC_AI_TEMPORAL_UI = 124;
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
constexpr UINT WM_LIVE_STATUS = WM_APP + 4;

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
constexpr RECT AiPanelRect = { kSidebarW + 36, 84, kSidebarW + 36 + 794, 84 + 340 };

struct GuiState {
    HWND hwnd = nullptr;
    int activeTab = 0;
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
    HWND aiUiRegions = nullptr;
    HWND aiTemporalUi = nullptr;
    HWND uiDetector = nullptr;
    HWND ffxRefine = nullptr;
    HWND occlusionFill = nullptr;
    HWND ffxOcclusion = nullptr;
    HWND aiDepth = nullptr;
    HWND aiDepthRate = nullptr;
    HWND aiBlendMask = nullptr;
    HWND aiRefinement = nullptr;
    HWND aiRefinementHelp = nullptr;
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
    std::vector<HWND> aiControls;
    HWND aiStatus = nullptr;
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
BrandLogo g_brandLogo;
BrandCharacter g_brandCharacter;
int g_offsetX = 0;   // added to x by addControl: the page area starts right of the sidebar

// ---- Animation (WM_TIMER kAnimTimer, animate()) ----
// Ambient motion (the emblem, the sidebar's lines, the status dot) runs
// only while the window is in front; transitions (the navigation
// highlight, hover fades, the page title) always finish.
constexpr UINT_PTR kAnimTimer = 1;
constexpr int kNavCount = 3;
constexpr RECT kNavRects[kNavCount] = { {16, 214, kSidebarW - 16, 260}, {16, 266, kSidebarW - 16, 312},
                                        {16, 318, kSidebarW - 16, 364} };
constexpr const char* kNavLabels[kNavCount] = { "Live Capture", "Offline Interpolation", "AI Model" };
double g_time = 0.0;                 // seconds of ambient motion
LARGE_INTEGER g_lastTick = {};
float g_navY = -1.0f;                // top of the sliding navigation highlight
float g_navHoverT[kNavCount] = {};           // hover fade per navigation item, 0..1
int g_navHover = -1;
bool g_trackingLeave = false;
float g_pageT = 1.0f;                // page title entrance, 0..1
std::vector<std::pair<HWND, float>> g_hoverFades;   // owner-drawn controls' hover fade, 0..1

float& hoverFade(HWND control) {
    for (auto& entry : g_hoverFades)
        if (entry.first == control) return entry.second;
    g_hoverFades.emplace_back(control, 0.0f);
    return g_hoverFades.back().second;
}

float easeOut(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return 1.0f - (1.0f - t) * (1.0f - t) * (1.0f - t);
}

bool fontExists(const char* face) {
    HDC hdc = GetDC(nullptr);
    LOGFONTA query = {};
    query.lfCharSet = DEFAULT_CHARSET;
    strncpy_s(query.lfFaceName, face, _TRUNCATE);
    bool found = false;
    EnumFontFamiliesExA(hdc, &query, [](const LOGFONTA*, const TEXTMETRICA*, DWORD, LPARAM found) -> int {
        *reinterpret_cast<bool*>(found) = true;
        return 0;
    }, reinterpret_cast<LPARAM>(&found), 0);
    ReleaseDC(nullptr, hdc);
    return found;
}

HFONT createFont(const char* face, int height, int weight) {
    return CreateFontA(
        height, 0, 0, 0, weight, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, face);
}

void ensureTheme() {
    if (g_fontBody) return;
    // Windows 11's variable Segoe where present.
    const char* text = fontExists("Segoe UI Variable Text") ? "Segoe UI Variable Text" : "Segoe UI";
    const char* display = fontExists("Segoe UI Variable Display") ? "Segoe UI Variable Display" : "Segoe UI";
    g_fontBody = createFont(text, -14, FW_NORMAL);
    g_fontLabel = createFont(text, -13, FW_NORMAL);
    g_fontButton = createFont(text, -14, FW_SEMIBOLD);
    g_fontSection = createFont(display, -16, FW_SEMIBOLD);
    g_fontTitle = createFont(display, -30, FW_SEMIBOLD);
    g_fontBrand = createFont(display, -13, FW_SEMIBOLD);
    g_fontNav = createFont(text, -15, FW_NORMAL);
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
                         : g_state.activeTab == 1 ? within(OfflinePanelRect) : within(AiPanelRect);
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
        case WM_ERASEBKGND:
            // Everything is painted in WM_DRAWITEM, off screen; erasing
            // first flashed the system's light button colour on a click.
            return 1;
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

void paintStyledComboItem(const DRAWITEMSTRUCT* draw);

// The theme's parts of a combo, over what the system drew into `hdc`: its
// square border painted out (the window behind draws a rounded frame,
// paintFieldFrames) and a plain caret for the dropdown button.
void paintComboChrome(HDC hdc, const RECT& rc) {
    HBRUSH field = CreateSolidBrush(ColField);
    for (int i = 0; i < 3; ++i) {
        RECT edge = rc;
        InflateRect(&edge, -i, -i);
        FrameRect(hdc, &edge, field);
    }
    DeleteObject(field);
    RECT button = { rc.right - 28, rc.top + 1, rc.right - 1, rc.bottom - 1 };
    FillRect(hdc, &button, g_brField);
    const int cx = (button.left + button.right) / 2;
    const int cy = (button.top + button.bottom) / 2;
    HPEN pen = CreatePen(PS_SOLID, 2, ColDim);
    HPEN oldPen = static_cast<HPEN>(SelectObject(hdc, pen));
    MoveToEx(hdc, cx - 5, cy - 2, nullptr);
    LineTo(hdc, cx, cy + 3);
    LineTo(hdc, cx + 5, cy - 2);
    SelectObject(hdc, oldPen);
    DeleteObject(pen);
}

LRESULT CALLBACK comboNcProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam,
                             UINT_PTR subclassId, DWORD_PTR refData) {
    switch (message) {
        case WM_ERASEBKGND:
            return 1;
        case WM_PAINT: {
            // Painted entirely here, off screen, and copied once: the field,
            // the selected item (the same painter as the list's items) and
            // the caret. The system's own paint drew a white square border
            // that the theme then painted over, which flickered on hover.
            PAINTSTRUCT paint = {};
            HDC screen = BeginPaint(hwnd, &paint);
            RECT rc = {};
            GetClientRect(hwnd, &rc);
            HDC memory = CreateCompatibleDC(screen);
            HBITMAP bitmap = CreateCompatibleBitmap(screen, std::max(1L, rc.right), std::max(1L, rc.bottom));
            HGDIOBJ old = SelectObject(memory, bitmap);
            HFONT font = reinterpret_cast<HFONT>(SendMessageA(hwnd, WM_GETFONT, 0, 0));
            HGDIOBJ oldFont = font ? SelectObject(memory, font) : nullptr;
            FillRect(memory, &rc, g_brField);
            DRAWITEMSTRUCT item = {};
            item.CtlType = ODT_COMBOBOX;
            item.CtlID = static_cast<UINT>(GetDlgCtrlID(hwnd));
            item.itemID = static_cast<UINT>(SendMessageA(hwnd, CB_GETCURSEL, 0, 0));
            item.itemAction = ODA_DRAWENTIRE;
            item.itemState = ODS_COMBOBOXEDIT | (IsWindowEnabled(hwnd) ? 0 : ODS_DISABLED);
            item.hwndItem = hwnd;
            item.hDC = memory;
            item.rcItem = { 1, 1, rc.right - 1, rc.bottom - 1 };
            paintStyledComboItem(&item);
            paintComboChrome(memory, rc);
            BitBlt(screen, 0, 0, rc.right, rc.bottom, memory, 0, 0, SRCCOPY);
            if (oldFont) SelectObject(memory, oldFont);
            SelectObject(memory, old);
            DeleteObject(bitmap);
            DeleteDC(memory);
            EndPaint(hwnd, &paint);
            return 0;
        }
    }
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
    }
    return result;
}

void paintStyledButton(const DRAWITEMSTRUCT* draw) {
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
    const float hover = disabled ? 0.0f : hoverFade(hwnd);
    int id = draw->CtlID;

    bool primary = (id == IDC_START_CAPTURE || id == IDC_RUN_OFFLINE || id == IDC_OPEN_TRAINER);
    // Style bits are ambiguous (BS_OWNERDRAW overlaps BS_CHECKBOX), so
    // checkboxes are tagged explicitly with a window property.
    bool isCheckbox = GetPropA(hwnd, "MeCheckbox") != nullptr;

    if (isCheckbox) {
        bool checked = SendMessageA(hwnd, BM_GETCHECK, 0, 0) == BST_CHECKED;
        if (hover > 0.01f) fillRoundRect(hdc, rc, ColPanelHi, 8, static_cast<int>(255 * hover));
        RECT box = { rc.left + 2, rc.top + (rc.bottom - rc.top - 16) / 2,
                     rc.left + 18, rc.top + (rc.bottom - rc.top + 16) / 2 };
        fillRoundRect(hdc, box, checked ? ColAccent : ColField, 5);
        frameRoundRect(hdc, box, checked ? ColAccent : lerpColor(ColBorderHi, ColAccentDk, hover), 5);
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

    int radius = (rc.bottom - rc.top) / 2;   // pills
    if (primary) {
        if (disabled) {
            fillRoundRect(hdc, rc, ColPanelHi, radius);
        } else {
            // A white pill lit from above; on hover a light band sweeps
            // across it.
            Gdiplus::Graphics g(hdc);
            prepare(g);
            const float x = static_cast<float>(rc.left), y = static_cast<float>(rc.top);
            const float w = static_cast<float>(rc.right - rc.left), h = static_cast<float>(rc.bottom - rc.top);
            Gdiplus::GraphicsPath pill;
            roundPath(pill, x, y, w, h, h / 2.0f);
            const COLORREF top = pressed ? ColAccentDk : lerpColor(RGB(236, 236, 240), RGB(255, 255, 255), hover);
            const COLORREF bottom = pressed ? ColAccentDk : lerpColor(RGB(196, 196, 204), RGB(228, 228, 234), hover);
            Gdiplus::LinearGradientBrush fill(Gdiplus::PointF(0.0f, y), Gdiplus::PointF(0.0f, y + h),
                                              gpColor(top), gpColor(bottom));
            g.FillPath(&fill, &pill);
            if (hover > 0.01f) {
                g.SetClip(&pill);
                const float sweep = static_cast<float>(std::fmod(g_time * 0.9, 1.6)) / 1.6f;
                const float bx = x - 60.0f + (w + 120.0f) * sweep;
                Gdiplus::LinearGradientBrush band(Gdiplus::PointF(bx - 40.0f, y), Gdiplus::PointF(bx + 40.0f, y),
                                                  gpColor(RGB(255, 255, 255), 0), gpColor(RGB(255, 255, 255), 0));
                const Gdiplus::Color colors[3] = {gpColor(RGB(255, 255, 255), 0),
                                                  gpColor(RGB(255, 255, 255), static_cast<int>(150 * hover)),
                                                  gpColor(RGB(255, 255, 255), 0)};
                const Gdiplus::REAL positions[3] = {0.0f, 0.5f, 1.0f};
                band.SetInterpolationColors(colors, positions, 3);
                g.FillRectangle(&band, bx - 40.0f, y, 80.0f, h);
                g.ResetClip();
            }
        }
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, disabled ? ColDim : ColTextOn);
    } else {
        fillRoundRect(hdc, rc, disabled ? ColPanel : lerpColor(ColPanelHi, RGB(38, 38, 44), hover), radius);
        frameRoundRect(hdc, rc, lerpColor(ColBorder, RGB(110, 110, 120), hover), radius);
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, disabled ? ColDim : ColText);
    }
    RECT textRect = rc;
    if (pressed) OffsetRect(&textRect, 1, 1);
    DrawTextA(hdc, text, -1, &textRect, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
}

// Runs `paint` on a copy of `draw` whose DC is an off-screen bitmap of the
// item, with the control's font, then copies the bitmap to the item.
template <typename Paint>
void drawBuffered(const DRAWITEMSTRUCT* draw, Paint paint) {
    const RECT& rc = draw->rcItem;
    const int w = rc.right - rc.left, h = rc.bottom - rc.top;
    if (w <= 0 || h <= 0) return;
    HDC memory = CreateCompatibleDC(draw->hDC);
    HBITMAP bitmap = CreateCompatibleBitmap(draw->hDC, w, h);
    HGDIOBJ oldBitmap = SelectObject(memory, bitmap);
    HFONT font = reinterpret_cast<HFONT>(SendMessageA(draw->hwndItem, WM_GETFONT, 0, 0));
    HGDIOBJ oldFont = font ? SelectObject(memory, font) : nullptr;
    DRAWITEMSTRUCT local = *draw;
    local.hDC = memory;
    OffsetRect(&local.rcItem, -rc.left, -rc.top);
    paint(&local);
    BitBlt(draw->hDC, rc.left, rc.top, w, h, memory, 0, 0, SRCCOPY);
    if (oldFont) SelectObject(memory, oldFont);
    SelectObject(memory, oldBitmap);
    DeleteObject(bitmap);
    DeleteDC(memory);
}

void drawStyledButton(const DRAWITEMSTRUCT* draw) {
    drawBuffered(draw, paintStyledButton);
}

void drawStyledComboItem(const DRAWITEMSTRUCT* draw) {
    drawBuffered(draw, paintStyledComboItem);
}

void paintStyledComboItem(const DRAWITEMSTRUCT* draw) {
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

// The capture's once-a-second status line ("... | flow AI 2.1 ms, diff ...")
// as the Live Capture status: which flow engine runs and its GPU time. The
// AI engine's fallback message is kept in front of it for that capture.
std::string* liveStatusFromLine(const std::string& line, std::string& fallback) {
    if (line.find("cannot select AI refinement:") != std::string::npos)
        return new std::string(line.substr(line.find("cannot select AI refinement:")));
    if (line.find("AI pyramid flow unavailable") != std::string::npos) {
        fallback = "AI model not loaded (export one on the AI Model page) - ";
        return new std::string(fallback + "using DIS.");
    }
    const size_t at = line.find("| flow ");
    if (at == std::string::npos) return nullptr;
    std::istringstream fields(line.substr(at + 7));
    std::string engine, ms;
    fields >> engine >> ms;
    const char* name = engine == "AI" ? "AI pyramid (your model)" : engine == "DIS" ? "DIS optical flow"
                     : engine == "FFX" ? "FFX" : engine == "neural" ? "neural flow" : "SAD";
    return new std::string(fallback + "Capture running - flow engine: " + name + ", " + ms +
                           " ms GPU per frame pair. Click Stop Capture to exit.");
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
    // Training reports its progress, and live capture which flow engine
    // actually runs (an AI model that fails to load falls back to DIS, which
    // only the capture's own output said) - both read from a pipe.
    const bool readOutput = jobType == JobType::Train || jobType == JobType::Live;
    if (readOutput) {
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
            readOutput,
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
            std::string fallback;   // this capture's AI fallback note
            while (ReadFile(logRead, buffer, sizeof(buffer), &length, nullptr) && length) {
                output.append(buffer, buffer + length);
                if (output.size() > 8192) output.erase(0, output.size() - 8192);
                pending.append(buffer, buffer + length);
                size_t end = 0;
                while ((end = pending.find('\n')) != std::string::npos) {
                    const std::string line = pending.substr(0, end);
                    if (jobType == JobType::Live) {
                        if (auto* status = liveStatusFromLine(line, fallback)) {
                            if (!PostMessageA(notifyWindow, WM_LIVE_STATUS, 0, reinterpret_cast<LPARAM>(status)))
                                delete status;
                        }
                    } else if (line.rfind("Epoch ", 0) == 0 ||
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

void updateAiRefinementControl();

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

    if (!validateInteger(g_state.gpuLevels, 1, FlowEngine::kMaxLevels, "Pyramid levels", g_state.liveStatus) ||
        !validateInteger(g_state.gpuMinRefine, 0, FlowEngine::kMaxLevels - 1, "Finest searched level", g_state.liveStatus) ||
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
    const char* flowEngineArg = flowSelection == 4 ? "ai" :
                                flowSelection == 3 ? "dis" :
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
    else if (SendMessageA(g_state.aiUiRegions, BM_GETCHECK, 0, 0) == BST_CHECKED) {
        const LRESULT selection = SendMessageA(g_state.uiDetector, CB_GETCURSEL, 0, 0);
        const auto choice = selection == 1 ? UiDetector::Choice::MatrixIcons : UiDetector::Choice::DirectMLIcons;
        const auto path = std::filesystem::path(UiDetector::defaultModelPath(choice));
        if (!std::filesystem::is_regular_file(path) ||
            (selection == 1 && !std::filesystem::is_regular_file(std::filesystem::path(path).replace_extension(L".bin")))) {
            CloseHandle(stopEvent);
            SetWindowTextA(g_state.liveStatus, "Selected UI detector missing: see tools/ui-detect/README.md. No fallback.");
            return;
        }
        captureArgs.push_back("--ai-ui-regions");
        captureArgs.push_back("--ui-detector");
        captureArgs.push_back(selection == 1 ? "matrix" : "icons");
    }
    // Engine-specific, ignored by the others: the occlusion fill is DIS's,
    // the refinement and its occlusion handling FFX's (occlusion handling
    // is part of the refinement, so it needs the refinement on).
    if (SendMessageA(g_state.occlusionFill, BM_GETCHECK, 0, 0) != BST_CHECKED) captureArgs.push_back("--no-occlusion-fill");
    if (SendMessageA(g_state.ffxRefine, BM_GETCHECK, 0, 0) != BST_CHECKED) captureArgs.push_back("--no-ffx-refine");
    if (SendMessageA(g_state.ffxOcclusion, BM_GETCHECK, 0, 0) == BST_CHECKED) captureArgs.push_back("--ffx-occlusion");
    if (flowSelection == 4 && SendMessageA(g_state.aiBlendMask, BM_GETCHECK, 0, 0) == BST_CHECKED)
        captureArgs.push_back("--ai-blend-mask");
    const bool temporalUi = flowSelection == 4 && SendMessageA(g_state.uiMask, BM_GETCHECK, 0, 0) == BST_CHECKED &&
                            SendMessageA(g_state.aiTemporalUi, BM_GETCHECK, 0, 0) == BST_CHECKED;
    if (temporalUi) {
        std::string reason;
        if (!AiFlowEstimator::temporalModelAvailable(AiFlowEstimator::defaultModelPath(true), reason)) {
            CloseHandle(stopEvent);
            SetWindowTextA(g_state.liveStatus, reason.c_str());
            updateAiRefinementControl();
            return;
        }
        captureArgs.push_back("--ai-temporal-ui");
    }
    if (flowSelection == 4 && SendMessageA(g_state.aiRefinement, BM_GETCHECK, 0, 0) == BST_CHECKED) {
        std::wstring selected;
        std::string reason;
        if (!AiFlowEstimator::selectRefinementModel(AiFlowEstimator::defaultModelPath(temporalUi), true, selected, reason)) {
            CloseHandle(stopEvent);
            SetWindowTextA(g_state.liveStatus, reason.c_str());
            updateAiRefinementControl();
            return;
        }
        captureArgs.push_back("--ai-refinement");
    }
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
    updateAiRefinementControl();
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
    if (!validateInteger(g_state.offlineLevels, 1, FlowEngine::kMaxLevels, "Pyramid levels", g_state.offlineStatus) ||
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
    const int page = g_state.activeTab;
    for (HWND control : g_state.liveControls) ShowWindow(control, page == 0 ? SW_SHOW : SW_HIDE);
    for (HWND control : g_state.offlineControls) ShowWindow(control, page == 1 ? SW_SHOW : SW_HIDE);
    for (HWND control : g_state.aiControls) ShowWindow(control, page == 2 ? SW_SHOW : SW_HIDE);
    InvalidateRect(g_state.hwnd, nullptr, FALSE);
}

// ---- AI Model page ----

std::wstring executableFolder() {
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring folder(path);
    const size_t slash = folder.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : folder.substr(0, slash);
}

// tools\ai-train (the trainer app and its Python), searched upwards from
// the executable: the build folders sit inside the repository.
std::wstring trainerFolder() {
    std::wstring folder = executableFolder();
    for (int up = 0; up < 6; ++up) {
        const std::wstring candidate = folder + L"\\tools\\ai-train";
        if (GetFileAttributesW((candidate + L"\\app.py").c_str()) != INVALID_FILE_ATTRIBUTES) return candidate;
        const size_t slash = folder.find_last_of(L"\\/");
        if (slash == std::wstring::npos) break;
        folder = folder.substr(0, slash);
    }
    return std::wstring();
}

// What models\ai_flow.onnx (and its .json from the trainer's export) says.
std::string aiModelSummary() {
    const bool temporalUi = g_state.aiTemporalUi && SendMessageA(g_state.aiTemporalUi, BM_GETCHECK, 0, 0) == BST_CHECKED;
    const std::wstring model = AiFlowEstimator::defaultModelPath(temporalUi);
    WIN32_FILE_ATTRIBUTE_DATA info = {};
    if (!GetFileAttributesExW(model.c_str(), GetFileExInfoStandard, &info))
        return "No AI model yet: open the trainer, build a dataset from your videos, train and export.";
    const size_t dot = model.find_last_of(L'.');
    std::ifstream file(model.substr(0, dot) + L".json");
    std::stringstream text;
    text << file.rdbuf();
    const std::string json = text.str();
    auto field = [&json](const char* name) {
        std::smatch m;
        const std::regex pattern(std::string("\"") + name + "\"\\s*:\\s*\"?([^\",}\\n]+)");
        return std::regex_search(json, m, pattern) ? m[1].str() : std::string("?");
    };
    SYSTEMTIME utc = {}, local = {};
    FileTimeToSystemTime(&info.ftLastWriteTime, &utc);
    SystemTimeToTzSpecificLocalTime(nullptr, &utc, &local);
    char line[400];
    // A model exported without a validation score has "psnr": null.
    std::string psnr = field("psnr");
    if (psnr.size() > 5) psnr = psnr.substr(0, 5);
    const std::string score = psnr == "null" || psnr == "?" ? std::string() : ", " + psnr + " dB on held-out clips";
    std::snprintf(line, sizeof(line), "Installed: models\\%s - %sx%s, %s model%s, exported %04d-%02d-%02d %02d:%02d",
                  temporalUi ? "ai_flow_ui.onnx (temporal UI)" : "ai_flow.onnx",
                  field("width").c_str(), field("height").c_str(), field("preset").c_str(), score.c_str(),
                  local.wYear, local.wMonth, local.wDay, local.wHour, local.wMinute);
    return line;
}

void updateAiRefinementControl() {
    if (!g_state.aiRefinement) return;
    EnableWindow(g_state.aiUiRegions, !g_state.captureStopEvent &&
        SendMessageA(g_state.uiMask, BM_GETCHECK, 0, 0) == BST_CHECKED);
    EnableWindow(g_state.uiDetector, !g_state.captureStopEvent &&
        SendMessageA(g_state.uiMask, BM_GETCHECK, 0, 0) == BST_CHECKED &&
        SendMessageA(g_state.aiUiRegions, BM_GETCHECK, 0, 0) == BST_CHECKED);
    const bool ai = SendMessageA(g_state.liveFlowEngine, CB_GETCURSEL, 0, 0) == 4;
    const bool ui = SendMessageA(g_state.uiMask, BM_GETCHECK, 0, 0) == BST_CHECKED;
    std::wstring selected;
    std::string reason;
    const bool temporalAvailable = AiFlowEstimator::temporalModelAvailable(AiFlowEstimator::defaultModelPath(true), reason);
    if (!temporalAvailable || !ui) SendMessageA(g_state.aiTemporalUi, BM_SETCHECK, BST_UNCHECKED, 0);
    EnableWindow(g_state.aiTemporalUi, ai && ui && temporalAvailable && !g_state.captureStopEvent);
    const bool temporal = ai && ui && SendMessageA(g_state.aiTemporalUi, BM_GETCHECK, 0, 0) == BST_CHECKED;
    const bool available = AiFlowEstimator::selectRefinementModel(AiFlowEstimator::defaultModelPath(temporal), true, selected, reason);
    if (!available) SendMessageA(g_state.aiRefinement, BM_SETCHECK, BST_UNCHECKED, 0);
    EnableWindow(g_state.aiRefinement, ai && available && !g_state.captureStopEvent);
    const char* help = g_state.captureStopEvent ? "Stop capture to change refinement."
                     : !ai ? "Select AI pyramid to choose its refiner."
                     : !available ? "AI refiner unavailable: export a refined model pair."
                     : "Checked: AI only. Unchecked: DIS. Choose before capture.";
    const std::string hint = temporal ? "Temporal UI: trained bundle, native pixel checks still required."
                           : !temporalAvailable && ai ? "Temporal UI: export best_ui.pt, then Refresh on AI Model page."
                           : std::string(help);
    SetWindowTextA(g_state.aiRefinementHelp, hint.c_str());
    InvalidateRect(g_state.aiRefinement, nullptr, FALSE);
    InvalidateRect(g_state.aiTemporalUi, nullptr, FALSE);
}

void refreshAiStatus() {
    if (g_state.aiStatus) SetWindowTextA(g_state.aiStatus, aiModelSummary().c_str());
    updateAiRefinementControl();
}

void openTrainer() {
    const std::wstring folder = trainerFolder();
    const std::wstring python = folder + L"\\.venv\\Scripts\\pythonw.exe";
    if (folder.empty() || GetFileAttributesW(python.c_str()) == INVALID_FILE_ATTRIBUTES) {
        const std::wstring setup = folder + L"\\Setup-AITrainer.ps1";
        if (!folder.empty() && GetFileAttributesW(setup.c_str()) != INVALID_FILE_ATTRIBUTES) {
            std::wstring command = L"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"" + setup + L"\" -Pause -Launch";
            STARTUPINFOW startup = {};
            startup.cb = sizeof(startup);
            PROCESS_INFORMATION process = {};
            if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NEW_CONSOLE, nullptr,
                               folder.c_str(), &startup, &process)) {
                CloseHandle(process.hThread);
                CloseHandle(process.hProcess);
                SetWindowTextA(g_state.aiStatus, "Trainer setup opened (Internet required, several GB). It opens the trainer when finished.");
                return;
            }
        }
        SetWindowTextA(g_state.aiStatus, folder.empty()
            ? "The trainer (tools\\ai-train) was not found above this program's folder."
            : "The trainer's Python environment is missing: see tools\\ai-train\\README.md to set it up.");
        return;
    }
    std::wstring command = L"\"" + python + L"\" \"" + folder + L"\\app.py\"";
    STARTUPINFOW startup = {};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process = {};
    if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, 0, nullptr, folder.c_str(), &startup, &process)) {
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        SetWindowTextA(g_state.aiStatus, "Trainer opened. Export there, then press Refresh.");
    } else {
        SetWindowTextA(g_state.aiStatus, "Could not start the trainer.");
    }
}

void switchPage(int page) {
    if (page == g_state.activeTab) return;
    g_state.activeTab = page;
    g_pageT = 0.0f;   // the title slides in
    applyPageVisibility();
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

    // The pages are placed right of the sidebar; its navigation is painted
    // (paintSidebar) and clicked through the window itself.
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

    HWND gpuLevelsLabel = makeLabel("Pyramid levels (1-10)", 56, 208, 160);
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
    SendMessageA(g_state.liveFlowEngine, CB_ADDSTRING, 0,
                 reinterpret_cast<LPARAM>("AI pyramid (trained model)"));
    SendMessageA(g_state.liveFlowEngine, CB_SETCURSEL, 3, 0);   // DIS
    // Static UI (HUD, subtitles, logos) is shown where it is instead of
    // being dragged along by the motion underneath it. See UiMaskCS.hlsl.
    g_state.uiMask = makeCheckbox("Keep static UI sharp (HUD, subtitles, logos)", 56, 318, 420, true);
    SetWindowLongPtrA(g_state.uiMask, GWLP_ID, IDC_UI_MASK);
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
    g_state.aiBlendMask = makeCheckbox("AI: blending mask", 238, 396, 184, false);
    g_state.aiRefinement = makeCheckbox("AI flow refinement (off: DIS)", 36, 434, 310, false);
    SetWindowLongPtrA(g_state.aiRefinement, GWLP_ID, IDC_AI_REFINEMENT);
    g_state.aiUiRegions = makeCheckbox("AI: UI regions (exp.)", 360, 434, 200, false);
    SetWindowLongPtrA(g_state.aiUiRegions, GWLP_ID, IDC_AI_UI_REGIONS);
    g_state.uiDetector = makeCombo(570, 430, 260, IDC_UI_DETECTOR);
    for (const char* option : {"DirectML: icons (default)", "Matrix: icons (WMMA)"})
        SendMessageA(g_state.uiDetector, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(option));
    SendMessageA(g_state.uiDetector, CB_SETCURSEL, 0, 0);
    g_state.aiTemporalUi = makeCheckbox("Trained temporal UI (exp.)", 36, 462, 280, false);
    SetWindowLongPtrA(g_state.aiTemporalUi, GWLP_ID, IDC_AI_TEMPORAL_UI);
    g_state.aiRefinementHelp = makeLabel("Select AI pyramid to choose its refiner.", 326, 466, 504);
    SetWindowSubclass(g_state.aiRefinementHelp, changingStaticProc, 3, 0);
    // How often the AI depth network runs (FlowSettings::aiDepthStride):
    // every frame, or every n-th with the frames between borrowing a
    // neighbour's depth (about 1.3-1.6 ms per run on an RX 9070 XT).
    HWND aiDepthRateLabel = makeLabel("AI DEPTH RATE", 430, 398, 175);
    g_state.aiDepthRate = makeCombo(610, 394, 180, IDC_AI_DEPTH_RATE);
    for (const char* rate : {"Every frame", "Every 2nd frame", "Every 3rd frame", "Every 4th frame"})
        SendMessageA(g_state.aiDepthRate, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(rate));
    SendMessageA(g_state.aiDepthRate, CB_SETCURSEL, 0, 0);
    g_state.liveStatus = addControl("STATIC", "", SS_LEFTNOWORDWRAP | SS_ENDELLIPSIS,
                                     36, 490, 794, 20, IDC_LIVE_STATUS);
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
        g_state.uiMask, g_state.occlusionFill, g_state.ffxRefine, g_state.ffxOcclusion, g_state.aiDepth, aiDepthRateLabel, g_state.aiDepthRate, g_state.aiBlendMask,
        g_state.aiRefinement, g_state.aiRefinementHelp, g_state.aiTemporalUi, g_state.aiUiRegions, g_state.uiDetector, g_state.captureButton, g_state.liveStatus,
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

    HWND levelsLabel = makeLabel("Pyramid levels (1-10)", 56, 312, 145);
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

    // ---- AI Model page ----
    HWND aiIntro = addControl("STATIC",
        "Train your own frame-interpolation model on footage like what you play, then use it as the flow engine. "
        "The model estimates motion; the program synthesizes intermediate frames. Select integrated AI refinement "
        "or DIS refinement on Live Capture. AI refinement requires a model trained with it and a paired export. Benchmark your model's speed.",
        SS_LEFT, 56, 124, 740, 60, 0);
    setControlFont(aiIntro, g_fontBody);
    g_state.aiStatus = addControl("STATIC", "", SS_LEFT, 56, 196, 740, 40, IDC_AI_STATUS);
    setControlFont(g_state.aiStatus, g_fontBody);
    SetWindowSubclass(g_state.aiStatus, changingStaticProc, 3, 0);
    HWND aiSteps = addControl("STATIC",
        "1.  Dataset - add your videos; the trainer cuts them into triplets of frames (repeats, cuts and stills left out).\n"
        "2.  Train - on the GPU with ROCm; it shows the quality on the part of each video it never trains on.\n"
        "3.  Export - refined models also write ai_flow.dis.onnx; keep both ONNX and JSON files together.\n"
        "4.  Choose \"AI pyramid (trained model)\" as the flow engine on the Live Capture page.",
        SS_LEFT, 56, 248, 740, 92, 0);
    setControlFont(aiSteps, g_fontLabel);
    HWND openTrainerButton = makeButton("OPEN AI TRAINER", 56, 360, 200, 40, IDC_OPEN_TRAINER);
    HWND useAiButton = makeButton("Use AI engine", 272, 365, 150, 30, IDC_USE_AI_ENGINE);
    HWND refreshAiButton = makeButton("Refresh", 434, 365, 110, 30, IDC_AI_REFRESH);
    g_state.aiControls = {aiIntro, g_state.aiStatus, aiSteps, openTrainerButton, useAiButton, refreshAiButton};
    refreshAiStatus();

    g_offsetX = 0;
    applyPageVisibility();
    refreshWindowList();
}

// ------------------------------------------------------------------
// Custom painting
// ------------------------------------------------------------------
// A section card: a small icon tile and the title.
enum class SectionIcon { Chip, Spark, Sliders };

void paintSectionPanel(HDC hdc, const RECT& rc, const char* title, SectionIcon icon) {
    paintCard(hdc, rc, ColPanel, 18);
    {
        Gdiplus::Graphics g(hdc);
        prepare(g);
        const float x = rc.left + 18.0f, y = rc.top + 8.0f, size = 22.0f;
        Gdiplus::GraphicsPath tile;
        roundPath(tile, x, y, size, size, 6.0f);
        Gdiplus::SolidBrush fill(gpColor(ColPanelHi));
        g.FillPath(&fill, &tile);
        Gdiplus::Pen edge(gpColor(ColBorderHi), 1.0f);
        g.DrawPath(&edge, &tile);
        Gdiplus::Pen pen(gpColor(ColText), 1.4f);
        const float cx = x + size / 2.0f, cy = y + size / 2.0f;
        if (icon == SectionIcon::Chip) {
            g.DrawRectangle(&pen, cx - 4.5f, cy - 4.5f, 9.0f, 9.0f);
            for (float d : {-2.5f, 2.5f}) {
                g.DrawLine(&pen, cx + d, cy - 7.5f, cx + d, cy - 5.0f);
                g.DrawLine(&pen, cx + d, cy + 5.0f, cx + d, cy + 7.5f);
                g.DrawLine(&pen, cx - 7.5f, cy + d, cx - 5.0f, cy + d);
                g.DrawLine(&pen, cx + 5.0f, cy + d, cx + 7.5f, cy + d);
            }
        } else if (icon == SectionIcon::Spark) {
            Gdiplus::PointF star[8];
            for (int k = 0; k < 8; ++k) {
                const float r = k % 2 ? 2.0f : 7.0f;
                const float a = k * 3.14159265f / 4.0f - 3.14159265f / 2.0f;
                star[k] = Gdiplus::PointF(cx + r * std::cos(a), cy + r * std::sin(a));
            }
            Gdiplus::SolidBrush ink(gpColor(ColText));
            g.FillPolygon(&ink, star, 8);
        } else {
            Gdiplus::SolidBrush knob(gpColor(ColText));
            for (int k = 0; k < 3; ++k) {
                const float ly = cy - 5.0f + 5.0f * k;
                g.DrawLine(&pen, cx - 7.0f, ly, cx + 7.0f, ly);
                const float kx = cx + (k == 1 ? 3.0f : -3.0f);
                g.FillEllipse(&knob, kx - 2.2f, ly - 2.2f, 4.4f, 4.4f);
            }
        }
    }
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, ColText);
    RECT titleRect = { rc.left + 50, rc.top + 8, rc.right - 16, rc.top + 30 };
    HFONT oldFont = static_cast<HFONT>(SelectObject(hdc, g_fontSection));
    DrawTextA(hdc, title, -1, &titleRect, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, oldFont);
}

// The original staged orbit/node/trail animation is shared with the installer.
struct EmblemState {
    float reveal = 1.0f;
    float time = -1.0f;    // seconds; < 0 = g_time
    bool assembling = false;
};

void paintEmblem(Gdiplus::Graphics& g, float cx, float cy, const EmblemState& e = {}) {
    const float t = e.time >= 0.0f ? e.time : static_cast<float>(g_time);
    g_brandLogo.draw(g, cx, cy, 104.0f, t, e.reveal, e.assembling);
}

// The navigation icons: a live signal, and two frames with the one made
// between them.
void paintNavIcon(Gdiplus::Graphics& g, int item, float cx, float cy, COLORREF ink) {
    Gdiplus::Pen pen(gpColor(ink), 1.7f);
    if (item == 0) {
        Gdiplus::SolidBrush dot(gpColor(ink));
        g.FillEllipse(&dot, cx - 3.0f, cy - 3.0f, 6.0f, 6.0f);
        for (float r : {7.0f, 11.5f}) {
            g.DrawArc(&pen, cx - r, cy - r, 2 * r, 2 * r, -50.0f, 100.0f);
            g.DrawArc(&pen, cx - r, cy - r, 2 * r, 2 * r, 130.0f, 100.0f);
        }
    } else if (item == 2) {
        // A pyramid of three levels, the network's coarse-to-fine.
        Gdiplus::PointF tri[3] = {Gdiplus::PointF(cx, cy - 9.0f), Gdiplus::PointF(cx + 10.0f, cy + 8.0f),
                                  Gdiplus::PointF(cx - 10.0f, cy + 8.0f)};
        g.DrawPolygon(&pen, tri, 3);
        g.DrawLine(&pen, cx - 4.2f, cy - 1.8f, cx + 4.2f, cy - 1.8f);
        g.DrawLine(&pen, cx - 7.0f, cy + 3.2f, cx + 7.0f, cy + 3.2f);
    } else {
        Gdiplus::GraphicsPath a, b;
        roundPath(a, cx - 10.0f, cy - 8.0f, 12.0f, 10.0f, 2.5f);
        roundPath(b, cx - 2.0f, cy - 2.0f, 12.0f, 10.0f, 2.5f);
        g.DrawPath(&pen, &a);
        g.DrawPath(&pen, &b);
    }
}

void paintSidebar(HDC hdc, const RECT& client) {
    RECT side = { 0, 0, kSidebarW, client.bottom };
    FillRect(hdc, &side, g_brSidebar);
    RECT edge = { kSidebarW - 1, 0, kSidebarW, client.bottom };
    HBRUSH line = CreateSolidBrush(RGB(30, 30, 34));
    FillRect(hdc, &edge, line);
    DeleteObject(line);
    const float t = static_cast<float>(g_time);

    Gdiplus::Graphics g(hdc);
    prepare(g);

    // The character's own orbit/particles fill the unused space below navigation.
    // Keep the icon, controls and status card clear. Short windows scale to fit.
    const float top = 384.0f, bottom = static_cast<float>(client.bottom) - 116.0f;
    const float characterTop = 370.0f;
    const float characterHeight = std::min(312.0f, static_cast<float>(client.bottom) - 98.0f - characterTop);
    if (g_brandCharacter.loaded() && characterHeight >= 100.0f) {
        const float width = characterHeight * CharacterAtlas::Width / CharacterAtlas::Height;
        g_brandCharacter.draw(g, Gdiplus::RectF((kSidebarW - width) / 2.0f, characterTop, width, characterHeight), t);
    } else if (bottom > top + 40.0f) {
        // Retain the original ambient treatment if the artwork is unavailable.
        for (int i = 0; i < 5; ++i) {
            const float o = 16.0f * i;
            const float drift = 14.0f * std::sin(t * 0.5f + i * 0.8f);
            Gdiplus::Pen wave(gpColor(RGB(255, 255, 255), 36 - 6 * i), 1.0f);
            g.DrawBezier(&wave, -10.0f, bottom - o, 70.0f + drift, bottom - 120.0f - o - drift,
                         140.0f - drift, top + 60.0f + o + drift, static_cast<float>(kSidebarW) + 10.0f, top + o);
        }
        const float stars[][2] = {{38, 380}, {176, 410}, {96, 470}, {200, 520}, {54, 560}, {150, 600}, {120, 360}};
        int k = 0;
        for (const auto& p : stars) {
            const float twinkle = 0.5f + 0.5f * std::sin(t * 1.3f + k++ * 1.9f);
            if (p[1] >= bottom) continue;
            Gdiplus::SolidBrush star(gpColor(RGB(255, 255, 255), 50 + static_cast<int>(150 * twinkle)));
            const float r = 0.8f + 0.7f * twinkle;
            g.FillEllipse(&star, p[0] - r, p[1] - r, 2 * r, 2 * r);
        }
    }

    paintEmblem(g, kSidebarW / 2.0f, 84.0f);

    // Navigation: a highlight card gliding to the active item (g_navY),
    // hover fades, and a small lit marker on the highlight's edge.
    if (g_navY < 0.0f) g_navY = static_cast<float>(kNavRects[g_state.activeTab].top);
    for (int i = 0; i < kNavCount; ++i)
        if (g_navHoverT[i] > 0.01f) {
            Gdiplus::GraphicsPath hover;
            const RECT& r = kNavRects[i];
            roundPath(hover, static_cast<float>(r.left), static_cast<float>(r.top), static_cast<float>(r.right - r.left),
                      static_cast<float>(r.bottom - r.top), 12.0f);
            Gdiplus::SolidBrush fill(gpColor(RGB(24, 24, 28), static_cast<int>(255 * g_navHoverT[i])));
            g.FillPath(&fill, &hover);
        }
    g.Flush(Gdiplus::FlushIntentionSync);
    const int highlightTop = static_cast<int>(std::lround(g_navY));
    RECT highlight = { 16, highlightTop, kSidebarW - 16, highlightTop + 46 };
    paintCard(hdc, highlight, ColPanelHi, 12);
    {
        const float my = highlightTop + 23.0f;
        for (int i = 3; i >= 1; --i) {
            Gdiplus::SolidBrush glow(gpColor(RGB(255, 255, 255), 14));
            g.FillEllipse(&glow, 16.0f - 2.0f * i, my - 10.0f - 2.0f * i, 6.0f + 4.0f * i, 20.0f + 4.0f * i);
        }
        Gdiplus::GraphicsPath marker;
        roundPath(marker, 17.0f, my - 9.0f, 3.0f, 18.0f, 1.5f);
        Gdiplus::SolidBrush white(gpColor(RGB(255, 255, 255)));
        g.FillPath(&white, &marker);
    }
    HFONT oldFont = static_cast<HFONT>(SelectObject(hdc, g_fontNav));
    SetBkMode(hdc, TRANSPARENT);
    for (int i = 0; i < kNavCount; ++i) {
        const RECT& r = kNavRects[i];
        // Lit by the highlight as it passes, and by the hover.
        const float lit = std::max(g_navHoverT[i], 1.0f - std::min(1.0f, std::fabs(g_navY - r.top) / 52.0f));
        const COLORREF ink = lerpColor(ColDim, ColText, lit);
        paintNavIcon(g, i, r.left + 30.0f, (r.top + r.bottom) / 2.0f, ink);
        g.Flush(Gdiplus::FlushIntentionSync);
        SetTextColor(hdc, ink);
        RECT label = { r.left + 52, r.top, r.right - 8, r.bottom };
        DrawTextA(hdc, kNavLabels[i], -1, &label, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    }

    SelectObject(hdc, g_fontBrand);
    SetTextColor(hdc, ColText);
    RECT brand = { 0, 144, kSidebarW, 164 };
    DrawTextA(hdc, "M O T I O N   E N H A N C E R", -1, &brand, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, g_fontLabel);
    SetTextColor(hdc, ColDim);
    RECT tagline = { 0, 166, kSidebarW, 184 };
    DrawTextA(hdc, "Real-time frame generation", -1, &tagline, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    // Status card: what the program is doing now; the dot breathes while
    // something runs.
    const bool capturing = g_state.captureStopEvent != nullptr;
    const bool busy = !capturing && g_state.trainButton && !IsWindowEnabled(g_state.trainButton);
    RECT card = { 16, client.bottom - 92, kSidebarW - 16, client.bottom - 16 };
    paintCard(hdc, card, ColPanel, 14);
    {
        const float dx = card.left + 26.0f, dy = (card.top + card.bottom) / 2.0f;
        if (capturing || busy) {
            const float pulse = 0.5f + 0.5f * std::sin(t * 3.0f);
            for (int i = 4; i >= 1; --i) {
                Gdiplus::SolidBrush glow(gpColor(RGB(255, 255, 255), 10 + static_cast<int>(18 * pulse)));
                const float r = 5.0f + (2.0f + 2.0f * pulse) * i;
                g.FillEllipse(&glow, dx - r, dy - r, 2 * r, 2 * r);
            }
        }
        Gdiplus::SolidBrush dot(gpColor(capturing || busy ? RGB(255, 255, 255) : RGB(90, 90, 98)));
        g.FillEllipse(&dot, dx - 5.0f, dy - 5.0f, 10.0f, 10.0f);
        g.Flush(Gdiplus::FlushIntentionSync);
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

// The page title and subtitle; after a page switch they slide up and fade
// in (g_pageT).
constexpr RECT kHeaderRect = { kSidebarW + 30, 0, kSidebarW + 840, 64 };

void paintPageHeader(HDC hdc) {
    const bool live = g_state.activeTab == 0;
    const float e = easeOut(g_pageT);
    const int rise = static_cast<int>(std::lround((1.0f - e) * 12.0f));
    SetBkMode(hdc, TRANSPARENT);
    HFONT oldFont = static_cast<HFONT>(SelectObject(hdc, g_fontTitle));
    SetTextColor(hdc, lerpColor(ColBg, ColText, e));
    RECT title = { kSidebarW + 36, 6 + rise, kSidebarW + 830, 44 + rise };
    DrawTextA(hdc, kNavLabels[g_state.activeTab], -1, &title, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, g_fontLabel);
    const float e2 = easeOut(g_pageT * 1.4f - 0.3f);
    SetTextColor(hdc, lerpColor(ColBg, ColDim, e2));
    const int rise2 = static_cast<int>(std::lround((1.0f - e2) * 8.0f));
    RECT subtitle = { kSidebarW + 36, 42 + rise2, kSidebarW + 830, 60 + rise2 };
    DrawTextA(hdc, live ? "Generate frames for any window in real time on the GPU."
                   : g_state.activeTab == 1 ? "Interpolate a frame between two images."
                                            : "Your own AI motion model, trained on your videos.",
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
    // Drawn off screen, then copied: no flicker. Only the parts the update
    // touches are drawn: an animation frame usually redraws the sidebar.
    HDC hdc = CreateCompatibleDC(screen);
    HBITMAP bitmap = CreateCompatibleBitmap(screen, client.right, client.bottom);
    HGDIOBJ oldBitmap = SelectObject(hdc, bitmap);
    const RECT& dirty = paint.rcPaint;

    if (dirty.left < kSidebarW) paintSidebar(hdc, client);
    if (dirty.right > kSidebarW) {
        RECT page = { kSidebarW, 0, client.right, client.bottom };
        FillRect(hdc, &page, g_brBg);
        paintPageHeader(hdc);
        if (dirty.bottom > kHeaderRect.bottom || dirty.right > kHeaderRect.right) {
            if (g_state.activeTab == 0) {
                paintSectionPanel(hdc, LivePanelRect, "Advanced GPU flow", SectionIcon::Chip);
                paintSectionPanel(hdc, TrainPanelRect, "Train neural flow", SectionIcon::Spark);
            } else if (g_state.activeTab == 1) {
                paintSectionPanel(hdc, OfflinePanelRect, "Advanced interpolation", SectionIcon::Sliders);
            } else {
                paintSectionPanel(hdc, AiPanelRect, "AI pyramid model", SectionIcon::Spark);
            }
            paintFieldFrames(hdc, hwnd);
        }
    }

    BitBlt(screen, dirty.left, dirty.top, dirty.right - dirty.left, dirty.bottom - dirty.top, hdc, dirty.left,
           dirty.top, SRCCOPY);
    SelectObject(hdc, oldBitmap);
    DeleteObject(bitmap);
    DeleteDC(hdc);
    EndPaint(hwnd, &paint);
}

// One animation step (kAnimTimer, about 60 per second).
void animate(HWND hwnd) {
    LARGE_INTEGER now, frequency;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&frequency);
    const float dt = g_lastTick.QuadPart
        ? std::min(0.1f, static_cast<float>(now.QuadPart - g_lastTick.QuadPart) / frequency.QuadPart) : 0.016f;
    g_lastTick = now;
    const bool front = GetForegroundWindow() == hwnd && !IsIconic(hwnd);
    bool sidebar = false;
    if (front) {
        g_time += dt;
        sidebar = true;
    }
    const auto approach = [dt](float& value, float target, float rate) {
        const float next = target + (value - target) * std::exp(-rate * dt);
        const bool moved = std::fabs(next - value) > 0.002f;
        value = std::fabs(next - target) < 0.004f ? target : next;
        return moved;
    };
    const float navTarget = static_cast<float>(kNavRects[g_state.activeTab].top);
    if (g_navY >= 0.0f && g_navY != navTarget) {
        const float before = g_navY;
        approach(g_navY, navTarget, 14.0f);
        if (std::fabs(g_navY - navTarget) < 0.3f) g_navY = navTarget;
        sidebar |= g_navY != before;
    }
    for (int i = 0; i < kNavCount; ++i) sidebar |= approach(g_navHoverT[i], g_navHover == i ? 1.0f : 0.0f, 12.0f);
    if (sidebar) {
        RECT side = { 0, 0, kSidebarW, 4000 };
        InvalidateRect(hwnd, &side, FALSE);
    }
    if (g_pageT < 1.0f) {
        g_pageT = std::min(1.0f, g_pageT + dt / 0.45f);
        InvalidateRect(hwnd, &kHeaderRect, FALSE);
    }
    for (auto& [control, fade] : g_hoverFades) {
        const bool hot = g_hover == control && IsWindowEnabled(control);
        const bool moved = approach(fade, hot ? 1.0f : 0.0f, 12.0f);
        // A hovered primary button keeps animating (its light sweep).
        const int id = GetDlgCtrlID(control);
        const bool sweep = hot && front && (id == IDC_START_CAPTURE || id == IDC_RUN_OFFLINE);
        if (moved || sweep) InvalidateRect(control, nullptr, FALSE);
    }
}

// ------------------------------------------------------------------
// Window procedure
// ------------------------------------------------------------------
LRESULT CALLBACK guiWindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_CREATE:
            g_state.hwnd = hwnd;
            createControls();
            SetTimer(hwnd, kAnimTimer, 16, nullptr);
            return 0;

        case WM_TIMER:
            if (wParam == kAnimTimer) {
                animate(hwnd);
                return 0;
            }
            break;

        // The sidebar's navigation is painted, so the window itself tracks
        // the pointer over it.
        case WM_MOUSEMOVE: {
            const POINT p = { static_cast<short>(LOWORD(lParam)), static_cast<short>(HIWORD(lParam)) };
            g_navHover = -1;
            for (int i = 0; i < kNavCount; ++i)
                if (PtInRect(&kNavRects[i], p)) g_navHover = i;
            if (!g_trackingLeave) {
                TRACKMOUSEEVENT track = {};
                track.cbSize = sizeof(track);
                track.dwFlags = TME_LEAVE;
                track.hwndTrack = hwnd;
                g_trackingLeave = TrackMouseEvent(&track) != FALSE;
            }
            return 0;
        }

        case WM_MOUSELEAVE:
            g_navHover = -1;
            g_trackingLeave = false;
            return 0;

        case WM_LBUTTONDOWN: {
            const POINT p = { static_cast<short>(LOWORD(lParam)), static_cast<short>(HIWORD(lParam)) };
            for (int i = 0; i < kNavCount; ++i)
                if (PtInRect(&kNavRects[i], p)) switchPage(i);
            return 0;
        }

        case WM_SETCURSOR:
            if (reinterpret_cast<HWND>(wParam) == hwnd && LOWORD(lParam) == HTCLIENT && g_navHover >= 0) {
                SetCursor(LoadCursor(nullptr, IDC_HAND));
                return TRUE;
            }
            break;

        // Ctrl+Tab switches the page from the keyboard.
        case WM_KEYDOWN:
            if (wParam == VK_TAB && (GetKeyState(VK_CONTROL) & 0x8000)) {
                switchPage((g_state.activeTab + 1) % kNavCount);
                return 0;
            }
            break;

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
            const bool onPanel = g_state.activeTab == 0 ? within(LivePanelRect) || within(TrainPanelRect)
                                 : g_state.activeTab == 1 ? within(OfflinePanelRect) : within(AiPanelRect);
            SetBkMode(hdc, OPAQUE);
            SetBkColor(hdc, onPanel ? ColPanel : ColBg);
            if (controlId == IDC_LIVE_STATUS || controlId == IDC_OFFLINE_STATUS || controlId == IDC_TRAIN_STATUS ||
                controlId == IDC_AI_STATUS) {
                SetTextColor(hdc, ColAccent);
            } else {
                SetTextColor(hdc, ColDim);
            }
            return reinterpret_cast<LRESULT>(onPanel ? g_brPanel : g_brBg);
        }

        case WM_CTLCOLORBTN: {
            HWND control = reinterpret_cast<HWND>(lParam);
            HDC hdc = reinterpret_cast<HDC>(wParam);
            const COLORREF surface = surfaceUnder(control);
            SetBkColor(hdc, surface);
            return reinterpret_cast<LRESULT>(surface == ColSidebar ? g_brSidebar
                                             : surface == ColPanel ? g_brPanel : g_brBg);
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
                if (box == g_state.uiMask || box == g_state.aiUiRegions || box == g_state.aiTemporalUi) refreshAiStatus();
                return 0;
            }
            switch (LOWORD(wParam)) {
                case IDC_LIVE_FLOW_ENGINE:
                    if (HIWORD(wParam) == CBN_SELCHANGE) updateAiRefinementControl();
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
                case IDC_OPEN_TRAINER: openTrainer(); return 0;
                case IDC_AI_REFRESH: refreshAiStatus(); return 0;
                case IDC_USE_AI_ENGINE:
                    SendMessageA(g_state.liveFlowEngine, CB_SETCURSEL, 4, 0);
                    updateAiRefinementControl();
                    switchPage(0);
                    return 0;
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
                updateAiRefinementControl();
                EnableWindow(g_state.trainButton, TRUE);
                SetWindowTextA(g_state.liveStatus, exitCode == 0 ? "Capture stopped." : "Capture ended with an error.");
            } else if (jobType == JobType::Offline) {
                EnableWindow(g_state.offlineButton, TRUE);
                EnableWindow(g_state.trainButton, TRUE);
                SetWindowTextA(g_state.offlineStatus, exitCode == 0 ? "Interpolation complete." : "Interpolation failed.");
            }
            return 0;
        }

        case WM_LIVE_STATUS: {
            std::string* status = reinterpret_cast<std::string*>(lParam);
            if (status) {
                if (g_state.captureStopEvent) SetWindowTextA(g_state.liveStatus, status->c_str());
                delete status;
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
            KillTimer(hwnd, kAnimTimer);
            g_state.hwnd = nullptr;
            PostQuitMessage(0);
            return 0;
        }
    }
    return DefWindowProcA(hwnd, message, wParam, lParam);
}

// ------------------------------------------------------------------
// Splash: the emblem builds itself before the window appears, then the
// window fades in under it. A click or a key skips to the end.
// ------------------------------------------------------------------
constexpr int kSplashW = 580, kSplashH = 360;
constexpr float kSplashLength = 2.6f;    // seconds, the last kSplashFade of them crossfading
constexpr float kSplashFade = 0.45f;
bool g_splashSkip = false;

float phase(float t, float start, float length) {
    return std::clamp((t - start) / length, 0.0f, 1.0f);
}

std::wstring widen(const char* text) {
    std::wstring wide;
    for (const char* p = text; *p; ++p) wide += static_cast<wchar_t>(*p);
    return wide;
}

// One frame of the splash at time t, on a transparent ARGB surface.
void paintSplash(Gdiplus::Graphics& target, float t) {
    target.Clear(Gdiplus::Color(0, 0, 0, 0));
    // Everything is drawn at full strength into a layer, then put on the
    // surface scaled and faded as a whole.
    Gdiplus::Bitmap layer(kSplashW, kSplashH, PixelFormat32bppPARGB);
    Gdiplus::Graphics g(&layer);
    prepare(g);
    g.SetTextRenderingHint(Gdiplus::TextRenderingHintAntiAliasGridFit);
    g.Clear(Gdiplus::Color(0, 0, 0, 0));

    // The card, with a soft shadow-glow around it.
    const float m = 24.0f;
    const float cw = kSplashW - 2 * m, ch = kSplashH - 2 * m;
    for (int i = 8; i >= 1; --i) {
        Gdiplus::GraphicsPath shade;
        roundPath(shade, m - 2.5f * i, m - 2.5f * i + 3.0f, cw + 5.0f * i, ch + 5.0f * i, 26.0f + 2.5f * i);
        Gdiplus::SolidBrush brush(Gdiplus::Color(static_cast<BYTE>(16), 0, 0, 0));
        g.FillPath(&brush, &shade);
    }
    Gdiplus::GraphicsPath card;
    roundPath(card, m, m, cw, ch, 26.0f);
    Gdiplus::LinearGradientBrush fill(Gdiplus::PointF(0.0f, m), Gdiplus::PointF(0.0f, m + ch),
                                      gpColor(RGB(20, 20, 24), 250), gpColor(RGB(10, 10, 12), 250));
    g.FillPath(&fill, &card);
    {
        Gdiplus::GraphicsPath edge;
        roundPath(edge, m + 0.5f, m + 0.5f, cw - 1.0f, ch - 1.0f, 26.0f);
        Gdiplus::LinearGradientBrush light(Gdiplus::PointF(0.0f, m - 1.0f), Gdiplus::PointF(0.0f, m + ch + 1.0f),
                                           gpColor(RGB(110, 110, 120)), gpColor(ColBorder));
        const Gdiplus::Color colors[3] = {gpColor(RGB(110, 110, 120)), gpColor(ColBorder), gpColor(ColBorder)};
        const Gdiplus::REAL positions[3] = {0.0f, 0.45f, 1.0f};
        light.SetInterpolationColors(colors, positions, 3);
        Gdiplus::Pen pen(&light, 1.0f);
        g.DrawPath(&pen, &edge);
    }

    // A separate character cameo, feathered into the right side of the card.
    // The orbital mark and all lettering retain their own uncluttered space.
    g_brandCharacter.draw(g, Gdiplus::RectF(356.0f, 44.0f, 184.0f, 266.0f), t,
                          easeOut(phase(t, 0.15f, 0.65f)));

    // Reveal the same refined orbit-and-motion emblem as the window icon.
    EmblemState e;
    e.time = t;
    e.assembling = true;
    const float brandCentre = 210.0f;
    const float ex = brandCentre, ey = 140.0f;
    g.TranslateTransform(ex, ey);
    g.ScaleTransform(1.35f, 1.35f);
    paintEmblem(g, 0.0f, 0.0f, e);
    g.ResetTransform();

    // The name, letter by letter, its spacing settling in.
    const wchar_t* family = fontExists("Segoe UI Variable Display") ? L"Segoe UI Variable Display" : L"Segoe UI";
    {
        Gdiplus::FontFamily face(family);
        Gdiplus::Font font(&face, 17.0f, Gdiplus::FontStyleBold, Gdiplus::UnitPixel);
        const std::wstring name = L"MOTION ENHANCER";
        const float track = 5.0f + 2.0f * (1.0f - easeOut(phase(t, 1.0f, 0.9f)));
        std::vector<float> widths;
        float total = 0.0f;
        for (wchar_t c : name) {
            Gdiplus::RectF box;
            const wchar_t one[2] = {c, 0};
            g.MeasureString(one, 1, &font, Gdiplus::PointF(0.0f, 0.0f), Gdiplus::StringFormat::GenericTypographic(), &box);
            const float w = c == L' ' ? 6.0f : box.Width;
            widths.push_back(w);
            total += w + track;
        }
        total -= track;
        float x = brandCentre - total / 2.0f;
        for (size_t i = 0; i < name.size(); ++i) {
            const float a = easeOut(phase(t, 1.05f + 0.035f * i, 0.35f));
            if (a > 0.0f && name[i] != L' ') {
                Gdiplus::SolidBrush ink(gpColor(RGB(255, 255, 255), static_cast<int>(255 * a)));
                const wchar_t one[2] = {name[i], 0};
                g.DrawString(one, 1, &font, Gdiplus::PointF(x, 232.0f + 6.0f * (1.0f - a)),
                             Gdiplus::StringFormat::GenericTypographic(), &ink);
            }
            x += widths[i] + track;
        }
    }
    {
        Gdiplus::FontFamily face(family);
        Gdiplus::Font font(&face, 13.0f, Gdiplus::FontStyleRegular, Gdiplus::UnitPixel);
        const float a = easeOut(phase(t, 1.55f, 0.45f));
        Gdiplus::SolidBrush ink(gpColor(ColDim, static_cast<int>(255 * a)));
        Gdiplus::StringFormat centred;
        centred.SetAlignment(Gdiplus::StringAlignmentCenter);
        g.DrawString(L"Real-time frame generation", -1, &font,
                     Gdiplus::RectF(30.0f, 262.0f + 4.0f * (1.0f - a), 360.0f, 20.0f), &centred, &ink);
    }
    // A thin progress line filling under it.
    {
        const float w = 140.0f, x = brandCentre - w / 2.0f, y = 296.0f;
        const float a = phase(t, 0.3f, 0.4f);
        Gdiplus::Pen track(gpColor(RGB(255, 255, 255), static_cast<int>(26 * a)), 2.0f);
        track.SetStartCap(Gdiplus::LineCapRound);
        track.SetEndCap(Gdiplus::LineCapRound);
        g.DrawLine(&track, x, y, x + w, y);
        const float p = easeOut(phase(t, 0.3f, kSplashLength - kSplashFade - 0.4f));
        if (p > 0.0f) {
            Gdiplus::LinearGradientBrush lit(Gdiplus::PointF(x - 1.0f, y), Gdiplus::PointF(x + w * p + 1.0f, y),
                                             gpColor(RGB(255, 255, 255), static_cast<int>(60 * a)),
                                             gpColor(RGB(255, 255, 255), static_cast<int>(240 * a)));
            Gdiplus::Pen bar(&lit, 2.0f);
            bar.SetStartCap(Gdiplus::LineCapRound);
            bar.SetEndCap(Gdiplus::LineCapRound);
            g.DrawLine(&bar, x, y, x + w * p, y);
        }
    }

    // The whole card: in with a slight zoom, out with a slight zoom.
    const float appear = easeOut(phase(t, 0.0f, 0.4f));
    const float vanish = phase(t, kSplashLength - kSplashFade, kSplashFade);
    const float alpha = appear * (1.0f - vanish * vanish);
    const float scale = (0.94f + 0.06f * appear) * (1.0f + 0.05f * vanish);
    Gdiplus::ColorMatrix fade = {{{1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, alpha, 0}, {0, 0, 0, 0, 1}}};
    Gdiplus::ImageAttributes attributes;
    attributes.SetColorMatrix(&fade);
    prepare(target);
    target.SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBicubic);
    const float w = kSplashW * scale, h = kSplashH * scale;
    target.DrawImage(&layer, Gdiplus::RectF((kSplashW - w) / 2.0f, (kSplashH - h) / 2.0f, w, h), 0.0f, 0.0f,
                     static_cast<float>(kSplashW), static_cast<float>(kSplashH), Gdiplus::UnitPixel, &attributes);
}

LRESULT CALLBACK splashProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_LBUTTONDOWN:
        case WM_RBUTTONDOWN:
        case WM_KEYDOWN:
            g_splashSkip = true;
            return 0;
        case WM_SETCURSOR:
            SetCursor(LoadCursor(nullptr, IDC_APPSTARTING));
            return TRUE;
    }
    return DefWindowProcA(hwnd, message, wParam, lParam);
}

// Plays the splash centred on `area`, and fades `window` (created hidden,
// WS_EX_LAYERED) in under its end. Returns once the window is shown.
void playSplash(HWND window, const RECT& area) {
    HINSTANCE instance = GetModuleHandleA(nullptr);
    WNDCLASSEXA splashClass = {};
    splashClass.cbSize = sizeof(splashClass);
    splashClass.lpfnWndProc = splashProc;
    splashClass.hInstance = instance;
    splashClass.hIcon = static_cast<HICON>(LoadImageW(instance, MAKEINTRESOURCEW(IDI_MOTION_ENHANCER),
        IMAGE_ICON, GetSystemMetrics(SM_CXICON), GetSystemMetrics(SM_CYICON), LR_SHARED));
    splashClass.hIconSm = static_cast<HICON>(LoadImageW(instance, MAKEINTRESOURCEW(IDI_MOTION_ENHANCER),
        IMAGE_ICON, GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), LR_SHARED));
    splashClass.lpszClassName = "MotionEnhancerSplashClass";
    RegisterClassExA(&splashClass);
    const int x = (area.left + area.right - kSplashW) / 2;
    const int y = (area.top + area.bottom - kSplashH) / 2;
    // Topmost while it plays, so the window can fade in beneath it.
    HWND splash = CreateWindowExA(WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_TOPMOST, splashClass.lpszClassName,
                                  "Motion Enhancer", WS_POPUP, x, y, kSplashW, kSplashH, nullptr, nullptr, instance,
                                  nullptr);
    if (!splash) {
        SetLayeredWindowAttributes(window, 0, 255, LWA_ALPHA);
        ShowWindow(window, SW_SHOW);
        return;
    }
    BITMAPINFO info = {};
    info.bmiHeader.biSize = sizeof(info.bmiHeader);
    info.bmiHeader.biWidth = kSplashW;
    info.bmiHeader.biHeight = -kSplashH;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    void* bits = nullptr;
    HDC screen = GetDC(nullptr);
    HDC memory = CreateCompatibleDC(screen);
    HBITMAP dib = CreateDIBSection(screen, &info, DIB_RGB_COLORS, &bits, nullptr, 0);
    HGDIOBJ old = SelectObject(memory, dib);
    Gdiplus::Bitmap surface(kSplashW, kSplashH, kSplashW * 4, PixelFormat32bppPARGB, static_cast<BYTE*>(bits));

    LARGE_INTEGER start, now, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    float skipShift = 0.0f;
    bool shown = false, first = true;
    for (;;) {
        QueryPerformanceCounter(&now);
        float t = static_cast<float>(now.QuadPart - start.QuadPart) / frequency.QuadPart + skipShift;
        // A skip jumps to the crossfade.
        if (g_splashSkip && t < kSplashLength - kSplashFade) {
            skipShift += kSplashLength - kSplashFade - t;
            t = kSplashLength - kSplashFade;
        }
        // The main window uses this same clock underneath the crossfade. Do
        // not restart the spark/nodes at zero when the splash goes away.
        g_time = std::min(t, kSplashLength);
        if (t >= kSplashLength) break;
        {
            Gdiplus::Graphics g(&surface);
            paintSplash(g, t);
        }
        POINT position = {x, y}, origin = {0, 0};
        SIZE size = {kSplashW, kSplashH};
        BLENDFUNCTION blend = {AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};
        UpdateLayeredWindow(splash, screen, &position, &size, memory, &origin, 0, &blend, ULW_ALPHA);
        if (first) {
            ShowWindow(splash, SW_SHOWNOACTIVATE);
            SetForegroundWindow(splash);
            first = false;
        }
        // The window fades in under the splash's end.
        if (t >= kSplashLength - kSplashFade) {
            const BYTE alpha = static_cast<BYTE>(255.0f * easeOut(phase(t, kSplashLength - kSplashFade, kSplashFade)));
            SetLayeredWindowAttributes(window, 0, alpha, LWA_ALPHA);
            if (!shown) {
                ShowWindow(window, SW_SHOWNOACTIVATE);
                UpdateWindow(window);
                shown = true;
            }
        }
        MSG pending;
        while (PeekMessageA(&pending, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&pending);
            DispatchMessageA(&pending);
        }
        DwmFlush();   // one frame per screen refresh
    }
    SelectObject(memory, old);
    DeleteObject(dib);
    DeleteDC(memory);
    ReleaseDC(nullptr, screen);
    DestroyWindow(splash);
    UnregisterClassA(splashClass.lpszClassName, instance);

    if (!shown) ShowWindow(window, SW_SHOW);
    // Fully opaque, and not layered any more (layering costs a copy of every frame).
    SetLayeredWindowAttributes(window, 0, 255, LWA_ALPHA);
    SetWindowLongA(window, GWL_EXSTYLE, GetWindowLongA(window, GWL_EXSTYLE) & ~WS_EX_LAYERED);
    QueryPerformanceCounter(&g_lastTick);
    RedrawWindow(window, nullptr, nullptr, RDW_ERASE | RDW_INVALIDATE | RDW_FRAME | RDW_ALLCHILDREN);
    SetForegroundWindow(window);
}

// Offline check: MOTION_ENHANCER_SPLASH_FRAMES=<prefix> writes the splash's
// frames (30 fps, over the README background) as <prefix>_NN.png.
bool writeSplashFrames(const char* prefix) {
    constexpr int fps = 30;
    Gdiplus::Bitmap surface(kSplashW, kSplashH, PixelFormat32bppPARGB);
    const int count = static_cast<int>(std::lround(kSplashLength * fps));
    for (int index = 0; index < count; ++index) {
        {
            Gdiplus::Graphics g(&surface);
            paintSplash(g, static_cast<float>(index) / fps);
        }
        Image image(kSplashW, kSplashH, 3);
        for (int y = 0; y < kSplashH; ++y)
            for (int x = 0; x < kSplashW; ++x) {
                Gdiplus::Color c;
                surface.GetPixel(x, y, &c);
                // Over GitHub's dark page colour (the README shows these).
                const float a = c.GetA() / 255.0f;
                image.set(x, y, 0, c.GetR() * a + 13.0f * (1.0f - a));
                image.set(x, y, 1, c.GetG() * a + 17.0f * (1.0f - a));
                image.set(x, y, 2, c.GetB() * a + 23.0f * (1.0f - a));
            }
        char name[MAX_PATH];
        std::snprintf(name, sizeof(name), "%s_%02d.png", prefix, index);
        if (!image.savePNG(name)) return false;
    }
    return true;
}

// A complete native 30-fps logo loop, plus the intro/ambient hand-off. This
// renders directly to memory: no window capture or hundreds of app launches.
bool writeLogoFrames(const char* prefix) {
    constexpr int size = 256, fps = 30;
    Gdiplus::Bitmap surface(size, size, PixelFormat32bppARGB);
    Image image(size, size, 3);
    Gdiplus::Bitmap sidebar(kSidebarW, 330, PixelFormat32bppARGB);
    Image sidebarImage(kSidebarW, 330, 3);
    auto save = [](Gdiplus::Bitmap& bitmap, Image& frame, const char* name) {
        Gdiplus::BitmapData data = {};
        Gdiplus::Rect area(0, 0, frame.width, frame.height);
        if (bitmap.LockBits(&area, Gdiplus::ImageLockModeRead, PixelFormat32bppARGB, &data) != Gdiplus::Ok) return false;
        for (int y = 0; y < frame.height; ++y) {
            const auto* row = static_cast<const BYTE*>(data.Scan0) + static_cast<std::ptrdiff_t>(y) * data.Stride;
            for (int x = 0; x < frame.width; ++x)
                for (int c = 0; c < 3; ++c) frame.set(x, y, c, row[x * 4 + 2 - c]);
        }
        bitmap.UnlockBits(&data);
        return frame.savePNG(name);
    };
    auto write = [&](const char* name, float time, bool assembling = false) {
        {
            Gdiplus::Graphics graphics(&surface);
            graphics.Clear(gpColor(ColSidebar));
            g_brandLogo.draw(graphics, size / 2.0f, size / 2.0f, 184, time, 1, assembling);
        }
        return save(surface, image, name);
    };
    char name[MAX_PATH];
    const int count = static_cast<int>(BrandLogo::AnimationPeriod * fps);
    for (int i = 0; i <= count; ++i) {
        if (std::snprintf(name, sizeof(name), "%s_%04d.png", prefix, i) >= static_cast<int>(sizeof(name))) return false;
        if (!write(name, static_cast<float>(i) / fps)) return false;
        // Use the actual sidebar painter/font metrics, not a composited mock-up.
        g_time = static_cast<double>(i) / fps;
        {
            Gdiplus::Graphics graphics(&sidebar);
            HDC dc = graphics.GetHDC();
            const RECT client = {0, 0, 1116, 781};
            paintSidebar(dc, client);
            graphics.ReleaseHDC(dc);
        }
        if (std::snprintf(name, sizeof(name), "%s_sidebar_%04d.png", prefix, i) >= static_cast<int>(sizeof(name))) return false;
        if (!save(sidebar, sidebarImage, name)) return false;
    }
    if (std::snprintf(name, sizeof(name), "%s_intro.png", prefix) >= static_cast<int>(sizeof(name)) || !write(name, kSplashLength, true)) return false;
    if (std::snprintf(name, sizeof(name), "%s_handoff.png", prefix) >= static_cast<int>(sizeof(name)) || !write(name, kSplashLength)) return false;
    return true;
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
    g_brandLogo.load(GetModuleHandleW(nullptr));
    g_brandCharacter.load(GetModuleHandleW(nullptr));
    ensureTheme();

    HINSTANCE instance = GetModuleHandleA(nullptr);
    WNDCLASSEXA windowClass = {};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = guiWindowProc;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursor(nullptr, IDC_ARROW);
    windowClass.hIcon = static_cast<HICON>(LoadImageW(instance, MAKEINTRESOURCEW(IDI_MOTION_ENHANCER),
        IMAGE_ICON, GetSystemMetrics(SM_CXICON), GetSystemMetrics(SM_CYICON), LR_SHARED));
    windowClass.hIconSm = static_cast<HICON>(LoadImageW(instance, MAKEINTRESOURCEW(IDI_MOTION_ENHANCER),
        IMAGE_ICON, GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), LR_SHARED));
    windowClass.hbrBackground = nullptr;
    windowClass.lpszClassName = "MotionEnhancerGuiClass";
    RegisterClassExA(&windowClass);

    char logoFrames[MAX_PATH] = {};
    if (GetEnvironmentVariableA("MOTION_ENHANCER_LOGO_FRAMES", logoFrames, sizeof(logoFrames))) {
        const bool written = writeLogoFrames(logoFrames);
        g_brandCharacter.reset();
        g_brandLogo.reset();
        Gdiplus::GdiplusShutdown(g_gdiplusToken);
        return written ? 0 : 1;
    }

    char splashFrames[MAX_PATH] = {};
    if (GetEnvironmentVariableA("MOTION_ENHANCER_SPLASH_FRAMES", splashFrames, sizeof(splashFrames))) {
        const bool written = writeSplashFrames(splashFrames);
        g_brandCharacter.reset();
        g_brandLogo.reset();
        Gdiplus::GdiplusShutdown(g_gdiplusToken);
        return written ? 0 : 1;
    }

    // Centred on the primary monitor's work area, where the splash plays.
    RECT area = {};
    SystemParametersInfoA(SPI_GETWORKAREA, 0, &area, 0);
    // Offline check: MOTION_ENHANCER_SPLASH_TEST plays the splash and the
    // fade-in far off screen, reports the window's state and exits.
    const bool splashTest = GetEnvironmentVariableA("MOTION_ENHANCER_SPLASH_TEST", nullptr, 0) > 0;
    if (splashTest) OffsetRect(&area, -30000, -30000);
    const int windowW = 900 + kSidebarW, windowH = 820;
    // Created hidden and layered: it fades in under the splash's end.
    HWND window = CreateWindowExA(
        WS_EX_LAYERED,
        windowClass.lpszClassName,
        "Motion Enhancer",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_CLIPCHILDREN,
        std::max<int>(area.left, (area.left + area.right - windowW) / 2),
        std::max<int>(area.top, (area.top + area.bottom - windowH) / 2), windowW, windowH,
        nullptr, nullptr, instance, nullptr
    );
    if (!window) {
        g_brandCharacter.reset();
        g_brandLogo.reset();
        Gdiplus::GdiplusShutdown(g_gdiplusToken);
        return 1;
    }
    SetLayeredWindowAttributes(window, 0, 0, LWA_ALPHA);

    // Dark immersive title bar where supported.
    BOOL dark = TRUE;
    DwmSetWindowAttribute(window, 20, &dark, sizeof(dark));
    DwmSetWindowAttribute(window, 19, &dark, sizeof(dark));
    // Windows 11: rounded corners, a dark border, caption and title text.
    const DWORD round = 2;   // DWMWCP_ROUND
    DwmSetWindowAttribute(window, 33, &round, sizeof(round));
    const COLORREF borderColor = RGB(34, 34, 38), captionColor = ColSidebar, captionText = ColText;
    DwmSetWindowAttribute(window, 34, &borderColor, sizeof(borderColor));
    DwmSetWindowAttribute(window, 35, &captionColor, sizeof(captionColor));
    DwmSetWindowAttribute(window, 36, &captionText, sizeof(captionText));

    // Test hook for offline UI checks: MOTION_ENHANCER_GUI_SNAPSHOT=<png>
    // renders this window off screen into a PNG and exits (page 1 when
    // MOTION_ENHANCER_GUI_PAGE=1). Nothing on the screen is read.
    char snapshot[MAX_PATH] = {};
    if (GetEnvironmentVariableA("MOTION_ENHANCER_GUI_SNAPSHOT", snapshot, sizeof(snapshot))) {
        char page[4] = {};
        if (GetEnvironmentVariableA("MOTION_ENHANCER_GUI_PAGE", page, sizeof(page)) && page[0] >= '1' && page[0] <= '2') {
            g_state.activeTab = page[0] - '0';
            g_navY = static_cast<float>(kNavRects[g_state.activeTab].top);
            applyPageVisibility();
        }
        // MOTION_ENHANCER_GUI_TIME: the ambient animation's time in seconds,
        // to render its frames one by one.
        char seconds[32] = {};
        if (GetEnvironmentVariableA("MOTION_ENHANCER_GUI_TIME", seconds, sizeof(seconds))) g_time = atof(seconds);
        SetLayeredWindowAttributes(window, 0, 255, LWA_ALPHA);
        SetWindowLongA(window, GWL_EXSTYLE, GetWindowLongA(window, GWL_EXSTYLE) & ~WS_EX_LAYERED);
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
        g_brandCharacter.reset();
        g_brandLogo.reset();
        Gdiplus::GdiplusShutdown(g_gdiplusToken);
        return saved ? 0 : 1;
    }

    // The splash, unless MOTION_ENHANCER_NO_SPLASH is set.
    if (GetEnvironmentVariableA("MOTION_ENHANCER_NO_SPLASH", nullptr, 0) > 0) {
        SetLayeredWindowAttributes(window, 0, 255, LWA_ALPHA);
        SetWindowLongA(window, GWL_EXSTYLE, GetWindowLongA(window, GWL_EXSTYLE) & ~WS_EX_LAYERED);
        ShowWindow(window, SW_SHOW);
        UpdateWindow(window);
    } else {
        LARGE_INTEGER start, end, frequency;
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&start);
        playSplash(window, area);
        QueryPerformanceCounter(&end);
        if (splashTest) {
            const bool layered = (GetWindowLongA(window, GWL_EXSTYLE) & WS_EX_LAYERED) != 0;
            char report[160];
            std::snprintf(report, sizeof(report), "splash %.2f s, window visible %d, layered %d\n",
                          static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart,
                          IsWindowVisible(window) ? 1 : 0, layered ? 1 : 0);
            HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
            DWORD written = 0;
            if (out && out != INVALID_HANDLE_VALUE) WriteFile(out, report, static_cast<DWORD>(strlen(report)), &written, nullptr);
            DestroyWindow(window);
        }
    }

    MSG message;
    while (GetMessageA(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageA(&message);
    }
    g_brandCharacter.reset();
    g_brandLogo.reset();
    Gdiplus::GdiplusShutdown(g_gdiplusToken);
    return static_cast<int>(message.wParam);
}
