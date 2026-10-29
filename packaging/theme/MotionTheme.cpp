// Installer-only skin. Native NSIS controls retain their keyboard, screen-reader
// and installation behavior; this plug-in supplies the app's monochrome skin.
#include <windows.h>
#include <commctrl.h>
#include <objidl.h>
#include <gdiplus.h>
#include <uxtheme.h>
#include <dwmapi.h>
#include <mmsystem.h>
#include <richedit.h>
#include <nsis/pluginapi.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {
constexpr COLORREF Bg = RGB(8, 8, 10), Sidebar = RGB(12, 12, 14);
constexpr COLORREF Panel = RGB(17, 17, 20), Hi = RGB(28, 28, 32);
constexpr COLORREF Field = RGB(13, 13, 15), Border = RGB(42, 42, 48);
constexpr COLORREF Text = RGB(238, 238, 242), Dim = RGB(138, 138, 148);
constexpr UINT MusicId = 1200;
constexpr UINT_PTR SkinId = 0x4d45, AnimationId = 0x4d46;
constexpr float Pi = 3.14159265358979323846f;
HMODULE g_module = nullptr;
HWND g_window = nullptr, g_musicButton = nullptr, g_page = nullptr;
HFONT g_body = nullptr, g_title = nullptr;
HBRUSH g_background = nullptr, g_panel = nullptr, g_field = nullptr, g_border = nullptr;
ULONG_PTR g_gdi = 0;
float g_scale = 1.0f;
int g_originalWidth = 450, g_originalHeight = 230, g_step = 0;
bool g_uninstall = false, g_playing = false, g_audioFailed = false;
std::wstring g_music;
ULONGLONG g_start = 0;
std::vector<HWND> g_controls;

struct Control {
    bool hover = false;
    std::wstring type;
};

int px(float value) { return static_cast<int>(std::lround(value * g_scale)); }
Gdiplus::Color color(COLORREF c, BYTE alpha = 255) {
    return Gdiplus::Color(alpha, GetRValue(c), GetGValue(c), GetBValue(c));
}
void rounded(Gdiplus::Graphics& g, Gdiplus::RectF r, float radius, COLORREF fill, COLORREF edge, bool filled = true) {
    Gdiplus::GraphicsPath p;
    const float d = std::min(radius * 2, std::min(r.Width, r.Height));
    p.AddArc(r.X, r.Y, d, d, 180, 90);
    p.AddArc(r.GetRight() - d, r.Y, d, d, 270, 90);
    p.AddArc(r.GetRight() - d, r.GetBottom() - d, d, d, 0, 90);
    p.AddArc(r.X, r.GetBottom() - d, d, d, 90, 90);
    p.CloseFigure();
    Gdiplus::SolidBrush brush(color(fill));
    Gdiplus::Pen pen(color(edge));
    if (filled) g.FillPath(&brush, &p);
    g.DrawPath(&pen, &p);
}
void label(Gdiplus::Graphics& g, const wchar_t* text, Gdiplus::RectF r,
           float size, COLORREF c, bool bold = false, bool centered = false) {
    Gdiplus::Font font(L"Segoe UI", size, bold ? Gdiplus::FontStyleBold : Gdiplus::FontStyleRegular,
                        Gdiplus::UnitPixel);
    Gdiplus::SolidBrush brush(color(c));
    Gdiplus::StringFormat format;
    format.SetLineAlignment(Gdiplus::StringAlignmentCenter);
    if (centered) format.SetAlignment(Gdiplus::StringAlignmentCenter);
    g.DrawString(text, -1, &font, r, &format, &brush);
}

void emblem(Gdiplus::Graphics& g, float t) {
    const float x = 114, y = 91;
    for (int i = 10; i > 0; --i) {
        Gdiplus::SolidBrush glow(Gdiplus::Color(static_cast<BYTE>(2 + (10 - i) / 2), 235, 235, 242));
        const float r = 43.0f + i * 3.4f;
        g.FillEllipse(&glow, x - r, y - r, r * 2, r * 2);
    }
    Gdiplus::Pen orbit(Gdiplus::Color(160, 220, 220, 228), 1.2f);
    g.DrawEllipse(&orbit, x - 34, y - 34, 68.0f, 68.0f);
    orbit.SetColor(Gdiplus::Color(85, 220, 220, 228));
    g.DrawEllipse(&orbit, x - 43, y - 43, 86.0f, 86.0f);
    for (int k = 0; k < 6; ++k) {
        const float a = (k * 60.0f + t * 6.0f) * Pi / 180;
        const float dx = std::cos(a), dy = std::sin(a);
        g.DrawLine(&orbit, x + 43 * dx, y + 43 * dy, x + 55 * dx, y + 55 * dy);
        Gdiplus::SolidBrush dot(Gdiplus::Color(205, 238, 238, 244));
        g.FillEllipse(&dot, x + 55 * dx - 2, y + 55 * dy - 2, 4.0f, 4.0f);
    }
    for (int k = 14; k > 0; --k) {
        Gdiplus::Pen trail(Gdiplus::Color(static_cast<BYTE>(230 * (15 - k) / 15), 240, 240, 246), 2.4f);
        trail.SetStartCap(Gdiplus::LineCapRound);
        trail.SetEndCap(Gdiplus::LineCapRound);
        g.DrawArc(&trail, x - 34, y - 34, 68.0f, 68.0f, t * 60 - k * 4, 4.4f);
    }
    Gdiplus::Pen streak(Gdiplus::Color(230, 235, 235, 241), 1.5f);
    for (int k = -1; k <= 1; ++k)
        g.DrawLine(&streak, x - 22.0f + std::abs(k) * 5, y + k * 8.0f, x + 5, y + k * 8.0f);
    Gdiplus::SolidBrush core(Gdiplus::Color(246, 244, 244, 248));
    g.FillEllipse(&core, x + 1, y - 7, 14.0f, 14.0f);
    g.DrawEllipse(&orbit, x - 4, y - 12, 24.0f, 24.0f);
}

void paintWindow(HWND window, HDC supplied = nullptr) {
    PAINTSTRUCT ps = {};
    HDC dc = supplied ? supplied : BeginPaint(window, &ps);
    RECT r;
    GetClientRect(window, &r);
    Gdiplus::Bitmap bitmap(r.right, r.bottom);
    Gdiplus::Graphics g(&bitmap);
    g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
    g.SetTextRenderingHint(Gdiplus::TextRenderingHintClearTypeGridFit);
    g.Clear(color(Bg));
    g.ScaleTransform(g_scale, g_scale);
    Gdiplus::SolidBrush side(color(Sidebar));
    g.FillRectangle(&side, 0, 0, 228, 600);
    Gdiplus::Pen line(color(Border));
    g.DrawLine(&line, 227.0f, 0.0f, 227.0f, 600.0f);
    const float time = static_cast<float>(GetTickCount64() - g_start) / 1000;
    // The same subtle flowing lines used by the application's sidebar.
    for (int k = 0; k < 7; ++k) {
        Gdiplus::Pen curve(Gdiplus::Color(22, 175, 175, 190), 0.7f);
        const float offset = std::sin(time * 0.15f + k) * 8;
        g.DrawBezier(&curve, -20.0f, 575.0f, 48.0f, 445.0f + k * 8,
                     157.0f, 470.0f + offset, 228.0f, 390.0f + k * 12);
    }
    emblem(g, time);
    label(g, L"M O T I O N  E N H A N C E R", {8, 162, 212, 24}, 12, Text, true, true);
    label(g, L"Real-time frame generation", {8, 188, 212, 22}, 12, Dim, false, true);
    label(g, g_uninstall ? L"UNINSTALL" : L"SETUP  /  WINDOWS X64", {16, 226, 196, 18}, 10, Dim);
    const wchar_t* steps[] = {L"Welcome", L"License", L"Components", L"Location", L"Install", L"Ready"};
    const wchar_t* unsteps[] = {L"Remove application", L"Removing", L"Complete"};
    const int count = g_uninstall ? 3 : 6;
    for (int i = 0; i < count; ++i) {
        const float y = 255.0f + i * 37;
        if (i == g_step) {
            rounded(g, {14, y, 200, 32}, 10, Hi, Border);
            Gdiplus::SolidBrush accent(color(Text));
            g.FillRectangle(&accent, 15.0f, y + 9, 2.0f, 14.0f);
        }
        wchar_t n[5];
        wsprintfW(n, L"%02d", i + 1);
        label(g, n, {27, y, 25, 32}, 11, i == g_step ? Text : Dim);
        label(g, g_uninstall ? unsteps[i] : steps[i], {61, y, 150, 32}, 13,
              i == g_step ? Text : Dim, i == g_step);
    }
    rounded(g, {244, 114, 632, 400}, 18, Panel, Border);
    Gdiplus::Graphics target(dc);
    target.DrawImage(&bitmap, 0, 0);
    if (!supplied) EndPaint(window, &ps);
}

void updateMusic(bool enable) {
    if (enable && !g_music.empty()) {
        g_playing = PlaySoundW(g_music.c_str(), nullptr, SND_FILENAME | SND_ASYNC | SND_LOOP | SND_NODEFAULT) != FALSE;
        g_audioFailed = !g_playing;
    } else {
        PlaySoundW(nullptr, nullptr, 0);
        g_playing = false;
    }
    // Read-only state also makes audio behavior verifiable by UI automation.
    if (g_window) SetPropW(g_window, L"MotionEnhancer.MusicPlaying", reinterpret_cast<HANDLE>(g_playing ? 1 : 2));
    if (g_musicButton) {
        SetWindowTextW(g_musicButton, g_audioFailed ? L"Music unavailable" : g_playing ? L"&Music: on" : L"&Music: off");
        EnableWindow(g_musicButton, !g_audioFailed);
        InvalidateRect(g_musicButton, nullptr, FALSE);
    }
}

LRESULT colorControl(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
    HDC dc = reinterpret_cast<HDC>(wParam);
    HWND control = reinterpret_cast<HWND>(lParam);
    const bool isRoot = window == g_window;
    const bool edit = message == WM_CTLCOLOREDIT;
    SetTextColor(dc, GetDlgCtrlID(control) == 1038 ? Dim : Text);
    SetBkColor(dc, edit ? Field : isRoot ? Bg : Panel);
    return reinterpret_cast<LRESULT>(edit ? g_field : isRoot ? g_background : g_panel);
}

LRESULT CALLBACK controlProc(HWND, UINT, WPARAM, LPARAM, UINT_PTR, DWORD_PTR);
void skinControl(HWND control);

LRESULT treeColors(LPARAM lParam) {
    auto* draw = reinterpret_cast<NMTVCUSTOMDRAW*>(lParam);
    if (draw->nmcd.dwDrawStage == CDDS_PREPAINT) return CDRF_NOTIFYITEMDRAW;
    if (draw->nmcd.dwDrawStage == CDDS_ITEMPREPAINT) {
        draw->clrText = Text;
        draw->clrTextBk = (draw->nmcd.uItemState & CDIS_SELECTED) ? Hi : Field;
        draw->nmcd.uItemState &= ~(CDIS_SELECTED | CDIS_HOT);
        return CDRF_NEWFONT;
    }
    return CDRF_DODEFAULT;
}

LRESULT CALLBACK rootProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam,
                          UINT_PTR, DWORD_PTR) {
    switch (message) {
    case WM_ERASEBKGND: return 1;
    case WM_PAINT: paintWindow(window); return 0;
    case WM_PRINTCLIENT: paintWindow(window, reinterpret_cast<HDC>(wParam)); return 0;
    case WM_CTLCOLORSTATIC: case WM_CTLCOLORBTN: case WM_CTLCOLOREDIT: case WM_CTLCOLORDLG:
        return colorControl(window, message, wParam, lParam);
    case WM_COMMAND:
        if (LOWORD(wParam) == MusicId && HIWORD(wParam) == BN_CLICKED) {
            updateMusic(!g_playing);
            return 0;
        }
        break;
    case WM_TIMER:
        if (wParam == AnimationId) {
            // No repaint churn while the setup window is minimized/inactive.
            if (!IsIconic(window) && GetForegroundWindow() == window) {
                RECT sidebar = {0, 0, px(228), px(535)};
                InvalidateRect(window, &sidebar, FALSE);
            }
            return 0;
        }
        break;
    case WM_DESTROY:
        KillTimer(window, AnimationId);
        updateMusic(false);
        break;
    }
    return DefSubclassProc(window, message, wParam, lParam);
}

void paintButton(HWND control, Control& state, HDC dc, const RECT& client) {
    const float w = client.right / g_scale, h = client.bottom / g_scale;
    const LONG style = GetWindowLongW(control, GWL_STYLE) & BS_TYPEMASK;
    if (style == BS_GROUPBOX) {
        // Group boxes overlap sibling edit fields. Never fill their interior:
        // doing so hides the path/browse controls during painting/printing.
        Gdiplus::Graphics g(dc);
        g.ScaleTransform(g_scale, g_scale);
        g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
        rounded(g, {0.5f, 8, w - 1, h - 9}, 10, Panel, Border, false);
        wchar_t title[256] = {};
        GetWindowTextW(control, title, 256);
        Gdiplus::Font font(L"Segoe UI", 12, Gdiplus::FontStyleRegular, Gdiplus::UnitPixel);
        Gdiplus::RectF measured;
        g.MeasureString(title, -1, &font, Gdiplus::PointF(0, 0), &measured);
        Gdiplus::SolidBrush background(color(Panel));
        g.FillRectangle(&background, 10.0f, 0.0f, measured.Width + 8, 18.0f);
        label(g, title, {14, 0, w - 28, 18}, 12, Dim);
        return;
    }
    Gdiplus::Bitmap bitmap(client.right, client.bottom);
    Gdiplus::Graphics g(&bitmap);
    g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
    g.SetTextRenderingHint(Gdiplus::TextRenderingHintClearTypeGridFit);
    const bool footer = GetParent(control) == g_window;
    g.Clear(color(footer ? (control == g_musicButton ? Sidebar : Bg) : Panel));
    g.ScaleTransform(g_scale, g_scale);
    wchar_t caption[1024] = {};
    GetWindowTextW(control, caption, 1024);
    const bool enabled = IsWindowEnabled(control) != FALSE;
    const bool focused = GetFocus() == control;
    const bool pressed = (SendMessageW(control, BM_GETSTATE, 0, 0) & BST_PUSHED) != 0;
    const bool check = style == BS_AUTOCHECKBOX || style == BS_CHECKBOX || style == BS_AUTO3STATE;
    const bool radio = style == BS_AUTORADIOBUTTON || style == BS_RADIOBUTTON;
    // Preserve native button text, mnemonics and check state for accessibility.
    std::wstring text;
    for (size_t i = 0; caption[i]; ++i) {
        if (caption[i] == L'&' && caption[i + 1] != L'&') continue;
        text += caption[i];
        if (caption[i] == L'&' && caption[i + 1] == L'&') ++i;
    }
    if (check || radio) {
        const bool checked = SendMessageW(control, BM_GETCHECK, 0, 0) != BST_UNCHECKED;
        const float y = (h - 16) / 2;
        rounded(g, {1, y, 16, 16}, radio ? 8.0f : 4.0f, checked ? Text : Field,
                state.hover || focused ? Dim : Border);
        if (checked) {
            Gdiplus::Pen tick(color(Bg), 1.8f);
            g.DrawLine(&tick, 5.0f, y + 8, 8.0f, y + 11);
            g.DrawLine(&tick, 8.0f, y + 11, 13.0f, y + 5);
        }
        label(g, text.c_str(), {26, 0, w - 28, h}, 13, enabled ? Text : Dim);
    } else {
        const bool primary = footer && GetDlgCtrlID(control) == IDOK && enabled;
        rounded(g, {0.5f, 0.5f, w - 1, h - 1}, std::min(18.0f, h / 2),
                primary ? (pressed ? RGB(178, 178, 186) : state.hover ? RGB(252, 252, 255) : Text)
                        : pressed ? Field : state.hover ? Hi : Panel,
                primary ? Text : focused ? Dim : Border);
        label(g, text.c_str(), {5, 0, w - 10, h}, 13, primary ? Bg : enabled ? Text : Dim, true, true);
    }
    if (focused) {
        // A visible focus outline is retained for keyboard-only operation.
        float focusWidth = w;
        if (check || radio) {
            Gdiplus::Font font(L"Segoe UI", 13, Gdiplus::FontStyleRegular, Gdiplus::UnitPixel);
            Gdiplus::RectF measured;
            g.MeasureString(text.c_str(), -1, &font, Gdiplus::PointF(0, 0), &measured);
            focusWidth = std::min(w, measured.Width + 34);
        }
        rounded(g, {3, 3, focusWidth - 6, h - 6}, check || radio ? 4.0f : std::min(16.0f, h / 2 - 4),
                Bg, Dim, false);
    }
    Gdiplus::Graphics target(dc);
    target.DrawImage(&bitmap, 0, 0);
}

void buttonRegion(HWND control) {
    const auto style = GetWindowLongW(control, GWL_STYLE) & BS_TYPEMASK;
    if (style != BS_PUSHBUTTON && style != BS_DEFPUSHBUTTON) return;
    RECT r;
    GetClientRect(control, &r);
    SetWindowRgn(control, CreateRoundRectRgn(0, 0, r.right + 1, r.bottom + 1, px(36), px(36)), TRUE);
}

LRESULT CALLBACK controlProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam,
                             UINT_PTR, DWORD_PTR reference) {
    auto* state = reinterpret_cast<Control*>(reference);
    if (message == WM_NCDESTROY) {
        RemoveWindowSubclass(window, controlProc, SkinId);
        g_controls.erase(std::remove(g_controls.begin(), g_controls.end(), window), g_controls.end());
        delete state;
        return DefSubclassProc(window, message, wParam, lParam);
    }
    if (message >= WM_CTLCOLORMSGBOX && message <= WM_CTLCOLORSTATIC)
        return colorControl(window, message, wParam, lParam);
    if (message == WM_NOTIFY) {
        auto* notification = reinterpret_cast<NMHDR*>(lParam);
        wchar_t type[80] = {};
        GetClassNameW(notification->hwndFrom, type, 80);
        if (notification->code == NM_CUSTOMDRAW && std::wstring(type) == WC_TREEVIEWW)
            return treeColors(lParam);
    }
    const bool button = state->type == L"Button";
    const bool progress = state->type == PROGRESS_CLASSW;
    const bool field = state->type == L"Edit" || state->type == WC_TREEVIEWW || state->type == WC_LISTVIEWW ||
        state->type.find(L"RichEdit") != std::wstring::npos || state->type.find(L"RICHEDIT") != std::wstring::npos;
    if (button && message == WM_SIZE) buttonRegion(window);
    if (button && message == WM_NCPAINT) return 0;
    if (field && message == WM_NCPAINT) {
        const LRESULT result = DefSubclassProc(window, message, wParam, lParam);
        RECT r;
        GetWindowRect(window, &r);
        OffsetRect(&r, -r.left, -r.top);
        HDC dc = GetWindowDC(window);
        FrameRect(dc, &r, g_border);
        ReleaseDC(window, dc);
        return result;
    }
    if (message == WM_ERASEBKGND && (button || progress)) return 1;
    if ((message == WM_PAINT || message == WM_PRINTCLIENT || message == WM_PRINT) && (button || progress)) {
        PAINTSTRUCT ps = {};
        HDC dc = message == WM_PAINT ? BeginPaint(window, &ps) : reinterpret_cast<HDC>(wParam);
        RECT r;
        GetClientRect(window, &r);
        if (button) paintButton(window, *state, dc, r);
        else {
            PBRANGE range = {};
            SendMessageW(window, PBM_GETRANGE, FALSE, reinterpret_cast<LPARAM>(&range));
            const auto position = SendMessageW(window, PBM_GETPOS, 0, 0);
            const float amount = range.iHigh > range.iLow
                ? std::clamp(static_cast<float>(position - range.iLow) / (range.iHigh - range.iLow), 0.0f, 1.0f) : 0;
            Gdiplus::Graphics g(dc);
            g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
            g.Clear(color(Panel));
            rounded(g, {0.5f, 0.5f, static_cast<float>(r.right - 1), static_cast<float>(r.bottom - 1)}, 5 * g_scale, Hi, Border);
            if (amount > 0) rounded(g, {1, 1, std::max(2.0f, (r.right - 2) * amount), static_cast<float>(r.bottom - 2)}, 5 * g_scale, Text, Text);
        }
        if (message == WM_PAINT) EndPaint(window, &ps);
        return 0;
    }
    if (button && message == WM_MOUSEMOVE && !state->hover) {
        state->hover = true;
        TRACKMOUSEEVENT track = {sizeof(track), TME_LEAVE, window, 0};
        TrackMouseEvent(&track);
        InvalidateRect(window, nullptr, FALSE);
    }
    if (button && message == WM_MOUSELEAVE) {
        state->hover = false;
        InvalidateRect(window, nullptr, FALSE);
    }
    const LRESULT result = DefSubclassProc(window, message, wParam, lParam);
    if (message == WM_SETTEXT && g_uninstall && window == GetDlgItem(g_window, 1037)) {
        wchar_t caption[100] = {};
        GetWindowTextW(window, caption, 100);
        if (std::wstring(caption) == L"Application removed.") {
            g_step = 2;
            InvalidateRect(g_window, nullptr, FALSE);
        }
    }
    if ((button || progress) && (message == WM_SETTEXT || message == WM_SETFOCUS || message == WM_KILLFOCUS ||
        message == WM_ENABLE || message == BM_SETSTYLE || message == BM_SETCHECK || message == PBM_SETPOS || message == PBM_SETRANGE32 ||
        message == WM_LBUTTONDOWN || message == WM_LBUTTONUP || message == WM_KEYDOWN || message == WM_KEYUP))
        InvalidateRect(window, nullptr, FALSE);
    return result;
}

void skinControl(HWND control) {
    if (!IsWindow(control)) return;
    DWORD_PTR reference;
    if (GetWindowSubclass(control, controlProc, SkinId, &reference)) return;
    auto* state = new Control;
    wchar_t type[80] = {};
    GetClassNameW(control, type, 80);
    state->type = type;
    SetWindowSubclass(control, controlProc, SkinId, reinterpret_cast<DWORD_PTR>(state));
    g_controls.push_back(control);
    SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(g_body), FALSE);
    SetWindowTheme(control, L"", L"");
    const auto buttonType = GetWindowLongW(control, GWL_STYLE) & BS_TYPEMASK;
    if (state->type == L"Button" && (buttonType == BS_PUSHBUTTON || buttonType == BS_DEFPUSHBUTTON)) {
        buttonRegion(control);
    } else if (state->type == WC_TREEVIEWW) {
        TreeView_SetBkColor(control, Field);
        TreeView_SetTextColor(control, Text);
        TreeView_SetLineColor(control, Border);
        TreeView_SetItemHeight(control, px(36));
    } else if (state->type == WC_LISTVIEWW) {
        ListView_SetBkColor(control, Field);
        ListView_SetTextBkColor(control, Field);
        ListView_SetTextColor(control, Text);
    } else if (state->type.find(L"RichEdit") != std::wstring::npos || state->type.find(L"RICHEDIT") != std::wstring::npos) {
        SendMessageW(control, EM_SETBKGNDCOLOR, 0, Field);
        CHARFORMAT2W format = {};
        format.cbSize = sizeof(format);
        format.dwMask = CFM_COLOR;
        format.crTextColor = Text;
        SendMessageW(control, EM_SETCHARFORMAT, SCF_ALL, reinterpret_cast<LPARAM>(&format));
    }
    const bool field = state->type == L"Edit" || state->type == WC_TREEVIEWW || state->type == WC_LISTVIEWW ||
        state->type.find(L"RichEdit") != std::wstring::npos || state->type.find(L"RICHEDIT") != std::wstring::npos;
    if (field) {
        SetWindowTheme(control, L"DarkMode_Explorer", nullptr);
        SetWindowLongW(control, GWL_EXSTYLE, GetWindowLongW(control, GWL_EXSTYLE) & ~WS_EX_CLIENTEDGE);
        SetWindowLongW(control, GWL_STYLE, GetWindowLongW(control, GWL_STYLE) | WS_BORDER);
        SetWindowPos(control, nullptr, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
    }
}

void move(HWND window, float x, float y, float width, float height) {
    if (window) SetWindowPos(window, nullptr, px(x), px(y), px(width), px(height), SWP_NOZORDER | SWP_NOACTIVATE);
}
std::wstring pop(stack_t** top) {
    if (!top || !*top) return {};
    stack_t* entry = *top;
    std::wstring text(entry->text);
    *top = entry->next;
    GlobalFree(entry);
    return text;
}

void cleanUp() {
    updateMusic(false);
    if (IsWindow(g_window)) {
        KillTimer(g_window, AnimationId);
        RemoveWindowSubclass(g_window, rootProc, SkinId);
        RemovePropW(g_window, L"MotionEnhancer.MusicPlaying");
    }
    for (HWND control : g_controls) {
        DWORD_PTR reference = 0;
        if (IsWindow(control) && GetWindowSubclass(control, controlProc, SkinId, &reference)) {
            RemoveWindowSubclass(control, controlProc, SkinId);
            delete reinterpret_cast<Control*>(reference);
        }
    }
    g_controls.clear();
    for (HFONT font : {g_body, g_title}) if (font) DeleteObject(font);
    for (HBRUSH brush : {g_background, g_panel, g_field, g_border}) if (brush) DeleteObject(brush);
    if (g_gdi) Gdiplus::GdiplusShutdown(g_gdi);
    g_gdi = 0;
    g_window = g_musicButton = g_page = nullptr;
}
UINT_PTR callback(NSPIM event) {
    if ((event == NSPIM_GUIUNLOAD || event == NSPIM_UNLOAD) && g_window) cleanUp();
    return 0;
}
} // namespace

extern "C" __declspec(dllexport) void Attach(HWND parent, int, wchar_t*, stack_t** stack, extra_parameters* extra) {
    const std::wstring music = pop(stack);
    const bool enabled = pop(stack) == L"1";
    g_uninstall = pop(stack) == L"1";
    if (!parent || g_window) return;
    extra->RegisterPluginCallback(g_module, callback);
    g_window = parent;
    g_music = music;
    g_start = GetTickCount64();
    HDC screen = GetDC(parent);
    g_scale = GetDeviceCaps(screen, LOGPIXELSX) / 96.0f;
    ReleaseDC(parent, screen);
    MONITORINFO monitor = {sizeof(monitor)};
    GetMonitorInfoW(MonitorFromWindow(parent, MONITOR_DEFAULTTONEAREST), &monitor);
    g_scale = std::min({g_scale, (monitor.rcWork.right - monitor.rcWork.left - 40) / 900.0f,
                       (monitor.rcWork.bottom - monitor.rcWork.top - 80) / 600.0f});
    Gdiplus::GdiplusStartupInput startup;
    Gdiplus::GdiplusStartup(&g_gdi, &startup, nullptr);
    auto font = [](int height, int weight) {
        return CreateFontW(-px(static_cast<float>(height)), 0, 0, 0, weight, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
            DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
    };
    g_body = font(14, FW_NORMAL);
    g_title = font(28, FW_SEMIBOLD);
    g_background = CreateSolidBrush(Bg);
    g_panel = CreateSolidBrush(Panel);
    g_field = CreateSolidBrush(Field);
    g_border = CreateSolidBrush(Border);
    RECT old = {};
    if (GetDlgItem(parent, 1018)) {
        GetClientRect(GetDlgItem(parent, 1018), &old);
        g_originalWidth = old.right;
        g_originalHeight = old.bottom;
    }
    RECT rect = {0, 0, px(900), px(600)};
    const auto style = GetWindowLongW(parent, GWL_STYLE);
    AdjustWindowRectEx(&rect, style, FALSE, GetWindowLongW(parent, GWL_EXSTYLE));
    const int width = rect.right - rect.left, height = rect.bottom - rect.top;
    SetWindowPos(parent, nullptr, monitor.rcWork.left + (monitor.rcWork.right - monitor.rcWork.left - width) / 2,
        monitor.rcWork.top + (monitor.rcWork.bottom - monitor.rcWork.top - height) / 2,
        width, height, SWP_NOZORDER | SWP_NOACTIVATE);
    SetWindowLongW(parent, GWL_STYLE, style | WS_CLIPCHILDREN);
    SetWindowSubclass(parent, rootProc, SkinId, 0);
    skinControl(GetDlgItem(parent, 1037));
    skinControl(GetDlgItem(parent, 1038));
    const BOOL dark = TRUE;
    DwmSetWindowAttribute(parent, 20, &dark, sizeof(dark));
    const DWORD corner = 2;
    DwmSetWindowAttribute(parent, 33, &corner, sizeof(corner));
    DwmSetWindowAttribute(parent, 35, &Sidebar, sizeof(Sidebar));
    DwmSetWindowAttribute(parent, 36, &Text, sizeof(Text));
    for (int id : {1018, 1044}) move(GetDlgItem(parent, id), 260, 130, 600, 362);
    for (int id : {1034, 1035, 1036, 1039, 1045, 1028, 1256, 1046}) ShowWindow(GetDlgItem(parent, id), SW_HIDE);
    move(GetDlgItem(parent, IDOK), 732, 540, 144, 38);
    move(GetDlgItem(parent, 3), 592, 540, 124, 38);
    move(GetDlgItem(parent, IDCANCEL), 452, 540, 124, 38);
    for (int id : {IDOK, IDCANCEL, 3}) skinControl(GetDlgItem(parent, id));
    if (!music.empty()) {
        g_musicButton = CreateWindowExW(0, L"Button", L"&Music: on", WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_PUSHBUTTON,
            px(16), px(540), px(196), px(38), parent, reinterpret_cast<HMENU>(MusicId), g_module, nullptr);
        skinControl(g_musicButton);
        updateMusic(enabled);
    }
    SetTimer(parent, AnimationId, 33, nullptr);
    InvalidateRect(parent, nullptr, TRUE);
}

extern "C" __declspec(dllexport) void Page(HWND, int, wchar_t*, stack_t** stack, extra_parameters*) {
    HWND page = reinterpret_cast<HWND>(static_cast<UINT_PTR>(std::wcstoull(pop(stack).c_str(), nullptr, 10)));
    const int step = std::wcstol(pop(stack).c_str(), nullptr, 10);
    const std::wstring title = pop(stack), subtitle = pop(stack);
    if (!g_window || !IsWindow(page)) return;
    g_page = page;
    g_step = step;
    for (int id : {1034, 1035, 1036, 1039, 1045, 1028, 1256, 1046}) ShowWindow(GetDlgItem(g_window, id), SW_HIDE);
    HWND heading = GetDlgItem(g_window, 1037), subheading = GetDlgItem(g_window, 1038);
    SetWindowTextW(heading, title.c_str());
    SetWindowTextW(subheading, subtitle.c_str());
    SendMessageW(heading, WM_SETFONT, reinterpret_cast<WPARAM>(g_title), FALSE);
    SendMessageW(subheading, WM_SETFONT, reinterpret_cast<WPARAM>(g_body), FALSE);
    move(heading, 260, 32, 616, 40);
    move(subheading, 260, 79, 616, 24);
    ShowWindow(heading, SW_SHOW);
    ShowWindow(subheading, SW_SHOW);
    move(page, 260, 130, 600, 362);
    skinControl(page);
    if (!GetPropW(page, L"MotionEnhancer.LayoutDone")) {
        const float sx = px(600.0f) / static_cast<float>(std::max(1, g_originalWidth));
        const float sy = px(362.0f) / static_cast<float>(std::max(1, g_originalHeight));
        for (HWND child = GetWindow(page, GW_CHILD); child; child = GetWindow(child, GW_HWNDNEXT)) {
            RECT rect;
            GetWindowRect(child, &rect);
            MapWindowPoints(nullptr, page, reinterpret_cast<POINT*>(&rect), 2);
            SetWindowPos(child, nullptr, static_cast<int>(rect.left * sx), static_cast<int>(rect.top * sy),
                static_cast<int>((rect.right - rect.left) * sx), static_cast<int>((rect.bottom - rect.top) * sy),
                SWP_NOZORDER | SWP_NOACTIVATE);
        }
        SetPropW(page, L"MotionEnhancer.LayoutDone", reinterpret_cast<HANDLE>(1));
    }
    EnumChildWindows(page, [](HWND child, LPARAM) -> BOOL { skinControl(child); return TRUE; }, 0);
    auto place = [page](int id, float x, float y, float width, float height) {
        move(GetDlgItem(page, id), x, y, width, height);
    };
    if (!g_uninstall && step == 1) {
        place(1040, 20, 14, 560, 24);
        place(1000, 20, 52, 560, 232);
        place(1006, 20, 304, 560, 52);
    } else if (!g_uninstall && step == 2) {
        place(1006, 20, 14, 560, 58);
        place(1022, 20, 98, 560, 24);
        place(1032, 20, 132, 560, 144);
        place(1023, 20, 315, 560, 24);
    } else if (!g_uninstall && step == 3) {
        place(1006, 20, 14, 560, 70);
        place(1020, 20, 130, 560, 126);
        place(1019, 36, 180, 400, 32);
        place(1001, 452, 180, 110, 32);
        place(1023, 20, 290, 560, 24);
        place(1024, 20, 320, 560, 24);
    } else if ((!g_uninstall && step == 4) || (g_uninstall && step >= 1)) {
        place(1006, 20, 16, 560, 26);
        place(1004, 20, 56, 560, 12);
        place(1016, 20, 92, 560, 246);
        place(1027, 20, 92, 136, 32);
    } else if (g_uninstall && step == 0) {
        place(1006, 20, 14, 560, 70);
        place(1029, 20, 130, 560, 24);
        place(1000, 20, 166, 560, 32);
    }
    if (!g_uninstall && (step == 0 || step == 5)) {
        // Welcome/finish use MUI's full-window nsDialogs template. Replace its
        // stock wizard bitmap and narrow text column with our shared sidebar
        // and a spacious card, keeping the original accessible labels/buttons.
        int labels = 0, checks = 0;
        for (HWND child = GetWindow(page, GW_CHILD); child; child = GetWindow(child, GW_HWNDNEXT)) {
            wchar_t type[80] = {};
            GetClassNameW(child, type, 80);
            const auto style = GetWindowLongW(child, GWL_STYLE);
            if (std::wstring(type) == L"Static") {
                if ((style & SS_TYPEMASK) == SS_BITMAP) ShowWindow(child, SW_HIDE);
                else if (labels++ == 0) ShowWindow(child, SW_HIDE);
                else move(child, 24, 16, 552, step == 0 ? 330.0f : 228.0f);
            } else if (std::wstring(type) == L"Button") {
                move(child, 24, 264.0f + checks++ * 38, 552, 32);
            }
        }
    }
    InvalidateRect(page, nullptr, TRUE);
    InvalidateRect(g_window, nullptr, TRUE);
}

extern "C" __declspec(dllexport) void Stop(HWND, int, wchar_t*, stack_t**, extra_parameters*) {
    updateMusic(false);
}

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_module = module;
    }
    return TRUE;
}
