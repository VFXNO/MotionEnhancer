#include "GuiApp.h"
#include "WindowHelper.h"

#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <dwmapi.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <thread>
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
constexpr UINT WM_PROCESS_COMPLETE = WM_APP + 1;

enum class JobType : LPARAM {
    Live = 1,
    Offline = 2
};

// ------------------------------------------------------------------
// Gaming theme
// ------------------------------------------------------------------
constexpr COLORREF ColBg        = RGB(9, 12, 17);
constexpr COLORREF ColBgAlt     = RGB(12, 16, 22);
constexpr COLORREF ColPanel     = RGB(15, 20, 28);
constexpr COLORREF ColPanelHi   = RGB(22, 29, 40);
constexpr COLORREF ColField     = RGB(11, 15, 21);
constexpr COLORREF ColBorder    = RGB(38, 50, 66);
constexpr COLORREF ColAccent    = RGB(0, 229, 255);
constexpr COLORREF ColAccentDk  = RGB(0, 158, 178);
constexpr COLORREF ColAccent2   = RGB(255, 45, 120);
constexpr COLORREF ColText      = RGB(232, 238, 244);
constexpr COLORREF ColDim       = RGB(132, 148, 164);
constexpr COLORREF ColTextOn    = RGB(3, 14, 18);

constexpr RECT LivePanelRect = { 36, 172, 36 + 794, 172 + 176 };
constexpr RECT OfflinePanelRect = { 36, 272, 36 + 794, 272 + 330 };

struct GuiState {
    HWND hwnd = nullptr;
    int activeTab = 0;
    HWND tabLive = nullptr;
    HWND tabOffline = nullptr;
    HWND windowList = nullptr;
    HWND refreshButton = nullptr;
    HWND captureButton = nullptr;
    HWND liveStatus = nullptr;
    HWND liveSourceFps = nullptr;
    HWND liveMultiplier = nullptr;
    HWND gpuLevels = nullptr;
    HWND gpuMinRefine = nullptr;
    HWND gpuCoarseRadius = nullptr;
    HWND gpuRefineRadius = nullptr;
    HWND gpuSmoothness = nullptr;
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
    HWND offlineOcclusion = nullptr;
    HWND offlineDetailStrength = nullptr;
    HWND offlinePhotoThreshold = nullptr;
    HWND offlinePhotoSigma = nullptr;
    HWND adaptiveBlock = nullptr;
    HWND spatialPredictors = nullptr;
    HWND gaussianWeights = nullptr;
    HWND bidirectional = nullptr;
    HWND photoGate = nullptr;
    HWND holeInpaint = nullptr;
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
HBRUSH g_brPanel = nullptr;
HBRUSH g_brPanelHi = nullptr;
HBRUSH g_brBgAlt = nullptr;

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
    g_fontSection = createFont("Segoe UI", -15, FW_BOLD);
    g_brField = CreateSolidBrush(ColField);
    g_brPanel = CreateSolidBrush(ColPanel);
    g_brPanelHi = CreateSolidBrush(ColPanelHi);
    g_brBgAlt = CreateSolidBrush(ColBgAlt);
}

COLORREF lerpColor(COLORREF a, COLORREF b, double t) {
    t = std::clamp(t, 0.0, 1.0);
    int r = static_cast<int>(GetRValue(a) + (GetRValue(b) - GetRValue(a)) * t);
    int g = static_cast<int>(GetGValue(a) + (GetGValue(b) - GetGValue(a)) * t);
    int bl = static_cast<int>(GetBValue(a) + (GetBValue(b) - GetBValue(a)) * t);
    return RGB(r, g, bl);
}

void fillRoundRect(HDC hdc, const RECT& rc, COLORREF color, int radius) {
    HBRUSH brush = CreateSolidBrush(color);
    HBRUSH oldBrush = static_cast<HBRUSH>(SelectObject(hdc, brush));
    HPEN pen = CreatePen(PS_NULL, 0, 0);
    HPEN oldPen = static_cast<HPEN>(SelectObject(hdc, pen));
    RoundRect(hdc, rc.left, rc.top, rc.right, rc.bottom, radius * 2, radius * 2);
    SelectObject(hdc, oldBrush);
    SelectObject(hdc, oldPen);
    DeleteObject(brush);
    DeleteObject(pen);
}

void frameRoundRect(HDC hdc, const RECT& rc, COLORREF color, int radius) {
    HPEN pen = CreatePen(PS_SOLID, 1, color);
    HPEN oldPen = static_cast<HPEN>(SelectObject(hdc, pen));
    HBRUSH oldBrush = static_cast<HBRUSH>(SelectObject(hdc, GetStockObject(NULL_BRUSH)));
    RoundRect(hdc, rc.left, rc.top, rc.right, rc.bottom, radius * 2, radius * 2);
    SelectObject(hdc, oldPen);
    SelectObject(hdc, oldBrush);
    DeleteObject(pen);
}

void setControlFont(HWND control, HFONT font) {
    SendMessageA(control, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
}

HWND addControl(const char* className, const char* text, DWORD style,
                int x, int y, int w, int h, int id) {
    HWND control = CreateWindowExA(
        0, className, text, WS_CHILD | WS_VISIBLE | style,
        x, y, w, h, g_state.hwnd,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
        GetModuleHandleA(nullptr), nullptr);
    return control;
}

// ------------------------------------------------------------------
// Hover tracking for owner-drawn controls
// ------------------------------------------------------------------
LRESULT CALLBACK styledControlProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam,
                                   UINT_PTR subclassId, DWORD_PTR refData) {
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

LRESULT CALLBACK comboNcProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam,
                             UINT_PTR subclassId, DWORD_PTR refData) {
    LRESULT result = DefSubclassProc(hwnd, message, wParam, lParam);
    if (message == WM_NCPAINT) {
        HDC hdc = GetWindowDC(hwnd);
        if (hdc) {
            RECT rc = {};
            GetWindowRect(hwnd, &rc);
            OffsetRect(&rc, -rc.left, -rc.top);
            bool active = (GetFocus() == hwnd);
            HPEN pen = CreatePen(PS_SOLID, 1, active ? ColAccent : ColBorder);
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
            RECT button = { rc.right - 26, rc.top + 2, rc.right - 3, rc.bottom - 2 };
            FillRect(hdc, &button, g_brField);
            frameRoundRect(hdc, button, ColBorder, 4);
            int cx = (button.left + button.right) / 2;
            int cy = (button.top + button.bottom) / 2;
            HPEN pen = CreatePen(PS_SOLID, 2, ColAccent);
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

    // Defensive base fill: guarantees the item is never left system-white.
    FillRect(hdc, &rc, g_brPanelHi);

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
        fillRoundRect(hdc, box, checked ? ColAccent : ColField, 4);
        frameRoundRect(hdc, box, checked ? ColAccent : (hovered ? ColAccentDk : ColBorder), 4);
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
        int radius = std::max(2, static_cast<int>((rc.bottom - rc.top) / 2 - 1));
        if (tabActive) {
            fillRoundRect(hdc, rc, ColAccent, radius);
            SetBkMode(hdc, TRANSPARENT);
            SetTextColor(hdc, ColTextOn);
        } else {
            fillRoundRect(hdc, rc, ColPanelHi, radius);
            frameRoundRect(hdc, rc, hovered ? ColAccentDk : ColBorder, radius);
            SetBkMode(hdc, TRANSPARENT);
            SetTextColor(hdc, hovered ? ColText : ColDim);
        }
        DrawTextA(hdc, text, -1, &rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        return;
    }

    int radius = 9;
    if (primary) {
        COLORREF fill = disabled ? RGB(30, 40, 52) : pressed ? ColAccentDk : ColAccent;
        fillRoundRect(hdc, rc, fill, radius);
        if (hovered && !disabled) frameRoundRect(hdc, rc, RGB(180, 245, 255), radius);
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, disabled ? ColDim : ColTextOn);
    } else {
        fillRoundRect(hdc, rc, disabled ? ColPanel : ColPanelHi, radius);
        frameRoundRect(hdc, rc, hovered ? ColAccent : ColBorder, radius);
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, disabled ? ColDim : (hovered ? ColAccent : ColText));
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

    char text[128] = {};
    SendMessageA(draw->hwndItem, CB_GETLBTEXT, draw->itemID,
                 reinterpret_cast<LPARAM>(text));

    bool isEditPart = (draw->itemState & ODS_COMBOBOXEDIT) != 0;
    bool highlighted = !isEditPart &&
        ((draw->itemState & ODS_SELECTED) != 0 || (draw->itemState & ODS_FOCUS) != 0);

    FillRect(hdc, &rc, highlighted ? g_brPanel : g_brField);
    if (highlighted) frameRoundRect(hdc, rc, ColAccent, 0);
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, highlighted ? ColAccent : ColText);

    RECT textRect = rc;
    textRect.left += 10;
    DrawTextA(hdc, text, -1, &textRect, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    if (isEditPart) {
        // Dropdown caret
        int cx = rc.right - 16;
        int cy = (rc.top + rc.bottom) / 2;
        HPEN pen = CreatePen(PS_SOLID, 2, ColAccent);
        HPEN oldPen = static_cast<HPEN>(SelectObject(hdc, pen));
        MoveToEx(hdc, cx - 5, cy - 2, nullptr);
        LineTo(hdc, cx, cy + 3);
        LineTo(hdc, cx + 5, cy - 2);
        SelectObject(hdc, oldPen);
        DeleteObject(pen);
    }
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
    const char* multipliers[] = { "2x", "3x", "4x", "Max" };
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

bool launchJob(const std::vector<std::string>& arguments, JobType jobType) {
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
    if (!CreateProcessA(
            executable,
            mutableCommand.data(),
            nullptr,
            nullptr,
            FALSE,
            CREATE_NO_WINDOW,
            nullptr,
            nullptr,
            &startup,
            &process)) {
        return false;
    }

    CloseHandle(process.hThread);
    HWND notifyWindow = g_state.hwnd;
    std::thread([handle = process.hProcess, notifyWindow, jobType]() {
        WaitForSingleObject(handle, INFINITE);
        DWORD exitCode = 1;
        GetExitCodeProcess(handle, &exitCode);
        CloseHandle(handle);
        PostMessageA(notifyWindow, WM_PROCESS_COMPLETE, exitCode, static_cast<LPARAM>(jobType));
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
    dialog.lpstrFilter = "PNG images (*.png)\0*.png\0All files (*.*)\0*.*\0";
    dialog.lpstrDefExt = "png";
    dialog.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | (saveDialog ? OFN_OVERWRITEPROMPT : OFN_FILEMUSTEXIST);

    BOOL selected = saveDialog ? GetSaveFileNameA(&dialog) : GetOpenFileNameA(&dialog);
    if (selected) SetWindowTextA(edit, path);
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
    int selection = static_cast<int>(SendMessageA(g_state.windowList, CB_GETCURSEL, 0, 0));
    if (selection == CB_ERR) {
        SetWindowTextA(g_state.liveStatus, "Choose a target window first.");
        return;
    }

    int length = static_cast<int>(SendMessageA(g_state.windowList, CB_GETLBTEXTLEN, selection, 0));
    std::string title(static_cast<size_t>(length) + 1, '\0');
    SendMessageA(g_state.windowList, CB_GETLBTEXT, selection, reinterpret_cast<LPARAM>(title.data()));
    title.resize(static_cast<size_t>(length));

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

    if (!launchJob({
            "--capture-window", title,
            "--source-fps", lowercaseWindowText(g_state.liveSourceFps),
            "--multiplier", lowercaseWindowText(g_state.liveMultiplier),
            "--gpu-levels", getWindowText(g_state.gpuLevels),
            "--gpu-min-refine", getWindowText(g_state.gpuMinRefine),
            "--gpu-coarse-radius", getWindowText(g_state.gpuCoarseRadius),
            "--gpu-refine-radius", getWindowText(g_state.gpuRefineRadius),
            "--gpu-smoothness", getWindowText(g_state.gpuSmoothness)
        }, JobType::Live)) {
        SetWindowTextA(g_state.liveStatus, "Could not start the capture process.");
        return;
    }

    EnableWindow(g_state.captureButton, FALSE);
    SetWindowTextA(g_state.liveStatus, "Capture running. Use Ctrl+Alt+F1 to toggle or Ctrl+Alt+Esc to exit.");
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
        !validateFloat(g_state.offlineOcclusion, 0.0f, 100.0f, "Occlusion threshold", g_state.offlineStatus) ||
        !validateFloat(g_state.offlineDetailStrength, 0.0f, 1.0f, "Detail strength", g_state.offlineStatus) ||
        !validateFloat(g_state.offlinePhotoThreshold, 0.0f, 255.0f, "Photo threshold", g_state.offlineStatus) ||
        !validateFloat(g_state.offlinePhotoSigma, 0.01f, 255.0f, "Photo sigma", g_state.offlineStatus)) {
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
            "--occlusion-threshold", getWindowText(g_state.offlineOcclusion),
            "--detail-strength", getWindowText(g_state.offlineDetailStrength),
            "--photo-threshold", getWindowText(g_state.offlinePhotoThreshold),
            "--photo-sigma", getWindowText(g_state.offlinePhotoSigma),
            "--adaptive-block", checkValue(g_state.adaptiveBlock),
            "--spatial-pred", checkValue(g_state.spatialPredictors),
            "--gaussian-weight", checkValue(g_state.gaussianWeights),
            "--bidirectional", checkValue(g_state.bidirectional),
            "--photo-gate", checkValue(g_state.photoGate),
            "--hole-inpaint", checkValue(g_state.holeInpaint),
            "--detail-restore", checkValue(g_state.detailRestore),
            "--color-clamp", checkValue(g_state.colorClamp)
        }, JobType::Offline)) {
        SetWindowTextA(g_state.offlineStatus, "Could not start offline processing.");
        return;
    }

    EnableWindow(g_state.offlineButton, FALSE);
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
    SendMessageA(checkbox, BM_SETCHECK, checked ? BST_CHECKED : BST_UNCHECKED, 0);
    SetPropA(checkbox, "MeCheckbox", reinterpret_cast<HANDLE>(static_cast<INT_PTR>(1)));
    setControlFont(checkbox, g_fontLabel);
    SetWindowSubclass(checkbox, styledControlProc, 1, 0);
    return checkbox;
}

void createControls() {
    ensureTheme();

    g_state.tabLive = makeButton("LIVE CAPTURE", 36, 14, 170, 38, IDC_TAB_LIVE);
    g_state.tabOffline = makeButton("OFFLINE INTERPOLATION", 216, 14, 240, 38, IDC_TAB_OFFLINE);

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

    HWND gpuLevelsLabel = makeLabel("Pyramid levels (1-8)", 56, 208, 160);
    g_state.gpuLevels = makeEdit("7", 224, 204, 80, IDC_GPU_LEVELS, true);
    HWND gpuMinRefineLabel = makeLabel("Finest searched level", 430, 208, 175);
    g_state.gpuMinRefine = makeEdit("0", 610, 204, 80, IDC_GPU_MIN_REFINE, true);
    HWND gpuCoarseLabel = makeLabel("Coarse radius (0-8)", 56, 248, 160);
    g_state.gpuCoarseRadius = makeEdit("8", 224, 244, 80, IDC_GPU_COARSE_RADIUS, true);
    HWND gpuRefineLabel = makeLabel("Refine radius (0-8)", 430, 248, 175);
    g_state.gpuRefineRadius = makeEdit("8", 610, 244, 80, IDC_GPU_REFINE_RADIUS, true);
    HWND gpuSmoothnessLabel = makeLabel("Smoothness (0-0.1)", 56, 288, 160);
    g_state.gpuSmoothness = makeEdit("0", 224, 284, 80, IDC_GPU_SMOOTHNESS, false);

    g_state.captureButton = makeButton("START CAPTURE", 36, 372, 180, 40, IDC_START_CAPTURE);
    HWND liveHelp = addControl(
        "STATIC",
        "Values remain fixed during capture. Ctrl+Alt+F1 toggles the overlay; Ctrl+Alt+Esc exits.",
        SS_LEFT, 36, 436, 760, 40, 0);
    setControlFont(liveHelp, g_fontLabel);
    g_state.liveStatus = addControl("STATIC", "", SS_LEFT, 36, 622, 760, 24, IDC_LIVE_STATUS);
    setControlFont(g_state.liveStatus, g_fontBody);

    g_state.liveControls = {
        liveHeading, g_state.windowList, g_state.refreshButton,
        liveFpsLabel, g_state.liveSourceFps, multiplierLabel, g_state.liveMultiplier,
        gpuLevelsLabel, g_state.gpuLevels, gpuMinRefineLabel, g_state.gpuMinRefine,
        gpuCoarseLabel, g_state.gpuCoarseRadius, gpuRefineLabel, g_state.gpuRefineRadius,
        gpuSmoothnessLabel, g_state.gpuSmoothness,
        g_state.captureButton, liveHelp, g_state.liveStatus
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

    HWND levelsLabel = makeLabel("Pyramid levels", 56, 302, 145);
    g_state.offlineLevels = makeEdit("8", 205, 298, 70, IDC_OFFLINE_LEVELS, true);
    HWND blockLabel = makeLabel("Block size", 56, 334, 145);
    g_state.offlineBlockSize = makeEdit("8", 205, 330, 70, IDC_OFFLINE_BLOCK_SIZE, true);
    HWND minBlockLabel = makeLabel("Minimum block", 56, 366, 145);
    g_state.offlineMinBlock = makeEdit("6", 205, 362, 70, IDC_OFFLINE_MIN_BLOCK, true);
    HWND maxBlockLabel = makeLabel("Maximum block", 56, 398, 145);
    g_state.offlineMaxBlock = makeEdit("14", 205, 394, 70, IDC_OFFLINE_MAX_BLOCK, true);
    HWND coarseLabel = makeLabel("Coarse radius", 56, 430, 145);
    g_state.offlineCoarseRadius = makeEdit("8", 205, 426, 70, IDC_OFFLINE_COARSE_RADIUS, true);
    HWND refineLabel = makeLabel("Refine radius", 56, 462, 145);
    g_state.offlineRefineRadius = makeEdit("4", 205, 458, 70, IDC_OFFLINE_REFINE_RADIUS, true);
    HWND gridLabel = makeLabel("Grid step", 56, 494, 145);
    g_state.offlineGridStep = makeEdit("2", 205, 490, 70, IDC_OFFLINE_GRID_STEP, true);
    HWND subpelLabel = makeLabel("Subpixel mode", 56, 526, 145);
    g_state.offlineSubpel = makeCombo(205, 522, 120, IDC_OFFLINE_SUBPEL);
    populateSubpixelModes(g_state.offlineSubpel);

    HWND smoothnessLabel = makeLabel("Smoothness", 430, 302, 150);
    g_state.offlineSmoothness = makeEdit("0.0005", 580, 298, 80, IDC_OFFLINE_SMOOTHNESS, false);
    HWND occlusionLabel = makeLabel("Occlusion threshold", 430, 334, 150);
    g_state.offlineOcclusion = makeEdit("2.5", 580, 330, 80, IDC_OFFLINE_OCCLUSION, false);
    HWND detailLabel = makeLabel("Detail strength", 430, 366, 150);
    g_state.offlineDetailStrength = makeEdit("0.65", 580, 362, 80, IDC_OFFLINE_DETAIL_STRENGTH, false);
    HWND photoThresholdLabel = makeLabel("Photo threshold", 430, 398, 150);
    g_state.offlinePhotoThreshold = makeEdit("20", 580, 394, 80, IDC_OFFLINE_PHOTO_THRESHOLD, false);
    HWND photoSigmaLabel = makeLabel("Photo sigma", 430, 430, 150);
    g_state.offlinePhotoSigma = makeEdit("25", 580, 426, 80, IDC_OFFLINE_PHOTO_SIGMA, false);

    g_state.adaptiveBlock = makeCheckbox("Adaptive blocks", 430, 462, 150, true);
    g_state.spatialPredictors = makeCheckbox("Spatial predictors", 580, 462, 170, true);
    g_state.gaussianWeights = makeCheckbox("Gaussian weights", 430, 488, 150, true);
    g_state.bidirectional = makeCheckbox("Bidirectional", 580, 488, 170, true);
    g_state.photoGate = makeCheckbox("Photometric gate", 430, 514, 150, true);
    g_state.holeInpaint = makeCheckbox("Hole inpainting", 580, 514, 170, true);
    g_state.detailRestore = makeCheckbox("Detail restoration", 430, 540, 150, true);
    g_state.colorClamp = makeCheckbox("Color clamp", 580, 540, 170, true);

    g_state.offlineButton = makeButton("INTERPOLATE", 36, 620, 180, 40, IDC_RUN_OFFLINE);
    g_state.offlineStatus = addControl("STATIC", "Select two PNG frames to begin.", SS_LEFT, 232, 628, 590, 24, IDC_OFFLINE_STATUS);
    setControlFont(g_state.offlineStatus, g_fontBody);

    g_state.offlineControls = {
        input0Label, g_state.input0, input0Browse, input1Label, g_state.input1, input1Browse,
        outputLabel, g_state.output, outputBrowse, timeLabel, g_state.time,
        levelsLabel, g_state.offlineLevels, blockLabel, g_state.offlineBlockSize,
        minBlockLabel, g_state.offlineMinBlock, maxBlockLabel, g_state.offlineMaxBlock,
        coarseLabel, g_state.offlineCoarseRadius, refineLabel, g_state.offlineRefineRadius,
        gridLabel, g_state.offlineGridStep, subpelLabel, g_state.offlineSubpel,
        smoothnessLabel, g_state.offlineSmoothness, occlusionLabel, g_state.offlineOcclusion,
        detailLabel, g_state.offlineDetailStrength, photoThresholdLabel, g_state.offlinePhotoThreshold,
        photoSigmaLabel, g_state.offlinePhotoSigma, g_state.adaptiveBlock, g_state.spatialPredictors,
        g_state.gaussianWeights, g_state.bidirectional, g_state.photoGate, g_state.holeInpaint,
        g_state.detailRestore, g_state.colorClamp,
        g_state.offlineButton, g_state.offlineStatus
    };

    applyPageVisibility();
    refreshWindowList();
}

// ------------------------------------------------------------------
// Custom painting
// ------------------------------------------------------------------
void paintSectionPanel(HDC hdc, const RECT& rc, const char* title) {
    fillRoundRect(hdc, rc, ColPanel, 10);
    frameRoundRect(hdc, rc, ColBorder, 10);

    RECT dot = { rc.left + 16, rc.top + 15, rc.left + 24, rc.top + 23 };
    fillRoundRect(hdc, dot, ColAccent, 4);

    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, ColText);
    RECT titleRect = { rc.left + 32, rc.top + 8, rc.right - 16, rc.top + 30 };
    HFONT oldFont = static_cast<HFONT>(SelectObject(hdc, g_fontSection));
    DrawTextA(hdc, title, -1, &titleRect, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, oldFont);
}

void paintHeader(HDC hdc, const RECT& client) {
    // Single clean accent divider under the tab row.
    int width = client.right - client.left;
    for (int x = 36; x < std::min(828, width); x += 3) {
        double t = static_cast<double>(x - 36) / 792.0;
        COLORREF color = lerpColor(ColAccent, ColBg, t);
        RECT segment = { x, 62, x + 3, 65 };
        HBRUSH brush = CreateSolidBrush(color);
        FillRect(hdc, &segment, brush);
        DeleteObject(brush);
    }
}

void paintClient(HWND hwnd) {
    PAINTSTRUCT paint = {};
    HDC hdc = BeginPaint(hwnd, &paint);
    RECT client = {};
    GetClientRect(hwnd, &client);

    HBRUSH background = CreateSolidBrush(ColBg);
    FillRect(hdc, &client, background);
    DeleteObject(background);

    paintHeader(hdc, client);
    if (g_state.activeTab == 0) {
        paintSectionPanel(hdc, LivePanelRect, "ADVANCED GPU FLOW");
    } else {
        paintSectionPanel(hdc, OfflinePanelRect, "ADVANCED INTERPOLATION");
    }

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
            int controlId = GetDlgCtrlID(reinterpret_cast<HWND>(lParam));
            SetBkMode(hdc, TRANSPARENT);
            if (controlId == IDC_LIVE_STATUS || controlId == IDC_OFFLINE_STATUS) {
                SetTextColor(hdc, ColAccent);
            } else {
                SetTextColor(hdc, ColDim);
            }
            return reinterpret_cast<LRESULT>(GetStockObject(NULL_BRUSH));
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
                case IDC_BROWSE_INPUT0: selectFile(g_state.input0, false); return 0;
                case IDC_BROWSE_INPUT1: selectFile(g_state.input1, false); return 0;
                case IDC_BROWSE_OUTPUT: selectFile(g_state.output, true); return 0;
                case IDC_RUN_OFFLINE: runOffline(); return 0;
            }
            break;

        case WM_PROCESS_COMPLETE: {
            DWORD exitCode = static_cast<DWORD>(wParam);
            JobType jobType = static_cast<JobType>(lParam);
            if (jobType == JobType::Live) {
                EnableWindow(g_state.captureButton, TRUE);
                SetWindowTextA(g_state.liveStatus, exitCode == 0 ? "Capture stopped." : "Capture ended with an error.");
            } else {
                EnableWindow(g_state.offlineButton, TRUE);
                SetWindowTextA(g_state.offlineStatus, exitCode == 0 ? "Interpolation complete." : "Interpolation failed.");
            }
            return 0;
        }

        case WM_DESTROY:
            g_state.hwnd = nullptr;
            PostQuitMessage(0);
            return 0;
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
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, 900, 748,
        nullptr, nullptr, instance, nullptr
    );
    if (!window) return 1;

    // Dark immersive title bar where supported.
    BOOL dark = TRUE;
    DwmSetWindowAttribute(window, 20, &dark, sizeof(dark));
    DwmSetWindowAttribute(window, 19, &dark, sizeof(dark));
    COLORREF captionColor = ColBg;
    DwmSetWindowAttribute(window, 1024, &captionColor, sizeof(captionColor));

    ShowWindow(window, SW_SHOW);
    UpdateWindow(window);

    MSG message;
    while (GetMessageA(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageA(&message);
    }
    return static_cast<int>(message.wParam);
}
