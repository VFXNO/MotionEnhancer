#include "WindowHelper.h"
#include <iostream>
#include <algorithm>
#include <thread>
#include <dwmapi.h>
#include <tlhelp32.h>

static DWORD getParentProcessId() {
    DWORD parentPid = 0;
    DWORD myPid = GetCurrentProcessId();
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnap != INVALID_HANDLE_VALUE) {
        PROCESSENTRY32 pe = {};
        pe.dwSize = sizeof(pe);
        if (Process32First(hSnap, &pe)) {
            do {
                if (pe.th32ProcessID == myPid) {
                    parentPid = pe.th32ParentProcessID;
                    break;
                }
            } while (Process32Next(hSnap, &pe));
        }
        CloseHandle(hSnap);
    }
    return parentPid;
}

bool WindowHelper::attachInteractiveDesktop() {
    HDESK hDesk = OpenDesktopA("Default", 0, FALSE, GENERIC_ALL);
    if (!hDesk) {
        hDesk = OpenInputDesktop(0, FALSE, GENERIC_ALL);
    }
    if (hDesk) {
        BOOL ok = SetThreadDesktop(hDesk);
        CloseDesktop(hDesk);
        return ok != FALSE;
    }
    return false;
}

static BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam) {
    if (!IsWindowVisible(hwnd)) return TRUE;

    // Check if cloaked by DWM (minimized virtual desktop or hidden UWP suspended window)
    int cloaked = 0;
    if (SUCCEEDED(DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, &cloaked, sizeof(cloaked))) && cloaked) {
        return TRUE;
    }

    // Exclude console window attached to this process
    if (hwnd == GetConsoleWindow()) return TRUE;

    // Exclude own process and parent process (e.g. terminal / shell that launched us)
    DWORD windowPid = 0;
    GetWindowThreadProcessId(hwnd, &windowPid);
    static DWORD s_myPid = GetCurrentProcessId();
    static DWORD s_parentPid = getParentProcessId();
    if (windowPid == s_myPid || (s_parentPid != 0 && windowPid == s_parentPid)) {
        return TRUE;
    }

    char title[256];
    int len = GetWindowTextA(hwnd, title, sizeof(title));
    if (len <= 0) return TRUE;

    RECT rect;
    GetWindowRect(hwnd, &rect);
    if ((rect.right - rect.left) <= 100 || (rect.bottom - rect.top) <= 100) return TRUE;

    // Exclude command prompt, PowerShell, terminal, or overlay windows that mirror our command line
    std::string lowerTitle = title;
    std::transform(lowerTitle.begin(), lowerTitle.end(), lowerTitle.begin(), ::tolower);
    if (lowerTitle.find("motion_enhancer") != std::string::npos ||
        lowerTitle.find("--capture-window") != std::string::npos ||
        lowerTitle.find("motion enhancer gpu overlay") != std::string::npos) {
        return TRUE;
    }

    auto* list = reinterpret_cast<std::vector<WindowInfo>*>(lParam);
    WindowInfo info;
    info.hwnd = hwnd;
    info.title = title;
    info.rect = rect;
    list->push_back(info);

    return TRUE;
}

std::vector<WindowInfo> WindowHelper::enumerateWindows() {
    std::vector<WindowInfo> windows;
    // Run enumeration in a clean thread attached to the interactive desktop
    std::thread worker([&]() {
        attachInteractiveDesktop();
        EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&windows));
    });
    worker.join();
    return windows;
}

HWND WindowHelper::findWindowByTitle(const std::string& query) {
    auto windows = enumerateWindows();
    std::string lowerQuery = query;
    std::transform(lowerQuery.begin(), lowerQuery.end(), lowerQuery.begin(), ::tolower);

    HWND bestHwnd = nullptr;
    size_t shortestLen = (size_t)-1;

    // First pass: exact match
    for (const auto& w : windows) {
        std::string lowerTitle = w.title;
        std::transform(lowerTitle.begin(), lowerTitle.end(), lowerTitle.begin(), ::tolower);
        if (lowerTitle == lowerQuery) {
            return w.hwnd;
        }
    }

    // Second pass: substring match (choose most specific / shortest title)
    for (const auto& w : windows) {
        std::string lowerTitle = w.title;
        std::transform(lowerTitle.begin(), lowerTitle.end(), lowerTitle.begin(), ::tolower);
        if (lowerTitle.find(lowerQuery) != std::string::npos) {
            if (lowerTitle.length() < shortestLen) {
                shortestLen = lowerTitle.length();
                bestHwnd = w.hwnd;
            }
        }
    }

    return bestHwnd;
}

void WindowHelper::printWindowList() {
    auto windows = enumerateWindows();
    std::cout << "\n===================================================\n"
              << " Active Desktop Windows Available for WGC Capture:\n"
              << "===================================================\n";
    for (size_t i = 0; i < windows.size(); ++i) {
        std::cout << " [" << i << "] HWND: 0x" << std::hex << (uintptr_t)windows[i].hwnd << std::dec
                  << " (" << (windows[i].rect.right - windows[i].rect.left) << "x"
                  << (windows[i].rect.bottom - windows[i].rect.top) << ") - \""
                  << windows[i].title << "\"\n";
    }
    std::cout << "===================================================\n\n";
}

static LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    OverlayWindow* self = reinterpret_cast<OverlayWindow*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

    switch (msg) {
        case WM_NCHITTEST:
            return HTTRANSPARENT;

        case WM_HOTKEY:
        case WM_KEYDOWN: {
            UINT key = msg == WM_HOTKEY ? HIWORD(lParam) : static_cast<UINT>(wParam);
            if (key == VK_ESCAPE) {
                if (self) self->isRunning = false;
                PostQuitMessage(0);
                return 0;
            } else if (key == VK_F1) {
                // Toggle overlay visibility
                BOOL visible = IsWindowVisible(hwnd);
                ShowWindow(hwnd, visible ? SW_HIDE : SW_SHOW);
                std::cout << "Overlay " << (visible ? "Hidden" : "Shown") << "\n";
                return 0;
            }
            break;
        }
        case WM_DESTROY: {
            UnregisterHotKey(hwnd, 1);
            UnregisterHotKey(hwnd, 2);
            if (self) self->isRunning = false;
            PostQuitMessage(0);
            return 0;
        }
    }

    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

OverlayWindow::~OverlayWindow() {
    if (hwnd && IsWindow(hwnd)) {
        DestroyWindow(hwnd);
        hwnd = nullptr;
    }
}

bool OverlayWindow::create(HWND target, const std::string& windowTitle) {
    targetHwnd = target;
    if (!targetHwnd || !IsWindow(targetHwnd)) {
        std::cerr << "Error: Invalid target window HWND provided to OverlayWindow\n";
        return false;
    }

    RECT clientRect;
    GetClientRect(targetHwnd, &clientRect);
    width = clientRect.right - clientRect.left;
    height = clientRect.bottom - clientRect.top;

    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Target window has invalid dimensions (" << width << "x" << height << "). Ensure it is not minimized.\n";
        return false;
    }

    POINT pt = { clientRect.left, clientRect.top };
    ClientToScreen(targetHwnd, &pt);

    HINSTANCE hInstance = GetModuleHandle(nullptr);
    const char* className = "MotionEnhancerOverlayClass";

    WNDCLASSEXA wc = {};
    wc.cbSize = sizeof(WNDCLASSEXA);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = OverlayWndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = className;

    RegisterClassExA(&wc);

    hwnd = CreateWindowExA(
        WS_EX_LAYERED | WS_EX_NOACTIVATE | WS_EX_TRANSPARENT |
            WS_EX_TOOLWINDOW | WS_EX_NOREDIRECTIONBITMAP,
        className,
        windowTitle.c_str(),
        WS_POPUP | WS_VISIBLE,
        pt.x, pt.y,
        width, height,
        targetHwnd,
        nullptr,
        hInstance,
        nullptr
    );

    if (!hwnd) {
        std::cerr << "Error: Failed to create overlay Win32 window (0x" << std::hex << GetLastError() << ")\n";
        return false;
    }

    SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
    // Keep the source visible until the presenter has submitted real content.
    SetLayeredWindowAttributes(hwnd, 0, 0, LWA_ALPHA);

    LONG_PTR appliedStyles = GetWindowLongPtr(hwnd, GWL_EXSTYLE);
    LONG_PTR requiredStyles = WS_EX_LAYERED | WS_EX_NOACTIVATE | WS_EX_TRANSPARENT;
    if ((appliedStyles & requiredStyles) != requiredStyles ||
        SendMessageA(hwnd, WM_NCHITTEST, 0, 0) != HTTRANSPARENT) {
        std::cerr << "Error: Overlay input pass-through could not be enabled.\n";
        DestroyWindow(hwnd);
        hwnd = nullptr;
        return false;
    }

    RegisterHotKey(hwnd, 1, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_F1);
    RegisterHotKey(hwnd, 2, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_ESCAPE);

#ifndef WDA_EXCLUDEFROMCAPTURE
#define WDA_EXCLUDEFROMCAPTURE 0x00000011
#endif
    SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);

    posX = pt.x;
    posY = pt.y;

    std::cout << "Overlay window created successfully (" << width << "x" << height << ") at (" << pt.x << ", " << pt.y << ").\n";
    std::cout << "Overlay input pass-through active (mouse and keyboard remain on source window).\n";
    std::cout << "Hotkeys: [Ctrl+Alt+F1] Toggle | [Ctrl+Alt+Esc] Exit\n";

    return true;
}

void OverlayWindow::updateTracking() {
    if (!targetHwnd || !IsWindow(targetHwnd) || !hwnd || !IsWindow(hwnd)) {
        isRunning = false;
        return;
    }

    RECT clientRect;
    GetClientRect(targetHwnd, &clientRect);
    POINT pt = { clientRect.left, clientRect.top };
    ClientToScreen(targetHwnd, &pt);

    int newW = clientRect.right - clientRect.left;
    int newH = clientRect.bottom - clientRect.top;

    if (newW > 0 && newH > 0 && (newW != (int)width || newH != (int)height || pt.x != posX || pt.y != posY)) {
        SetWindowPos(
            hwnd,
            nullptr,
            pt.x, pt.y,
            newW, newH,
            SWP_NOACTIVATE | SWP_NOREDRAW | SWP_NOZORDER
        );
        width = newW;
        height = newH;
        posX = pt.x;
        posY = pt.y;
    }
}

bool OverlayWindow::processMessages() {
    MSG msg;
    while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            isRunning = false;
            return false;
        }
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    return isRunning;
}

void OverlayWindow::show(bool visible) {
    if (hwnd && IsWindow(hwnd)) {
        ShowWindow(hwnd, visible ? SW_SHOW : SW_HIDE);
    }
}
