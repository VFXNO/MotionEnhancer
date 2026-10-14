#pragma once

#include <windows.h>
#include <string>
#include <vector>

struct WindowInfo {
    HWND hwnd = nullptr;
    std::string title;
    RECT rect = {};
};

class WindowHelper {
public:
    static bool attachInteractiveDesktop();
    static std::vector<WindowInfo> enumerateWindows();
    static HWND findWindowByTitle(const std::string& query);
    static void printWindowList();
};

class OverlayWindow {
public:
    HWND hwnd = nullptr;
    HWND targetHwnd = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    int posX = 0;
    int posY = 0;
    bool isRunning = true;

    OverlayWindow() = default;
    ~OverlayWindow();

    bool create(HWND targetWindow, const std::string& windowTitle = "Motion Enhancer Overlay");
    void updateTracking();
    bool processMessages();
    void show(bool visible = true);
};
