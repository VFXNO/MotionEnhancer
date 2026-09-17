#include "GuiApp.h"
#include "WindowHelper.h"

#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int IDC_TABS = 100;
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

struct GuiState {
    HWND hwnd = nullptr;
    HWND tabs = nullptr;
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

HWND addControl(
    const char* className,
    const char* text,
    DWORD style,
    int x,
    int y,
    int width,
    int height,
    int id
) {
    HWND control = CreateWindowExA(
        0, className, text, WS_CHILD | WS_VISIBLE | style,
        x, y, width, height, g_state.hwnd,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
        GetModuleHandleA(nullptr), nullptr
    );
    SendMessageA(control, WM_SETFONT, reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
    return control;
}

void refreshWindowList() {
    SendMessageA(g_state.windowList, CB_RESETCONTENT, 0, 0);
    auto windows = WindowHelper::enumerateWindows();
    for (const auto& window : windows) {
        SendMessageA(g_state.windowList, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(window.title.c_str()));
    }
    if (!windows.empty()) {
        SendMessageA(g_state.windowList, CB_SETCURSEL, 0, 0);
        SetWindowTextA(g_state.liveStatus, "Select a window and start capture.");
    } else {
        SetWindowTextA(g_state.liveStatus, "No capturable windows found.");
    }
}

void selectFile(HWND edit, bool saveDialog) {
    char path[MAX_PATH] = {};
    std::string current = getWindowText(edit);
    if (!current.empty()) lstrcpynA(path, current.c_str(), MAX_PATH);

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

void showSelectedTab() {
    bool showLive = TabCtrl_GetCurSel(g_state.tabs) == 0;
    for (HWND control : g_state.liveControls) ShowWindow(control, showLive ? SW_SHOW : SW_HIDE);
    for (HWND control : g_state.offlineControls) ShowWindow(control, showLive ? SW_HIDE : SW_SHOW);
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

void createControls() {
    g_state.tabs = addControl(WC_TABCONTROLA, "", WS_CLIPSIBLINGS, 12, 12, 860, 650, IDC_TABS);
    TCITEMA item = {};
    item.mask = TCIF_TEXT;
    item.pszText = const_cast<char*>("Live Capture");
    TabCtrl_InsertItem(g_state.tabs, 0, &item);
    item.pszText = const_cast<char*>("Offline Interpolation");
    TabCtrl_InsertItem(g_state.tabs, 1, &item);

    HWND liveHeading = addControl("STATIC", "Target window", SS_LEFT, 36, 58, 760, 20, 0);
    g_state.windowList = addControl("COMBOBOX", "", CBS_DROPDOWNLIST | WS_VSCROLL, 36, 80, 650, 260, IDC_WINDOW_LIST);
    g_state.refreshButton = addControl("BUTTON", "Refresh", BS_PUSHBUTTON, 700, 79, 130, 28, IDC_REFRESH_WINDOWS);
    HWND liveFpsLabel = addControl("STATIC", "Source FPS", SS_LEFT, 36, 126, 100, 20, 0);
    g_state.liveSourceFps = addControl("COMBOBOX", "", CBS_DROPDOWNLIST | WS_VSCROLL, 140, 122, 130, 140, IDC_LIVE_SOURCE_FPS);
    populateSourceRates(g_state.liveSourceFps);
    HWND multiplierLabel = addControl("STATIC", "Output multiplier", SS_LEFT, 330, 126, 125, 20, 0);
    g_state.liveMultiplier = addControl("COMBOBOX", "", CBS_DROPDOWNLIST | WS_VSCROLL, 460, 122, 130, 140, IDC_LIVE_MULTIPLIER);
    populateMultipliers(g_state.liveMultiplier);
    HWND liveAdvanced = addControl("BUTTON", "Advanced GPU flow", BS_GROUPBOX, 36, 164, 794, 174, 0);
    HWND gpuLevelsLabel = addControl("STATIC", "Pyramid levels (1-8)", SS_LEFT, 56, 198, 160, 20, 0);
    g_state.gpuLevels = addControl("EDIT", "7", WS_BORDER | ES_NUMBER, 224, 194, 80, 24, IDC_GPU_LEVELS);
    HWND gpuMinRefineLabel = addControl("STATIC", "Finest searched level", SS_LEFT, 430, 198, 170, 20, 0);
    g_state.gpuMinRefine = addControl("EDIT", "0", WS_BORDER | ES_NUMBER, 610, 194, 80, 24, IDC_GPU_MIN_REFINE);
    HWND gpuCoarseLabel = addControl("STATIC", "Coarse radius (0-8)", SS_LEFT, 56, 238, 160, 20, 0);
    g_state.gpuCoarseRadius = addControl("EDIT", "8", WS_BORDER | ES_NUMBER, 224, 234, 80, 24, IDC_GPU_COARSE_RADIUS);
    HWND gpuRefineLabel = addControl("STATIC", "Refine radius (0-8)", SS_LEFT, 430, 238, 170, 20, 0);
    g_state.gpuRefineRadius = addControl("EDIT", "8", WS_BORDER | ES_NUMBER, 610, 234, 80, 24, IDC_GPU_REFINE_RADIUS);
    HWND gpuSmoothnessLabel = addControl("STATIC", "Smoothness (0-0.1)", SS_LEFT, 56, 278, 160, 20, 0);
    g_state.gpuSmoothness = addControl("EDIT", "0", WS_BORDER | ES_AUTOHSCROLL, 224, 274, 80, 24, IDC_GPU_SMOOTHNESS);
    g_state.captureButton = addControl("BUTTON", "Start Capture", BS_DEFPUSHBUTTON, 36, 362, 170, 34, IDC_START_CAPTURE);
    HWND liveHelp = addControl("STATIC", "Values remain fixed during capture. Ctrl+Alt+F1 toggles the overlay; Ctrl+Alt+Esc exits.", SS_LEFT, 36, 420, 760, 42, 0);
    g_state.liveStatus = addControl("STATIC", "", SS_LEFT, 36, 610, 760, 26, IDC_LIVE_STATUS);
    g_state.liveControls = {
        liveHeading, g_state.windowList, g_state.refreshButton, liveFpsLabel, g_state.liveSourceFps,
        multiplierLabel, g_state.liveMultiplier,
        liveAdvanced, gpuLevelsLabel, g_state.gpuLevels, gpuMinRefineLabel, g_state.gpuMinRefine,
        gpuCoarseLabel, g_state.gpuCoarseRadius, gpuRefineLabel, g_state.gpuRefineRadius,
        gpuSmoothnessLabel, g_state.gpuSmoothness,
        g_state.captureButton, liveHelp, g_state.liveStatus
    };

    HWND input0Label = addControl("STATIC", "First frame", SS_LEFT, 36, 52, 760, 18, 0);
    g_state.input0 = addControl("EDIT", "", WS_BORDER | ES_AUTOHSCROLL, 36, 72, 650, 24, IDC_INPUT0);
    HWND input0Browse = addControl("BUTTON", "Browse...", BS_PUSHBUTTON, 700, 70, 130, 28, IDC_BROWSE_INPUT0);
    HWND input1Label = addControl("STATIC", "Second frame", SS_LEFT, 36, 102, 760, 18, 0);
    g_state.input1 = addControl("EDIT", "", WS_BORDER | ES_AUTOHSCROLL, 36, 122, 650, 24, IDC_INPUT1);
    HWND input1Browse = addControl("BUTTON", "Browse...", BS_PUSHBUTTON, 700, 120, 130, 28, IDC_BROWSE_INPUT1);
    HWND outputLabel = addControl("STATIC", "Output PNG", SS_LEFT, 36, 152, 760, 18, 0);
    g_state.output = addControl("EDIT", "", WS_BORDER | ES_AUTOHSCROLL, 36, 172, 650, 24, IDC_OUTPUT);
    HWND outputBrowse = addControl("BUTTON", "Browse...", BS_PUSHBUTTON, 700, 170, 130, 28, IDC_BROWSE_OUTPUT);
    HWND timeLabel = addControl("STATIC", "Interpolation time (0-1)", SS_LEFT, 36, 212, 180, 20, 0);
    g_state.time = addControl("EDIT", "0.5", WS_BORDER | ES_AUTOHSCROLL, 220, 208, 80, 24, IDC_TIME);

    HWND offlineAdvanced = addControl("BUTTON", "Advanced interpolation", BS_GROUPBOX, 36, 242, 794, 330, 0);
    HWND levelsLabel = addControl("STATIC", "Pyramid levels", SS_LEFT, 56, 272, 145, 20, 0);
    g_state.offlineLevels = addControl("EDIT", "8", WS_BORDER | ES_NUMBER, 205, 268, 70, 24, IDC_OFFLINE_LEVELS);
    HWND blockLabel = addControl("STATIC", "Block size", SS_LEFT, 56, 304, 145, 20, 0);
    g_state.offlineBlockSize = addControl("EDIT", "8", WS_BORDER | ES_NUMBER, 205, 300, 70, 24, IDC_OFFLINE_BLOCK_SIZE);
    HWND minBlockLabel = addControl("STATIC", "Minimum block", SS_LEFT, 56, 336, 145, 20, 0);
    g_state.offlineMinBlock = addControl("EDIT", "6", WS_BORDER | ES_NUMBER, 205, 332, 70, 24, IDC_OFFLINE_MIN_BLOCK);
    HWND maxBlockLabel = addControl("STATIC", "Maximum block", SS_LEFT, 56, 368, 145, 20, 0);
    g_state.offlineMaxBlock = addControl("EDIT", "14", WS_BORDER | ES_NUMBER, 205, 364, 70, 24, IDC_OFFLINE_MAX_BLOCK);
    HWND coarseLabel = addControl("STATIC", "Coarse radius", SS_LEFT, 56, 400, 145, 20, 0);
    g_state.offlineCoarseRadius = addControl("EDIT", "8", WS_BORDER | ES_NUMBER, 205, 396, 70, 24, IDC_OFFLINE_COARSE_RADIUS);
    HWND refineLabel = addControl("STATIC", "Refine radius", SS_LEFT, 56, 432, 145, 20, 0);
    g_state.offlineRefineRadius = addControl("EDIT", "4", WS_BORDER | ES_NUMBER, 205, 428, 70, 24, IDC_OFFLINE_REFINE_RADIUS);
    HWND gridLabel = addControl("STATIC", "Grid step", SS_LEFT, 56, 464, 145, 20, 0);
    g_state.offlineGridStep = addControl("EDIT", "2", WS_BORDER | ES_NUMBER, 205, 460, 70, 24, IDC_OFFLINE_GRID_STEP);
    HWND subpelLabel = addControl("STATIC", "Subpixel mode", SS_LEFT, 56, 496, 145, 20, 0);
    g_state.offlineSubpel = addControl("COMBOBOX", "", CBS_DROPDOWNLIST | WS_VSCROLL, 205, 492, 120, 120, IDC_OFFLINE_SUBPEL);
    populateSubpixelModes(g_state.offlineSubpel);

    HWND smoothnessLabel = addControl("STATIC", "Smoothness", SS_LEFT, 420, 272, 150, 20, 0);
    g_state.offlineSmoothness = addControl("EDIT", "0.0005", WS_BORDER | ES_AUTOHSCROLL, 580, 268, 80, 24, IDC_OFFLINE_SMOOTHNESS);
    HWND occlusionLabel = addControl("STATIC", "Occlusion threshold", SS_LEFT, 420, 304, 150, 20, 0);
    g_state.offlineOcclusion = addControl("EDIT", "2.5", WS_BORDER | ES_AUTOHSCROLL, 580, 300, 80, 24, IDC_OFFLINE_OCCLUSION);
    HWND detailLabel = addControl("STATIC", "Detail strength", SS_LEFT, 420, 336, 150, 20, 0);
    g_state.offlineDetailStrength = addControl("EDIT", "0.65", WS_BORDER | ES_AUTOHSCROLL, 580, 332, 80, 24, IDC_OFFLINE_DETAIL_STRENGTH);
    HWND photoThresholdLabel = addControl("STATIC", "Photo threshold", SS_LEFT, 420, 368, 150, 20, 0);
    g_state.offlinePhotoThreshold = addControl("EDIT", "20", WS_BORDER | ES_AUTOHSCROLL, 580, 364, 80, 24, IDC_OFFLINE_PHOTO_THRESHOLD);
    HWND photoSigmaLabel = addControl("STATIC", "Photo sigma", SS_LEFT, 420, 400, 150, 20, 0);
    g_state.offlinePhotoSigma = addControl("EDIT", "25", WS_BORDER | ES_AUTOHSCROLL, 580, 396, 80, 24, IDC_OFFLINE_PHOTO_SIGMA);

    g_state.adaptiveBlock = addControl("BUTTON", "Adaptive blocks", BS_AUTOCHECKBOX, 420, 434, 150, 22, 0);
    g_state.spatialPredictors = addControl("BUTTON", "Spatial predictors", BS_AUTOCHECKBOX, 580, 434, 160, 22, 0);
    g_state.gaussianWeights = addControl("BUTTON", "Gaussian weights", BS_AUTOCHECKBOX, 420, 460, 150, 22, 0);
    g_state.bidirectional = addControl("BUTTON", "Bidirectional", BS_AUTOCHECKBOX, 580, 460, 160, 22, 0);
    g_state.photoGate = addControl("BUTTON", "Photometric gate", BS_AUTOCHECKBOX, 420, 486, 150, 22, 0);
    g_state.holeInpaint = addControl("BUTTON", "Hole inpainting", BS_AUTOCHECKBOX, 580, 486, 160, 22, 0);
    g_state.detailRestore = addControl("BUTTON", "Detail restoration", BS_AUTOCHECKBOX, 420, 512, 150, 22, 0);
    g_state.colorClamp = addControl("BUTTON", "Color clamp", BS_AUTOCHECKBOX, 580, 512, 160, 22, 0);
    for (HWND checkbox : { g_state.adaptiveBlock, g_state.spatialPredictors, g_state.gaussianWeights,
                          g_state.bidirectional, g_state.photoGate, g_state.holeInpaint,
                          g_state.detailRestore, g_state.colorClamp }) {
        SendMessageA(checkbox, BM_SETCHECK, BST_CHECKED, 0);
    }

    g_state.offlineButton = addControl("BUTTON", "Interpolate", BS_DEFPUSHBUTTON, 36, 590, 170, 34, IDC_RUN_OFFLINE);
    g_state.offlineStatus = addControl("STATIC", "Select two PNG frames to begin.", SS_LEFT, 230, 596, 590, 26, IDC_OFFLINE_STATUS);
    g_state.offlineControls = {
        input0Label, g_state.input0, input0Browse, input1Label, g_state.input1, input1Browse,
        outputLabel, g_state.output, outputBrowse, timeLabel, g_state.time,
        offlineAdvanced, levelsLabel, g_state.offlineLevels, blockLabel, g_state.offlineBlockSize,
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

    showSelectedTab();
    refreshWindowList();
}

LRESULT CALLBACK guiWindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_CREATE:
            g_state.hwnd = hwnd;
            createControls();
            return 0;

        case WM_NOTIFY:
            if (reinterpret_cast<NMHDR*>(lParam)->idFrom == IDC_TABS &&
                reinterpret_cast<NMHDR*>(lParam)->code == TCN_SELCHANGE) {
                showSelectedTab();
            }
            return 0;

        case WM_COMMAND:
            switch (LOWORD(wParam)) {
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
    controls.dwICC = ICC_TAB_CLASSES | ICC_STANDARD_CLASSES;
    InitCommonControlsEx(&controls);

    HINSTANCE instance = GetModuleHandleA(nullptr);
    WNDCLASSEXA windowClass = {};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.lpfnWndProc = guiWindowProc;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursor(nullptr, IDC_ARROW);
    windowClass.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    windowClass.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    windowClass.lpszClassName = "MotionEnhancerGuiClass";
    RegisterClassExA(&windowClass);

    HWND window = CreateWindowExA(
        0,
        windowClass.lpszClassName,
        "Motion Enhancer",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, 900, 720,
        nullptr, nullptr, instance, nullptr
    );
    if (!window) return 1;

    ShowWindow(window, SW_SHOW);
    UpdateWindow(window);

    MSG message;
    while (GetMessageA(&message, nullptr, 0, 0) > 0) {
        if (!IsDialogMessageA(window, &message)) {
            TranslateMessage(&message);
            DispatchMessageA(&message);
        }
    }
    return static_cast<int>(message.wParam);
}
