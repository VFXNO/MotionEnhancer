cmake_minimum_required(VERSION 3.20)
project(MotionEnhancerInstallerTheme LANGUAGES CXX)

# NSIS itself is 32-bit, including when its payload is an x64 application.
if(NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "Build the NSIS theme with the MSVC Win32 generator.")
endif()
if(NOT EXISTS "${NSIS_DIR}/Examples/Plugin/nsis/pluginapi.h")
    message(FATAL_ERROR "NSIS_DIR must point to a complete NSIS 3 installation.")
endif()
add_library(MotionTheme SHARED MotionTheme.cpp)
target_compile_features(MotionTheme PRIVATE cxx_std_17)
target_compile_definitions(MotionTheme PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(MotionTheme PRIVATE "${NSIS_DIR}/Examples/Plugin")
target_link_libraries(MotionTheme PRIVATE comctl32 gdiplus uxtheme dwmapi winmm)
target_compile_options(MotionTheme PRIVATE /W4 /O2 /EHsc)
# The installer plug-in must not require a preinstalled Visual C++ runtime.
set_property(TARGET MotionTheme PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
