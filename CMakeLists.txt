cmake_minimum_required(VERSION 3.16)
project(MotionEnhancer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Source files
set(SOURCES
    src/main.cpp
    src/Image.cpp
    src/Pyramid.cpp
    src/ZNCCMatcher.cpp
    src/FrameInterpolator.cpp
    src/D3D11Context.cpp
    src/WGCCapture.cpp
    src/GPUInterpolator.cpp
    src/WindowHelper.cpp
    src/GuiApp.cpp
    src/RealtimePresenter.cpp
)

# Header files
set(HEADERS
    include/Image.h
    include/Pyramid.h
    include/MotionVector.h
    include/ZNCCMatcher.h
    include/FrameInterpolator.h
    include/D3D11Context.h
    include/WGCCapture.h
    include/GPUInterpolator.h
    include/WindowHelper.h
    include/GuiApp.h
    include/RealtimePresenter.h
    third_party/stb_image.h
    third_party/stb_image_write.h
)

add_executable(motion_enhancer ${SOURCES} ${HEADERS})

target_include_directories(motion_enhancer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# OpenMP multi-threading support for CPU fallback
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling multi-threaded CPU acceleration")
    target_link_libraries(motion_enhancer PRIVATE OpenMP::OpenMP_CXX)
endif()

# Windows libraries for DirectX 11, WGC, and DWM
if(WIN32)
    target_link_libraries(motion_enhancer PRIVATE
        d3d11
        dxgi
        d3dcompiler
        dwmapi
        comctl32
        comdlg32
        windowsapp
    )
    if(MSVC)
        target_compile_options(motion_enhancer PRIVATE /O2 /Oi /Ot /fp:fast /W3 /permissive- /await)
        target_compile_definitions(motion_enhancer PRIVATE 
            _CRT_SECURE_NO_WARNINGS 
            NOMINMAX 
            WIN32_LEAN_AND_MEAN
        )
    endif()

    add_custom_command(TARGET motion_enhancer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "$<TARGET_FILE_DIR:motion_enhancer>/shaders"
        COMMENT "Copying HLSL shaders to target binary directory"
    )
else()
    target_compile_options(motion_enhancer PRIVATE -O3 -ffast-math -Wall -Wextra)
endif()
