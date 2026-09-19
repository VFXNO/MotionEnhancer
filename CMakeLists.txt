cmake_minimum_required(VERSION 3.16)
project(MotionEnhancer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FIDELITYFX_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/FidelityFX-SDK-v1.1.4"
    CACHE PATH "FidelityFX SDK root for native Optical Flow")
option(MOTION_ENHANCER_ENABLE_FFX_OPTICAL_FLOW
        "Build the native AMD FidelityFX Optical Flow backend" ON)
if(MOTION_ENHANCER_ENABLE_FFX_OPTICAL_FLOW)
    if(NOT WIN32 OR NOT FIDELITYFX_SDK_DIR)
        message(FATAL_ERROR
            "Native FidelityFX Optical Flow requires Windows and "
            "-DFIDELITYFX_SDK_DIR=<FidelityFX-SDK-v1.1.4 root>.")
    endif()
    if(NOT EXISTS "${FIDELITYFX_SDK_DIR}/CMakeLists.txt" OR
       NOT EXISTS "${FIDELITYFX_SDK_DIR}/tools/binary_store/FidelityFX_SC.exe")
        message(FATAL_ERROR
            "FIDELITYFX_SDK_DIR must contain CMakeLists.txt and "
            "tools/binary_store/FidelityFX_SC.exe from FidelityFX SDK v1.1.4.")
    endif()
    # Use AMD's CMake path so FidelityFX_SC generates the required DX12 shader
    # permutation headers instead of copying generated SDK artifacts into this tree.
    set(FFX_API_BACKEND DX12_X64 CACHE STRING "" FORCE)
    set(FFX_OF ON CACHE BOOL "" FORCE)
    set(FFX_FSR3 OFF CACHE BOOL "" FORCE)
    set(FFX_ALL OFF CACHE BOOL "" FORCE)
    add_subdirectory("${FIDELITYFX_SDK_DIR}" "${CMAKE_BINARY_DIR}/fidelityfx-sdk" EXCLUDE_FROM_ALL)
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/Image.cpp
    src/Pyramid.cpp
    src/ZNCCMatcher.cpp
    src/FrameInterpolator.cpp
    src/D3D11Context.cpp
    src/D3D12Context.cpp
    src/WGCCapture.cpp
    src/GPUInterpolator.cpp
    src/WindowHelper.cpp
    src/GuiApp.cpp
    src/RealtimePresenter.cpp
    app.rc
)

# Header files
set(HEADERS
    include/Image.h
    include/Pyramid.h
    include/MotionVector.h
    include/ZNCCMatcher.h
    include/FrameInterpolator.h
    include/D3D11Context.h
    include/D3D12Context.h
    include/WGCCapture.h
    include/GPUInterpolator.h
    include/WindowHelper.h
    include/GuiApp.h
    include/RealtimePresenter.h
    third_party/stb_image.h
    third_party/stb_image_write.h
)

add_executable(motion_enhancer ${SOURCES} ${HEADERS})

# The manifest is embedded manually via app.rc; disable the linker's
# auto-generated manifest to avoid a duplicate resource.
if(MSVC)
    set_target_properties(motion_enhancer PROPERTIES LINK_FLAGS "/MANIFEST:NO")
endif()

target_include_directories(motion_enhancer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

if(MOTION_ENHANCER_ENABLE_FFX_OPTICAL_FLOW)
    target_link_libraries(motion_enhancer PRIVATE ffx_opticalflow_x64 ffx_backend_dx12_x64)
    target_compile_definitions(motion_enhancer PRIVATE MOTION_ENHANCER_FFX_OPTICAL_FLOW=1)
endif()

# OpenMP multi-threading support for CPU fallback
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling multi-threaded CPU acceleration")
    target_link_libraries(motion_enhancer PRIVATE OpenMP::OpenMP_CXX)
endif()

# Windows libraries for D3D12, the D3D11On12 bridge, WGC, and DWM
if(WIN32)
    target_link_libraries(motion_enhancer PRIVATE
        d3d11
        d3d12
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

    find_program(DXC_EXECUTABLE dxc
        HINTS "C:/VulkanSDK/1.4.309.0/Bin"
        DOC "DirectX Shader Compiler")
    if(DXC_EXECUTABLE)
        set(DXC_OUTPUTS)
        set(MOTION_ENHANCER_SHADERS Common.hlsli LuminanceCS.hlsl PyramidCS.hlsl MotionSearchCS.hlsl
                                     FilterFlowCS.hlsl InterpolateCS.hlsl PresentFrameCS.hlsl)
        if(MOTION_ENHANCER_ENABLE_FFX_OPTICAL_FLOW)
            list(APPEND MOTION_ENHANCER_SHADERS FfxFlowConvertCS.hlsl)
        endif()
        foreach(SHADER IN LISTS MOTION_ENHANCER_SHADERS)
            if(SHADER MATCHES "\\.hlsl$")
                string(REGEX REPLACE "\\.hlsl$" ".cso" SHADER_OBJECT "${SHADER}")
                add_custom_command(
                    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_OBJECT}"
                    COMMAND ${CMAKE_COMMAND} -E make_directory
                            "${CMAKE_CURRENT_BINARY_DIR}/shaders"
                    COMMAND "${DXC_EXECUTABLE}" -E CSMain -T cs_5_0 -O3
                            -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                            -Fo "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_OBJECT}"
                            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}"
                            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/Common.hlsli"
                    VERBATIM)
                list(APPEND DXC_OUTPUTS "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_OBJECT}")
            endif()
        endforeach()
        add_custom_target(motion_enhancer_shaders ALL DEPENDS ${DXC_OUTPUTS})
        add_dependencies(motion_enhancer motion_enhancer_shaders)
        add_custom_command(TARGET motion_enhancer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_BINARY_DIR}/shaders"
                "$<TARGET_FILE_DIR:motion_enhancer>/shaders")
        target_compile_definitions(motion_enhancer PRIVATE MOTION_ENHANCER_DXC=1)
    else()
        message(WARNING "DXC not found; runtime D3DCompile fallback will be used")
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
