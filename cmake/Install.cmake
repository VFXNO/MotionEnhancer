# Runtime-only layout: no SDK headers, import libraries or build intermediates.
install(TARGETS motion_enhancer RUNTIME DESTINATION .)
install(FILES ${DXC_OUTPUTS} DESTINATION shaders)
install(FILES LICENSE THIRD_PARTY_NOTICES.md packaging/INSTALLATION.md DESTINATION .)

if(MSVC)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION .)
    set(CMAKE_INSTALL_OPENMP_LIBRARIES TRUE)
    include(InstallRequiredSystemLibraries)
endif()

if(EXISTS "${MOTION_ENHANCER_AI_DEPTH_DIR}/lib/onnxruntime.lib")
    install(FILES
        "${MOTION_ENHANCER_AI_DEPTH_DIR}/bin/onnxruntime.dll"
        "${MOTION_ENHANCER_AI_DEPTH_DIR}/bin/onnxruntime_providers_shared.dll"
        "${MOTION_ENHANCER_AI_DEPTH_DIR}/bin/DirectML.dll" DESTINATION .)
    install(FILES "${MOTION_ENHANCER_AI_DEPTH_DIR}/models/midas_v21_small_256_fp16.onnx" DESTINATION models)
    install(FILES
        "${MOTION_ENHANCER_AI_DEPTH_DIR}/ORT_LICENSE"
        "${MOTION_ENHANCER_AI_DEPTH_DIR}/ORT_ThirdPartyNotices.txt"
        "${MOTION_ENHANCER_AI_DEPTH_DIR}/DirectML_LICENSE.txt"
        "${MOTION_ENHANCER_AI_DEPTH_DIR}/DirectML_LICENSE-CODE.txt"
        "${MOTION_ENHANCER_AI_DEPTH_DIR}/DirectML_ThirdPartyNotices.txt" DESTINATION licenses)
endif()
if(MOTION_ENHANCER_ENABLE_FFX_OPTICAL_FLOW)
    install(FILES "${FIDELITYFX_SDK_DIR}/LICENSE.txt" DESTINATION licenses RENAME FidelityFX-LICENSE.txt)
endif()
if(MOTION_ENHANCER_ENABLE_UI_MATRIX)
    install(FILES "${UI_MATRIX_SOURCE}/LICENSE.txt" DESTINATION licenses RENAME UI-Matrix-ncnn.txt)
    install(FILES "${UI_MATRIX_SOURCE}/glslang/LICENSE.txt" DESTINATION licenses RENAME UI-Matrix-glslang.txt)
    # UI detector weights remain an optional, separately licensed download.
    # Do not package matrix or ONNX UI weights implicitly.
endif()
install(DIRECTORY packaging/licenses/ DESTINATION licenses)

set(MOTION_ENHANCER_AI_FLOW_MODEL "${CMAKE_SOURCE_DIR}/models/ai_flow.onnx"
    CACHE FILEPATH "AI flow ONNX model to stage and install")
option(MOTION_ENHANCER_STAGE_AI_FLOW_MODEL
       "Explicitly overwrite the build runtime's AI flow model from the packaging source" OFF)
if(EXISTS "${MOTION_ENHANCER_AI_FLOW_MODEL}")
    get_filename_component(_model_dir "${MOTION_ENHANCER_AI_FLOW_MODEL}" DIRECTORY)
    get_filename_component(_model_name "${MOTION_ENHANCER_AI_FLOW_MODEL}" NAME_WE)
    set(_model_json "${_model_dir}/${_model_name}.json")
    if(NOT EXISTS "${_model_json}")
        message(FATAL_ERROR "AI flow model needs its matching resolution metadata: ${_model_json}")
    endif()
    install(FILES "${MOTION_ENHANCER_AI_FLOW_MODEL}" DESTINATION models RENAME ai_flow.onnx)
    install(FILES "${_model_json}" DESTINATION models RENAME ai_flow.json)
    # A local trained/activated model is authoritative. An ordinary C++ rebuild
    # must not replace it with the repository's older packaging model.
    if(MOTION_ENHANCER_STAGE_AI_FLOW_MODEL)
      add_custom_command(TARGET motion_enhancer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:motion_enhancer>/models"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MOTION_ENHANCER_AI_FLOW_MODEL}"
            "$<TARGET_FILE_DIR:motion_enhancer>/models/ai_flow.onnx"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_model_json}"
            "$<TARGET_FILE_DIR:motion_enhancer>/models/ai_flow.json")
    endif()
endif()
install(FILES neural_model.bin DESTINATION models)
install(FILES packaging/Setup-AITrainer.cmd DESTINATION .)
install(FILES tools/ai-train/app.py tools/ai-train/README.md tools/ai-train/requirements.txt
    tools/ai-train/Setup-AITrainer.ps1 DESTINATION tools/ai-train)
install(DIRECTORY tools/ai-train/mefi DESTINATION tools/ai-train
    FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)

set(CPACK_PACKAGE_NAME MotionEnhancer)
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "MotionEnhancer-${PROJECT_VERSION}-win64")
set(CPACK_GENERATOR ZIP)
include(CPack)
