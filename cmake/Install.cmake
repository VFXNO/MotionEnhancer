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
endif()

# Every optional AI mode's model, so each GUI option works after install.
# A missing file here only leaves that option without its model; the package
# script (tools/Build-Package.ps1) checks the complete set.
set(_models "${CMAKE_SOURCE_DIR}/models")
# Trained temporal UI (GUI "Trained temporal UI", --ai-temporal-ui): the
# integrated model and its DIS companion, exported by the trainer.
foreach(_file ai_flow_ui.onnx ai_flow_ui.json ai_flow_ui.dis.onnx ai_flow_ui.dis.json)
    if(EXISTS "${_models}/${_file}")
        install(FILES "${_models}/${_file}" DESTINATION models)
    endif()
endforeach()
# OmniParser icon detector ("AI: UI regions", DirectML): AGPL-3.0 weights,
# shipped with their licence and provenance.
foreach(_file ui_detect.onnx ui_detect.source.json ui_detect.LICENSE.txt)
    if(EXISTS "${_models}/${_file}")
        install(FILES "${_models}/${_file}" DESTINATION models)
    endif()
endforeach()
if(EXISTS "${_models}/ui_detect.LICENSE.txt")
    install(FILES "${_models}/ui_detect.LICENSE.txt" DESTINATION licenses RENAME OmniParser-icon_detect-AGPL-3.0.txt)
endif()
# Its Vulkan matrix conversion ("Matrix: icons (WMMA)"), only useful when that
# backend is compiled in.
if(MOTION_ENHANCER_ENABLE_UI_MATRIX)
    foreach(_file ui_detect.matrix.param ui_detect.matrix.bin ui_detect.matrix.source.json ui_detect.matrix.LICENSE.txt)
        if(EXISTS "${_models}/${_file}")
            install(FILES "${_models}/${_file}" DESTINATION models)
        endif()
    endforeach()
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
    # A refined export is a pair: the integrated model and the backbone-only
    # companion its JSON names ("dis_model"), which the default AI engine
    # (refinement off) loads. Without the companion the AI engine refuses to
    # start a capture (AiFlowEstimator::selectRefinementModel).
    file(READ "${_model_json}" _model_meta)
    if(_model_meta MATCHES "\"dis_model\"[ \t\r\n]*:[ \t\r\n]*\"([^\"]+)\"")
        set(_dis_model "${_model_dir}/${CMAKE_MATCH_1}")
        get_filename_component(_dis_name "${_dis_model}" NAME_WLE)   # "ai_flow.dis", not "ai_flow"
        set(_dis_json "${_model_dir}/${_dis_name}.json")
        if(NOT EXISTS "${_dis_model}" OR NOT EXISTS "${_dis_json}")
            message(FATAL_ERROR "The AI flow model is a refined pair; its companion is missing: ${_dis_model} (and .json)")
        endif()
        install(FILES "${_dis_model}" DESTINATION models RENAME ai_flow.dis.onnx)
        install(FILES "${_dis_json}" DESTINATION models RENAME ai_flow.dis.json)
    endif()
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
install(FILES tools/ai-train/app.py tools/ai-train/README.md tools/ai-train/UI_TRAINING.md
    tools/ai-train/UI_LABELING.md tools/ai-train/requirements.txt
    tools/ai-train/Setup-AITrainer.ps1 DESTINATION tools/ai-train)
install(DIRECTORY tools/ai-train/mefi DESTINATION tools/ai-train
    FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)

set(CPACK_PACKAGE_NAME MotionEnhancer)
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "MotionEnhancer-${PROJECT_VERSION}-win64")
set(CPACK_GENERATOR ZIP)
include(CPack)
