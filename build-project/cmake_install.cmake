# Install script for directory: C:/Users/bsher/Documents/GitHub/MotionEnhancer2

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/MotionEnhancer")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}" TYPE EXECUTABLE FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/Debug/motion_enhancer.exe")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}" TYPE EXECUTABLE FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/Release/motion_enhancer.exe")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}" TYPE EXECUTABLE FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/MinSizeRel/motion_enhancer.exe")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}" TYPE EXECUTABLE FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/RelWithDebInfo/motion_enhancer.exe")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/shaders" TYPE FILE FILES
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/LuminanceCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/PyramidCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/MotionSearchCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/InterpolateCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/PresentFrameCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/PackLumaCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/LearnedFeaturesCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/FillFlowCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/UiMaskCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/FrameStatsCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/DisPatchCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/DisDensifyCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/DisToRingCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/OcclusionFillCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/ProjectClearCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/ProjectSplatCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/FlowMedianCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/DepthInputCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/DepthFinishCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/DepthViewCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/DepthDilateCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/AiFlowInputCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/AiFlowToRingCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/UiDetectInputCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/UiDebugViewCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/FfxFlowConvertCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/FfxRefineInitCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/FfxRefineIterateCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/FfxRefineToRingCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/FfxRefineConsistCS.cso"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/shaders/FfxRepairCS.cso"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}" TYPE FILE FILES
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/LICENSE"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/THIRD_PARTY_NOTICES.md"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/packaging/INSTALLATION.md"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}" TYPE PROGRAM FILES
    "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/x64/Microsoft.VC145.CRT/msvcp140.dll"
    "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/x64/Microsoft.VC145.CRT/msvcp140_1.dll"
    "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/x64/Microsoft.VC145.CRT/msvcp140_2.dll"
    "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/x64/Microsoft.VC145.CRT/msvcp140_atomic_wait.dll"
    "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/x64/Microsoft.VC145.CRT/msvcp140_codecvt_ids.dll"
    "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/x64/Microsoft.VC145.CRT/vcruntime140_1.dll"
    "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/x64/Microsoft.VC145.CRT/vcruntime140.dll"
    "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/x64/Microsoft.VC145.CRT/concrt140.dll"
    "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/x64/Microsoft.VC145.OPENMP/vcomp140.dll"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}" TYPE DIRECTORY FILES "")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}" TYPE FILE FILES
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/third_party/ai-depth/bin/onnxruntime.dll"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/third_party/ai-depth/bin/onnxruntime_providers_shared.dll"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/third_party/ai-depth/bin/DirectML.dll"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/models" TYPE FILE FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/third_party/ai-depth/models/midas_v21_small_256_fp16.onnx")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/licenses" TYPE FILE FILES
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/third_party/ai-depth/ORT_LICENSE"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/third_party/ai-depth/ORT_ThirdPartyNotices.txt"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/third_party/ai-depth/DirectML_LICENSE.txt"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/third_party/ai-depth/DirectML_LICENSE-CODE.txt"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/third_party/ai-depth/DirectML_ThirdPartyNotices.txt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/licenses" TYPE FILE RENAME "FidelityFX-LICENSE.txt" FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/third_party/FidelityFX-SDK-v1.1.4/LICENSE.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/licenses" TYPE FILE RENAME "UI-Matrix-ncnn.txt" FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/third_party/ui-matrix/ncnn/LICENSE.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/licenses" TYPE FILE RENAME "UI-Matrix-glslang.txt" FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/third_party/ui-matrix/ncnn/glslang/LICENSE.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/licenses" TYPE DIRECTORY FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/packaging/licenses/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/models" TYPE FILE RENAME "ai_flow.onnx" FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/models/ai_flow.onnx")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/models" TYPE FILE RENAME "ai_flow.json" FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/models/ai_flow.json")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/models" TYPE FILE FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/neural_model.bin")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}" TYPE FILE FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/packaging/Setup-AITrainer.cmd")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tools/ai-train" TYPE FILE FILES
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/tools/ai-train/app.py"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/tools/ai-train/README.md"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/tools/ai-train/requirements.txt"
    "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/tools/ai-train/Setup-AITrainer.ps1"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tools/ai-train" TYPE DIRECTORY FILES "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/tools/ai-train/mefi" FILES_MATCHING REGEX "/[^/]*\\.py$" REGEX "/\\_\\_pycache\\_\\_$" EXCLUDE)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/bsher/Documents/GitHub/MotionEnhancer2/build-project/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
