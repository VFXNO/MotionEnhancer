# Install script for directory: C:/Users/vfxno/Documents/GitHub/Motion enhancer/third_party/FidelityFX-SDK-v1.1.4

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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/opticalflow/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/frameinterpolation/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/fsr3/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/fsr3upscaler/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/fsr2/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/fsr1/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/spd/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/cacao/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/lpm/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/blur/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/vrs/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/cas/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/dof/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/lens/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/parallelsort/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/denoiser/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/sssr/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/brixelizer/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/brixelizergi/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/classifier/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/components/breadcrumbs/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/libs/pix/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/src/backends/dx12/cmake_install.cmake")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/vfxno/Documents/GitHub/Motion enhancer/build-vendored-ffx/fidelityfx-sdk/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
