# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

cmake_minimum_required(VERSION 3.25)
project(bp_pipeline_flat_folder VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Hide symbols by default (including statically bundled Abseil) so the Python
# extension module only exports its pybind11 entry point and cannot collide with
# Abseil symbols in other loaded extensions.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

option(
  FAST_FLAT_FOLDER_BUILD_PYTHON
  "Build the pybind11 _fast_flat_folder Python extension"
  ON
)
set(_ff_default_install_cpp OFF)
if(PROJECT_IS_TOP_LEVEL AND NOT SKBUILD)
  set(_ff_default_install_cpp ON)
endif()
option(
  FAST_FLAT_FOLDER_INSTALL_CPP
  "Install C++ static library, headers, and CMake package config"
  ${_ff_default_install_cpp}
)
set(
  FAST_FLAT_FOLDER_SANITIZERS
  ""
  CACHE STRING
  "Semicolon-separated sanitizers to enable (e.g. address;undefined or thread)"
)
option(
  BP_USE_SYSTEM_ABSL
  "Use pre-installed system Abseil instead of FetchContent"
  OFF
)

# Minimum Abseil LTS version required for the reference `absl::MutexLock`
# constructor introduced in Abseil 20250814.
set(BP_MIN_ABSL_VERSION "20250814")
set(
  BP_ABSEIL_URL
  "https://github.com/abseil/abseil-cpp/archive/refs/tags/20250814.1.tar.gz"
  CACHE STRING "URL or local path to abseil-cpp source archive"
)
set(
  BP_ABSEIL_URL_HASH
  "SHA256=1692f77d1739bacf3f94337188b78583cf09bab7e420d2dc6c5605a4f86785a1"
  CACHE STRING "Expected hash of BP_ABSEIL_URL"
)

if(FAST_FLAT_FOLDER_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
endif()

# Compute optional sanitizer compile/link flags and apply them before
# FetchContent_MakeAvailable(abseil-cpp) so bundled Abseil (e.g., absl::Mutex)
# is instrumented consistently under ASan/UBSan/TSan.
set(_ff_sanitizer_compile_opts "")
set(_ff_sanitizer_link_opts "")
if(FAST_FLAT_FOLDER_SANITIZERS)
  if(MSVC)
    if("address" IN_LIST FAST_FLAT_FOLDER_SANITIZERS)
      list(APPEND _ff_sanitizer_compile_opts "/fsanitize=address")
    else()
      message(
        FATAL_ERROR
        "Unsupported FAST_FLAT_FOLDER_SANITIZERS='${FAST_FLAT_FOLDER_SANITIZERS}' on MSVC (only 'address' is supported)."
      )
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    string(REPLACE ";" "," _san_csv "${FAST_FLAT_FOLDER_SANITIZERS}")
    list(
      APPEND
      _ff_sanitizer_compile_opts
      "-fsanitize=${_san_csv}"
      "-fno-omit-frame-pointer"
    )
    list(APPEND _ff_sanitizer_link_opts "-fsanitize=${_san_csv}")
  else()
    message(
      FATAL_ERROR
      "FAST_FLAT_FOLDER_SANITIZERS is not supported for compiler '${CMAKE_CXX_COMPILER_ID}'."
    )
  endif()
  add_compile_options(${_ff_sanitizer_compile_opts})
  add_link_options(${_ff_sanitizer_link_opts})
endif()

if(BP_USE_SYSTEM_ABSL)
  # Abseil's CMake package version file accepts only an exact LTS match, so
  # `find_package(absl <version>)` would reject every newer LTS release. Find
  # any installed Abseil and enforce the minimum LTS here instead.
  find_package(absl CONFIG REQUIRED)
  if(NOT absl_VERSION)
    message(
      WARNING
      "System Abseil at ${absl_DIR} does not report a version; an LTS release >= ${BP_MIN_ABSL_VERSION} is required."
    )
  elseif(absl_VERSION VERSION_LESS BP_MIN_ABSL_VERSION)
    message(
      FATAL_ERROR
      "System Abseil ${absl_VERSION} at ${absl_DIR} is older than the required LTS ${BP_MIN_ABSL_VERSION}; upgrade it or configure with -DBP_USE_SYSTEM_ABSL=OFF."
    )
  else()
    message(STATUS "Using system Abseil ${absl_VERSION} from ${absl_DIR}")
  endif()
else()
  include(FetchContent)
  set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
  if(FAST_FLAT_FOLDER_INSTALL_CPP)
    set(ABSL_ENABLE_INSTALL ON CACHE BOOL "" FORCE)
  else()
    set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  endif()
  set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(_ff_absl_fetch_args
    URL "${BP_ABSEIL_URL}"
    URL_HASH "${BP_ABSEIL_URL_HASH}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    INACTIVITY_TIMEOUT 30
    TIMEOUT 120
  )
  if(NOT FAST_FLAT_FOLDER_INSTALL_CPP AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.28")
    list(APPEND _ff_absl_fetch_args EXCLUDE_FROM_ALL)
  endif()
  FetchContent_Declare(abseil-cpp ${_ff_absl_fetch_args})
  message(
    STATUS
    "Fetching abseil-cpp from ${BP_ABSEIL_URL}. "
    "For offline or firewalled builds, set "
    "CMAKE_ARGS=\"-DBP_USE_SYSTEM_ABSL=ON\" (requires system Abseil >= "
    "${BP_MIN_ABSL_VERSION}) or "
    "CMAKE_ARGS=\"-DBP_ABSEIL_URL=file:///path/to/abseil-cpp.tar.gz\"."
  )
  FetchContent_MakeAvailable(abseil-cpp)
endif()

set(FF_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/bp_pipeline_flat_folder")

# Shared compiler warning flags for first-party targets only (kept off bundled
# third-party dependencies such as Abseil).
add_library(fast_flat_folder_build_options INTERFACE)
target_compile_options(
  fast_flat_folder_build_options
  INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wall;-Wextra>
)

set(FF_PUBLIC_HEADERS
  "${FF_SRC_DIR}/fast_flat_folder_types.h"
  "${FF_SRC_DIR}/fast_flat_folder_geometry.h"
  "${FF_SRC_DIR}/fast_flat_folder_core.h"
)

# Python-free static C++20 core library (geometry + solver engine).
add_library(fast_flat_folder_core STATIC
  "${FF_SRC_DIR}/fast_flat_folder_geometry.cc"
  "${FF_SRC_DIR}/fast_flat_folder_core.cc"
)
add_library(
  bp_pipeline_flat_folder::fast_flat_folder_core
  ALIAS fast_flat_folder_core
)
target_compile_features(fast_flat_folder_core PUBLIC cxx_std_20)
target_include_directories(fast_flat_folder_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${FF_SRC_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/bp_pipeline_flat_folder>
)
target_link_libraries(fast_flat_folder_core
  PUBLIC
    absl::status
    absl::statusor
    absl::strings
    absl::span
  PRIVATE
    $<BUILD_INTERFACE:fast_flat_folder_build_options>
    absl::absl_check
    absl::algorithm_container
    absl::core_headers
    absl::flat_hash_map
    absl::flat_hash_set
    absl::int128
    absl::str_format
    absl::synchronization
    absl::time
)

if(FAST_FLAT_FOLDER_INSTALL_CPP)
  include(CMakePackageConfigHelpers)

  set(_ff_cmake_install_dir
    "${CMAKE_INSTALL_LIBDIR}/cmake/bp_pipeline_flat_folder"
  )

  install(
    TARGETS fast_flat_folder_core
    EXPORT bp_pipeline_flat_folderTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  )
  install(
    FILES ${FF_PUBLIC_HEADERS}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/bp_pipeline_flat_folder"
  )
  install(
    EXPORT bp_pipeline_flat_folderTargets
    FILE bp_pipeline_flat_folderTargets.cmake
    NAMESPACE bp_pipeline_flat_folder::
    DESTINATION "${_ff_cmake_install_dir}"
  )

  if(BP_USE_SYSTEM_ABSL)
    set(_ff_installed_bundled_absl OFF)
  else()
    set(_ff_installed_bundled_absl ON)
  endif()
  set(_ff_config_in
    "${CMAKE_CURRENT_BINARY_DIR}/bp_pipeline_flat_folderConfig.cmake.in"
  )
  file(WRITE "${_ff_config_in}"
    "@PACKAGE_INIT@\n"
    "\n"
    "include(CMakeFindDependencyMacro)\n"
    "if(@_ff_installed_bundled_absl@ AND NOT DEFINED absl_DIR AND EXISTS \"\${CMAKE_CURRENT_LIST_DIR}/../absl/abslConfig.cmake\")\n"
    "  find_dependency(absl CONFIG PATHS \"\${CMAKE_CURRENT_LIST_DIR}/../absl\" NO_DEFAULT_PATH)\n"
    "else()\n"
    "  find_dependency(absl CONFIG)\n"
    "endif()\n"
    "if(absl_VERSION AND absl_VERSION VERSION_LESS \"@BP_MIN_ABSL_VERSION@\")\n"
    "  set(bp_pipeline_flat_folder_FOUND FALSE)\n"
    "  set(\n"
    "    bp_pipeline_flat_folder_NOT_FOUND_MESSAGE\n"
    "    \"bp_pipeline_flat_folder requires Abseil >= @BP_MIN_ABSL_VERSION@, found \${absl_VERSION} at \${absl_DIR}.\"\n"
    "  )\n"
    "  return()\n"
    "endif()\n"
    "\n"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/bp_pipeline_flat_folderTargets.cmake\")\n"
    "check_required_components(bp_pipeline_flat_folder)\n"
  )
  configure_package_config_file(
    "${_ff_config_in}"
    "${CMAKE_CURRENT_BINARY_DIR}/bp_pipeline_flat_folderConfig.cmake"
    INSTALL_DESTINATION "${_ff_cmake_install_dir}"
    NO_SET_AND_CHECK_MACRO
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/bp_pipeline_flat_folderConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
  )
  install(
    FILES
      "${CMAKE_CURRENT_BINARY_DIR}/bp_pipeline_flat_folderConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/bp_pipeline_flat_folderConfigVersion.cmake"
    DESTINATION "${_ff_cmake_install_dir}"
  )
endif()

if(FAST_FLAT_FOLDER_BUILD_PYTHON)
  pybind11_add_module(_fast_flat_folder MODULE
    "${FF_SRC_DIR}/_fast_flat_folder.cc"
  )
  target_link_libraries(_fast_flat_folder PRIVATE
    fast_flat_folder_core
    fast_flat_folder_build_options
    absl::statusor
    absl::strings
  )
  install(TARGETS _fast_flat_folder DESTINATION bp_pipeline_flat_folder)
endif()
