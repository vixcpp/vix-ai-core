cmake_minimum_required(VERSION 3.20)

project(vix-ai-core
  VERSION 0.1.0
  DESCRIPTION "Vix AI Core: tiny device/tensor/engine primitives"
  LANGUAGES CXX)

# Options
option(VIX_AI_CORE_BUILD_TESTS "Build vix-ai-core tests" ON)

if (VIX_AI_CORE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
option(VIX_AI_CORE_BUILD_TESTS "Build vix-ai-core tests" OFF)
option(VIX_AI_CORE_INSTALL     "Install vix-ai-core targets" ON)
option(VIX_AI_CORE_WARNINGS    "Enable extra warnings" ON)

# Library
add_library(vix_ai_core
  src/Version.cpp
  src/Tensor.cpp
  src/Device.cpp
  src/Engine.cpp
)

add_library(Vix::AI::core ALIAS vix_ai_core)

# Public headers
target_include_directories(vix_ai_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# C++ std & defs
target_compile_features(vix_ai_core PUBLIC cxx_std_20)
target_compile_definitions(vix_ai_core PUBLIC VIX_AI_CORE_VERSION="${PROJECT_VERSION}")

# Warnings (optional but handy)
if (VIX_AI_CORE_WARNINGS)
  if (MSVC)
    target_compile_options(vix_ai_core PRIVATE /W4 /permissive-)
  else()
    target_compile_options(vix_ai_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
endif()

# Good defaults
set_target_properties(vix_ai_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  VERSION   ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  EXPORT_NAME core
)

# Install / package config
if (VIX_AI_CORE_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  install(TARGETS vix_ai_core
    EXPORT vix_ai_coreTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  # Export targets for find_package()
  install(EXPORT vix_ai_coreTargets
    FILE vix_ai_coreTargets.cmake
    NAMESPACE Vix::AI::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai_core
  )

  # Config + version files
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/vix_ai_coreConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  # Simple, relocatable config that just includes the targets file
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/vix_ai_coreConfig.cmake
"include(\"\${CMAKE_CURRENT_LIST_DIR}/vix_ai_coreTargets.cmake\")\n")

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/vix_ai_coreConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/vix_ai_coreConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai_core
  )
endif()

# (Optionnel) Tests
if (VIX_AI_CORE_BUILD_TESTS)
  enable_testing()
  add_executable(vix_ai_core_smoke tests/smoke.cpp)
  target_link_libraries(vix_ai_core_smoke PRIVATE vix_ai_core)
  add_test(NAME smoke COMMAND vix_ai_core_smoke)
endif()
