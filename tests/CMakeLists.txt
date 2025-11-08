cmake_minimum_required(VERSION 3.20)

#[[
Vix.AI Core — Unit-like tests (header-only, assert-based)
Each test is a tiny executable registered with CTest.
]]

# Smoke test ---------------------------------------------------------------
add_executable(vix_ai_core_test_smoke smoke.cpp)
target_link_libraries(vix_ai_core_test_smoke PRIVATE vix_ai_core)
target_compile_features(vix_ai_core_test_smoke PRIVATE cxx_std_20)
add_test(NAME core_smoke COMMAND vix_ai_core_test_smoke)

# Tensor tests -------------------------------------------------------------
add_executable(vix_ai_core_test_tensor tensor_tests.cpp)
target_link_libraries(vix_ai_core_test_tensor PRIVATE vix_ai_core)
target_compile_features(vix_ai_core_test_tensor PRIVATE cxx_std_20)
add_test(NAME core_tensor COMMAND vix_ai_core_test_tensor)

# Device tests -------------------------------------------------------------
add_executable(vix_ai_core_test_device device_tests.cpp)
target_link_libraries(vix_ai_core_test_device PRIVATE vix_ai_core)
target_compile_features(vix_ai_core_test_device PRIVATE cxx_std_20)
add_test(NAME core_device COMMAND vix_ai_core_test_device)

# Engine tests -------------------------------------------------------------
add_executable(vix_ai_core_test_engine engine_tests.cpp)
target_link_libraries(vix_ai_core_test_engine PRIVATE vix_ai_core)
target_compile_features(vix_ai_core_test_engine PRIVATE cxx_std_20)
add_test(NAME core_engine COMMAND vix_ai_core_test_engine)
