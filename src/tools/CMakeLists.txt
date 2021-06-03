add_executable(build_halide_h build_halide_h.cpp)
add_executable(HalideHost::build_halide_h ALIAS build_halide_h)
target_compile_options(build_halide_h PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/wd4996>)

add_executable(binary2cpp binary2cpp.cpp)
add_executable(HalideHost::binary2cpp ALIAS binary2cpp)
target_compile_options(binary2cpp PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/wd4996>)

# Use of ${CMAKE_BINARY_DIR} here is intentional; exported targets should
# be available by using the top-level build directory as <Pkg>_ROOT.
export(TARGETS build_halide_h binary2cpp
       NAMESPACE HalideHost::
       FILE "${CMAKE_BINARY_DIR}/share/HalideHost/HalideHost-exports.cmake")

file(GENERATE
     OUTPUT "${CMAKE_BINARY_DIR}/share/HalideHost/HalideHostConfig.cmake"
     CONTENT [[
include(${CMAKE_CURRENT_LIST_DIR}/HalideHost-exports.cmake)

add_executable(HalideHost::llvm-as IMPORTED)
set_target_properties(
  HalideHost::llvm-as
  PROPERTIES
  IMPORTED_LOCATION "$<TARGET_FILE:llvm-as>"
)

add_executable(HalideHost::clang IMPORTED)
set_target_properties(
  HalideHost::clang
  PROPERTIES
  IMPORTED_LOCATION "$<TARGET_FILE:clang>"
)
]])
