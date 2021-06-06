##
# Third-party dependencies in their own subdirectories
##

add_subdirectory(llvm)
add_subdirectory(wasm) # Needs cache vars set by llvm, do not reorder.

##
# Standard CMake find-module dependencies
##

# Set this to LAST by default since Mono has a framework with the
# libjpeg and libpng headers present -- so CMake finds the headers
# from Mono but the libraries from Homebrew, and hilarity ensues.
set(CMAKE_FIND_FRAMEWORK LAST
    CACHE STRING "Make find_* commands search for macOS Frameworks FIRST/LAST/ONLY/NEVER")

find_package(Threads REQUIRED)
find_package(OpenMP)

# TODO(https://github.com/halide/Halide/issues/5633): verify this is still correct / necessary for OpenGLCompute
find_package(OpenGL)

find_package(JPEG)
find_package(PNG)

foreach (target IN ITEMS JPEG::JPEG OpenGL::GL OpenMP::OpenMP_CXX PNG::PNG Threads::Threads)
    if (TARGET ${target})
        set_property(TARGET ${target} PROPERTY IMPORTED_GLOBAL TRUE)
    endif ()
endforeach ()
