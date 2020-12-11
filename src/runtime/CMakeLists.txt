if (NOT CMAKE_GENERATOR MATCHES "Make|Ninja")
    message(STATUS "Notice: ${CMAKE_GENERATOR} does not support depfile dependencies. Incremental builds may fail.")
endif ()

# Keep these lists in alphabetical order.
set(runtime_modules
    aarch64_cpu_features
    alignment_128
    alignment_32
    alignment_64
    allocation_cache
    android_clock
    android_host_cpu_count
    android_io
    arm_cpu_features
    cache
    can_use_target
    cuda
    destructors
    device_interface
    errors
    fake_get_symbol
    fake_thread_pool
    float16_t
    fuchsia_clock
    fuchsia_host_cpu_count
    fuchsia_yield
    gpu_device_selection
    halide_buffer_t
    hexagon_cache_allocator
    hexagon_cpu_features
    hexagon_dma
    hexagon_dma_pool
    hexagon_host
    ios_io
    linux_clock
    linux_host_cpu_count
    linux_yield
    matlab
    metadata
    metal
    metal_objc_arm
    metal_objc_x86
    mips_cpu_features
    module_aot_ref_count
    module_jit_ref_count
    msan
    msan_stubs
    opencl
    opengl
    opengl_egl_context
    opengl_glx_context
    openglcompute
    osx_clock
    osx_get_symbol
    osx_host_cpu_count
    osx_opengl_context
    osx_yield
    posix_allocator
    posix_clock
    posix_error_handler
    posix_get_symbol
    posix_io
    posix_print
    posix_threads
    posix_threads_tsan
    powerpc_cpu_features
    prefetch
    profiler
    profiler_inlined
    pseudostack
    qurt_allocator
    qurt_hvx
    qurt_hvx_vtcm
    qurt_init_fini
    qurt_threads
    qurt_threads_tsan
    qurt_yield
    riscv_cpu_features
    runtime_api
    ssp
    to_string
    trace_helper
    tracing
    wasm_cpu_features
    windows_clock
    windows_cuda
    windows_d3d12compute_x86
    windows_d3d12compute_arm
    windows_get_symbol
    windows_io
    windows_opencl
    windows_profiler
    windows_threads
    windows_threads_tsan
    windows_yield
    write_debug_image
    x86_cpu_features
    )

set(runtime_ll
    aarch64
    arm
    arm_no_neon
    hvx_128
    mips
    posix_math
    powerpc
    ptx_dev
    wasm_math
    win32_math
    x86
    x86_avx
    x86_avx2
    x86_sse41
    )

set(runtime_bc
    compute_20
    compute_30
    compute_35
    )

set(runtime_header_files
    HalideBuffer.h
    HalidePyTorchCudaHelpers.h
    HalidePyTorchHelpers.h
    HalideRuntime.h
    HalideRuntimeCuda.h
    HalideRuntimeD3D12Compute.h
    HalideRuntimeHexagonDma.h
    HalideRuntimeHexagonHost.h
    HalideRuntimeMetal.h
    HalideRuntimeOpenCL.h
    HalideRuntimeOpenGL.h
    HalideRuntimeOpenGLCompute.h
    HalideRuntimeQurt.h
    )

# Need to create an object library for this because CMake
# doesn't support using target_sources on a target declared
# in a different directory ONLY IF that source was created
# by add_custom_command, as is the case in this directory.
add_library(Halide_initmod OBJECT)

# Note: ensure that these flags match the flags in the Makefile.
# Note: this always uses Clang-from-LLVM for compilation, so none
#       of these flags should need conditionalization.
set(runtime_cxx_flags
    -O3
    -ffreestanding
    -fno-blocks
    -fno-exceptions
    -fno-unwind-tables
    -fno-vectorize
    # Note: we don't want static locals to get thread synchronization stuff.
    -fno-threadsafe-statics
    -Wall
    -Wcast-qual
    -Werror
    -Wignored-qualifiers
    -Wno-comment
    -Wno-psabi
    -Wno-unknown-warning-option
    -Wno-unused-function
    -Wsign-compare
    )

foreach (module IN LISTS runtime_modules)
    foreach (bits IN ITEMS 32 64)
        # Target-specific flags
        set(fpic -fpic)
        set(fshort-wchar "") # for the generic windows 64-bit target, we need -fshort-wchar

        # Determine clang target for module
        if (module MATCHES "^windows(_[^_]+)*_(x86|arm)?([^_]+)?$")
            set(fpic "") # Windows doesn't have PIC as a concept.

            set(arch "${CMAKE_MATCH_2}") # either 'x86', 'arm', or ''.
            if ("${arch}|${bits}" STREQUAL "x86|32")
                # win32 uses the stdcall calling convention, which is x86-specific
                set(target "i386-unknown-windows-unknown")
            elseif ("${arch}|${bits}" STREQUAL "x86|64")
                set(target "x86_64-unknown-windows-unknown")
            elseif ("${arch}|${bits}" STREQUAL "arm|32")
                set(target "arm-unknown-windows-unknown")
            elseif ("${arch}|${bits}" STREQUAL "arm|64")
                set(target "aarch64-unknown-windows-unknown")
            elseif ("${arch}|${bits}" STREQUAL "|32")
                # TODO(marcos): generic code won't hold for ARM32... If ARM32 support becomes necessary,
                # all windows-related runtime modules will have to be wrapped in windows_*_arm.cpp files
                # for now, generic Windows 32bit code just assumes x86 (i386)
                set(target "i386-unknown-windows-unknown")
            elseif ("${arch}|${bits}" STREQUAL "|64")
                set(target "le64-unknown-windows-unknown")
                # clang doesn't automatically set this flag even though the ABI is msvc on windows
                set(fshort-wchar -fshort-wchar)
            endif ()
        elseif (bits EQUAL 32)
            # The 'nacl' is a red herring. This is just a generic 32-bit little-endian target.
            set(target "le32-unknown-nacl-unknown")
        else ()
            # generic 64-bit code
            set(target "le64-unknown-unknown-unknown")
        endif ()

        set(module_cpp "${CMAKE_CURRENT_SOURCE_DIR}/${module}.cpp")

        set(runtime_defines -DCOMPILING_HALIDE_RUNTIME -DBITS_${bits})
        set(runtime_defines_debug -g -DDEBUG_RUNTIME ${runtime_defines})

        foreach (debug IN ITEMS "" "_debug")
            set(basename "initmod_${module}_${bits}${debug}")
            set(ll "${basename}.ll")
            set(bc "${basename}.bc")
            set(initmod_cpp "_${basename}.cpp")
            set(symbol "halide_internal_${basename}")

            set(clang_flags ${runtime_cxx_flags} ${fpic} ${fshort-wchar} ${runtime_defines${debug}} -m${bits} -target ${target} -emit-llvm -S)

            # Dep-files are subtle and require clang to run using *just* the right
            # relative paths to the build root, NOT the Halide build root. This is
            # a perfect storm of bad behavior from CMake, Ninja, and Clang.
            file(RELATIVE_PATH ll_path "${CMAKE_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${ll}")
            file(TO_NATIVE_PATH "${ll_path}" ll_path)

            if (CMAKE_GENERATOR MATCHES "Ninja")
                list(APPEND clang_flags -MD -MF "$<SHELL_PATH:${CMAKE_CURRENT_BINARY_DIR}/${basename}.d>")
                set(dep_args DEPFILE "${CMAKE_CURRENT_BINARY_DIR}/${basename}.d")
            elseif (NOT CMAKE_GENERATOR MATCHES "Make")
                set(dep_args "")
            endif ()

            if (Halide_CLANG_TIDY_BUILD)
                # Create a 'fake' entry just so that clang-tidy will see a C++ compilation command that it
                # can use for tidy-checking. (This branch should never be taken by 'normal' compilations.)
                add_library(${basename} STATIC "${module_cpp}")
                target_compile_options(${basename} PRIVATE ${runtime_cxx_flags} ${fpic} ${fshort-wchar} -m${bits} -target ${target})
                target_compile_definitions(${basename} PRIVATE ${runtime_defines})
            else ()
                add_custom_command(OUTPUT "${ll}"
                                   COMMAND clang ${clang_flags} -o "${ll_path}" "$<SHELL_PATH:${module_cpp}>"
                                   DEPENDS "${module_cpp}"
                                   # Note: IMPLICIT_DEPENDS only works for Makefile generators, ${dep_args} handles Ninja.
                                   IMPLICIT_DEPENDS CXX "${module_cpp}"
                                   WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                                   ${dep_args}
                                   VERBATIM)
            endif ()

            add_custom_command(OUTPUT "${bc}"
                               COMMAND llvm-as "${ll}" -o "${bc}"
                               DEPENDS "${ll}"
                               VERBATIM)

            add_custom_command(OUTPUT "${initmod_cpp}"
                               COMMAND binary2cpp ${symbol} < "${bc}" > "${initmod_cpp}"
                               DEPENDS "${bc}" binary2cpp
                               VERBATIM)

            target_sources(Halide_initmod PRIVATE ${initmod_cpp})
        endforeach ()
    endforeach ()
endforeach ()


foreach (module IN LISTS runtime_ll)
    set(ll "${module}.ll")
    set(bc "initmod_${module}.bc")
    set(initmod "_initmod_${module}.cpp")

    add_custom_command(OUTPUT "${bc}"
                       COMMAND llvm-as "$<SHELL_PATH:${CMAKE_CURRENT_SOURCE_DIR}/${ll}>" -o "${bc}"
                       DEPENDS "${ll}"
                       VERBATIM)
    add_custom_command(OUTPUT "${initmod}"
                       COMMAND binary2cpp "halide_internal_initmod_${module}_ll" < "${bc}" > "${initmod}"
                       DEPENDS "${bc}" binary2cpp
                       VERBATIM)
    target_sources(Halide_initmod PRIVATE ${initmod})
endforeach ()

foreach (bc IN LISTS runtime_bc)
    set(initmod "_initmod_ptx_${bc}.cpp")
    set(rt_bc "${CMAKE_CURRENT_SOURCE_DIR}/nvidia_libdevice_bitcode/libdevice.${bc}.10.bc")

    add_custom_command(OUTPUT "${initmod}"
                       COMMAND binary2cpp "halide_internal_initmod_ptx_${bc}_ll" < "$<SHELL_PATH:${rt_bc}>" > "${initmod}"
                       DEPENDS binary2cpp "${rt_bc}"
                       VERBATIM)
    target_sources(Halide_initmod PRIVATE ${initmod})
endforeach ()

add_custom_command(OUTPUT "_initmod_inlined_c.cpp"
                   COMMAND binary2cpp "halide_internal_initmod_inlined_c" < "$<SHELL_PATH:${CMAKE_CURRENT_SOURCE_DIR}/halide_buffer_t.cpp>" > "_initmod_inlined_c.cpp"
                   DEPENDS "halide_buffer_t.cpp" binary2cpp
                   VERBATIM)
target_sources(Halide_initmod PRIVATE "_initmod_inlined_c.cpp")

foreach (header IN LISTS runtime_header_files)
    string(REPLACE "." "_" sym_name "${header}")
    add_custom_command(OUTPUT "_initmod_${sym_name}.cpp"
                       COMMAND binary2cpp "halide_internal_runtime_header_${sym_name}" < "$<SHELL_PATH:${CMAKE_CURRENT_SOURCE_DIR}/${header}>" > "_initmod_${sym_name}.cpp"
                       DEPENDS "${header}" binary2cpp
                       VERBATIM)
    target_sources(Halide_initmod PRIVATE "_initmod_${sym_name}.cpp")

    configure_file(${header} "${Halide_BINARY_DIR}/include/${header}" COPYONLY)
endforeach ()

##
# Target for the runtime
##

add_library(Halide_Runtime INTERFACE)
add_library(Halide::Runtime ALIAS Halide_Runtime)
target_include_directories(Halide_Runtime INTERFACE $<BUILD_INTERFACE:${Halide_BINARY_DIR}/include>)
set_target_properties(Halide_Runtime PROPERTIES EXPORT_NAME Runtime)

