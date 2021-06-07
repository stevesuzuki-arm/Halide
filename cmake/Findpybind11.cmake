set(pybind11_VERSION "${${CMAKE_FIND_PACKAGE_NAME}_FIND_VERSION}")

find_package(pybind11 "${pybind11_VERSION}" QUIET
             HINTS ${Python_SITELIB} ${Python3_SITELIB})

if (NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
                         GIT_REPOSITORY https://github.com/pybind/pybind11.git
                         GIT_TAG v${pybind11_VERSION})
    FetchContent_MakeAvailable(pybind11)

    # Make FPHSA think a config package was found
    set(pybind11_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(pybind11 CONFIG_MODE)
