cmake_minimum_required(VERSION 3.18)
project(fidumap LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/eigen" CACHE PATH "" FORCE)
set(EIGEN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/eigen" CACHE PATH "" FORCE)

# OpenGV still declares compatibility with CMake 2.8. CMake 4 removed that
# compatibility mode, so give only the vendored subdirectory the supported
# policy floor it needs to configure.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_SKIP_INSTALL_RULES TRUE)
add_subdirectory(vendor/opengv EXCLUDE_FROM_ALL)
set(CMAKE_SKIP_INSTALL_RULES FALSE)

# FiduMap links only OpenGV's absolute-pose APIs. Avoid compiling the unrelated
# generated relative-pose solvers, which are pathological under MSVC.
get_target_property(_opengv_sources opengv SOURCES)
list(FILTER _opengv_sources EXCLUDE REGEX "^src/(relative_pose|triangulation|point_cloud)/")
list(FILTER _opengv_sources EXCLUDE REGEX "^src/sac_problems/(relative_pose|point_cloud)/")
set_property(TARGET opengv PROPERTY SOURCES "${_opengv_sources}")

if(NOT MSVC)
    get_property(
        _opengv_directory_compile_options
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vendor/opengv"
        PROPERTY COMPILE_OPTIONS
    )
    if(_opengv_directory_compile_options)
        list(FILTER _opengv_directory_compile_options EXCLUDE REGEX "^-m(arch|cpu)=")
        set_property(
            DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vendor/opengv"
            PROPERTY COMPILE_OPTIONS "${_opengv_directory_compile_options}"
        )
    endif()

    get_target_property(_opengv_compile_options opengv COMPILE_OPTIONS)
    if(_opengv_compile_options)
        list(FILTER _opengv_compile_options EXCLUDE REGEX "^-m(arch|cpu)=")
        set_property(TARGET opengv PROPERTY COMPILE_OPTIONS "${_opengv_compile_options}")
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(amd64)")
        set(_fidumap_arch_option -march=x86-64)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm64)|(ARM64)|(aarch64)|(AARCH64)")
        set(_fidumap_arch_option -march=armv8-a)
    endif()
    if(_fidumap_arch_option)
        # OpenGV adds -march=native on x86. Override it with a wheel-safe
        # baseline and use the same Eigen ABI in OpenGV and the extension.
        target_compile_options(opengv PRIVATE "${_fidumap_arch_option}")
    endif()
    target_compile_definitions(
        opengv PUBLIC EIGEN_MAX_ALIGN_BYTES=16 EIGEN_MAX_STATIC_ALIGN_BYTES=16
    )
endif()

if(MSVC)
    set(_opengv_generated_solver_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/vendor/opengv/src/absolute_pose/modules/upnp2.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/vendor/opengv/src/absolute_pose/modules/upnp4.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/vendor/opengv/src/absolute_pose/modules/gp3p/code.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/vendor/opengv/src/absolute_pose/modules/gp3p/init.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/vendor/opengv/src/absolute_pose/modules/gp3p/reductors.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/vendor/opengv/src/absolute_pose/modules/gp3p/spolynomials.cpp"
    )
    set_source_files_properties(
        ${_opengv_generated_solver_sources}
        TARGET_DIRECTORY opengv
        PROPERTIES COMPILE_OPTIONS "/Od"
    )
endif()

pybind11_add_module(_opengv_native NO_EXTRAS src/absolute_pose_solver/native/bindings.cpp)
target_compile_features(_opengv_native PRIVATE cxx_std_14)
target_link_libraries(_opengv_native PRIVATE opengv)

if(_fidumap_arch_option)
    target_compile_options(_opengv_native PRIVATE "${_fidumap_arch_option}")
endif()

if(MSVC)
    target_compile_options(_opengv_native PRIVATE /bigobj)
endif()

install(TARGETS _opengv_native LIBRARY DESTINATION absolute_pose_solver)
