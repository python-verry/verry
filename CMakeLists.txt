cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15)
        message(FATAL_ERROR "Verry requires clang >= 15")
    endif()

    add_compile_options(-O2 -frounding-math)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
        message(FATAL_ERROR "Verry requires GCC >= 11")
    endif()

    add_compile_options(-O2 -ffp-contract=off -frounding-math)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
        message(FATAL_ERROR "Verry requires MSVC >= 19")
    endif()

    add_compile_options(/O2 /fp:strict)
else()
    message(FATAL_ERROR "Verry does not support compilng with ${CMAKE_CXX_COMPILER_ID}")
endif()

add_subdirectory(src/verry/interval)
