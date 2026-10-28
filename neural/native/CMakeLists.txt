cmake_minimum_required(VERSION 3.18)
project(wime_llama LANGUAGES C CXX)

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(WIME_LLAMA_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/llama.cpp"
    CACHE PATH "Imported llama.cpp source")
add_subdirectory("${WIME_LLAMA_SOURCE}" llama)

add_library(wime_llama SHARED llama_bridge.cpp)
target_compile_features(wime_llama PRIVATE cxx_std_17)
target_link_libraries(wime_llama PRIVATE llama)
set_target_properties(wime_llama PROPERTIES INSTALL_RPATH "$ORIGIN")
