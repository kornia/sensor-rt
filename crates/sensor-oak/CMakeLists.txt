cmake_minimum_required(VERSION 3.16)
project(oak_bridge CXX)

# depthai-core built into vendor/depthai by `pixi run depthai-build`. CMAKE_PREFIX_PATH
# is passed by build.rs so find_package resolves the config + all transitive deps.
find_package(depthai CONFIG REQUIRED)

add_library(oak_bridge STATIC oak_bridge.cpp)
target_compile_features(oak_bridge PUBLIC cxx_std_17)
target_link_libraries(oak_bridge PUBLIC depthai::core)

install(TARGETS oak_bridge ARCHIVE DESTINATION lib)
