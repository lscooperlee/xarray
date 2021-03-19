cmake_minimum_required(VERSION 3.15)

# Set the project name to your project name, my project isn't very descriptive
project(cpp_starter CXX)

add_subdirectory(tests)

add_executable(main src/main.cpp)

target_compile_features(main PUBLIC cxx_std_20)
set_target_properties(main PROPERTIES CXX_EXTENSIONS OFF)
