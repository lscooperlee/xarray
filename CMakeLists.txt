cmake_minimum_required(VERSION 3.15)

# Set the project name to your project name, my project isn't very descriptive
project(xarray CXX)

add_subdirectory(tests)

file(GLOB sources CONFIGURE_DEPENDS "src/*.cpp")
add_executable(main ${sources})

target_compile_features(main PUBLIC cxx_std_20)
target_compile_options(main PUBLIC -Wall -Wextra -pedantic)
set_target_properties(main PROPERTIES CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED core imgcodecs imgproc calib3d)
target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS})

