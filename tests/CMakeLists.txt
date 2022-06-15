Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.9)

FetchContent_MakeAvailable(Catch2)

file(GLOB sources CONFIGURE_DEPENDS "*.cpp")

add_executable(unit_test ${sources})
target_compile_features(unit_test PUBLIC cxx_std_20)
target_compile_options(unit_test PUBLIC -Wall -Wextra -pedantic -Wno-self-assign)

target_link_libraries(unit_test PRIVATE Catch2::Catch2)
target_include_directories(unit_test PRIVATE "../src/")

find_package(OpenCV REQUIRED core imgcodecs imgproc calib3d)
target_include_directories(unit_test PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(unit_test PRIVATE ${OpenCV_LIBS})

