Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.1)

FetchContent_MakeAvailable(Catch2)

file(GLOB sources CONFIGURE_DEPENDS "*.cpp")

add_executable(unit_test ${sources})

target_link_libraries(unit_test PRIVATE Catch2::Catch2)
