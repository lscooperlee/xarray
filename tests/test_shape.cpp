
#include "shape.h"
#include "xarray.h"
#include <iostream>

#include <catch2/catch.hpp>
#include <vector>

using namespace xa;

TEST_CASE("shape", "[shape]")
{
    Shape s1(5, 2, 1);
    Shape s2(3, 1);
    Shape s3(4);

    REQUIRE(s1.size() == 3);
    REQUIRE(s2.size() == 2);
    REQUIRE(s3.size() == 1);

    Shape ss1({ 5, 2, 1 });
    REQUIRE(ss1.size() == 3);
}
