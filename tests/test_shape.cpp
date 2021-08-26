
#include "xarray/shape.h"
#include "xarray/xarray.h"
#include <iostream>

#include <catch2/catch.hpp>
#include <vector>

using namespace xa;

TEST_CASE("shape")
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

TEST_CASE("_shape")
{
    _Shape<5, 2, 1> s1;
    _Shape<3, 1> s2;
    _Shape<4> s3;
    _Shape<> s4;

    REQUIRE(s1.size() == 3);
    REQUIRE(s2.size() == 2);
    REQUIRE(s3.size() == 1);
    REQUIRE(s4.size() == 0);

    // std::cout << s1 << std::endl;
}
