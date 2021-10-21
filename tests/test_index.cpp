
#include <catch2/catch.hpp>

#include "xarray/index.h"
#include "xarray/xarray.h"

using namespace xa;

TEST_CASE("index")
{
    constexpr auto a = Index1D({ 1, 2, 3 });
    static_assert(a.step() == 3);

    constexpr auto b = Index({ 1, 2 }, { 3, 4, 10 });

    static_assert(b.get<0, 0>() == 1);
    static_assert(b.get<1, 2>() == 10);

    Xarray t = { { 1.0, 2.0, 3.0, 4.0, 5.0 }, { 1.1, 2.1, 3.1, 4.1, 5.1 } };

    REQUIRE(isclose(Xarray({ 1.1, 2.1, 3.1, 4.1, 5.1 }), t[Index(1)]));

    auto c1 = t[Index({ 0 }, { 0, 4, 2 })];
    REQUIRE(c1.shape_size == 1);
    REQUIRE(c1.shape == Shape(2));

    REQUIRE(isclose(Xarray({ 1.0, 3.0 }), t[Index({ 0 }, { 0, 4, 2 })]));
    REQUIRE(t[Index(0, 2)] == 3.0);

    auto m1 = t[Index({ -1 }, { 0, 4, 2 })];
    auto c2 = t[Index({ 1 }, { 0, 4, 2 })];
    REQUIRE(isclose(m1, c2));

    Xarray t1d = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    t1d[Index({1, 2})];
}