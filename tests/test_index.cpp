
#include <catch2/catch.hpp>

#include "index.h"
#include "xarray.h"

using namespace xa;

TEST_CASE("index")
{
    constexpr auto a = Index1D({ 1, 2, 3, 4 });
    constexpr auto b = Index({ 1, 2 }, { 3, 4, 10 });

    static_assert(a.get<0>() == 1);
    static_assert(a.get<1>() == 2);
    static_assert(a.get<2>() == 3);
    static_assert(a.get<3>() == 4);

    static_assert(b.start<0>() == 1);
    static_assert(b.start<1>() == 3);

    static_assert(b.stop<0>() == 2);
    static_assert(b.stop<1>() == 4);

    static_assert(b.step<0>() == 1);
    static_assert(b.step<1>() == 10);

    Xarray t = { { 1.0, 2.0, 3.0, 4.0, 5.0 }, { 1.1, 2.1, 3.1, 4.1, 5.1 } };
    auto c1 = t[Index({ 0 }, { 0, 4, 2 })];
    REQUIRE(c1.shape_size == 1);
    REQUIRE(c1.shape == Shape(2));

    REQUIRE(isclose(Xarray({ 1.0, 3.0 }), t[Index({ 0 }, { 0, 4, 2 })]));
    REQUIRE(t[Index({ 0 }, { 2 })] == 3.0);
}