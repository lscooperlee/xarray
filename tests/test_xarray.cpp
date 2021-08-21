#include "shape.h"
#include "xarray.h"
#include <iostream>

#include <catch2/catch.hpp>
#include <vector>
#include <initializer_list>

using namespace xa;

TEST_CASE("xarray", "[xarray]")
{
    auto x1 = Xarray({ 1.0, 2.0 });
    auto x2 = Xarray({ 1.0, 2.0 });
    REQUIRE(x1.shape[0] == 2);
    REQUIRE(x1.shape.size() == 1);
    REQUIRE(isclose(x1, x2));

    auto x3 = x1 * 2;
    REQUIRE(isclose(x3, Xarray({ 2.2, 4.0 })));
    // auto x3 = 2 * x1;
    // REQUIRE(all(x2 == x3));
    // auto x4 = x1 * x1;

    // auto x32 = Xarray({ { 0.1, 0.2 }, { 0.3, 0.4 }, { 0.5, 0.6 } });
    // auto x22 = Xarray({ { 3.1, 0.2 }, { 1.3, 3.4 } });
    // auto newx32 = x32 * x22;
    // REQUIRE(isclose(newx32, Xarray({ { 0.57, 0.7 }, { 1.45, 1.42 }, { 2.33, 2.14 } })));

    // REQUIRE(isclose(x22.inv(), Xarray({ { 0.330739, -0.0194553 }, { -0.126459, 0.301556 } })));

    // REQUIRE(x1.dot(x2) == 10.42);
    // REQUIRE(all(x32.T() == Xarray({ { 0.1, 0.3, 0.5 }, { 0.2, 0.4, 0.6 } })));

    // REQUIRE(all((x1 + x2 - x3) == Xarray({ 1.1, 2.0 })));
    // REQUIRE(all(x2 / 2 == Xarray({ 1.1, 2.0 })));
    // REQUIRE(all(x2 / x3 == Xarray({ 1.0, 1.0 })));

    // x2 -= x1;
    // REQUIRE(all(x2 == Xarray({ 1.1, 2.0 })));
    // x2 += x1;
    // REQUIRE(all(x2 == Xarray({ 2.2, 4.0 })));

    // auto xdet = Xarray<float>({ { 50, 29 }, { 30, 44 } });
    // REQUIRE(det(xdet) == 1330);
}
