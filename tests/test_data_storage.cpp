
#include <array>
#include <catch2/catch.hpp>
#include <vector>

#include "xarray/data_storage.h"
#include "xarray/index.h"
#include "xarray/shape.h"

using namespace xa;

TEST_CASE("data_storage")
{
    auto ds = DataStorage(std::vector({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 }));

    Index i0(0);
    Shape s0(2, 6);
    auto m0 = ds.get_mask(i0, s0);
    REQUIRE(m0 == std::array<std::array<int, 3>, s0.size()>({ { { 0, 1, 1 }, { 0, 6, 1 } } }));

    auto st0 = ds.get_stride(s0);
    REQUIRE(st0 == std::array<int, st0.size()>({ { 6, 1 } }));
    REQUIRE(ds.get_shape(i0, s0) == Shape(6));

    ds.copy(i0, s0);

    Index i1({ -1 }, { -1, -13, -2 });
    Shape s1(2, 6);

    auto m1 = ds.get_mask(i1, s1);
    REQUIRE(m1 == std::array<std::array<int, 3>, s1.size()>({ { { 1, 2, 1 }, { 5, -1, -2 } } }));

    auto st1 = ds.get_stride(s1);
    REQUIRE(st1 == std::array<int, st1.size()>({ { 6, 1 } }));

    REQUIRE(ds.get_shape(i1, s1) == Shape(3));

    ds.copy(i1, s1);

    Index i2({ 1 }, { -1, -13, -1 });
    Shape s2(2, 3, 2);

    auto m2 = ds.get_mask(i2, s2);
    REQUIRE(m2 == std::array<std::array<int, 3>, s2.size()>({ { { 1, 2, 1 }, { 2, -1, -1 }, { 0, 2, 1 } } }));

    auto st2 = ds.get_stride(s2);
    REQUIRE(st2 == std::array<int, st2.size()>({ { 6, 2, 1 } }));
    REQUIRE(ds.get_shape(i2, s2) == Shape(3, 2));

    ds.copy(i2, s2);

    Index i3({ 0, 3 }, { 0, 3 });
    Shape s3(3, 4);

    auto m3 = ds.get_mask(i3, s3);
    REQUIRE(m3 == std::array<std::array<int, 3>, s3.size()>({ { { 0, 3, 1 }, { 0, 3, 1 } } }));

    auto st3 = ds.get_stride(s3);
    REQUIRE(st3 == std::array<int, st3.size()>({ { 4, 1 } }));
    REQUIRE(ds.get_shape(i3, s3) == Shape(3, 3));

    // auto mm = ds.copy(i2, s2);
    auto ds1 = DataStorage<double, 1, 2>(std::vector({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 }));

    // ds = ds1;
}