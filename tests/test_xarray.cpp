#include "xarray/common.h"
#include "xarray/shape.h"
#include "xarray/xarray.h"
#include <iostream>

#include <catch2/catch.hpp>
#include <initializer_list>
#include <vector>

using namespace xa;

TEST_CASE("xarray")
{

    static_assert(XBaseType<Xarray<int, 2>>);

    auto x1 = Xarray({ 1.0, 2.0 });
    auto x2 = Xarray({ 1.0, 2.0 });
    REQUIRE(x1.shape[0] == 2);
    REQUIRE(x1.shape.size() == 1);
    REQUIRE(isclose(x1, x2));

    auto i1 = Xarray({ 1, 2, 3, 4, 5, 6 });
    auto tt1 = i1.reshape(Shape(2, 3)) == Xarray({ { 1, 2, 3 }, { 4, 5, 6 } });
    REQUIRE(all(tt1));

    auto x3 = x1 * 2;
    REQUIRE(isclose(x3, Xarray({ 2.0, 4.0 })));
    REQUIRE(isclose(2 * x1, x3));
    REQUIRE(isclose(x1 * x1, Xarray({ 1.0, 4.0 })));

    auto x4 = Xarray({ { 1.0, 2.0, 3.0 }, { 3.0, 2.0, 1.0 } });
    auto xe = Xarray({ 2.0, 1.0, 0.0 });
    REQUIRE(isclose(xe * x4, Xarray<double, 2>({ { 2, 2, 0 }, { 6, 2, 0 } })));
    REQUIRE(isclose(xe * x4, x4 * xe));

    REQUIRE(isclose(x1 / 2, Xarray({ 0.5, 1.0 })));
    REQUIRE(isclose(2 / x1, Xarray({ 2.0, 1.0 })));
    REQUIRE(isclose(x1 / x1, Xarray({ 1.0, 1.0 })));

    REQUIRE(x1.dot(x2) == 5);
    REQUIRE(x2.dot(x1) == 5);
    auto x32 = Xarray({ { 0.1, 0.2 }, { 0.3, 0.4 }, { 0.5, 0.6 } });
    auto x22 = Xarray({ { 3.1, 0.2 }, { 1.3, 3.4 } });
    REQUIRE(isclose(x32.dot(x22), Xarray({ { 0.57, 0.7 }, { 1.45, 1.42 }, { 2.33, 2.14 } })));

    auto x31 = Xarray({ 1.0, 0.0, 1.0 });
    auto x21 = Xarray({ -1.0, 1.0 });
    REQUIRE(isclose(x32.matmul(x21), Xarray({ 0.1, 0.1, 0.1 })));
    REQUIRE(isclose(x31.matmul(x32), Xarray({ 0.6, 0.8 })));
    REQUIRE(isclose(x32.matmul(x22), Xarray({ { 0.57, 0.7 }, { 1.45, 1.42 }, { 2.33, 2.14 } })));

    REQUIRE(isclose(x22.inv(), Xarray({ { 0.330739, -0.0194553 }, { -0.126459, 0.301556 } })));

    auto xsvd = Xarray({ { 0.29658888, -0.19843645 }, { -0.63928474, -0.62971046 }, { 0.21215595, 0.55278224 } });
    auto [u, w, vt] = svd(xsvd);
    REQUIRE(isclose(u,
        Xarray({ { 0.030069, -0.862362, -0.505398 }, { -0.844925, 0.248218, -0.473804 }, { 0.534039, 0.44127, -0.721168 } })));
    REQUIRE(isclose(w, Xarray({ 1.05511, 0.412167 })));
    REQUIRE(isclose(vt, Xarray({ { 0.627768, 0.7784 }, { -0.7784, 0.627768 } })));

    REQUIRE(isclose(x32.T(), Xarray({ { 0.1, 0.3, 0.5 }, { 0.2, 0.4, 0.6 } })));

    REQUIRE(all(x32 < 10));

    REQUIRE(all(-x32 == Xarray { { -0.1, -0.2 }, { -0.3, -0.4 }, { -0.5, -0.6 } }));

    REQUIRE(all((x1 + x2 - x3) == Xarray<double, 1>({ 0, 0 })));
    REQUIRE(all(x2 / 2 == Xarray({ 0.5, 1.0 })));
    REQUIRE(all(x2 / x3 == Xarray({ 0.5, 0.5 })));

    auto xdet = Xarray<float, 2>({ { 50, 29 }, { 30, 44 } });
    REQUIRE(det(xdet) == 1330);

    x2 -= x1;
    REQUIRE(all(x2 == Xarray({ 0.0, 0.0 })));
    x2 += x1;
    REQUIRE(all(x2 == Xarray({ 1.0, 2.0 })));

    REQUIRE(sum(Xarray({ 0.5, 1.5 })) == 2);

    auto xm1 = Xarray({ { 1, 2, 3 }, { 4, 5, 6 } });

    REQUIRE(all(xm1 - Xarray({ 1, 1, 1 }) == Xarray({ { 0, 1, 2 }, { 3, 4, 5 } })));
    REQUIRE(all(Xarray({ 1, 1, 1 }) - xm1 == Xarray({ { 0, -1, -2 }, { -3, -4, -5 } })));

    REQUIRE(all(xm1 + Xarray({ 1, 1, 1 }) == Xarray({ { 2, 3, 4 }, { 5, 6, 7 } })));
    REQUIRE(all(Xarray({ 1, 1, 1 }) + xm1 == Xarray({ { 2, 3, 4 }, { 5, 6, 7 } })));

    REQUIRE(all(repeat(x1, 2) == Xarray({ 1.0, 2.0, 1.0, 2.0 })));

    auto xm2 = Xarray<double, 2>({ { 1, 2, 3 }, { 3, 2, 1 } });

    REQUIRE(mean(xm2) == 2.0);
    REQUIRE(all(mean(xm2, 0) == Xarray({ 2.0, 2.0, 2.0 })));
    REQUIRE(all(mean(xm2, 1) == Xarray({ 2.0, 2.0 })));

    REQUIRE(all(abs(-xm2) == xm2));

    auto xchar = Xarray<char, 2>({ { 1, 2 }, { 3, 4 } });
    REQUIRE(all(Xarray<int, 2>({ { 1, 2 }, { 3, 4 } }) == xchar.astype<int>()));

    auto xdouble = Xarray<double, 2>({ { 1.1, 2.2 }, { 3.3, 4.4 } });
    REQUIRE(all(Xarray<int, 2>({ { 1, 2 }, { 3, 4 } }) == xdouble.astype<int>()));

    auto xlinsp = linspace(2.0, 3.0, 5);
    auto xlinspf = linspace(2.0, 3.0, 5, false);

    REQUIRE(all(Xarray({ 2.0, 2.25, 2.5, 2.75, 3.0 }) == xlinsp));
    REQUIRE(isclose(Xarray({ 2.0, 2.2, 2.4, 2.6, 2.8 }), xlinspf));

    xdouble[Index(1)] = Xarray<double, 1>({ 8.8, 9.9 });
    REQUIRE(all(Xarray<double, 2>({ { 1.1, 2.2 }, { 8.8, 9.9 } }) == xdouble));

    std::cout << xdouble[Index(1, 1)] << std::endl;
    xdouble[Index(1, 1)] = 0.0;
    std::cout << xdouble[Index(1, 1)] << std::endl;
}
