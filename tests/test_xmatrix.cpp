
#include "xarray/shape.h"
#include "xarray/xmatrix.h"

#include <iostream>
#include <type_traits>

#include <catch2/catch.hpp>

using namespace xa;

TEST_CASE("xmatrix", "[xtest]")
{
    Xmatrix m2 = { { 1.0, 2.0, 3.0 }, { 3.0, 2.0, 1.0 } };

    REQUIRE(isclose(m2, Xmatrix({ { 1.0, 2.0, 3.0 }, { 3.0, 2.0, 1.0 } })));
    REQUIRE(isclose(m2.T(), Xmatrix({ { 1.0, 3.0 }, { 2.0, 2.0 }, { 3.0, 1.0 } })));

    REQUIRE(isclose(m2 * 2, Xmatrix({ { 2.0, 4.0, 6.0 }, { 6.0, 4.0, 2.0 } })));
    REQUIRE(isclose(2 * m2, m2 * 2));

    REQUIRE(isclose(m2 / 2, Xmatrix({ { 0.5, 1.0, 1.5 }, { 1.5, 1.0, 0.5 } })));
    REQUIRE(isclose(m2 + 2, Xmatrix({ { 3.0, 4.0, 5.0 }, { 5.0, 4.0, 3.0 } })));
    REQUIRE(isclose(m2 - 2, Xmatrix({ { -1.0, 0.0, 1.0 }, { 1.0, 0.0, -1.0 } })));
    REQUIRE(isclose(m2 - m2, Xmatrix({ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } })));
    REQUIRE(isclose(m2 + m2, Xmatrix({ { 2.0, 4.0, 6.0 }, { 6.0, 4.0, 2.0 } })));

    m2 += 2;
    REQUIRE(isclose(m2, Xmatrix({ { 3.0, 4.0, 5.0 }, { 5.0, 4.0, 3.0 } })));
    m2 -= 2;
    REQUIRE(isclose(m2, Xmatrix({ { 1.0, 2.0, 3.0 }, { 3.0, 2.0, 1.0 } })));
    m2 += m2;
    REQUIRE(isclose(m2, Xmatrix({ { 2.0, 4.0, 6.0 }, { 6.0, 4.0, 2.0 } })));
    m2 -= m2;
    REQUIRE(isclose(m2, Xmatrix({ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } })));

    Xmatrix m3 = { { 1.0 }, { 2.0 }, { 3.0 } };
    REQUIRE(m3.dot(m3) == 14);

    std::vector<Xmatrix<double>> v = { m3, m3 * 2, m3 * 3 };
    auto sum_m3 = std::accumulate(v.begin(), v.end(), Xmatrix({ { 0.0 }, { 0.0 }, { 0.0 } }));
    REQUIRE(isclose(sum_m3, m3 * 6));

    auto x22 = Xmatrix({ { 3.1, 0.2 }, { 1.3, 3.4 } });
    REQUIRE(isclose(x22.inv(), Xmatrix({ { 0.330739, -0.0194553 }, { -0.126459, 0.301556 } })));

    auto x33 = Xmatrix({ { 0.0, -3.0, -2.0 }, { 1.0, -4.0, -2.0 }, { -3.0, 4.0, 1.0 } });
    REQUIRE(isclose(x33.inv(), Xmatrix({ { 4.0, -5.0, -2.0 }, { 5.0, -6.0, -2.0 }, { -8.0, 9.0, 3.0 } })));

    Xmatrix<double>(Shape(2, 3), 1.0);

    // std::cout << Xrand<Xmatrix<double>>(Shape(2, 3)) << std::endl;
    // std::cout << Xrand<Xmatrix<double>>(Shape(2, 3)) << std::endl;
    // std::cout << Xrand<Xmatrix<double>>(Shape(2, 3)) << std::endl;
    REQUIRE(Xnorm(x33) - 7.74597 < 0.0001);

    auto m33 = Xvstack(m3.T(), Xmatrix<double>({ { 1.0, 1.0, 1.0 } }));
    REQUIRE(isclose(m33, Xmatrix({ { 1.0, 2.0, 3.0 }, { 1.0, 1.0, 1.0 } })));

    auto mexp = Xmatrix<double>({ { 1.0, 0.0, 10.0 } });
    REQUIRE(isclose(exp(mexp), Xmatrix({ { 2.71828, 1.0, 22026.5 } })));

    auto mpow = Xmatrix<double>({ { 1.0, 0.0, 10.0 } });
    REQUIRE(isclose(power(mpow, 2), Xmatrix({ { 1.0, 0.0, 100.0 } })));

    auto msqrt = Xmatrix<double>({ { 1.0, 100.0, 2.0 } });
    REQUIRE(isclose(sqrt(msqrt), Xmatrix({ { 1.0, 10.0, 1.41421356 } })));

    auto sm1 = Xmatrix<double>({ { 1, 2 }, { 3, 5 } });
    auto sm2 = Xmatrix<double>({ { 1 }, { 2 } });

    auto sm3 = solve(sm1, sm2);
    REQUIRE(isclose(sm3, Xmatrix({ { -1.0 }, { 1.0 } })));

    Xmatrix im2 = { { 1.0, 2.0, 3.0 }, { 3.0, 2.0, 1.0 } };
    REQUIRE(isclose(im2[Index( 0,  2 )], Xmatrix({ { 3.0 } })));
    REQUIRE(isclose(im2[Index({ 1 }, { 0, 3, 2 })], Xmatrix({ { 3.0, 1.0 } })));
    REQUIRE(isclose(im2[Index({ 0, 2 }, { 2 })], Xmatrix({ { 3.0 }, { 1.0 } })));

    auto xr = Xrand<Xmatrix<double>>(Shape(2, 10));
    auto xrn = Xrandn<Xmatrix<double>>(Shape(2, 10));
}

TEST_CASE("xtestmix")
{
    Xmatrix<double> m1 = { { 1.0 } };
    Xmatrix<float> m2 = { { 1.0 } };

    // m1 + m2;
}