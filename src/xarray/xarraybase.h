#ifndef XARRAYBASE_H
#define XARRAYBASE_H

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include <opencv2/core.hpp>

#include "xarray/common.h"
#include "xarray/data_storage.h"
#include "xarray/index.h"
#include "xarray/shape.h"
#include "xarray/xarraybase_imp.h"

namespace xa {

template <typename A, int N>
using XarrayBaseImp = CVXarrayBaseImp<A, N>;

template <typename A, int N>
class XBase {
public:
    constexpr XBase() = default;

    XBase(const Shape<N>& shape_, const A* start, unsigned int len)
        : shape(shape_)
        , data_storage(start, len)
    {
    }

    template <class C>
    XBase(const Shape<N>& shape_, C&& data_)
        : shape(shape_)
        , data_storage(std::forward<C>(data_))
    {
    }

    XBase(const Shape<N>& shape_, A data_)
        : shape(shape_)
        , data_storage(shape.total(), data_)
    {
    }

    XBase(const Shape<N>& shape_)
        : shape(shape_)
        , data_storage(shape.total(), A {})
    {
    }

    const auto* raw() const
    {
        return this->data_storage.data();
    }

    auto* raw()
    {
        return this->data_storage.data();
    }

    Shape<N> shape = {};

protected:
    DataStorage<A> data_storage = {};
};

template <typename A, int N, typename I = XarrayBaseImp<A, N>>
class XarrayBase : public XBase<A, N> {
    using This_t = XarrayBase<A, N, I>;

public:
    using value_type = A;
    using imp_type = I;
    static constexpr int shape_size = N;

    using XBase<A, N>::XBase;
    using XBase<A, N>::shape;
    using XBase<A, N>::data_storage;

    XarrayBase(XBase<A, N> base)
        : XBase<A, N>(base)
    {
    }

    template <int... M>
    auto operator[](const Index<M...>& idx) const requires(get_final_shape_size<N, M...>() == 0)
    {
        auto new_data = data_storage.copy(idx, shape);
        return new_data[0];
    }

    template <int... M>
    auto& operator[](const Index<M...>& idx) requires(get_final_shape_size<N, M...>() == 0)
    {
        auto new_data = data_storage.copy(idx, shape);
        return new_data[0];
    }

    template <int... M>
    requires(get_final_shape_size<N, M...>() > 0) auto operator[](const Index<M...>& idx) const
    {
        auto new_shape = data_storage.get_shape(idx, shape);
        auto new_data = data_storage.copy(idx, shape);

        const auto ret = XarrayBase<A, new_shape.size(), XarrayBaseImp<A, new_shape.size()>>(new_shape, std::move(new_data));
        return ret;
    }

    template <int... M>
    requires(get_final_shape_size<N, M...>() > 0) auto operator[](const Index<M...>& idx)
    {
        auto new_shape = data_storage.get_shape(idx, shape);
        auto new_data = data_storage.copy(idx, shape);

        return XarrayBase<A, new_shape.size(), XarrayBaseImp<A, new_shape.size()>>(new_shape, std::move(new_data));
    }

    auto operator[](int idx) const
    {
        return (*this)[Index(idx)];
    }

    This_t copy() const
    {
        return This_t(shape, data_storage.data(), data_storage.size());
    }

    A item(const int idx = 0) const
    {
        return this->data_storage[idx];
    }

    This_t inv() const
    {
        return I(*this).inv();
    }

    This_t T() const
    {
        return I(*this).T();
    }

    template <typename U>
    auto astype() const
    {
        auto new_data = DataStorage<U>(data_storage);
        return XarrayBase<U, N, XarrayBaseImp<U, N>>(shape, std::move(new_data));
    }

    template <int M>
    auto reshape(const Shape<M>& shape)
    {
        return XarrayBase<A, M, XarrayBaseImp<A, M>>(shape, data_storage);
    }

    template <typename U>
    requires(arithmetic<U> || XBaseType<U>) auto dot(const U& op2)
        const
    {
        return I(*this).dot(op2);
    }

    template <typename U>
    requires(XBaseType<U>) auto matmul(const U& op2)
        const
    {
        return I(*this).matmul(op2);
    }

    template <typename U>
    requires(arithmetic<U> || XBaseType<U>) auto operator+(const U& op2) const
    {
        return I(*this) + op2;
    }

    template <typename U>
    requires(arithmetic<U> || XBaseType<U>) This_t& operator+=(const U& op2)
    {
        I(*this) += op2;
        return *this;
    }

    template <typename U>
    requires(arithmetic<U> || XBaseType<U>) This_t& operator-=(const U& op2)
    {
        I(*this) -= op2;
        return *this;
    }

    auto operator==(const XarrayBase<A, N, I>& op2) const
    {
        return I(*this) == op2;
    }
};

template <typename A, int N, typename I>
std::ostream& operator<<(std::ostream& stream, const XarrayBase<A, N, I>& x)
{
    stream << "[";
    if constexpr (N == 1) {
        for (int j = 0; j < x.shape[0]; ++j) {
            if constexpr (std::is_same_v<A, char>) {
                stream << int(x.raw()[j]);
            } else if constexpr (std::is_same_v<A, bool>) {
                stream << ((x.raw()[j]) ? "true" : "false");
            } else {
                stream << x.raw()[j];
            }
            if (j != x.shape[0] - 1) {
                stream << ", ";
            }
        }

    } else if constexpr (N == 2) {
        for (int i = 0; i < x.shape[0]; ++i) {
            stream << x[Index(i)];
            if (i != x.shape[0] - 1) {
                stream << "\n ";
            }
        }
    } else if (N == 3) {
        for (int i = 0; i < x.shape[0]; ++i) {
            stream << x[Index(i)];
            if (i != x.shape[0] - 1) {
                stream << "\n\n ";
            }
        }
    } else {
        std::cout << "not implemented" << std::endl;
    }
    stream << "]";

    return stream;
}

template <typename A, int N, typename I>
auto operator-(const XarrayBase<A, N, I>& op1)
{
    return -I(op1);
}

template <typename U, typename A, int N, typename I>
requires(arithmetic<U> || XBaseType<U>) auto operator-(const XarrayBase<A, N, I>& op1, const U& op2)
{
    return I(op1) - op2;
}

template <arithmetic U, typename A, int N, typename I>
auto operator-(const U& op2, const XarrayBase<A, N, I>& op1)
{
    return op2 - I(op1);
}

template <typename U, typename A, int N, typename I>
requires(arithmetic<U> || XBaseType<U>) auto operator*(const XarrayBase<A, N, I>& op1, const U& op2)
{
    return I(op1) * op2;
}

template <arithmetic U, typename A, int N, typename I>
auto operator*(const U& op2, const XarrayBase<A, N, I>& op1)
{
    return I(op1) * op2;
}

template <typename U, typename A, int N, typename I>
requires(arithmetic<U> || XBaseType<U>) auto operator/(const XarrayBase<A, N, I>& op1, const U& op2)
{
    return I(op1) / op2;
}

template <typename U, typename A, int N, typename I>
requires(arithmetic<U>) auto operator/(const U& op2, const XarrayBase<A, N, I>& op1)
{
    return op2 / I(op1);
}

template <typename U, typename A, int N, typename I>
requires(arithmetic<U> || ((!std::is_same_v<U, XarrayBase<A, N, I>>)&&(XBaseType<U>))) auto operator==(const XarrayBase<A, N, I>& op1, const U& op2)
{
    return I(op1) == op2;
}

template <typename U, typename A, int N, typename I>
requires(arithmetic<U> || ((!std::is_same_v<U, XarrayBase<A, N, I>>)&&(XBaseType<U>))) auto operator==(const U& op2, const XarrayBase<A, N, I>& op1)
{
    return I(op1) == op2;
}

template <typename U, typename A, int N, typename I>
requires(arithmetic<U> || XBaseType<U>) auto operator<(const XarrayBase<A, N, I>& op1, const U& op2)
{
    return I(op1) < op2;
}

template <typename A, int N, typename I>
A det(const XarrayBase<A, N, I>& op1)
{
    return I(op1).det();
}

template <typename A, int N, typename I>
auto svd(const XarrayBase<A, N, I>& op1)
{
    return I(op1).svd();
}

template <typename A, int N, typename I>
bool all(const XarrayBase<A, N, I>& op1)
{
    return I(op1).all();
}

template <typename A, int N, typename I>
bool isclose(const XarrayBase<A, N, I>& op1, const XarrayBase<A, N, I>& op2)
{
    return I(op1).isclose(op2);
}

template <typename T, int N>
T Xrand(Shape<N> shape)
{
    return T::imp_type::rand(shape);
}

template <typename T, int N>
T Xrandn(Shape<N> shape)
{
    return T::imp_type::randn(shape);
}

template <typename A, int N, typename I, typename C, typename P>
XarrayBase<A, N, I> choice(C&& population, Shape<N> shape, P&& probability)
{
    return I::choice(std::forward<C>(population), shape, std::forward<P>(probability));
}

template <typename A, int N, typename I>
A Xnorm(const XarrayBase<A, N, I>& op1)
{
    return I(op1).norm();
}

template <typename A, int N, typename I>
XarrayBase<A, N, I> Xvstack(const XarrayBase<A, N, I>& op1, const XarrayBase<A, N, I>& op2)
{
    return I(op1).vstack(op2);
}

template <typename A, int N, typename I>
XarrayBase<A, N, I> Xhstack(const XarrayBase<A, N, I>& op1, const XarrayBase<A, N, I>& op2)
{
    return I(op1).hstack(op2);
}

template <typename A, int N, typename I>
XarrayBase<A, N, I> exp(const XarrayBase<A, N, I>& op1)
{
    return I(op1).exp();
}

template <typename A, int N, typename I>
XarrayBase<A, N, I> abs(const XarrayBase<A, N, I>& op1)
{
    return I(op1).abs();
}

template <typename D, typename A, int N, typename I>
XarrayBase<A, N, I> power(const XarrayBase<A, N, I>& op1, D power)
{
    return I(op1).power(power);
}

template <typename A, int N, typename I>
XarrayBase<A, N, I> sqrt(const XarrayBase<A, N, I>& op1)
{
    return I(op1).sqrt();
}

template <typename A, int N, typename I>
XarrayBase<A, N, I> repeat(const XarrayBase<A, N, I>& op1, int repeats)
{
    return I(op1).repeat(repeats);
}

template <typename A, int N, typename I>
A sum(const XarrayBase<A, N, I>& op1)
{
    return I(op1).sum();
}

template <typename A, int N, typename I, typename AXIS = void*>
auto mean(const XarrayBase<A, N, I>& op1, AXIS axis = {})
{
    return I(op1).mean(axis);
}

template <typename A, int N = 1, typename I = XarrayBaseImp<A, N>>
XarrayBase<A, N, I> linspace(A start, A end, int num, bool endpoint = true)
{
    return I::linspace(start, end, num, endpoint);
}

template <typename A, int N, typename I>
XarrayBase<A, N, I> solve(const XarrayBase<A, N, I>& op1, const XarrayBase<A, N, I>& op2)
{
    return I(op1).solve(op2);
}
}
#endif
