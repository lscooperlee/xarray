#ifndef XARRAYBASE_H
#define XARRAYBASE_H

#include <algorithm>
#include <iostream>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

#include <opencv2/core.hpp>

#include "common.h"
#include "data_storage.h"
#include "index.h"
#include "shape.h"
#include "xarraybase_imp.h"

template <typename S>
using XarrayBaseImp = CVXarrayBaseImp<S>;

template <typename S, typename A, typename I, int N>
class XBase {
public:
    constexpr XBase() = default;

    explicit XBase(const Shape<N>& shape_)
        : shape(shape_)
        , data_storage()
    {
    }

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

    Shape<N> shape = {};

protected:
    DataStorage<A> data_storage = {};
};

template <typename S, typename A, int N, typename I = XarrayBaseImp<S>>
class XarrayBase : XBase<S, A, I, N> {
public:
    using value_type = A;
    using sub_type = S;
    using imp_type = I;
    static constexpr int shape_size = N;

    using XBase<S, A, I, N>::XBase;
    using XBase<S, A, I, N>::shape;
    using XBase<S, A, I, N>::data_storage;

    using base_class = XarrayBase<S, A, N, I>;

    operator S() const
    {
        return static_cast<const S&>(*this);
    }

    XarrayBase<S, A, N, I> copy() const
    {
        return XarrayBase(shape, data_storage.data(), data_storage.size());
    }

    template <int... M>
    auto operator[](const Index<M...>& idx) const
    {
        auto new_shape = data_storage.get_shape(idx, shape);
        auto new_data = data_storage.copy(idx, shape);
        if constexpr (new_shape.size() == 0) {
            assert (new_data.size() == 1);
            return new_data[0];
        } else {
            return XarrayBase<S, A, new_shape.size(), I>(new_shape, std::move(new_data));
        }
    }

    A item(const int idx = 0) const
    {
        return this->data_storage[idx];
    }

    const A* raw() const
    {
        return this->data_storage.data();
    }

    A* raw()
    {
        return this->data_storage.data();
    }

    XarrayBase<S, A, N, I> inv() const
    {
        return I(*this).inv();
    }

    XarrayBase<S, A, N, I> T() const
    {
        return I(*this).T();
    }

    XarrayBase<S, A, N, I> t() const
    {
        return T();
    }

    template <typename U>
    requires XBaseType<U> A dot(const U& op2)
    const
    {
        return I(*this).dot(op2);
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U> XarrayBase<S, A, N, I>
    operator*(const U& op2) const
    {
        return I(*this) * op2;
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U> XarrayBase<S, A, N, I>
    operator/(const U& op2) const
    {
        return I(*this) / op2;
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U> XarrayBase<S, A, N, I>
    operator-(const U& op2) const
    {
        return I(*this) - op2;
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U> XarrayBase<S, A, N, I>
    operator+(const U& op2) const
    {
        return I(*this) + op2;
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U> XarrayBase<S, A, N, I>
    &operator+=(const U& op2)
    {
        I(*this) += op2;
        return *this;
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U> XarrayBase<S, A, N, I>
    &operator-=(const U& op2)
    {
        I(*this) -= op2;
        return *this;
    }
};

template <typename S, typename A, int N, typename I>
std::ostream& operator<<(std::ostream& stream, const XarrayBase<S, A, N, I>& x)
{
    auto sz = x.shape.size();
    stream << "[";
    if (sz == 1) {
        for (int j = 0; j < x.shape[0]; ++j) {
            if constexpr (std::is_same_v<A, char>) {
                stream << int(x.raw()[j]);
            } else {
                stream << x.raw()[j];
            }
            if (j != x.shape[0] - 1) {
                stream << ", ";
            }
        }

    } else if (sz == 2) {
        for (int i = 0; i < x.shape[0]; ++i) {
            stream << "[";
            for (int j = 0; j < x.shape[1]; ++j) {
                if constexpr (std::is_same_v<A, unsigned char>) {
                    stream << int(x.raw()[i * x.shape[1] + j]);
                } else {
                    stream << x.raw()[i * x.shape[1] + j];
                }
                if (j != x.shape[1] - 1) {
                    stream << ", ";
                }
            }
            stream << "]";
            if (i != x.shape[0] - 1) {
                stream << "\n";
            }
        }
    } else {
        std::cout << "not implemented" << std::endl;
    }
    stream << "]";

    return stream;
}

template <arithmetic U, typename S, typename A, int N, typename I>
XarrayBase<S, A, N, I> operator*(const U& op2, const XarrayBase<S, A, N, I>& op1)
{
    return op1 * op2;
}

template <typename S, typename A, int N, typename I>
A det(const XarrayBase<S, A, N, I>& op1)
{
    return I(op1).det();
}

template <typename S, typename A = typename S::value_type, int N, typename I>
bool all(const XarrayBase<S, A, N, I>& op1)
{
    return I(op1).all();
}

template <typename S, typename A = typename S::value_type, int N, typename I>
bool isclose(const XarrayBase<S, A, N, I>& op1, const XarrayBase<S, A, N, I>& op2)
{
    return I(op1).isclose(op2);
}

template <typename S, typename A = typename S::value_type, int N = S::shape_size, typename I = typename S::imp_type>
XarrayBase<S, A, N, I> Xrand(Shape<N> shape)
{
    return I::rand(shape);
}

template <typename S, typename A = typename S::value_type, int N = S::shape_size, typename I = typename S::imp_type>
XarrayBase<S, A, N, I> Xrandn(Shape<N> shape)
{
    return I::randn(shape);
}

template <typename S, typename A = typename S::value_type, int N, typename I, typename C, typename P>
XarrayBase<S, A, N, I> choice(C&& population, Shape<N> shape, P&& probability)
{
    return I::choice(std::forward<C>(population), shape, std::forward<P>(probability));
}

template <typename S, typename A = typename S::value_type, int N, typename I>
A Xnorm(const XarrayBase<S, A, N, I>& op1)
{
    return I(op1).norm();
}

template <typename S, typename A = typename S::value_type, int N, typename I>
XarrayBase<S, A, N, I> Xvstack(const XarrayBase<S, A, N, I>& op1, const XarrayBase<S, A, N, I>& op2)
{
    return I(op1).vstack(op2);
}

template <typename S, typename A = typename S::value_type, int N, typename I>
XarrayBase<S, A, N, I> Xhstack(const XarrayBase<S, A, N, I>& op1, const XarrayBase<S, A, N, I>& op2)
{
    return I(op1).hstack(op2);
}

template <typename S, typename A = typename S::value_type, int N, typename I>
XarrayBase<S, A, N, I> exp(const XarrayBase<S, A, N, I>& op1)
{
    return I(op1).exp();
}

template <typename S, typename D, typename A = typename S::value_type, int N, typename I>
XarrayBase<S, A, N, I> power(const XarrayBase<S, A, N, I>& op1, D power)
{
    return I(op1).power(power);
}

template <typename S, typename A = typename S::value_type, int N, typename I>
XarrayBase<S, A, N, I> sqrt(const XarrayBase<S, A, N, I>& op1)
{
    return I(op1).sqrt();
}

template <typename S, typename A = typename S::value_type, int N, typename I>
XarrayBase<S, A, N, I> solve(const XarrayBase<S, A, N, I>& op1, const XarrayBase<S, A, N, I>& op2)
{
    return I(op1).solve(op2);
}

#endif
