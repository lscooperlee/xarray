#ifndef XARRAYBASE_IMP_H
#define XARRAYBASE_IMP_H

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

#include <opencv2/core.hpp>

#include "xarray/common.h"
#include "xarray/data_storage.h"
#include "xarray/shape.h"

namespace xa {

template <typename A, int N, typename I>
class XarrayBase;

template <typename I, int M>
requires(M > 0) class CVXarrayBaseImp : public cv::Mat {

    template <typename A, int N>
    using TT = CVXarrayBaseImp<A, N>;

    template <typename U>
    using IMP = CVXarrayBaseImp<typename U::value_type, U::shape_size>;

    template <typename A, int N>
    using TYP = XarrayBase<A, N, TT<A, N>>;

private:
    static constexpr int get_type()
    {
        int type = 0;
        if constexpr (std::is_same_v<I, float>) {
            type = CV_32F;
        } else if constexpr (std::is_same_v<I, double>) {
            type = CV_64F;
        } else if constexpr (std::is_same_v<I, char>) {
            type = CV_8S;
        } else if constexpr (std::is_same_v<I, unsigned char>) {
            type = CV_8U;
        } else if constexpr (std::is_same_v<I, int>) {
            type = CV_32S;
        } else if constexpr (std::is_same_v<I, bool>) {
            type = CV_8U;
        } else {
            throw std::runtime_error("float double bool when create XarrayImp");
        }

        return type;
    }

    template <int N>
    static constexpr auto get_rc(Shape<N> shape) -> std::tuple<int, int>
    {
        int rows, cols;
        if (shape.size() == 1) {
            rows = 1;
            cols = shape[0];
        } else if (shape.size() == 2) {
            rows = shape[0];
            cols = shape[1];
        } else {
            throw std::runtime_error("shape is wrong");
        }
        return std::tie(rows, cols);
    }

public:
    template <int N>
    static TYP<I, M> rand(Shape<N> shape)
    {
        auto [rows, cols] = get_rc(shape);

        cv::Mat m(rows, cols, get_type());
        cv::randu(m, cv::Scalar(0), cv::Scalar(1));

        return TYP<I, M> { TT<I, M>(m, shape) };
    }

    template <int N>
    static TYP<I, M> randn(Shape<N> shape)
    {
        auto [rows, cols] = get_rc(shape);

        cv::Mat m(rows, cols, get_type());
        cv::randn(m, cv::Scalar(0), cv::Scalar(1));

        return TYP<I, M> { TT<I, M>(m, shape) };
    }

    template <typename C, typename P, int N>
    static TYP<I, M> choice(C&& population, Shape<N> shape, P&& probability)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution p(probability.begin(), probability.end());

        DataStorage<I> d = {};

        for (int i = 0; i < shape.total(); ++i) {
            auto c = p(gen);
            d.append(population[c]);
        }

        return TYP<I, M>(shape, d);
    }

public:
    CVXarrayBaseImp(const cv::MatExpr& e, Shape<M> shape_ = {})
        : cv::Mat(e)
        , shape(shape_)
    {
    }

    CVXarrayBaseImp(const cv::Mat& e, Shape<M> shape_ = {})
        : cv::Mat(e)
        , shape(shape_)
    {
    }

    template <typename U>
    requires XBaseType<U>
    CVXarrayBaseImp(const U& x)
        : shape(x.shape)
    {
        auto [rows, cols] = get_rc(x.shape);
        static_cast<cv::Mat&>(*this) = cv::Mat(rows, cols, get_type(), const_cast<void*>(reinterpret_cast<const void*>((x.raw()))));
        //shape = x.shape; //shape changed by *this = ..., static_cast<cv::Mat&>(*this) for the fix
    }

    explicit operator TYP<I, M>() const
    {
        return TYP<I, M>(shape, ptr<I>(), rows * cols);
    }

    bool all() const
    {
        cv::Mat _;
        cv::findNonZero(*this, _);
        return _.total() == size_t(cols * rows);
    }

    template <typename U>
    bool isclose(const U& op1) const
    {
        auto tmp = static_cast<const cv::Mat&>(*this) - IMP<U>(op1);
        return TT<I, M>(tmp < 0.00001).all();
    }

    I det() const
    {
        return cv::determinant(*this);
    }

    std::tuple<TYP<I, M>, TYP<I, 1>, TYP<I, M>> svd() const
    {
        if constexpr (M == 2) {

            auto s = cv::SVD(*this, cv::SVD::FULL_UV);

            auto u = TYP<I, M>(TT<I, M>(s.u));
            u.shape = Shape(shape[0], shape[0]); // if for FULL_UV

            auto vt = TYP<I, M>(TT<I, M>(s.vt));
            vt.shape = Shape(shape[1], shape[1]); // if for FULL_UV

            auto w = TYP<I, 1>(TT<I, 1>(s.w));
            w.shape = Shape(std::min(shape[0], shape[1])); // if for FULL_UV

            return std::tie(u, w, vt);
        }
    }

    TYP<I, M> inv() const
    {
        return TYP<I, M>(TT<I, M>(this->cv::Mat::inv(cv::DECOMP_SVD), shape));
    }

    TYP<I, M> T() const
    {
        if constexpr (M == 1) {
            return TYP<I, M>(*this);
        } else {
            return TYP<I, M>(TT<I, M>(this->cv::Mat::t(), Shape(shape[1], shape[0])));
        }
    }

    TYP<I, M> repeat(int repeats) const
    {
        if constexpr (M == 1) {
            return TYP<I, M>(TT<I, M>(cv::repeat(*this, repeats, 1), Shape(shape[0] * repeats)));
        } else {
            return TYP<I, M>(TT<I, M>(cv::repeat(*this, repeats, 1), Shape(shape[0], shape[1] * repeats)));
        }
    }

    template <typename U>
    auto matmul(const U& op2) const
    {
        if constexpr (M == U::shape_size && M == 2) {
            if (shape[1] == op2.shape[0]) {
                return TYP<I, M>(TT<I, M>(static_cast<const cv::Mat&>(*this) * IMP<U>(op2), Shape(shape[0], op2.shape[1])));
            } else {
                throw std::runtime_error("shape error in imp matmul");
            }
        } else if constexpr (U::shape_size == 1 && M == 2) {
            auto new_op2 = (TT<I, 1>(op2)).cv::Mat::reshape(1, op2.shape[0]);
            auto m = TT<I, 1>(static_cast<const cv::Mat&>(*this) * new_op2, Shape(shape[0]));
            TYP<I, 1> ret(m);
            return ret;
        } else if constexpr (U::shape_size == 2 && M == 1) {
            auto new_this = this->cv::Mat::reshape(1, 1);
            return TYP<I, M>(TT<I, 1>(new_this * TT<I, 2>(op2), Shape(op2.shape[1])));
        } else {
            throw std::runtime_error("matmul not imp yet");
        }
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U>
    auto dot(const U& op2) const
    {
        if constexpr (std::is_arithmetic_v<U>) {
            return *this * op2;
        } else if constexpr (M == U::shape_size && M == 1) {
            if (shape == op2.shape) {
                return this->cv::Mat::dot(IMP<U>(op2));
            } else {
                throw std::runtime_error("shape error in dot");
            }

        } else if constexpr (M == U::shape_size && M == 2) {
            return this->matmul(op2);
        } else {
            throw std::runtime_error("not impliment for dot");
        }
    }

    template <typename U>
    requires XBaseType<U> TYP<I, M> vstack(const U& op2)
    const
    {
        auto is_vector_same_column = (shape.size() == 2) && (op2.shape.size() == 2) && (shape[1] == shape[1]);
        if (is_vector_same_column) {
            cv::Mat _;
            cv::vconcat(*this, IMP<U>(op2), _);
            return TYP<I, M> { TT<I, M>(_, Shape(shape[0] + op2.shape[0], shape[1])) };
        } else {
            throw std::runtime_error("shape error in vstack");
        }
    }

    template <typename U>
    requires XBaseType<U> TYP<I, M> hstack(const U& op2)
    const
    {
        auto is_vector_same_row = (shape.size() == 2) && (op2.shape.size() == 2) && (shape[0] == shape[0]);
        if (is_vector_same_row) {
            cv::Mat _;
            cv::hconcat(*this, IMP<U>(op2), _);
            return TYP<I, M> { TT<I, M>(_, Shape(shape[0], op2.shape[1] + shape[1])) };
        } else {
            throw std::runtime_error("shape error in hstack");
        }
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U>
    auto operator<(const U& op2) const
    {
        if constexpr (std::is_arithmetic_v<U>) {
            return TYP<I, M>(TT<bool, M>(static_cast<const cv::Mat&>(*this) < op2, shape));
        } else if constexpr (std::derived_from<U, TYP<I, M>>) {
            if (shape == op2.shape) {
                return TYP<I, M>(TT<bool, M>(static_cast<const cv::Mat&>(*this) < IMP<U>(op2), shape));
            } else {
                throw std::runtime_error("shape error in <");
            }
        }
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U>
    auto operator==(const U& op2) const
    {
        if constexpr (std::is_arithmetic_v<U>) {
            return TYP<I, M>(TT<bool, M>(static_cast<const cv::Mat&>(*this) == op2, shape));
        } else if constexpr (std::derived_from<U, TYP<I, M>>) {
            if (shape == op2.shape) {
                return TYP<bool, M>(TT<bool, M>(static_cast<const cv::Mat&>(*this) == IMP<U>(op2), shape));
            } else {
                throw std::runtime_error("shape error in <");
            }
        }
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U> TYP<I, M>
    operator*(const U& op2) const
    {
        if constexpr (std::is_arithmetic_v<U>) {
            return TYP<I, M>(TT<I, M>(static_cast<const cv::Mat&>(*this) * op2, shape));
        } else if constexpr (std::derived_from<U, TYP<I, M>>) {
            if (shape == op2.shape) {
                return TYP<I, M>(TT<I, M>(this->mul(IMP<U>(op2)), shape));
            } else {
                throw std::runtime_error("shape error in imp *");
            }
        }
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U> TYP<I, M>
    operator/(const U& op2) const
    {
        if constexpr (std::is_arithmetic_v<U>) {
            return TYP<I, M>(TT<I, M>(static_cast<const cv::Mat&>(*this) / op2, shape));
        } else if constexpr (std::derived_from<U, TYP<I, M>>) {
            if (shape == op2.shape) {
                return TYP<I, M>(TT<I, M>(static_cast<const cv::Mat&>(*this) / IMP<U>(op2), shape));
            } else {
                throw std::runtime_error("shape error in imp /");
            }
        }
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U>
    auto operator+(const U& op2) const
    {
        if constexpr (std::is_arithmetic_v<U>) {
            return TYP<I, M>(TT<I, M>(static_cast<const cv::Mat&>(*this) + op2, shape));
        } else if constexpr (XBaseType<U>) {

            auto constexpr SZ = (U::shape_size > M) ? U::shape_size : M;
            Shape<SZ> new_shape;
            if constexpr (U::shape_size > M) {
                new_shape = op2.shape;
            } else {
                new_shape = shape;
            }

            if (shape == op2.shape) {
                return TYP<I, SZ>(TT<I, SZ>(static_cast<const cv::Mat&>(*this) + IMP<U>(op2), new_shape));
            } else if ((shape.total() % op2.shape.total()) == 0) {
                auto times = shape.total() / op2.shape.total();
                auto op = cv::repeat(static_cast<const cv::Mat&>(IMP<U>(op2)), times, 1);
                auto [r, _] = get_rc(shape);

                return TYP<I, SZ>(TT<I, SZ>(static_cast<const cv::Mat&>(*this) + op.reshape(1, r), new_shape));

            } else if ((op2.shape.total() % shape.total()) == 0) {
                auto times = op2.shape.total() / shape.total();
                auto op = cv::repeat(*this, times, 1);
                auto [r, _] = get_rc(op2.shape);

                return TYP<I, SZ>(TT<I, SZ>(op.reshape(1, r) + IMP<U>(op2), new_shape));
            } else {
                throw std::runtime_error("shape error in imp +");
            }
        }
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U>
    auto operator-(const U& op2) const
    {
        if constexpr (std::is_arithmetic_v<U>) {
            return TYP<I, M>(TT<I, M>(static_cast<const cv::Mat&>(*this) - op2, shape));
        } else if constexpr (XBaseType<U>) {

            auto constexpr SZ = (U::shape_size > M) ? U::shape_size : M;
            Shape<SZ> new_shape;
            if constexpr (U::shape_size > M) {
                new_shape = op2.shape;
            } else {
                new_shape = shape;
            }

            if (shape == op2.shape) {
                return TYP<I, SZ>(TT<I, SZ>(static_cast<const cv::Mat&>(*this) - IMP<U>(op2), new_shape));
            } else if ((shape.total() % op2.shape.total()) == 0) {
                auto times = shape.total() / op2.shape.total();
                auto op = cv::repeat(static_cast<const cv::Mat&>(IMP<U>(op2)), times, 1);
                auto [r, _] = get_rc(shape);

                return TYP<I, SZ>(TT<I, SZ>(static_cast<const cv::Mat&>(*this) - op.reshape(1, r), new_shape));

            } else if ((op2.shape.total() % shape.total()) == 0) {
                auto times = op2.shape.total() / shape.total();
                auto op = cv::repeat(*this, times, 1);
                auto [r, _] = get_rc(op2.shape);

                return TYP<I, SZ>(TT<I, SZ>(op.reshape(1, r) - IMP<U>(op2), new_shape));
            } else {
                throw std::runtime_error("shape error in imp -");
            }
        }
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U>
    void operator+=(const U& op2)
    {
        if constexpr (std::is_arithmetic_v<U>) {
            static_cast<cv::Mat&>(*this) += op2;
        } else if constexpr (std::derived_from<U, TYP<I, M>>) {
            if (shape == op2.shape) {
                static_cast<cv::Mat&>(*this) += IMP<U>(op2);
            } else {
                throw std::runtime_error("shape error in imp +=");
            }
        }
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U>
    void operator-=(const U& op2)
    {
        if constexpr (std::is_arithmetic_v<U>) {
            static_cast<cv::Mat&>(*this) -= op2;
        } else if constexpr (std::derived_from<U, TYP<I, M>>) {
            if (shape == op2.shape) {
                static_cast<cv::Mat&>(*this) -= IMP<U>(op2);
            } else {
                throw std::runtime_error("shape error in imp -=");
            }
        }
    }

    I norm() const
    {
        return cv::norm(*this);
    }

    TYP<I, M> exp() const
    {
        cv::Mat _;
        cv::exp(*this, _);

        TYP<I, M> ret { TT<I, M>(_) };
        ret.shape = shape;

        return ret;
    }

    template <typename D>
    TYP<I, M> power(D power) const
    {
        cv::Mat _;
        cv::pow(*this, power, _);

        return TYP<I, M> { TT<I, M>(_, shape) };
    }

    TYP<I, M> sqrt() const
    {
        cv::Mat _;
        cv::sqrt(*this, _);
        return TYP<I, M> { TT<I, M>(_, shape) };
    }

    I sum() const
    {
        return cv::sum(*this)[0];
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U> TYP<I, M> solve(const U& op2)
    const
    {
        cv::Mat _;
        cv::solve(*this, IMP<U>(op2), _, cv::DECOMP_SVD);
        return TYP<I, M> { TT<I, M>(_, Shape(shape[1], 1)) };
    }

private:
    Shape<M> shape = {};
};

template <typename I>
struct CVXarrayBaseImp<I, 0> {
    I value;
    CVXarrayBaseImp(I v)
        : value(v) {};
};

}
#endif