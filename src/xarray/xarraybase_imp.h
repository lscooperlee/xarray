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

template <typename A, int N>
class XBase;

// template <typename I, int M=0> requires (M == 0)
// class CVXarrayBaseImp {
// };

template <typename I, int M>
requires(M > 0) class CVXarrayBaseImp : public cv::Mat {
    using This_t = CVXarrayBaseImp<I, M>;
    //using Type_t = XBase<I, M>;
    using Type_t = XarrayBase<I, M, This_t>;

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
        } else {
            std::cout << "float double bool when create XarrayImp";
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
    static Type_t rand(Shape<N> shape)
    {
        auto [rows, cols] = get_rc(shape);

        cv::Mat m(rows, cols, get_type());
        cv::randu(m, cv::Scalar(0), cv::Scalar(1));

        Type_t ret { This_t(m) };
        ret.shape = shape;

        return ret;
    }

    template <int N>
    static Type_t randn(Shape<N> shape)
    {
        auto [rows, cols] = get_rc(shape);

        cv::Mat m(rows, cols, get_type());
        cv::randn(m, cv::Scalar(0), cv::Scalar(1));

        Type_t ret { This_t(m) };
        ret.shape = shape;

        return ret;
    }

    template <typename C, typename P, int N>
    static Type_t choice(C&& population, Shape<N> shape, P&& probability)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution p(probability.begin(), probability.end());

        DataStorage<I> d = {};

        for (int i = 0; i < shape.total(); ++i) {
            auto c = p(gen);
            d.append(population[c]);
        }

        return Type_t(shape, d);
    }

public:
    //using cv::Mat::Mat;
    CVXarrayBaseImp(const cv::MatExpr& e)
        : cv::Mat(e)
    {
    }

    CVXarrayBaseImp(const cv::Mat& e)
        : cv::Mat(e)
    {
    }

    CVXarrayBaseImp(const Type_t& x)
    {
        auto [rows, cols] = get_rc(x.shape);
        *this = cv::Mat(rows, cols, get_type(), const_cast<void*>(reinterpret_cast<const void*>((x.raw()))));
        shape = x.shape; //shape changed by *this = ...
    }

    explicit operator Type_t() const
    {
        return Type_t(shape, this->ptr<I>(), (this->rows * this->cols));
    }

    bool all() const
    {
        cv::Mat _;
        const auto& self = static_cast<const cv::Mat&>(*this);
        cv::findNonZero(self, _);
        return _.total() == size_t(self.cols * self.rows);
    }

    template <typename U>
    bool isclose(const U& op1) const
    {
        auto tmp = static_cast<const cv::Mat&>(*this) - This_t(op1);
        return This_t(tmp < 0.00001).all();
    }

    I det() const
    {
        return cv::determinant(static_cast<const cv::Mat&>(*this));
    }

    Type_t inv() const
    {
        Type_t ret(This_t(static_cast<const cv::Mat*>(this)->inv(cv::DECOMP_SVD)));
        ret.shape = shape;
        return ret;
    }

    Type_t T() const
    {
        if (shape.size() == 1) {
            Type_t ret(This_t(*this));
            return ret;
        } else {
            Type_t ret(This_t(static_cast<const cv::Mat*>(this)->t()));
            ret.shape[0] = shape[1];
            ret.shape[1] = shape[0];
            return ret;
        }
    }

    template <typename U>
    auto matmul(const U& op2) const
    {
        if constexpr (M == U::shape_size && M == 2) {
            if (shape[1] == op2.shape[0]) {
                Type_t ret(This_t(static_cast<const cv::Mat&>(*this) * This_t(op2)));
                ret.shape = Shape(shape[0], op2.shape[1]);
                return ret;
            } else {
                throw std::runtime_error("shape error in imp matmul");
            }
        } else if constexpr (U::shape_size == 1 && M == 2) {
            auto new_op2 = static_cast<const cv::Mat&>(CVXarrayBaseImp<I, 1>(op2)).reshape(1, op2.shape[0]);
            auto m = CVXarrayBaseImp<I, 1>(static_cast<const cv::Mat&>(*this) * new_op2);
            XarrayBase<I, 1, CVXarrayBaseImp<I, 1>> ret(m);
            ret.shape = Shape(shape[0]);
            return ret;
        } else if constexpr (U::shape_size == 2 && M == 1) {
            auto new_this = static_cast<const cv::Mat&>(*this).reshape(1, 1);
            Type_t ret(CVXarrayBaseImp<I, 1>(new_this * CVXarrayBaseImp<I, 2>(op2)));
            ret.shape = Shape(op2.shape[1]);
            return ret;
        } else {
            throw std::runtime_error("matmul not imp yet");
        }
    }

    template <typename U>
    requires arithmetic<U> || std::derived_from<U, Type_t>
    auto dot(const U& op2) const
    {
        if constexpr (std::is_arithmetic_v<U>) {
            return *this * op2;
        } else if constexpr (M == U::shape_size && M == 1) {
            if (shape == op2.shape) {
                I ret = static_cast<const cv::Mat*>(this)->dot(This_t(op2));
                return ret;
            } else {
                throw std::runtime_error("shape error in dot");
            }

        } else if constexpr (M == U::shape_size && M == 2) {
            return this->matmul(op2);
        } else {

                throw std::runtime_error("not impliment for dot");
            // auto is_vector_2d = (shape.size() == 2) && (op2.shape.size() == 2)
            //     && (shape == op2.shape) && (shape[0] == 1 || shape[1] == 1);
            // auto is_vector_1d = (shape.size() == 1) && (op2.shape.size() == 1) && (shape == op2.shape);

            // if (is_vector_2d || is_vector_1d) {
            //     I ret = static_cast<const cv::Mat*>(this)->dot(This_t(op2));
            //     return ret;
            // } else {
            //     throw std::runtime_error("shape error in dot");
            // }
        }
    }

    template <typename U>
    requires std::derived_from<U, Type_t> Type_t vstack(const U& op2)
    const
    {
        auto is_vector_same_column = (shape.size() == 2) && (op2.shape.size() == 2) && (shape[1] == shape[1]);
        // auto is_vector_1d = (shape.size() == 1) && (op2.shape.size() == 1) && (shape == op2.shape);
        if (is_vector_same_column) {

            cv::Mat _;

            auto m1 = static_cast<const cv::Mat&>(*this);
            auto m2 = static_cast<const cv::Mat&>(This_t(op2));

            cv::vconcat(m1, m2, _);

            Type_t ret { This_t(_) };
            ret.shape = Shape(shape[0] + op2.shape[0], shape[1]);
            return ret;

        } else {
            throw std::runtime_error("shape error in dot");
        }
    }

    template <typename U>
    requires std::derived_from<U, Type_t> Type_t hstack(const U& op2)
    const
    {
        auto is_vector_same_row = (shape.size() == 2) && (op2.shape.size() == 2) && (shape[0] == shape[0]);
        // auto is_vector_1d = (shape.size() == 1) && (op2.shape.size() == 1) && (shape == op2.shape);
        if (is_vector_same_row) {

            cv::Mat _;

            auto m1 = static_cast<const cv::Mat&>(*this);
            auto m2 = static_cast<const cv::Mat&>(This_t(op2));

            cv::hconcat(m1, m2, _);

            Type_t ret { This_t(_) };
            ret.shape = Shape(shape[0], op2.shape[1] + shape[1]);
            return ret;

        } else {
            throw std::runtime_error("shape error in dot");
        }
    }

    template <typename U>
    requires arithmetic<U> || std::derived_from<U, Type_t>
    bool operator<(const U&) const
    {
        if constexpr (std::is_arithmetic_v<U>) {
            // const auto& self = static_cast<const cv::Mat&>(*this);
            // Type_t ret(This_t(static_cast<const cv::Mat&>(*this) * op2));
            // ret.shape = shape;
            // return ret;
        } else if constexpr (std::derived_from<U, Type_t>) {
            // if ((shape.size() == 2)
            //     && (op2.shape.size() == 2)
            //     && (shape[1] == op2.shape[0])) {
            //     Type_t ret(This_t(static_cast<const cv::Mat&>(*this) * This_t(op2)));
            //     ret.shape = Shape(shape[0], op2.shape[1]);
            //     return ret;
            // } else if (shape == op2.shape) {

            //     Type_t ret(This_t(this->mul(This_t(op2))));
            //     ret.shape = shape;
            //     return ret;
            // } else {
            //     throw std::runtime_error("shape error in imp *");
            // }
        }
        return true;
    }

    template <typename U>
    requires arithmetic<U> || std::derived_from<U, Type_t> Type_t operator*(const U& op2) const
    {
        if constexpr (std::is_arithmetic_v<U>) {
            Type_t ret(This_t(static_cast<const cv::Mat&>(*this) * op2));
            ret.shape = shape;
            return ret;
        } else if constexpr (std::derived_from<U, Type_t>) {
            if (shape == op2.shape) {
                Type_t ret(This_t(this->mul(This_t(op2))));
                ret.shape = shape;
                return ret;
            } else {
                throw std::runtime_error("shape error in imp *");
            }
        }
    }

    template <typename U>
    requires arithmetic<U> || std::derived_from<U, Type_t> Type_t operator/(const U& op2) const
    {
        if constexpr (std::is_arithmetic_v<U>) {
            Type_t ret(This_t(static_cast<const cv::Mat&>(*this) / op2));
            ret.shape = shape;
            return ret;
        } else if constexpr (std::derived_from<U, Type_t>) {
            if (shape == op2.shape) {
                Type_t ret(This_t(static_cast<const cv::Mat&>(*this) - This_t(op2)));
                ret.shape = shape;
                return ret;
            } else {
                std::cout << shape << std::endl;
                std::cout << op2.shape << std::endl;
                throw std::runtime_error("shape error in imp /");
            }
        }
    }

    template <typename U>
    requires arithmetic<U> || XBaseType<U> Type_t operator+(const U& op2) const
    {
        if constexpr (arithmetic<U>) {
            Type_t ret(This_t(static_cast<const cv::Mat&>(*this) + op2));
            ret.shape = shape;
            return ret;
        } else if constexpr (XBaseType<U>) {
            if (shape == op2.shape) {
                Type_t ret(This_t(static_cast<const cv::Mat&>(*this) + This_t(op2)));
                ret.shape = shape;
                return ret;
            } else {
                std::cout << shape << ", " << op2.shape << std::endl;
                throw std::runtime_error("shape error in imp +");
            }
        }
    }

    template <typename U>
    requires arithmetic<U> || std::derived_from<U, Type_t> Type_t
    operator-(const U& op2) const
    {
        if constexpr (std::is_arithmetic_v<U>) {
            Type_t ret(This_t(static_cast<const cv::Mat&>(*this) - op2));
            ret.shape = shape;
            return ret;
        } else if constexpr (std::derived_from<U, Type_t>) {
            if (shape == op2.shape) {
                Type_t ret(This_t(static_cast<const cv::Mat&>(*this) - This_t(op2)));
                ret.shape = shape;
                return ret;
            } else {
                throw std::runtime_error("shape error in imp -");
            }
        }
    }

    template <typename U>
    requires arithmetic<U> || std::derived_from<U, Type_t>
    void operator+=(const U& op2)
    {
        if constexpr (std::is_arithmetic_v<U>) {
            static_cast<cv::Mat&>(*this) += op2;
        } else if constexpr (std::derived_from<U, Type_t>) {
            if (shape == op2.shape) {
                static_cast<cv::Mat&>(*this) += This_t(op2);
            } else {
                throw std::runtime_error("shape error in imp +=");
            }
        }
    }

    template <typename U>
    requires arithmetic<U> || std::derived_from<U, Type_t>
    void operator-=(const U& op2)
    {
        if constexpr (std::is_arithmetic_v<U>) {
            static_cast<cv::Mat&>(*this) -= op2;
        } else if constexpr (std::derived_from<U, Type_t>) {
            if (shape == op2.shape) {
                static_cast<cv::Mat&>(*this) -= This_t(op2);
            } else {
                throw std::runtime_error("shape error in imp -=");
            }
        }
    }

    I norm() const
    {
        return cv::norm(static_cast<const cv::Mat&>(*this));
    }

    Type_t exp() const
    {
        cv::Mat _;
        const auto& self = static_cast<const cv::Mat&>(*this);
        cv::exp(self, _);

        Type_t ret { This_t(_) };
        ret.shape = shape;

        return ret;
    }

    template <typename D>
    Type_t power(D power) const
    {
        cv::Mat _;
        const auto& self = static_cast<const cv::Mat&>(*this);
        cv::pow(self, power, _);

        Type_t ret { This_t(_) };
        ret.shape = shape;

        return ret;
    }

    Type_t sqrt() const
    {
        cv::Mat _;
        const auto& self = static_cast<const cv::Mat&>(*this);
        cv::sqrt(self, _);

        Type_t ret { This_t(_) };
        ret.shape = shape;

        return ret;
    }

    template <typename U>
    requires arithmetic<U> || std::derived_from<U, Type_t>
        Type_t solve(const U& op2)
    const
    {
        cv::Mat _;
        const auto& m1 = static_cast<const cv::Mat&>(*this);
        const auto& m2 = static_cast<const cv::Mat&>(This_t(op2));

        cv::solve(m1, m2, _, cv::DECOMP_SVD);

        Type_t ret { This_t(_) };
        ret.shape = Shape(shape[1], 1);

        return ret;
    }

private:
    Shape<M> shape = {};
};

template <typename I>
class CVXarrayBaseImp<I, 0> {
};

}
#endif