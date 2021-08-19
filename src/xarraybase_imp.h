#ifndef XARRAYBASE_IMP_H
#define XARRAYBASE_IMP_H

#include <algorithm>
#include <iostream>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>
#include <random>

#include <opencv2/core.hpp>

#include "common.h"
#include "data_storage.h"
#include "shape.h"

template <typename S>
class CVXarrayBaseImp : public cv::Mat {
    using Type_t = typename S::base_class;
    using I = typename Type_t::value_type;

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
            //            static_assert(false);
            std::cout << "float double bool when create XarrayImp";
        }

        return type;
    }

public:
    template <int N>
    static Type_t rand(Shape<N> shape)
    {
        int rows, cols;
        if (shape.size() == 1) {
            rows = 1;
            cols = shape[0];
        } else {
            rows = shape[0];
            cols = shape[1];
        }

        cv::Mat m(rows, cols, get_type());
        cv::randu(m, cv::Scalar(0), cv::Scalar(1));

        Type_t ret { CVXarrayBaseImp<S>(m) };
        ret.shape = shape;

        return ret;
    }

    template <int N>
    static Type_t randn(Shape<N> shape)
    {
        int rows, cols;
        if (shape.size() == 1) {
            rows = 1;
            cols = shape[0];
        } else {
            rows = shape[0];
            cols = shape[1];
        }

        cv::Mat m(rows, cols, get_type());
        cv::randn(m, cv::Scalar(0), cv::Scalar(1));

        Type_t ret { CVXarrayBaseImp<S>(m) };
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

        int total;
        if (shape.size() == 1) {
            total = shape[0];
        } else {
            total = shape[0] * shape[1];
        }

        for (int i = 0; i < total; ++i) {
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
        int rows, cols;
        if (x.shape.size() == 1) {
            rows = 1;
            cols = x.shape[0];
        } else {
            rows = x.shape[0];
            cols = x.shape[1];
        }

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
        auto tmp = static_cast<const cv::Mat&>(*this) - CVXarrayBaseImp<S>(op1);
        return CVXarrayBaseImp<S>(tmp < 0.00001).all();
    }

    I det() const
    {
        return cv::determinant(static_cast<const cv::Mat&>(*this));
    }

    Type_t inv() const
    {
        Type_t ret(CVXarrayBaseImp<S>(static_cast<const cv::Mat*>(this)->inv(cv::DECOMP_SVD)));
        ret.shape = shape;
        return ret;
    }

    Type_t T() const
    {
        if (shape.size() == 1) {
            Type_t ret(CVXarrayBaseImp<S>(*this));
            return ret;
        } else {
            Type_t ret(CVXarrayBaseImp<S>(static_cast<const cv::Mat*>(this)->t()));
            ret.shape[0] = shape[1];
            ret.shape[1] = shape[0];
            return ret;
        }
    }

    template <typename U>
    requires std::derived_from<U, Type_t> I dot(const U& op2)
    const
    {
        auto is_vector_2d = (shape.size() == 2) && (op2.shape.size() == 2)
            && (shape == op2.shape) && (shape[0] == 1 || shape[1] == 1);

        auto is_vector_1d = (shape.size() == 1) && (op2.shape.size() == 1) && (shape == op2.shape);

        if (is_vector_2d || is_vector_1d) {

            I ret = static_cast<const cv::Mat*>(this)->dot(CVXarrayBaseImp<S>(op2));
            return ret;
        } else {
            throw std::runtime_error("shape error in dot");
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
            auto m2 = static_cast<const cv::Mat&>(CVXarrayBaseImp<S>(op2));

            cv::vconcat(m1, m2, _);

            Type_t ret { CVXarrayBaseImp<S>(_) };
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
            auto m2 = static_cast<const cv::Mat&>(CVXarrayBaseImp<S>(op2));

            cv::hconcat(m1, m2, _);

            Type_t ret { CVXarrayBaseImp<S>(_) };
            ret.shape = Shape(shape[0], op2.shape[1] + shape[1]);
            return ret;

        } else {
            throw std::runtime_error("shape error in dot");
        }
    }

    template <typename U>
    requires arithmetic<U> || std::derived_from<U, Type_t>
    bool operator<(const U& op2) const
    {
        if constexpr (std::is_arithmetic_v<U>) {
            const auto& self = static_cast<const cv::Mat&>(*this);
            // Type_t ret(CVXarrayBaseImp<S>(static_cast<const cv::Mat&>(*this) * op2));
            // ret.shape = shape;
            // return ret;
        } else if constexpr (std::derived_from<U, Type_t>) {
            // if ((shape.size() == 2)
            //     && (op2.shape.size() == 2)
            //     && (shape[1] == op2.shape[0])) {
            //     Type_t ret(CVXarrayBaseImp<S>(static_cast<const cv::Mat&>(*this) * CVXarrayBaseImp<S>(op2)));
            //     ret.shape = Shape(shape[0], op2.shape[1]);
            //     return ret;
            // } else if (shape == op2.shape) {

            //     Type_t ret(CVXarrayBaseImp<S>(this->mul(CVXarrayBaseImp<S>(op2))));
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
            Type_t ret(CVXarrayBaseImp<S>(static_cast<const cv::Mat&>(*this) * op2));
            ret.shape = shape;
            return ret;
        } else if constexpr (std::derived_from<U, Type_t>) {
            if ((shape.size() == 2)
                && (op2.shape.size() == 2)
                && (shape[1] == op2.shape[0])) {
                Type_t ret(CVXarrayBaseImp<S>(static_cast<const cv::Mat&>(*this) * CVXarrayBaseImp<S>(op2)));
                ret.shape = Shape(shape[0], op2.shape[1]);
                return ret;
            } else if (shape == op2.shape) {

                Type_t ret(CVXarrayBaseImp<S>(this->mul(CVXarrayBaseImp<S>(op2))));
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
            Type_t ret(CVXarrayBaseImp<S>(static_cast<const cv::Mat&>(*this) / op2));
            ret.shape = shape;
            return ret;
        } else if constexpr (std::derived_from<U, Type_t>) {
            if (shape == op2.shape) {
                Type_t ret(CVXarrayBaseImp<S>(static_cast<const cv::Mat&>(*this) - CVXarrayBaseImp<S>(op2)));
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
            Type_t ret(CVXarrayBaseImp<S>(static_cast<const cv::Mat&>(*this) + op2));
            ret.shape = shape;
            return ret;
        } else if constexpr (XBaseType<U>) {
            if (shape == op2.shape) {
                Type_t ret(CVXarrayBaseImp<S>(static_cast<const cv::Mat&>(*this) + CVXarrayBaseImp<S>(op2)));
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
            Type_t ret(CVXarrayBaseImp<S>(static_cast<const cv::Mat&>(*this) - op2));
            ret.shape = shape;
            return ret;
        } else if constexpr (std::derived_from<U, Type_t>) {
            if (shape == op2.shape) {
                Type_t ret(CVXarrayBaseImp<S>(static_cast<const cv::Mat&>(*this) - CVXarrayBaseImp<S>(op2)));
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
                static_cast<cv::Mat&>(*this) += CVXarrayBaseImp<S>(op2);
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
                static_cast<cv::Mat&>(*this) -= CVXarrayBaseImp<S>(op2);
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

        Type_t ret { CVXarrayBaseImp<S>(_) };
        ret.shape = shape;

        return ret;
    }

    template <typename D>
    Type_t power(D power) const
    {
        cv::Mat _;
        const auto& self = static_cast<const cv::Mat&>(*this);
        cv::pow(self, power, _);

        Type_t ret { CVXarrayBaseImp<S>(_) };
        ret.shape = shape;

        return ret;
    }

    Type_t sqrt() const
    {
        cv::Mat _;
        const auto& self = static_cast<const cv::Mat&>(*this);
        cv::sqrt(self, _);

        Type_t ret { CVXarrayBaseImp<S>(_) };
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
        const auto& m2 = static_cast<const cv::Mat&>(CVXarrayBaseImp<S>(op2));

        cv::solve(m1, m2, _, cv::DECOMP_SVD);

        Type_t ret { CVXarrayBaseImp<S>(_) };
        ret.shape = Shape(shape[1], 1);

        return ret;
    }

private:
    Shape<Type_t::shape_size> shape = {};
};

#endif