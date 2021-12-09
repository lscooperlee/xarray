#ifndef XMATRIX_H
#define XMATRIX_H

#include "xarray/xarraybase.h"

namespace xa {

template <typename A>
class Xmatrix : public XarrayBase<A, 2> {
public:
    using XarrayBase<A, 2>::XarrayBase;

    Xmatrix(const std::initializer_list<std::initializer_list<A>>& data_)
    {
        this->shape[0] = data_.size();
        for (const auto& l1 : data_) {
            this->shape[1] = l1.size();
            for (auto l2 : l1) {
                this->data_storage.append(l2);
            }
        }
    }

    template <int... M>
    auto operator[](const Index<M...>& idx) const
    {
        auto new_shape = this->data_storage.get_shape(idx, this->shape);
        auto new_data = this->data_storage.copy(idx, this->shape);
        if constexpr (new_shape.size() == 0) {
            return Xmatrix<A>(Shape({ 1, 1 }), std::move(new_data));
        } else if constexpr (new_shape.size() == 1) {
            if constexpr (get_first_num<M...>() == 1) {
                return Xmatrix<A>(Shape({ 1, new_shape[0] }), std::move(new_data));
            } else {
                return Xmatrix<A>(Shape({ new_shape[0], 1 }), std::move(new_data));
            }
        } else if constexpr (new_shape.size() == 2) {
            return Xmatrix<A>(new_shape, std::move(new_data));
        }
    }

    template <typename U>
    auto dot(const U& op2) const
    {
        if (this->shape == op2.shape) {
            int sp = this->shape[0] == 1 ? this->shape[1] : this->shape[0];
            Shape<1> new_shape(sp);

            auto t1 = XarrayBase<A, 1>(new_shape, this->data_storage);
            auto t2 = XarrayBase<A, 1>(new_shape, op2.data_storage);
            return t1.dot(t2);
        } else {
            throw std::runtime_error("matrix dot, shape wrong");
        }
    }

private:
    template <int K, int M>
    static constexpr int get_first_num()
    {
        return K;
    }
};

}

#endif