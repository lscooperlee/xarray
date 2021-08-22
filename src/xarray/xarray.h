#ifndef XARRAY_H
#define XARRAY_H

#include "xarray/xarraybase.h"

namespace xa
{


template <typename A, int N>
class Xarray: public XarrayBase<A, N> {
public:
    using XarrayBase<A, N>::XarrayBase;

    Xarray(const std::initializer_list<std::initializer_list<A>>& data_)
    {
        this->shape[0] = data_.size();
        for (const auto& l1 : data_) {
            this->shape[1] = l1.size();
            for (auto l2 : l1) {
                this->data_storage.append(l2);
            }
        }
    }

    Xarray(const std::initializer_list<A>& data_):XarrayBase<A, N>(Shape(data_.size()), data_)
    {
    }

    // template <int... M>
    // auto operator[](const Index<M...>& idx) const
    // {
    //     auto new_shape = this->data_storage.get_shape(idx, this->shape);
    //     auto new_data = this->data_storage.copy(idx, this->shape);
    //     if constexpr (new_shape.size() == 0) {
    //         assert (new_data.size() == 1);
    //         return new_data[0];
    //     } else {
    //         return Xarray<A, new_shape.size()>(new_shape, std::move(new_data));
    //     }
    // }
};

template <typename A>
Xarray(const std::initializer_list<std::initializer_list<A>>& data_) -> Xarray<A, 2>;

template <typename A>
Xarray(const std::initializer_list<A>& data_) -> Xarray<A, 1>;

}

#endif