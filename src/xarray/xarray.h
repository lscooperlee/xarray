#ifndef XARRAY_H
#define XARRAY_H

#include "xarray/xarraybase.h"

namespace xa {

template <typename A, int N>
class Xarray : public XarrayBase<A, N> {
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

    Xarray(const std::initializer_list<A>& data_)
        : XarrayBase<A, N>(Shape(data_.size()), data_)
    {
    }

    template <template <typename> typename C>
    Xarray(Shape<N> shape, const C<A>& container)
        : XarrayBase<A, N>(shape, container)
    {
    }
};

template <typename A>
Xarray(const std::initializer_list<std::initializer_list<A>>& data_) -> Xarray<A, 2>;

template <typename A>
Xarray(const std::initializer_list<A>& data_) -> Xarray<A, 1>;

}

#endif