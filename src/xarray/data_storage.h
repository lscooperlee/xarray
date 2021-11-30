#ifndef DATA_STORAGE_H
#define DATA_STORAGE_H

#include "xarray/index.h"
#include "xarray/shape.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace xa {

template <typename T, int... N>
class DataStorage {
public:
    template <typename U>
    using InternalData = std::conditional_t<std::is_same_v<U, bool>, std::vector<unsigned char>, std::vector<U>>;
    using size_type = typename InternalData<T>::size_type;

    DataStorage(const T* start, unsigned int len)
        : idata(std::make_shared<InternalData<T>>(start, start + len))
    {
    }

    DataStorage(unsigned int size, T t)
        : idata(std::make_shared<InternalData<T>>(size, t))
    {
    }

    template <template <class, class...> class U>
    DataStorage(const U<T>& u)
        : idata(std::make_shared<InternalData<T>>(u))
    {
    }

    template <int... K>
    DataStorage(const DataStorage<T, K...>& u)
        : idata(u.idata)
    {
    }

    template <typename U>
    DataStorage(const DataStorage<U>& u)
        : idata(std::make_shared<InternalData<T>>(u.size()))
    {
        std::copy(u.idata->begin(), u.idata->end(), idata->begin());
    }

    DataStorage()
        : idata(std::make_shared<InternalData<T>>())
    {
    }

    template <typename U>
    void append(U&& u)
    {
        (*idata).push_back(std::forward<U>(u));
    }

    T& operator[](size_type idx)
    {
        return (*idata)[idx];
    }

    const T& operator[](size_type idx) const
    {
        return (*idata)[idx];
    }

    auto* data()
    {
        return (*idata).data();
    }

    const auto* data() const
    {
        return (*idata).data();
    }

    size_type size() const
    {
        return (*idata).size();
    }

    // template <typename U>
    // DataStorage<T> as_type(const DataStorage<U, K...>& storage) const
    // {
    //     auto new_idata = std::make_shared<InternalData<U>>(storage.size());
    //     std::copy(idata->begin(), idata->end(), new_idata->begin());

    //     DataStorage
    // }

    template <int K, int... M>
    DataStorage<T> copy(const Index<M...>& index, Shape<K> shape) const
    {
        auto msk = get_mask(index, shape);
        auto stride = get_stride(shape);

        std::vector<T> tmp = {};
        std::array<int, msk.size()> out = {};
        loop<msk.size()>(msk, out, [&stride, &tmp, this](const auto& o) {
            auto idx = std::transform_reduce(stride.begin(), stride.end(), o.begin(), 0);
            tmp.push_back((*this)[idx]);
        });

        return DataStorage<T>(tmp.data(), tmp.size());
    }

    template <int K, int... M>
    auto get_mask(const Index<M...>& index, Shape<K> shape) const
    {
        std::array<std::array<int, 3>, shape.size()> ret = {};

        int count = 0;
        auto fill = [&ret, &count, &shape](auto&& idx1d) {
            ret[count][0] = idx1d.start(shape[count]);
            ret[count][1] = idx1d.stop(shape[count]);
            ret[count][2] = idx1d.step();
            count++;
        };
        std::apply([&fill](auto&&... args) { (fill(args), ...); }, index.data);

        for (int i = sizeof...(M); i < shape.size(); ++i) {
            ret[i][0] = 0;
            ret[i][1] = shape[i];
            ret[i][2] = 1;
        }

        return ret;
    }

    template <int K>
    auto loop(const auto& mask, auto& out, const auto& func) const
    {
        auto is_in_range = [](auto a, auto b, auto c) {
            if (a <= b) {
                return c >= a && c < b;
            } else {
                return c <= a && c > b;
            }
        };

        auto start = mask[K - 1][0];
        auto stop = mask[K - 1][1];
        auto step = mask[K - 1][2];

        if constexpr (K == 1) {
            for (int i = start; is_in_range(start, stop, i); i += step) {
                out[K - 1] = i;
                func(out);
            }
        } else {
            for (int i = start; is_in_range(start, stop, i); i += step) {
                out[K - 1] = i;
                loop<K - 1>(mask, out, func);
            }
        }
    }

    template <int K>
    auto get_stride(Shape<K> shape) const
    {
        int total = shape.total();
        std::array<int, K> stride = {};
        for (int i = 0; i < shape.size(); ++i) {
            stride[i] = total / shape[i];
            total = stride[i];
        }

        return stride;
    }

    template <int K, int... M>
    auto get_shape(const Index<M...>& index, Shape<K> shape) const
    {
        constexpr int m = get_shape_size<M...>() + shape.size() - sizeof...(M);

        std::array<int, m> val = {};
        int val_idx = 0;
        int count = 0;

        auto _get_shape = [&val, &val_idx, &count, &shape](auto&& idx1d) {
            if (idx1d.size != 1) {

                auto is_in_range = [](auto a, auto b, auto c) {
                    if (a <= b) {
                        return c >= a && c < b;
                    } else {
                        return c <= a && c > b;
                    }
                };

                auto start = idx1d.start(shape[count]);
                auto stop = idx1d.stop(shape[count]);
                auto step = idx1d.step();

                int sp = 0;
                for (int i = start; is_in_range(start, stop, i); i += step) {
                    sp++;
                }

                val[val_idx++] = sp;
            }

            count++;
        };

        std::apply([&_get_shape](auto&&... args) { (_get_shape(args), ...); }, index.data);

        for (int i = sizeof...(M); i < shape.size(); ++i) {
            val[val_idx++] = shape[i];
        }

        return Shape<m>(val);
    }

    std::shared_ptr<InternalData<T>> idata = {};

private:
    template <int K, int... M>
    static constexpr int get_shape_size()
    {
        if constexpr (sizeof...(M) == 0) {
            return K == 1 ? 0 : 1;
        } else {
            constexpr auto tmp = K == 1 ? 0 : 1;
            return get_shape_size<M...>() + tmp;
        }
    }
};

}

#endif