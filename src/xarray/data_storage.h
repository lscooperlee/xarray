#ifndef DATA_STORAGE_H
#define DATA_STORAGE_H

#include "xarray/index.h"
#include "xarray/shape.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace xa {

template <int K, int... M>
static constexpr int get_shape_size()
{
    if constexpr (sizeof...(M) == 0) {
        return K == 1 ? 0 : 1;
    } else {
        constexpr auto tmp = (K == 1) ? 0 : 1;
        return get_shape_size<M...>() + tmp;
    }
}

template <int K, int... M>
static constexpr int get_final_shape_size()
{
    return get_shape_size<M...>() + K - sizeof...(M);
}

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

    DataStorage(const DataStorage<T, N...>& u)
        : idata(u.idata)
        , sdata(u.sdata)
    {
    }

    DataStorage<T, N...>& operator=(const DataStorage<T, N...>& u)
    {
        if (sdata) {
            assert((*sdata).size() == u.size());
            for (size_t i = 0; i < u.size(); ++i) {
                *(*sdata)[i] = u[i];
            }
        }
        idata = u.idata;

        return *this;
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
        if (sdata) {
            return *(*sdata)[idx];
        } else {
            return (*idata)[idx];
        }
    }

    const T& operator[](size_type idx) const
    {
        return (*const_cast<DataStorage<T, N...>*>(this))[idx];
    }

    auto* data()
    {
        if (sdata) {
            idata = std::make_shared<InternalData<T>>((*sdata).size());
            std::transform(sdata->begin(), sdata->end(), idata->begin(), [](const auto t) {
                return *t;
            });
            sdata.reset();
        }

        return (*idata).data();
    }

    const auto* data() const
    {
        return const_cast<DataStorage<T, N...>*>(this)->data();
    }

    size_type size() const
    {
        if (sdata) {
            return (*sdata).size();
        } else {
            return (*idata).size();
        }
    }

    template <int K, int... M>
    DataStorage<T> copy(const Index<M...>& index, Shape<K> shape) const
    {
        std::vector<T> tmp = {};
        for_all_indexed_item(index, shape, [&tmp, this](const auto idx) {
            tmp.push_back((*this)[idx]);
        });

        return DataStorage<T>(tmp.data(), tmp.size());
    }

    template <int K, int... M>
    DataStorage<T> copy(const Index<M...>& index, Shape<K> shape)
    {
        if (sdata) {

            std::vector<T*> tmp = {};
            for_all_indexed_item(index, shape, [&tmp, this](const auto idx) {
                tmp.push_back((*sdata)[idx]);
            });

            auto d = DataStorage<T>(*this);
            d.sdata = std::make_shared<InternalData<T*>>(tmp);

            return d;

        } else {

            std::vector<T*> tmp = {};
            for_all_indexed_item(index, shape, [&tmp, this](const auto idx) {
                tmp.push_back(&(*this)[idx]);
            });

            auto d = DataStorage<T>(*this);
            d.sdata = std::make_shared<InternalData<T*>>(tmp);

            return d;
        }
    }

    mutable std::shared_ptr<InternalData<T>> idata = {};
    mutable std::shared_ptr<InternalData<T*>> sdata = {};

    // private:
    template <int K, int... M>
    auto get_shape(const Index<M...>& index, Shape<K> shape) const
    {
        constexpr int m = get_final_shape_size<K, M...>();

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

        return Shape<m>(std::move(val));
    }

    template <int K>
    auto get_stride(Shape<K> shape) const
    {
        /**
         * A stride is for getting an actual distance for visiting a flatten array give a binary idx.
         * eg for a shape (4,3,2), stride would be like:
         * {6, 2, 1}
         * that is, for a binary index (3, 2, 1), the final stride is {3*6 + 2*2 + 1*0}
         */
        int total = shape.total();
        std::array<int, K> stride = {};
        for (int i = 0; i < shape.size(); ++i) {
            stride[i] = total / shape[i];
            total = stride[i];
        }

        return stride;
    }

    template <int K = 0, int E>
    auto loop(const auto& mask, std::array<int, E>& bin_idx, const auto& func) const
    {
        auto is_in_range = [](auto a, auto b, auto c) {
            if (a <= b) {
                return c >= a && c < b;
            } else {
                return c <= a && c > b;
            }
        };

        auto start = mask[K][0];
        auto stop = mask[K][1];
        auto step = mask[K][2];

        /**
         * Note the order has to be from first indexed to last indexed,
         * eg: bin_idx has to be 000, 001, 010, 011 for forward order
         * Then func could simply use push_back for keeping right order after filter.
        */
        if constexpr (K == E - 1) {
            for (int i = start; is_in_range(start, stop, i); i += step) {
                bin_idx[K] = i;
                func(bin_idx);
            }
        } else {
            for (int i = start; is_in_range(start, stop, i); i += step) {
                bin_idx[K] = i;
                loop<K + 1, E>(mask, bin_idx, func);
            }
        }
    }

    template <int K, int... M>
    auto get_mask(const Index<M...>& index, Shape<K> shape) const
    {
        /**
         * A mask is an array of {start, end, step} data filtered with index.
         * It extends the index to its full format.
         * eg for a shape (4,3,2) and index({0, 4}, {0, 3, 2}), mask would be like:
         * {
         *  {0, 4, 1},
         *  {0, 3, 2},
         *  {0, 2, 1},
         * }
         */
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

    template <int K, int... M>
    auto for_all_indexed_item(const Index<M...>& index, Shape<K> shape, const auto& func) const
    {
        auto msk = get_mask(index, shape);
        auto stride = get_stride(shape);

        std::array<int, shape.size()> targetidx = {};
        loop<0, shape.size()>(msk, targetidx, [&func, &stride](const auto& o /*o in out*/) {
            auto idx = std::transform_reduce(stride.begin(), stride.end(), o.begin(), 0);
            func(idx);
        });
    }
};
}

#endif