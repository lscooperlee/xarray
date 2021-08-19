#ifndef DATA_STORAGE_H
#define DATA_STORAGE_H

#include "index.h"
#include "shape.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

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

    template <typename U>
    DataStorage(U&& u)
        : idata(std::make_shared<InternalData<T>>(std::forward<U>(u)))
    {
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

    T* data()
    {
        return (*idata).data();
    }

    const T* data() const
    {
        return (*idata).data();
    }

    size_type size() const
    {
        return (*idata).size();
    }

    template <int K, int... M>
    auto mask(const Index<M...>& index, Shape<K> shape) const
    {
        assert(sizeof...(M) <= shape.size());

        std::vector<std::vector<unsigned char>> mask = {};
        int count = 0;

        auto index1 = [&mask, &shape, &count](auto&& index1d) {
            auto start = index1d.start();
            auto stop = index1d.stop();
            auto step = index1d.step();

            auto stride = shape[count++];
            std::vector<unsigned char> idxidx(stride, 0);

            auto real_stop = stride < stop ? stride : stop;
            for (int i = start; i < real_stop; i += step) {
                idxidx[i] = 1;
            }

            mask.push_back(idxidx);
        };

        std::apply([&index1](auto&&... args) { (index1(args), ...); },
            index.data);

        for (int i = sizeof...(M); i < shape.size(); ++i) {
            std::vector<unsigned char> idxidx(shape[i], 0);
            for (int j = 0; j < shape[i]; ++j) {
                idxidx[j] = 1;
            }
            mask.push_back(idxidx);
        }

        return mask;
    }

    template <int K>
    auto gen_idx(auto idx_mask, Shape<K> shape) const
    {
        int total = shape.total();
        std::vector<int> indices(idx_mask.size(), 0);

        std::vector<int> stride(shape.size());
        int last = 1;
        stride[shape.size() - 1] = last;
        for (int i = 1; i < shape.size(); ++i) {
            last = shape[i] * last;
            stride[shape.size() - i - 1] = last;
        }

        std::vector<bool> ret = {};

        for (int j = 0; j < total; ++j) {

            int tmp = j;
            for (int i = shape.size() - 1; i >= 0; --i) {
                indices[i] = tmp % shape[i];
                tmp = tmp / shape[i];
            }

            int idxidx = 0;
            bool is_indexed = std::all_of(idx_mask.begin(), idx_mask.end(), [&idxidx, &indices](const auto& c) {
                return c[indices[idxidx++]];
            });

            ret.push_back(is_indexed);
        }

        return ret;
    }

    template <int K, int... M>
    auto get_shape(const Index<M...>& index, Shape<K> shape) const
    {
        std::vector<int> val = {};
        int count = 0;

        auto _get_shape = [&val, &shape, &count](auto&& index1d) {
            if (index1d.size != 1) {
                auto start = index1d.start();
                auto stop = index1d.stop();
                auto step = index1d.step();

                auto stride = shape[count];
                auto real_stop = stride < stop ? stride : stop;
                // std::cout << start << ", " << real_stop << ", " << step << std::endl;
                // std::cout << (real_stop - start) / step << std::endl;
                int sp = 0;
                for (int i = start; i < real_stop; i += step) {
                    sp++;
                }
                val.push_back(sp);
            }
            count++;
        };

        std::apply([&_get_shape](auto&&... args) { (_get_shape(args), ...); },
            index.data);

        for (int i = sizeof...(M); i < shape.size(); ++i) {
            val.push_back(shape[i]);
        }
        constexpr int m = get_shape_size<M...>() + shape.size() - sizeof...(M);

        if constexpr (m == 0) {
            return Shape<m>();
        } else if constexpr (m == 1) {
            return Shape(val[0]);
        } else if constexpr (m == 2) {
            return Shape(val[0], val[1]);
        } else {
            return Shape(val[0], val[1], val[2]);
        }
    }

    template <int K, int... M>
    DataStorage<T> copy(const Index<M...>& index, Shape<K> shape) const
    {
        auto msk = mask(index, shape);
        auto idx = gen_idx(msk, shape);

        std::vector<T> tmp = {};
        for (size_type i = 0; i < idata->size(); ++i) {
            if (idx[i]) {
                tmp.push_back((*this)[i]);
            }
        }

        return DataStorage<T>(tmp.data(), tmp.size());
    }

    std::shared_ptr<InternalData<T>> idata = {};

private:
    template <int K, int... M>
    static constexpr int get_shape_size()
    {
        if constexpr (sizeof...(M) == 0) {
            return K == 1 ? 0 : 1;
        } else {
            return get_shape_size<M...>() + K == 1 ? 0 : 1;
        }
    }
};

#endif