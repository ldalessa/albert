#pragma once

#include "albert/Index.hpp"
#include "albert/concepts.hpp"
#include "albert/utils/min.hpp"
#include <ce/cvector.hpp>
#include <utility>

namespace albert
{
    template <int N>
    struct TensorIndex
    {
        using tensor_index_tag = void;

        ce::cvector<char, N> is;

        constexpr TensorIndex() = default;

        template <char... Is>
        constexpr explicit TensorIndex(Index<Is...>)
        {
            (is.emplace_back(Is), ...);
            static_assert(N == sizeof...(Is));
        }

        constexpr auto operator[](int i) const -> decltype(auto)
        {
            return is[i];
        }

        constexpr auto operator[](int i) -> decltype(auto)
        {
            return is[i];
        }

        constexpr auto begin() const -> decltype(auto)
        {
            return std::begin(is);
        }

        constexpr auto end() const -> decltype(auto)
        {
            return std::end(is);
        }

        constexpr auto size() const -> decltype(auto)
        {
            return is.size();
        }

        constexpr auto index_of(char c) const -> int
        {
            for (int i = 0; i < N; ++i) {
                if (is[i] == c) {
                    return i;
                }
            }
            __builtin_abort(); // could not find char
        }

        constexpr auto count(char c) const -> int
        {
            int n = 0;
            for (char i : is) {
                n += c == i;
            }
            return n;
        }

        constexpr void push(char c)
        {
            is.push_back(c);
        }

        constexpr auto exclusive() const -> TensorIndex
        {
            TensorIndex b;
            for (char c : is) {
                if (c != projected_index_id and count(c) == 1) {
                    b.push(c);
                }
            }
            return b;
        }

        constexpr auto repeated() const -> TensorIndex
        {
            TensorIndex b;
            for (char c : is) {
                if (c != projected_index_id and count(c) > 1 and b.count(c) == 0) {
                    b.push(c);
                }
            }
            return b;
        }

        constexpr auto n_repeated() const -> int
        {
            return repeated().size();
        }

        constexpr auto projected() const -> TensorIndex
        {
            TensorIndex b;
            for (char c : is) {
                if (c == projected_index_id) {
                    b.push(c);
                }
            }
            return b;
        }

        constexpr auto n_projected() const -> int
        {
            int n = 0;
            for (char c : is) {
                n += (c == projected_index_id);
            }
            return n;
        }

        constexpr auto reverse() const -> TensorIndex
        {
            TensorIndex b;
            for (int i = 0, e = size(); i < e; ++i) {
                b.push(is[e - i - 1]);
            }
            return b;
        }
    };

    template <char... Is>
    TensorIndex(Index<Is...>) -> TensorIndex<sizeof...(Is)>;

    template <int N, int M>
    constexpr inline bool operator==(TensorIndex<N> const& a, TensorIndex<M> const& b)
    {
        return a.is == b.is;
    }

    template <int N, int M>
    constexpr inline auto operator<=>(TensorIndex<N> const& a, TensorIndex<M> const& b)
    {
        return a.is <=> b.is;
    }

    template <int N, int M>
    constexpr inline auto operator+(TensorIndex<N> const& a, TensorIndex<M> const& b)
        -> TensorIndex<N + M>
    {
        TensorIndex<N + M> out;
        for (char c : a) out.push(c);
        for (char c : b) out.push(c);
        return out;
    }

    template <int N, int M>
    constexpr inline auto operator-(TensorIndex<N> const& a, TensorIndex<M> const& b)
        -> TensorIndex<N>
    {
        TensorIndex<N> difference;
        for (char c : a) {
            if (b.count(c) == 0) {
                difference.push(c);
            }
        }
        return difference;
    }

    template <int N, int M>
    constexpr inline auto operator^(TensorIndex<N> const& a, TensorIndex<M> const& b)
        -> TensorIndex<N + M>
    {
        TensorIndex<N + M> disjoint_union;
        for (char c : a) {
            if (b.count(c) == 0) {
                disjoint_union.push(c);
            }
        }
        for (char c : b) {
            if (a.count(c) == 0) {
                disjoint_union.push(c);
            }
        }
        return disjoint_union;
    }

    template <int N, int M>
    constexpr inline auto operator&(TensorIndex<N> const& a, TensorIndex<M> const& b)
        -> TensorIndex<utils::min(N, M)>
    {
        TensorIndex<utils::min(N, M)> intersection;
        for (char c : a) {
            if (b.count(c) != 0) {
                intersection.push(c);
            }
        }
        return intersection;
    }

    template <int N, int M>
    constexpr inline auto is_permutation(TensorIndex<N> const& a, TensorIndex<M> const& b)
        -> bool
    {
        return (a - b).size() == 0 and (b - a).size() == 0;
    }
}
