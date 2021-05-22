#ifndef ALBERT_INCLUDE_INDEX_HPP
#define ALBERT_INCLUDE_INDEX_HPP

#include "albert/concepts.hpp"
#include <ce/cvector.hpp>
#include <array>
#include <concepts>
#include <utility>

namespace albert
{
  template <char... Is>
  struct Index
  {
    using index_tag = void;
  };

  template <char... As, char... Bs>
  constexpr inline auto operator+(Index<As...>, Index<Bs...>)
    -> Index<As..., Bs...>
  {
    return {};
  }

  template <std::size_t N>
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

    constexpr auto begin() const -> decltype(auto)
    {
      return std::begin(is);
    }

    constexpr auto end() const -> decltype(auto)
    {
      return std::end(is);
    }

    constexpr auto size() const -> std::size_t
    {
      return is.size();
    }

    constexpr auto count(char c) const -> std::size_t
    {
      std::size_t n = 0;
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
      for (char a : is) {
        if (count(a) == 1) {
          b.push(a);
        }
      }
      return b;
    }

    constexpr auto repeated() const -> TensorIndex
    {
      TensorIndex b;
      for (char a : is) {
        if (count(a) > 1) {
          b.push(a);
        }
      }
      return b;
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

  template <std::size_t N, std::size_t M>
  constexpr inline auto operator==(TensorIndex<N> a, TensorIndex<M> b)
  {
    return a.is == b.is;
  }

  template <std::size_t N, std::size_t M>
  constexpr inline auto operator<=>(TensorIndex<N> a, TensorIndex<M> b)
  {
    return a.is <=> b.is;
  }

  template <std::size_t N, std::size_t M>
  constexpr inline auto operator+(TensorIndex<N> a, TensorIndex<M> b)
    -> TensorIndex<N + M>
  {
    TensorIndex<N + M> out;
    for (char c : a) out.push(c);
    for (char c : b) out.push(c);
    return out;
  }

  template <std::size_t N, std::size_t M>
  constexpr inline auto operator-(TensorIndex<N> a, TensorIndex<M> b)
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

  template <std::size_t N, std::size_t M>
  constexpr inline auto operator^(TensorIndex<N> a, TensorIndex<M> b)
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

  // constexpr inline auto operator&(is_tensor_index auto a, is_tensor_index auto b)
  //   TensorIndex<std::min(a.size(), b.size())>
  // {
  //   TensorIndex<std::min(a.size(), b.size())> intersection;
  //   for (char c : a) {
  //     if (b.count(c) != 0) {
  //       intersection.push(c);
  //     }
  //   }
  //   return intersection;
  // }

  template <std::size_t N, std::size_t M>
  constexpr inline auto is_permutation(TensorIndex<N> a, TensorIndex<M> b)
    -> bool
  {
    return (a - b).size() == 0 and (b - a).size() == 0;
  }
}

#endif // ALBERT_INCLUDE_INDEX_HPP
