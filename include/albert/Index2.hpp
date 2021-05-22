#ifndef ALBERT_INCLUDE_INDEX_HPP
#define ALBERT_INCLUDE_INDEX_HPP

#include <ce/cvector.hpp>
#include <ce/concepts.hpp>
#include <array>
#include <concepts>
#include <utility>

namespace albert
{
  // enum index_t : char32_t {};

  enum index_t : char {};

  template <char32_t... Is>
  struct Index
  {
    using index_tag = void;
  };

  template <char32_t... As, char32_t... Bs>
  constexpr inline auto operator+(Index<As...>, Index<Bs...>)
    -> Index<As..., Bs...>
  {
    return {};
  }

  template <class T>
  concept is_index = requires {
    typename std::remove_cvref_t<T>::index_tag;
  };

  template <class T>
  concept is_tensor_index = requires {
    typename std::remove_cvref_t<T>::tensor_index_tag;
  };

  template <std::size_t N>
  struct TensorIndex
  {
    using tensor_index_tag = void;

    ce::cvector<index_t, N> is;

    constexpr TensorIndex() = default;

    template <char32_t... Is>
    constexpr explicit TensorIndex(Index<Is...>)
        : is { std::in_place, index_t(Is)... }
    {
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

    constexpr auto count(index_t c) const -> std::size_t
    {
      std::size_t n = 0;
      for (index_t i : is) {
        n += c == i;
      }
      return n;
    }

    constexpr void push(index_t c)
    {
      is.push_back(c);
    }

    constexpr auto exclusive() const -> TensorIndex
    {
      TensorIndex b;
      for (index_t a : is) {
        if (count(a) == 1) {
          b.push(a);
        }
      }
      return b;
    }

    constexpr auto repeated() const -> TensorIndex
    {
      TensorIndex b;
      for (index_t a : is) {
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

  template <char32_t... Is>
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
    for (index_t c : a) out.push(c);
    for (index_t c : b) out.push(c);
    return out;
  }

  // template <std::size_t N, std::size_t M>
  // constexpr inline auto operator-(TensorIndex<N> a, TensorIndex<M> b)
  //   -> TensorIndex<N>
  // {
  //   TensorIndex<N> difference;
  //   for (index_t c : a) {
  //     if (b.count(c) == 0) {
  //       difference.push(c);
  //     }
  //   }
  //   return difference;
  // }

  template <std::size_t N, std::size_t M>
  constexpr inline auto operator^(TensorIndex<N> a, TensorIndex<M> b)
    -> TensorIndex<N + M>
  {
    TensorIndex<N + M> disjoint_union;
    for (index_t c : a) {
      if (b.count(c) == 0) {
        disjoint_union.push(c);
      }
    }
    for (index_t c : b) {
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
  //   for (index_t c : a) {
  //     if (b.count(c) != 0) {
  //       intersection.push(c);
  //     }
  //   }
  //   return intersection;
  // }

}

#endif // ALBERT_INCLUDE_INDEX_HPP
