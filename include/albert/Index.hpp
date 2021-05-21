#ifndef ALBERT_INCLUDE_INDEX_HPP
#define ALBERT_INCLUDE_INDEX_HPP

#include <ce/cvector.hpp>
#include <ce/concepts.hpp>
#include <concepts>
#include <utility>

namespace albert
{
  template <class T>
  concept is_index = std::same_as<T, int> or requires {
    typename std::remove_cvref_t<T>::index_tag;
  };

  enum index_t : char32_t {};

  template <index_t... Is>
  struct Index
  {
    using index_tag = void;

    constexpr static auto is = { Is... };

    constexpr auto begin() const -> decltype(auto)
    {
      return std::begin(is);
    }

    constexpr auto end() const -> decltype(auto)
    {
      return std::end(is);
    }

    constexpr friend auto operator<=>(Index, Index) = default;

    template <auto op>
    constexpr static auto to_index = []
    {
      constexpr ce::is_cvector auto a = op();
      return [&]<std::size_t... i>(std::index_sequence<i...>) {
        return Index<a[i]...>();
      }(std::make_index_sequence<a.size()>{});
    };

    constexpr static auto size() -> std::size_t
    {
      return sizeof...(Is);
    }

    constexpr static auto count(index_t c) -> std::size_t
    {
      return ((c == Is) + ...);
    }

    constexpr static auto exclusive()
    {
      return to_index<[]
      {
        ce::cvector<index_t, size()> b;
        for (index_t a : is) {
          if (count(a) == 1) {
            b.push_back(a);
          }
        }
        return b;
      }>();
    }

    constexpr static auto repeated()
    {
      return to_index<[]
      {
        ce::cvector<index_t, size()> b;
        for (index_t a : is) {
          if ((count(a) > 1)) {
            b.push_back(a);
          }
        }
        return b;
      }>();
    }

    template <index_t... Js>
    constexpr auto operator+(Index<Js...>) const -> Index<Is..., Js...>
    {
      return {};
    }

    constexpr auto operator+(std::same_as<int> auto) const -> Index
    {
      return {};
    }

    constexpr auto operator-(is_index auto b) const
    {
      return to_index<[b]
      {
        ce::cvector<index_t, size()> difference;
        for (index_t a : is) {
          if (b.count(a) == 0) {
            difference.push_back(a);
          }
        }
        return difference;
      }>();
    }

    constexpr auto operator^(is_index auto b) const
    {
      return to_index<[b]
      {
        ce::cvector<index_t, size() + b.size()> disjoint_union;
        for (index_t c : is) {
          if (b.count(c) == 0) {
            disjoint_union.push_back(c);
          }
        }
        for (index_t c : b) {
          if (count(c) == 0) {
            disjoint_union.push_back(c);
          }
        }
        return disjoint_union;
      }>();
    }

    constexpr auto operator&(is_index auto b) const
    {
      return to_index<[b]
      {
        ce::cvector<index_t, b.size()> intersection;
        for (index_t c : is) {
          if (b.count(c) != 0) {
            intersection.push_back(c);
          }
        }
        return intersection;
      }>();
    }
  };

  template <char I>
  constexpr inline Index<index_t(I)> index = {};
}

#endif // ALBERT_INCLUDE_INDEX_HPP
