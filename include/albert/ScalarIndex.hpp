#ifndef ALBERT_INCLUDE_SCALAR_INDEX_HPP
#define ALBERT_INCLUDE_SCALAR_INDEX_HPP

#include "albert/concepts.hpp"
#include "albert/traits.hpp"
#include <array>

namespace albert
{
  template <std::size_t Rank>
  struct ScalarIndex
  {
    std::array<int, Rank> _data = {};

    constexpr ScalarIndex() = default;

    constexpr static auto size() -> std::size_t
    {
      return Rank;
    }

    template <std::size_t O>
    requires (O < Rank)
      constexpr ScalarIndex(ScalarIndex<O> const& b)
    {
      for (int i = 0; i < O; ++i) {
        _data[i] = b[i];
      }
    }

    constexpr auto operator[](std::size_t i) const -> int
    {
      assert(i < Rank);
      return _data[i];
    }

    constexpr auto operator[](std::size_t i) -> int&
    {
      assert(i < Rank);
      return _data[i];
    }

    template <is_tensor_index auto from, is_tensor_index auto to>
    constexpr friend auto select(ScalarIndex const& in)
      -> ScalarIndex<to.size()>
    {
      static_assert(from.size() == size());
      static_assert(from.repeated().size() == 0);
      constexpr std::array<int, to.size()> map = [] {
        std::array<int, to.size()> map;
        for (int i = 0; i < to.size(); ++i) {
          map[i] = from.index_of(to[i]);
        }
        return map;
      }();

      ScalarIndex<to.size()> out;
      for (int i = 0; i < map.size(); ++i) {
        out[i] = in[map[i]];
      }
      return out;
    }

    template <std::size_t N, std::size_t n = 0>
    constexpr friend bool carry_sum_inc(ScalarIndex& index)
    {
      for (int i = n; i < Rank; ++i) {
        if (++index[i] < N) {
          return true;                          // no carry
        }
        index[i] = 0;                           // reset and carry
      }
      return false;                             // overflow
    }
  };
}

#endif // ALBERT_INCLUDE_SCALAR_INDEX_HPP
