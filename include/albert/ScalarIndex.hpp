#ifndef ALBERT_INCLUDE_SCALAR_INDEX_HPP
#define ALBERT_INCLUDE_SCALAR_INDEX_HPP

#include "albert/concepts.hpp"
#include "albert/utils.hpp"
#include <ce/cvector.hpp>

namespace albert
{
  /// A scalar index represents a runtime tensor index.
  ///
  /// Scalar indices are stored "big endian" in the sense that the outermost
  /// index is at offset 0.
  template <int Order>
  struct ScalarIndex
  {
    ce::cvector<int, Order> _data = { Order };

    constexpr ScalarIndex() = default;

    template <int B>
    constexpr ScalarIndex(ScalarIndex<B> const& b)
    {
      std::copy_n(b.begin(), b.size(), _data.begin());
    }

    template <int B>
    constexpr ScalarIndex(ce::cvector<int, B> const& b)
    {
      std::copy_n(b.begin(), b.size(), _data.begin());
    }

    constexpr ScalarIndex(std::same_as<int> auto... is)
        : _data { std::in_place, is... }
    {
    }

    constexpr static auto size() -> int
    {
      return Order;
    }

    constexpr auto begin() const -> decltype(auto)
    {
      return _data.begin();
    }

    constexpr auto end() const -> decltype(auto)
    {
      return _data.end();
    }

    constexpr auto operator[](int i) const -> decltype(auto)
    {
      return _data[i];
    }

    constexpr auto operator[](int i) -> decltype(auto)
    {
      return _data[i];
    }

    template <is_tensor_index auto from, is_tensor_index auto to>
    constexpr auto select(nttp_args<from, to>) const -> ScalarIndex<to.size()>
    {
      static_assert(from.size() == size());
      constexpr int A = to.size();
      constexpr ce::cvector map = []
      {
        // We use `j` to keep track of anonymous index matching.
        auto index_of = [j=0](char c) mutable {
          if (c != '\0') {
            return from.index_of(c);
          }
          for (int i = 0, k = j++; i < from.size(); ++i) {
            if (from[i] == '\0' and 0 == k--) {
              return i;
            }
          }
          __builtin_abort();
        };

        ce::cvector<int, A> map;
        for (char c : to) {
          map.push_back(index_of(c));
        }
        return map;
      }();

      ScalarIndex<A> out;
      for (int i = 0; i < A; ++i) {
        out[i] = _data[map[i]];
      }
      return out;
    }

    template <is_tensor_index auto from, is_tensor_index auto to>
    constexpr friend auto select(ScalarIndex const& in) -> ScalarIndex<to.size()>
    {
      return in.select(nttp<from, to>);
    }

    template <int N, int n = 0>
    constexpr friend bool carry_sum_inc(ScalarIndex& index)
    {
      for (int i = n; i < Order; ++i) {
        if (++index[i] < N) {
          return true;                          // no carry
        }
        index[i] = 0;                           // reset and carry
      }
      return false;                             // overflow
    }
  };

  /// Infer the ScalarIndex Order for this constructor.
  ScalarIndex(std::same_as<int> auto... is) -> ScalarIndex<sizeof...(is)>;

  /// Non-friend operator+ because it reduces the number of instances that the
  /// compiler needs to generate.
  template <int A, int B>
  constexpr auto operator+(ScalarIndex<A> const& a, ScalarIndex<B> const& b)
    -> ScalarIndex<A + B>
  {
    ScalarIndex<A + B> c;
    int i = 0;
    for (int a : a) c[i++] = a;
    for (int b : b) c[i++] = b;
    return c;
  }
}

#endif // ALBERT_INCLUDE_SCALAR_INDEX_HPP
