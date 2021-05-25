#ifndef ALBERT_INCLUDE_BIND_HPP
#define ALBERT_INCLUDE_BIND_HPP

#include "albert/Index.hpp"
#include "albert/ScalarIndex.hpp"
#include "albert/TensorIndex.hpp"
#include "albert/concepts.hpp"
#include "albert/traits.hpp"
#include "albert/utils.hpp"
#include <ce/cvector.hpp>
#include <utility>

namespace albert
{
  template <class T>
  struct Bindable;

  template <is_tree A, is_tensor_index auto index>
  struct Bind : Bindable<Bind<A, index>>
  {
    using tree_node_tag = void;
    using unary_node_tag = void;

    constexpr static int M = index.scalars().size();

    A a;
    ScalarIndex<M> _scalars;

    constexpr Bind(A a, ce::cvector<int, M> const& scalars, nttp_args<index>)
        : a(std::move(a))
        , _scalars(scalars)
    {
      constexpr auto l = rank_v<A>;
      constexpr auto r = index.size();
      static_assert(l == r);
    }

    constexpr static auto outer()
    {
      return index.exclusive();
    }

    constexpr static auto dim() -> int
    {
      return dim_v<A>;
    }

    /// Evaluate a bind node when there's a contraction and/or projection.
    constexpr auto evaluate(ScalarIndex<rank_v<Bind>> const& i) const
      -> auto requires(index.repeated().size() != 0 or
                       index.scalars().size() != 0)
    {
      constexpr TensorIndex  outer = index.exclusive();
      constexpr TensorIndex slices = index.scalars();
      constexpr TensorIndex  inner = index.repeated();
      constexpr TensorIndex    all = outer + slices + inner;
      constexpr int      N = dim();
      constexpr int   Rank = outer.size() + slices.size();
      constexpr int      I = inner.size();

      auto rhs = [&](auto const& i) {
        return a.evaluate(select<all, index>(i));
      };

      ScalarIndex<Rank + I> j(i + _scalars);
      decltype(rhs(j)) temp{};
      do {
        temp += rhs(j);
      } while (carry_sum_inc<N, Rank>(j));
      return temp;
    }

    constexpr auto evaluate(ScalarIndex<rank_v<Bind>> i) const -> decltype(auto)
      requires((index.scalars().size() + index.repeated().size()) == 0)
    {
      return a.evaluate(i);
    }

    constexpr auto evaluate(ScalarIndex<rank_v<Bind>> i) -> decltype(auto)
      requires((index.scalars().size() + index.repeated().size()) == 0)
    {
      return a.evaluate(i);
    }
  };

  template <class T>
  struct Bindable
  {
    template <is_index I, class... Is> requires (all_index<Is...>)
    constexpr auto operator()(I i, Is... is) const & -> decltype(auto)
    {
      constexpr cat_index_type_t<I, Is...> all = {};
      constexpr TensorIndex index(all);
      constexpr int n_scalars = index.scalars().size();
      ce::cvector<int, n_scalars> scalars;
      if (std::integral<I>) {
        scalars.push_back(i);
      }
      ([&] {
        if constexpr (std::integral<Is>) {
          scalars.push_back(is);
        }
      }(), ...);

      return Bind {
        *static_cast<T const*>(this),
        scalars,
        nttp_pack<index>
      };
    }

    template <is_index I, class... Is> requires (all_index<Is...>)
    constexpr auto operator()(I i, Is... is) && -> decltype(auto)
    {
      constexpr cat_index_type_t<I, Is...> all = {};
      constexpr TensorIndex index(all);
      constexpr int n_scalars = index.scalars().size();
      ce::cvector<int, n_scalars> scalars;
      if (std::integral<I>) {
        scalars.push_back(i);
      }
      ([&] {
        if constexpr (std::integral<Is>) {
          scalars.push_back(is);
        }
      }(), ...);

      return Bind {
        std::move(*static_cast<T*>(this)),
        scalars,
        nttp_pack<index>
      };
    }

    template <is_index I, class... Is> requires (all_index<Is...>)
    constexpr auto operator()(is_index auto i, Is... is) & -> decltype(auto)
    {
      constexpr cat_index_type_t<I, Is...> all = {};
      constexpr TensorIndex index(all);
      constexpr int n_scalars = index.scalars().size();
      ce::cvector<int, n_scalars> scalars;
      if (std::integral<I>) {
        scalars.push_back(i);
      }
      ([&] {
        if constexpr (std::integral<Is>) {
          scalars.push_back(is);
        }
      }(), ...);

      return Bind {
        *static_cast<T*>(this),
        scalars,
        nttp_pack<index>
      };
    }

    template <is_tensor_index auto index>
    constexpr auto rebind() const & -> decltype(auto)
    {
      return Bind { *static_cast<T const*>(this), {}, nttp_pack<index> };
    }

    template <is_tensor_index auto index>
    constexpr auto rebind() && -> decltype(auto)
    {
      return Bind { std::move(*static_cast<T*>(this)), {}, nttp_pack<index> };
    }

    template <is_tensor_index auto index>
    constexpr auto rebind() & -> decltype(auto)
    {
      return Bind{ *static_cast<T*>(this), {}, nttp_pack<index> };
    }
  };
}

#endif // ALBERT_INCLUDE_BIND_HPP
