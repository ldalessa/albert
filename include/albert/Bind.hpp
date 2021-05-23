#ifndef ALBERT_INCLUDE_BIND_HPP
#define ALBERT_INCLUDE_BIND_HPP

#include "albert/Index.hpp"
#include "albert/ScalarIndex.hpp"
#include "albert/concepts.hpp"
#include "albert/traits.hpp"
#include <utility>

namespace albert
{
  template <class T>
  struct Bindable;

  template <class A, is_tensor_index auto index>
  struct Bind : Bindable<Bind<A, index>>
  {
    using tree_node_tag = void;
    using unary_node_tag = void;

    A a;

    constexpr Bind(A a)
        : a(std::move(a))
    {
      constexpr auto l = rank_v<A>;
      constexpr auto r = index.exclusive().size();
      static_assert(l == r);
    }

    constexpr static auto outer()
    {
      return index.exclusive();
    }

    constexpr static auto dim() -> std::size_t
    {
      return dim_v<A>;
    }

    constexpr auto evaluate(ScalarIndex<rank_v<Bind>> const& i) const
      -> auto requires(index.repeated().size() != 0)
    {
      constexpr TensorIndex outer = index.exclusive();
      constexpr TensorIndex inner = index.repeated();
      constexpr TensorIndex   all = outer + inner;
      constexpr std::size_t     N = dim();
      constexpr std::size_t  Rank = outer.size();
      constexpr std::size_t     I = inner.rank();
      static_assert(Rank + I == rank_v<A>);

      auto rhs = [&](auto const& i) {
        return a.evaluate(select<all, index>(i));
      };

      ScalarIndex<Rank + I> j(i);
      decltype(rhs(j)) temp{};
      do {
        temp += rhs(j);
      } while (carry_sum_inc<N, Rank>(j));
      return temp;
    }

    constexpr auto evaluate(ScalarIndex<rank_v<Bind>> i) const -> decltype(auto)
      requires(index.repeated().size() == 0)
    {
      return a.evaluate(i);
    }

    constexpr auto evaluate(ScalarIndex<rank_v<Bind>> i) -> decltype(auto)
      requires(index.repeated().size() == 0)
    {
      return a.evaluate(i);
    }
  };

  template <class T>
  struct Bindable
  {
    constexpr auto operator()(is_index auto i, is_index auto... is) const & -> decltype(auto)
    {
      constexpr Index all = (i + ... + is);
      constexpr TensorIndex index(all);
      return Bind<T const, index>{*static_cast<T const*>(this)};
    }

    constexpr auto operator()(is_index auto i, is_index auto... is) && -> decltype(auto)
    {
      constexpr Index all = (i + ... + is);
      constexpr TensorIndex index(all);
      return Bind<T, index>{std::move(*static_cast<T*>(this))};
    }

    constexpr auto operator()(is_index auto i, is_index auto... is) & -> decltype(auto)
    {
      constexpr Index all = (i + ... + is);
      constexpr TensorIndex index(all);
      return Bind<T, index>{*static_cast<T*>(this)};
    }

    template <is_tensor_index auto index>
    constexpr auto rebind() const & -> decltype(auto)
    {
      return Bind<T const, index>{*static_cast<T const*>(this)};
    }

    template <is_tensor_index auto index>
    constexpr auto rebind() && -> decltype(auto)
    {
      return Bind<T, index>{std::move(*static_cast<T*>(this))};
    }

    template <is_tensor_index auto index>
    constexpr auto rebind() & -> decltype(auto)
    {
      return Bind<T, index>{*static_cast<T*>(this)};
    }
  };
}

#endif // ALBERT_INCLUDE_BIND_HPP
