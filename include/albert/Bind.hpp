#ifndef ALBERT_INCLUDE_BIND_HPP
#define ALBERT_INCLUDE_BIND_HPP

#include "albert/Index.hpp"
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
  };

  template <class T>
  struct Bindable
  {
    constexpr auto operator()(is_index auto... is) const & -> decltype(auto)
    {
      constexpr Index all = (is + ... + Index<>{});
      constexpr TensorIndex index(all);
      return Bind<T const, index>{*static_cast<T const*>(this)};
    }

    constexpr auto operator()(is_index auto... is) && -> decltype(auto)
    {
      constexpr Index all = (is + ... + Index<>{});
      constexpr TensorIndex index(all);
      return Bind<T, index>{std::move(*static_cast<T*>(this))};
    }

    constexpr auto operator()(is_index auto... is) & -> decltype(auto)
    {
      constexpr Index all = (is + ... + Index<>{});
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
