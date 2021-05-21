#ifndef ALBERT_INCLUDE_ALBERT_HPP
#define ALBERT_INCLUDE_ALBERT_HPP

#include "albert/Index2.hpp"
#include <concepts>

#define FWD(x) static_cast<decltype(x)&&>(x)

namespace albert
{
  template <class T>
  inline constexpr auto outer_v = std::remove_cvref_t<T>::outer();

  template <class T>
  inline constexpr auto rank_v = [] {
    if constexpr (requires { std::remove_cvref_t<T>::rank(); }) {
      return std::remove_cvref_t<T>::rank();
    }
    else {
      return std::remove_cvref_t<T>::outer().size();
    }
  }();

  template <class T>
  concept is_tree_node = requires {
    typename std::remove_cvref_t<T>::tree_node_tag;
  };

  template <class T>
  concept is_binary_node = is_tree_node<T> and requires {
    typename std::remove_cvref_t<T>::binary_node_tag;
  };

  template <class T>
  concept is_unary_node = is_tree_node<T> and requires {
    typename std::remove_cvref_t<T>::unary_node_tag;
  };

  template <class T>
  concept is_leaf_node = is_tree_node<T> and requires {
    typename std::remove_cvref_t<T>::leaf_node_tag;
  };

  template <class T>
  concept is_tensor = is_leaf_node<T> and requires {
    typename std::remove_cvref_t<T>::tensor_tag;
  };

  template <class T>
  concept is_tree = is_tree_node<T> || std::integral<T> || std::floating_point<T>;

  template <class A, is_tensor_index auto index>
  struct Bind
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
  };

  template <class A, class B>
  struct Addition
  {
    using tree_node_tag = void;
    using binary_node_tag = void;

    A a;
    B b;

    constexpr Addition(A a, B b)
        : a(std::move(a))
        , b(std::move(b))
    {
      constexpr auto l = outer_v<A>;
      constexpr auto r = outer_v<B>;
      assert(l == r);
    }

    constexpr static auto outer()
    {
      return outer_v<A>;
    }
  };

  template <class A, class B>
  struct Sum : Addition<A, B>, Bindable<Sum<A, B>>
  {
    using Addition<A, B>::Addition;
  };

  template <class A, class B>
  Sum(A, B) -> Sum<A, B>;

  template <class A, class B>
  struct Diff : Addition<A, B>, Bindable<Diff<A, B>>
  {
    using Addition<A, B>::Addition;
  };

  template <class A, class B>
  Diff(A, B) -> Diff<A, B>;

  template <class A, class B>
  struct Contraction
  {
    using tree_node_tag = void;
    using binary_node_tag = void;

    A a;
    B b;

    constexpr Contraction(A a, B b)
        : a(std::move(a))
        , b(std::move(b))
    {
    }

    constexpr static auto outer()
    {
      constexpr TensorIndex a = outer_v<A>;
      constexpr TensorIndex b = outer_v<B>;
      constexpr TensorIndex c = a ^ b;
      return c;
    }
  };

  template <class A, class B>
  struct Product : Contraction<A, B>, Bindable<Product<A, B>>
  {
    using Contraction<A, B>::Contraction;
  };

  template <class A, class B>
  Product(A, B) -> Product<A, B>;

  template <class A, class B>
  struct Ratio : Contraction<A, B>
  {
    using Contraction<A, B>::Contraction;
  };

  template <class A, class B>
  Ratio(A, B) -> Ratio<A, B>;

  template <class A>
  struct Negate : Bindable<Negate<A>>
  {
    using tree_node_tag = void;
    using unary_node_tag = void;

    A a;

    constexpr Negate(A a)
        : a(std::move(a))
    {
    }

    constexpr static auto outer()
    {
      return outer_v<A>;
    }
  };

  template <class A, is_tensor_index auto index>
  struct Partial : Bindable<Partial<A, index>>
  {
    using tree_node_tag = void;
    using unary_node_tag = void;

    A a;

    template <index_t... Is>
    constexpr Partial(A a)
        : a(std::move(a))
    {
    }

    constexpr static auto outer()
    {
      return (outer_v<A> + index).exclusive();
    }
  };

  template <class T>
  struct Literal
  {
    using tree_node_tag = void;
    using leaf_node_tag = void;

    T x;

    constexpr Literal(T x)
        : x(x)
    {
    }

    constexpr static auto rank() -> std::size_t
    {
      return 0;
    }

    constexpr static auto outer() -> TensorIndex<0>
    {
      return {};
    }
  };

  template <TensorIndex<2> index>
  struct Delta
  {
    using tree_node_tag = void;
    using leaf_node_tag = void;

    constexpr static auto rank() -> std::size_t
    {
      return 2;
    }

    constexpr static auto outer() -> TensorIndex<2>
    {
      return index;
    }
  };

  template <is_tensor_index auto index>
  struct Epsilon
  {
    using tree_node_tag = void;
    using leaf_node_tag = void;

    constexpr static auto rank() -> std::size_t
    {
      return index.size();
    }

    constexpr static auto outer()
    {
      return index;
    }
  };

  template <class T, std::size_t Rank, std::size_t N>
  struct Tensor : Bindable<Tensor<T, Rank, N>>
  {
    using tensor_tag = void;
    using tree_node_tag = void;
    using leaf_node_tag = void;

    constexpr static std::size_t rank()
    {
      return Rank;
    }
  };

  template <is_tree A>
  constexpr auto promote(A&& a)
  {
    return FWD(a);
  }

  template <is_tensor A>
  constexpr auto promote(A&& a)
  {
    static_assert(rank_v<A> == 0);
    return FWD(a)();
  }

  constexpr auto promote(std::integral auto&& i)
  {
    return Literal(i);
  }

  constexpr auto promote(std::floating_point auto&& d)
  {
    return Literal(d);
  }

  template <is_tree A>
  constexpr auto operator+(A&& a)
  {
    return promote(FWD(a));
  }

  template <is_tree A, is_tree B>
  constexpr auto operator+(A&& a, B&& b)
  {
    return Sum { promote(FWD(a)), promote(FWD(b)) };
  }

  template <is_tree A>
  constexpr auto operator-(A&& a)
  {
    return Negate { promote(FWD(a)) };
  }

  template <is_tree A, is_tree B>
  constexpr auto operator-(A&& a, B&& b)
  {
    return Diff { promote(FWD(a)), promote(FWD(b)) };
  }

  template <is_tree A, is_tree B>
  constexpr auto operator*(A&& a, B&& b)
  {
    return Product { promote(FWD(a)), promote(FWD(b)) };
  }

  template <is_tree A, is_tree B>
  constexpr auto operator/(A&& a, B&& b)
  {
    return Ratio { promote(FWD(a)), promote(FWD(b)) };
  }

  template <is_tree A>
  constexpr auto D(A&& a, is_index auto i, is_index auto... is)
  {
    constexpr Index index = (i + ... + is);
    constexpr TensorIndex jindex(index);
    return Partial<decltype(promote(FWD(a))), jindex> { promote(FWD(a)) };
  }

  constexpr auto δ(is_index auto i, is_index auto j)
  {
    constexpr Index index = (j + j);
    constexpr TensorIndex jindex(index);
    return Delta<jindex>{};
  }

  constexpr auto ε(is_index auto i, is_index auto... is)
  {
    constexpr Index index = (i + ... + is);
    constexpr TensorIndex jindex(index);
    return Epsilon<jindex>{};
  }
}

#endif // ALBERT_INCLUDE_ALBERT_HPP
