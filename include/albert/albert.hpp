#ifndef ALBERT_INCLUDE_ALBERT_HPP
#define ALBERT_INCLUDE_ALBERT_HPP

#include "albert/cmath.hpp"
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

  template <class A, CMathTag tag>
  struct CMath : Bindable<CMath<A, tag>>
  {
    static_assert(!has_immediate<tag>);
    using tree_node_tag = void;
    using unary_node_tag = void;

    A a;

    constexpr CMath(A a, cmath_tag<tag>)
        : a(std::move(a))
    {
    }

    constexpr static auto outer()
    {
      return outer_v<A>;
    }
  };

  template <class A, class B, CMathTag tag>
  struct CMath2 : Bindable<CMath<A, tag>>
  {
    using tree_node_tag = void;
    using binary_node_tag = void;

    A a;
    B b;

    constexpr CMath2(A a, B b, cmath_tag<tag>)
        : a(std::move(a))
        , b(std::move(b))
    {
    }

    constexpr static auto outer()
    {
      return outer_v<A>;
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
  struct Delta : Bindable<Delta<index>>
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
  struct Epsilon : Bindable<Delta<index>>
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
    return Delta<jindex> {};
  }

  constexpr auto ε(is_index auto i, is_index auto... is)
  {
    constexpr Index index = (i + ... + is);
    constexpr TensorIndex jindex(index);
    return Epsilon<jindex> {};
  }

  template <is_tree A>
  constexpr auto symmetrize(A&& a)
  {
    auto&& b = promote(FWD(a));
    constexpr TensorIndex j = std::remove_cvref_t<decltype(b)>::outer().reverse();
    return promote(1) / promote(2) * (b + b.template rebind<j>());
  }

  template <is_tree A, is_tree B>
  constexpr auto fmin(A&& a, B&& b)
  {
    assert(a.rank() == 0);
    return CMath2(promote(FWD(a)), promote(FWD(b)), cmath_tag_v<FMIN>);
  }

  template <is_tree A, is_tree B>
  constexpr auto fmax(A&& a, B&& b)
  {
    assert(a.rank() == 0);
    return CMath2(promote(FWD(a)), promote(FWD(b)), cmath_tag_v<FMAX>);
  }

  template <is_tree A, is_tree B>
  constexpr auto pow(A&& a, B&& b)
  {
    return CMath2(promote(FWD(a)), promote(FWD(b)), cmath_tag_v<POW>);
  }

  template <is_tree A>
  constexpr auto abs(A&& a)
  {
    assert(a.rank() == 0);
    return CMath(promote(FWD(a)), cmath_tag_v<ABS>);
  }

  template <is_tree A>
  constexpr auto exp(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<EXP>);
  }

  template <is_tree A>
  constexpr auto log(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<LOG>);
  }

  template <is_tree A>
  constexpr auto sqrt(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<SQRT>);
  }

    template <is_tree A>
    constexpr auto sin(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<SIN>);
  }

  template <is_tree A>
  constexpr auto cos(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<COS>);
  }

  template <is_tree A>
  constexpr auto tan(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<TAN>);
  }

  template <is_tree A>
  constexpr auto asin(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<ASIN>);
  }

  template <is_tree A>
  constexpr auto acos(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<ACOS>);
  }

  template <is_tree A>
  constexpr auto atan(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<ATAN>);
  }

  template <is_tree A>
  constexpr auto atan2(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<ATAN2>);
  }

  template <is_tree A>
  constexpr auto sinh(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<SINH>);
  }

  template <is_tree A>
  constexpr auto cosh(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<COSH>);
  }

  template <is_tree A>
  constexpr auto tanh(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<TANH>);
  }

  template <is_tree A>
  constexpr auto asinh(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<ASINH>);
  }

  template <is_tree A>
  constexpr auto acosh(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<ACOSH>);
  }

  template <is_tree A>
  constexpr auto atanh(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<ATANH>);
  }

  template <is_tree A>
  constexpr auto ceil(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<CEIL>);
  }

  template <is_tree A>
  constexpr auto floor(A&& a)
  {
    return CMath(promote(FWD(a)), cmath_tag_v<FLOOR>);
  }
}

#endif // ALBERT_INCLUDE_ALBERT_HPP
