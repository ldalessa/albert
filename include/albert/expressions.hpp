#ifndef ALBERT_INCLUDE_EXPRESSIONS_HPP
#define ALBERT_INCLUDE_EXPRESSIONS_HPP

#include "albert/Bind.hpp"
#include "albert/Index.hpp"
#include "albert/cmath.hpp"
#include "albert/concepts.hpp"
#include <utility>

namespace albert
{
  template <is_tree A, is_tree B>
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
      constexpr auto l_index = outer_v<A>;
      constexpr auto r_index = outer_v<B>;
      static_assert(is_permutation(l_index, r_index)); // tensor expression addition must have compatible indices
    }

    constexpr static auto outer()
    {
      return outer_v<A>;
    }
  };

  template <is_tree A, is_tree B>
  struct Sum : Addition<A, B>, Bindable<Sum<A, B>>
  {
    using Addition<A, B>::Addition;
  };

  template <is_tree A, is_tree B>
  Sum(A, B) -> Sum<A, B>;

  template <is_tree A, is_tree B>
  struct Diff : Addition<A, B>, Bindable<Diff<A, B>>
  {
    using Addition<A, B>::Addition;
  };

  template <is_tree A, is_tree B>
  Diff(A, B) -> Diff<A, B>;

  template <is_tree A, is_tree B>
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

  template <is_tree A, is_tree B>
  struct Product : Contraction<A, B>, Bindable<Product<A, B>>
  {
    using Contraction<A, B>::Contraction;
  };

  template <is_tree A, is_tree B>
  Product(A, B) -> Product<A, B>;

  template <is_tree A, is_tree B>
  struct Ratio : Contraction<A, B>
  {
    using Contraction<A, B>::Contraction;
  };

  template <is_tree A, is_tree B>
  Ratio(A, B) -> Ratio<A, B>;

  template <is_tree A>
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

  template <is_tree A, is_tensor_index auto index>
  struct Partial : Bindable<Partial<A, index>>
  {
    using tree_node_tag = void;
    using unary_node_tag = void;

    A a;

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
}

#endif // ALBERT_INCLUDE_EXPRESSIONS_HPP
