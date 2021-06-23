#ifndef ALBERT_INCLUDE_BIND_HPP
#define ALBERT_INCLUDE_BIND_HPP

#include "albert/Index.hpp"
#include "albert/ScalarIndex.hpp"
#include "albert/TensorIndex.hpp"
#include "albert/concepts.hpp"
#include "albert/evaluate.hpp"
#include "albert/utils.hpp"
#include <ce/cvector.hpp>
#include <utility>

namespace albert
{
  template <class T>
  struct Bindable;

  /// The bind node.
  ///
  /// The bind node is the most semantically important type of node in the
  /// albert expression tree. A bind node associates some subtree with an
  /// Einstein notation index mapping, potentially including self-contraction,
  /// and/or a set of projected indices.
  ///
  /// Bind nodes may have full subtrees or just leafs (like tensor nodes).
  ///
  /// Bind nodes look like:
  ///
  ///     is_index i, j, k, l;                     // indices
  ///     is_tensor A;                             // actual matrix tensor
  ///     is_expression B;                         // matrix expression
  ///     is_expression C;                         // vector expression
  ///     auto bind = A(i, j);                     // basic bind
  ///     auto bind = A(i, i);                     // trace
  ///     auto bind = A(1, i);                     // projection
  ///     auto bind = A(1, 0);                     // fully projected
  ///
  ///     auto product = A(i, j) * B(j);           // normal vector product
  ///     auto product = A(i, k) * B(k, j);        // normal matrix product
  ///     auto product = A(i, j) * B(i, j);        // inner product
  ///     auto product = A(i, j) * B(k, l);        // kronecker product
  ///     auto product = A(k, i) * B(j, k);        // some odd contraction
  ///     auto product = A(i, j) * B(j, k) * C(k); // expression
  ///     auto product = (A(i, j) + B(j, i)) / 2;  // symmetric transformation
  ///
  ///     A(i, j) = B(j, i);                       // transpose assignment
  ///     A(1, j) = B(j, 0);                       // projection assignment
  ///     A(0, i) = B(i, j) * C(j);                // projection assignment
  ///     A(0, 0) = 42;                            // scalar assignment
  ///
  /// @param     A The type of the subtree.
  /// @param index The index binding to the subtree.
  template <is_tensor A, is_tensor_index auto index>
  struct Bind : Bindable<Bind<A, index>>
  {
    using scalar_type = scalar_type_t<A>;

    constexpr static int Order = order_v<Bind>;
    constexpr static int M = index.n_projected(); //!< number of projected indices

    A a;                                        //!< subtree
    ScalarIndex<M> _projected;                  //!< projected indices

    /// Construct a bind node for a subtree.
    ///
    /// This uses a trick to infer the right type for the subtree.
    /// - If the subtree is an lvalue reference then A is a reference type and
    ///   we'll capture the reference.
    /// - If the subree is an rvalue reference then A is a value type and we'll
    ///   move from it.
    ///
    /// This allows use of bound tensors for the lhs of expressions.
    ///
    /// @param         a The subtree we're binding.
    /// @param projected Scalars associated with projections (if any).
    /// @param           Helper to specify the `index` CNTTP.
    template <is_tensor B>
    constexpr Bind(B&& a, ce::cvector<int, M> const& projected, nttp_args<index>)
        : a(std::forward<A>(a))
        , _projected(projected)
    {
      constexpr auto l = order_v<A>;
      constexpr auto r = index.size();
      static_assert(l == r);
    }

    /// Default copy and move will prevent implicit operator= generation, which
    /// means that the `is_expression` version will match _all_ instances of
    /// Bind::operator=.
    /// @{
    constexpr Bind(Bind const&) = default;
    constexpr Bind(Bind&&) = default;
    /// @}

    template <is_expression B>
    constexpr auto operator=(B&& b)
      -> Bind&
    {
      return assign(FWD(b), [](auto&& a, auto&&b) {
          FWD(a) = FWD(b);
      });
    }

    template <is_expression B>
    constexpr auto operator+=(B&& b)
      -> Bind&
    {
      return assign(FWD(b), [](auto&& a, auto&&b) {
          FWD(a) += FWD(b);
      });
    }

    constexpr auto operator-=(is_expression auto && b)
      -> Bind&
    {
      return assign(FWD(b), [](auto&& a, auto&&b) {
          FWD(a) -= FWD(b);
      });
    }

    template <is_expression B>
    constexpr auto assign(B&& b, auto&& op)
      -> Bind&
    {
      constexpr TensorIndex l = outer_v<Bind>;
      constexpr TensorIndex r = outer_v<B>;
      static_assert(is_permutation(l, r), "indices don't match in assignment");

      constexpr bool transpose = (l != r and std::remove_cvref_t<B>::contains(tag()));
      if constexpr (transpose or std::remove_cvref_t<B>::may_alias(tag())) {
        return albert::evaluate_via_temp(*this, FWD(b), FWD(op));
      }
      else {
        return albert::evaluate(*this, FWD(b), FWD(op));
      }
    }

    /// Evaluate into a scalar.
    constexpr operator scalar_type() const requires (Order == 0)
    {
      return evaluate(ScalarIndex<0>{});
    }

    constexpr static bool may_alias(auto&& tag)
    {
      return std::remove_cvref_t<A>::may_alias(FWD(tag));
    }

    constexpr static bool contains(auto&& tag)
    {
      return std::remove_cvref_t<A>::contains(FWD(tag));
    }

    constexpr static auto tag() -> decltype(auto)
    {
      return std::remove_cvref_t<A>::tag();
    }

    /// CPO support.
    constexpr static auto outer()
      -> decltype(auto)
    {
      return index.exclusive();
    }

    constexpr static auto dim()
      -> decltype(auto)
    {
      return dim_v<A>;
    }

    constexpr static auto order()
      -> decltype(auto)
    {
      return outer().size();
    }

    /// Evaluate a bind node when there's a contraction.
    ///
    /// Contracted binds can only exist on the right-hand-side of an equation,
    /// because they don't represent lvalues when evaluated.
    constexpr auto evaluate(ScalarIndex<Order> const& i) const
      requires(index.n_repeated() != 0)
    {
      constexpr TensorIndex     outer = index.exclusive();
      constexpr TensorIndex projected = index.projected();
      constexpr TensorIndex     inner = index.repeated();
      constexpr TensorIndex       all = outer + projected + inner;
      constexpr int     N = dim();
      constexpr int Order = outer.size() + projected.size();
      constexpr int     I = inner.size();

      auto rhs = [&](auto const& i) {
        return a.evaluate(select<all, index>(i));
      };

      ScalarIndex<Order + I> j(i + _projected);
      decltype(rhs(j)) temp{};
      do {
        temp += rhs(j);
      } while (carry_sum_inc<N, Order>(j));
      return temp;
    }

    /// Evaluate a bind node when there's only a projection.
    constexpr auto evaluate(ScalarIndex<Order> const& i) const -> decltype(auto)
      requires(index.n_repeated() == 0 and index.n_projected() != 0)
    {
      constexpr TensorIndex  outer = index.exclusive();
      constexpr TensorIndex slices = index.projected();
      constexpr TensorIndex    all = outer + slices;
      return a.evaluate(select<all, index>(i + _projected));
    }

    /// Evaluate a bind node when there's only a projection.
    ///
    /// This version of the bind represents an assignable lvalue, assuming that
    /// the underlying expression is assignable. This is currently only true for
    /// raw tensors.
    constexpr auto evaluate(ScalarIndex<Order> const& i) -> decltype(auto)
      requires(index.n_repeated() == 0 and index.n_projected() != 0)
    {
      constexpr TensorIndex  outer = index.exclusive();
      constexpr TensorIndex slices = index.projected();
      constexpr TensorIndex    all = outer + slices;
      return a.evaluate(select<all, index>(i + _projected));
    }

    /// Evaluate a bind that contains neither contraction nor projection.
    constexpr auto evaluate(ScalarIndex<Order> i) const -> decltype(auto)
      requires((index.n_projected() + index.n_repeated()) == 0)
    {
      return a.evaluate(i);
    }

    /// Evaluate a bind that contains neither contraction nor projection.
    ///
    /// This version of the bind represents an assignable lvalue, assuming that
    /// the underlying expression is assignable. This is currently only true for
    /// raw tensors.
    constexpr auto evaluate(ScalarIndex<Order> i) -> decltype(auto)
      requires((index.n_projected() + index.n_repeated()) == 0)
    {
      return a.evaluate(i);
    }
  };

  /// Deduction guide allows us to infer lvalue reference types.
  template <is_tensor A, is_tensor_index auto index, int M = 0>
  Bind(A&&, ce::cvector<int, M> const&, nttp_args<index>)
    -> Bind<A, index>;

  template <class T>
  struct Bindable
  {
    constexpr static int Order = order_v<T>;

    constexpr auto derived() const & -> T const*
    {
      return static_cast<T const*>(this);
    }

    constexpr auto derived() && -> T*
    {
      return static_cast<T*>(this);
    }

    constexpr auto derived() & -> T*
    {
      return static_cast<T*>(this);
    }

    template <class... Is>
    requires (all_index<Is...>)
    constexpr auto operator()(Is... is) const &
      -> decltype(auto)
    {
      static_assert(sizeof...(Is) == Order, "Tensor index must be fully specified");

      if constexpr (all_integral_index<Is...>) {
        ScalarIndex<Order> index(is...);
        return derived()->evaluate(index);
      }
      else {
        // Build an index sequence that has all of the characters as specified
        // (any projected indices are mapped to `\0`).
        constexpr cat_index_type_t<Is...> all = {};

        // Build a tensor index for the index sequence.
        constexpr TensorIndex index(all);

        // Extract the projected indices.
        ce::cvector<int, index.n_projected()> projected;
        ([&] {
          if constexpr (std::integral<Is>) {
            projected.push_back(is);
          }
        }(), ...);

        return Bind { *derived(), projected, nttp<index> };
      }
    }

    template <class... Is>
    requires (all_index<Is...>)
    constexpr auto operator()(Is... is) &&
      -> decltype(auto)
    {
      static_assert(sizeof...(Is) == Order, "Tensor index must be fully specified");

      if constexpr (all_integral_index<Is...>) {
        ScalarIndex<Order> index(is...);
        return derived()->evaluate(index);
      }
      else {
        // Build an index sequence that has all of the characters as specified
        // (any projected indices are mapped to `\0`).
        constexpr cat_index_type_t<Is...> all = {};

        // Build a tensor index for the index sequence.
        constexpr TensorIndex index(all);

        // Extract the projected indices.
        ce::cvector<int, index.n_projected()> projected;
        ([&] {
          if constexpr (std::integral<Is>) {
            projected.push_back(is);
          }
        }(), ...);

        return Bind { *derived(), projected, nttp<index> };
      }
    }

    template <class... Is>
    requires (all_index<Is...>)
    constexpr auto operator()(Is... is) &
      -> decltype(auto)
    {
      static_assert(sizeof...(Is) == Order, "Tensor index must be fully specified");

      if constexpr (all_integral_index<Is...>) {
        ScalarIndex<Order> index(is...);
        return derived()->evaluate(index);
      }
      else {
        // Build an index sequence that has all of the characters as specified
        // (any projected indices are mapped to `\0`).
        constexpr cat_index_type_t<Is...> all = {};

        // Build a tensor index for the index sequence.
        constexpr TensorIndex index(all);

        // Extract the projected indices.
        ce::cvector<int, index.n_projected()> projected;
        ([&] {
          if constexpr (std::integral<Is>) {
            projected.push_back(is);
          }
        }(), ...);

        return Bind { *derived(), projected, nttp<index> };
      }
    }

    template <is_tensor_index auto index>
    constexpr auto rebind() const &
      -> decltype(auto)
    {
      return Bind { *derived(), {}, nttp<index> };
    }

    template <is_tensor_index auto index>
    constexpr auto rebind() &&
      -> decltype(auto)
    {
      return Bind { std::move(*derived()), {}, nttp<index> };
    }

    template <is_tensor_index auto index>
    constexpr auto rebind() &
      -> decltype(auto)
    {
      return Bind{ *derived(), {}, nttp<index> };
    }
  };
}

#endif // ALBERT_INCLUDE_BIND_HPP
