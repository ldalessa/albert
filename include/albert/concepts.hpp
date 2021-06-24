#ifndef ALBERT_INCLUDE_CONCEPTS_HPP
#define ALBERT_INCLUDE_CONCEPTS_HPP

#include "albert/cpos.hpp"
#include "albert/utils.hpp"
#include <concepts>

namespace albert
{
  namespace traits
  {
    // Allow types other than integral and floating_point to advertise that
    // they're scalars (e.g., a rational class)
    template <class>
    struct is_scalar : std::false_type {};

    template <class A, class B>
    struct may_alias : std::is_same<A, B> {};
  }

  // Scalars.
  template <class T>
  concept is_scalar = std::integral<T> || std::floating_point<T> || traits::is_scalar<T>::value;

  // Extract the scalar type of some expression type.
  template <class T>
  struct scalar_type;

  template <class T>
  requires requires { typename std::remove_cvref_t<T>::scalar_type; }
  struct scalar_type<T>
  {
    using type = typename std::remove_cvref_t<T>::scalar_type;
  };

  template <class T>
  requires is_scalar<T>
  struct scalar_type<T>
  {
    using type = T;
  };

  template <class T>
  using scalar_type_t = typename scalar_type<T>::type;

  /// For our purposes, a tensor is anything with a order and dimension.
  template <class T>
  concept is_tensor = requires(T t)
  {
    typename scalar_type<T>::type;
    { albert::order(t) } -> std::same_as<int>;
    { albert::dim(t) } -> std::same_as<int>;
  };

  template <class T>
  concept is_tensor_index = requires {
    typename std::remove_cvref_t<T>::tensor_index_tag;
  };

  inline constexpr auto clang_hack = []{};

  template <class T>
  concept is_expression = is_tensor<T> and requires (T t) {
    { albert::outer(t) } -> is_tensor_index;
    { albert::contains(t, clang_hack) } -> std::same_as<bool>;
    { albert::may_alias(t, clang_hack) } -> std::same_as<bool>;
  };

  template <class T>
  concept is_index = std::integral<T> or requires {
    typename std::remove_cvref_t<T>::index_tag;
  };

  template<typename... Ts>
  concept all_index = (is_index<Ts> && ...);

  template<typename... Ts>
  concept all_integral_index = all_index<Ts...> && (std::integral<Ts> && ...);

  template <class T>
  inline constexpr auto outer_v = std::remove_cvref_t<T>::outer();

  template <class T>
  inline constexpr int order_v = std::remove_cvref_t<T>::order();

  template <class T>
  inline constexpr int dim_v = std::remove_cvref_t<T>::dim();

  template <class T, auto tag>
  inline constexpr int contains_v = std::remove_cvref_t<T>::contains(tag);

  template <class T, auto tag>
  inline constexpr int may_alias_v = std::remove_cvref_t<T>::may_alias(tag);

  template <class T>
  inline constexpr int size_v = std::remove_cvref_t<T>::size();
}

#endif // ALBERT_INCLUDE_CONCEPTS_HPP
