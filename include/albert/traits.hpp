#ifndef ALBERT_INCLUDE_TRAITS_HPP
#define ALBERT_INCLUDE_TRAITS_HPP

#include <type_traits>

namespace albert
{
  template <class T>
  inline constexpr auto outer_v = std::remove_cvref_t<T>::outer();

  template <class T>
  inline constexpr std::size_t rank_v = [] {
    if constexpr (requires { std::remove_cvref_t<T>::rank(); }) {
      return std::remove_cvref_t<T>::rank();
    }
    else {
      return std::remove_cvref_t<T>::outer().size();
    }
  }();

  template <class T>
  inline constexpr std::size_t dim_v = std::remove_cvref_t<T>::dim();
}

#endif // ALBERT_INCLUDE_TRAITS_HPP
