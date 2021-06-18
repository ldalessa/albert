#ifndef ALBERT_TESTS_COMMON_HPP
#define ALBERT_TESTS_COMMON_HPP

#include <cstdio>
#include <type_traits>
#include <experimental/source_location>

#define ALBERT_CHECK(expr, ...) albert::tests::check((expr), #expr, ##__VA_ARGS__)

namespace albert::tests
{
  using std::experimental::source_location;

  constexpr static inline bool
  check(bool condition, const char *expr,
        source_location&& src = source_location::current())
  {
    if (condition) {
      return true;
    }

    if (!std::is_constant_evaluated()) {
      std::printf("%s:%d failed unit test %s\n", src.file_name(), src.line(), expr);
    }

    throw src;
  }

  constexpr void unused(...) {}

  template <class...>
  struct type_args {};

  template <class... Ts>
  constexpr inline type_args<Ts...> args = {};
}

#endif// ALBERT_TESTS_COMMON_HPP
