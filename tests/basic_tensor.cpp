#include "albert/Tensor.hpp"
#include "common.hpp"

using albert::Tensor;
using albert::tests::type_args;
using albert::tests::args;

template <class T>
constexpr static bool default_ctor(type_args<T> = {})
{
  Tensor<T, 0, 0> a;
  Tensor<T, 0, 1> b;
  Tensor<T, 1, 1> c;
  Tensor<T, 1, 3> d;
  Tensor<T, 2, 1> e;
  Tensor<T, 2, 4> f;
  Tensor<T, 8, 8> g;
  return true;
}

template <class T>
constexpr static bool linear_access(type_args<T> = {})
{
  bool passed = true;

  Tensor<T, 0, 0> a;
  a[0] = 1;
  passed &= ALBERT_CHECK(a[0] == 1);

  Tensor<T, 0, 3> b;
  b[0] = 1;
  passed &= ALBERT_CHECK(b[0] == 1);

  Tensor<T, 1, 1> c;
  c[0] = 1;
  ALBERT_CHECK(c[0] == 1);

  Tensor<T, 1, 3> d;
  d[0] = 1;
  d[1] = 2;
  d[2] = 3;
  passed &= ALBERT_CHECK(d[0] == 1);
  passed &= ALBERT_CHECK(d[1] == 2);
  passed &= ALBERT_CHECK(d[2] == 3);

  Tensor<T, 2, 1> e;
  e[0] = 1;
  passed &= ALBERT_CHECK(e[0] == 1);

  Tensor<T, 2, 2> f;
  f[0] = 1;
  f[1] = 2;
  f[2] = 3;
  f[3] = 4;
  passed &= ALBERT_CHECK(f[0] == 1);
  passed &= ALBERT_CHECK(f[1] == 2);
  passed &= ALBERT_CHECK(f[2] == 3);
  passed &= ALBERT_CHECK(f[3] == 4);

  return passed;
}

template <class T>
constexpr static bool aggregate_ctor(type_args<T>)
{
  bool passed = true;

  Tensor<T, 0, 0> a{ T(1) };
  passed &= ALBERT_CHECK(a[0] == 1);

  Tensor<T, 0, 1> b{ T(1) };
  passed &= ALBERT_CHECK(b[0] == 1);

  Tensor<T, 1, 1> c{ T(1) };
  passed &= ALBERT_CHECK(c[0] == 1);

  Tensor<T, 1, 3> d{ T(1), T(2) };
  passed &= ALBERT_CHECK(d[0] == 1);
  passed &= ALBERT_CHECK(d[1] == 2);

  Tensor<T, 1, 3> e{ T(1), T(2), T(3) };
  passed &= ALBERT_CHECK(e[0] == 1);
  passed &= ALBERT_CHECK(e[1] == 2);
  passed &= ALBERT_CHECK(e[2] == 3);

  Tensor<T, 2, 1> f{ T(1) };
  passed &= ALBERT_CHECK(f[0] == 1);

  Tensor<T, 2, 2> g{ T(1), T(2), T(3) };
  passed &= ALBERT_CHECK(g[0] == 1);
  passed &= ALBERT_CHECK(g[1] == 2);
  passed &= ALBERT_CHECK(g[2] == 3);

  Tensor<T, 2, 2> h{ T(1), T(2), T(3), T(4) };
  passed &= ALBERT_CHECK(h[0] == 1);
  passed &= ALBERT_CHECK(h[1] == 2);
  passed &= ALBERT_CHECK(h[2] == 3);
  passed &= ALBERT_CHECK(h[3] == 4);

  return passed;
}

template <class T>
constexpr static bool copy_ctor(type_args<T> = {})
{
  bool passed = true;

  Tensor<T, 0, 0> aa{ T(1) };
  Tensor a = aa;
  passed &= ALBERT_CHECK(a[0] == 1);

  Tensor<T, 0, 1> bb{ T(1) };
  Tensor b = bb;
  passed &= ALBERT_CHECK(b[0] == 1);

  Tensor<T, 1, 1> cc{ T(1) };
  Tensor c = cc;
  passed &= ALBERT_CHECK(c[0] == 1);

  Tensor<T, 1, 3> dd{ T(1), T(2) };
  Tensor d = dd;
  passed &= ALBERT_CHECK(d[0] == 1);
  passed &= ALBERT_CHECK(d[1] == 2);

  Tensor<T, 1, 3> ee{ T(1), T(2), T(3) };
  Tensor e = ee;
  passed &= ALBERT_CHECK(e[0] == 1);
  passed &= ALBERT_CHECK(e[1] == 2);
  passed &= ALBERT_CHECK(e[2] == 3);

  Tensor<T, 2, 1> ff{ T(1) };
  Tensor f = ff;
  passed &= ALBERT_CHECK(f[0] == 1);

  Tensor<T, 2, 2> gg{ T(1), T(2), T(3) };
  Tensor g = gg;
  passed &= ALBERT_CHECK(g[0] == 1);
  passed &= ALBERT_CHECK(g[1] == 2);
  passed &= ALBERT_CHECK(g[2] == 3);

  Tensor<T, 2, 2> hh{ T(1), T(2), T(3), T(4) };
  Tensor h = hh;
  passed &= ALBERT_CHECK(h[0] == 1);
  passed &= ALBERT_CHECK(h[1] == 2);
  passed &= ALBERT_CHECK(h[2] == 3);
  passed &= ALBERT_CHECK(h[3] == 4);

  return passed;
}

template <class T>
constexpr static bool move_ctor(type_args<T> = {})
{
  bool passed = true;

  Tensor<T, 0, 0> aa{ T(1) };
  Tensor a = std::move(aa);
  passed &= ALBERT_CHECK(a[0] == 1);

  Tensor<T, 0, 1> bb{ T(1) };
  Tensor b = std::move(bb);
  passed &= ALBERT_CHECK(b[0] == 1);

  Tensor<T, 1, 1> cc{ T(1) };
  Tensor c = std::move(cc);
  passed &= ALBERT_CHECK(c[0] == 1);

  Tensor<T, 1, 3> dd{ T(1), T(2) };
  Tensor d = std::move(dd);
  passed &= ALBERT_CHECK(d[0] == 1);
  passed &= ALBERT_CHECK(d[1] == 2);

  Tensor<T, 1, 3> ee{ T(1), T(2), T(3) };
  Tensor e = std::move(ee);
  passed &= ALBERT_CHECK(e[0] == 1);
  passed &= ALBERT_CHECK(e[1] == 2);
  passed &= ALBERT_CHECK(e[2] == 3);

  Tensor<T, 2, 1> ff{ T(1) };
  Tensor f = std::move(ff);
  passed &= ALBERT_CHECK(f[0] == 1);

  Tensor<T, 2, 2> gg{ T(1), T(2), T(3) };
  Tensor g = std::move(gg);
  passed &= ALBERT_CHECK(g[0] == 1);
  passed &= ALBERT_CHECK(g[1] == 2);
  passed &= ALBERT_CHECK(g[2] == 3);

  Tensor<T, 2, 2> hh{ T(1), T(2), T(3), T(4) };
  Tensor h = std::move(hh);
  passed &= ALBERT_CHECK(h[0] == 1);
  passed &= ALBERT_CHECK(h[1] == 2);
  passed &= ALBERT_CHECK(h[2] == 3);
  passed &= ALBERT_CHECK(h[3] == 4);

  return passed;
}

template <class T>
constexpr static bool copy(type_args<T> = {})
{
  bool passed = true;

  Tensor<T, 0, 0> aa{ T(1) };
  Tensor<T, 0, 0> a;
  a = aa;
  passed &= ALBERT_CHECK(a[0] == 1);

  Tensor<T, 0, 1> bb{ T(1) };
  Tensor<T, 0, 1> b;
  b = bb;
  passed &= ALBERT_CHECK(b[0] == 1);

  Tensor<T, 1, 1> cc{ T(1) };
  Tensor<T, 1, 1> c;
  c = cc;
  passed &= ALBERT_CHECK(c[0] == 1);

  Tensor<T, 1, 3> dd{ T(1), T(2) };
  Tensor<T, 1, 3> d;
  d = dd;
  passed &= ALBERT_CHECK(d[0] == 1);
  passed &= ALBERT_CHECK(d[1] == 2);

  Tensor<T, 1, 3> ee{ T(1), T(2), T(3) };
  Tensor<T, 1, 3> e;
  e = ee;
  passed &= ALBERT_CHECK(e[0] == 1);
  passed &= ALBERT_CHECK(e[1] == 2);
  passed &= ALBERT_CHECK(e[2] == 3);

  Tensor<T, 2, 1> ff{ T(1) };
  Tensor<T, 2, 1> f;
  f = ff;
  passed &= ALBERT_CHECK(f[0] == 1);

  Tensor<T, 2, 2> gg{ T(1), T(2), T(3) };
  Tensor<T, 2, 2> g;
  g = gg;
  passed &= ALBERT_CHECK(g[0] == 1);
  passed &= ALBERT_CHECK(g[1] == 2);
  passed &= ALBERT_CHECK(g[2] == 3);

  Tensor<T, 2, 2> hh{ T(1), T(2), T(3), T(4) };
  Tensor<T, 2, 2> h;
  h = hh;
  passed &= ALBERT_CHECK(h[0] == 1);
  passed &= ALBERT_CHECK(h[1] == 2);
  passed &= ALBERT_CHECK(h[2] == 3);
  passed &= ALBERT_CHECK(h[3] == 4);

  return passed;
}

template <class T>
constexpr static bool move(type_args<T> = {})
{
  bool passed = true;

  Tensor<T, 0, 0> a, aa{ T(1) };
  a = std::move(aa);
  passed &= ALBERT_CHECK(a[0] == 1);

  Tensor<T, 0, 1> b, bb{ T(1) };
  b = std::move(bb);
  passed &= ALBERT_CHECK(b[0] == 1);

  Tensor<T, 1, 1> c, cc{ T(1) };
  c = std::move(cc);
  passed &= ALBERT_CHECK(c[0] == 1);

  Tensor<T, 1, 3> d, dd{ T(1), T(2) };
  d = std::move(dd);
  passed &= ALBERT_CHECK(d[0] == 1);
  passed &= ALBERT_CHECK(d[1] == 2);

  Tensor<T, 1, 3> e, ee{ T(1), T(2), T(3) };
  e = std::move(ee);
  passed &= ALBERT_CHECK(e[0] == 1);
  passed &= ALBERT_CHECK(e[1] == 2);
  passed &= ALBERT_CHECK(e[2] == 3);

  Tensor<T, 2, 1> f, ff{ T(1) };
  f = std::move(ff);
  passed &= ALBERT_CHECK(f[0] == 1);

  Tensor<T, 2, 2> g, gg{ T(1), T(2), T(3) };
  g = std::move(gg);
  passed &= ALBERT_CHECK(g[0] == 1);
  passed &= ALBERT_CHECK(g[1] == 2);
  passed &= ALBERT_CHECK(g[2] == 3);

  Tensor<T, 2, 2> h, hh{ T(1), T(2), T(3), T(4) };
  h = std::move(hh);
  passed &= ALBERT_CHECK(h[0] == 1);
  passed &= ALBERT_CHECK(h[1] == 2);
  passed &= ALBERT_CHECK(h[2] == 3);
  passed &= ALBERT_CHECK(h[3] == 4);

  return passed;
}

template <class T>
constexpr static bool md_access(type_args<T> = {})
{
  bool passed = true;

  Tensor<T, 0, 0> a;
  a() = 1;
  passed &= ALBERT_CHECK(a[0] == 1);

  Tensor<T, 0, 3> b;
  b() = 1;
  passed &= ALBERT_CHECK(b[0] == 1);

  Tensor<T, 1, 1> c;
  c(0) = 1;
  ALBERT_CHECK(c[0] == 1);

  Tensor<T, 1, 3> d;
  d(0) = 1;
  d(1) = 2;
  d(2) = 3;
  passed &= ALBERT_CHECK(d[0] == 1);
  passed &= ALBERT_CHECK(d[1] == 2);
  passed &= ALBERT_CHECK(d[2] == 3);

  Tensor<T, 2, 1> e;
  e(0,0) = 1;
  passed &= ALBERT_CHECK(e[0] == 1);

  Tensor<T, 2, 2> f;
  f(0,0) = 1;
  f(0,1) = 2;
  f(1,0) = 3;
  f(1,1) = 4;
  passed &= ALBERT_CHECK(f[0] == 1);
  passed &= ALBERT_CHECK(f[1] == 2);
  passed &= ALBERT_CHECK(f[2] == 3);
  passed &= ALBERT_CHECK(f[3] == 4);

  Tensor<T, 3, 2> g;
  g(0,0,0) = 1;
  g(0,0,1) = 2;
  g(0,1,0) = 3;
  g(0,1,1) = 4;
  g(1,0,0) = 5;
  g(1,0,1) = 6;
  g(1,1,0) = 7;
  g(1,1,1) = 8;
  passed &= ALBERT_CHECK(g[0] == 1);
  passed &= ALBERT_CHECK(g[1] == 2);
  passed &= ALBERT_CHECK(g[2] == 3);
  passed &= ALBERT_CHECK(g[3] == 4);
  passed &= ALBERT_CHECK(g[4] == 5);
  passed &= ALBERT_CHECK(g[5] == 6);
  passed &= ALBERT_CHECK(g[6] == 7);
  passed &= ALBERT_CHECK(g[7] == 8);

  return passed;
}

template <class T>
constexpr static bool tests()
{
  bool passed = true;
  passed &= ALBERT_CHECK( default_ctor(args<T>)   );
  passed &= ALBERT_CHECK( linear_access(args<T>)   );
  passed &= ALBERT_CHECK( aggregate_ctor(args<T>) );
  passed &= ALBERT_CHECK( copy_ctor(args<T>) );
  passed &= ALBERT_CHECK( move_ctor(args<T>) );
  passed &= ALBERT_CHECK( copy(args<T>) );
  passed &= ALBERT_CHECK( move(args<T>) );
  passed &= ALBERT_CHECK( md_access(args<T>) );

  return passed;
}

int main()
{
  constexpr bool i = tests<int>();
  constexpr bool f = tests<float>();
  constexpr bool d = tests<double>();
}
