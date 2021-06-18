#include "albert/Tensor.hpp"
#include "albert/grammar.hpp"
#include "common.hpp"

using albert::Tensor;
using albert::tests::type_args;
using albert::tests::args;

constexpr static albert::Index<'i'> i;
constexpr static albert::Index<'j'> j;
constexpr static albert::Index<'k'> k;
constexpr static albert::Index<'l'> l;

template <class T>
constexpr static bool bind(type_args<T> = {})
{
  bool passed = true;
  return passed;
}

template <class T>
constexpr static bool addition(type_args<T> = {})
{
  bool passed = true;
  albert::Tensor<T, 0, 0> a(1), b = a + a;
  passed &= ALBERT_CHECK( a + a == 2 );
  passed &= ALBERT_CHECK( b == 2 );

  albert::Tensor<T, 1, 3> c = { 1, 2, 3}, d = c(i) + c(i);
  passed &= ALBERT_CHECK( d[0] == 2 );
  passed &= ALBERT_CHECK( d[1] == 4 );
  passed &= ALBERT_CHECK( d[2] == 6 );

  albert::Tensor<T, 2, 3> A = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  }, B = A(i,j) + A(i,j);

  passed &= ALBERT_CHECK( B[0] == 2 );
  passed &= ALBERT_CHECK( B[1] == 4 );
  passed &= ALBERT_CHECK( B[2] == 6 );
  passed &= ALBERT_CHECK( B[3] == 8 );
  passed &= ALBERT_CHECK( B[4] == 10 );
  passed &= ALBERT_CHECK( B[5] == 12 );
  passed &= ALBERT_CHECK( B[6] == 14 );
  passed &= ALBERT_CHECK( B[7] == 16 );
  passed &= ALBERT_CHECK( B[8] == 18 );

  return passed;
}

template <class T>
constexpr static bool subtraction(type_args<T> = {})
{
  bool passed = true;
  albert::Tensor<T, 0, 0> a(1), b = a - a;
  passed &= ALBERT_CHECK( a - a == 0 );
  passed &= ALBERT_CHECK( b == 0 );

  albert::Tensor<T, 1, 3> c = { 1, 2, 3}, d = c(i) - c(i);
  passed &= ALBERT_CHECK( d[0] == 0 );
  passed &= ALBERT_CHECK( d[1] == 0 );
  passed &= ALBERT_CHECK( d[2] == 0 );

  albert::Tensor<T, 2, 3> A = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  }, B = A(i,j) - A(i,j);

  passed &= ALBERT_CHECK( B[0] == 0 );
  passed &= ALBERT_CHECK( B[1] == 0 );
  passed &= ALBERT_CHECK( B[2] == 0 );
  passed &= ALBERT_CHECK( B[3] == 0 );
  passed &= ALBERT_CHECK( B[4] == 0 );
  passed &= ALBERT_CHECK( B[5] == 0 );
  passed &= ALBERT_CHECK( B[6] == 0 );
  passed &= ALBERT_CHECK( B[7] == 0 );
  passed &= ALBERT_CHECK( B[8] == 0 );

  return passed;
}

template <class T>
constexpr static bool projection(type_args<T> = {})
{
  bool passed = true;
  albert::Tensor<T, 2, 3> A = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };

  for (int i = 0; i < 3; ++i) {
    auto a = A(i,j);
    for (int j = 0; j < 3; ++j) {
      T n = a(j);
      T m = A(i,j);
      passed &= ALBERT_CHECK( n == m );
    }
  }

  for (int j = 0; j < 3; ++j) {
    auto a = A(i,j);
    for (int i = 0; i < 3; ++i) {
      T n = a(i);
      T m = A(i,j);
      passed &= ALBERT_CHECK( n == m );
    }
  }

  albert::Tensor<T, 3, 2> B = {
    1, 2, 3, 4,
    5, 6, 7, 8
  };


  for (int i = 0; i < 2; ++i) {
    auto b = B(i,j,k);
    for (int j = 0; j < 2; ++j) {
      auto bb = b(j,k);
      for (int k = 0; k < 2; ++k) {
        T n = bb(k);
        T m = B(i,j,k);
        passed &= ALBERT_CHECK( n == m );
      }
    }
  }

  for (int j = 0; j < 2; ++j) {
    auto b = B(i,j,k);
    for (int i = 0; i < 2; ++i) {
      auto bb = b(i,k);
      for (int k = 0; k < 2; ++k) {
        T n = bb(k);
        T m = B(i,j,k);
        passed &= ALBERT_CHECK( n == m );
      }
    }
  }

  for (int j = 0; j < 2; ++j) {
    auto b = B(i,j,k);
    for (int k = 0; k < 2; ++k) {
      auto bb = b(i,k);
      for (int i = 0; i < 2; ++i) {
        T n = bb(i);
        T m = B(i,j,k);
        passed &= ALBERT_CHECK( n == m );
      }
    }
  }

  return passed;
}

template <class T>
constexpr static bool trace(type_args<T> = {})
{
  bool passed = true;
  albert::Tensor<T, 0, 3> A = {
    1
  };

  albert::Tensor<T, 2, 3> B = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };

  albert::Tensor<T, 4, 2> C = {
    0 , 1 , 2 , 3 ,
    4 , 5 , 6 , 7 ,
    8 , 9 , 10, 11,
    12, 13, 14, 15
  };

  T a = A();
  passed &= ALBERT_CHECK( a == 1 );

  T b = B(i,i);
  passed &= ALBERT_CHECK( b == 15 );

  T c = C(i,i,i,i);
  passed &= ALBERT_CHECK( c == 15 );
  return passed;
}

template <class...> struct print;

template <class T>
constexpr static bool transposition(type_args<T> = {})
{
  bool passed = true;

  albert::Tensor<T, 2, 3> A = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  }, B;

  B(j,i) = A(i,j);

  passed &= ALBERT_CHECK( B(0,0) == A(0,0) );
  passed &= ALBERT_CHECK( B(0,1) == A(1,0) );
  passed &= ALBERT_CHECK( B(0,2) == A(2,0) );
  passed &= ALBERT_CHECK( B(1,0) == A(0,1) );
  passed &= ALBERT_CHECK( B(1,1) == A(1,1) );
  passed &= ALBERT_CHECK( B(1,2) == A(2,1) );
  passed &= ALBERT_CHECK( B(2,0) == A(0,2) );
  passed &= ALBERT_CHECK( B(2,1) == A(1,2) );
  passed &= ALBERT_CHECK( B(2,2) == A(2,2) );

  albert::Tensor<T, 4, 2> C = {
    0 , 1 , 2 , 3 ,
    4 , 5 , 6 , 7 ,
    8 , 9 , 10, 11,
    12, 13, 14, 15
  }, D;

  D(l,k,j,i) = C(i,j,k,l);

  passed &= ALBERT_CHECK( D(0,0,0,0) == C(0,0,0,0) );
  passed &= ALBERT_CHECK( D(0,0,0,1) == C(1,0,0,0) );
  passed &= ALBERT_CHECK( D(0,0,1,0) == C(0,1,0,0) );
  passed &= ALBERT_CHECK( D(0,0,1,1) == C(1,1,0,0) );
  passed &= ALBERT_CHECK( D(0,1,0,0) == C(0,0,1,0) );
  passed &= ALBERT_CHECK( D(0,1,0,1) == C(1,0,1,0) );
  passed &= ALBERT_CHECK( D(0,1,1,0) == C(0,1,1,0) );
  passed &= ALBERT_CHECK( D(0,1,1,1) == C(1,1,1,0) );
  passed &= ALBERT_CHECK( D(1,0,0,0) == C(0,0,0,1) );
  passed &= ALBERT_CHECK( D(1,0,0,1) == C(1,0,0,1) );
  passed &= ALBERT_CHECK( D(1,0,1,0) == C(0,1,0,1) );
  passed &= ALBERT_CHECK( D(1,0,1,1) == C(1,1,0,1) );
  passed &= ALBERT_CHECK( D(1,1,0,0) == C(0,0,1,1) );
  passed &= ALBERT_CHECK( D(1,1,0,1) == C(1,0,1,1) );
  passed &= ALBERT_CHECK( D(1,1,1,0) == C(0,1,1,1) );
  passed &= ALBERT_CHECK( D(1,1,1,1) == C(1,1,1,1) );

  D(j,i,l,k) = C(i,j,k,l);

  passed &= ALBERT_CHECK( D(0,0,0,0) == C(0,0,0,0) );
  passed &= ALBERT_CHECK( D(0,1,0,0) == C(1,0,0,0) );
  passed &= ALBERT_CHECK( D(1,0,0,0) == C(0,1,0,0) );
  passed &= ALBERT_CHECK( D(1,1,0,0) == C(1,1,0,0) );
  passed &= ALBERT_CHECK( D(0,0,0,1) == C(0,0,1,0) );
  passed &= ALBERT_CHECK( D(0,1,0,1) == C(1,0,1,0) );
  passed &= ALBERT_CHECK( D(1,0,0,1) == C(0,1,1,0) );
  passed &= ALBERT_CHECK( D(1,1,0,1) == C(1,1,1,0) );
  passed &= ALBERT_CHECK( D(0,0,1,0) == C(0,0,0,1) );
  passed &= ALBERT_CHECK( D(0,1,1,0) == C(1,0,0,1) );
  passed &= ALBERT_CHECK( D(1,0,1,0) == C(0,1,0,1) );
  passed &= ALBERT_CHECK( D(1,1,1,0) == C(1,1,0,1) );
  passed &= ALBERT_CHECK( D(0,0,1,1) == C(0,0,1,1) );
  passed &= ALBERT_CHECK( D(0,1,1,1) == C(1,0,1,1) );
  passed &= ALBERT_CHECK( D(1,0,1,1) == C(0,1,1,1) );
  passed &= ALBERT_CHECK( D(1,1,1,1) == C(1,1,1,1) );

  return passed;
}

template <class T>
constexpr static bool tests(type_args<T> type = {})
{
  bool passed = true;
  passed &= bind(type);
  passed &= addition(type);
  passed &= subtraction(type);
  passed &= projection(type);
  passed &= trace(type);
  passed &= transposition(type);
  return passed;
}

int main()
{
  //constexpr
  bool i = tests(args<int>);
  // constexpr bool f = tests(args<float>);
  // constexpr bool d = tests(args<double>);
}
