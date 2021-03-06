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
constexpr static bool scalar_multiplication(type_args<T> = {})
{
  bool passed = true;
  albert::Tensor<T, 1, 3> a = { 1, 2, 3 }, b = 2 * a(i);

  passed &= ALBERT_CHECK( b[0] == 2 );
  passed &= ALBERT_CHECK( b[1] == 4 );
  passed &= ALBERT_CHECK( b[2] == 6 );

  albert::Tensor<T, 2, 3> A = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  }, B = 2 * A(i,j);

  passed &= ALBERT_CHECK( B[0] == 2  );
  passed &= ALBERT_CHECK( B[1] == 4  );
  passed &= ALBERT_CHECK( B[2] == 6  );
  passed &= ALBERT_CHECK( B[3] == 8  );
  passed &= ALBERT_CHECK( B[4] == 10 );
  passed &= ALBERT_CHECK( B[5] == 12 );
  passed &= ALBERT_CHECK( B[6] == 14 );
  passed &= ALBERT_CHECK( B[7] == 16 );
  passed &= ALBERT_CHECK( B[8] == 18 );
  return passed;
}

template <class T>
constexpr static bool scalar_division(type_args<T> = {})
{
  bool passed = true;
  albert::Tensor<T, 1, 3> a = { 2, 4, 6 };
  albert::Tensor b = a(i) / 2;

  passed &= ALBERT_CHECK( b[0] == 1 );
  passed &= ALBERT_CHECK( b[1] == 2 );
  passed &= ALBERT_CHECK( b[2] == 3 );

  albert::Tensor<T, 2, 3> A = {
    2,  4,  6,
    8,  10, 12,
    14, 16, 18
  }, B = A(i,j) / 2;

  passed &= ALBERT_CHECK( B[0] == 1 );
  passed &= ALBERT_CHECK( B[1] == 2 );
  passed &= ALBERT_CHECK( B[2] == 3 );
  passed &= ALBERT_CHECK( B[3] == 4 );
  passed &= ALBERT_CHECK( B[4] == 5 );
  passed &= ALBERT_CHECK( B[5] == 6 );
  passed &= ALBERT_CHECK( B[6] == 7 );
  passed &= ALBERT_CHECK( B[7] == 8 );
  passed &= ALBERT_CHECK( B[8] == 9 );
  return passed;
}

template <class T>
constexpr static bool contraction(type_args<T> = {})
{
  bool passed = true;

  albert::Tensor<T, 1, 3> a = { 1, 2, 3 };
  T dot = a(i) * a(i);
  passed &= ALBERT_CHECK( dot == 14 );
  passed &= ALBERT_CHECK( a(i) * a(i) == 14 );

  albert::Tensor<T, 2, 3> A = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };

  T dot2 = A(i,j) * A(i,j);
  passed &= ALBERT_CHECK( dot2 == 285 );
  passed &= ALBERT_CHECK( A(i,j) * A(i,j) == 285 );

  T dot3 = A(i,j) * A(j,i);
  passed &= ALBERT_CHECK( dot3 == 261 );
  passed &= ALBERT_CHECK( A(i,j) * A(j,i) == 261 );

  albert::Tensor Aa = A(i,j) * a(j);
  passed &= ALBERT_CHECK( Aa(0) == 14 );
  passed &= ALBERT_CHECK( Aa(1) == 32 );
  passed &= ALBERT_CHECK( Aa(2) == 50 );

  albert::Tensor Ata = A(j,i) * a(j);
  passed &= ALBERT_CHECK( Ata(0) == 30 );
  passed &= ALBERT_CHECK( Ata(1) == 36 );
  passed &= ALBERT_CHECK( Ata(2) == 42 );

  a = A(i,j) * a(j);
  passed &= ALBERT_CHECK( a(0) == 14 );
  passed &= ALBERT_CHECK( a(1) == 32 );
  passed &= ALBERT_CHECK( a(2) == 50 );

  albert::Tensor<T, 2, 3> A2 = {
    30,  36,  42,
    66,  81,  96,
    102, 126, 150
  };

  albert::Tensor B = A(i,k) * A(k,j);

  passed &= ALBERT_CHECK( B[0] == A2[0] );
  passed &= ALBERT_CHECK( B[1] == A2[1] );
  passed &= ALBERT_CHECK( B[2] == A2[2] );
  passed &= ALBERT_CHECK( B[3] == A2[3] );
  passed &= ALBERT_CHECK( B[4] == A2[4] );
  passed &= ALBERT_CHECK( B[5] == A2[5] );
  passed &= ALBERT_CHECK( B[6] == A2[6] );
  passed &= ALBERT_CHECK( B[7] == A2[7] );
  passed &= ALBERT_CHECK( B[8] == A2[8] );

  albert::Tensor<T, 2, 3> C;
  C(i,k) = A(i,j) * albert::??(j,k);

  passed &= ALBERT_CHECK( C[0] == A[0] );
  passed &= ALBERT_CHECK( C[1] == A[1] );
  passed &= ALBERT_CHECK( C[2] == A[2] );
  passed &= ALBERT_CHECK( C[3] == A[3] );
  passed &= ALBERT_CHECK( C[4] == A[4] );
  passed &= ALBERT_CHECK( C[5] == A[5] );
  passed &= ALBERT_CHECK( C[6] == A[6] );
  passed &= ALBERT_CHECK( C[7] == A[7] );
  passed &= ALBERT_CHECK( C[8] == A[8] );

  albert::Tensor<T, 2, 2> D = {
    1, 2,
    3, 4
  };
  T b = D(i,j) * albert::??(i,j);
  passed &= ALBERT_CHECK( b == -1 );

  albert::Tensor<T, 3, 3> E = {
    1,  2,  3,
    4,  5,  6,
    7,  8,  9,

    10, 11, 12,
    13, 14, 15,
    16, 17, 18,

    19, 20, 21,
    22, 23, 24,
    25, 26, 27
  };

  T c = E(i,j,k) * albert::??(i,j,k);

  // 0: {0, 1, 2} parity  1
  // 1: {0, 2, 1} parity -1
  // 2: {1, 0, 2} parity -1
  // 3: {1, 2, 0} parity  1
  // 4: {2, 0, 1} parity  1
  // 5: {2, 1, 0} parity -1
  T d = E(0,1,2) + E(1,2,0) + E(2,0,1) -
        E(0,2,1) - E(1,0,2) - E(2,1,0);
  passed &= ALBERT_CHECK( c == d );

  albert::Tensor kronecker = A(i,j) * A(k,l);

  passed &= ALBERT_CHECK( kronecker(0,0,0,0) == A(0,0) * A(0,0) );
  passed &= ALBERT_CHECK( kronecker(0,0,0,1) == A(0,0) * A(0,1) );
  passed &= ALBERT_CHECK( kronecker(0,0,0,2) == A(0,0) * A(0,2) );
  passed &= ALBERT_CHECK( kronecker(0,0,1,0) == A(0,0) * A(1,0) );
  passed &= ALBERT_CHECK( kronecker(0,0,1,1) == A(0,0) * A(1,1) );
  passed &= ALBERT_CHECK( kronecker(0,0,1,2) == A(0,0) * A(1,2) );
  passed &= ALBERT_CHECK( kronecker(0,0,2,0) == A(0,0) * A(2,0) );
  passed &= ALBERT_CHECK( kronecker(0,0,2,1) == A(0,0) * A(2,1) );
  passed &= ALBERT_CHECK( kronecker(0,0,2,2) == A(0,0) * A(2,2) );

  passed &= ALBERT_CHECK( kronecker(0,1,0,0) == A(0,1) * A(0,0) );
  passed &= ALBERT_CHECK( kronecker(0,1,0,1) == A(0,1) * A(0,1) );
  passed &= ALBERT_CHECK( kronecker(0,1,0,2) == A(0,1) * A(0,2) );
  passed &= ALBERT_CHECK( kronecker(0,1,1,0) == A(0,1) * A(1,0) );
  passed &= ALBERT_CHECK( kronecker(0,1,1,1) == A(0,1) * A(1,1) );
  passed &= ALBERT_CHECK( kronecker(0,1,1,2) == A(0,1) * A(1,2) );
  passed &= ALBERT_CHECK( kronecker(0,1,2,0) == A(0,1) * A(2,0) );
  passed &= ALBERT_CHECK( kronecker(0,1,2,1) == A(0,1) * A(2,1) );
  passed &= ALBERT_CHECK( kronecker(0,1,2,2) == A(0,1) * A(2,2) );

  passed &= ALBERT_CHECK( kronecker(0,2,0,0) == A(0,2) * A(0,0) );
  passed &= ALBERT_CHECK( kronecker(0,2,0,1) == A(0,2) * A(0,1) );
  passed &= ALBERT_CHECK( kronecker(0,2,0,2) == A(0,2) * A(0,2) );
  passed &= ALBERT_CHECK( kronecker(0,2,1,0) == A(0,2) * A(1,0) );
  passed &= ALBERT_CHECK( kronecker(0,2,1,1) == A(0,2) * A(1,1) );
  passed &= ALBERT_CHECK( kronecker(0,2,1,2) == A(0,2) * A(1,2) );
  passed &= ALBERT_CHECK( kronecker(0,2,2,0) == A(0,2) * A(2,0) );
  passed &= ALBERT_CHECK( kronecker(0,2,2,1) == A(0,2) * A(2,1) );
  passed &= ALBERT_CHECK( kronecker(0,2,2,2) == A(0,2) * A(2,2) );

  passed &= ALBERT_CHECK( kronecker(1,0,0,0) == A(1,0) * A(0,0) );
  passed &= ALBERT_CHECK( kronecker(1,0,0,1) == A(1,0) * A(0,1) );
  passed &= ALBERT_CHECK( kronecker(1,0,0,2) == A(1,0) * A(0,2) );
  passed &= ALBERT_CHECK( kronecker(1,0,1,0) == A(1,0) * A(1,0) );
  passed &= ALBERT_CHECK( kronecker(1,0,1,1) == A(1,0) * A(1,1) );
  passed &= ALBERT_CHECK( kronecker(1,0,1,2) == A(1,0) * A(1,2) );
  passed &= ALBERT_CHECK( kronecker(1,0,2,0) == A(1,0) * A(2,0) );
  passed &= ALBERT_CHECK( kronecker(1,0,2,1) == A(1,0) * A(2,1) );
  passed &= ALBERT_CHECK( kronecker(1,0,2,2) == A(1,0) * A(2,2) );

  passed &= ALBERT_CHECK( kronecker(1,1,0,0) == A(1,1) * A(0,0) );
  passed &= ALBERT_CHECK( kronecker(1,1,0,1) == A(1,1) * A(0,1) );
  passed &= ALBERT_CHECK( kronecker(1,1,0,2) == A(1,1) * A(0,2) );
  passed &= ALBERT_CHECK( kronecker(1,1,1,0) == A(1,1) * A(1,0) );
  passed &= ALBERT_CHECK( kronecker(1,1,1,1) == A(1,1) * A(1,1) );
  passed &= ALBERT_CHECK( kronecker(1,1,1,2) == A(1,1) * A(1,2) );
  passed &= ALBERT_CHECK( kronecker(1,1,2,0) == A(1,1) * A(2,0) );
  passed &= ALBERT_CHECK( kronecker(1,1,2,1) == A(1,1) * A(2,1) );
  passed &= ALBERT_CHECK( kronecker(1,1,2,2) == A(1,1) * A(2,2) );

  passed &= ALBERT_CHECK( kronecker(1,2,0,0) == A(1,2) * A(0,0) );
  passed &= ALBERT_CHECK( kronecker(1,2,0,1) == A(1,2) * A(0,1) );
  passed &= ALBERT_CHECK( kronecker(1,2,0,2) == A(1,2) * A(0,2) );
  passed &= ALBERT_CHECK( kronecker(1,2,1,0) == A(1,2) * A(1,0) );
  passed &= ALBERT_CHECK( kronecker(1,2,1,1) == A(1,2) * A(1,1) );
  passed &= ALBERT_CHECK( kronecker(1,2,1,2) == A(1,2) * A(1,2) );
  passed &= ALBERT_CHECK( kronecker(1,2,2,0) == A(1,2) * A(2,0) );
  passed &= ALBERT_CHECK( kronecker(1,2,2,1) == A(1,2) * A(2,1) );
  passed &= ALBERT_CHECK( kronecker(1,2,2,2) == A(1,2) * A(2,2) );

  passed &= ALBERT_CHECK( kronecker(2,0,0,0) == A(2,0) * A(0,0) );
  passed &= ALBERT_CHECK( kronecker(2,0,0,1) == A(2,0) * A(0,1) );
  passed &= ALBERT_CHECK( kronecker(2,0,0,2) == A(2,0) * A(0,2) );
  passed &= ALBERT_CHECK( kronecker(2,0,1,0) == A(2,0) * A(1,0) );
  passed &= ALBERT_CHECK( kronecker(2,0,1,1) == A(2,0) * A(1,1) );
  passed &= ALBERT_CHECK( kronecker(2,0,1,2) == A(2,0) * A(1,2) );
  passed &= ALBERT_CHECK( kronecker(2,0,2,0) == A(2,0) * A(2,0) );
  passed &= ALBERT_CHECK( kronecker(2,0,2,1) == A(2,0) * A(2,1) );
  passed &= ALBERT_CHECK( kronecker(2,0,2,2) == A(2,0) * A(2,2) );

  passed &= ALBERT_CHECK( kronecker(2,1,0,0) == A(2,1) * A(0,0) );
  passed &= ALBERT_CHECK( kronecker(2,1,0,1) == A(2,1) * A(0,1) );
  passed &= ALBERT_CHECK( kronecker(2,1,0,2) == A(2,1) * A(0,2) );
  passed &= ALBERT_CHECK( kronecker(2,1,1,0) == A(2,1) * A(1,0) );
  passed &= ALBERT_CHECK( kronecker(2,1,1,1) == A(2,1) * A(1,1) );
  passed &= ALBERT_CHECK( kronecker(2,1,1,2) == A(2,1) * A(1,2) );
  passed &= ALBERT_CHECK( kronecker(2,1,2,0) == A(2,1) * A(2,0) );
  passed &= ALBERT_CHECK( kronecker(2,1,2,1) == A(2,1) * A(2,1) );
  passed &= ALBERT_CHECK( kronecker(2,1,2,2) == A(2,1) * A(2,2) );

  passed &= ALBERT_CHECK( kronecker(2,2,0,0) == A(2,2) * A(0,0) );
  passed &= ALBERT_CHECK( kronecker(2,2,0,1) == A(2,2) * A(0,1) );
  passed &= ALBERT_CHECK( kronecker(2,2,0,2) == A(2,2) * A(0,2) );
  passed &= ALBERT_CHECK( kronecker(2,2,1,0) == A(2,2) * A(1,0) );
  passed &= ALBERT_CHECK( kronecker(2,2,1,1) == A(2,2) * A(1,1) );
  passed &= ALBERT_CHECK( kronecker(2,2,1,2) == A(2,2) * A(1,2) );
  passed &= ALBERT_CHECK( kronecker(2,2,2,0) == A(2,2) * A(2,0) );
  passed &= ALBERT_CHECK( kronecker(2,2,2,1) == A(2,2) * A(2,1) );
  passed &= ALBERT_CHECK( kronecker(2,2,2,2) == A(2,2) * A(2,2) );

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
constexpr static bool accumulation(type_args<T> = {})
{
  bool passed = true;
  albert::Tensor<T, 1, 3> c = { 1, 2, 3};
  c(i) += c(i);
  passed &= ALBERT_CHECK( c(0) == 2 );
  passed &= ALBERT_CHECK( c(1) == 4 );
  passed &= ALBERT_CHECK( c(2) == 6 );

  c(i) -= c(i);
  passed &= ALBERT_CHECK( c(0) == 0 );
  passed &= ALBERT_CHECK( c(1) == 0 );
  passed &= ALBERT_CHECK( c(2) == 0 );

  albert::Tensor<T, 2, 3> A = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };

  A(i,j) += A(i,j);
  passed &= ALBERT_CHECK( A(0,0) == 2  );
  passed &= ALBERT_CHECK( A(0,1) == 4  );
  passed &= ALBERT_CHECK( A(0,2) == 6  );
  passed &= ALBERT_CHECK( A(1,0) == 8  );
  passed &= ALBERT_CHECK( A(1,1) == 10 );
  passed &= ALBERT_CHECK( A(1,2) == 12 );
  passed &= ALBERT_CHECK( A(2,0) == 14 );
  passed &= ALBERT_CHECK( A(2,1) == 16 );
  passed &= ALBERT_CHECK( A(2,2) == 18 );

  A(i,j) -= A(i,j);
  passed &= ALBERT_CHECK( A(0,0) == 0 );
  passed &= ALBERT_CHECK( A(0,1) == 0 );
  passed &= ALBERT_CHECK( A(0,2) == 0 );
  passed &= ALBERT_CHECK( A(1,0) == 0 );
  passed &= ALBERT_CHECK( A(1,1) == 0 );
  passed &= ALBERT_CHECK( A(1,2) == 0 );
  passed &= ALBERT_CHECK( A(2,0) == 0 );
  passed &= ALBERT_CHECK( A(2,1) == 0 );
  passed &= ALBERT_CHECK( A(2,2) == 0 );

  albert::Tensor<T, 2, 3> B = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };

  B(i,j) += B(j,i);
  passed &= ALBERT_CHECK( B(0,0) == 2 );
  passed &= ALBERT_CHECK( B(0,1) == 6 );
  passed &= ALBERT_CHECK( B(0,2) == 10 );
  passed &= ALBERT_CHECK( B(1,0) == 6 );
  passed &= ALBERT_CHECK( B(1,1) == 10 );
  passed &= ALBERT_CHECK( B(1,2) == 14 );
  passed &= ALBERT_CHECK( B(2,0) == 10 );
  passed &= ALBERT_CHECK( B(2,1) == 14 );
  passed &= ALBERT_CHECK( B(2,2) == 18 );

  return passed;
}

template <class T>
constexpr static bool tests(type_args<T> type = {})
{
  bool passed = true;
  passed &= bind(type);
  passed &= addition(type);
  passed &= subtraction(type);
  passed &= scalar_multiplication(type);
  passed &= scalar_division(type);
  passed &= contraction(type);
  passed &= projection(type);
  passed &= trace(type);
  passed &= transposition(type);
  passed &= accumulation(type);
  return passed;
}

int main()
{
  //constexpr
  bool i = tests(args<int>);
  // constexpr bool f = tests(args<float>);
  // constexpr bool d = tests(args<double>);
}
