#include "albert/albert.hpp"

template <class> struct print;

using namespace albert::grammar;
int main() {
  albert::Index<'i'> i;
  albert::Index<'j'> j;
  albert::Tensor<double, 1, 3> a = { 1, 2, 3 };
  a(0) = 1;
  albert::Tensor<double, 1, 3> b;
  albert::Tensor<double, 0, 0> c = { 2 };
  albert::Tensor<double, 2, 3> A = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };

  auto aa = A(0, j);
  aa = A(1, j);

  // b = d;

  // auto d = ((a(i) + b(i)) * δ(i,j)) / c;
  // auto e = pow(sin(tanh(exp(D(symmetrize(D((a(i) + b(i))(j),i) * c), i)))), d);

  // b = d;

  // print<decltype(d)> _;
}
