#ifndef MATRIX_H
#define MATRIX_H

#include <functional>
#include <vector>
using namespace std;

class Matrix {
  double** _m;

 public:
  int rows, cols;
  Matrix();
  Matrix(int, int);
  ~Matrix();

  double*& operator[](const int);
  Matrix& operator+=(const double);
  Matrix& operator+=(const Matrix&);
  Matrix& operator-=(const double);
  Matrix& operator-=(const Matrix&);
  Matrix& operator*=(const double);
  Matrix& operator*=(const Matrix&);
  Matrix operator+(const double);
  Matrix operator+(const Matrix&);
  Matrix operator-(const double);
  Matrix operator-(const Matrix&);
  Matrix operator*(const double);
  Matrix operator*(const Matrix&);

  static Matrix identity(int);
  void copy(Matrix&);
  void dot(Matrix&, Matrix&);
  void hadamard(Matrix&, Matrix&);
  void applyFunction(function<double(double)>);
  void initWith(double**, int, int);
  void transpose();
  void example(int);
  void random();
  void print();
  void deconst();

 private:
  void add(double);
  void add(Matrix);
  Matrix subtract(double);
  Matrix subtract(const Matrix&);
  void multiply(double);
  void multiply(Matrix);
  static double _rand();
};

#endif