#include "Matrix.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>

using namespace std;

Matrix::Matrix() {}
Matrix::Matrix(int rows, int cols) {
  this->rows = rows;
  this->cols = cols;
  this->_m = new double*[rows];
  for (int i = 0; i < rows; ++i) {
    this->_m[i] = new double[cols];

    for (int j = 0; j < cols; j++) {
      this->_m[i][j] = 0.0;
    }
  }
}
Matrix::~Matrix() {}

// OPERATORS

double*& Matrix::operator[](const int index) { return this->_m[index]; }

Matrix& Matrix::operator+=(const double rhs) {
  this->add(rhs);
  return *this;
}

Matrix Matrix::operator+(const double rhs) {
  this->add(rhs);
  return *this;
}

Matrix& Matrix::operator+=(const Matrix& rhs) {
  this->add(rhs);
  return *this;
}

Matrix Matrix::operator+(const Matrix& rhs) {
  this->add(rhs);
  return *this;
}

Matrix& Matrix::operator-=(const double rhs) {
  this->subtract(rhs);
  return *this;
}

Matrix Matrix::operator-(const double rhs) { return this->subtract(rhs); }

Matrix& Matrix::operator-=(const Matrix& rhs) {
  this->subtract(rhs);
  return *this;
}

Matrix Matrix::operator-(const Matrix& rhs) { return this->subtract(rhs); }

Matrix& Matrix::operator*=(const double rhs) {
  this->multiply(rhs);
  return *this;
}

Matrix Matrix::operator*(const double rhs) {
  this->multiply(rhs);
  return *this;
}
Matrix& Matrix::operator*=(const Matrix& rhs) {
  this->multiply(rhs);
  return *this;
}

Matrix Matrix::operator*(const Matrix& rhs) {
  this->multiply(rhs);
  return *this;
}

void Matrix::print() {
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      printf("%.5f ", _m[i][j]);
    }
    printf("\n");
  }
}

void Matrix::initWith(double** v, int I, int J) {
  assert(I == this->rows && J == this->cols);
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      this->_m[i][j] = v[i][j];
    }
  }
}

void Matrix::transpose() {
  Matrix clone = *this;
  swap(this->cols, this->rows);

  *this = Matrix(this->rows, this->cols);

  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      this->_m[i][j] = clone[j][i];
    }
  }
  clone.deconst();
}

Matrix Matrix::subtract(double number) {
  Matrix sub = Matrix(this->rows, this->cols);
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      sub[i][j] = this->_m[i][j] - number;
    }
  }
  return sub;
}
Matrix Matrix::subtract(const Matrix& b) {
  assert(b.rows == this->rows && b.cols == this->cols);
  Matrix sub = Matrix(this->rows, this->cols);
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      sub[i][j] = this->_m[i][j] - b._m[i][j];
    }
  }
  return sub;
}
void Matrix::add(double number) {
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      this->_m[i][j] += number;
    }
  }
}
void Matrix::add(Matrix b) {
  assert(b.rows == this->rows && b.cols == this->cols);
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      this->_m[i][j] += b[i][j];
    }
  }
}

void Matrix::multiply(double number) {
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      this->_m[i][j] *= number;
    }
  }
}

void Matrix::multiply(Matrix b) {
  assert(b.rows == this->rows && b.cols == this->cols);
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      this->_m[i][j] *= b[i][j];
    }
  }
}

void Matrix::applyFunction(function<double(double)> l) {
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      this->_m[i][j] = l(this->_m[i][j]);
    }
  }
}
void Matrix::copy(Matrix& ref) {
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      ref[i][j] = this->_m[i][j];
    }
  }
}
void Matrix::dot(Matrix& b, Matrix& ref) {
  assert(this->cols == b.rows);
  double help;

  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < b.cols; ++j) {
      help = 0;

      for (int k = 0; k < this->cols; ++k) {
        help += this->_m[i][k] * b._m[k][j];
      }

      ref[i][j] += help;
    }
  }
}
void Matrix::hadamard(Matrix& b, Matrix& ref) {
  assert(b.rows == this->rows && b.cols == this->cols);

  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      ref[i][j] = this->_m[i][j] * b[i][j];
    }
  }
}

Matrix Matrix::identity(int cols) {
  Matrix I = Matrix(cols, cols);
  for (int i = 0; i < cols; ++i) {
    I[i][i] = 1;
  }
  return I;
}
void Matrix::random() {
  for (int i = 0; i < this->rows; ++i) {
    for (int j = 0; j < this->cols; ++j) {
      this->_m[i][j] = Matrix::_rand();
    }
  }
}
void Matrix::deconst() {
  for (int i = 0; i < this->rows; i++) {
    if (this->_m[i] != NULL) {
      delete this->_m[i];
    }
  }
  if (this->_m != NULL) {
    delete[] this->_m;
  }
}

double Matrix::_rand() { return 2.0 * rand() / (RAND_MAX)-1; }

void Matrix::example(int x) {
  vector<vector<double>> e{{3, 4}, {4, 1}};
  vector<vector<vector<double>>> d{{
                                       {0.16219, 0.56182, 0.70785, -0.14477},
                                       {0.94583, 0.31048, 1.01085, 0.57410},
                                       {0.99160, 2.87010, 1.76342, 1.13101},
                                   },
                                   {
                                       {0.88314},
                                       {0.34502},
                                       {0.43294},
                                       {0.74408},
                                   }};
  for (int i = 0; i < e[x][0]; ++i) {
    for (int j = 0; j < e[x][1]; ++j) {
      this->_m[i][j] = d[x][i][j];
    }
  }
}