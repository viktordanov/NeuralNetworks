#include <cstdio>
#include <vector>
#include "Matrix.hpp"

using namespace std;

class NeuralNetwork {
  Matrix *L;
  Matrix *W;

  int layersLength;
  int weightsLength;

  Matrix *deltas;
  Matrix lastError;

 public:
  NeuralNetwork(vector<int>);
  ~NeuralNetwork();
  vector<double> guess(vector<double>);
  /**
   *  pair.first -> input layer as Matrix
   *    |
   *    -> length = neurons in input layer      and cols in input matrix
   *    -> width  = number of data sets         and rows in input matrix
   *
   *  pair.second -> output layer as Matrix
   *    |
   *    -> length = expected result in output   and cols in input matrix
   *    -> width  = number of data sets         and rows in input matrix
   *
   *   Rows and columns are flipped so as to avoid transposing when multiplying
   */
  void train(pair<Matrix, Matrix>, int);
  void printLayers();
  void printWeights();

 private:
  void feedforward();
  void backpropagation(Matrix);
};