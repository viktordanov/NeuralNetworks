#include <cstdio>
#include <vector>
#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;

class NeuralNetwork {
  MatrixXd *L;
  MatrixXd *W;

  int layersLength;
  int weightsLength;

  MatrixXd *deltas;
  MatrixXd lastError;

 public:
  NeuralNetwork(vector<int>);
  ~NeuralNetwork();
  // vector<double> guess(vector<double>);
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
  void train(pair<MatrixXd, MatrixXd>, int);
  void printLayers();
  void printWeights();
  void saveWeightsToFile();

 private:
  void feedforward();
  void backpropagation(MatrixXd);
};