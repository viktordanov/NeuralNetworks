#include <chrono>
#include <iostream>
#include "Eigen/Dense"
#include "NeuralNetwork.hpp"
using Eigen::MatrixXd;

double ssigmoid(double n) { return 1.0 / (1.0 + std::exp(-n)); }
int main(int argc, char **args) {
  srand(time(NULL));

  NeuralNetwork nn = NeuralNetwork({3, 4, 1});

  // Note: Using 3 (neurons) as the # of columns
  MatrixXd input(4, 3);
  MatrixXd output(4, 1);

  input << 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1;
  output << 0, 1, 1, 0;

  int iteration = argc > 1 ? stoi(args[1]) : 10;

  printf("Training initialized\n");
  auto start = chrono::steady_clock::now();
  nn.train({input, output}, iteration);
  auto end = chrono::steady_clock::now();
  chrono::duration<double> duration_cast = end - start;

  printf("Time elapsed: %fs\n", duration_cast.count());

  printf("Layers:\n");
  nn.printLayers();
}