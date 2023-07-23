#include <chrono>
#include <ctime>
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"

int main(int argc, char **args) {
  srand(time(NULL));

  NeuralNetwork nn = NeuralNetwork({3, 4, 1});

  // Note: Using 3 (neurons) as the # of columns
  Matrix input = Matrix(4, 3);
  Matrix output = Matrix(4, 1);

  double **input_data = new double *[4];
  input_data[0] = new (double[3]){0, 0, 1};
  input_data[1] = new (double[3]){0, 1, 1};
  input_data[2] = new (double[3]){1, 0, 1};
  input_data[3] = new (double[3]){1, 1, 1};

  double **output_data = new double *[4];
  output_data[0] = new (double[1]){0};
  output_data[1] = new (double[1]){1};
  output_data[2] = new (double[1]){1};
  output_data[3] = new (double[1]){0};

  input.initWith(input_data, 4, 3);
  output.initWith(output_data, 4, 1);

  int iteration = argc > 1 ? stoi(args[1]) : 10;

  printf("Training initialized\n");
  auto start = chrono::steady_clock::now();
  nn.train({input, output}, iteration);
  auto end = chrono::steady_clock::now();
  chrono::duration<double> duration_cast = end - start;

  printf("Time elapsed: %fs\n", duration_cast.count());

  printf("Layers:\n");
  nn.printLayers();

  std::vector<double> guess  = nn.guess({0, 0});
}
