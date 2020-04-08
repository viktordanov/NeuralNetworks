#include <algorithm>
#include <chrono>
#include <iterator>

#include "Eigen/Dense"
#include "NeuralNetwork.hpp"
#include "csv_parser.h"

using Eigen::MatrixXd;

// void* operator new(std::size_t bytes) {
//   std::cout << "Allocated " << bytes << " bytes\n";
//   return malloc(bytes);
// }
// void operator delete(void* ptr, std::size_t bytes) {
//   std::cout << "Freeing " << bytes << " bytes\n";
//   free(ptr);
// }

template <typename T>
void print(const std::vector<T> &vec, std::ostream &o) {
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(o, " "));
}

auto timeStart() { return chrono::steady_clock::now(); }
void timeEnd(const std::chrono::_V2::steady_clock::time_point &start, const char *msg) {
    auto end = chrono::steady_clock::now();
    chrono::duration<double> duration_cast = end - start;
    printf("[%s] Took: %fs\n", msg, duration_cast.count());
}

int main(int argc, char **args) {
    srand(time(NULL));

    NeuralNetwork nn({2, 3, 1});

    MatrixXd input(4, 2);
    MatrixXd output(4, 1);

    input << 0, 0, 0, 1, 1, 0, 1, 1;
    output << 0, 1, 1, 0;

    int iteration = argc > 1 ? stoi(args[1]) : 10;

    printf("Training initialized, running on %d threads\n", Eigen::nbThreads());

    auto start = timeStart();
    nn.train({input, output}, iteration);
    timeEnd(start, "Training");
    nn.printOutput();
    /*
        NeuralNetwork nn = NeuralNetwork({784, 20, 20, 20, 10});

        MatrixXd input(42000, 784);
        MatrixXd output(42000, 10);

        auto start = timeStart();
        loadFromCSV("./train.csv", input, output);
        timeEnd(start, "Loading train data");

        int iteration = argc > 1 ? stoi(args[1]) : 10;

        printf("Training initialized, running on %d threads\n", Eigen::nbThreads());

        start = timeStart();
        nn.train({input, output}, iteration);
        timeEnd(start, "Training");

        start = timeStart();
        loadTestFromCSV("./test.csv", input);
        timeEnd(start, "Loading train data");

        ofstream outCSV("./out.csv");

        if (outCSV.is_open()) {
            outCSV << "ImageId,Label\n";
        } else
            throw std::runtime_error("Main :: couldn't open out.csv for writing!");

        for (size_t i = 0; i < input.rows(); i++) {
            MatrixXd guess = nn.guess(input.row(i));
            if (i % 10000 == 0) {
                // std::cout << input.row(i)(215) << std::endl;
            }
            double maxValue = -1e9, index;
            for (int j{}; j < guess.cols(); ++j) {
                if (maxValue < guess.row(0)(j)) {
                    maxValue = guess.row(0)(j);
                    index = j;
                }
            }
            outCSV << (i + 1) << "," << index << "\n";
        }
        outCSV.close();
        // nn.printLayers();
        // nn.saveWeightsToFile();*/
}