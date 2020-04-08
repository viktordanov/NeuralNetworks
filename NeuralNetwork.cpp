#include "NeuralNetwork.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>

const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, " ", "\n");

void writeToCSVfile(string name, MatrixXd matrix) {
    ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
    file.close();
}

double sigmoid(double n) { return 1.0 / (1.0 + std::exp(-n)); }
double sigmoid_derivative(double n) { return n * (1.0 - n); }

NeuralNetwork::NeuralNetwork(vector<int> layers) {
    this->layersLength = layers.size();
    this->weightsLength = layers.size() - 1;

    this->L = new MatrixXd[this->layersLength];

    int i = 0;
    for (int l : layers) {
        L[i++] = MatrixXd(1, l);
    }

    this->W = new MatrixXd[this->weightsLength];

    int j = 0;
    i = 0;

    for (int l : layers) {
        if (j) {
            W[i] = MatrixXd(layers[j - 1], l);
            W[i++].setRandom();
            // W[i++].example(j - 1);
        }
        j++;
    }

    this->deltas;
}
NeuralNetwork::~NeuralNetwork() {
    delete[] this->L;
    delete[] this->W;
}
// vector<double> NeuralNetwork::guess(vector<double>);

void NeuralNetwork::printLayers() {
    for (int i = 0; i < this->layersLength; i++) std::cout << this->L[i] << endl;
}
void NeuralNetwork::printOutput() { std::cout << this->L[layersLength - 1] << endl; }
void NeuralNetwork::printWeights() {
    for (int i = 0; i < this->weightsLength; i++) std::cout << this->W[i] << endl;
}
void NeuralNetwork::saveWeightsToFile() {
    for (int i = 0; i < this->weightsLength; i++) {
        writeToCSVfile("weights." + to_string(i) + ".csv", this->W[i]);
    }
    writeToCSVfile("layer.o.csv", this->L[this->layersLength - 1]);
}

void NeuralNetwork::train(pair<MatrixXd, MatrixXd>&& inOutSets, int repetitions) {
    this->L[0] = inOutSets.first;

    for (int i = 0; i < repetitions; i++) {
        this->feedforward();
        this->backpropagation(inOutSets.second);
    }
}

MatrixXd NeuralNetwork::guess(MatrixXd testRow) {
    MatrixXd layers[this->layersLength];
    for (int i = 0; i < this->layersLength; ++i) {
        layers[i] = MatrixXd(1, this->L[i].cols());
    }
    layers[0] << testRow;
    for (int i = 0; i < this->layersLength - 1; ++i) {
        // W[i], L[i] -> L[i + 1]
        layers[i + 1] = layers[i] * this->W[i];
        layers[i + 1] = layers[i + 1].unaryExpr(&sigmoid);
    }
    return layers[this->layersLength - 1];
}

void NeuralNetwork::feedforward() {
    for (int i = 0; i < this->layersLength - 1; ++i) {
        // W[i], L[i] -> L[i + 1]
        this->L[i + 1] = this->L[i] * this->W[i];
        this->L[i + 1] = this->L[i + 1].unaryExpr(&sigmoid);
    }
}
void NeuralNetwork::backpropagation(MatrixXd& desiredOutputs) {
    this->deltas = new MatrixXd[this->layersLength];

    MatrixXd outputLayerCopy = this->L[this->layersLength - 1];
    this->lastError = desiredOutputs - outputLayerCopy;
    outputLayerCopy = outputLayerCopy.unaryExpr(&sigmoid_derivative);

    int id, ie;
    this->deltas[id] = this->lastError.cwiseProduct(outputLayerCopy);

    for (int i = this->layersLength - 1; i > 0; --i) {
        MatrixXd prevLayer = this->L[i - 1];
        MatrixXd weights = this->W[i - 1];
        weights.transposeInPlace();

        this->lastError = this->deltas[id] * weights;
        prevLayer = prevLayer.unaryExpr(&sigmoid_derivative);

        this->deltas[++id] = this->lastError.cwiseProduct(prevLayer);
    }
    for (int i = this->weightsLength - 1; i >= 0; --i) {
        MatrixXd _layer = this->L[i];
        _layer.transposeInPlace();

        MatrixXd res = _layer * deltas[this->weightsLength - 1 - i];

        this->W[i] += res * 0.1;
    }

    for (int n = 0; n < this->layersLength; ++n) this->deltas[n].resize(0, 0);
    delete[] this->deltas;
}