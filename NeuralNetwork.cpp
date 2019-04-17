#include "NeuralNetwork.hpp"
#include <algorithm>
#include <cassert>
#include "Util.hpp"

NeuralNetwork::NeuralNetwork(vector<int> layers) {
  this->layersLength = layers.size();
  this->weightsLength = layers.size() - 1;

  this->L = new Matrix[this->layersLength];

  int i = 0;
  for (int l : layers) {
    L[i++] = Matrix(1, l);
  }

  this->W = new Matrix[this->weightsLength];

  int j = 0;
  i = 0;

  for (int l : layers) {
    if (j) {
      W[i] = Matrix(layers[j - 1], l);
      W[i++].random();
      // W[i++].example(j - 1);
    }
    j++;
  }

  this->deltas;
}
NeuralNetwork::~NeuralNetwork() {
  delete this->L;
  delete this->W;
}
// vector<double> NeuralNetwork::guess(vector<double>);

void NeuralNetwork::printLayers() {
  for (int i = 0; i < this->layersLength; i++) this->L[i].print(), printf("\n");
}
void NeuralNetwork::printWeights() {
  for (int i = 0; i < this->weightsLength; i++)
    this->W[i].print(), printf("\n");
}

void NeuralNetwork::train(pair<Matrix, Matrix> inOutSets, int repetitions) {
  this->L[0] = inOutSets.first;

  for (int i = 0; i < repetitions; i++) {
    this->feedforward();
    this->backpropagation(inOutSets.second);
  }
}

void NeuralNetwork::feedforward() {
  for (int i = 0; i < this->layersLength - 1; ++i) {
    // W[i], L[i] -> L[i + 1]
    this->L[i + 1].deconst();
    this->L[i + 1] = Matrix(this->L[i].rows, this->W[i].cols);
    this->L[i].dot(this->W[i], this->L[i + 1]);
    this->L[i + 1].applyFunction(Util::sigmoid);
  }
}
void NeuralNetwork::backpropagation(Matrix desiredOutputs) {
  this->deltas = new Matrix[this->layersLength];

  Matrix outputLayerCopy = Matrix(this->L[this->layersLength - 1].rows,
                                  this->L[this->layersLength - 1].cols);
  this->L[this->layersLength - 1].copy(outputLayerCopy);
  this->lastError = desiredOutputs - outputLayerCopy;
  outputLayerCopy.applyFunction(Util::sigmoid_derivative);

  int id, ie;
  this->deltas[id] = Matrix(this->lastError.rows, this->lastError.cols);
  this->lastError.hadamard(outputLayerCopy, this->deltas[id]);

  for (int i = this->layersLength - 1; i > 0; --i) {
    Matrix prevLayer = Matrix(this->L[i - 1].rows, this->L[i - 1].cols);
    Matrix weights = Matrix(this->W[i - 1].rows, this->W[i - 1].cols);
    this->L[i - 1].copy(prevLayer);
    this->W[i - 1].copy(weights);
    weights.transpose();

    this->lastError = Matrix(this->deltas[id].rows, weights.cols);
    this->deltas[id].dot(weights, this->lastError);
    prevLayer.applyFunction(Util::sigmoid_derivative);

    this->deltas[++id] = Matrix(this->lastError.rows, this->lastError.cols);
    this->lastError.hadamard(prevLayer, this->deltas[id]);

    prevLayer.deconst();
    weights.deconst();
  }
  for (int i = this->weightsLength - 1; i >= 0; --i) {
    Matrix _layer = Matrix(this->L[i].rows, this->L[i].cols);
    this->L[i].copy(_layer);
    _layer.transpose();

    Matrix res = Matrix(_layer.rows, deltas[this->weightsLength - 1 - i].cols);
    _layer.dot(deltas[this->weightsLength - 1 - i], res);

    this->W[i] += res;
    res.deconst();
    _layer.deconst();
  }
  outputLayerCopy.deconst();

  for (int n = 0; n < this->layersLength; ++n) this->deltas[n].deconst();
  delete[] this->deltas;
}