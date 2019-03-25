#include "neural_network.hh"
#include "nn_utils/nn_exception.hh"

NeuralNetwork::NeuralNetwork(float learning_rate) :
	learning_rate(learning_rate)
{ }

NeuralNetwork::~NeuralNetwork() {
	for (auto layer : layers) {
		delete layer;
	}
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix X) {
	Matrix Z = X;

	for (auto layer : layers) {
		Z = layer->forward(Z);
	}

	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix target) {
	dY.allocateMemoryIfNotAllocated(predictions.shape);
	/*
	for(int i = 0 ; i < target.shape.x ; i++){
    for(int j = 0 ; j < target.shape.y ; j++){
      std::cout << target.data_host.get()[i * target.shape.y + j] << " ";
    }
    std::cout << std::endl;
  }
	*/
	Matrix error = cce_cost.dCost(predictions, target, dY);

	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		error = (*it)->backprop(error, learning_rate);
	}

	cudaDeviceSynchronize();
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}
