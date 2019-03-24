#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"
#include "nn_utils/cce_cost.hh"
#include "layers/softmax_activation.hh"

#include "coordinates_dataset.hh"


void printmatrix(const Matrix& m){
	for(int i = 0 ; i < m.shape.x ; i++){
		for(int j = 0 ; j < m.shape.y ; j++)
			std::cout << m[i * m.shape.y + j] << " ";
		std::cout << std::endl;
	}

}



float computeAccuracy(const Matrix& predictions, const Matrix& targets);
int main() {

	srand( time(NULL) );

	CoordinatesDataset dataset(1000, 21);
	BCECost cce_cost;

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(2, 50)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(50, 1)));
	nn.addLayer(new SigmoidActivation("softmax_output"));

	// network training
	Matrix Y;
	for (int epoch = 0; epoch < 1001; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			Y = nn.forward(dataset.getBatches().at(batch));
			//Y.copyDeviceToHost();
			//dataset.getTargets().at(batch).copyDeviceToHost();
			//printmatrix(dataset.getTargets().at(batch));
			nn.backprop(Y, dataset.getTargets().at(batch));
			cost += cce_cost.cost(Y, dataset.getTargets().at(batch));
		}

		if (epoch % 100 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / dataset.getNumOfBatches()
						<< std::endl;
		}
	}

	// compute accuracy
	Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
	Y.copyDeviceToHost();

	float accuracy = computeAccuracy(
			Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
	std::cout 	<< "Accuracy: " << accuracy << std::endl;

	return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	int m = predictions.shape.x;
	int correct_predictions = 0;
	//printmatrix(predictions);

	for (int i = 0; i < m; i++) {
		float prediction = predictions[i] > 0.5 ? 1 : 0;
		if (prediction == targets[i]) {
			correct_predictions++;
		}
	}

	return static_cast<float>(correct_predictions) / m;
}
