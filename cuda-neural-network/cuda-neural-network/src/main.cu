#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/linear_relu.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"
#include "nn_utils/cce_cost.hh"
#include "layers/softmax_activation.hh"

#include "coordinates_dataset.hh"
#include "mnist_dataset.hh"

#define num_batches_train 100
#define batch_size 100

#define num_batches_test 20
#define classes 10



void printmatrix1(const Matrix& m){
	for(int i = 0 ; i < m.shape.x ; i++){
		for(int j = 0 ; j < m.shape.y ; j++)
			std::cout << m[j * m.shape.x + i] << " ";
		std::cout << std::endl;
	}

}



float computeAccuracy_mnist(const Matrix& predictions, const Matrix& targets);
int computeAccuracyClasses_mnist(const Matrix& predictions, const Matrix& targets, int k);
int main() {

	srand( time(NULL) );
  Matrix Y;

  CCECost cce_cost;

  NeuralNetwork nn;
  nn.addLayer(new LinearLayer("linear_1", Shape(28 * 28, 256)));
  nn.addLayer(new ReLUActivation("relu_1"));
	//nn.addLayer(new LinearReluLayer("linear_relu", Shape(256, 128)));
	/*
  nn.addLayer(new LinearLayer("linear_2", Shape(512, 512)));
	nn.addLayer(new SigmoidActivation("relu_2"));
	*/
	/*

  nn.addLayer(new LinearLayer("linear_3", Shape(512, 512)));
	nn.addLayer(new SigmoidActivation("relu_3"));
	nn.addLayer(new LinearLayer("linear_4", Shape(512, 512)));
	nn.addLayer(new SigmoidActivation("relu_4"));
	nn.addLayer(new LinearLayer("linear_5", Shape(512, 256)));
	nn.addLayer(new SigmoidActivation("relu_5"));
	nn.addLayer(new LinearLayer("linear_6", Shape(256, 256)));
	nn.addLayer(new ReLUActivation("relu_6"));
	nn.addLayer(new LinearLayer("linear_7", Shape(256, 128)));
	nn.addLayer(new ReLUActivation("relu_7"));

*/
	nn.addLayer(new LinearLayer("linear_8", Shape(256, 10)));
  nn.addLayer(new softmaxActivation("softmax_output"));

  MNISTDataset mnist(num_batches_train, batch_size, classes);
  //float cost = 0.0;

  for (int epoch = 0; epoch < 601; epoch++) {
		float cost = 0.0;
    for(int batch = 0 ; batch < num_batches_train ; batch++){
       //Y = mnist.getBatches().at(batch);
       //printmatrix1(Y);
       Y = nn.forward(mnist.getBatches().at(batch));
       nn.backprop(Y, mnist.getTargets().at(batch));
       cost += cce_cost.cost(Y, mnist.getTargets().at(batch));
    }
    if(epoch%100 == 0){
      std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / mnist.getNumOfBatches()
						<< std::endl;
    }
  }


	std::cout << "Testing..." << std::endl;

	int correct_predictions = 0;

	MNISTDataset mnist_test(num_batches_test, batch_size, classes, TEST);

	for(int batch = 0 ; batch < num_batches_test;  batch++){
		Y = nn.forward(mnist_test.getBatches().at(batch));
		Y.copyDeviceToHost();
		correct_predictions += computeAccuracyClasses_mnist(Y, mnist_test.getTargets().at(batch), classes);
	}

	float accuracy = (float)correct_predictions / (num_batches_test * batch_size);
	std::cout << "Accuracy: " << accuracy << std::endl;




	return 0;
}

float computeAccuracy_mnist(const Matrix& predictions, const Matrix& targets) {
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


int computeAccuracyClasses_mnist(const Matrix& predictions, const Matrix& targets, int k) {
	int m = predictions.shape.x;
	int correct_predictions = 0;
	//printmatrix(predictions);

	for (int i = 0; i < m; i++) {
		float _max = 0.0;
		float _maxt = 0.0;
		int label = 0;
		int labely = 0;
		for(int j = 0 ; j < k ; j++){
			if(predictions[j * m + i] > _max){
				_max = predictions[j * m + i];
				label = j;
			}
			if(targets[j * m + i] > _maxt){
				_maxt = targets[j * m + i];
				labely = j;
			}
		}
		if(label == labely)	correct_predictions++;
	}
	return correct_predictions;
	//return static_cast<float>(correct_predictions) / m;
}
