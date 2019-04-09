#include <iostream>
#include <string>
#include <time.h>
#include <chrono>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/linear_relu.hh"
#include "layers/linear_sigmoid.hh"
#include "layers/linear_tanh.hh"
#include "layers/linear_softmax.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "layers/tanh_activation.hh"
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
int main(int argc, char* argv[]) {

	srand( time(NULL) );
  Matrix Y;

  CCECost cce_cost;

  NeuralNetwork nn;

	if(argc != 2){
		std::cout << "Usage: ./a.out <number>" << std::endl;
		return 0;
	}

	int arg = std::stoi(argv[1]);

	//std::cout << arg <<std::endl;

  //nn.addLayer(new LinearLayer("linear_1", Shape(28 * 28, 256)));
  //nn.addLayer(new	tanhActivation("relu_1"));

	//nn.addLayer(new LinearTanhLayer("Ltanh", Shape(28 * 28, 256)));
	/*nn.addLayer(new LinearTanhLayer("Ltanh", Shape(256, 256)));
	nn.addLayer(new LinearTanhLayer("Ltanh", Shape(256, 256)));
	nn.addLayer(new LinearTanhLayer("Ltanh", Shape(256, 256)));
	nn.addLayer(new LinearTanhLayer("Ltanh", Shape(256, 256)));
	nn.addLayer(new LinearTanhLayer("Ltanh", Shape(256, 256)));
	nn.addLayer(new LinearTanhLayer("Ltanh", Shape(256, 256)));
	nn.addLayer(new LinearTanhLayer("Ltanh", Shape(256, 256)));
	*/

	if(arg == 1){
		nn.addLayer(new LinearLayer("linear_1", Shape(28*28, 256)));
	  nn.addLayer(new tanhActivation("relu_1"));

		nn.addLayer(new LinearLayer("linear_1", Shape(256, 256)));
	  nn.addLayer(new tanhActivation("relu_1"));

		nn.addLayer(new LinearLayer("linear_1", Shape(256, 256)));
	  nn.addLayer(new tanhActivation("relu_1"));

		nn.addLayer(new LinearLayer("linear_1", Shape(256, 256)));
	  nn.addLayer(new tanhActivation("relu_1"));

		nn.addLayer(new LinearLayer("linear_1", Shape(256, 256)));
	  nn.addLayer(new tanhActivation("relu_1"));

		nn.addLayer(new LinearLayer("linear_1", Shape(256, 256)));
	  nn.addLayer(new tanhActivation("relu_1"));

		nn.addLayer(new LinearLayer("linear_1", Shape(256, 256)));
	  nn.addLayer(new tanhActivation("relu_1"));

		nn.addLayer(new LinearLayer("linear_8", Shape(256, 10)));
	  nn.addLayer(new softmaxActivation("softmax_output"));

	}else if(arg == 2){
		nn.addLayer(new LinearTanhLayer("Ltanh", Shape(28*28, 256)));
		nn.addLayer(new LinearTanhLayer("Ltanh", Shape(256, 256)));
		nn.addLayer(new LinearTanhLayer("Ltanh", Shape(256, 256)));
		nn.addLayer(new LinearTanhLayer("Ltanh", Shape(256, 256)));
		nn.addLayer(new LinearTanhLayer("Ltanh", Shape(256, 256)));
		nn.addLayer(new LinearTanhLayer("Ltanh", Shape(256, 256)));
		nn.addLayer(new LinearTanhLayer("Ltanh", Shape(256, 256)));
		nn.addLayer(new LinearSoftmaxLayer("linear_8", Shape(256, 10)));

	}else{
		nn.addLayer(new LinearLayer("linear_1", Shape(28*28, 256)));
	  nn.addLayer(new ReLUActivation("relu_1"));

		nn.addLayer(new LinearLayer("linear_8", Shape(256, 10)));
	  nn.addLayer(new softmaxActivation("softmax_output"));
	}

	//nn.addLayer(new LinearLayer("linear_1", Shape(256, 256)));
  //nn.addLayer(new tanhActivation("relu_1"));
/*
	nn.addLayer(new LinearLayer("linear_1", Shape(256, 256)));
  nn.addLayer(new tanhActivation("relu_1"));

	nn.addLayer(new LinearLayer("linear_1", Shape(256, 256)));
  nn.addLayer(new tanhActivation("relu_1"));

	nn.addLayer(new LinearLayer("linear_1", Shape(256, 256)));
  nn.addLayer(new tanhActivation("relu_1"));

	nn.addLayer(new LinearLayer("linear_1", Shape(256, 256)));
  nn.addLayer(new tanhActivation("relu_1"));

	nn.addLayer(new LinearLayer("linear_1", Shape(256, 256)));
  nn.addLayer(new tanhActivation("relu_1"));
	*/


	//nn.addLayer(new LinearSoftmaxLayer("linear_8", Shape(256, 10)));

  MNISTDataset mnist(num_batches_train, batch_size, classes);
  //float cost = 0.0;



	double ftime = 0.0;
	double btime = 0.0;
	double ttime = 0.0;

	auto st = std::chrono::steady_clock::now();

  for (int epoch = 0; epoch < 601; epoch++) {
		float cost = 0.0;
    for(int batch = 0 ; batch < num_batches_train ; batch++){
       //Y = mnist.getBatches().at(batch);
       //printmatrix1(Y);
			 auto start = std::chrono::steady_clock::now();
       Y = nn.forward(mnist.getBatches().at(batch));
			 auto end = std::chrono::steady_clock::now();
			 ftime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			 auto start1 = std::chrono::steady_clock::now();
       nn.backprop(Y, mnist.getTargets().at(batch));
			 auto end1 = std::chrono::steady_clock::now();
			 btime += std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();

       cost += cce_cost.cost(Y, mnist.getTargets().at(batch));
    }
    if(epoch%100 == 0){
      std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / mnist.getNumOfBatches()
						<< std::endl;
    }
  }

	auto en = std::chrono::steady_clock::now();
	ttime = std::chrono::duration_cast<std::chrono::milliseconds>(en - st).count();

	std::cout <<  "forward time: " << ftime/(602) << std::endl;
	std::cout << "backward time: " << btime/(602) << std::endl;
	std::cout << "Total time: " << ttime/(602) << std::endl;

	std::cout << "Testing..." << std::endl;

	int correct_predictions = 0;

	MNISTDataset mnist_test(num_batches_test, batch_size, classes, TEST);

	for(int batch = 0 ; batch < num_batches_test;  batch++){
		Y = nn.forward(mnist_test.getBatches().at(batch));
		Y.copyDeviceToHost();
		correct_predictions += computeAccuracyClasses_mnist(Y, mnist_test.getTargets().at(batch), classes);
	}

	float accuracy = (float)correct_predictions / (num_batches_test * batch_size);
	std::cout << "Accuracy: " << accuracy*100 << std::endl;




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
