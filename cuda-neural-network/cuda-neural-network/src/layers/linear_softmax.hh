#pragma once
#include "nn_layer.hh"


class LinearSoftmaxLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;

	Matrix W;
	Matrix b;

	Matrix T;
	Matrix Z;
	Matrix A;
	Matrix dA;

	Matrix AT;
	Matrix WT;

	void initializeBiasWithZeros_softmax();
	void initializeWeightsRandomly_softmax();

	void computeAndStoreBackpropError_softmax(Matrix& dZ);
	void computeAndStoreLayerOutput_softmax(Matrix& A);
	void updateWeights_softmax(Matrix& dZ, float learning_rate);
	void updateBias_softmax(Matrix& dZ, float learning_rate);


public:
	LinearSoftmaxLayer(std::string name, Shape W_shape);
	~LinearSoftmaxLayer();

	Matrix& forward(Matrix& A);
	Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;


};
