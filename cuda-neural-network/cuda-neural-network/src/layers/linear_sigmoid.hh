#pragma once
#include "nn_layer.hh"


class LinearSigmoidLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;

	Matrix W;
	Matrix b;

	Matrix Z;
	Matrix A;
	Matrix dA;
	Matrix dZeta;

	Matrix AT;
	Matrix WT;

	void initializeBiasWithZeros_sigmoid();
	void initializeWeightsRandomly_sigmoid();

	void computeAndStoreBackpropError_sigmoid(Matrix& dZ);
	void computeAndStoreLayerOutput_sigmoid(Matrix& A);
	void updateWeights_sigmoid(Matrix& dZ, float learning_rate);
	void updateBias_sigmoid(Matrix& dZ, float learning_rate);
  void SigmoidBackProp(Matrix& dZ);

public:
	LinearSigmoidLayer(std::string name, Shape W_shape);
	~LinearSigmoidLayer();

	Matrix& forward(Matrix& A);
	Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;


};
