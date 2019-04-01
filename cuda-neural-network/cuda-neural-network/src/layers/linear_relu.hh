#pragma once
#include "nn_layer.hh"
class LinearReluLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;

	Matrix W;
	Matrix b;

	Matrix Z;
	Matrix A;
	Matrix dA;

	Matrix AT;
	Matrix WT;

	void initializeBiasWithZeros_Relu();
	void initializeWeightsRandomly_Relu();

  void ReluBackProp(Matrix& dZ);
	void computeAndStoreBackpropError_Relu(Matrix& dZ);
	void computeAndStoreLayerOutput_Relu(Matrix& A);
	void updateWeights_Relu(Matrix& dZ, float learning_rate);
	void updateBias_Relu(Matrix& dZ, float learning_rate);

public:
	LinearReluLayer(std::string name, Shape W_shape);
	~LinearReluLayer();

	Matrix& forward(Matrix& A);
	Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;


};
