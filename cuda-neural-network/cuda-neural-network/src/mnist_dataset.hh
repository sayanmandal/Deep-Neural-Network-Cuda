#pragma once

#include "nn_utils/matrix.hh"

#include <vector>

enum DataSetType{
    TRAIN,
    TEST
};

class MNISTDataset {
private:
	size_t batch_size;
  int num_batches;
  int size;
  float** images;
  float** labels;
	//size_t number_of_batches;

	std::vector<Matrix> batches;
	std::vector<Matrix> targets;

public:

	MNISTDataset(int num_batches, size_t batch_size, int classes, DataSetType type = TRAIN);

	int getNumOfBatches();
  int getSize();
  //void shuffle();
	std::vector<Matrix>& getBatches();
	std::vector<Matrix>& getTargets();

};
