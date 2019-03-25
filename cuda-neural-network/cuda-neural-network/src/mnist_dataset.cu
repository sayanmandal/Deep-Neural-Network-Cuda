#include "mnist_dataset.hh"
#include <iostream>


#define BEGIN_OF_PIXELS 16
#define BEGIN_OF_LABELS 8

#define NUMBER_OF_PROBE_IMAGES 1000

void printmatrix2(const Matrix& m){
	for(int i = 0 ; i < m.shape.x ; i++){
		for(int j = 0 ; j < m.shape.y ; j++)
			std::cout << m[j * m.shape.x + i] << " ";
		std::cout << std::endl;
	}

}

MNISTDataset::MNISTDataset(int num_batches, size_t batch_size, int classes, DataSetType type){
  // Prepare some placeholders for our dataset
    this->batch_size = batch_size;
    this->num_batches = num_batches;
    FILE *file;
    long length;

    // Open file with images and check how long it is
    if (type == TRAIN) {
        file = fopen("data/train-images", "rb");
    } else if (type == TEST) {
        file = fopen("data/test-images", "rb");
    }
    fseek(file, 0, SEEK_END);
    length = ftell(file);
    rewind(file);

    //std::cout << length << std::endl;



    // Read whole file with images to the buffer
    unsigned char* bufferImages = (unsigned char *)malloc((length+1)*sizeof(unsigned char));
    fread(bufferImages, length, 1, file);
    fclose(file);

    // Open file with labels and check how long it is
    if (type == TRAIN) {
        file = fopen("data/train-labels", "rb");
    } else if (type == TEST) {
        file = fopen("data/test-labels", "rb");
    }
    fseek(file, 0, SEEK_END);
    length = ftell(file);
    rewind(file);

    // Read whole file with labels to the buffer
    unsigned char* bufferLabels = (unsigned char *)malloc((length+1)*sizeof(unsigned char));
    fread(bufferLabels, length, 1, file);
    fclose(file);

    // Keep size of the dataset in the property as we'll be using this value a lot!
    this->size = (int)((bufferImages[4] << 24) + (bufferImages[5] << 16) + (bufferImages[6] << 8) + bufferImages[7]);

    std::cout << this->size << std::endl;

    // Prepare value for mean image - we don't have to calculate exact mean value. Mean value of a 1000 images will be enough!
    float* meanImage = new float[28*28];
    for (int i = 0; i < 28*28; i++) {
        meanImage[i] = 0.0;
    }
    for (int image = 0; image < NUMBER_OF_PROBE_IMAGES; image++) {
        for (int i = 0; i < 28*28; i++) {
            meanImage[i] += bufferImages[BEGIN_OF_PIXELS + image*28*28 + i];
        }
    }
    for (int i = 0; i < 28*28; i++) {
        meanImage[i] /= NUMBER_OF_PROBE_IMAGES;
    }

    // Prepare value for standard deviation. It's the same story as above - 1000 images will be enough!
    float* stdDevImage = new float[28*28];
    for (int i = 0; i < 28*28; i++) {
        stdDevImage[i] = 0.0;
    }
    for (int image = 0; image < NUMBER_OF_PROBE_IMAGES; image++) {
        for (int i = 0; i < 28*28; i++) {
            stdDevImage[i] += pow(bufferImages[BEGIN_OF_PIXELS + image*28*28 + i] - meanImage[i], 2.0);
        }
    }
    for (int i = 0; i < 28*28; i++) {
        stdDevImage[i] = sqrt(stdDevImage[i] / (NUMBER_OF_PROBE_IMAGES - 1));
    }

    // Now let's read all images and convert them to floating point values
    // Together with this operation, let's perform simple image preprocessing
    //   Final image = (Input Image - Mean Image) / Std Dev Image
    this->images = new float*[this->size];
    *this->images = new float[this->size*28*28];
    for (int image = 1; image < this->size; image++) this->images[image] = this->images[image-1] + 28*28;
    for (int image = 0; image < this->size; image++) {
        for (int i = 0; i < 28*28; i++) {
            if (stdDevImage[i] > 1e-10) {
                // TODO: Test set shouldn't apply its mean and std dev values!
                // TODO: It should use the same values as were used in the training dataset!
                this->images[image][i] = (float)(bufferImages[BEGIN_OF_PIXELS + image*28*28 + i] - meanImage[i]) / stdDevImage[i];
            } else {
                this->images[image][i] = 0.0;
            }
        }
    }


    // And now let's read all labels from the MNIST dataset
    // Once we've read this values, let's convert them to the one-hot encoding
    //   4 -> 0000100000
    this->labels = new float*[this->size];
    *this->labels = new float[this->size*10];
    for (int image = 1; image < this->size; image++) this->labels[image] = this->labels[image-1] + 10;
    for (int image = 0; image < this->size; image++) {
        for(int i = 0; i < 10; i++) {
            this->labels[image][i] = 0;
        }
        int label = (int)bufferLabels[BEGIN_OF_LABELS + image];
        this->labels[image][label] = 1;
    }



    for(int i = 0 ; i < num_batches ; i++){
      batches.push_back(Matrix(Shape(batch_size, 28 * 28)));
      targets.push_back(Matrix(Shape(batch_size, classes)));

      batches[i].allocateMemory();
      targets[i].allocateMemory();

      //std::cout << "Allocated" << std::endl;

      for(int j = 0 ; j < batch_size ; j++){
        int index = i * num_batches + j;
        for(int k = 0 ; k < 28 * 28 ; k++){
          batches[i][k * batch_size + j] = this->images[index][k];
        }

        for(int k = 0 ; k < classes ; k++){
          targets[i][k * batch_size + j] = this->labels[index][k];
        }

      }

      //printmatrix2(targets[i]);


      batches[i].copyHostToDevice();
  		targets[i].copyHostToDevice();
      //break;
    }


}


int MNISTDataset::getNumOfBatches() {
	return num_batches;
}


std::vector<Matrix>& MNISTDataset::getBatches() {
	return batches;
}

std::vector<Matrix>& MNISTDataset::getTargets() {
	return targets;
}
