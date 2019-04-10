# Deep-Neural-Network-Cuda
Cuda Implementation of Feedforward Neural Network

- To run: <br\>
  - cd cuda-neural-network/cuda-neural-network/src/ <br\>
  - sh get_data.sh <br\>
  - nvcc *.cu layers/*.cu nn_utils/*.cu -std=c++11 -o main <br\>
  - ./main 3 <br\>
