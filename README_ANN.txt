Implement a Multi Layer ( One Input, One Output and One or more Hidden Layers)
ANN for handwritten digit classification using MNIST dataset.
---------------------------------------------------------------------------------------------------
Objective:-The  main aim of the problem is to recognize the handwritten digit.

>>The method used to do the classification is Artificial Neural Network. 

>>MNIST dataset is loaded from tensorflow module(70000 ,28 pixel image) with 10 labels.

>>Neural Network having one input layer, 2 hidden layer and 1 output layer is created.

>>Then MNIST dataset is trained using this neural network with specified number of 
epoch(iterations) and model is saved.

>>During testing, the trained model is loaded and new handwritten image is grayscaled and reshaped to 28 pixel.

>>Then the image is given to the trained model and digit is categorized to either of 10 labels.