# Machine-Learning-Model-for-CT-scans

Train machine learning models in order to classify the provided images. You are allowed to use all the necessary libraries to execute the tasks.
Write a python program to perform the following tasks:
1. Load the dataset for all three classes and resize each image to (32 x 32). Apply the required pre-processing steps to employ the data into Machine Learning / Deep Learning algorithms.
2. Construct a Convolutional Neural Network (CNN) architecture from scratch to extract features from the images. (HINT: Extract features for train and test set separately. Extract the features constructed by the convolutional layers from an intermediate dense layer. Please refrain from using any pre-trained model for implementing this step)
3. Apply the K-Nearest Neighbor (KNN) algorithm to the extracted features from CNN and find the optimal value of K. The value of K can be considered as [3, 5, 7, 9]. Determine the performance of the model using an appropriate performance metric. Draw a graph of K values and their corresponding performance in order to represent your results.
4. Apply Random Forest (RF) algorithm to the extracted features from CNN. Tune at least two hyperparameters using random search. Determine the model's optimal performance, the confusion matrix, and the value of hyperparameters producing the optimal performance.
5. Report the performance of each model and explain your results. (eg. overfitting, underfitting, etc.)
For steps 3 and 4, save the models into pickle files after training and load the saved models in order to predict the class of the test data. Make sure the models are validated.
