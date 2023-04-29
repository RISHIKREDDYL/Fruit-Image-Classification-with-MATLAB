# Fruit-Image-Classification-with-MATLAB

This code implements a fruit and vegetable classification using a pre-trained Google Inception neural network. The dataset used consists of 33 different fruits and vegetables classes. The validation and test accuracy of this model are both 100%.

# Dataset
The dataset used in this project can be found at [Kaggle.](https://www.kaggle.com/datasets/sshikamaru/fruit-recognition/code)

# Implementation Details
The code starts by loading the image data and then reducing all labels to 200 images, before splitting the dataset between training, validation, and testing sets. The pre-trained GoogleNet is then loaded, and the last layers are replaced for the classification task. The new feature learner and classifier layer are defined, and the layer graph is replaced with the new layers.

The model is trained using stochastic gradient descent with momentum (SGDM) as the optimization algorithm. The training options include a learn rate schedule, a learn rate drop factor, a learn rate drop period, a mini-batch size, and a maximum number of epochs. The model is trained for five epochs, and the validation data is used to validate the model every 50 iterations.

After training the model, it is saved along with the training progress. The test set is then classified, and a confusion matrix is created to show the performance of the model. The accuracy is calculated and displayed on the test set.

# Usage
To use this code, download the dataset from the Kaggle link mentioned above and modify the path in the code to point to the correct location of the dataset on your system. The code can then be run on any platform that supports MATLAB.
