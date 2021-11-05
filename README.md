# digit-recognizer
The Digit Recognizer Competition is a Kaggle challenge that aims to classify hand written digits (MNIST).
MNIST Dataset obtained from: https://www.kaggle.com/c/digit-recognizer/data

The keras model used for the classification is the following:

![model](https://user-images.githubusercontent.com/93343055/140495468-b463c740-6a15-4ea1-8527-5062e5536cd4.png)

Obtaining an accuracy of 98.75% in the training set.

Code folder contains:
- DataLoader.py --> loads and standardizes the dataset.
- Classifier.py --> creates and trains the model, it also allows its visualization and generates a csv with the predictions for the submission of the competition.
