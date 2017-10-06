kaggle classical intro contest : https://www.kaggle.com/c/digit-recognizer/data

this data set is acutally very pure and easy that a simple logistic regressoin with 100 iterations of gradient descent gets an accuracy score of 92.071%

the same logistic regression gets a score of 95% with 50 iterations  of GD and using batches of size 750

using a NN of 2 hidden layers each with 100 neurons and batches of size 2000 and 100 iterations gets a score of 97.071%

the code in main.py contains the code for the third model with some helpers in DataLoader.py and ModelLoader.py