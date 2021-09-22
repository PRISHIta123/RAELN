from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

class NB:
    def __init__(self,training_data,labels,testing_data,tlabels):
        self.training_data= training_data
        self.labels= labels
        self.testing_data= testing_data
        self.tlabels=tlabels

    def training(self):
        classifier = GaussianNB()
        classifier.fit(self.training_data, self.labels)
        y_train_pred = classifier.predict(self.training_data)
        y_pred = classifier.predict(self.testing_data)
        y_pred = np.array(y_pred) 

        Train_Accuracy=accuracy_score(self.labels, y_train_pred)*100
        Test_Accuracy=accuracy_score(self.tlabels, y_pred)*100

        print("Training Accuracy: ",Train_Accuracy)
        print("Testing Accuracy: ",Test_Accuracy)

        return y_pred
