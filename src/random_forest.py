from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RF:
    def __init__(self,training_data,labels,testing_data,tlabels):
        self.training_data= training_data
        self.labels= labels
        self.testing_data= testing_data
        seld.tlabels=tlabels

    def training(self):
        classifier = RandomForestClassifier() 
        classifier.fit(self.training_data, self.labels)
        y_pred = classifier.predict(self.testing_data)
        y_pred = np.array(y_pred) 

        Accuracy=accuracy_score(self.tlabels, y_pred)*100

        print(Accuracy)
