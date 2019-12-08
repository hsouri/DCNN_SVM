import numpy as np
from sklearn import metrics
from sklearn import svm
import tools



class SVM:

    def __init__(self, train_data, train_labels, test_data, test_labels, file):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.num_of_classes = max(train_labels) - min(train_labels) + 1
        self.file = file
        self.choice = "None"
        self.linear_svm = svm.SVC()
        self.poly_svm = svm.SVC(C=10, kernel='poly')
        self.RBF_svm = svm.SVC(C=10, kernel='rbf')

    def train(self, choice):
        self.choice = choice
        if choice == "linear":
            self.linear_svm.fit(self.train_data, self.train_labels)
        elif choice == "polynomial":
            self.poly_svm.fit(self.train_data, self.train_labels)
        elif choice == "RBF":
            self.RBF_svm.fit(self.train_data, self.train_labels)
        return self

    def test(self):

        if self.choice == "linear":
            classifier  = self.linear_svm
        elif self.choice == "polynomial":
            classifier = self.poly_svm
        elif self.choice == "RBF":
            classifier = self.RBF_svm

        predicted_labels = classifier.predict(self.test_data)
        np.save(self.choice + '_svm_predicted_labels.npy', predicted_labels)
        accuracy = 100 * metrics.accuracy_score(self.test_labels, predicted_labels)
        experiment_name = self.choice + '_svm'
        tools.write_results(accuracy, self.file, experiment_name)
        print(experiment_name + " accuracy:", accuracy),
        # confusion_matrix = tools.confusion_matrix(predicted_labels)
        # tools.plot_confusion_matrix(confusion_matrix, self.num_of_classes, experiment_name)











