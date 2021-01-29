import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as _RFC
from sklearn.linear_model import LogisticRegression as _LR
from sklearn import naive_bayes as _NB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class Classifier:

    train_data = None
    train_labels = []
    test_data = None
    test_labels = []
    classifier = None
    model = None
    prediction = []
    evaluator = None

    def __init__(self, train_feature_matrix, train_labels, test_feature_matrix, test_labels, method=None):
        self.train_data = sparse.csr_matrix(train_feature_matrix)
        self.train_labels = train_labels
        self.test_data = sparse.csr_matrix(test_feature_matrix)
        self.test_labels = test_labels

        if method is not None:
            if method in ["NB", "SV", "LR", "RF"]:
                self.classifier = self.__getattribute__(method)
                self.classifier()
            else:
                print(":: [ERROR] Invalid Classifier Requested [NB / SV / LR / RF] ... --ABORTING")
                exit(500)

    def NB(self):   # Naive Bayes
        self.model = _NB.MultinomialNB()
        self.model.fit(self.train_data, self.train_labels)
        self.prediction = self.model.predict(self.test_data)

    def LR(self):   # logistic regression
        self.model = _LR()
        self.model.fit(self.train_data, self.train_labels)
        self.prediction = self.model.predict(self.test_data)

    def SV(self):   # Support-Vector Machine
        self.model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        self.model.fit(self.train_data, self.train_labels)
        self.prediction = self.model.predict(self.test_data)

    def RF(self):   # Random Forest
        self.model = _RFC(max_depth=2, random_state=0)
        self.model.fit(self.train_data, self.train_labels)
        self.prediction = self.model.predict(self.test_data)

    def get_prediction(self, classifier):
        if classifier in ["NB", "SV", "LR", "RF"]:
            self.classifier = self.__getattribute__(classifier)
            self.classifier()
        else:
            print(":: [ERROR] Invalid Classifier Requested [NB / SV / LR / RF] ... --ABORTING")
            exit(500)
        return self.prediction

    def evaluation(self, evaluator):
        if evaluator in ["accuracy", "precision", "recall", "f1"]:
            self.evaluator = self.__getattribute__(evaluator)
            self.evaluator()
        else:
            print(":: [ERROR] Invalid Evaluator Requested [accuracy / precision / recall / f1] ... --ABORTING")
            exit(500)

    def accuracy(self):
        return accuracy_score(y_true=self.test_labels, y_pred=self.prediction)

    def precision(self):
        return precision_score(y_true=self.test_labels, y_pred=self.prediction)

    def recall(self):
        return recall_score(y_true=self.test_labels, y_pred=self.prediction)

    def f1(self):
        return f1_score(y_true=self.test_labels, y_pred=self.prediction)
