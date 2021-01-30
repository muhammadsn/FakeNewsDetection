from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn import naive_bayes as NB
from scipy import sparse


class CrossValidation:

    dataset = None
    labels = []
    scores = []
    k = 5

    def __init__(self, train, labels, n_fold):
        self.dataset = sparse.csr_matrix(train)
        self.labels = labels
        self.k = n_fold

    def validate(self, method, metric):
        classifier = self.__getattribute__(method)
        clf = classifier()
        self.scores = cross_val_score(estimator=clf, X=self.dataset, y=self.labels, cv=self.k, scoring=metric)
        return self.scores

    def NB(self):
        return NB.MultinomialNB()

    def SV(self):
        return SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

    def RF(self):
        return RF(max_depth=2, random_state=0)

    def LR(self):
        return LR(max_iter=1500)

    def get_validation_scores(self):
        return self.scores
