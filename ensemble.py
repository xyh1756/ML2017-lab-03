import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import math

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier=DecisionTreeClassifier, n_weakers_limit=5):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''

        # number of classifiers
        self.M = self.n_weakers_limit
        # number of samples
        self.N = len(X)
        # weights of samples
        self.w = np.array([1.0 / self.N for i in range(self.N)])
        # weights of classifiers
        self.D = np.zeros(self.M)
        a = self.w
        clfs = []
        for i in range(self.M):
            clf = self.weak_classifier()
            clf.fit(X = X, y = y, sample_weight=a)
            error_rate = max(1 - clf.score(X, y, sample_weight=self.w), 10 ** (-8))
            # update weights of classifiers
            self.D[i] = 0.5 * math.log((1 - error_rate) / error_rate)
            # update weights of samples
            y_pre = clf.predict(X)
            Z =0
            for j in range(self.N):
                Z += self.w[j] * math.exp(-self.w[j] * y_pre[j] * y[j])
            for j in range(self.N):
                self.w[j] = self.w[j] * math.exp(-self.w[j] * y_pre[j] * y[j])
            score = 0
            for j in range(self.N):
                if y[j] == y_pre[j]:
                    score += 1
            score = float(score)/self.N
            print('Classifier %d, score: %5f' % (i, score))
            clfs.append(clf)

        self.clfs = clfs


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        y_predict_list = [[] for i in range(self.n_weakers_limit)]
        for i in range(self.n_weakers_limit):
            y_predict_list[i].append(self.clfs[i].predict(X))
        y_predict_score = []
        for i in range(self.n_weakers_limit):
            s = 0
            for j in range(self.N):
                s += self.D[i] * y_predict_list[i][j]
            y_predict_score.append(s)
        return y_predict_score

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        y_predict_list = []
        for i in range(self.n_weakers_limit):
            y_predict_list.append(self.clfs[i].predict(X))
        y_predict = []
        for i in range(len(X)):
            s = 0
            for j in range(self.n_weakers_limit):
                s += self.D[j] * y_predict_list[j][i]
            if s > threshold:
                y_predict.append(1)
            else:
                y_predict.append(-1)
        print(self.D)
        return y_predict

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
