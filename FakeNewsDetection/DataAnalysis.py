from .FileHandler import Importer as load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class DataAnalysis:

    def __init__(self):
        self.feature_no = [10, 100, 250, 500, 800, 1000, 1500]
        self.classifiers = ["NB", "SV", "LR", "RF"]
        self.classifiers_names = ["Naive Bayes", "SVM", "Logistic Regression", "Random Forest"]
        self.metrics = ['accuracy', 'precision', 'recall', 'f1']
        self.scorer = ['tf', 'tf_idf', 'log_tf_idf', 'log_tf_1']
        self.scorer_names = ['TF', 'TF-IDF', 'log(TF+1)*IDF', 'log(TF+1)']


    def GetCharts(self):
        dataSet = load(_path='Output/result.json', _format='json')

        data = dataSet.get_data()

        # x = np.array(self.feature_no)
        # x_smooth = np.linspace(x.min(), x.max(), 300)

        yt = list(x / 100 for x in range(30, 100, 5))

        for c in self.classifiers:
            for m in self.metrics:
                filtered = data.query('classifier == "' + c + '"')[['classifier', m, 'function', 'featureCount']]
                plt.plot(self.feature_no, filtered[filtered['function'] == "tf"][m].tolist(), label='TF', marker='.')
                plt.plot(self.feature_no, filtered[filtered['function'] == "tf_idf"][m].tolist(), label='TF-IDF', marker='.')
                plt.plot(self.feature_no, filtered[filtered['function'] == "log_tf_idf"][m].tolist(), label='log(TF+1)*IDF', marker='.')
                plt.plot(self.feature_no, filtered[filtered['function'] == "log_tf_1"][m].tolist(), label='log(TF+1)', marker='.')
                plt.xticks(self.feature_no)
                plt.yticks(yt)
                plt.xlabel("Number of features")
                plt.ylabel(m + " value")
                plt.title(m + " metric values for " + c + " method")
                plt.legend()
                # plt.savefig("Evals/ALL.png")
                plt.show()

        filtered = data.query('classifier == "' + "RF" + '"')
        print(filtered)

        exit()


        for fun in self.scorer:

            for cla in self.classifiers:
                ls = []
                for fea in self.feature_no:
                    ls.append()
                for met in self.metrics:
                    pass
