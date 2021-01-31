from .FileHandler import Importer as load
import numpy as np

class DataAnalysis:

    def __init__(self):
        self.features = [10, 100, 250, 500, 800, 1000, 1500]
        self.classifiers = ["NB", "SV", "LR", "RF"]
        self.metrics = ['accuracy', 'precision', 'recall', 'f1']
        self.functions = ['tf', 'tf_idf', 'log_tf_idf', 'log_tf_1']


    def GetCharts(self):
        dataSet = load(_path='Resources/Result/result.json',_format='json')
        data = dataSet.get_data()
        print(data)
        exit()
        x = np.array(self.features)
        x_smooth = np.linspace(x.min(), x.max(), 300)

        for fun in self.functions:

            for cla in self.classifiers:
                ls = []
                for fea in self.features:
                    ls.append()
                for met in self.metrics:
                    pass
