from .FileHandler import Importer as load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Plotter:

    def __init__(self):
        self.feature_no = ['10', '100', '250', '500', '800', '1000', '1500']
        self.classifiers = ["NB", "SV", "LR", "RF"]
        self.classifiers_names = ["Naive Bayes", "SVM", "Logistic Regression", "Random Forest"]
        self.metrics = ['accuracy', 'precision', 'recall', 'f1']
        self.scorer = ['tf', 'tf_idf', 'log_tf_idf', 'log_tf_1']
        self.scorer_names = ['TF', 'TF-IDF', 'log(TF+1)*IDF', 'log(TF+1)']


    def generate_plots(self):
        dataSet = load(_path='Output/result.json', _format='json')

        data = dataSet.get_data()

        # x = np.array(self.feature_no)
        # x_smooth = np.linspace(x.min(), x.max(), 300)

        yt = list(x / 100 for x in range(20, 85, 5))

        for c in self.classifiers:
            for m in self.metrics:
                filtered = data.query('classifier == "' + c + '"')[['classifier', m, 'function', 'featureCount']]
                plt.plot(self.feature_no, filtered[filtered['function'] == "tf"][m].tolist(), label='TF', marker='.')
                plt.plot(self.feature_no, filtered[filtered['function'] == "tf_idf"][m].tolist(), label='TF-IDF', marker='.')
                plt.plot(self.feature_no, filtered[filtered['function'] == "log_tf_idf"][m].tolist(), label='log(TF+1)*IDF', marker='.')
                plt.plot(self.feature_no, filtered[filtered['function'] == "log_tf_1"][m].tolist(), label='log(TF+1)', marker='.')
                # plt.xticks(None)
                plt.yticks(yt)
                plt.tick_params(labelbottom=False, bottom=False,)
                plt.xlabel(None)
                plt.ylabel(m + " value")
                plt.title(m + " metric values for " + self.classifiers_names[self.classifiers.index(c)] + " method")
                plt.legend()
                filtered.sort_values(by=['featureCount'], inplace=True)
                filtered['featureCount'] = filtered['featureCount'].astype(int)
                table = pd.pivot_table(filtered, values=m, index=['function'], columns=['featureCount']).to_numpy()
                the_table = plt.table(cellText=np.around(table, 7), loc='bottom', rowLabels=self.scorer_names, colLabels=self.feature_no)
                the_table.auto_set_font_size(False)
                the_table.set_fontsize(7)
                plt.savefig("Output/Figures/" + m + "_" + c + ".png")
                plt.show()
