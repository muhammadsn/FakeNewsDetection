from .FileHandler import Importer as load
from .FileHandler import Exporter as save
import pandas as pd
import numpy as np
import sys

pd.set_option("display.max_rows", None, "display.max_columns", None)
np.set_printoptions(threshold=sys.maxsize)


class FeatureExtractor:
    all_words = []
    vocabulary = []
    dataset = pd.DataFrame()
    tf = pd.DataFrame()
    idf = pd.DataFrame()
    feature_scores = pd.DataFrame()
    feature_count = 0
    feature_list = []
    resource_path = ""
    method = ""

    def __init__(self, dataset, resource_path, feature_count, selection_method):
        self.feature_count = feature_count
        self.dataset = dataset
        self.resource_path = resource_path
        self.method = selection_method

        for idx, row in dataset.iterrows():
            words = row['text'] + row['title'] + row['description']
            self.all_words += words
        self.vocabulary = list(dict.fromkeys(self.all_words))
        self.dataset['body'] = None
        df = pd.DataFrame(columns=['body', 'class'])
        df['body'] = dataset['text'] + dataset['title'] + dataset['description']
        df['class'] = dataset['class']
        for idx, row in df.iterrows():
            self.dataset['body'].at[idx] = " ".join(row['body'])

    def get_features(self):
        if self.method == "MI":
            self.mutual_information_scorer()
            self.feature_scores.columns = ['word', 'score']
        elif self.method == "TF":
            self.tf_based_scorer()
            self.feature_scores.columns = ['word', 'score']
        f = self.feature_scores.sort_values(by=['score'], ascending=False)
        f = f.head(self.feature_count)
        self.feature_list = f.to_dict('records')
        return self.feature_list

    def mutual_information_scorer(self):
        self.feature_scores = load("json", self.resource_path + "mi_scores.json")
        if self.feature_scores.get_status():
            self.feature_scores = self.feature_scores.get_data()
        else:
            print(":: Calculating Term Frequencies for Vocabulary...", end="\t")
            score_list = []
            collections = [self.dataset.loc[self.dataset['class'] == 1], self.dataset.loc[self.dataset['class'] == 0]]
            cnt = 1
            for word in self.vocabulary:
                mi = [0, 0]
                A = [0, 0]
                C = [0, 0]
                for cat in [0, 1]:
                    N = collections[cat].shape[0]
                    for idx, row in collections[cat].iterrows():
                        words = row['text'] + row['title'] + row['description']
                        if word in words:
                            A[cat] += words.count(word)
                        else:
                            C[cat] += 1
                B = [A[1], A[0]]
                mi[0] = np.log((A[0] * N)/((A[0] + C[0]) * (A[0] + B[0])))
                mi[1] = np.log((A[1] * N)/((A[1] + C[1]) * (A[1] + B[1])))
                # print(cnt, word, np.max(mi))
                cnt += 1
                score_list.append({"word": word, "mi_max": np.max(mi)})
            df = pd.DataFrame(score_list, columns=["word", "mi_max"])
            print("--DONE!")

            print(":: Saving MI Scores to file...", end="\t")
            save(df, "json", self.resource_path + "mi_scores.json")
            print("--DONE!")

    def tf_based_scorer(self):
        self.calculate_tf()
        self.feature_scores = self.tf.sort_values(by=['tf'], ascending=False)
        f = self.feature_scores.head(self.feature_count)
        self.feature_list = f['word'].tolist()

    def calculate_tf(self):
        self.tf = load("json", self.resource_path + "tf.json")
        if self.tf.get_status():
            self.tf = self.tf.get_data()
        else:
            print(":: Calculating Term Frequencies for Dataset...", end="\t")
            wcd = {}

            for w in self.all_words:
                if w not in wcd.keys():
                    wcd[w] = self.all_words.count(w)
            self.tf = pd.DataFrame.from_dict(wcd, orient='index').reset_index()
            self.tf.columns = ['word', 'tf']
            print("--DONE!")

            print(":: Saving Term Frequencies to file...", end="\t")
            save(self.tf, "json", self.resource_path + "tf.json")
            print("--DONE!")

    def calculate_idf(self):
        self.idf = load("json", self.resource_path + "idf.json")
        if self.idf.get_status():
            self.idf = self.idf.get_data()
        else:
            print(":: Calculating Term Frequencies for Dataset...", end="\t")
            widf = {}
            N = self.dataset.shape[0]
            for w in self.vocabulary:
                df = 0
                res = self.dataset[self.dataset['body'].str.contains(w)]
                df += res.shape[0]
                if df == 0:
                    for idx, row in self.dataset.iterrows():
                        words = row['text'] + row['title'] + row['description']
                        if w in words:
                            df += 1
                widf[w] = np.log((N+1)/df)

            self.idf = pd.DataFrame.from_dict(widf, orient='index').reset_index()
            self.idf.columns = ['word', 'idf']
            print("--DONE!")

            print(":: Saving Term Frequencies to file...", end="\t")
            save(self.idf, "json", self.resource_path + "idf.json")
            print("--DONE!")

    def calculate_tf_in_cat(self):
        self.tf = load("json", self.resource_path + "tf_in_cat.json")
        if self.tf.get_status():
            self.tf = self.tf.get_data()
        else:
            self.calculate_tf()
            self.tf['tf_real'] = 0
            self.tf['tf_fake'] = 0
            print(":: Calculating Term Frequencies for Each Class...", end="\t")

            real_words = self.get_words_in_cat(1)
            for w in real_words:
                self.tf[w]['tf_real'] = real_words.count(w)

            fake_words = self.get_words_in_cat(0)
            for w in fake_words:
                self.tf[w]['tf_fake'] = fake_words.count(w)

            print("--DONE!")

            print(":: Saving Term Frequencies to file...", end="\t")
            save(self.tf, "json", self.resource_path + "tf_in_cat.json")
            print("--DONE!")

    def get_words_in_cat(self, cat_no):
        cat_data = self.dataset.loc[self.dataset['class'] == cat_no]
        cat_word_list = []
        for idx, row in cat_data.iterrows():
            words = row['text'] + row['title'] + row['description']
            cat_word_list += words
        return cat_word_list