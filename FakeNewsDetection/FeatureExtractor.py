from .FileHandler import Importer as load
from .FileHandler import Exporter as save
import pandas as pd
import numpy as np
from nltk.book import FreqDist
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
            real_set = self.dataset.loc[self.dataset['class'] == 1]
            fake_set = self.dataset.loc[self.dataset['class'] == 0]
            collections = [real_set, fake_set]
            collection_words = [[], []]

            for idx, row in collections[0].iterrows():
                collection_words[0] += row['text'] + row['title'] + row['description']

            for idx, row in collections[1].iterrows():
                collection_words[1] += row['text'] + row['title'] + row['description']

            term_freq = [FreqDist(collection_words[0]), FreqDist(collection_words[1])]
            all_term_freq = FreqDist(self.all_words)
            total_word_count = sum(all_term_freq.values())
            N = len(collections[0]) + len(collections[1])
            M = [0, 0]
            pci = [len(collections[0])/N, len(collections[1])/N]
            for word in self.vocabulary:
                pi_w = [0, 0]
                p_w = all_term_freq[word] / total_word_count
                for cat in [0, 1]:
                    pi_w[cat] = term_freq[cat][word]/sum(term_freq[cat].values())
                    print(cat, word, pci[cat], p_w)
                    M[cat] = np.log(pi_w[cat]/(pci[cat] * p_w))
                score_list.append({"word": word, "score": np.max(M)})

            df = pd.DataFrame(score_list, columns=["word", "score"])
            self.feature_scores = df
            print("--DONE!")

            print(":: Saving MI Scores to file...", end="\t")
            save(df, "json", self.resource_path + "mi_scores.json")
            print("--DONE!")

    def tf_based_scorer(self):
        self.calculate_tf()
        self.feature_scores = self.tf.sort_values(by=['tf'], ascending=False)
        f = self.feature_scores.head(self.feature_count)
        self.feature_list = f['word'].tolist()

    def calculate_tf(self):     # TODO: USE FREQDIST...
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

    def get_tf_in_doc(self):
        tf_matrix = np.zeros([self.dataset.shape[0], len(self.feature_list)])
        # self.tf = load("json", self.resource_path + "tf_in_cat.json")
        # if self.tf.get_status():
        #     self.tf = self.tf.get_data()
        # else:
        for idx, row in self.dataset.iterrows():
            words = row['text'] + row['title'] + row['description']
            doc_term_freq = FreqDist(words)
            print(words)
            for fw in self.feature_list:
                print(fw['word'], doc_term_freq[fw['word']])
                # tf_matrix[idx][self.feature_list.index(fw)] = doc_term_freq[fw['word']]

        # print(tf_matrix)
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