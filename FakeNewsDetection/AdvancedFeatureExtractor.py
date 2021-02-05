from .FileHandler import Importer as load
from .FileHandler import Exporter as save
import pandas as pd
import numpy as np
from nltk.book import FreqDist
import sys

pd.set_option("display.max_rows", None, "display.max_columns", None)
np.set_printoptions(threshold=sys.maxsize)


class AdvancedFeatureExtractor:
    all_words = []
    vocabulary = []
    dataset = pd.DataFrame()
    _tf = pd.DataFrame()
    _idf = pd.DataFrame()
    feature_scores = pd.DataFrame()
    feature_count = 0
    feature_list = []
    settings = ""
    method = ""
    phase = ""

    def __init__(self, dataset, feature_list, settings):
        self.dataset = dataset
        self.settings = settings
        self.feature_count = self.settings['feature_count']
        self.method = self.settings['feature_extraction_method']
        self.feature_list = feature_list

        if self.dataset.empty or not isinstance(self.dataset, pd.DataFrame):
            print(":: [ERROR] Invalid Dataset Provided ... --ABORTING")
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


    def get_text_features(self):
        if self.feature_list is None:
            self.phase = "train"
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
        self.feature_scores = load("json", self.settings['train_dataset_path'] + "mi_scores.json")
        if self.feature_scores.get_status():
            self.feature_scores = self.feature_scores.get_data()
        else:
            print(":: Calculating Term Frequencies for Vocabulary...", end="\t")
            score_list = []
            realer_set = self.dataset.loc[self.dataset['class'] == 3]
            real_set = self.dataset.loc[self.dataset['class'] == 2]
            fake_set = self.dataset.loc[self.dataset['class'] == 1]
            faker_set = self.dataset.loc[self.dataset['class'] == 0]
            collections = [realer_set, real_set, fake_set, faker_set]
            collection_words = [[], [], [], []]

            for idx, row in collections[0].iterrows():
                collection_words[0] += row['text'] + row['title'] + row['description']

            for idx, row in collections[1].iterrows():
                collection_words[1] += row['text'] + row['title'] + row['description']

            for idx, row in collections[2].iterrows():
                collection_words[2] += row['text'] + row['title'] + row['description']

            for idx, row in collections[3].iterrows():
                collection_words[3] += row['text'] + row['title'] + row['description']

            term_freq = [FreqDist(collection_words[0]), FreqDist(collection_words[1]),
                         FreqDist(collection_words[2]), FreqDist(collection_words[3])]
            all_term_freq = FreqDist(self.all_words)
            total_word_count = sum(all_term_freq.values())
            # N = len(collections[0]) + len(collections[1])
            M = [0, 0, 0, 0]
            # pci = [len(collections[0])/N, len(collections[1])/N]
            pci = [sum(term_freq[0].values()) / total_word_count, sum(term_freq[1].values()) / total_word_count,
                   sum(term_freq[2].values()) / total_word_count, sum(term_freq[3].values()) / total_word_count]
            for word in self.vocabulary:
                pi_w = [0, 0, 0, 0]
                p_w = all_term_freq[word] / total_word_count
                for cat in [0, 1, 2, 3]:
                    ############## WE USED LAPLACE SMOOTHING
                    pi_w[cat] = (term_freq[cat][word] + 1) / (sum(term_freq[cat].values()) + len(self.vocabulary))
                    # print(cat, word, pi_w[cat], pci[cat], p_w)
                    M[cat] = np.log(pi_w[cat]/(pci[cat] * p_w))
                # score_list.append({"word": word, "score": np.max(M)})
                score_list.append({"word": word, "score": (pci[0] * p_w * M[0]) + (pci[1] * p_w * M[1]) + (pci[2] * p_w * M[2]) + (pci[3] * p_w * M[3])})

            df = pd.DataFrame(score_list, columns=["word", "score"])
            self.feature_scores = df
            print("--DONE!")

            print(":: Saving MI Scores to file...", end="\t")
            save(df, "json", self.settings['train_dataset_path'] + "mi_scores.json")
            print("--DONE!")

    def tf_based_scorer(self):
        self.calculate_tf()
        self.feature_scores = self._tf.sort_values(by=['tf'], ascending=False)
        f = self.feature_scores.head(self.feature_count)
        self.feature_list = f['word'].tolist()

    def calculate_tf(self):     # TODO: USE FREQDIST...
        self._tf = load("json", self.settings['resource_path'] + "tf.json")
        if self._tf.get_status():
            self._tf = self._tf.get_data()
        else:
            print(":: Calculating Term Frequencies for Dataset...", end="\t")
            wcd = {}

            for w in self.all_words:
                if w not in wcd.keys():
                    wcd[w] = self.all_words.count(w)
            self._tf = pd.DataFrame.from_dict(wcd, orient='index').reset_index()
            self._tf.columns = ['word', 'tf']
            print("--DONE!")

            print(":: Saving Term Frequencies to file...", end="\t")
            save(self._tf, "json", self.settings['resource_path'] + "tf.json")
            print("--DONE!")

    def idf(self):
        idf_list = load("csv", self.settings[self.phase + '_dataset_path'] + "adv_idf_list.csv")
        if idf_list.get_status():
            idf_list = idf_list.get_data()
        else:
            idf_list = np.zeros(len(self.feature_list))
            print(":: Calculating Inverse Document Frequencies for Dataset...", end="\t")
            N = self.dataset.shape[0]
            feature_words = [x['word'] for x in self.feature_list]
            for w in feature_words:
                df = 0
                res = self.dataset[self.dataset['body'].str.contains(w)]
                df += res.shape[0]
                if df == 0:
                    for idx, row in self.dataset.iterrows():
                        words = row['text'] + row['title'] + row['description']
                        if w in words:
                            df += 1
                idf_list[feature_words.index(w)] = np.log((N+1)/df)
                self.feature_list[feature_words.index(w)]['idf'] = np.log((N+1)/df)
            print("--DONE!")

            print(":: Saving Inverse Document Frequencies to file...", end="\t")
            save(idf_list, "csv", self.settings[self.phase + '_dataset_path'] + "adv_idf_list.csv")
            print("--DONE!")
        return idf_list

    def tf(self):
        tf_matrix = load("csv", self.settings[self.phase + '_dataset_path'] + "adv_tf_matrix.csv")
        if tf_matrix.get_status():
            tf_matrix = tf_matrix.get_data()
        else:
            tf_matrix = np.zeros([self.dataset.shape[0], len(self.feature_list)])
            for idx, row in self.dataset.iterrows():
                words = row['text'] + row['title'] + row['description']
                doc_term_freq = FreqDist(words)
                for fw in self.feature_list:
                    tf_matrix[idx][self.feature_list.index(fw)] = doc_term_freq[fw['word']]
            print("--DONE!")

            print(":: Saving Term Frequencies to file...", end="\t")
            save(tf_matrix, "csv", self.settings[self.phase + '_dataset_path'] + "adv_tf_matrix.csv")
            print("--DONE!")
        return tf_matrix

    def tf_idf(self):
        tf_idf_matrix = load("csv", self.settings[self.phase + '_dataset_path'] + "adv_tf_idf_matrix.csv")
        if tf_idf_matrix.get_status():
            tf_idf_matrix = tf_idf_matrix.get_data()
        else:
            idf_list = load("csv", self.settings[self.phase + '_dataset_path'] + "adv_idf_list.csv")
            if idf_list.get_status():
                idf_list = idf_list.get_data()
            else:
                idf_list = self.idf()

            tf_idf_matrix = np.zeros([self.dataset.shape[0], len(self.feature_list)])

            for idx, row in self.dataset.iterrows():
                words = row['text'] + row['title'] + row['description']
                doc_term_freq = FreqDist(words)
                for fw in self.feature_list:
                    tf_idf_matrix[idx][self.feature_list.index(fw)] = doc_term_freq[fw['word']] * idf_list[self.feature_list.index(fw)]

            print("--DONE!")

            print(":: Saving TF-IDF Data to file...", end="\t")
            save(tf_idf_matrix, "csv", self.settings[self.phase + '_dataset_path'] + "adv_tf_idf_matrix.csv")
            print("--DONE!")
        return tf_idf_matrix

    def log_tf_idf(self):
        tf_idf_matrix = load("csv", self.settings[self.phase + '_dataset_path'] + "adv_log_tf_idf_matrix.csv")
        if tf_idf_matrix.get_status():
            tf_idf_matrix = tf_idf_matrix.get_data()
        else:
            idf_list = load("csv", self.settings[self.phase + '_dataset_path'] + "adv_idf_list.csv")
            if idf_list.get_status():
                idf_list = idf_list.get_data()
            else:
                idf_list = self.idf()

            tf_idf_matrix = np.zeros([self.dataset.shape[0], len(self.feature_list)])
            for idx, row in self.dataset.iterrows():
                words = row['text'] + row['title'] + row['description']
                doc_term_freq = FreqDist(words)
                for fw in self.feature_list:
                    tf_idf_matrix[idx][self.feature_list.index(fw)] = np.log(doc_term_freq[fw['word']] + 1) * idf_list[self.feature_list.index(fw)]
            print("--DONE!")

            print(":: Saving Processed Data to file...", end="\t")
            save(tf_idf_matrix, "csv", self.settings[self.phase + '_dataset_path'] + "adv_log_tf_idf_matrix.csv")
            print("--DONE!")
        return tf_idf_matrix

    def log_tf_1(self,):
        tf_matrix = load("csv", self.settings[self.phase + '_dataset_path'] + "adv_tf_plus_one_matrix.csv")
        if tf_matrix.get_status():
            tf_matrix = tf_matrix.get_data()
        else:
            tf_matrix = np.zeros([self.dataset.shape[0], len(self.feature_list)])
            for idx, row in self.dataset.iterrows():
                words = row['text'] + row['title'] + row['description']
                doc_term_freq = FreqDist(words)
                for fw in self.feature_list:
                    tf_matrix[idx][self.feature_list.index(fw)] = np.log(doc_term_freq[fw['word']] + 1)
            print("--DONE!")

            print(":: Saving Term Frequencies to file...", end="\t")
            save(tf_matrix, "csv", self.settings[self.phase + '_dataset_path'] + "adv_tf_plus_one_matrix.csv")
            print("--DONE!")
        return tf_matrix

    def get_labels(self):
        if self.phase == "train":
            return self.dataset['class'].tolist()
        else:
            print(":: [WARNING] Getting Labels is Only Available in Train Phase")
            return []

    def get_feature_scores(self, method, phase):
        self.phase = phase
        if phase != "test" and phase != "train":
            print(":: [ERROR] Wrong Value for Phase Parameter ['test' or 'train']... --ABORTING")
            exit(419)

        if method not in ['tf', 'tf_idf', 'log_tf_idf', 'log_tf_1']:
            print(":: [ERROR] Wrong Value for feature scoring method ['tf' / 'tf_idf' / 'log_tf_idf' / 'log_tf_1']... --ABORTING")
            exit(419)

        scorer = self.__getattribute__(method)
        return scorer()
