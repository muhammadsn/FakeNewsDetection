import pandas as pd
from sklearn.feature_selection import chi2


class FeatureExtractor:

    all_words = []
    tf = pd.DataFrame()
    idf = pd.DataFrame()

    def __init__(self, real, fake):

        for idx, row in real.iterrows():
            words = row['text'] + row['title'] + row['description']
            self.all_words += words

        for idx, row in fake.iterrows():
            words = row['text'] + row['title'] + row['description']
            self.all_words += words

        self.calculate_tf()

    def calculate_tf(self):
        wcd = {}

        for w in self.all_words:
            if w not in wcd.keys():
                wcd[w] = self.all_words.count(w)

