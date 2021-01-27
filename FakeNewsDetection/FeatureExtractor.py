import pandas as pd


class FeatureExtractor:

    all_words = []
    tf = pd.DataFrame()
    idf = pd.DataFrame()

    def __init__(self, real, fake):

        for idx, row in real.iterrow():
            words = row['text'] + row['title'] + row['description']
            self.all_words += words

        for idx, row in fake.iterrow():
            words = row['text'] + row['title'] + row['description']
            self.all_words += words

    def calculate_tf(self):
        wcd = {}

        for w in self.all_words:
            if w not in wcd.keys():
                wcd[w] = self.all_words.count(w)
                print(wcd[w])
