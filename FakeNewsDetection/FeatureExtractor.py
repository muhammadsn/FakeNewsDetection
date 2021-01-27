import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest

pd.set_option("display.max_rows", None, "display.max_columns", None)
np.set_printoptions(threshold=sys.maxsize)

class FeatureExtractor:

    all_words = []
    tf = pd.DataFrame()
    idf = pd.DataFrame()

    def __init__(self, dataset):
        df = pd.DataFrame(columns=['body', 'class'])
        df['body'] = dataset['text'] + dataset['title'] + dataset['description']
        df['class'] = dataset['class']
        for idx, row in df.iterrows():
            df['body'].at[idx] = " ".join(row['body'])
        # X = df['body'].tolist()
        bag_words = CountVectorizer(ngram_range=(1, 1), min_df=5, max_df=0.7)
        # tfidf = TfidfVectorizer(max_features=10, lowercase=True, ngram_range=(1, 1))


        features1 = bag_words.fit_transform(df['body'], df['class']).toarray()
        # features2 = tfidf.fit_transform(df['body'], df['class'])
        chi2_selector = SelectKBest(chi2, k=10)
        x_kbest = chi2_selector.fit_transform(X=features1, y=df['class'].tolist())
        print(x_kbest)


    def calculate_tf(self):
        wcd = {}

        for w in self.all_words:
            if w not in wcd.keys():
                wcd[w] = self.all_words.count(w)

