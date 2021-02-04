from .FileHandler import Importer as load
from .FileHandler import Exporter as save
import pandas as pd
import numpy as np

pd.set_option("display.max_rows", None, "display.max_columns", None)


class AuthorScorer:
    authors_scores = pd.DataFrame()
    doc_author_scores = pd.DataFrame()
    authors = pd.DataFrame()

    def __init__(self):
        self.load_authors()
        self.get_author_list()

        for idx, row in self.doc_author_scores.iterrows():
            doc_score = 0
            for a in row['authors']:
                doc_score += self.authors.query('author == "' + a + '"')['score'].tolist()[0]
            if len(row['authors']):
                self.doc_author_scores['author_score'].at[idx] = doc_score/len(row['authors'])
            if row['class'] == 2:
                if len(row['authors']) != 0 and doc_score/len(row['authors']) == 1:
                    self.doc_author_scores['class'].at[idx] = 3
            else:
                if len(row['authors']) != 0 and doc_score/len(row['authors']) == 0:
                    self.doc_author_scores['class'].at[idx] = 0
            # print(doc_score, len(row['authors']), self.doc_author_scores.iloc[idx]['score'])
            # print("=====================================")


    def load_authors(self):
        self.doc_author_scores = load('json', 'Resources/step4/RealAuthors.json')
        if self.doc_author_scores.get_status():
            self.doc_author_scores = self.doc_author_scores.get_data()
        else:
            exit()

        fake_authors = load('json', 'Resources/step4/FakeAuthors.json')
        if fake_authors.get_status():
            self.doc_author_scores = pd.concat([self.doc_author_scores, fake_authors.get_data()], ignore_index=True)
        else:
            exit()
        self.doc_author_scores['author_score'] = 0.0

    def get_author_list(self):
        authors = {}

        for idx, row in self.doc_author_scores.iterrows():
            for a in row['authors']:
                if a in authors.keys():
                    if row['class'] == 2:
                        authors[a]['real'] += 1
                    else:
                        authors[a]['fake'] += 1
                else:
                    if row['class'] == 2:
                        authors[a] = {'real': 1, 'fake': 0, 'score': 0.0}
                    else:
                        authors[a] = {'real': 0, 'fake': 1, 'score': 0.0}

        self.authors = pd.DataFrame.from_dict(authors, orient="index")
        self.authors.reset_index(inplace=True)
        self.authors.columns = ['author', 'real', 'fake', 'score']
        self.authors['score'] = self.authors['real'] / (self.authors['real'] + self.authors['fake'])

    def get_doc_author_scores(self):
        return self.doc_author_scores
