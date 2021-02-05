from .FileHandler import Importer as load
import pandas as pd

pd.set_option("display.max_rows", None, "display.max_columns", None)


class AuthorScorer:
    authors_scores = pd.DataFrame()
    doc_author_scores = pd.DataFrame()
    authors = pd.DataFrame()
    authors_file = ''
    advanced_mode = False
    phase = ""

    def __init__(self, phase, advanced=False):
        if phase == "train":
            if advanced:
                self.authors_file = f"Resources/step4/train/"
            else:
                self.authors_file = f"Resources/step2/train/"
        elif phase == "test":
            if advanced:
                self.authors_file = f"Resources/step4/test/"
            else:
                self.authors_file = f"Resources/step2/test/"
        else:
            print(":: [ERROR] Wrong phase... --ABORTING")
            exit()

        self.advanced_mode = advanced
        self.phase = phase

        self.load_authors()
        self.get_author_list()

        for idx, row in self.doc_author_scores.iterrows():
            doc_score = 0
            for a in row['authors']:
                doc_score += self.authors.query('author == "' + a + '"')['score'].tolist()[0]
            if len(row['authors']):
                self.doc_author_scores['author_score'].at[idx] = doc_score/len(row['authors'])
            if self.advanced_mode and phase == "train":
                if row['class'] == 2:
                    if len(row['authors']) != 0 and doc_score/len(row['authors']) == 1:
                        self.doc_author_scores['class'].at[idx] = 3
                else:
                    if len(row['authors']) != 0 and doc_score/len(row['authors']) == 0:
                        self.doc_author_scores['class'].at[idx] = 0
            # print(doc_score, len(row['authors']), self.doc_author_scores.iloc[idx]['score'])
            # print("=====================================")


    def load_authors(self):
        if self.phase == "train":
            self.doc_author_scores = load('json', self.authors_file + 'RealAuthors.json')
            if self.doc_author_scores.get_status():
                self.doc_author_scores = self.doc_author_scores.get_data()
            else:
                exit()

            fake_authors = load('json', self.authors_file + 'FakeAuthors.json')
            if fake_authors.get_status():
                self.doc_author_scores = pd.concat([self.doc_author_scores, fake_authors.get_data()], ignore_index=True)
            else:
                exit()
            self.doc_author_scores['author_score'] = 0.0
        if self.phase == "test":
            test_authors = load('json', self.authors_file + 'TestAuthors.json')
            if test_authors.get_status():
                self.doc_author_scores = test_authors.get_data()
            else:
                exit()
            self.doc_author_scores['author_score'] = 0.0

    def get_author_list(self):
        authors = {}
        for idx, row in self.doc_author_scores.iterrows():
            if self.advanced_mode:
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
            else:
                for a in row['authors']:
                    if a in authors.keys():
                        if row['class'] == 1:
                            authors[a]['real'] += 1
                        else:
                            authors[a]['fake'] += 1
                    else:
                        if row['class'] == 1:
                            authors[a] = {'real': 1, 'fake': 0, 'score': 0.0}
                        else:
                            authors[a] = {'real': 0, 'fake': 1, 'score': 0.0}

        self.authors = pd.DataFrame.from_dict(authors, orient="index")
        self.authors.reset_index(inplace=True)
        self.authors.columns = ['author', 'real', 'fake', 'score']
        self.authors['score'] = self.authors['real'] / (self.authors['real'] + self.authors['fake'])

    def get_doc_author_scores(self):
        return self.doc_author_scores
