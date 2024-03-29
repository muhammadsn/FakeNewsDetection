import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from .TextProcessor import TextProcessor as tp
from .FeatureExtractor import FeatureExtractor as fe
from .AdvancedFeatureExtractor import AdvancedFeatureExtractor as afe
from .CrossValidator import CrossValidation as cv
from .AuthorScorer import AuthorScorer
from .Classifier import Classifier as cl
from .FileHandler import Importer as load
from .FileHandler import Exporter as save
# import warnings
# warnings.filterwarnings('ignore')

pd.set_option("display.max_rows", None, "display.max_columns", None)


class FakeDetector:

    settings = {}
    real_train_dataset = pd.DataFrame()
    fake_train_dataset = pd.DataFrame()
    train_dataset = pd.DataFrame()
    test_dataset = pd.DataFrame()
    train_processed = None
    feature_list = []
    classifiers = []
    metrics = []
    cross_validator = None
    validation_results = []
    advanced_mode = False
    extra_feature_mode = False
    train_doc_author_scores = pd.DataFrame()
    test_doc_author_scores = pd.DataFrame()

    def __init__(self, settings):
        self.settings = settings
        self.advanced_mode = self.settings['advanced_mode_step4']
        self.extra_feature_mode = self.settings['advanced_mode_step2']

        if self.advanced_mode or self.extra_feature_mode:
            Train_AS = AuthorScorer('train', advanced=self.advanced_mode)
            self.train_doc_author_scores = Train_AS.get_doc_author_scores()
            Test_AS = AuthorScorer('test', advanced=self.advanced_mode)
            self.test_doc_author_scores = Test_AS.get_doc_author_scores()

        self.load_train_data()
        self.load_test_data()

        self.classifiers = self.settings['classifiers']
        self.metrics = self.settings['metrics']

        if self.advanced_mode:
            self.train_processed = afe(dataset=self.train_dataset, feature_list=None, settings=self.settings)
        else:
            self.train_processed = fe(dataset=self.train_dataset, feature_list=None, settings=self.settings)
        self.feature_list = self.train_processed.get_text_features()

        train_data = self.train_processed.get_feature_scores(self.settings['feature_scoring_function'], 'train')
        self.cross_validator = cv(train=train_data,
                                  labels=self.train_processed.get_labels(),
                                  n_fold=self.settings['cross_validation_fold_no'])



    def load_train_data(self):
        # if self.advanced_mode:
        #     self.train_dataset = load("json", self.settings["train_dataset_path"] + "Advanced_Train.json")
        # else:
        self.train_dataset = load("json", self.settings["train_dataset_path"] + "Train.json")

        if self.train_dataset.get_status():
            self.train_dataset = self.train_dataset.get_data()

            if self.advanced_mode or self.extra_feature_mode:
                self.train_dataset.drop(['class'], axis=1, inplace=True)
                self.train_dataset = pd.merge(self.train_dataset, self.train_doc_author_scores, on=['file'], how='inner')

        else:
            self.real_train_dataset = load("json", self.settings["train_dataset_path"] + "Real.json")
            self.fake_train_dataset = load("json", self.settings["train_dataset_path"] + "Fake.json")

            if self.real_train_dataset.get_status():
                self.real_train_dataset = self.real_train_dataset.get_data()
                self.real_train_dataset['class'] = 1
            else:
                rt = load("json", self.settings["resource_path"] + "Real.json")
                if rt.get_status():
                    rt = rt.get_data()
                else:
                    print(":: ERROR OCCURRED!")
                    exit(404)
                print(":: Processing Real Dataset From file...", end="\t")
                rl = []
                for idx, row in rt.iterrows():
                    text_tokens = tp(row['text'], self.settings["stemmer"])
                    title_tokens = tp(row['title'], self.settings["stemmer"])
                    description_tokens = tp(row['description'], self.settings["stemmer"])
                    d = {"file": row["file"], "text": text_tokens.get_words(), "title": title_tokens.get_words(), "description": description_tokens.get_words(), "class": 1}
                    rl.append(d)
                self.real_train_dataset = pd.DataFrame(rl)
                print("--Done!")
                print(":: Saving Processed Real Train Set...", end='\t')
                save(self.real_train_dataset, "json", self.settings["train_dataset_path"] + "Real.json")
                print("--Done!")

            if self.fake_train_dataset.get_status():
                self.fake_train_dataset = self.fake_train_dataset.get_data()
                self.fake_train_dataset['class'] = 0
            else:
                ft = load("json", self.settings["resource_path"] + "Fake.json")

                if ft.get_status():
                    ft = ft.get_data()
                else:
                    print(":: ERROR OCCURRED!")
                    exit(404)

                print(":: Processing Fake Dataset From file...", end="\t")
                fl = []
                for idx, row in ft.iterrows():
                    text_tokens = tp(row['text'], self.settings["stemmer"])
                    title_tokens = tp(row['title'], self.settings["stemmer"])
                    description_tokens = tp(row['description'], self.settings["stemmer"])
                    d = {"file": row["file"], "text": text_tokens.get_words(), "title": title_tokens.get_words(), "description": description_tokens.get_words(), "class": 0}
                    fl.append(d)
                self.fake_train_dataset = pd.DataFrame(fl)
                print("--Done!")

                print(":: Saving Processed Fake Train Set...", end='\t')
                save(self.fake_train_dataset, "json", self.settings["train_dataset_path"] + "Fake.json")
                print("--Done!")

            self.train_dataset = pd.concat([self.real_train_dataset, self.fake_train_dataset])
            # self.train_dataset = self.train_dataset.sample(frac=1).reset_index(drop=True)
            self.train_dataset = shuffle(self.train_dataset)

            if self.advanced_mode or self.extra_feature_mode:
                self.train_dataset.drop(['class'], axis=1, inplace=True)
                self.train_dataset = pd.merge(self.train_dataset, self.train_doc_author_scores, on=['file'], how='inner')

            print(":: Saving Processed Train Set...", end='\t')
            if self.advanced_mode:
                save(self.train_dataset, "json", self.settings["train_dataset_path"] + "Advanced_Train.json")
            else:
                save(self.train_dataset, "json", self.settings["train_dataset_path"] + "Train.json")
            print("--Done!")
            return True

    def load_test_data(self):
        # if self.advanced_mode:
        #     self.test_dataset = load("json", self.settings["test_dataset_path"] + "Advanced_Test.json")
        # else:
        self.test_dataset = load("json", self.settings["test_dataset_path"] + "Test.json")

        if self.test_dataset.get_status():
            self.test_dataset = self.test_dataset.get_data()
            self.test_dataset['class'] = None

            if self.advanced_mode or self.extra_feature_mode:
                self.test_dataset.drop(['class'], axis=1, inplace=True)
                self.test_dataset = pd.merge(self.test_dataset, self.test_doc_author_scores, on=['file'], how='inner')
        else:
            test = load("json", self.settings["resource_path"] + "Test.json")
            if test.get_status():
                test = test.get_data()
            else:
                print(":: ERROR OCCURRED!")
                exit(404)
            print(":: Processing Test Dataset From file...", end="\t")
            tl = []
            for idx, row in test.iterrows():
                text_tokens = tp(row['text'], self.settings["stemmer"])
                title_tokens = tp(row['title'], self.settings["stemmer"])
                description_tokens = tp(row['description'], self.settings["stemmer"])
                d = {"file": row["file"], "text": text_tokens.get_words(), "title": title_tokens.get_words(), "description": description_tokens.get_words(), "class": None}
                tl.append(d)
            self.test_dataset = pd.DataFrame(tl)
            if self.advanced_mode or self.extra_feature_mode:
                self.test_dataset.drop(['class'], axis=1, inplace=True)
                self.test_dataset = pd.merge(self.test_dataset, self.test_doc_author_scores, on=['file'], how='inner')
            print("--Done!")
            print(":: Saving Processed Real Test Set...", end='\t')
            if self.advanced_mode:
                save(self.test_dataset, "json", self.settings["test_dataset_path"] + "Advanced_Test.json")
            else:
                save(self.test_dataset, "json", self.settings["test_dataset_path"] + "Test.json")
            print("--Done!")
            return True

    def train(self, classifiers=None):

        if classifiers is not None:
            self.classifiers = classifiers

        for c in self.classifiers:
            s = []
            for m in self.metrics:
                s.append(self.cross_validator.validate(c, m))
            self.validation_results.append({
                'classifier': c,
                'accuracy': np.mean(s[0]),
                'precision': np.mean(s[1]),
                'recall': np.mean(s[2]),
                'f1': np.mean(s[3])
            })
        self.validation_results = pd.DataFrame(self.validation_results)
        save(self.validation_results, 'json', self.settings['output_path'] + self.settings['feature_scoring_function'] + "/result_fn@" + str(self.settings['feature_count']) + ".json")

    def get_best_classifier(self, metric):
        res = self.validation_results.loc[self.validation_results[metric].idxmax()]
        return res['classifier'], res[metric]

    def predict(self):
        test = fe(dataset=self.test_dataset, feature_list=self.feature_list, settings=self.settings)
        res = cl(train_feature_matrix=self.train_processed.get_feature_scores(method=self.settings['feature_scoring_function'], phase='train'),
                 train_labels=self.train_processed.get_labels(),
                 test_feature_matrix=test.get_feature_scores(method=self.settings['feature_scoring_function'], phase='test'),
                 test_labels=None, method=None)
        df = self.test_dataset
        df['class'] = res.get_prediction("RF")
        df['file'] = 'newscontent_' + df['file'].astype(str)
        return df[['file', 'class']]