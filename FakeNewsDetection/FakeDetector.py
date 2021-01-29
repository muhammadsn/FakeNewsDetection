import pandas as pd
from .TextProcessor import TextProcessor as tp
from .FeatureExtractor import FeatureExtractor as fe
from .Classifier import Classifier
from .FileHandler import Importer as load
from .FileHandler import Exporter as save

pd.set_option("display.max_rows", None, "display.max_columns", None)


class FakeDetector:

    settings = {}

    real_train_dataset = pd.DataFrame()
    fake_train_dataset = pd.DataFrame()
    train_dataset = pd.DataFrame()
    test_dataset = pd.DataFrame()

    def __init__(self, settings):
        self.settings = settings
        self.load_train_data()
        self.load_test_data()

        # self.settings['feature_extraction_method']

        train = fe(dataset=self.train_dataset, feature_list=None, settings=self.settings)
        test = fe(dataset=self.test_dataset, feature_list=train.get_features(), settings=self.settings)

        # print(a.get_tf())
        # print(a.get_tf_plus_one())
        # print(a.get_idf())
        classifiers = ["NB", "SV", "LR", "RF"]
        a = Classifier(train.get_tf_idf('train'), train.get_labels(), test.get_tf_idf('test'), None)
        for c in classifiers:
            print(a.get_prediction(c))
        # print(a.get_log_tf_idf())


    def load_train_data(self):
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
        return True

    def load_test_data(self):
        self.test_dataset = load("json", self.settings["test_dataset_path"] + "Test.json")

        if self.test_dataset.get_status():
            self.test_dataset = self.test_dataset.get_data()
            self.test_dataset['class'] = None
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
            print("--Done!")
            print(":: Saving Processed Real Train Set...", end='\t')
            save(self.test_dataset, "json", self.settings["test_dataset_path"] + "Test.json")
            print("--Done!")
            return True
