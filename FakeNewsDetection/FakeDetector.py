import pandas as pd
from .TextProcessor import TextProcessor as tp
from .FeatureExtractor import FeatureExtractor as fe
from .FileHandler import Importer as load
from .FileHandler import Exporter as save

pd.set_option("display.max_rows", None, "display.max_columns", None)


class FakeDetector:

    settings = {}

    real_train_dataset = pd.DataFrame()
    fake_train_dataset = pd.DataFrame()
    train_dataset = pd.DataFrame()

    def __init__(self, settings):
        self.settings = settings
        self.load_train_data()
        self.train_dataset = pd.concat([self.real_train_dataset, self.fake_train_dataset])

        # self.settings['feature_extraction_method']

        a = fe(self.train_dataset, self.settings["resource_path"], self.settings['feature_count'], "MI")

        print(a.get_features())
        a.get_tf_in_doc()





    def load_train_data(self):
        self.real_train_dataset = load("json", self.settings["real_dataset_path"])
        self.fake_train_dataset = load("json", self.settings["fake_dataset_path"])

        if self.real_train_dataset.get_status():
            self.real_train_dataset = self.real_train_dataset.get_data()
            self.real_train_dataset['class'] = 1
        else:
            rt = load("json", self.settings["real_file_path"])
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
                d = {"file": row["file"], "text": text_tokens.get_words(), "title": title_tokens.get_words(), "description": description_tokens.get_words()}
                rl.append(d)
            self.real_train_dataset = pd.DataFrame(rl)
            print("--Done!")
            print(":: Saving Processed Real Train Set...", end='\t')
            save(self.real_train_dataset, "json", self.settings["real_dataset_path"])
            print("--Done!")

        if self.fake_train_dataset.get_status():
            self.fake_train_dataset = self.fake_train_dataset.get_data()
            self.fake_train_dataset['class'] = 0
        else:
            ft = load("json", self.settings["fake_file_path"])

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
                d = {"file": row["file"], "text": text_tokens.get_words(), "title": title_tokens.get_words(), "description": description_tokens.get_words()}
                fl.append(d)
            self.fake_train_dataset = pd.DataFrame(fl)
            print("--Done!")

            print(":: Saving Processed Fake Train Set...", end='\t')
            save(self.fake_train_dataset, "json", self.settings["fake_dataset_path"])
            print("--Done!")
