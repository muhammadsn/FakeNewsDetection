import pandas as pd
from .textProcessor import TextProcessor as tp
from .fileHandler import Importer as load
from .fileHandler import Exporter as save

pd.set_option("display.max_rows", None, "display.max_columns", None)


class FakeDetector:

    settings = {}

    def __init__(self, settings):
        self.settings = settings


        rt = load("json", self.settings["real_file_path"])
        ft = load("json", self.settings["fake_file_path"])

        if rt.get_status():
            rt = rt.get_data()
        else:
            print(":: ERROR OCCURRED!")
            exit()

        if ft.get_status():
            ft = ft.get_data()
        else:
            print(":: ERROR OCCURRED!")
            exit()

        rl = []
        for idx, row in rt.iterrows():
            text_tokens = tp(row['text'])
            title_tokens = tp(row['title'])
            description_tokens = tp(row['description'])
            d = {"file": row["file"], "text": text_tokens.get_words(), "title": title_tokens.get_words(), "description": description_tokens.get_words()}
            rl.append(d)
        real_train = pd.DataFrame(rl)
        save(real_train, "json", self.settings["real_dataset_path"])

        fl = []
        for idx, row in ft.iterrows():
            text_tokens = tp(row['text'])
            title_tokens = tp(row['title'])
            description_tokens = tp(row['description'])
            d = {"file": row["file"], "text": text_tokens.get_words(), "title": title_tokens.get_words(), "description": description_tokens.get_words()}
            fl.append(d)
        fake_train = pd.DataFrame(fl)
        save(fake_train, "json", self.settings["fake_dataset_path"])
