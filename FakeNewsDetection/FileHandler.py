import os.path
import pandas as pd
import numpy as np


class Exporter:

    data = pd.DataFrame()
    path = ""

    def __init__(self, _data, _format, _path):
        self.data = _data
        self.path = _path
        if _format == "json":
            self.to_json()
        elif _format == "csv":
            self.to_csv()

    def to_json(self):
        self.data.to_json(self.path, "table")
        return True

    def to_csv(self):
        np.savetxt(self.path, self.data, delimiter=',')
        return True


class Importer:
    data = pd.DataFrame()
    path = ""


    def __init__(self, _format, _path):
        self.path = _path
        self.is_error = False

        if self.check_file(self.path):
            if _format == "json":
                self.from_json()
            elif _format == "csv":
                self.from_csv()
        else:
            self.is_error = True
            print(":: Data File in \"" + self.path + "\" NOT Found ...\t --ABORTING")

    def from_json(self):
        print(":: Loading Data From File: \"" + self.path + "\" ...", end="\t ")
        self.data = pd.read_json(path_or_buf=self.path, orient="table")
        print("--DONE!")

    def from_csv(self):
        print(":: Loading Data From File: \"" + self.path + "\" ...", end="\t ")
        self.data = np.genfromtxt(self.path, delimiter=',')
        print("--DONE!")

    def get_status(self):
        return not self.is_error

    def get_data(self):
        return self.data

    @staticmethod
    def check_file(path_to_file):
        if os.path.isfile(path_to_file):
            return True
        else:
            return False