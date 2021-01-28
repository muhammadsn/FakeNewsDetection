from FakeNewsDetection.FileHandler import Exporter as save
from FakeNewsDetection.FileHandler import Importer as load
from FakeNewsDetection.FakeDetector import FakeDetector as fd

def main():

    ## TODO: rewrite with os path

    settings = {
        "stemmer": "porter",
        "feature_count": 1000,
        "feature_extraction_method": "gini_index",
        "resource_path": "Resources/",
        "real_file_path": "Resources/Real.json",
        "fake_file_path": "Resources/Fake.json",
        "real_dataset_path": "Resources/Dataset/Real.json",
        "fake_dataset_path": "Resources/Dataset/Fake.json",
    }

    A = fd(settings)




if __name__ == "__main__":
    main()
