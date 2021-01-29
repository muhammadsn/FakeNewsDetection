from FakeNewsDetection.FileHandler import Exporter as save
from FakeNewsDetection.FileHandler import Importer as load
from FakeNewsDetection.FakeDetector import FakeDetector as fd

def main():

    ## TODO: rewrite with os path

    settings = {
        "stemmer": "porter",
        "feature_count": 10,
        "feature_extraction_method": "MI",
        "resource_path": "Resources/",
        "train_dataset_path": "Resources/Dataset/Train/",
        "test_dataset_path": "Resources/Dataset/Test/",
    }

    A = fd(settings)




if __name__ == "__main__":
    main()
