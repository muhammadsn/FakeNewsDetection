from FakeNewsDetection.fileHandler import Exporter as save
from FakeNewsDetection.fileHandler import Importer as load
from FakeNewsDetection.FakeDetector import FakeDetector as fd

def main():

    ## TODO: rewrite with os path

    settings = {
        "stemmer": "porter",
        "real_file_path": "Resources/Real.json",
        "fake_file_path": "Resources/Fake.json",
        "real_dataset_path": "Resources/Dataset/Real.json",
        "fake_dataset_path": "Resources/Dataset/Fake.json",
    }

    A = fd(settings)




if __name__ == "__main__":
    main()