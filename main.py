from FakeNewsDetection.FileHandler import Exporter as save
from FakeNewsDetection.FileHandler import Importer as load
from FakeNewsDetection.FakeDetector import FakeDetector as fd
from pandas import ExcelWriter


def main():

    ## TODO: rewrite with os path

    settings = {
        "stemmer": "porter",                                    # POSSIBLE VALUES => porter, english
        "classifiers": ["NB", "SV", "LR", "RF"],
        "metrics": ['accuracy', 'precision', 'recall', 'f1'],
        "desired_metric": "f1",
        "feature_count": 10,
        "feature_scoring_function": 'tf_idf',                       # POSSIBLE VALUES => tf, tf_idf, log_tf_idf, log_tf_1
        "feature_extraction_method": "MI",                      # POSSIBLE VALUES => MI, TF
        "cross_validation_fold_no": 5,
        "resource_path": "Resources/",
        "output_path": "Output/",
        "train_dataset_path": "Resources/Dataset/Train/",
        "test_dataset_path": "Resources/Dataset/Test/",
    }

    A = fd(settings)
    B = A.train()
    C = A.get_best_classifier(settings['desired_metric'])
    D = A.predict()

    # YOU HAVE TO INSTALL "OPENPYXL" FOR THIS...
    with ExcelWriter('FinalResults.xlsx') as writer:
        D.to_excel(writer, index=False)




if __name__ == "__main__":
    main()
