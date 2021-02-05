# from FakeNewsDetection.FileHandler import Exporter as save
# from FakeNewsDetection.FileHandler import Importer as load
from FakeNewsDetection.FakeDetector import FakeDetector as fd
from FakeNewsDetection.Plotter import Plotter
from pandas import ExcelWriter


def main():

    settings = {
        "stemmer": "porter",                                        # POSSIBLE VALUES => porter, english
        "advanced_mode_step4": False,                               # Activate/Deactivate advanced 4-category classification mode
        "advanced_mode_step2": False,                               # Activate/Deactivate using authors as extra features
        "classifiers": ["RF"],                                      # ["NB", "SV", "LR", "RF"],
        "metrics": ['accuracy', 'precision', 'recall', 'f1'],       # IF ADVANCED => ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        "desired_metric": "f1",
        "feature_count": 1500,
        "feature_scoring_function": 'log_tf_1',                     # POSSIBLE VALUES => tf, tf_idf, log_tf_idf, log_tf_1
        "feature_extraction_method": "MI",                          # POSSIBLE VALUES => MI, TF
        "cross_validation_fold_no": 5,
        "resource_path": "Resources/",
        "output_path": "Output/step4/",
        "train_dataset_path": "Resources/Dataset/Train/",
        "test_dataset_path": "Resources/Dataset/Test/",
    }

    A = fd(settings)
    B = A.train()

    ### UNCOMMENT THESE LINES IF YOU WANT TO PERFORM CLASSIFICATION ON TEST DATASET
    # C = A.get_best_classifier(settings['desired_metric'])
    # print(C)
    # D = A.predict()
    # YOU HAVE TO INSTALL "OPENPYXL" FOR THIS...
    # with ExcelWriter('FinalResults.xlsx') as writer:
    #     D.to_excel(writer, index=False)

    # F = Plotter()
    # F.generate_plots()


if __name__ == "__main__":
    main()
