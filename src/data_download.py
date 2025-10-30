"""
data_download.py
----------------
Handles KaggleHub downloads and returns full paths to the exact files used in modeling.
"""

import os
import kagglehub

def get_beth_paths():
    path = kagglehub.dataset_download("katehighnam/beth-dataset")
    files = {
        "train": os.path.join(path, "labelled_training_data.csv"),
        "val": os.path.join(path, "labelled_validation_data.csv"),
        "test": os.path.join(path, "labelled_testing_data.csv"),
    }
    print(f"[INFO] BETH dataset files found: {list(files.values())}")
    return files


def get_unsw_paths():
    path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
    files = {
        "train": os.path.join(path, "UNSW_NB15_training-set.csv"),
        "test": os.path.join(path, "UNSW_NB15_testing-set.csv"),
    }
    print(f"[INFO] UNSW-NB15 dataset files found: {list(files.values())}")
    return files