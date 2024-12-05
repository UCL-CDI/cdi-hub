import pandas as pd
import os
from pathlib import Path
import yaml
from loguru import logger
from amir.utils.datasets import CheXNet_CNN_Dataset


with open(str(Path().absolute())+"/tests/config_test.yml", "r") as file:
    config_yaml = yaml.load(file, Loader=yaml.FullLoader)


def test_CheXNet_CNN_Dataset():
    """
    Test CheXNet_CNN_Dataset class
    pytest -vs tests/test_datasets.py::test_CheXNet_CNN_Dataset
        TODO:
            - Use 640x400 image size
            - Test mask to show sclera, iris, pupil and background
    """
    # Define transforms - note we do ToTensor in the dataset class

    
    DATASET_PATH = os.path.join(str(Path.home()), config_yaml['ABS_DATA_PATH'])
    logger.info(f"")
    logger.info(f"DATASET_PATH: {DATASET_PATH}")

    df_projections = pd.read_csv( str(DATASET_PATH) + '/indiana_projections.csv')
    df_reports = pd.read_csv( str(DATASET_PATH) + '/indiana_reports.csv')


    logger.info(f"len(df_projections) : {len(df_projections)}")
    logger.info(f"len(df_reports) : {len(df_reports)}")

    assert len(df_projections) == 7466, f"Expected lenght of projections 7466"
    assert len(df_reports) == 3851, f"Expected length of reports 3851"
