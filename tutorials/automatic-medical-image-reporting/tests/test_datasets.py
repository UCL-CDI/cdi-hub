import os
from pathlib import Path

import pandas as pd
import yaml
from amir.utils.datasets import CheXNet_CNN_Dataset
from amir.utils.utils import preprocess_text, display_image
from loguru import logger
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset



with open(str(Path().absolute())+"/tests/config_test.yml", "r") as file:
    config_yaml = yaml.load(file, Loader=yaml.FullLoader)


def test_CheXNet_CNN_Dataset():
    """
    Test CheXNet_CNN_Dataset class
    pytest -vs tests/test_datasets.py::test_CheXNet_CNN_Dataset
        TODO:
            - Use 640x400 image size
            - Test mask to show sclera, iris, pupil and background
    References:
        https://www.kaggle.com/code/esenin/chestxnet2-0
    """
    # Define transforms - note we do ToTensor in the dataset class


    DATASET_PATH = os.path.join(str(Path.home()), config_yaml['ABS_DATA_PATH'])
    logger.info(f"")
    logger.info(f"DATASET_PATH: {DATASET_PATH}")

    df_projections = pd.read_csv( str(DATASET_PATH) + '/indiana_projections.csv')
    df_reports = pd.read_csv( str(DATASET_PATH) + '/indiana_reports.csv')


    logger.info(f"len(df_projections) : {len(df_projections)}")
    logger.info(f"len(df_reports) : {len(df_reports)}")
    logger.info(f"df_projections : {df_projections}")
    logger.info(f"df_reports: {df_reports}")


    assert len(df_projections) == 7466, f"Expected length of projections 7466"
    assert len(df_reports) == 3851, f"Expected length of reports 3851"


    df_frontal_projections = df_projections[df_projections['projection'] == 'Frontal']
    df_frontal_projections['projection'].unique()


    images_captions_df = pd.DataFrame({'image': [], 'diagnosis': [],
                                        'caption': [],'number_of_words':[]})
    for i in range(len(df_frontal_projections)):
        uid = df_frontal_projections.iloc[i]['uid']
        image = df_frontal_projections.iloc[i]['filename']
        index = df_reports.loc[df_reports['uid'] ==uid]

        if not index.empty:
            index = index.index[0]
            caption = df_reports.iloc[index]['findings']
            diagnosis = df_reports.iloc[index]['MeSH']

            number_of_words = len(str(caption).split())

            if type(caption) == float:
                # TO DO: handle NaN
                continue
            images_captions_df = pd.concat([images_captions_df, pd.DataFrame([{'image': image, 'diagnosis':diagnosis, 'caption': caption ,'number_of_words':number_of_words}])], ignore_index=True)

    images_captions_df["number_of_words"] =  images_captions_df["caption"].apply(lambda text: len(str(text).split()))
    images_captions_df['number_of_words'] = images_captions_df['number_of_words'].astype(int)

    logger.info(f"len(images_captions_df) {len(images_captions_df)}")
    logger.info(f"images_captions_df[:10] {images_captions_df[:10]}")






    # Apply preprocessing to the 'caption' and 'diagnosis' columns
    images_captions_df['caption'] = images_captions_df['caption'].apply(preprocess_text)
    images_captions_df['diagnosis'] = images_captions_df['diagnosis'].apply(preprocess_text)

    logger.info(f"images_captions_df[['image', 'diagnosis', 'caption']].head() {images_captions_df[['image', 'diagnosis', 'caption']].head()}")

    filtered_df = images_captions_df[images_captions_df['diagnosis'] != 'normal']
    logger.info(f"filtered_df['diagnosis'] {filtered_df['diagnosis']}")


    # Define pneumonia keywords
    pulmonary_keywords = ['pulmonary']
    # pneumonia_keywords = ['Lung', 'Lungs', 'pneumonia', 'Pulmonary','Pulmonary', 'asthma']
    # pneumonia_keywords = ['alveolitis', 'bronchopneumonia', 'pneumonia', 'pneumonitis', 'lung infection',
    #                      'Alveolitis', 'Bronchopneumonia', 'Pneumonia', 'Pneumonitis', 'Lung infection',
    #                      'Lung']

    # Function to classify diagnosis as 'normal' or 'pneumonia'
    def classify_diagnosis(diagnosis):
        if any(keyword in str(diagnosis) for keyword in pulmonary_keywords):
            return 'pulmonary'
        if str(diagnosis) == 'normal':
            return 'normal'
        return 'other'

    # Apply the function to the diagnosis column
    images_captions_df['diagnosis'] = images_captions_df['diagnosis'].apply(classify_diagnosis)

    pulmonary_len = len(images_captions_df[images_captions_df['diagnosis'] == 'pulmonary'])
    normal_len = len(images_captions_df[images_captions_df['diagnosis'] == 'normal'])
    other_len = len(images_captions_df[images_captions_df['diagnosis'] == 'other'])

    logger.info(f"len(images_captions_df) {len(images_captions_df)}")
    logger.info(f"pulmonary_len {pulmonary_len}")
    logger.info(f"normal_len {normal_len}")
    logger.info(f"other_len {other_len}")


    # Data preprocessing using transforms
    data_train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(15),  # Random rotation (15 degrees)
        transforms.RandomAffine(translate=(0.1, 0.1), scale=(0.9, 1.1), degrees=(-10, 10)),  # Random affine transformation (translation and scaling)
        transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Data preprocessing using transforms
    data_test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # Splitting the DataFrame into train and test sets
    train_df, test_df = train_test_split(images_captions_df, test_size=0.2, stratify=images_captions_df['diagnosis'])

    # # Initialize the train and test datasets
    img_base_folder = str(DATASET_PATH) + '/images/images_normalized'
    # logger.info(f"img_base_folder: {img_base_folder}")

    train_dataset = CheXNet_CNN_Dataset(train_df, img_base_folder, image_size=224, transform=data_train_transforms)
    test_dataset = CheXNet_CNN_Dataset(test_df, img_base_folder, image_size=224, transform=data_test_transforms)

    # Create DataLoaders for batching
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    logger.info(f"len(train_dataloader) {len(train_dataloader)}")
    logger.info(f"len(test_dataloader) {len(test_dataloader)}")

    assert len(train_dataloader) == 166, f"Expected length of train_dataloader 166"
    assert len(test_dataloader) == 42, f"Expected length of test_dataloader 42"

    for images, labels in train_dataloader:
        display_image(images[0])  # Display the first image in the batch
        break

    for images, labels in test_dataloader:
        display_image(images[0])  # Display the first image in the batch
        break
