"""
References:
https://www.kaggle.com/code/esenin/chestxnet2-0
python src/amir/apis/data-preprocessing.py
"""

import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')


import os

import matplotlib.pyplot as plt
from amir.utils.datasets import CheXNet_CNN_Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

HOME_PATH = Path().home()
DATA_PATH = HOME_PATH / 'datasets/chest-xrays-indiana-university/unzip'

df_projections = pd.read_csv( str(DATA_PATH) + '/indiana_projections.csv')
df_reports = pd.read_csv( str(DATA_PATH) + '/indiana_reports.csv')


logger.info(f"len(df_projections) : {len(df_projections)}")
logger.info(f"len(df_reports) : {len(df_reports)}")
logger.info(f"df_projections : {df_projections}")
logger.info(f"df_reports: {df_reports}")

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




# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = text.replace('/', ' ').replace(';', ' ')
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Join back into a single string
    return " ".join(tokens)

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


def display_image(image_tensor):
    """
    Displays an image.

    Args:
        image_tensor (torch.Tensor): The image tensor to display.
    """
    image_np = image_tensor.permute(1, 2, 0).numpy()  # Convert to HWC format
    plt.imshow(image_np)
    plt.axis('off')  # Hide axis labels
    plt.show()

# Splitting the DataFrame into train and test sets
train_df, test_df = train_test_split(images_captions_df, test_size=0.2, stratify=images_captions_df['diagnosis'])

# Initialize the train and test datasets
img_base_folder = DATA_PATH / 'images/images_normalized'

train_dataset = CheXNet_CNN_Dataset(train_df, img_base_folder, image_size=224, transform=data_train_transforms)
test_dataset = CheXNet_CNN_Dataset(test_df, img_base_folder, image_size=224, transform=data_test_transforms)

# Create DataLoaders for batching
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


for images, labels in train_dataloader:
    display_image(images[0])  # Display the first image in the batch
    break

for images, labels in test_dataloader:
    display_image(images[0])  # Display the first image in the batch
    break
