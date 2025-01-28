# SPDX-FileCopyrightText: Copyright (c) University College London. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""unit test for data and ml workflow"""

import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from amir.utils.datasets import CheXNet_CNN_Dataset
from amir.utils.utils import display_image, preprocess_text
from loguru import logger
from pretrainedmodels import \
    se_resnext101_32x4d  # Use SENet-154 from pretrainedmodels
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

with open(str(Path().absolute())+"/tests/config_test.yml", "r") as file:
    config_yaml = yaml.load(file, Loader=yaml.FullLoader)


def test_CheXNet_CNN_Dataset():
    """
    Test CheXNet_CNN_Dataset class
    pytest -vs tests/test_ml_pipeline.py::test_CheXNet_CNN_Dataset

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

def test_train_eval_model():
    """
    Test train and evaluation of model pipeline
    pytest -vs tests/test_ml_pipeline.py::test_train_eval_model

        TRAIN 1 epoch in CPU
        Epoch [1/1], Loss: 0.9344, Accuracy: 0.5474
        Elapsed time for the training loop: 5.300666999816895 (mins)

        EVAL 1 epoch in CPU
        Test Accuracy: 0.5801
        Precision (Macro): 0.3852
        Recall (Macro): 0.4343
        F1 Score (Macro): 0.4074
        Elapsed time for the eval loop: 1.2158390720685324 (mins)

    References:
        https://www.kaggle.com/code/esenin/chestxnet2-0
    """


    # Load the pre-trained SENet-154 model
    base_model = se_resnext101_32x4d(pretrained='imagenet')  # Load pre-trained weights
    # Downloading 187M
    # "http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth"
    # to $USER/.cache/torch/hub/checkpoints/se_resnext101_32x4d-3b2fe3d8.pth

    # Modify the final layer (Inception-ResNet uses a 'last_linear' layer for classification)
    num_features = base_model.last_linear.in_features
    base_model.last_linear = nn.Linear(num_features, 3)  # Binary classification


    # Move the model to the appropriate device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device {device}")


    model = base_model.to(device)

    # Freeze all layers except the classifier
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    for param in model.last_linear.parameters():
        param.requires_grad = True  # Fine-tune only the classifier

    # Define loss function and optimizer for multi-class classification
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss is suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


    # Learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=1e-6)


    # Training function
    def train(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
        model.train()  # Set model to training mode
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)  # Move to GPU/CPU
                optimizer.zero_grad()  # Zero the gradients

                outputs = model(images)  # Get model predictions
                # print("outputs", outputs)
                # print("labels", labels)
                # print("preds", preds)
                loss = criterion(outputs, labels)  # Compute loss for multi-class classification
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model parameters

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)  # Get predicted class (index of max output)
                correct += (preds == labels).sum().item()  # Count correct predictions
                total += labels.size(0)  # Total number of samples

            # Calculate average loss and accuracy for the epoch
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            scheduler.step(epoch_loss)  # Step the learning rate scheduler

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')



    # Evaluation function
    def evaluate(model, test_loader):
        model.eval()  # Set model to evaluation mode
        total = 0
        correct = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)  # Move to GPU/CPU
                outputs = model(images)  # Forward pass
                _, preds = torch.max(outputs, 1)  # Get predicted class (index of max output)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                correct += (preds == labels).sum().item()  # Count correct predictions
                total += labels.size(0)  # Total number of samples

        # Calculate accuracy
        accuracy = correct / total

        # Calculate precision, recall, and F1 score for multi-class classification
        precision = precision_score(all_labels, all_preds, average='macro',zero_division=0)  # or 'weighted' if you want to weight by class size
        recall = recall_score(all_labels, all_preds, average='macro')  # or 'weighted'
        f1_score_val = f1_score(all_labels, all_preds, average='macro')  # or 'weighted'

        # Print the evaluation metrics
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Precision (Macro): {precision:.4f}')
        print(f'Recall (Macro): {recall:.4f}')
        print(f'F1 Score (Macro): {f1_score_val:.4f}')

    DATASET_PATH = os.path.join(str(Path.home()), config_yaml['ABS_DATA_PATH'])


    MODEL_PATH = os.path.join(DATASET_PATH, "models")
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)


    df_projections = pd.read_csv( str(DATASET_PATH) + '/indiana_projections.csv')
    df_reports = pd.read_csv( str(DATASET_PATH) + '/indiana_reports.csv')

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


    # Splitting the DataFrame into train and test sets
    train_df, test_df = train_test_split(images_captions_df, test_size=0.2, stratify=images_captions_df['diagnosis'])

    # # Initialize the train and test datasets
    img_base_folder = str(DATASET_PATH) + '/images/images_normalized'
    # logger.info(f"img_base_folder: {img_base_folder}")

    train_dataset = CheXNet_CNN_Dataset(train_df, img_base_folder, image_size=224, transform=None)
    test_dataset = CheXNet_CNN_Dataset(test_df, img_base_folder, image_size=224, transform=None)

    # Create DataLoaders for batching
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    logger.info(f"len(train_dataloader) {len(train_dataloader)}")
    logger.info(f"len(test_dataloader) {len(test_dataloader)}")

    # Train the model
    start_train_time = time.time()
    logger.info(f"start_train_time {start_train_time}")

    train(model, train_dataloader, criterion, optimizer, scheduler, num_epochs=1)

    end_train_time = time.time()
    elapsedtime = end_train_time - start_train_time
    logger.info(f"Elapsed time for the training loop: {elapsedtime/60} (mins)")

    # Evaluate the model
    start_train_time = time.time()
    logger.info(f"start_eval_time {start_train_time}")

    evaluate(model, test_dataloader)

    end_train_time = time.time()
    elapsedtime = end_train_time - start_train_time
    logger.info(f"Elapsed time for the eval loop: {elapsedtime/60} (mins)")

    current_time_stamp= datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    path_model_name = MODEL_PATH+"/_weights_" + current_time_stamp + ".pth"
    torch.save(model.state_dict(), path_model_name)
    logger.info(f"Saved PyTorch Model State to {path_model_name}")
