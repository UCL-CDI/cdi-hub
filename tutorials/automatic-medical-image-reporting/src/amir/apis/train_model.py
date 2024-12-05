"""
References:
https://www.kaggle.com/code/esenin/chestxnet2-0
python src/amir/apis/train_model.py
"""


import re
import time
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

########################
# data-preprocessing.py




nltk.download('stopwords')


import os

import cv2
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

HOME_PATH = Path().home()
DATA_PATH = HOME_PATH / 'datasets/chest-xrays-indiana-university/unzip'

df_projections = pd.read_csv( str(DATA_PATH) + '/indiana_projections.csv')
df_reports = pd.read_csv( str(DATA_PATH) + '/indiana_reports.csv')

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

filtered_df = images_captions_df[images_captions_df['diagnosis'] != 'normal']

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



class CheXNet_CNN_Dataset(Dataset):
    def __init__(self, dataframe, image_folder, image_size=224, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image file names and labels.
            image_folder (str): Path to the folder containing the images.
            image_size (int, optional): The target size to which images are resized. Defaults to 224.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_paths = [os.path.join(image_folder, filename) for filename in dataframe['image']]
        self.labels = dataframe['diagnosis'].tolist()
        self.image_size = image_size
        self.transform = transform

        # Encode labels to integers
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.encoded_labels[idx]

        # Load and preprocess image
        image = self.preprocess_image(image_path)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Return image and label
        return image, label

    def preprocess_image(self, image_path):
        """
        Loads and preprocesses an image.

        Args:
            image_path (str): Path to the image to load.

        Returns:
            torch.Tensor: The preprocessed image as a tensor.
        """
        # Load the image using OpenCV
        img = cv2.imread(image_path)
        # Resize the image
        img = cv2.resize(img, (self.image_size, self.image_size))
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize the image (0 to 1)
        img = img / 255.0
        # Convert to torch tensor
        img = torch.tensor(img, dtype=torch.float32)
        # Reorder dimensions to (C, H, W) for PyTorch
        img = img.permute(2, 0, 1)

        return img


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

# Initialize the train and test datasets
img_base_folder = DATA_PATH / 'images/images_normalized'

train_dataset = CheXNet_CNN_Dataset(train_df, img_base_folder, image_size=224, transform=data_train_transforms)
test_dataset = CheXNet_CNN_Dataset(test_df, img_base_folder, image_size=224, transform=data_test_transforms)

# Create DataLoaders for batching
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)







##############################################################################
# train_model.py

import ssl
import urllib.request

# Bypass SSL verification for se_resnext101_32x4d
ssl._create_default_https_context = ssl._create_unverified_context
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from pretrainedmodels import \
    se_resnext101_32x4d  # Use SENet-154 from pretrainedmodels
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Load the pre-trained SENet-154 model
base_model = se_resnext101_32x4d(pretrained='imagenet')  # Load pre-trained weights
# Downloading 187M/187M [09:11<00:00, 356kB/s]: "http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth" to $USER/.cache/torch/hub/checkpoints/se_resnext101_32x4d-3b2fe3d8.pth

# Modify the final layer (Inception-ResNet uses a 'last_linear' layer for classification)
num_features = base_model.last_linear.in_features
base_model.last_linear = nn.Linear(num_features, 3)  # Binary classification

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
#             print("outputs", outputs)
#             print("labels", labels)
#             print("preds", preds)
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



# Train the model

start_train_time = time.time()
print(f"start_train_time: {start_train_time}")
train(model, train_dataloader, criterion, optimizer, scheduler, num_epochs=1)
end_train_time = time.time()

elapsedtime = end_train_time - start_train_time
print(f"Elapsed time for the training loop: {elapsedtime/60} (mins)")


# Evaluate the model
evaluate(model, test_dataloader)
