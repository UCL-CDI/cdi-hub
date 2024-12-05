import os

import cv2
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


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
