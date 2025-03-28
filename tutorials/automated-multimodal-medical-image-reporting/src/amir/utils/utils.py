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
"""various utility functions"""

import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
import matplotlib.pyplot as plt


# Preprocessing function
def preprocess_text(text):

    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

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
