
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
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

