import os
import re
import nltk
import string
import numpy as np
import pandas as pd
from src.logger import logging
from nltk.corpus import stopwords
# for lemmatization i.e. converting words to their base form
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')  # Download WordNet data for lemmatization
nltk.download('stopwords')  # Download stopwords list


def preprocess_dataframe(df, col='txt'):

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))  # Define stopwords set

    def preprocess_text(text):
        """Helper function to preprocess a single text string."""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove numbers
        text = ''.join([char for char in text if not char.isdigit()])
        # Convert to lowercase
        text = text.lower()
        # Remove punctuations
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        # Remove stop words
        text = " ".join([word for word in text.split()
                        if word not in stop_words])
        # Lemmatization
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
