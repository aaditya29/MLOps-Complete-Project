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

    # apply preprocessing to the specified column
    df[col] = df[col].apply(preprocess_text)
    df = df.dropna(subset=[col])
    logging.info("Data pre-processing completed")
    return df


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info('data loaded properly')

        # Transform the data
        train_processed_data = preprocess_dataframe(train_data, 'review')
        test_processed_data = preprocess_dataframe(test_data, 'review')

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(
            data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(
            data_path, "test_processed.csv"), index=False)

        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error(
            'Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
