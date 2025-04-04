import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load datasets
phishtank_data = pd.read_csv('phishtank_dataset.csv')
alexa_data = pd.read_csv('alexa_dataset.csv')
enron_data = pd.read_csv('enron_email_dataset.csv')
uci_data = pd.read_csv('uci_phishing_dataset.csv')

# Data cleaning & preprocessing
def preprocess_url_data(df):
    # Handle missing values
    df = df.dropna(subset=['url', 'is_phishing'])
    
    # Convert label to binary
    df['is_phishing'] = df['is_phishing'].astype(int)
    
    return df

# Merge relevant datasets
processed_data = preprocess_url_data(phishtank_data)
processed_data = pd.concat([processed_data, preprocess_url_data(uci_data)])

# Split into training and testing sets
X = processed_data.drop('is_phishing', axis=1)
y = processed_data['is_phishing']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)