import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def create_sequences(X, y, timesteps=5):
    X_seq, y_seq = [], []

    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i+timesteps])
        y_seq.append(y[i+timesteps])

    return np.array(X_seq), np.array(y_seq)


def load_and_preprocess(file_path, target_col):

    df = pd.read_csv(file_path)

    # Drop missing values
    df = df.dropna()

    # Split features and target
    X = df.drop(columns=[target_col])

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    y = df[target_col]

    # Normalize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split (70-30)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, shuffle=False
    )

    # Convert to sequences (timesteps = 5)
    X_train, y_train = create_sequences(X_train, y_train.values, timesteps=5)
    X_test, y_test = create_sequences(X_test, y_test.values, timesteps=5)

    return X_train, X_test, y_train, y_test