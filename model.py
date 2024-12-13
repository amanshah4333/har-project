import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import requests
import zipfile
import io

# Step 1: Download and preprocess dataset
def download_and_preprocess_dataset():
    # Dataset URL and directory
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    dataset_dir = "./UCI_HAR_Dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    # Download dataset if not already present
    if not os.path.exists(f"{dataset_dir}/dataset.zip"):
        print("Downloading dataset...")
        response = requests.get(dataset_url)
        print("Dataset download response status code:", response.status_code)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(dataset_dir)
        print("Dataset downloaded and extracted.")

    # Load feature names
    print("Loading feature names...")
    features = pd.read_csv(f"{dataset_dir}/UCI HAR Dataset/features.txt", delim_whitespace=True, header=None)
    feature_names = features[1].values
    print("Feature names loaded.")

    # Load training and test data
    print("Loading training and test datasets...")
    X_train = pd.read_csv(f"{dataset_dir}/UCI HAR Dataset/train/X_train.txt", delim_whitespace=True, header=None)
    X_test = pd.read_csv(f"{dataset_dir}/UCI HAR Dataset/test/X_test.txt", delim_whitespace=True, header=None)

    y_train = pd.read_csv(f"{dataset_dir}/UCI HAR Dataset/train/y_train.txt", delim_whitespace=True, header=None)
    y_test = pd.read_csv(f"{dataset_dir}/UCI HAR Dataset/test/y_test.txt", delim_whitespace=True, header=None)
    print("Training and test datasets loaded.")

    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Features normalized.")

    # Map activity labels to binary classes (Walking: 0, Running: 1)
    print("Mapping activity labels to binary classes...")
    y_train = y_train[0].map({1: 0, 2: 1}).fillna(-1).astype(int).values  # Adjust as needed for binary classification
    y_test = y_test[0].map({1: 0, 2: 1}).fillna(-1).astype(int).values
    print("Activity labels mapped.")

    print("Dataset preprocessing complete.")
    return X_train, X_test, y_train, y_test

# Step 2: Train and evaluate model
def train_and_evaluate_model():
    # Load and preprocess dataset
    print("Starting dataset preprocessing...")
    X_train, X_test, y_train, y_test = download_and_preprocess_dataset()
    print("Dataset preprocessing finished.")

    # Define and train a Logistic Regression model
    print("Defining Logistic Regression model...")
    model = LogisticRegression()
    print("Training the model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy}")

    # Save the model
    print("Saving the model...")
    import joblib
    joblib.dump(model, "har_model.pkl")
    print("Model saved as 'har_model.pkl'.")

# Run the entire workflow
if __name__ == "__main__":
    print("Starting training and evaluation workflow...")
    train_and_evaluate_model()
    print("Workflow complete.")
