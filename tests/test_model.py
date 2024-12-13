from ..model import download_and_preprocess_dataset
import joblib

def test_dataset_preprocessing():
    X_train, X_test, y_train, y_test = download_and_preprocess_dataset()
    assert X_train.shape[0] > 0, "X_train is empty!"
    assert X_test.shape[0] > 0, "X_test is empty!"
    assert len(y_train) == X_train.shape[0], "Mismatch in X_train and y_train length!"
    assert len(y_test) == X_test.shape[0], "Mismatch in X_test and y_test length!"

def test_model_saving():
    model = joblib.load("har_model.pkl")
    assert model is not None, "Model failed to load!"
