import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
import warnings

def load_libsvm(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from a LIBSVM formatted file.

    Args:
        file_path (str): Path to the LIBSVM file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and labels as numpy arrays.
    """
    features: List[List[float]] = []
    labels: List[float] = []
    with open(file_path, 'r') as f:
        for line in f:
            parts: List[str] = line.strip().split()
            labels.append(float(parts[0]))
            feat: Dict[int, float] = {int(item.split(':')[0]): float(item.split(':')[1]) for item in parts[1:]}
            features.append([feat.get(i, 0.) for i in range(1, max(feat.keys()) + 1)])
    return np.array(features), np.array(labels)

# In phase3.py
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """Train a Multi-Layer Perceptron Classifier."""
    layers: List[int] = [20, 10, 8, 6, 5]
    model = MLPClassifier(
        hidden_layer_sizes=layers,
        max_iter=1000,  # Increased from 200
        random_state=1234,
        batch_size='auto',  # Changed from 128
        learning_rate='adaptive',
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        solver='adam',
        alpha=0.0001
    )
    
    # Remove the sample_weight parameter
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, np.ndarray, str]:
    """
    Evaluate the trained model.

    Args:
        model (MLPClassifier): Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        Tuple[float, np.ndarray, str]: Accuracy, confusion matrix, and classification report.
    """
    predictions: np.ndarray = model.predict(X_test)
    accuracy: float = accuracy_score(y_test, predictions)
    cm: np.ndarray = confusion_matrix(y_test, predictions)
    cr: str = classification_report(y_test, predictions, zero_division=1)
    return accuracy, cm, cr, predictions

def save_results(y_test: np.ndarray, X_test: np.ndarray, predictions: np.ndarray) -> None:
    """
    Save the results to a CSV file.

    Args:
        y_test (np.ndarray): True labels.
        X_test (np.ndarray): Test features.
        predictions (np.ndarray): Predicted labels.
    """
    results = pd.DataFrame({
        'label': y_test,
        'features': [';'.join(map(str, feat)) for feat in X_test],
        'prediction': predictions
    })
    results.to_csv("resources2/outputMLP.csv", index=False, sep=';', header=False)

def preprocess_data(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the data."""
    # Handle missing values
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def phase_process() -> None:
    """
    Main function to process the data, train the model, evaluate it, and save results.
    """
    try:
        # Load training and test data
        X_train, y_train = load_libsvm("resources2/GATableListTraining.txt")
        X_test, y_test = load_libsvm("resources2/GATableListTest.txt")

        # Preprocess data
        X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
        X_test_scaled: np.ndarray = scaler.transform(X_test)

        # Train the model
        model: MLPClassifier = train_model(X_train_scaled, y_train)

        # Evaluate the model
        accuracy, cm, cr, predictions = evaluate_model(model, X_test_scaled, y_test)

        # Print results
        print(f"Test set accuracy = {accuracy}")
        print("Confusion matrix:")
        print(cm)
        print("Classification Report:")
        print(cr)

        # Save results
        save_results(y_test, X_test, predictions)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    phase_process()

