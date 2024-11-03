import csv
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
from scipy.special import softmax
import numpy as np

@dataclass
class ModelConfig:
    """Configuration parameters for the neural network model."""
    hidden_layers: List[int] = None
    max_iterations: int = 200
    random_state: int = 1234
    batch_size: Union[int, str] = "auto"
    activation: str = "logistic"
    solver: str = "lbfgs"

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [20, 10, 8, 6, 5]

@dataclass
class ModelMetrics:
    """Store model evaluation metrics."""
    accuracy: float
    confusion_matrix: np.ndarray
    classification_report: str
    predictions: np.ndarray

class MLTrader:
    """
    Neural Network-based trading model that handles data preprocessing,
    model training, evaluation, and result storage.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the MLPTrader with configuration.

        Args:
            config (ModelConfig, optional): Model configuration parameters.
        """
        self.config = config or ModelConfig()
        self.model: Optional[MLPClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the MLPClassifier with configured parameters."""
        self.model = MLPClassifier(
            tol=1e-6,
            solver=self.config.solver,
            activation=self.config.activation,
            hidden_layer_sizes=self.config.hidden_layers,
            max_iter=self.config.max_iterations,
            random_state=self.config.random_state,
            batch_size=self.config.batch_size,
            alpha=0.0
        )
        self.scaler = StandardScaler()

    def load_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and parse data.

        Args:
            file_path (str): Path to the file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels arrays.

        Raises:
            FileNotFoundError: If the specified file doesn't exist.
            ValueError: If the file format is invalid.
        """
        try:
            features: List[List[float]] = []
            labels: List[float] = []
            
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    labels.append(float(parts[0]))
                    
                    # Parse features
                    feat_dict = {
                        int(item.split(':')[0]): float(item.split(':')[1]) 
                        for item in parts[1:]
                    }
                    features.append([
                        feat_dict.get(i, 0.) 
                        for i in range(1, max(feat_dict.keys()) + 1)
                    ])
                    
            return np.array(features), np.array(labels)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error parsing data file: {str(e)}")

    def one_hot_encode(self, features: np.ndarray) -> np.ndarray:
        """
        Apply one-hot encoding to features following Spark's implementation.
        
        Args:
            features (np.ndarray): Input features array
            
        Returns:
            np.ndarray: One-hot encoded features
        """
        # Validate input values
        if np.any(features < 0):
            raise ValueError("Negative values found. Input can't be negative.")
        if not np.all(np.equal(np.mod(features, 1), 0)):
            raise ValueError("Values must be indices (integers).")
        if np.any(features > np.iinfo(np.int32).max):
            raise ValueError(f"OneHotEncoder only supports up to {np.iinfo(np.int32).max} indices.")

        n_samples = features.shape[0]
        encoded_features = []
        
        # Process each feature column
        for col_idx in range(features.shape[1]):
            col_data = features[:, col_idx]
            category_size = self.category_sizes[col_idx]
            
            # Determine the size of the encoded vector
            if self.get_drop_last() and not self.keep_invalid():
                output_size = category_size - 1
            elif not self.get_drop_last() and self.keep_invalid():
                output_size = category_size + 1
            else:
                output_size = category_size
                
            # Create encoded column
            encoded_col = np.zeros((n_samples, output_size))
            
            for i in range(n_samples):
                idx = int(col_data[i])
                if idx >= 0 and idx < category_size:
                    if idx < output_size:  # Only set 1 if idx is within output size
                        encoded_col[i, idx] = 1.0
                else:
                    if self.keep_invalid():
                        if not self.get_drop_last():
                            encoded_col[i, -1] = 1.0  # Set last position for invalid
                    else:
                        raise ValueError(f"Unseen value: {idx}. To handle unseen values, "
                                    "set handle_invalid to 'keep'.")
            
            encoded_features.append(encoded_col)
        
        # Concatenate all encoded features
        return np.hstack(encoded_features)

    def keep_invalid(self) -> bool:
        """Check if invalid values should be kept."""
        return self.config.handle_invalid == "keep"

    def get_drop_last(self) -> bool:
        """Get dropLast parameter value."""
        return self.config.drop_last

    def prepare_data(self, train_path: str, test_path: str) -> Tuple[
        Tuple[np.ndarray, np.ndarray], 
        Tuple[np.ndarray, np.ndarray]
    ]:
        """
        Load and preprocess both training and test data.

        Args:
            train_path (str): Path to training data file.
            test_path (str): Path to test data file.

        Returns:
            Tuple containing:
                - Training features and labels
                - Test features and labels
        """
        # Load raw data
        X_train, y_train = self.load_data(train_path)
        X_test, y_test = self.load_data(test_path)

        # Apply one-hot encoding
        X_train_encoded = self.one_hot_encode(X_train)
        X_test_encoded = self.one_hot_encode(X_test)

        # Scale features if needed
        if self.scaler is not None:
            X_train_scaled = self.scaler.fit_transform(X_train_encoded)
            X_test_scaled = self.scaler.transform(X_test_encoded)
        else:
            X_train_scaled = X_train_encoded
            X_test_scaled = X_test_encoded

        return (X_train_scaled, y_train), (X_test_scaled, y_test)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the neural network model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        """
        Evaluate model performance on test data using probability-based predictions.

        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): True labels.

        Returns:
            ModelMetrics: Collection of evaluation metrics.
        """
        # Get probability predictions for each class
        probabilities = self.model.predict_proba(X_test)
        
        # Apply custom softmax
        softmax_probabilities = self.custom_softmax(probabilities)
        
        # Get predicted labels from probabilities
        predictions = np.argmax(softmax_probabilities, axis=1)
        
        return ModelMetrics(
            accuracy=accuracy_score(y_test, predictions),
            confusion_matrix=confusion_matrix(y_test, predictions),
            classification_report=classification_report(y_test, predictions, zero_division=1),
            predictions=predictions
        )

    def custom_softmax(self, data: np.ndarray) -> np.ndarray:
        """
        Implementation of softmax function based on the provided Scala code.
        
        Args:
            data (np.ndarray): Input data of shape (n_samples, n_classes)
            
        Returns:
            np.ndarray: Softmax probabilities of same shape as input
        """
        output = np.zeros_like(data)
        
        # Process each sample
        for j in range(data.shape[1]):  # for each column
            # Find max value for numerical stability
            max_val = np.max(data[:, j])
            
            # Compute exp(x - max) for each value
            exp_vals = np.exp(data[:, j] - max_val)
            
            # Compute sum of exponentials
            sum_exp = np.sum(exp_vals)
            
            # Normalize by dividing by sum
            output[:, j] = exp_vals / sum_exp
        
        return output

    def save_results(self, y_test: np.ndarray, X_test: np.ndarray, 
                    predictions: np.ndarray, output_path: str) -> None:
        """
        Save prediction results to CSV file.

        Args:
            y_test (np.ndarray): True labels.
            X_test (np.ndarray): Test features.
            predictions (np.ndarray): Model predictions.
            output_path (str): Path for output file.
        """
        results = pd.DataFrame({
            'label': y_test,
            'features': [';'.join(map(str, feat)) for feat in X_test],
            'prediction': predictions
        })
        
        results.to_csv(output_path, index=False, sep=';', header=False)

    def process_financial_predictions(
        self, 
        predictions: np.ndarray, 
        rsi_test_path: str,
        output_path: str
        ) -> None:
            """
            Process predictions and RSI data to generate and save financial predictions.

            Args:
                predictions (np.ndarray): Model predictions
                rsi_test_path (str): Path to RSI test data
                output_path (str): Path for output file
            """
            # Read RSI test data
            with open(rsi_test_path, 'r') as file:
                rsi_test_data = list(csv.reader(file, delimiter=';'))

            # Process predictions
            builder: List[str] = []
            counter_zeros, counter_ones, counter_twos = 0, 0, 0
            row_price = 0

            for n in range(len(predictions)):
                if predictions[n] == 0:
                    counter_zeros += 1
                elif predictions[n] == 1:
                    counter_ones += 1
                elif predictions[n] == 2:
                    counter_twos += 1

                if (n + 1) % 20 == 0:
                    if counter_zeros > 14:
                        builder.append(f"{rsi_test_data[row_price][0]};0.0\n")
                    elif counter_ones > 14:
                        builder.append(f"{rsi_test_data[row_price][0]};1.0\n")
                    elif counter_twos > 14:
                        builder.append(f"{rsi_test_data[row_price][0]};2.0\n")
                    else:
                        builder.append(f"{rsi_test_data[row_price][0]};0.0\n")

                    counter_zeros, counter_ones, counter_twos = 0, 0, 0
                    row_price += 1

            # Write results
            with open(output_path, "w") as writer:
                writer.writelines(builder)