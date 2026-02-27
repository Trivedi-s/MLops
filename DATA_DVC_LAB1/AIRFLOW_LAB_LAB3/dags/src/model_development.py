import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_data():
    """
    Loads advertising data from CSV file.
    Returns:
        DataFrame: Loaded data
    """
    print("=" * 50)
    print("TASK 1: Loading Data")
    print("=" * 50)
    
    # Update this path to match your VM path
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/advertising.csv"))
    
    print(f"Data loaded successfully!")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print("=" * 50)
    
    return data


def data_preprocessing(data):
    """
    Preprocesses data: drops unnecessary columns, scales features, splits into train/test.
    Returns:
        tuple: (X_train, X_test, y_train, y_test) as lists for serialization
    """
    print("=" * 50)
    print("TASK 2: Data Preprocessing")
    print("=" * 50)
    
    # Drop non-numeric and target columns
    X = data.drop(['Timestamp', 'Clicked on Ad', 'Ad Topic Line', 'Country', 'City'], axis=1)
    y = data['Clicked on Ad']
    
    print(f"Features used: {list(X.columns)}")
    print(f"Target: Clicked on Ad")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Define columns for scaling
    num_columns = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']
    
    # Create column transformer for preprocessing
    ct = make_column_transformer(
        (MinMaxScaler(), num_columns),
        (StandardScaler(), num_columns),
        remainder='passthrough'
    )
    
    # Transform the data
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)
    
    print("Data scaling completed!")
    print("=" * 50)
    
    # Return as lists for Airflow serialization
    return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()


def build_model(data, filename):
    """
    Builds and saves a Random Forest classifier.
    Args:
        data: Tuple of (X_train, X_test, y_train, y_test)
        filename: Name for saving the model
    """
    print("=" * 50)
    print("TASK 3: Building Random Forest Model")
    print("=" * 50)
    
    # Convert lists back to numpy arrays
    X_train = np.array(data[0])
    X_test = np.array(data[1])
    y_train = np.array(data[2])
    y_test = np.array(data[3])
    
    # Create Random Forest classifier with custom parameters
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest Classifier...")
    print(f"Parameters: n_estimators=100, max_depth=10")
    
    # Train the model
    rf_clf.fit(X_train, y_train)
    
    print("Model training completed!")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    
    # Save the model
    pickle.dump(rf_clf, open(output_path, 'wb'))
    
    print(f"Model saved to: {output_path}")
    print("=" * 50)


def evaluate_model(data, filename):
    """
    Loads the saved model and evaluates its performance.
    Args:
        data: Tuple of (X_train, X_test, y_train, y_test)
        filename: Name of the saved model file
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("=" * 50)
    print("TASK 4: Model Evaluation")
    print("=" * 50)
    
    # Convert lists back to numpy arrays
    X_train = np.array(data[0])
    X_test = np.array(data[1])
    y_train = np.array(data[2])
    y_test = np.array(data[3])
    
    # Load the saved model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", filename)
    loaded_model = pickle.load(open(model_path, 'rb'))
    
    print("Model loaded successfully!")
    
    # Make predictions
    y_pred = loaded_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "=" * 30)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 30)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 30)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Click', 'Click']))
    
    # Save metrics to file
    metrics_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    metrics_path = os.path.join(metrics_dir, "model_metrics.txt")
    
    with open(metrics_path, 'w') as f:
        f.write("=" * 40 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write(f"Model: Random Forest Classifier\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['No Click', 'Click']))
    
    print(f"\nMetrics saved to: {metrics_path}")
    print("=" * 50)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


if __name__ == '__main__':
    # Test locally
    data = load_data()
    processed_data = data_preprocessing(data)
    build_model(processed_data, 'random_forest_model.sav')
    evaluate_model(processed_data, 'random_forest_model.sav')