import pandas as pd
from os import path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "placement-predictor-dataset.csv")

def logistic_regression_analysis():
    # Load the dataset
    df = pd.read_csv(DATA_FILE)
    
    # Independent and target variables
    X = df[['cgpa']] 
    y = df['placement'] 
    
    # Split the data into training and testing sets (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Create and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for class 1
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    sum_X = X_test['cgpa'].sum()
    sum_Y = y_test.sum()
    beta_0 = model.intercept_[0]
    beta_1 = model.coef_[0][0]
    
    # Function to display analysis results
    def display_analysis():
        print("\n--- Analysis ---")
        print(f"∑X (Sum of {'cgpa'}): {sum_X:.3f}")
        print(f"∑Y (Sum of {'placement'}): {sum_Y}")
        print(f"β0 (Intercept): {beta_0:.4f}")
        print(f"β1 (Coefficient for {'cgpa'}): {beta_1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(report)
    
    # Function to display first 20 actual vs predicted values
    def display_comparison():
        output_table = pd.DataFrame({
            'CGPS (X)': X_test['cgpa'],
            'Placement (Y)': y_test.values,
            'Logistic Regression (π)': y_pred_proba
        }).reset_index(drop=True)


        print("\n--- Logistic Regression Predictions ---")
        print(output_table.head(20).to_string(index=False))
    
    # Function to plot Actual vs Predicted probabilities
    def plot_actual_vs_predicted_probabilities():
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test['cgpa'], y_test, color='blue', label='CGPA')
        plt.scatter(X_test['cgpa'], y_pred_proba, color='red', label='Predicted Probability', alpha=0.6)
        X_sorted = np.linspace(X_test['cgpa'].min(), X_test['cgpa'].max(), 300).reshape(-1, 1)
        X_sorted_scaled = scaler.transform(pd.DataFrame(X_sorted, columns=['cgpa']))
        y_curve = model.predict_proba(X_sorted_scaled)[:, 1]
        plt.plot(X_sorted, y_curve, color='red', label='Logistic Regression Curve', linewidth=2)
        plt.xlabel('Feature (cgpa)') 
        plt.ylabel('Placement')
        plt.title('CGPA vs. Placement Probability')
        plt.legend()
        plt.grid(True)
        plt.show()
   
    # Function to plot Confusion Matrix
    def plot_confusion_matrix():
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    # Menu for user interaction
    menu_options = {
        '1': display_analysis,
        '2': display_comparison,
        '3': plot_actual_vs_predicted_probabilities,
        '4': plot_confusion_matrix,
        '5': lambda: print("Exiting...")
    }
    
    while True:
        print("\nChoose an option:")
        print("1. Display Analysis")
        print("2. Display First 20 Actual vs Predicted")
        print("3. Plot Actual vs Predicted Probabilities")
        print("4. Plot Confusion Matrix")
        print("5. Exit")
        
        choice = input("Enter the number of your choice: ")
        if choice in menu_options:
            if choice == '5':  # Exit case
                menu_options[choice]()
                break
            else:
                menu_options[choice]()
        else:
            print("Invalid choice, please select a valid option.")

# Run the analysis
logistic_regression_analysis()