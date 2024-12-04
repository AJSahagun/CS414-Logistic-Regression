# CS414-Logistic-Regression

---

#### Overview

This Python script performs a logistic regression analysis to predict student placements based on their CGPA (Cumulative Grade Point Average). The key features include data preprocessing, model training, evaluation, and visualization of results. A simple menu-driven interface allows users to interact with the analysis, view results, and generate visualizations.

---

#### Requirements

- Python 3.x  
- Required Python Libraries:  
  - `pandas`  
  - `numpy`  
  - `matplotlib`  
  - `seaborn`  
  - `scikit-learn`  
  - `os`  

---

#### Setup

1. **Dataset Location**  
   - Place your dataset named `placement-predictor-dataset.csv` in the `Datasets` folder within your project directory.  
   - Ensure the dataset includes at least the following columns:  
     - `cgpa`: Feature (independent variable)  
     - `placement`: Target variable (binary: 0 or 1)  

2. **Run the Script**  
Execute the script in a Python environment. The menu interface will guide you through the available options.

---

#### Functionalities

1. **Display Analysis**
   - Display the Following:
   - Summation  of  CGPA Test values (`∑X`)
   - Summation of Placement Test values (`∑Y`)
   - Intercept (`β0`) and Coefficient (`β1`)
   - Model Accuracy  
    - Confusion Matrix  
    - Classification Report 
2. **Display First 20 CGPA  vs Value Probability **
   -  Shows a table comparing the actual and predicted probabilities for the **first 20 samples in the test set**, specifically:  
  - `CGPA (X)`: The input values.  
  - `Placement (Y)`: Actual labels.  
  - `Logistic Regression (π)`: Predicted probabilities.  

3. **Plot Actual vs Predicted Probabilities**  
- Scatter plot of CGPA vs. Placement with a logistic regression curve:  
  - Blue points represent actual placements.  
  - Red points represent predicted probabilities.  

4. **Plot Confusion Matrix**  
- Heatmap visualization of the confusion matrix.

---

### Usage Instructions

- Run the script, and choose from the following menu options:  

Choose an option:

1. Display Analysis  
2. Display First 20 Actual vs Predicted  
3. Plot Actual vs Predicted Probabilities  
4. Plot Confusion Matrix  
5. Exit

----

The **first 20 samples** from the test set are used for the prediction comparison table.   


---

#### Notes

- Ensure the dataset is clean and appropriately formatted before running the script.  
- Adjust hyperparameters or add more features if needed to improve model performance. 

  
