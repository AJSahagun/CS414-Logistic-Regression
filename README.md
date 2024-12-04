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

---

### Usage Instructions

1. **Run the Script**  
   Execute the script in a Python environment. The menu interface will guide you through the available options.  
   
   To run the script, follow these steps:

   - Open a terminal (or command prompt) on your system.
   - Navigate to the directory where your script is saved using the `cd` command. For example:
     ```
     cd path/to/your/script/directory
     ```
   - Run the script using Python by typing the following command:
     ```
     python script.py
     ```
     Replace `script.py` with the actual name of your script file if it's different.
   
   The script will execute and display the following menu options for you to choose from:

   Choose an option:
   1. Display Analysis  
   2. Display First 20 Actual vs Predicted  
   3. Plot Actual vs Predicted Probabilities  
   4. Plot Confusion Matrix  
   5. Exit

   - Enter the corresponding number to perform the desired action.
   - Follow the instructions displayed on the screen for each option

----

The **first 20 samples** from the test set are used for the prediction comparison table.   


---

#### Notes

- Ensure the dataset is clean and appropriately formatted before running the script.  
- Adjust hyperparameters or add more features if needed to improve model performance. 

  
