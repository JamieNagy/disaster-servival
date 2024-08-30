
# Ship Disaster Survival Prediction

## Overview

This repository contains three predictive models for determining the survival of passengers in a ship disaster scenario. The goal of the project is to analyze the data and build a CatBoost model that can predict whether a passenger survived based on various features.

The project includes:

- **Jupyter Notebook (`disaster_survival.ipynb`)**: Contains the data analysis, feature engineering, model building, and evaluation.
- **Training Data (`train.csv`)**: The dataset used to train the machine learning model.
- **Test Data (`test.csv`)**: The dataset used to test the model's predictions, without the survival labels.

## Project Structure

```
├── disaster_survival.ipynb  # Jupyter Notebook with code and analysis
├── train.csv                # Training dataset
└── test.csv                 # Test dataset (without survival labels)
```

## Dataset Description

### Training Data (`train.csv`)

The training dataset contains the following columns:

- `PassengerId`: Unique identifier for each passenger.
- `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd).
- `Sex`: Gender of the passenger.
- `Age`: Age of the passenger.
- `SibSp`: Number of siblings/spouses aboard the ship.
- `Parch`: Number of parents/children aboard the ship.
- `Fare`: Ticket fare.
- `Embarked`: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).
- `Survived`: Survival status (0 = No, 1 = Yes) - **This is the target variable.**

### Test Data (`test.csv`)

The test dataset contains the same columns as the training data, **except for the `Survived` column**. The task is to predict the `Survived` status for each passenger in this dataset using the model built.

## Jupyter Notebook (`disaster_survival.ipynb`)

### Key Sections:

1. **Data Exploration & Preprocessing, and Feature Engineering**:
   - Visualization of data distributions with Matplotlib in an organized dashboard.
   - Handling missing values.
   - Encoding categorical variables.
   - Selection of the most relevant features for prediction.

3. **Model Building**:
   - Training CatBoost models.
   - Hyperparameter tuning using Grid Search and Bayesian Optimization.
   - Model evaluation with cross-validation.

4. **Prediction and Application**:
   - Apply the best model to the test data.
   - Generation of the final predictions for submission.

## How to Use

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Required Python libraries (see `disaster_survival.ipynb` for a list of imports)

### Steps to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ship-disaster-survival.git
   cd ship-disaster-survival
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook disaster_survival.ipynb
   ```

4. **Run the notebook**:
   - Follow the step-by-step code blocks in the notebook to train the model and generate predictions.
   - Tip: You can skip the Grid Search and Bayesian Optimization and just look at the outputted best parameters.
   - Modify and experiment with the code as needed.

### Application

The final predictions for the test dataset will be saved in a CSV file named `submission.csv`. This file contains two columns:

- `PassengerId`: The unique identifier for each passenger.
- `Survived`: The predicted survival status.

## Results and Analysis

- **Model Performance**: The notebook documents the models' performance during cross-validation, showing accuracy and other relevant metrics.
- **Final Accuracy**: The best model achieved an accuracy of 83.80%.


## Acknowledgments

- [Kaggle](https://www.kaggle.com/c/titanic) for the inspiration and the datasets.
- Various open-source libraries used in this project.
