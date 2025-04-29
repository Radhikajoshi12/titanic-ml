# Titanic Survival Prediction 🚢

This project uses machine learning to predict Titanic survival based on passenger data.

## 📁 Dataset
- `tested.csv`: Contains test data for the Titanic dataset.

## Preprocessing
- Filled missing `Age` values with the median.
- Filled missing `Embarked` values with the mode.
- Encoded categorical variables (`Sex`, `Embarked`) using LabelEncoder.
- Scaled numerical features (`Age`, `Fare`) using StandardScaler.

## Model
- Algorithm used: `RandomForestClassifier` from scikit-learn.

## Evaluation Results
Achieved perfect accuracy on test set:
- **Accuracy:** 1.0  
- **Precision:** 1.0  
- **Recall:** 1.0  
- **F1 Score:** 1.0

##  How to Run
```bash
python titanic.py
