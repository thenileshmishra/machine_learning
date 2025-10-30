# Student Performance Prediction Project

## Project Overview
A machine learning project that predicts student performance using both Linear Regression and Neural Networks, demonstrating proficiency in data preprocessing, feature engineering, and model optimization.

## Technical Implementation

### Data Preprocessing & Feature Engineering
- **Categorical Encoding**:
  - Gender: Binary encoding (male: 0, female: 1)
  - Race/Ethnicity: One-hot encoding with drop_first to avoid multicollinearity
  - Parental Education: Ordinal encoding (0-5 scale)
  - Lunch Type: Binary encoding (standard: 0, free/reduced: 1)
  - Test Preparation: Binary encoding (none: 0, completed: 1)

### Models Implemented

#### 1. Linear Regression Pipeline
- **Preprocessing Pipeline**:
  ```python
  - Numeric features: SimpleImputer(median) → StandardScaler
  - Categorical features: SimpleImputer(most_frequent) → OneHotEncoder
  - Ordinal features: SimpleImputer → (preserved numeric encoding)
  ```
- **Benefits**: Prevents data leakage, ensures reproducible preprocessing

#### 2. Neural Network Model
- **Architecture**:
  ```python
  Input Layer → Dense(64) → ReLU → Dropout(0.3) 
  → Dense(32) → ReLU → Dropout(0.2) → Dense(1)
  ```
- **Hyperparameter Optimization**:
  - Learning rates: [0.01, 0.001]
  - Weight decay: [1e-3, 1e-4]
  - Dropout rates: [0.3, 0.5] and [0.2, 0.4]
  - Validation-based model selection

## Interview Discussion Points

### 1. Data Preprocessing Decisions
- Why use different encoding strategies for different categorical variables?
  - Ordinal encoding for ordered categories (education levels)
  - One-hot for nominal categories (race/ethnicity)
  - Binary for binary categories (gender, lunch type)
- How did you handle potential data leakage?
  - Used scikit-learn Pipelines to ensure preprocessing steps are fit only on training data
  - Applied transformations consistently across train/test splits
- Why standardize numeric features?
  - Ensures all features contribute equally to the model
  - Improves neural network convergence
  - Makes regularization more effective

### 2. Model Architecture Choices
- Why implement both Linear Regression and Neural Networks?
  - Compare simple vs complex models
  - Demonstrate understanding of both approaches
  - Evaluate if complexity is justified
- How did you decide on the neural network architecture?
  - Progressive reduction in layer sizes (64→32→1)
  - ReLU for non-linearity
  - Dropout for regularization
- What's the significance of using dropout layers?
  - Prevents overfitting
  - Improves model generalization
  - Acts as model ensemble during inference

### 3. Model Evaluation & Validation
- How did you prevent overfitting?
  - Dropout layers in neural network
  - Early stopping based on validation loss
  - L2 regularization (weight decay)
- Why use MSE as the loss function?
  - Natural choice for regression problems
  - Penalizes larger errors more heavily
  - Differentiable for gradient descent
- How did you validate the model's performance?
  - Train/test split for honest evaluation
  - Monitoring validation loss during training
  - Grid search for hyperparameter optimization

### 4. Technical Skills Demonstrated
- **Python Libraries**: pandas, numpy, scikit-learn, PyTorch
- **ML Concepts**: Feature engineering, pipeline creation, hyperparameter tuning
- **Software Engineering**: Modular code, version control, model serialization

### 5. Potential Improvements
- Feature importance analysis
- Cross-validation implementation
- Ensemble methods exploration
- Advanced feature engineering

## Project Structure
```
LinearRegression/
├── EDA.ipynb                    # Exploratory Data Analysis
├── FeatureEngineering.ipynb    # Feature preprocessing
├── ModelNN.ipynb               # Neural Network implementation
├── model/
│   └── pipeline.py            # Linear Regression pipeline
├── StudentsPerformance.csv    # Raw data
└── StudentCleaned.csv         # Processed data
```

## Model Artifacts
- `neural_regressor.pth`: PyTorch model weights
- `neural_regressor.pkl`: Serialized model for deployment
- `scaler.pkl`: Feature standardization parameters

## Key Results
- Successfully implemented both traditional and deep learning approaches
- Created robust preprocessing pipelines
- Demonstrated practical ML deployment considerations
- Applied systematic hyperparameter optimization

## Running the Project
```bash
# Install dependencies
pip install -r requirements.txt

# Run Linear Regression
python LinearRegression/model/pipeline.py

# Run Neural Network Training
jupyter notebook LinearRegression/ModelNN.ipynb
```