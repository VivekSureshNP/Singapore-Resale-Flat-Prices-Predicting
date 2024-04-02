# Singapore-Resale-Flat-Prices-Predicting
Singapore Resale Flat Prices Predicting
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Data Collection
data = pd.read_csv('resale_flat_prices.csv')  # Assuming 'resale_flat_prices.csv' is your dataset

# Step 2: Data Preprocessing
# Handle missing values, encode categorical variables, scale numerical features if necessary

# Step 3: Feature Engineering
# Create new features or transform existing ones

# Step 4: Model Selection
model = LinearRegression()  # You can use any other regression model here

# Step 5: Model Training
X = data.drop(columns=['resale_price'])  # Features
y = data['resale_price']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data
model.fit(X_train, y_train)  # Train the model

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Step 7: Hyperparameter Tuning (if necessary)

# Step 8: Prediction
# Use the trained model to make predictions on new data
