import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd 

# Generate sample data
np.random.seed(0)
income = np.random.randint(20000, 100000, size=100)
happiness = np.random.randint(1, 11, size=100)

# Create a DataFrame
merged_train_data = pd.DataFrame({'Income Measurement': income, 'Happiness Measurement': happiness})

# Extract features and target
X = merged_train_data[["Income Measurement"]]
Y = merged_train_data["Happiness Measurement"]

# Transforming features into polynomial features
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Fit polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, Y)

# Predicting on new data
predict_x = np.linspace(20000, 100000, 100).reshape(-1, 1)
predict_x_poly = poly_features.transform(predict_x)
predict_y = poly_model.predict(predict_x_poly)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.scatter(X, Y)
plt.plot(predict_x, predict_y, color='red')
plt.xlabel('Income')
plt.ylabel('Happiness')
plt.title("Polynomial Regression")
plt.show()
