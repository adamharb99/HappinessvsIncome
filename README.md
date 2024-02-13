# HappinessvsIncome

Changing this particular code:
import matplotlib.pyplot as plt
import sklearn.linear_model



X = np.c_[merged_train_data["Income Measurement"]]
Y = np.c_[merged_train_data["Happiness Measurement"]]
x = X.tolist()
y = Y.tolist()

# plot data
out1 = widgets.Output()
with out1:
  plt.scatter(x, y)
  plt.xlabel('Income')
  plt.ylabel('Happiness')
  plt.title("Data Plot")
  plt.show()

# fit linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, Y)

# plot predictions
predict_x = [x for x in range(901)]
predict_x = [[x/100] for x in predict_x]
predict_y = model.predict(predict_x)

out2 = widgets.Output()
with out2:
  plt.scatter(predict_x, predict_y)
  plt.scatter(x, y)
  plt.xlabel('Income')
  plt.ylabel('Happiness')
  plt.title("Prediction Line")
  plt.show()

display(widgets.HBox([out1,out2]))

from linear regression to non-linear regression to improve the data for happiness vs income
