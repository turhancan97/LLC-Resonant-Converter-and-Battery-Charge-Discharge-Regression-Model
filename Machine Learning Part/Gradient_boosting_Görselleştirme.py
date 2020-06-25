import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor

# veri yukleme
veriler = pd.read_csv('Battery.csv')

#data frame dilimleme (slice)
y = veriler.iloc[:,4:5]
x = veriler.iloc[:,3:4]

#NumPY dizi (array) dönüşümü
X = x
Y = y


model = GradientBoostingRegressor().fit(X,Y)
y_pred = model.predict(X)

# Gorsellestirme
plt.scatter(X,Y,color='red')
plt.plot(x,y_pred, color = 'blue')
plt.xlabel("Battery Percent")
plt.ylabel("Estimated Time")
plt.title("Gradient Boosting Regressor")
plt.show()