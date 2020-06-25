import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# veri yukleme
df = pd.read_csv('Battery.csv')
y = df["Estimate_time"]
x = df[["Batt","Discharge_Current","Temp","Battery(+)"]]




#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
model = lin_reg2.fit(x_poly,y)
y_pred = model.predict(x_poly)
RMSE = np.sqrt(mean_squared_error(y,y_pred))
r2score = r2_score(y,y_pred)
