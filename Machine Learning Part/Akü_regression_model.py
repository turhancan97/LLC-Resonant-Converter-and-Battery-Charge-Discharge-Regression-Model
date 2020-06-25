#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('Battery.csv')

#data frame dilimleme (slice)
y = veriler.iloc[:,-1]
X = veriler.iloc[:,3:4]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)

#polynomial regression
#doğrusal olmayan (nonlinear model) oluşturma
#2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_train)
y_pred2 = lin_reg2.predict(poly_reg.fit_transform(X_test))

# 4. dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X_train)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y_train)
y_pred3 = lin_reg3.predict(poly_reg3.fit_transform(X_test))
# Gorsellestirme
plt.scatter(X_train,y_train,color='red')
plt.plot(X_test,y_pred, color = 'blue')
plt.show()

plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_test,y_pred2, color = 'blue')
plt.show()

plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_test,y_pred3, color = 'blue')
plt.show()