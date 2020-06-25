#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('Battery.csv')

#data frame dilimleme (slice)
y = veriler.iloc[:,4:5]
x = veriler.iloc[:,3:4]

#NumPY dizi (array) dönüşümü
X = x
Y = y


#linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
y_pred = lin_reg.predict(X)

#polynomial regression
#doğrusal olmayan (nonlinear model) oluşturma
#2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
y_pred2 = lin_reg2.predict(poly_reg.fit_transform(X))

# 6. dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 6)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)
y_pred3 = lin_reg3.predict(poly_reg3.fit_transform(X))

# Gorsellestirme
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.xlabel("Battery Percent")
plt.ylabel("Estimated Time")
plt.title("Linear Regression")
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.xlabel("Battery Percent")
plt.ylabel("Estimated Time")
plt.title("Polynominal Regression (Degree = 2)")
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.xlabel("Battery Percent")
plt.ylabel("Estimated Time")
plt.title("Polynominal Regression (Degree = 6)")
plt.show()












    

