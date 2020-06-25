import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# veri yukleme
veriler = pd.read_csv('Battery.csv')
#data frame dilimleme (slice)
y = veriler.iloc[:,4:5]
X = veriler.iloc[:,3:4]
#NumPY dizi (array) dönüşümü
X = x
Y = y

# Random Forest
modelRF = RandomForestRegressor().fit(X,Y) 
y_predRF = modelRF.predict(X)
# GradientBoostingRegressor
modelGBR = GradientBoostingRegressor().fit(X,Y)
y_predGBR = modelGBR.predict(X)
# LGBMRegressor
modelLGBR = GradientBoostingRegressor().fit(X,Y)
y_predLGBR = modelLGBR.predict(X)
# XGBRegressor
modelXGBR = LGBMRegressor().fit(X,Y)
y_predXGBR = modelXGBR.predict(X)
# DecisionTreeRegressor
modelDCR = DecisionTreeRegressor().fit(X,Y)
y_predDCR = modelDCR.predict(X)
# MLPRegressor
modelMLP = MLPRegressor().fit(X,Y)
y_predMLP = modelMLP.predict(X)
# KNeighborsRegressor
modelKn = KNeighborsRegressor().fit(X,Y)
y_predKn = modelKn.predict(X)
# SVR
modelSVR = SVR().fit(X,Y)
y_predSVR = modelSVR.predict(X)

# Gorsellestirme Random Forest
plt.scatter(X,Y, color = 'red') 
plt.plot(x,y_predRF, color ='blue') 
plt.xlabel("Battery Percent (%)") 
plt.ylabel("Estimated Time (sec)") 
plt.title("Random Forest Regression")
plt.show()
# Gorsellestirme GradientBoostingRegressor
plt.scatter(Y,X,color='red')
plt.plot(y_predGBR,x, color = 'blue')
plt.xlabel("Battery Percent (%)")
plt.ylabel("Estimated Time (sec)")
plt.title("Gradient Boosting Regressor")
plt.show()
# Gorsellestirme LGBMRegressor
plt.scatter(X,Y,color='red')
plt.plot(x,y_predLGBR, color = 'blue')
plt.xlabel("Battery Percent (%)")
plt.ylabel("Estimated Time (sec)")
plt.title("LGBMRegressor")
plt.show()
# Gorsellestirme XGBRegressor
plt.scatter(X,Y,color='red')
plt.plot(x,y_predXGBR, color = 'blue')
plt.xlabel("Battery Percent (%)")
plt.ylabel("Estimated Time (sec)")
plt.title("XGBRegressor")
plt.show()
# Gorsellestirme DecisionTreeRegressor
plt.scatter(X,Y,color='red')
plt.plot(x,y_predDCR, color = 'blue')
plt.xlabel("Battery Percent (%)")
plt.ylabel("Estimated Time (sec)")
plt.title("DecisionTreeRegressor")
plt.show()
# Gorsellestirme MLPRegressor
plt.scatter(X,Y,color='red')
plt.plot(x,y_predMLP, color = 'blue')
plt.xlabel("Battery Percent")
plt.ylabel("Estimated Time (sec)")
plt.title("MLPRegressor")
plt.show()
# Gorsellestirme KNeighborsRegressor
plt.scatter(X,Y,color='red')
plt.plot(x,y_predKn, color = 'blue')
plt.xlabel("Battery Percent (%)")
plt.ylabel("Estimated Time (sec)")
plt.title("KNeighborsRegressor")
plt.show()
# Gorsellestirme SVR
plt.scatter(X,Y,color='red')
plt.plot(x,y_predSVR, color = 'blue')
plt.xlabel("Battery Percent (%)")
plt.ylabel("Estimated Time (sec)")
plt.title("SVR")
plt.show()







