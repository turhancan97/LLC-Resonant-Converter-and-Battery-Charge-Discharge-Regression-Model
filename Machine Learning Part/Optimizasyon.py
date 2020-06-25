import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv("Battery.csv")
y = df["Estimate_time"]
X = df[["Batt","Discharge_Current","Temp","Battery(+)"]]   
    
# Train-Test Ayrımı
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

# Modelleme
rf_model = RandomForestRegressor(random_state=42).fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
r2score=r2_score(y_test,y_pred)

#Optimizasyon
#rf_params = {"max_depth": [5,3],"max_features": [2,4],"n_estimators":[200,1000],"min_samples_split":[2,4]}

#rf_cv_model = GridSearchCV(rf_model,rf_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
#rf_cv_model.best_params_

#Tuned Model
rf_model = RandomForestRegressor(random_state=42,
                                max_depth=5,
                                max_features=4,
                                min_samples_split=2,
                                n_estimators=500)
rf_tuned=rf_model.fit(X_train,y_train)


y_pred = rf_tuned.predict(X_test)
RMSE_tuned = np.sqrt(mean_squared_error(y_test,y_pred))
r2_tuned = r2_score(y_test,y_pred)


# Importances of Variables
Importance = pd.DataFrame({'Importance':rf_tuned.feature_importances_*100},
                         index=X_train.columns)

Importance.sort_values(by='Importance',
                      axis=0,
                      ascending = True).plot(kind = 'barh',
                                           color = 'b',)

plt.xlabel('Variable Importance')
plt.gca().legend_=None
