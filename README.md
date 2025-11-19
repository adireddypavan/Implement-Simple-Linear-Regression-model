# Implement-Simple-Linear-Regression-model
python program to implement  Simple Linear Regression model for given dataset

 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

data = {
    'Hours_studied':[1,2,3,4,5,6,7,8,9,10],
    'Marks_obtained':[10,20,28,35,45,50,60,70,75,85]
    }

df = pd.DataFrame(data)
print("Dataset: \n",df)

x = df[['Hours_studied']]
y = df['Marks_obtained']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3 , random_state = 1)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("\n Intercept(b0):",model.intercept_)
print("coefficient(b1):",model.coef_)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("\n Mean Square Error:",mse)
print("R-Squared value:",r2)

result = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

print("\n Actual VS Predicted Result: \n",result)

plt.scatter(x,y,color='blue',label='Actual Data')
plt.plot(x,model.predict(x),color='red',linewidth=2,label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Obtained')
plt.title('Simple Linear Regression Example')
plt.legend()
plt.show()
