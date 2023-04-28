import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#read excel data from both excel sheets
Data = pd.read_excel('Solar_Data.xlsx', sheet_name = 'Day_1')
Data2 = pd.read_excel('Solar_Data.xlsx', sheet_name = 'Day_2')
df = pd.read_excel('Solar_Data.xlsx')

#extract collumns of data and store them as individual variables
time = Data.iloc[:, 0]
x = Data.iloc[:, 1:2]
y = Data.iloc[:, 2]
x2 = Data2.iloc[:, 1:2]
y2 = Data2.iloc[:, 2]

#perform polynomail regression
lin = LinearRegression()
lin.fit(x, y)
poly = PolynomialFeatures(degree = 3)
x_poly = poly.fit_transform(x)
poly.fit(x_poly, y)
lin2 = LinearRegression()
lin2.fit(x_poly, y)

#use polynomial regression to predict day 2 data
#display Actual and Predicted values for Solar PV vs. Irradiance
plt.scatter(x, y, label = 'Actual')
plt.plot(x, lin2.predict(poly.fit_transform(x)), label = 'Predicted', color = 'Red')
plt.title('Solar PV vs. Solar Irradiance')
plt.xlabel('Solar Irradiance')
plt.ylabel('Solar Power (kW)')
plt.legend(fontsize = 'x-small')
plt.show()

#display Actual and Predicted values for Solar PV vs. Time
plt.scatter(time, y, label = 'Actual Power')
plt.plot(time, lin2.predict(poly.fit_transform(x)), label = 'Predicted Power', color = 'Red')
plt.title('Solar PV vs. Time')
plt.xlabel('Time (sec)')
plt.ylabel('Solar Power (kW)')
plt.legend(fontsize = 'x-small')
plt.show()

#calculate error of predicted values and display using a histogram
ans = lin2.predict(poly.fit_transform(x))
error = ans - y2
plt.hist(error, bins = 30)
plt.title('Error')
plt.xlabel('Prediction - Actual')
plt.ylabel('Count')
plt.show()

#calculate total squared error of prediciton for 3rd degree poly regres
errorSq = error * error
print(errorSq)

#calculate 5th degree polynomial regression
lin = LinearRegression()
lin.fit(x, y)
poly = PolynomialFeatures(degree = 5)
x_poly = poly.fit_transform(x)
poly.fit(x_poly, y)
lin3 = LinearRegression()
lin3.fit(x_poly, y)

#calculate error of predicted values
ans = lin3.predict(poly.fit_transform(x))
error = ans - y2

#calculate total squared error of prediciton for 5th degree poly regres
errorSq = error * error
print(errorSq)