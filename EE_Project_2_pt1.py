import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

#read excel data: sheet one
Data = pd.read_excel('Solar_Data.xlsx', sheet_name = 'Day_1')
df = pd.read_excel('Solar_Data.xlsx')

#extract collumns of data by label
time = Data['Time']
x = Data['Irradiance']
y = Data['Kw']

#perform linear regression
slope, intercept, r, p, std_err = stats.linregress(x,y)

#display regression model parameters
print('Intercept: \n', intercept)
print('Coefficients: \n', slope)
print('y = (',slope,' * x ) +',intercept)

#define function based on regression parameters
def f(x):
    y = slope* x + intercept
    return y

#read excel data: sheet 2
Data2 = pd.read_excel('Solar_Data.xlsx', sheet_name = 'Day_2')
x2 = Data2['Irradiance']
y2 = Data2['Kw']
time2 = Data2['Time']

#predict y values for day 2 based on day 1 linear regression
predictedY = list(map(f, x2))

#display Actual and predicted values for Solar PV vs. Solar Irradiance
plt.scatter(x2, y2, label = 'Actual')
plt.scatter(x2, predictedY, label = 'Predicted')
plt.title('Solar PV vs. Solar Irradiance')
plt.xlabel('Solar Irradiance')
plt.ylabel('Solar Power (kW)')
plt.legend(fontsize = 'x-small')
plt.show()

#display Actual and predicted values for Solar PV vs.Time
plt.scatter(time, y2, label = 'Actual Power')
plt.scatter(time, predictedY, label = 'Predicted Power')
plt.title('Solar PV vs. Time')
plt.xlabel('Time (sec)')
plt.ylabel('Solar Power (kW)')
plt.legend(fontsize = 'x-small')
plt.show()

#calculate error in predicted values and display using historgram
error = predictedY - y2
plt.hist(error, bins = 50)
plt.title('Error')
plt.xlabel('Prediction - Actual')
plt.ylabel('Count')
plt.show()