# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step1:
Import the needed packages.

### Step2:
Read the txt file using read_csv.

### Step3:
Use numpy to find theta,x,y values.

### Step4:
To visualize the data use plt.plot.
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('/content/ex1.txt', header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta)
def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))
  return theta,J_history
theta,J_history=gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color='r')
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))
predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![Screenshot 2023-09-20 194055](https://github.com/s-adhithya/Implementation-of-Linear-Regres![Screenshot 2023-09-20 194305](https://github.com/s-adhithya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497423/cea5f224-490f-4ffa-92bc-3ef173aacdbb)
sion-Using-Gradient-Descent/assets/113497423/f78d7838-4c67-4982-887f-653f77214ca6)
![Screenshot 2023-09-20 194323](https://github.com/s-adhithya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497423/11aee399-615e-48f3-afa7-b9a1b700bc83)
![Screenshot 2023-09-20 194343](https://github.com/s-adhithya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497423/45cd7318-9b37-46b8-ad8b-fb5aa01d9544)
![Screenshot 2023-09-20 194402](https://github.com/s-adhithya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497423/e9651843-81ef-4602-8f1a-c008e23beac9)
![Screenshot 2023-09-20 194415](https://github.com/s-adhithya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497423/c55306fc-4403-4ad7-9284-0c8324c8932c)
![Screenshot 2023-09-20 194424](https://github.com/s-adhithya/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497423/8dab1673-112e-4fd5-92ef-e4bb5d3ecacb)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
