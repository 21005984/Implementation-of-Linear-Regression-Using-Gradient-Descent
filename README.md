# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Meiyarasi.V
RegisterNumber:  212221230058
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    """"
    Take in a numpy array X,y,theta and generate the cost function of using theta as a parameter in a linera regression tool   
    """
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    """"
    Take in numpy array X,y and theta and update theta by taking num_iters gradient steps with learning rate of alpha 
    return theta and the list of the cost of the theta during each iteration
    """
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    """"
    Takes in numpy array of x and theta and return the predicted valude of y based on theta
    """
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))
*/
```

## Output:
![Screenshot 2022-09-28 134057](https://user-images.githubusercontent.com/94748389/192728540-c4c4c597-de82-4036-ab0f-747daac6d152.png)
![Screenshot 2022-09-28 134548](https://user-images.githubusercontent.com/94748389/192728629-0cbc1b25-cbf1-4de6-97ac-5f9a95a8fac9.png)
![Screenshot 2022-09-28 134605](https://user-images.githubusercontent.com/94748389/192728657-94a74a18-116a-4489-88a0-4841b6f3d0f2.png)
![Screenshot 2022-09-28 134131](https://user-images.githubusercontent.com/94748389/192728576-e1fa2ece-ee8b-405f-86d2-72b5677651e3.png)
![Screenshot 2022-09-28 134147](https://user-images.githubusercontent.com/94748389/192728608-3c54c16b-69e8-4e4e-b8bc-a5ce51884e7b.png)
![Screenshot 2022-09-28 134623](https://user-images.githubusercontent.com/94748389/192728710-5fbe1613-b9b6-4632-884e-2513ca7cd44d.png)
![Screenshot 2022-09-28 134638](https://user-images.githubusercontent.com/94748389/192728740-fd897a3b-ba4b-4586-ad7d-1a4d91494eee.png)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
