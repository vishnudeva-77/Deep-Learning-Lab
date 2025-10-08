import numpy as np 
import matplotlib.pyplot as plt
import sys

x = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
y = np.array([[1], [1], [0], [0]])

num_i_units = 2
num_h_units = 2
num_o_units = 1

learning_rate = 0.01
reg_param = 0
max_iter = 5000
m = 4 

np.random.seed(1)
w1 = np.random.normal(0, 1, (num_h_units, num_i_units))
w2 = np.random.normal(0, 1, (num_o_units, num_h_units))
b1 = np.random.random((num_h_units, 1))
b2 = np.random.random((num_o_units, 1))

def sigmoid(z, derv=False):
    if derv:
        return z * (1 - z) 
    return 1 / (1 + np.exp(-z))

cost = np.zeros((max_iter, 1))

def train(_w1, _w2, _b1, _b2):
    for i in range(max_iter):
        c = 0
        dw1 = np.zeros_like(_w1)
        dw2 = np.zeros_like(_w2)
        db1 = np.zeros_like(_b1)
        db2 = np.zeros_like(_b2)

        for j in range(m):
            sys.stdout.write(f"\r Iteration: {i+1} Sample: {j+1}")
            sys.stdout.flush()

            a0 = x[j].reshape(-1, 1)           
            yj = y[j].reshape(-1, 1)           

            z1 = _w1 @ a0 + _b1              
            a1 = sigmoid(z1)                   

            z2 = _w2 @ a1 + _b2              
            a2 = sigmoid(z2)                  

            dz2 = a2 - yj                    
            dw2 += dz2 @ a1.T             
            db2 += dz2                         

            dz1 = (_w2.T @ dz2) * sigmoid(a1, derv=True)
            dw1 += dz1 @ a0.T                 
            db1 += dz1                         

            c += - (y[j] * np.log(a2) + (1 - yj) * np.log(1 - a2))

        _w1 -= learning_rate * (dw1 / m) + (reg_param / m) * _w1
        _w2 -= learning_rate * (dw2 / m) + (reg_param / m) * _w2
        _b1 -= learning_rate * (db1 / m)
        _b2 -= learning_rate * (db2 / m)

        cost[i] = c / m + (reg_param / m) * (np.sum(_w1 ** 2) + np.sum(_w2 ** 2))

    return _w1, _w2, _b1, _b2

w1, w2, b1, b2 = train(w1, w2, b1, b2)

plt.plot(range(max_iter), cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Training Loss")
plt.show()

print("\nPredictions after training:")
for i in range(4):
    a0 = x[i].reshape(-1, 1)
    z1 = w1 @ a0 + b1
    a1 = sigmoid(z1)
    z2 = w2 @ a1 + b2
    a2 = sigmoid(z2)
    print(f"Input: {x[i]} -> Predicted: {a2[0][0]:.4f} | Expected: {y[i][0]}")
