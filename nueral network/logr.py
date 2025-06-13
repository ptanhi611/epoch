import numpy as np
import matplotlib.pyplot as plt


def sigmoid(self,x):
    return (np.exp(x)/(1+np.exp(x)))



class Logistic_reg():
   
    

    def __init__(self):
        self.w = np.random.randn()
        self.b = np.random.randn()


    def gen_line(self,X,Y):
        self.Y = Y
        self.X = X
        yi = [(self.w*x + self.b) for x in X]
        return yi
    
    def Calculate_gradients(self,yi):
        sigma = [sigmoid(y) for y in (yi)]
        dw = (-1* sum([x*s*(1-s) for x,s in zip(self.X,sigma)]))/len(yi)
        db = (-1* sum([s*(1-s) for x,s in zip(self.X,sigma)]))/len(yi)

        cross_entropy_loss = (-1* sum([np.log(Yi)*Y + (1-Y)*np.log(1-Yi)] for Y,Yi in zip(self.Y,yi)))/len(yi)
        

        return dw,db,cross_entropy_loss
    

    def Tune_params(self,dw,db,a):
        self.w -= (dw * a)
        self.b -= (db * a)


# X = []
# Y = []



# model = Logistic_reg()
# epochs = 200
# Loss = []


# #training loop
# for epoch in range(epochs):

#     y = model.gen_line(X,Y)
#     dw,db,cross_entropy_loss = model.Calculate_gradients(y)
#     Loss.append(cross_entropy_loss)
#     model.Tune_params(dw,db,a=0.01)
#     print(f"Epoch {epoch}  Current LOSS = {np.sqrt(cross_entropy_loss)}")

# w = model.w
# b = model.b

# print(w,b)
    

# import numpy as np
# import matplotlib.pyplot as plt

# # ✅ sigmoid function as standalone
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# class Logistic_reg():
#     def __init__(self):
#         self.w = np.random.randn()
#         self.b = np.random.randn()

#     def gen_line(self, X, Y):
#         self.Y = Y
#         self.X = X
#         yi = [(self.w * x + self.b) for x in X]
#         return yi

#     def Calculate_gradients(self, yi):
#         sigma = [sigmoid(y) for y in yi]

#         # Gradients based on cross-entropy loss
#         dw = -1 * sum([(y_true - y_pred) * x for x, y_true, y_pred in zip(self.X, self.Y, sigma)]) / len(yi)
#         db = -1 * sum([(y_true - y_pred) for y_true, y_pred in zip(self.Y, sigma)]) / len(yi)

#         # Cross-entropy loss
#         eps = 1e-15  # To avoid log(0)
#         loss = -1 * sum([y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
#                          for y_true, y_pred in zip(self.Y, sigma)]) / len(yi)

#         return dw, db, loss

#     def Tune_params(self, dw, db, a):
#         self.w -= (dw * a)
#         self.b -= (db * a)

# ✅ Define a simple dataset (X: input, Y: label 0 or 1)
X = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 6.5])
Y = np.array([0,   0,   0,   0,   1,   1,   1,   1,   1,   1])

model = Logistic_reg()
epochs = 10000
Loss = []

# ✅ Training loop
for epoch in range(epochs):
    y = model.gen_line(X, Y)
    dw, db, cross_entropy_loss = model.Calculate_gradients(y)
    Loss.append(cross_entropy_loss)
    model.Tune_params(dw, db, a=0.1)
    print(f"Epoch {epoch:3} | Loss = {cross_entropy_loss:.4f}")

# Final weights
w = model.w
b = model.b
print(f"\nFinal Weights: w = {w:.4f}, b = {b:.4f}")

# ✅ Plotting decision boundary
x_vals = np.linspace(min(X) - 1, max(X) + 1, 100)
y_vals = [sigmoid(w * x + b) for x in x_vals]

plt.figure(figsize=(10, 5))

# Plot sigmoid curve
plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, color='blue', label='Sigmoid Output')
plt.scatter(X, Y, color='red', label='Data Points')
plt.title("Logistic Regression - Sigmoid Fit")
plt.xlabel("X")
plt.ylabel("Predicted Probability")
plt.legend()

# Plot loss curve
plt.subplot(1, 2, 2)
plt.plot(Loss, color='green')
plt.title("Cross Entropy Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()
