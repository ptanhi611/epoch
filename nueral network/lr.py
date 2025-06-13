import numpy as np
import matplotlib.pyplot as plt






class Linear_reg():
    def __init__(self,n_params):
        self.w = np.random.randn(n_params)
        self.b = np.random.randn()


    def gen_line(self,X,Y):
        self.X = X
        yi = (self.X @ self.w) + self.b
        return yi
    
    def Calculate_gradients(self,Y,yi):
        diff = Y - yi
        dw = (-2 * self.X.T @ diff)/len(yi)
        db = (-2 * np.sum(diff))/len(yi)
        mse = np.mean(np.power(diff,2))

        return dw,db,mse
    

    def Tune_params(self,dw,db,a):
        self.w = self.w - (dw * a)
        self.b = self.b - (db * a)



X = np.array([
    [ 1.7853, -0.1315, -4.3632,  3.3620],
    [ 5.0021,  6.0122,  1.7298, -7.3646],
    [-6.4107, -9.9771,  6.2256, -0.0641],
    [ 4.6116,  1.2884, -1.3315,  7.9862],
    [ 2.0414,  6.0406, -2.1630, -1.3806],
    [ 0.2311, -6.6966, -0.6241, -7.2561],
    [-5.6516,  6.5520, -8.9352,  8.3607],
    [-1.4887, -8.3801,  3.8379, -2.7662],
    [ 8.0074,  1.3328, -1.3450, -0.2086],
    [ 0.3680, -7.4078,  7.0765,  3.7490],
    [-4.7689,  7.1501,  5.0950,  8.3060],
    [ 4.9713,  6.3650, -3.7611,  5.7589],
    [ 2.8298,  3.4496,  1.9339, -3.9761],
    [ 9.2746, -2.3543, -4.8645, -5.6002],
    [ 3.7127, -7.6779,  3.2405, -6.8699],
    [-5.3731,  5.8349,  2.4702, -6.4880],
    [-7.0667,  0.9929, -9.7176,  4.4121],
    [-6.3466, -5.9808,  3.6635,  6.5296],
    [-9.2585, -5.3347, -7.5053,  5.2332],
    [ 5.5512,  6.5864, -1.2487,  2.9738],
    [ 0.7055,  0.1130, -2.5196,  8.9382],
    [ 7.1910, -3.9939, -4.2067, -9.0106],
    [-2.3818, -0.1284,  3.3076,  0.2984],
    [-3.4598, -6.8307,  7.2046,  4.3372],
    [ 8.1885, -6.5678, -3.4930, -2.6784],
    [ 6.5624, -3.3252, -3.4885, -6.5125],
    [ 3.6881,  7.3941, -6.1347,  1.2861],
    [-4.8675, -6.2172,  1.5877,  6.2941],
    [-5.9255,  5.6660,  2.2974,  4.3843],
    [ 2.6481, -5.3263,  0.8720,  3.0274],
    [-9.3673,  1.7110, -0.1901, -3.4716],
    [ 4.5356, -4.4049,  1.2977, -5.3062],
    [ 3.6949, -2.0879, -0.3207, -7.5654],
    [-0.5059, -3.0601,  6.2400, -1.8731],
    [ 4.9176,  3.4034, -0.4785, -7.1434],
    [-9.5714, -9.6723, -3.1544,  7.6770],
    [-3.6934,  6.9396, -2.5394,  4.0457],
    [ 3.0402,  4.2706, -4.3446, -6.6238],
    [-0.5835, -4.4353, -2.0951, -8.8463],
    [-4.5272, -1.4262,  2.0164, -1.5107]
])
X = (X-np.mean(X,axis=0))/np.std(X,axis=0)




Y = np.array([
     14.82,  10.25,   9.91,  23.12,   1.09,  25.01, -14.57,  28.19,  25.49,  37.21,
     -4.65,   5.95,  15.27,  46.64,  41.45, -19.31, -9.98,  -3.79, -15.67,  11.79,
      9.78,  41.69,  -0.94,  30.53,  48.28,  33.36,  -1.84,  -0.20,  23.78, -17.23,
     33.13,  30.33,  24.25,  14.33,  32.23, -4.18,   2.03,  -0.61,   2.70,   7.80
])

model = Linear_reg(4)
learning_rate = 0.001
epochs = 100000
MSE = []


#training loop
for epoch in range(epochs):

    y = model.gen_line(X,Y)
    dw,db,mse = model.Calculate_gradients(Y,y)
    MSE.append(mse)
    model.Tune_params(dw,db,learning_rate)
    print(f"Epoch {epoch+1}  Current RMSE = {np.sqrt(mse)}")

w = model.w
b = model.b

print(w,b)
    


# # Plot 1: Data points and final predicted line
# plt.figure(figsize=(12, 5))

# # Subplot 1: Predicted line vs actual data
# plt.subplot(1, 2, 1)
# plt.scatter(X, Y, color='blue', label='Actual Data')
# y_pred = [w * x + b for x in X]
# plt.plot(X, y_pred, color='red', label='Predicted Line')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Final Regression Line vs Data')
# plt.legend()

# Subplot 2: RMSE over epochs
plt.subplot(1, 2, 2)
RMSE = [np.sqrt(m) for m in MSE]
plt.plot(range(epochs), RMSE, color='green')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE Over Epochs')

plt.tight_layout()
plt.show()
