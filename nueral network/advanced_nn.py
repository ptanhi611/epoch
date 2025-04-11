import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class Neural_network(nn.Module):
    def __init__(self,input_size):
        super(Neural_network,self).__init__()
        self.l1 = nn.Linear(input_size,128)
        self.act1 = nn.ReLU()
        self.d1 = nn.Dropout(0.3)
        self.l2 = nn.Linear(128,64)
        self.act2 = nn.ReLU()
        self.d2 = nn.Dropout(0.3)
        self.l3 = nn.Linear(64,32)
        self.act3 = nn.ReLU()
        self.d3 = nn.Dropout(0.3)
        self.l4 = nn.Linear(32,1)

    def forward(self,x):
        x = self.d1(self.act1(self.l1(x)))
        x = self.d2(self.act2(self.l2(x)))
        x = self.d3(self.act3(self.l3(x)))
        x = self.l4(x)
        return x
    
# class LSTM():
    



data = pd.read_excel("house_data.xlsx")
X=data.drop(columns="Price")
Y=data["Price"]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
Y_scaled=scaler.fit_transform(Y.values.reshape(-1,1)).flatten()


X_train,X_test,Y_train,Y_test = train_test_split(X_scaled,Y_scaled,test_size=0.2,random_state=40)


X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1,1)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1,1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= Neural_network(X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)



X_train_tensor  = X_train_tensor.to(device)
Y_train_tensor = Y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
Y_test_tensor = Y_test_tensor.to(device)



batch_size=32
epochs=2000



for epoch in range(epochs):
    permutation = torch.randperm(X_train_tensor.size()[0])
    total_loss = 0

    for i in range(0,X_train_tensor.size()[0],batch_size):
        indices=permutation[i: i+batch_size]
        X_batch = X_train_tensor[indices]
        Y_batch = Y_train_tensor[indices]

        
        optimizer.zero_grad()
        output=model(X_batch)
        loss=criterion(output,Y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss/(len(range(0, X_train_tensor.size(0), batch_size)))

    if (epoch+1)%50==0:
        print(f"Epoch: {(epoch+1)}/{epochs}, Loss: {avg_loss: .6f}")


with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy()
    real_preds = scaler.inverse_transform(predictions)
    real_labels = scaler.inverse_transform(Y_test_tensor.cpu().numpy())

    mse = np.mean((real_preds - real_labels)**2)
    rmse = np.sqrt(mse)
    print(f"\nTest MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

