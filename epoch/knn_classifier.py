import numpy as np 

def Euclidian_distance(X,Y):
    return np.sqrt(np.sum((X-Y)**2))

data = [
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
]

test_data = np.array([
    [118, 6.2, 0],  # Expected: Banana
    [160, 7.3, 1],  # Expected: Apple
    [185, 7.7, 2]   # Expected: Orange
])


labels = {"Apple":0,"Banana":1,"Orange":2}
rev_labels = {v:k for k,v in labels.items()}

X = np.array([row[:3] for row in data])
Y = np.array([labels.get(row[3]) for row in data])

true_labels = ["Banana", "Apple", "Orange"]

class KNN():
    def __init__(self,k):
        self.k=k
    
    def fit(self,X,y):
        self.X=X
        self.y=y

    def predict_one (self,x):
        distances=[Euclidian_distance(x,y) for y in self.X]
        K_indices = np.argsort(distances)[:self.k]
        y_labels = [self.y[i] for i in K_indices]


        unique_labels = np.unique(self.y)
        label_counts = {label: 0 for label in unique_labels}

        for label in y_labels:
            label_counts[label] += 1

        
        most_frequent_label = max(label_counts, key=label_counts.get)
        return most_frequent_label
        
    
    def predict(self,X_test):
        return [self.predict_one (x) for x in X_test]
    
knn = KNN(3)
knn.fit(X,Y)

predictions = knn.predict(test_data)

predicted_names = [rev_labels[pred] for pred in predictions]

print("True Labels     :", true_labels)
print("Predicted Labels:", predicted_names)   