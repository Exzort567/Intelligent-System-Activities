import random  as rd

class Perceptron:
    def __init__(self, size):
        self.weight = [rd.uniform(0, 1) for _ in range(size)]
        self.bias = 1

    def update(self, learning_rate, x, actual, predicted):
        new_weight = []
        for i in range(len(x)):
            new_weight.append(self.weight[i] + (learning_rate * (actual - predicted) * x[i]))
        self.weight = new_weight
        self.bias = self.bias + (learning_rate * (actual - predicted))

    def activation(self, sum):
        return 1 if sum > 0 else 0
    
    def predict(self, x):
        sum = 0
        for i in range(len(x)):
            sum = sum + (x[i] * self.weight[i])
        y = self.bias + sum

        return self.activation(y)
    def fit(self, learning_rate, x, y, epochs, threshold):
        epoch = 1
        while(epoch <= epochs):
            total_error = 0
            arr = list(range(len(x)))
            rd.shuffle(arr)

            for i in arr:
                predicted = self.predict(x[i])
                error = abs(y[i] - predicted)
                self.update(learning_rate, x[i], y[i], predicted)
                total_error += error
            total_error /= len(x)
            print(f"Epoch: {epoch} Error: {total_error}")

            if total_error <= threshold:
                print("Stop training due to reaching the threshold")
                break
            epoch += 1

x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
y = [0, 0, 0, 1]

learning_rate = 0.1
epochs = 100
threshold = 0

perceptron = Perceptron(len(x[0]))
perceptron.fit(learning_rate, x, y, epochs, threshold)
print("\nTesting the perceptron...")

for i in range(len(x)):
    print(f"Input: {x[i]} Actual: {y[i]} Predicted: {perceptron.predict(x[i])}")