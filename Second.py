import random

class Perceptron:
    def __init__(self, input_size=4, max_epochs=100):
        self.max_epochs = max_epochs
        self.input_size = input_size

        self.weights = [random.uniform(-1, 1) for _ in range(self.input_size + 1)]

    def activation(self, net):
        return 1 if net >= 0 else 0

    def predict(self, x):
        x = [1] + x
        net = sum(w * xi for w, xi in zip(self.weights, x))
        return self.activation(net)

    def train(self, X, D):
        X = [[1] + x for x in X]  

        for epoch in range(self.max_epochs):
            total_error = 0
            errors = 0

            for xm, d in zip(X, D):
                net = sum(w * xi for w, xi in zip(self.weights, xm))
                y = self.activation(net)
                error = d - y



                if y != d:
                    errors += 1
                    for i in range(len(self.weights)):
                        if error == 1: 
                            self.weights[i] += xm[i]
                        elif error == -1:
                            self.weights[i] -= xm[i]


            print(f"Эпоха {epoch + 1}: ошибок {errors}, веса {self.weights}")
            if errors == 0:
                print(f"Обучение завершено на эпохе {epoch + 1}")
                break

    def print_weights(self):
        print("Обученные веса:", self.weights)

    def generate_data(self, seed=10, n_samples=1000000):
        random.seed(seed)

        true_weights = [1] + [random.uniform(-1, 1) for _ in range(self.input_size)]
        X = []
        Y = []

        for _ in range(n_samples):
            inputs = [random.uniform(0, 1) for _ in range(self.input_size)]
            z = sum(w * x for w, x in zip(true_weights, [1] + inputs))
            target = 1 if z >= 0 else 0
            X.append(inputs)
            Y.append(target)
        return X, Y

if __name__ == "__main__":
    perceptron = Perceptron(input_size=8, max_epochs=5000)
    X, D = perceptron.generate_data()
    perceptron.train(X, D)
    perceptron.print_weights()

    print("\nПроверка:")
    X, D = perceptron.generate_data(25, 200)
    correct = 0
    for i in range(len(X)):
        y = perceptron.predict(X[i])
        if y == D[i]:
            correct += 1
    print(f"Точность на тестовых данных: {correct / len(X) * 100}%")
