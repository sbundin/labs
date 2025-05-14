import random

class Perceptron:
    def __init__(self, input_size=4, learning_rate=0.1, max_epochs=100, error_threshold=0.00001):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.error_threshold = error_threshold
        self.input_size = input_size
        self.weights = [random.uniform(-1, 1) for _ in range(self.input_size + 1)]

    def activation(self, net):
        return 1 if net >= 0 else 0

    def predict(self, x):
        x = [1] + x  # Добавляем x0 = 1
        net = sum(w * xi for w, xi in zip(self.weights, x))
        return self.activation(net)

    def train(self, X, D):
        X = [[1] + x for x in X]

       

        for epoch in range(self.max_epochs):
            total_error = 0

            delta_w = [0.0] * len(self.weights) 

            for xm, d in zip(X, D):
                net = sum(w * xi for w, xi in zip(self.weights, xm))
                y = self.activation(net)
                error = d - y
                total_error += error ** 2

                for i in range(len(self.weights)):
                    delta_w[i] += self.learning_rate * error * xm[i]

            

            if total_error <= self.error_threshold * len(X): # Не забыть подгонять error_threshold под размер данных
                print(f"Обучение завершено на эпохе {epoch + 1}")
                print(f"Эпоха {epoch + 1}: ошибка {total_error}, веса {self.weights}")
                break

            for i in range(len(self.weights)):
                self.weights[i] += delta_w[i] / len(X)

            print(f"Эпоха {epoch + 1}: ошибка {total_error}, веса {self.weights}")

    def generate_data(self, seed=10, n_samples=1000000):
        random.seed(seed)

        true_weights = [1] + [random.uniform(-1, 1) for _ in range(self.input_size)]

        X = []
        Y = []

        for _ in range(n_samples):
            inputs = [random.uniform(0, 1) for _ in range(self.input_size)]
            
            z = sum(w * x for w, x in zip(true_weights,[1] + inputs))
            
            target = 1 if z >= 0 else 0
            
            # Сохраняем пример
            X.append(inputs)
            Y.append(target)
        print("Данные готовы")
        return X, Y

if __name__ == "__main__":
    perceptron = Perceptron(input_size=10, learning_rate=0.1, max_epochs=5000)
    X, D = perceptron.generate_data()
    perceptron.train(X, D)
    print("Обученные веса:", perceptron.weights)

    # Проверка
    print("\nПроверка:")
    X, D = perceptron.generate_data(25, 100000)
    correct = 0
    for i in range(len(X)):
        y = perceptron.predict(X[i])
        if y == D[i]:
            correct += 1
    print(f"Точность на тестовых данных: {correct / len(X) * 100}%")
