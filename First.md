# Алгоритм обучения методом градиентного спуска

## Шаг 1. Подготоваить обучающую выборку

В коде за это отвечает функция generate_data класса Perceptron

```
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
```
где 

```
true_weights = [1] + [random.uniform(-1, 1) for _ in range(self.input_size)]
```
Инициализация значений истинных весов (они нужны только для генерации данных)

Далее в цикле происходит сначала генерация входных значений 

```
inputs = [random.uniform(0, 1) for _ in range(self.input_size)]
```

а затем расчитываются выходные значения

```
z = sum(w * x for w, x in zip(true_weights,[1] + inputs))
```

## Шаг 2. Генератором случайных чисел всем синаптическим весам wi,j и нейронным смещениям w0,j (i=0,…,n; j=1,…,k) присваиваются некоторые малые случайные значения

в init функции класса Perceptron

```
self.weights = [random.uniform(-1, 1) for _ in range(self.input_size + 1)]
```

## Шаги 3-8

Здесь у нас начинается обучение по эпохам

В начале эпохи суммарную ошибку приравниваем к нулю 

```
total_error = 0
```

Далее обрабатываем все входные и выходные значения спомощью цикла

```
for xm, d in zip(X, D):
```

Рассчитываем выходное значение 

```
net = sum(w * xi for w, xi in zip(self.weights, xm))
y = self.activation(net)
```

где activation

```
def activation(self, net):
    return 1 if net >= 0 else 0

```

Считаем ошибку 

```
error = d - y
total_error += error ** 2
```

Считаем величину корреции весов

```
for i in range(len(self.weights)):
    delta_w[i] += self.learning_rate * error * xm[i]
```

Проверяем критерий останова 

```
if total_error <= self.error_threshold * len(X):
    print(f"Обучение завершено на эпохе {epoch + 1}")
    print(f"Эпоха {epoch + 1}: ошибка {total_error}, веса {self.weights}")
    break
```

Обновляем веса 

```
for i in range(len(self.weights)):
    self.weights[i] += delta_w[i] / len(X)
```

## Проверка работоспособности

Создаем и обучаем модель на тестовых данных

```
perceptron = Perceptron(input_size=10, learning_rate=0.1, max_epochs=5000)
    X, D = perceptron.generate_data()
    perceptron.train(X, D)
```

Генерируем проверочные данные и оцениваем точность модели

```
X, D = perceptron.generate_data(25, 100000)
correct = 0
for i in range(len(X)):
    y = perceptron.predict(X[i])
    if y == D[i]:
        correct += 1
print(f"Точность на тестовых данных: {correct / len(X) * 100}%")
```