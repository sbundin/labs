#  Алгоритм обучения методом положительного и отрицательного подкрепления

## Шаг 1. Инициализация весов случайными малыми значениями

```
self.weights = [random.uniform(-1, 1) for _ in range(self.input_size + 1)]
```

## Шаг 2. Подача входного вектора

```
X = [[1] + x for x in X]  
```

## Шаг 3. Вычисление взвешенной суммы и применение пороговой функции активации

```
net = sum(w * xi for w, xi in zip(self.weights, xm))
y = self.activation(net)
```

## Шаг 4. Проверка ошибки и корректировка весов

```
error = d - y

if y != d:
    errors += 1
    for i in range(len(self.weights)):
        if error == 1: 
            self.weights[i] += xm[i]
        elif error == -1:
            self.weights[i] -= xm[i]
```

## Шаг 5. Завершение обучения при отсутствии ошибок

if errors == 0:
    print(f"Обучение завершено на эпохе {epoch + 1}")
    break

