import numpy as np

# Функции активации
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    return x * (1 - x)

def softmax(x):
    # Стабильный Softmax: вычитаем max для защиты от переполнения exp
    shift_x = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shift_x)
    return exps / np.sum(exps, axis=1, keepdims=True)

# Задания 2 и 3
# Входы: все комбинации 4-битного кода (0-15)
x_train = np.array([[int(b) for b in format(i, '04b')] for i in range(16)])
# Выходы: матрица 16x16
y_train = np.eye(16)

# Задание 4
input_dim = 4
hidden_dim = 32  # Увеличили слой для стабильности
output_dim = 16
lr = 0.05        # Умеренная скорость для плавной сходимости
epochs = 15000

# Инициализация весов 
np.random.seed(42)
w0 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1/input_dim)
w1 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1/hidden_dim)

print(f"Запуск обучения на {epochs} эпох...")

# Цикл обучения
for epoch in range(epochs):
    # Forward pass
    l1 = sigmoid(np.dot(x_train, w0))
    l2 = softmax(np.dot(l1, w1))
    
    # Расчет ошибки 
    if epoch % 3000 == 0:
        loss = -np.mean(np.sum(y_train * np.log(l2 + 1e-10), axis=1))
        print(f"Эпоха {epoch:5} | Loss: {loss:.6f}")

    # Backward pass
    # Градиент Softmax + Cross-Entropy упрощается до (pred - true)
    l2_delta = (l2 - y_train) 
    l1_delta = l2_delta.dot(w1.T) * sigmoid_deriv(l1)
    
    # Обновление весов
    w1 -= lr * l1.T.dot(l2_delta)
    w0 -= lr * x_train.T.dot(l1_delta)

# Проверка результатов
print("\n Финальная проверка (0-15)")
correct = 0
for i in range(16):
    raw_in = x_train[i:i+1]
    # Предсказание
    hid = sigmoid(np.dot(raw_in, w0))
    out = softmax(np.dot(hid, w1))
    
    pred = np.argmax(out)
    prob = np.max(out)
    
    status = "✓" if pred == i else "✗"
    print(f"Вход {raw_in[0]} | Ожидалось: {i:2} | Предсказано: {pred:2} | Уверенность: {prob:.2%}")
    if pred == i: correct += 1

print(f"\nИтоговая точность: {correct/16:.1%}")