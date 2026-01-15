import numpy as np

def neural_networks(inp, weights):
    return inp.dot(weights)

def get_error(target, pred):
    return np.sqrt(np.mean((target - pred) ** 2))

def train(inp, targets, w, lr, epochs):
    for i in range(epochs):
        err_log = 0
        delta = np.zeros_like(w)
        
        for j in range(len(inp)):
            pred = neural_networks(inp[j], w)
            err_log += get_error(targets[j], pred)
            # Градиент
            delta += (pred - targets[j]) * inp[j]
        
        w -= (delta / len(inp)) * lr
        
        if i % 1000 == 0:
            print(f"Эпоха {i} | Ошибка: {err_log/len(inp):.6f}")
    return w

def test_it(a, b, w):
    pred = a * w[0] + b * w[1]
    correct = a + b
    # Считаем реальный процент ошибки
    diff_percent = (abs(correct - pred) / abs(correct)) * 100 if correct != 0 else abs(pred)*100
    
    print(f"Вход: [{a}, {b}]")
    print(f"Нейросеть: {pred:.4f} | Истина: {correct}")
    print(f"Ошибка: {diff_percent:.4f}%")
    print("-" * 20)

# Данные для обучения
X = np.array([[10, 5], [0, -5], [2, 6]])
y = np.array([15, -5, 8])

# Стартуем с кривых весов
weights = np.array([0.1, 0.1])

# Параметры 
learning_rate = 0.01
epochs = 5000

# Учимся
weights = train(X, y, weights, learning_rate, epochs)

print("\nРезультаты тестов:")
test_it(12, 4, weights)
test_it(3, -8, weights)