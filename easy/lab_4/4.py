import numpy as np

def neural_network(inp, weights):
    # Прямой проход: входной вектор (1x2) умножаем на матрицу весов (2x2)
    # Результат: выходной вектор (1x2)
    return inp.dot(weights)

def train(inp, weights, targets, lr, epochs):
    print(f"Старт обучения: Вход {inp} -> Цели {targets}")
    
    for i in range(1, epochs + 1):
        # 1. Предикшн
        pred = neural_network(inp, weights)
        
        # 2. Ошибка
        error = (pred - targets) ** 2
        
        # 3. Расчет градиента 
        delta = np.outer(inp, (pred - targets)) * lr
        
        # 4. Обновление весов
        weights -= delta
        
        if i % 10 == 0:
            print(f"Эпоха {i}: Предсказание {pred.round(2)}, Ошибка {error.mean().round(4)}")

# Вход: 
inp_data = np.array([150, 40])
# Цель: 
true_target = np.array([70, 110])
# Инициализация весов (2 входа - 2 выхода = матрица 2x2)
weights = np.random.rand(2, 2)

train(inp_data, weights, true_target, 0.00001, 100)