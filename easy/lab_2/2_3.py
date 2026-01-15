import numpy as np

def neural_network(inp, weights):
    #  Вход - Слой 1 - Слой 2 - Выход
    res = inp
    for w in weights:
        res = res.dot(w)
    return res

# Входные данные 
inp = np.array([23, 45])

# Слой 1: 2 входа - 3 нейрона (матрица 2x3)
# Слой 2: 3 входа - 4 нейрона (матрица 3x4)
# Слой 3: 4 входа - 2 выхода  (матрица 4x2)

# Генерируем случайные веса в виде матриц нужного размера
weights = [
    np.random.rand(2, 3), # Матрица весов для 1-го слоя
    np.random.rand(3, 4), # Матрица весов для 2-го слоя
    np.random.rand(4, 2)  # Матрица весов для выходного слоя
]

# Прогон через сеть
prediction = neural_network(inp, weights)

print("Веса первого слоя (фрагмент):\n", weights[0])
print(f"\nИтоговое предсказание: {prediction}")