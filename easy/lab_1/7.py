def network(inp: list, weights: list) -> list:
    # Инициализируем список под результаты каждого нейрона
    predictions = [0] * len(weights)
    
    for i in range(len(weights)):
        # Считаем скалярное произведение для текущего нейрона
        # (умножаем каждый вход на соответствующий ему вес и складываем)
        predictions[i] = sum([inp[j] * weights[i][j] for j in range(len(inp))])
        
    return predictions

# Исходные данные
data = [50, 165, 45]

# Наборы весов 
w_1 = [0.2, 0.1, 0.65]
w_2 = [0.3, 0.1, 0.7]
w_3 = [0.5, 0.4, 0.34]
w_4 = [0.4, 0.2, 0.1]

# Формируем матрицу весов слоя
layer_weights = [w_1, w_2, w_3, w_4]

# Считаем результат
results = network(data, layer_weights)

print("Результаты по каждому нейрону:")
for idx, res in enumerate(results, 1):
    print(f"Нейрон {idx}: {round(res, 2)}")
