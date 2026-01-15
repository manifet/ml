def neural_network(inp: float, weights: list) -> list:
    # Возвращаем список выходов
    return [inp * w for w in weights]

# Исходные данные
weights = [0.001, 0.5]
inp = 4
STEP = 0.01 
LIMIT = 0.5

# Считаем стартовый прогноз
predict = neural_network(inp, weights)
counter = 0

# Цикл крутится, пока хотя бы один выход меньше лимита
while any(p <= LIMIT for p in predict):
    # Обновляем каждый вес только если его прогноз еще не дополз до 0.5
    for i in range(len(weights)):
        if predict[i] <= LIMIT:
            weights[i] += STEP
            
    # Пересчитываем результаты
    predict = neural_network(inp, weights)
    counter += 1

print(f"Потребовалось шагов: {counter}")
print(f"Финальные веса: {[round(w, 4) for w in weights]}")
print(f"Выходные значения: {[round(p, 4) for p in predict]}")