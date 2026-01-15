import numpy as np

def neural_network(inp, weights):
    return inp.dot(weights)

def calc_error(target, pred):
    return (target - pred) ** 2

# Данные
true_target = 50
inputs = np.array([150, 40])
weights = np.array([0.2, 0.3])

# Параметры поиска
step = 0.0001
limit = 0.001
iterations = 0

# Цикл подгонки веса
while True:
    prediction = neural_network(inputs, weights)
    error = calc_error(true_target, prediction)
    
    if error <= limit:
        break
        
    # Если прогноз меньше цели — прибавляем, если больше — убавляем
    if prediction < true_target:
        weights[0] += step
    else:
        weights[0] -= step
        
    iterations += 1

print(f"Результат достигнут за {iterations} итераций")
print(f"Финальные веса: {weights}")
print(f"Итоговая ошибка: {error:.6f}")