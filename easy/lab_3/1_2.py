import numpy as np

def neural_network(inp, weights):
    # Прогноз: сумма произведений входов на веса
    return inp.dot(weights)

def calc_error(target, prediction):
    return (target - prediction) ** 2

# Данные
target = 50
inputs = np.array([150, 40])

weights = np.array([0.253333, 0.3])

# Прогон
prediction = neural_network(inputs, weights)
error = calc_error(target, prediction)

print(f"Прогноз: {prediction:.6f}")
print(f"Ошибка:  {error:.10f}")

# Проверка условия
if error < 0.001:
    print("Результат достигнут!")