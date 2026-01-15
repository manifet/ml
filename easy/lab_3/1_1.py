import numpy as np

def neural_network(inp, weights):
    # Скалярное произведение входа и весов
    return inp.dot(weights)

def calc_mse(true_val, pred_val):
    return (true_val - pred_val) ** 2

# Исходные данные
inp = np.array([150, 40])
weights = np.array([0.2, 0.3])
true_label = 50 

# Прогноз
pred = neural_network(inp, weights)

# Оценка
raw_error = true_label - pred      # На сколько ошиблись 
ms_error = calc_mse(true_label, pred)  # Квадрат ошибки

print(f"Прогноз сети: {pred}")
print(f"Чистая разница: {raw_error}")
print(f"Квадрат ошибки: {ms_error}")
