import numpy as np

def neural_network(inp, weight):
    return inp * weight

def get_error(target, pred):
    return (target - pred) ** 2

def train(inp, weight, target, iterations=10):
    print(f"--- Обучение: Цель {target}, Вход {inp} ---")
    
    for i in range(iterations):
        pred = neural_network(inp, weight)
        error = get_error(target, pred)
        
        # Находим delta 
        # Чем дальше target от pred, тем больше будет этот шаг
        delta = (pred - target) * inp
        weight -= delta
        
        print(f"Шаг {i+1}: Прогноз: {pred:.6f} | Ошибка: {error:.10f}")

# Сравним два сценария
train(inp=0.9, weight=0.2, target=0.18) # Цель близко к начальному прогнозу (0.18)
print()
train(inp=0.9, weight=0.2, target=0.8)  # Цель далеко от начального прогноза (0.18)