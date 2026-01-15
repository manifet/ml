import numpy as np

def train_multi_input(inp, weights, target, lr, max_iters):
    print(f"Старт обучения (LR={lr}):")
    
    for i in range(1, max_iters + 1):
        # 1. Прогноз
        pred = inp.dot(weights)
        
        # 2. Ошибка
        error = (pred - target) ** 2
        
        # 3. Вычисление дельты для всех весов сразу
        # delta - это вектор того же размера, что и weights
        delta = (pred - target) * inp * lr
        
        # Блокировка обучения первого нейрона
        delta[0] = 0 
        
        # 4. Обновление весов
        weights -= delta
        
        # Печать прогресса каждые 100 шагов
        if i % 100 == 0 or i == 1:
            print(f"Шаг {i}: Прогноз {pred:.6f}, Ошибка {error:.10f}")
            
        # Условие ранней остановки 
        if error < 1e-12:
            print(f">>> Обучение завершено на шаге {i}!")
            break
            
    return weights

# Тестирование
inp_data = np.array([150, 40])
initial_weights = np.array([0.2, 0.3])
target_val = 1.0

final_w = train_multi_input(inp_data, initial_weights, target_val, 0.0001, 1000)
print(f"Финальные веса: {final_w}")