import numpy as np

def train_multi_output(inp, weights, targets, lr, epochs):
    print(f"Старт: Вход={inp}, Цели={targets}")
    
    for i in range(1, epochs + 1):
        preds = inp * weights
        
        # Ошибки по каждому выходу
        errors = (preds - targets) ** 2
        
        # Обновление весов
        delta = (preds - targets) * inp * lr
        weights -= delta
        
        if i % 5 == 0 or i == 1:
            print(f"Эпоха {i}: Прогнозы {preds}, Ошибки {errors}")
            
        # Условие остановки: средняя ошибка по всем выходам меньше порога
        if np.mean(errors) < 1e-10:
            print(f">>> Обучение завершено на шаге {i}!")
            break
            
    return weights

inp_val = 200
initial_weights = np.array([0.2, 0.3])
target_vals = np.array([50, 120])

final_w = train_multi_output(inp_val, initial_weights, target_vals, 0.00001, 50)