import numpy as np

def train(inp, weight, target, lr, epochs):
    print(f"\n--- Тест: LR = {lr} ---")
    for i in range(1, epochs + 1):
        pred = inp * weight
        error = (target - pred) ** 2
        
        # Если ошибка стала ничтожно малой, останавливаемся
        if error < 1e-10:
            print(f"Идеал достигнут на {i} шаге!")
            break
            
        # Обновление веса
        delta = (pred - target) * inp
        weight -= delta * lr
        
        # Печатаем только значимые этапы
        if i % 5 == 0 or i == 1:
            print(f"Эпоха {i}: Прогноз {pred:.4f}, Ошибка {error:.8f}")

train(30, 0.2, 70, 0.001, 100)
