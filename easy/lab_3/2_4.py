import numpy as np

def neural_network(inp, weight):
    return inp * weight

def gradient_descent(inp, weight, target, epochs, tol=1e-12):
    """
    epochs: максимальное кол-во шагов
    tol: порог остановки
    """
    print(f"--- Старт обучения: Вход {inp}, Цель {target} ---")
    
    for i in range(1, epochs + 1):
        pred = neural_network(inp, weight)
        error = (pred - target) ** 2
        
        # Градиентный шаг
        delta = (pred - target) * inp
        weight -= delta
        
        # Печатаем только каждую 10-ю итерацию, чтобы не засорять консоль
        if i % 10 == 0 or i == 1:
            print(f"Эпоха {i}: Прогноз {pred:.6f} | Ошибка {error:.12f}")
            
        # Условие раннего выхода
        if error < tol:
            print(f"--- Цель достигнута на шаге {i}! Финальный вес: {weight:.5f} ---")
            break
    return weight

# Сценарий 1: Быстрая сходимость
gradient_descent(inp=0.9, weight=0.2, target=0.5, epochs=100)

print("\n" + "="*50 + "\n")

# Сценарий 2: "Ленивое" обучение
gradient_descent(inp=0.01, weight=0.1, target=0.9, epochs=1000)