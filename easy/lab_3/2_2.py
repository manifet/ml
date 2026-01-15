import numpy as np

def neural_network(inp, weight):
    return inp * weight

def gradient_descent(inp, weight, target, iterations=10, alpha=0.1):
    print(f"Start: Inp={inp}, Weight={weight}, Target={target}, Alpha={alpha}")
    
    for i in range(iterations):
        pred = neural_network(inp, weight)
        error = (pred - target) ** 2
        
        # Умножаем на alpha, чтобы контролировать скорость спуска
        direction_and_amount = (pred - target) * inp
        weight -= direction_and_amount * alpha
        
        print(f"It {i+1}: Pred: {pred:.5f} | Error: {error:.10f} | New Weight: {weight:.5f}")

target = 0.8

# Пример 1: Стабильное обучение 
print("\n--- Сценарий 1: Большой вход, умеренный Alpha ---")
gradient_descent(inp=2.0, weight=0.5, target=target, alpha=0.1)

# Пример 2: Проблема затухания 
print("\n--- Сценарий 2: Крошечный вход (затухание градиента) ---")
gradient_descent(inp=0.0001, weight=0.5, target=target, alpha=0.1)

# Пример 3: Взрыв градиента 
print("\n--- Сценарий 3: Риск взрыва (высокий Alpha и большой вход) ---")
gradient_descent(inp=10.0, weight=0.5, target=target, alpha=1.0)