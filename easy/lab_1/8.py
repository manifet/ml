def network(inp: list, weights: list) -> list:
    """Считает выходы для каждого нейрона в слое."""
    predictions = []
    for w_set in weights:
        # Считаем скалярное произведение через генератор
        dot_product = sum(i * w for i, w in zip(inp, w_set))
        predictions.append(round(dot_product, 2))
    return predictions

# Входные данные
inp = [50, 165, 45]

# Чтобы результаты совпали (50*0.3 + 165*0.1 + 45*0.7 = 63.0),
# наборы весов должны давать одинаковую сумму. 
# Самый надежный способ — сделать их идентичными.
w1 = [0.3, 0.1, 0.7]
w2 = [0.3, 0.1, 0.7]

weights = [w1, w2]
result = network(inp, weights)

print(f"Выходы нейронов: {result}")
print(f"Веса: {weights}")

if result[0] == result[1]:
    print("Выходные данные идентичны.")