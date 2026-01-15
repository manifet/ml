def neural_network(inp: float, weight: float, bias: float) -> float:
    # Теперь учитываем не только вес, но и смещение
    return inp * weight + bias

# Фиксируем вход и вес, меняем только смещение
val, w = 150, 0.3

# Проверка разных вариантов bias
res_1 = neural_network(val, w, 0)   # Базовый вариант
res_2 = neural_network(val, w, 10)  # Сдвиг вверх
res_3 = neural_network(val, w, -5)  # Сдвиг вниз
res_4 = neural_network(val, w, 20)  # Сильный положительный сдвиг

print(f"Bias 0:  {res_1}")
print(f"Bias 10: {res_2}")
print(f"Bias -5: {res_3}")
print(f"Bias 20: {res_4}")
