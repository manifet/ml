def neural_network(input_data: float, weight: float) -> float:
    """Простейший перцептрон: умножаем вход на вес."""
    prediction = input_data * weight
    return round(prediction, 3)

# Базовые тесты
out_1 = neural_network(150, 0.3)
out_2 = neural_network(130, 0.4)

print(f"Результат 1: {out_1}")
print(f"Результат 2: {out_2}")

# Меняем параметры и смотрим на реакцию системы
print("\nПроверка с новыми весами:")

# Увеличили вход и вес -> прогноз вырос
print(neural_network(200, 0.5))  # 100.0
# Маленький вход, но большой вес
print(neural_network(50, 0.8))   # 40.0
# Минимальное влияние веса
print(neural_network(100, 0.1))  # 10.0
