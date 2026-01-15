def neural_network(inp: float, weight: float) -> float:
    # Простое перемножение входа на вес
    return inp * weight

# Тестовый набор данных
inputs = [150, 160, 170, 180, 190]
weight = 0.3

print("Результаты прогона нейросети:")

# Проходим циклом по списку и считаем предикты
for val in inputs:
    result = neural_network(val, weight)
    # Вывод с форматированием для наглядности
    print(f"Вход: {val} \t| Прогноз: {result:.1f}")