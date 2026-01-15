def neural_network(inp: float, weights: list) -> list:
    return [inp * w for w in weights]

inp = 4

# Проверка разных комбинаций весов
test_cases = [
    [0.2, 0.5],    
    [0.15, 0.15],  
    [0.13, 0.13],  
    [0.12, 0.12]  
]

print("Результаты подбора весов:")
for weights in test_cases:
    result = neural_network(inp, weights)
    
    # Проверяем условие задачи для наглядности
    status = "OK (>0.5)" if all(r > 0.5 for r in result) else "Мало"
    
    # Округление для красивого вывода
    clean_res = [round(r, 2) for r in result]
    print(f"Веса: {weights} -> Выход: {clean_res} | {status}")