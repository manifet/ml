def network(inp, weight):
    # Считаем взвешенную сумму для каждого нейрона
    return [sum(i * w for i, w in zip(inp, w_set)) for w_set in weight]

inp = [50, 165, 45]
target_weights = [0.3, 0.1, 0.7]
target_output = sum(i * w for i, w in zip(inp, target_weights))

found = False

print(f"Целевое значение выхода: {target_output}")

for w1 in [x * 0.1 for x in range(0, 20)]:
    for w2 in [x * 0.1 for x in range(0, 20)]:
        for w3 in [x * 0.1 for x in range(0, 20)]:
            current_weights = [w1, w2, w3]
            
            # Пропускаем, если веса идентичны целевым
            if current_weights == target_weights:
                continue
                
            prediction = sum(i * w for i, w in zip(inp, current_weights))
            
            # Сравниваем с допустимой погрешностью
            if abs(prediction - target_output) < 0.001:
                print(f"Найдена комбинация!")
                print(f"Веса 1-го нейрона: {current_weights}")
                print(f"Выход: {prediction} (совпадает с {target_output})")
                found = True
                break
        if found: break
    if found: break