def neural_network(inputs, weights):
    prediction = 0
    products = []  # Список для хранения вкладов каждого входа
    
    # Считаем взвешенную сумму
    for i in range(len(weights)):
        step = inputs[i] * weights[i]
        products.append(step)
        prediction += step
        
    return prediction, products

# Тестовые данные
inp_data = [150, 40]
weights = [0.3, 0.4]

# Получаем итоговый прогноз и детализацию расчетов
res, steps = neural_network(inp_data, weights)

print(f"Итоговый выход: {res}")
print(f"Вклад каждого нейрона: {steps}")

# Еще пара прогонов для проверки
print(f"Тест 1: {neural_network([150, 40], [0.3, 0.4])[0]}")
print(f"Тест 2: {neural_network([80, 60], [0.2, 0.4])[0]}")