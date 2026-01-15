def layer_step(inp: list, weights: list) -> list:
    """Просчет одного слоя: входы умножаются на матрицу весов."""
    outputs = []
    for w_neuron in weights:
        # Считаем сумму произведений входа на веса текущего нейрона
        dot_product = sum(i * w for i, w in zip(inp, w_neuron))
        outputs.append(dot_product)
    return outputs

# Входные данные
inp = [9, 9]

# Начальные веса скрытого слоя
w_h = [[0.4, 0.1], [0.3, 0.2]]
# Веса выходного слоя 
w_out = [[0.4, 0.1], [0.3, 0.1]]

print(f"Старт. Скрытый слой: {layer_step(inp, w_h)}")

# Увеличиваем веса, пока оба нейрона не выдадут > 5
step = 0.05
iteration = 0

while any(val < 5 for val in layer_step(inp, w_h)):
    for i in range(len(w_h)):
        if layer_step(inp, w_h)[i] < 5:
            # Слегка увеличиваем веса конкретного нейрона
            w_h[i][0] += step
            w_h[i][1] += step
    iteration += 1

predict_h = layer_step(inp, w_h)
final_predict = layer_step(predict_h, w_out)

print("--- Результат ---")
print(f"Итераций подбора: {iteration}")
print(f"Веса скрытого слоя: {w_h}")
print(f"Выход скрытого слоя > 5: {predict_h}")
print(f"Финальный прогноз сети: {final_predict}")