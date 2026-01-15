import numpy as np

def train_batch(inp, weights, targets, epochs, lr, batch_size=2):
    print(f"Обучение: {epochs} эпох, Batch Size: {batch_size}")
    
    for epoch in range(1, epochs + 1):
        total_error = 0
        # Проходим по данным с шагом batch_size
        for i in range(0, len(inp), batch_size):
            batch_inp = inp[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            # Накапливаем градиент для всего пакета
            batch_delta = np.zeros_like(weights)
            
            for j in range(len(batch_inp)):
                pred = batch_inp[j].dot(weights)
                error = (pred - batch_targets[j]) ** 2
                total_error += error
                
                # Суммируем поправки
                batch_delta += (pred - batch_targets[j]) * batch_inp[j]
            
            # Обновляем веса один раз по среднему значению пакета
            weights -= (batch_delta / len(batch_inp)) * lr
            
        if epoch % 20 == 0:
            rmse = np.sqrt(total_error / len(inp))
            print(f"Эпоха {epoch}: RMSE = {rmse:.4f}")
            
    return weights

# Данные
inp_data = np.array([[150, 40], [170, 80], [160, 90]])
target_data = np.array([50, 120, 140])
initial_w = np.array([0.1, 0.1])

# Запуск мини-пакетного обучения
final_w = train_batch(inp_data, initial_w, target_data, 100, 0.0001, batch_size=2)