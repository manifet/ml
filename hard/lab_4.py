import numpy as np
from tensorflow.keras.datasets import mnist

# Активация
def relu(x): return (x > 0) * x
def relu_deriv(x): return (x > 0)

# 1. Загрузка и подготовка
(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_size = 1000
test_size = 10000
pixels = 784
num_classes = 10

# Решейп и нормализация
x_train_proc = x_train[:train_size].reshape(train_size, pixels) / 255
x_test_proc = x_test[:test_size].reshape(test_size, pixels) / 255

y_train_oh = np.zeros((train_size, num_classes))
y_train_oh[np.arange(train_size), y_train[:train_size]] = 1

y_test_oh = np.zeros((test_size, num_classes))
y_test_oh[np.arange(test_size), y_test[:test_size]] = 1

# 2. Настройки сети
hidden_size = 100 # Увеличим слой, раз используем Dropout
batch_size = 100  # Размер пакета
lr = 0.005
epochs = 300 

np.random.seed(1)
w_in_hid = 0.2 * np.random.random((pixels, hidden_size)) - 0.1
w_hid_out = 0.2 * np.random.random((hidden_size, num_classes)) - 0.1

# 3. Обучение
for epoch in range(epochs):
    epoch_correct = 0
    
    # Идем по батчам
    for j in range(int(train_size / batch_size)):
        start, end = j * batch_size, (j + 1) * batch_size
        
        layer_in = x_train_proc[start:end]
        
        # Forward pass + Dropout
        layer_hid = relu(np.dot(layer_in, w_in_hid))
        
        # Маска: 50% нейронов "выключаются". Умножаем на 2 для сохранения масштаба сигнала
        dropout_mask = np.random.randint(2, size=layer_hid.shape)
        layer_hid *= dropout_mask * 2
        
        layer_out = np.dot(layer_hid, w_hid_out)

        # Считаем точность в пакете
        epoch_correct += np.sum(np.argmax(layer_out, axis=1) == np.argmax(y_train_oh[start:end], axis=1))

        # Backward pass
        delta_out = (layer_out - y_train_oh[start:end]) / batch_size
        # Применяем маску и к градиенту
        delta_hid = delta_out.dot(w_hid_out.T) * relu_deriv(layer_hid) * dropout_mask
        
        # Апдейт весов
        w_hid_out -= lr * layer_hid.T.dot(delta_out)
        w_in_hid -= lr * layer_in.T.dot(delta_hid)

    if (epoch + 1) % 50 == 0:
        print(f"Эпоха {epoch+1} | Точность обучения: {epoch_correct/train_size:.2%}")

# 4. Тест
test_preds = relu(x_test_proc.dot(w_in_hid)).dot(w_hid_out)
test_acc = np.mean(np.argmax(test_preds, axis=1) == y_test[:test_size])
print(f"\nТочность на тесте: {test_acc:.2%}")