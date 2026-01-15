import numpy as np

class Tensor:
    id_count = 0

    def __init__(self, data, creators=None, op=None, autograd=False, id=None):
        self.data = np.array(data)
        self.creators = creators
        self.op = op
        self.grad = None
        self.autograd = autograd
        self.children = {}
        
        if id is None:
            Tensor.id_count += 1
            self.id = Tensor.id_count
        else:
            self.id = id

        # Считаем, сколько раз тензор используется в графе
        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def all_children_grads_arrived(self):
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        if not self.autograd:
            return

        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        if grad_origin is not None:
            if self.children[grad_origin.id] > 0:
                self.children[grad_origin.id] -= 1
            else:
                raise Exception("Can't backprop more than once!")

        # Накапливаем градиент
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        # Идем дальше по графу, только если получили все градиенты от «детей»
        if self.creators is not None and (self.all_children_grads_arrived() or grad_origin is None):
            
            if self.op == "+":
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad, self)

            elif self.op == "-1":
                self.creators[0].backward(self.grad.__neg__(), self)

            elif self.op == "-":
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad.__neg__(), self)

            elif self.op == "*":
                self.creators[0].backward(self.grad * self.creators[1], self)
                self.creators[1].backward(self.grad * self.creators[0], self)

            elif "sum" in self.op:
                axis = int(self.op.split("_")[1])
                self.creators[0].backward(self.grad.expand(axis, self.creators[0].data.shape[axis]), self)

            elif "expand" in self.op:
                axis = int(self.op.split("_")[1])
                self.creators[0].backward(self.grad.sum(axis), self)

            elif self.op == "transpose":
                self.creators[0].backward(self.grad.transpose(), self)

            elif self.op == "dot":
                # Классика матричного backprop
                self.creators[0].backward(self.grad.dot(self.creators[1].transpose()), self)
                self.creators[1].backward(self.creators[0].transpose().dot(self.grad), self)

            elif self.op == "sigmoid":
                ones = Tensor(np.ones_like(self.grad.data))
                self.creators[0].backward(self.grad * (self * (ones - self)), self)

            elif self.op == "tanh":
                ones = Tensor(np.ones_like(self.grad.data))
                self.creators[0].backward(self.grad * (ones - self * self), self)

            elif self.op == "relu":
                mask = Tensor((self.creators[0].data > 0).astype(float))
                self.creators[0].backward(self.grad * mask, self)
            
            elif self.op == "softmax":
                # Для простоты прокидываем градиент как есть
                self.creators[0].backward(self.grad, self)

            elif self.op == "pow":
                p = self.creators[1].data
                self.creators[0].backward(self.grad * Tensor(p * (self.creators[0].data ** (p - 1))), self)

    # --- Магия операторов ---
    
    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, [self, other], "-", True)
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, [self, other], "*", True)
        return Tensor(self.data * other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, [self], "-1", True)
        return Tensor(self.data * -1)

    def __pow__(self, power):
        if self.autograd:
            return Tensor(self.data ** power, [self, Tensor(power)], "pow", True)
        return Tensor(self.data ** power)

    def sum(self, axis):
        if self.autograd:
            return Tensor(self.data.sum(axis), [self], f"sum_{axis}", True)
        return Tensor(self.data.sum(axis))

    def expand(self, axis, count):
        # Нужен для корректного сложения градиентов при broadcasting
        shape = list(self.data.shape)
        shape.insert(axis, count)
        new_data = np.expand_dims(self.data, axis).repeat(count, axis)
        if self.autograd:
            return Tensor(new_data, [self], f"expand_{axis}", True)
        return Tensor(new_data)

    def dot(self, other):
        if self.autograd:
            return Tensor(self.data.dot(other.data), [self, other], "dot", True)
        return Tensor(self.data.dot(other.data))

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.T, [self], "transpose", True)
        return Tensor(self.data.T)

    def sigmoid(self):
        res = 1 / (1 + np.exp(-self.data))
        if self.autograd: return Tensor(res, [self], "sigmoid", True)
        return Tensor(res)

    def tanh(self):
        res = np.tanh(self.data)
        if self.autograd: return Tensor(res, [self], "tanh", True)
        return Tensor(res)

    def relu(self):
        res = np.maximum(0, self.data)
        if self.autograd: return Tensor(res, [self], "relu", True)
        return Tensor(res)

    def softmax(self):
        # Трюк с max для численной стабильности
        e_x = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
        res = e_x / e_x.sum(axis=1, keepdims=True)
        if self.autograd: return Tensor(res, [self], "softmax", True)
        return Tensor(res)

    def __repr__(self):
        return str(self.data)

# --- NN модули ---

class SGD:
    def __init__(self, parameters, lr=0.1):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                p.data -= p.grad.data * self.lr
                # Обнуляем после шага
                p.grad.data *= 0

class Linear:
    def __init__(self, n_in, n_out):
        # He-init: хорош для ReLU/Tanh
        self.w = Tensor(np.random.randn(n_in, n_out) * np.sqrt(2/n_in), autograd=True)
        self.b = Tensor(np.zeros(n_out), autograd=True)

    def forward(self, x):
        return x.dot(self.w) + self.b.expand(0, len(x.data))

    def get_params(self):
        return [self.w, self.b]

class Sequential:
    def __init__(self, layers=[]):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def get_params(self):
        params = []
        for l in self.layers:
            if hasattr(l, 'get_params'):
                params += l.get_params()
        return params

class MSELoss:
    def forward(self, pred, target):
        diff = pred - target
        return (diff * diff).sum(0) * Tensor(1.0 / pred.data.shape[0])

# Обучение

x = Tensor([[80, 25], [90, 30], [85, 35], [45, 22], [55, 28], [50, 35]], autograd=True)
y = Tensor([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], autograd=True)

model = Sequential([
    Linear(2, 8),
    Linear(8, 2)
])

# Чтобы не перегружать граф, активации выносим или делаем через тензоры напрямую
optimizer = SGD(model.get_params(), lr=0.001)
criterion = MSELoss()

for i in range(2000):
    # Forward
    pred = model.forward(x).sigmoid() # применяем сигмоиду в конце
    loss = criterion.forward(pred, y)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    if i % 500 == 0:
        print(f"Epoch {i} | Loss: {loss.data.sum()}")

def predict(weight, age):
    return model.forward(Tensor([[weight, age]])).sigmoid().data

print("\nТест мужчина 80/25:", predict(80, 25))
print("Тест женщина 50/30:", predict(50, 30))