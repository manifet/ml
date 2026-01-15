import numpy as np

class Tensor:
    id_count = 0

    def __init__(self, data, creators=None, op=None, autograd=False, id=None):
        # Приводим к float, чтобы numpy не ругался на типы при расчетах
        self.data = np.array(data).astype(float)
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
            
        # Считаем, скольким "детям" нужен градиент этого тензора
        if self.creators:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def backward(self, grad=None, grad_origin=None):
        if not self.autograd:
            return

        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        # Ждем градиенты от всех потомков (проверка на тупиковые ветки)
        if grad_origin:
            if self.children[grad_origin.id] > 0:
                self.children[grad_origin.id] -= 1
            else:
                raise Exception(f"Странно: пришел лишний градиент от {grad_origin.id}")

        # Накапливаем градиент
        if self.grad is None:
            self.grad = grad
        else:
            self.grad.data += grad.data

        # Идем вглубь, только если собрали все ответы от детей
        if self.creators and (self.check_grads() or grad_origin is None):
            
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
                ax = int(self.op.split("_")[1])
                self.creators[0].backward(self.grad.expand(ax, self.creators[0].data.shape[ax]), self)

            elif "expand" in self.op:
                ax = int(self.op.split("_")[1])
                self.creators[0].backward(self.grad.sum(ax), self)

            elif self.op == "dot":
                # Классика: dL/dA = G @ B.T
                self.creators[0].backward(self.grad.dot(self.creators[1].transpose()), self)
                self.creators[1].backward(self.creators[0].transpose().dot(self.grad), self)

            elif self.op == "sigmoid":
                # Производная сигмоиды: s * (1 - s)
                ones = Tensor(np.ones_like(self.grad.data))
                self.creators[0].backward(self.grad * (self * (ones - self)), self)

            elif self.op == "tanh":
                # Производная тангенса: 1 - tanh^2
                ones = Tensor(np.ones_like(self.grad.data))
                self.creators[0].backward(self.grad * (ones - self * self), self)

            elif self.op == "relu":
                mask = Tensor((self.creators[0].data > 0).astype(float))
                self.creators[0].backward(self.grad * mask, self)

            elif self.op == "transpose":
                self.creators[0].backward(self.grad.transpose(), self)

    def check_grads(self):
        # Проверяем, все ли ветки сошлись
        return all(count == 0 for count in self.children.values())

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

    def sum(self, axis):
        if self.autograd:
            return Tensor(self.data.sum(axis), [self], f"sum_{axis}", True)
        return Tensor(self.data.sum(axis))

    def dot(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data.dot(other.data), [self, other], "dot", True)
        return Tensor(self.data.dot(other.data))

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), [self], "transpose", True)
        return Tensor(self.data.transpose())

    def sigmoid(self):
        res = 1 / (1 + np.exp(-self.data))
        if self.autograd: return Tensor(res, [self], "sigmoid", True)
        return Tensor(res)

    def relu(self):
        res = np.maximum(0, self.data)
        if self.autograd: return Tensor(res, [self], "relu", True)
        return Tensor(res)

    def expand(self, axis, copies):
        t_dims = list(range(len(self.data.shape)))
        t_dims.insert(axis, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        out = self.data.repeat(copies).reshape(new_shape).transpose(t_dims)
        if self.autograd: return Tensor(out, [self], f"expand_{axis}", True)
        return Tensor(out)

    def __repr__(self):
        return f"Tensor({self.data})"

class SGD:
    def __init__(self, weights, lr=0.01):
        self.weights = weights
        self.lr = lr

    def step(self):
        for w in self.weights:
            w.data -= self.lr * w.grad.data
            
    def zero_grad(self):
        # Обнуляем перед новой итерацией
        for w in self.weights:
            w.grad = None

# Тестим
np.random.seed(0)
weights = [Tensor(np.random.randn(3, 3), autograd=True) for _ in range(2)]
weights.append(Tensor(np.random.randn(3, 1), autograd=True))

optimizer = SGD(weights, lr=0.001)

train_data = [([1, 4, 5], 20), ([1, 5, 5], 25)]

for epoch in range(101):
    epoch_loss = 0
    for x, y in train_data:
        optimizer.zero_grad() # Чистим старое
        
        inp = Tensor([x], autograd=True)
        target = Tensor([[y]], autograd=True)

        # Прогон вперед
        l1 = inp.dot(weights[0]).sigmoid()
        l2 = l1.dot(weights[1]).sigmoid()
        pred = l2.dot(weights[2])

        loss = (pred - target) * (pred - target)
        loss.backward()
        
        optimizer.step()
        epoch_loss += loss.data

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {epoch_loss}")