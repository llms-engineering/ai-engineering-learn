# JAX 入门

> PyTorch 会就地修改张量。TensorFlow 会构建计算图。JAX 则编译纯函数。最后这一点会改变你对深度学习的思考方式。

**类型：** 动手构建
**语言：** Python
**前置：** 阶段 03 第 01–10 课、基础 NumPy
**建议时长：** 约 90 分钟

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数神经网络代码
- 说明 PyTorch 的即时可变执行与 JAX 的函数式编译模型之间的关键设计差异
- 用 jit 编译和 vmap 向量化加速训练循环，并与朴素 Python 实现对比
- 在 JAX 中训练一个简单网络，并对照 PyTorch 面向对象写法中的显式状态管理

## 问题

你已经会用 PyTorch 搭建神经网络：定义一个 `nn.Module`，调用 `.backward()`，再让优化器迈一步。它能用，也有数百万人在用。

但 PyTorch 的基因里有一个约束：它按 Python 的节奏、一步一步地即时追踪运算。每一次 `tensor + tensor` 都是一次独立的 kernel 启动；每一个训练步都会重新解释同一段 Python。这在常规规模下没问题，但当你要在 2,048 块 TPU 上训练 5400 亿参数的模型时，开销就会把你拖垮。

Google DeepMind 用 JAX 训练 Gemini。Anthropic 用 JAX 训练 Claude。这些不是小规模试验——它们是地球上最大规模的神经网络训练。他们选择 JAX，是因为它把训练循环当作可编译的程序，而不是一连串 Python 调用。

JAX 就是带三项超能力的 NumPy：自动微分、JIT 编译到 XLA、以及自动向量化。你写一个处理单个样本的函数；JAX 能给你一个处理整批、求梯度、编译成机器码并在多设备上运行的函数——且不必改动原函数。

## 概念

### JAX 的设计哲学

JAX 是函数式框架。没有类、没有可变状态、没有 `.backward()` 方法。取而代之的是：

| PyTorch | JAX |
|---------|-----|
| 带状态的 `nn.Module` 类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 即时执行 | 经 XLA 的 JIT 编译 |
| `for x in batch:` 手写循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变的 `model.parameters()` | 不可变的数组 pytree |

这不是风格偏好，而是编译器约束。JIT 要求纯函数——相同输入永远得到相同输出，且无副作用。正是这个限制，才换来可达 100 倍的加速。

### jax.numpy：熟悉的表层 API

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

函数名相同、广播规则相同、切片语义相同。但数组驻留在 GPU/TPU 上，且每一步运算都能被编译器追踪。

一个关键差异：JAX 数组不可变。不能写 `a[0] = 5`，而要写 `a = a.at[0].set(5)`。头一周会别扭，随后就会明白——不可变性正是 `grad`、`jit`、`vmap` 能够组合的前提。

### jax.grad：函数式自动微分

PyTorch 把梯度挂在张量上（`.grad`）。JAX 把梯度挂在函数上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接收一个函数，返回一个计算梯度的新函数。没有 `.backward()`，也没有存在张量上的计算图。梯度本身只是另一个你可以调用、组合或 JIT 编译的函数。

这种组合可以任意嵌套：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导、三阶导、Jacobian、Hessian——全靠组合 `grad`。PyTorch 也能做（`torch.autograd.functional.hessian`），但那是外挂；在 JAX 里，这是根基。

约束：`grad` 只对纯函数有效。函数里不能有打印（打印会在追踪阶段执行，而不是真正运行时）；不能改外部状态；没有显式 key 管理时也不能生成随机数。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

第一次调用时，JAX 会追踪函数——记录会发生哪些运算，却不真正执行。然后把追踪结果交给 XLA（Accelerated Linear Algebra，Google 面向 TPU/GPU 的编译器）。XLA 会融合运算、去掉多余内存拷贝，并生成优化机器码。

之后的调用完全绕过 Python，编译后的代码以接近 C++ 的速度在加速器上运行。

JIT 有帮助的情况：
- 训练步（同一计算重复成千上万次）
- 推理（同一模型、不同输入）
- 任意被多次调用、输入形状相近的函数

JIT 会吃亏的情况：
- 控制流依赖具体数值的 Python 分支（如 `if x > 0`，且 x 是被追踪的数组）
- 一次性计算（编译开销大于运行时间）
- 调试（追踪会掩盖真实执行路径）

控制流限制是真实存在的。`jax.lax.cond` 替代 `if/else`，`jax.lax.scan` 替代 `for` 循环。这些不是可选项——它们是编译的代价。

### vmap：自动向量化

你写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 把它提升为处理一整批：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 的含义是：不对 `params` 做 batch（共享），沿 `x` 的第 0 维做 batch。无需手写 `for`，无需改形状，也无需手动穿 batch 维。JAX 会推断 batch 维并向量化整段计算。

这不是语法糖。`vmap` 生成的融合向量化代码通常比 Python 循环快 10–100 倍，并且能与 `jit`、`grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本梯度，一行搞定。在 PyTorch 里没有黑科技几乎做不到。

### pmap：跨设备数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 把函数复制到所有可用设备（GPU/TPU）上，并拆分 batch。函数内部可用 `jax.lax.pmean`、`jax.lax.psum` 跨设备同步梯度。

Google 用 `pmap`（及其继任者 `shard_map`）在数千颗 TPU v5e 上训练 Gemini。编程模型是：先写单设备版本，再包一层 `pmap`，完成。

### Pytree：通用数据结构

JAX 操作的是「pytree」——由 list、tuple、dict 和数组嵌套而成的结构。模型参数就是一棵 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每一种 JAX 变换——`grad`、`jit`、`vmap`——都知道如何遍历 pytree。`jax.tree.map(f, tree)` 会对每个叶子应用 `f`。优化器正是这样一次性更新全部参数：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()`，也没有参数注册。树结构本身就是模型。

### 函数式 vs 面向对象

PyTorch 把状态存在对象里：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 用带显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数从外面传入。什么也不存，什么也不改。于是每个函数都可测试、可组合、可编译。代价是你要自己管理 params——或使用 Flax、Equinox 这类库。

### JAX 生态

JAX 提供原语，库提供人体工学：

| 库 | 角色 | 风格 |
|---------|------|-------|
| **Flax**（Google） | 神经网络层 | 带显式状态的 `nn.Module` |
| **Equinox**（Patrick Kidger） | 神经网络层 | 基于 pytree，更 Pythonic |
| **Optax**（DeepMind） | 优化器 + 学习率日程 | 可组合的梯度变换 |
| **Orbax**（Google） | 检查点 | 保存/恢复 pytree |
| **CLU**（Google） | 指标与日志 | 训练循环工具 |

Optax 是事实上的标准优化器库。它把梯度变换（Adam、SGD、裁剪）与参数更新分开，组合起来很轻松：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时用 JAX，何时用 PyTorch

| 维度 | JAX | PyTorch |
|--------|-----|---------|
| TPU 支持 | 一等公民（Google 两者都造） | 社区维护（torch_xla） |
| GPU 支持 | 良好（经 XLA 走 CUDA） | 业界顶尖（原生 CUDA） |
| 调试 | 难（追踪 + 编译） | 易（即时、逐行） |
| 生态 | 偏研究（Flax、Equinox） | 庞大（HuggingFace、torchvision 等） |
| 招聘 | 小众（Google/DeepMind/Anthropic） | 主流（到处都是） |
| 大规模训练 | 更强（XLA、pmap、mesh） | 良好（FSDP、DeepSpeed） |
| 原型速度 | 较慢（函数式开销） | 较快（改完就能跑） |
| 生产推理 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 谁在用 | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

诚实的答案：除非有明确理由，否则优先用 PyTorch。那些理由通常是——能用上 TPU、需要逐样本梯度、超大规模多设备训练，或你就在 Google / DeepMind / Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每次随机运算都需要显式的 PRNG key：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始会觉得麻烦，但它能保证跨设备和跨编译结果可复现——这是 PyTorch 的 `torch.manual_seed` 在多 GPU 场景下无法保证的性质。

```figure
batchnorm-effect
```

## 动手构建

### 步骤 1：环境与数据

我们将用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP：784 维输入，两个隐藏层（256 与 128 个神经元），10 个输出类别。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 步骤 2：初始化参数

没有类。只有一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He 初始化，手工完成。从一个种子拆出三把 PRNG key。每个权重都是嵌套字典里的不可变数组。

### 步骤 3：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数：参数进，预测出。没有 `self`，没有存状态。`loss_fn` 从零计算交叉熵——softmax、取对数、再取负均值。

### 步骤 4：JIT 编译的训练步

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 一次返回损失值与梯度。`@jax.jit` 把两个函数都编译到 XLA。第一次调用之后，每个训练步都不再碰 Python。

### 步骤 5：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 个 epoch，测试准确率约 97%。第一个 epoch 较慢（JIT 编译），第 2–10 个会很快。

注意缺了什么：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新就是一次组合好的函数调用。梯度计算、经 Adam 变换、再应用到参数——全部发生在 `train_step` 内部。

## 实际使用

### Flax：Google 的主流选择

Flax 是最常见的 JAX 神经网络库。它把 `nn.Module` 加了回来，但状态管理仍是显式的：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构像 PyTorch，但 `params` 与模型对象分离。`model.init()` 创建参数；`model.apply(params, x)` 跑前向。模型对象本身不持有状态。

### Equinox：更 Pythonic 的替代

Equinox（Patrick Kidger）把模型表示成 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是一棵 pytree，不需要 `.apply()`。参数就是模型的叶子。这更接近 JAX 的思维方式。

### Optax：可组合的优化器

Optax 把梯度变换与参数更新解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减——都作为一串变换组合起来。每个变换看到梯度、修改它，再交给下一个。没有庞大的单一优化器类。

## 落地交付

**安装：**

```bash
pip install jax jaxlib optax flax
```

GPU 支持：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能踩坑：**

- 第一次 JIT 调用很慢（编译）。做基准测试前先预热。
- 不要在 JIT 里用 Python 循环遍历 JAX 数组。改用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 在 JIT 内可用；普通 `print()` 不行。
- 用 `jax.profiler` 或 TensorBoard 做分析。XLA 编译可能掩盖瓶颈。
- JAX 默认预分配 75% 的 GPU 显存。设 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可关闭。

**检查点：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**
- `outputs/prompt-jax-optimizer.md` —— 选择合适 JAX 优化器配置的提示词
- `outputs/skill-jax-patterns.md` —— 覆盖 JAX 函数式模式的技能说明

## 练习

1. 给 MLP 加 dropout。在 JAX 里，dropout 需要 PRNG key——把 key 贯穿前向，并为每一层 dropout 拆分。对比有无 dropout 的测试准确率。

2. 用 `jax.vmap` 计算一批 32 张 MNIST 图的逐样本梯度，再算每个样本的梯度范数。哪些样本梯度最大？为什么？

3. 把手工前向换成通用的 `mlp_forward(params, x)`，使其支持任意层数。用 `jax.tree.leaves` 自动确定深度。

4. 对比有无 `@jax.jit` 的训练步性能：各计时 100 步。你的硬件上加速比有多大？第一次调用的编译开销是多少？

5. 用 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。分别在有无裁剪下训练，并画出训练过程中的梯度范数以观察效果。

## 关键词

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| XLA | 「让 JAX 变快的那个东西」 | Accelerated Linear Algebra——融合运算、并从计算图生成优化 GPU/TPU kernel 的编译器 |
| JIT | 「即时编译」 | JAX 在首次调用时追踪函数、编译到 XLA，之后运行编译版本 |
| 纯函数 | 「没有副作用」 | 输出只取决于输入——无全局状态、无就地修改、无随机（除非显式传 key） |
| vmap | 「自动组 batch」 | 把处理单样本的函数变成处理整批的函数，无需改写 |
| pmap | 「自动并行」 | 把函数复制到多设备上，并拆分输入 batch |
| Pytree | 「嵌套的数组字典」 | 任意由 list、tuple、dict 与数组嵌套的结构，JAX 可遍历并变换 |
| 追踪（Tracing） | 「记录计算」 | JAX 用抽象值执行函数以构建计算图，并不算出真实结果 |
| 函数式自动微分 | 「对函数求 grad」 | 通过变换函数来求导数，而不是把梯度存储挂在张量上 |
| Optax | 「JAX 的优化器库」 | 可组合的梯度变换库——Adam、SGD、裁剪、日程——可以串起来 |
| Flax | 「JAX 版的 nn.Module」 | Google 为 JAX 做的神经网络库：加层抽象，同时保持状态显式 |

## 进一步阅读

- JAX 文档：https://jax.readthedocs.io/ —— 官方文档，grad / jit / vmap 教程出色
- "JAX: composable transformations of Python+NumPy programs"（Bradbury 等，2018）—— 阐述设计哲学的原始论文
- Flax 文档：https://flax.readthedocs.io/ —— Google 的 JAX 神经网络库
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations"（2021）—— 更 Pythonic 的 Flax 替代
- DeepMind, "Optax: composable gradient transformation and optimisation" —— 标准优化器库
- "You Don't Know JAX"（Colin Raffel，2020）—— 实用的 JAX 坑点与模式指南，出自 T5 作者之一
