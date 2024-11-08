[TOC]

# 以下调用tensorflow包完成

```python
import tensorflow as tf
```



# 基础中的基础

*全连接层*（fully-connected layer）， 因为它的每一个输入都通过矩阵-向量乘法得到它的每个输出。

tensor 张量

## 交叉熵误差

交叉熵误差(Cross Entropy Error) ， 评估模型输出的概率分布和真实概率分布的差异

- 相比于均方误差，交叉熵误差更能得到更大的差异

- 使用标签来对概率进行计算，当标签完全不同时，如(0,1),(1,0),则在计算时，只有真实类别对应的哪一项被计算在内
- ![交叉熵误差](C:\Users\LEGION\Desktop\笔记\Dive into deep learning图片\交叉熵误差.png)

## 均方误差

均方误差：预测概率和标记值差的平方和再求平均

- ![均方误差公式](C:\Users\LEGION\Desktop\笔记\Dive into deep learning图片\均方误差公式.png)

## 独热编码

```
tf.one_hot(y, depth=y_hat.shape[-1])))
```

1. **`tf.one_hot(y, depth=y_hat.shape[-1])`**：
   - 将真实标签 `y` 转换为独热编码（one-hot encoding）。`depth` 参数指定了类别的总数。
   - 例如，如果 `y` 是 `[0, 2]`，且 `depth` 是 3，则结果为 `[[1, 0, 0], [0, 0, 1]]`。

# 快速查阅文档方法

```
help(tf.ones)
```

为了知道模块中可以调用哪些函数和类，可以调用`dir`函数。 例如，我们可以查询随机数生成模块中的所有属性：

```
import tensorflow as tf

print(dir(tf.random))
```

# 基础操作

## 行向量

### 创建张量

arange创建一个行向量x，这个行向量包含以0开始的前12个整数，除非额外指定，新的张量将存储在内存中，并采用基于CPU的计算。

```python
x = tf.range(12)
```

> ```
> tf.Tensor([ 0  1  2  3  4  5  6  7  8  9 10 11], shape=(12,), dtype=int32)
> ```

#### 创建一个m * n的矩阵

```python
A = tf.reshape(tf.range(20), (5, 4))
```

> ```
> <tf.Tensor: shape=(5, 4), dtype=int32, numpy=
> array([[ 0,  1,  2,  3],
>        [ 4,  5,  6,  7],
>        [ 8,  9, 10, 11],
>        [12, 13, 14, 15],
>        [16, 17, 18, 19]], dtype=int32)>
> ```

#### 元素全0创建

```
tf.zeros((2, 3, 4))
```

创建和原张量相同形状的元素全0的张量

```
tf.zeros_like(x)
```

#### 元素全1创建

```
tf.ones((2, 3, 4))
```

创建和原张量相同形状的元素全1的张量

```
tf.ones_like(x)
```

#### 随机采样创建(正态分布)

每个元素都从均值为0，标准差为1 的标准高斯分布(正太分布)中随机采样

```
tf.random.normal(shape=[3, 4])
```

> tf.Tensor(
> [[-0.04797238  0.12064157  0.9641921  -1.1451155 ]
>  [ 0.651513    0.20602256  0.75781673  0.39021862]
>  [-0.15830263 -2.008918   -2.136549    0.02532925]], shape=(3, 4), dtype=float32)

#### 手动创建

```
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

> ```
> <tf.Tensor: shape=(3, 4), dtype=int32, numpy=
> array([[2, 1, 4, 3],
>        [1, 2, 3, 4],
>        [4, 3, 2, 1]], dtype=int32)>
> ```

### 获得shape

```python
x.shape
```

> (12,)

### 元素总数

```python
tf.size(x)
```

> tf.Tensor(12, shape=(), dtype=int32)

或者也可以通过调用Python的内置`len()`函数来访问张量的长度

```
len(x)
```

> 12

## 改变张量形状而不改变其他

```
X = tf.reshape(x, (3, 4))
```

> tf.Tensor(
> [[ 0  1  2  3]
>  [ 4  5  6  7]
>  [ 8  9 10 11]], shape=(3, 4), dtype=int32)

#### 自动计算后面的纬度

数量对不上会报错

```
print(tf.reshape(x,(3,-1)))
```

#### 转置矩阵

```python
A = tf.reshape(tf.range(20), (5, 4))
print(tf.transpose(A))
```

> tf.Tensor(
> [[ 0  4  8 12 16]
>  [ 1  5  9 13 17]
>  [ 2  6 10 14 18]
>  [ 3  7 11 15 19]], shape=(4, 5), dtype=int32)

## 针对元素总量求和

reduce_sum适合一维和二维的情况，高维得分片

```python
tensor = tf.constant([[1, 2, 3], 
                      [4, 5, 6]])
# 对所有元素求和
total_sum = tf.reduce_sum(tensor)

# 沿着第一个轴（行）求和
sum_axis0 = tf.reduce_sum(tensor, axis=0)

# 沿着第二个轴（列）求和
sum_axis1 = tf.reduce_sum(tensor, axis=1)

# 保持维度
sum_keepdims = tf.reduce_sum(tensor, axis=1, keepdims=True)

# 打印结果
print(total_sum)
print(sum_axis0)
print(sum_axis1)
print(sum_keepdims)
```

> tf.Tensor(21, shape=(), dtype=int32)
> tf.Tensor([5 7 9], shape=(3,), dtype=int32)
> tf.Tensor([ 6 15], shape=(2,), dtype=int32)
> tf.Tensor(
> [[ 6]
>  [15]], shape=(2, 1), dtype=int32)
>
> Process finished with exit code 0

## 运算符

### 按元素计算

两个矩阵的按**元素**乘法称为*Hadamard积*（Hadamard product）（数学符号⊙），也就是 x * y

```python
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
```

> ```
> (<tf.Tensor: shape=(4,), dtype=float32, numpy=array([ 3.,  4.,  6., 10.], dtype=float32)>,
>  <tf.Tensor: shape=(4,), dtype=float32, numpy=array([-1.,  0.,  2.,  6.], dtype=float32)>,
>  <tf.Tensor: shape=(4,), dtype=float32, numpy=array([ 2.,  4.,  8., 16.], dtype=float32)>,
>  <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.5, 1. , 2. , 4. ], dtype=float32)>,
>  <tf.Tensor: shape=(4,), dtype=float32, numpy=array([ 1.,  4., 16., 64.], dtype=float32)>)
> ```

### 点积

 另一个最基本的操作之一是点积。 给定两个向量x,y∈Rd， 它们的*点积*（dot product）x⊤y （或⟨x,y⟩） 是相同位置的按元素乘积的和：x⊤y=∑i=1dxiyi。

```python
y = tf.ones(4, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

> ```
> (<tf.Tensor: shape=(4,), dtype=float32, numpy=array([0., 1., 2., 3.], dtype=float32)>,
>  <tf.Tensor: shape=(4,), dtype=float32, numpy=array([1., 1., 1., 1.], dtype=float32)>,
>  <tf.Tensor: shape=(), dtype=float32, numpy=6.0>)
> ```

### 向量积

```
A.shape, x.shape, tf.linalg.matvec(A, x)
```

> ```
> (TensorShape([5, 4]),
>  TensorShape([4]),
>  <tf.Tensor: shape=(5,), dtype=float32, numpy=array([ 14.,  38.,  62.,  86., 110.], dtype=float32)>)
> ```

### 矩阵相乘

```
tf.matmul(A, B)
```

### 范数

L2，向量元素平方和的平方根

```
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

> ```
> <tf.Tensor: shape=(), dtype=float32, numpy=5.0>
> ```

L1，向量元素的绝对值之和

```
tf.reduce_sum(tf.abs(u))
```

> ```
> <tf.Tensor: shape=(), dtype=float32, numpy=7.0>
> ```

### 求以e为底的幂

```
tf.exp(x)
```

> ```
> <tf.Tensor: shape=(4,), dtype=float32, numpy=
> array([2.7182817e+00, 7.3890562e+00, 5.4598148e+01, 2.9809580e+03],
>       dtype=float32)>
> ```

### 张量连结(concatenate)

axis 轴 0为行，1为列

```python
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

> ```
> (<tf.Tensor: shape=(6, 4), dtype=float32, numpy=
>  array([[ 0.,  1.,  2.,  3.],
>         [ 4.,  5.,  6.,  7.],
>         [ 8.,  9., 10., 11.],
>         [ 2.,  1.,  4.,  3.],
>         [ 1.,  2.,  3.,  4.],
>         [ 4.,  3.,  2.,  1.]], dtype=float32)>,
>  <tf.Tensor: shape=(3, 8), dtype=float32, numpy=
>  array([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
>         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
>         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]], dtype=float32)>)
> ```

pytorch版本

```
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

结合成张量 => `[2][3][4]`

```
result = torch.stack((a, b))
```



### 逻辑运算（按元素计算）

通过*逻辑运算符*构建二元张量

```
X == Y
```

> ```
> <tf.Tensor: shape=(3, 4), dtype=bool, numpy=
> array([[False,  True, False,  True],
>        [False, False, False, False],
>        [False, False, False, False]])>
> ```

#### 对称矩阵比较

```python
B = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B == tf.transpose(B)

all_equal = tf.reduce_all(B)  # 判断所有元素是否都为 True
any_equal = tf.reduce_any(B)   # 判断是否有任何元素为 True
```

> ```
> <tf.Tensor: shape=(3, 3), dtype=bool, numpy=
> array([[ True,  True,  True],
>        [ True,  True,  True],
>        [ True,  True,  True]])>
> ```

### 广播机制

即使形状不同，我们仍然可以通过调用 *广播机制*（broadcasting mechanism）来执行按元素操作。 这种机制的工作方式如下：

1. 通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状；
2. 对生成的数组执行按元素操作。

```
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))

a + b
```

> tf.Tensor(
> [[0 1]
>  [1 2]
>  [2 3]], shape=(3, 2), dtype=int32)

### 索引和切片

```
X[-1], X[1:3]
```

> tf.Tensor([ 8  9 10 11], shape=(4,), dtype=int32) 
>
> tf.Tensor(7, shape=(), dtype=int32)

### 赋值

Tensors是不可变的，也不能背赋值，使用Variables，梯度不会通过Variables反向传播

可通过索引来写入元素

```
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
```

> <tf.Variable 'Variable:0' shape=(3, 4) dtype=int32, numpy=
> array([[ 0,  1,  2,  3],
>        [ 4,  5,  9,  7],
>        [ 8,  9, 10, 11]])>

#### 多个元素赋值

```
X_var = tf.Variable(X)
X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
X_var
```

> ```
> <tf.Variable 'Variable:0' shape=(3, 4) dtype=float32, numpy=
> array([[12., 12., 12., 12.],
>        [12., 12., 12., 12.],
>        [ 8.,  9., 10., 11.]], dtype=float32)>
> ```

## 节省内存

人话，尽量用Variable和assign

如果我们用`Y = X + Y`，我们将取消引用`Y`指向的张量，而是指向新分配的内存处的张量。

```
before = id(Y)
Y = Y + X
id(Y) == before
```

> ```
> False
> ```

我们可以这样

```
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

> ```
> id(Z): 139772097851776
> id(Z): 139772097851776
> ```

TensorFlow没有提供一种明确的方式来原地运行单个操作。

但是，TensorFlow提供了`tf.function`修饰符， 将计算封装在TensorFlow图中，该图在运行前经过编译和优化。 这允许TensorFlow删除未使用的值，并复用先前分配的且不再需要的值。 这样可以最大限度地减少TensorFlow计算的内存开销。(也就是js中的let，及用及删)

```
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # 这个未使用的值将被删除
    A = X + Y  # 当不再需要时，分配将被复用
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

> ```
> <tf.Tensor: shape=(3, 4), dtype=float32, numpy=
> array([[ 8.,  9., 26., 27.],
>        [24., 33., 42., 51.],
>        [56., 57., 58., 59.]], dtype=float32)>
> ```

## 转换为其他Python对象

张量转换为NumPy张量（`ndarray`）很容易，反之也同样容易。 转换后的结果不共享内存

人话，numpy为取值

```
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

> ```
> (numpy.ndarray, tensorflow.python.framework.ops.EagerTensor)
> ```

要将大小为1的张量转换为Python标量，我们可以调用`item`函数或Python的内置函数。

```
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

> ```
> (array([3.5], dtype=float32), 3.5, 3.5, 3)
> ```

## zip()

```python

list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']

zipped = zip(list1, list2)

# 将 zip 对象转换为列表
result = list(zipped)
print(result)  # 输出: [(1, 'a'), (2, 'b'), (3, 'c')]
```

## 获取处理的时间

```python
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```



# 数据预处理

## 创建/获取并写入数据集

```python
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

`data/house_tiny.csv`

> ```
> NumRooms,Alley,Price
> NA,Pave,127500
> 2,NA,106000
> 4,NA,178100
> NA,NA,140000
> ```

## 加载原始数据堆

```python
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

> ```
>    NumRooms Alley   Price
> 0       NaN  Pave  127500
> 1       2.0   NaN  106000
> 2       4.0   NaN  178100
> 3       NaN   NaN  140000
> ```

## 处理缺失值

`data.iloc 获取特定杭`

`array.fillna` 填充为na的缺失值

需要注意的是，mean在使用的时候不能进入string，会自动绕过NAN

- mean也可以这么来

  ```python
  tf.reduce_sum(A) / tf.size(A).numpy()
  ```

```
inputs, outputs = data.iloc[:, 0:1], data.iloc[:, 1:3]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

>    NumRooms
> 0       3.0
> 1       2.0
> 2       4.0
> 3       3.0

### 虚拟变量（哑变量）处理分类变量

`pd.get_dummies()`

使用：方法

- ```python
  import pandas as pd
  
  # 示例数据框
  data = pd.DataFrame({
      'Color': ['Red', 'Blue', 'Green'],
      'Size': ['S', 'M', 'L']
  })
  
  # 使用 get_dummies 转换分类变量
  dummies = pd.get_dummies(data)
  
  print(dummies)
  ```

  >    Color_Blue  Color_Green  Color_Red  Size_L  Size_M  Size_S
  > 0           0            0          1       0       0       1
  > 1           1            0          0       0       1       0
  > 2           0            1          0       1       0       0

由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”， `pandas`可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。 巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。 缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。

```python
inputs1,inputs2, outputs = data.iloc[:, 0:1],data.iloc[:, 1],data.iloc[:, 2]
inputs1 = inputs1.fillna(inputs1.mean())
inputs = pd.concat([inputs1,inputs2],axis=1)
inputs = pd.get_dummies(inputs, dummy_na=True)
```

>    NumRooms  Alley_Pave  Alley_nan
> 0       3.0        True      False
> 1       2.0       False       True
> 2       4.0       False       True
> 3       3.0       False       True
>
> Process finished with exit code 0

## 转换成张量

```python
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(outputs.to_numpy(dtype=float))
```



> tf.Tensor(
> [[3. 1. 0.]
>  [2. 0. 1.]
>  [4. 0. 1.]
>  [3. 0. 1.]], shape=(4, 3), dtype=float64) tf.Tensor([127500. 106000. 178100. 140000.], shape=(4,), dtype=float64)
>
> Process finished with exit code 0

# 画图

### 矢量库设置

设置图片显示格式为SVG格式(可缩放矢量图形)

```
d2l.use_svg_display()
```



## 二维图，包含函数

```python
import numpy as np 
import matplotlib.pyplot as plt 

def f(x):
    return 3 * x ** 2 - 4 * x

def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize  # 修改全局图表大小设置

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)  # 设置x轴标签
    axes.set_ylabel(ylabel)  # 设置y轴标签
    axes.set_xscale(xscale)  # 设置x轴的刻度类型（线性或对数）
    axes.set_yscale(yscale)  # 设置y轴的刻度类型（线性或对数）
    axes.set_xlim(xlim)  # 设置x轴的范围
    axes.set_ylim(ylim)  # 设置y轴的范围
    if legend:
        axes.legend(legend)  # 如果提供了图例，添加图例
    axes.grid()  # 显示网格线

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 3.5), axes=None):
    if legend is None:
        legend = []  # 如果没有提供图例，初始化为空列表

    set_figsize(figsize)  # 设置图表大小
    axes = axes if axes else plt.gca()  # 如果没有提供轴，获取当前轴

    def has_one_axis(X):
        """检查X是否为一维数组或列表,ndim 通常用于 NumPy 数组，表示数组的维度数"""
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]  # 如果X是一维的，转换为列表
    if Y is None:
        X, Y = [[]] * len(X), X  # 如果Y为空，将Y设置为X
    elif has_one_axis(Y):
        Y = [Y]  # 如果Y是一维的，将其转换为列表
    if len(X) != len(Y):
        X = X * len(Y)  # 如果X和Y的长度不一致，将X扩展到与Y相同的长度
    axes.cla()  # 清除当前轴上的内容
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)  # 绘制x和y的图形
        else:
            axes.plot(y, fmt)  # 如果x为空，仅绘制y
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)  # 设置轴属性


x = np.arange(0, 3, 0.1) 
plot(x, [f(x), 2 * x - 3, x], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)', 'x'])  
plt.show() 
```

> ![二维图绘制展示](C:\Users\LEGION\Desktop\笔记\Dive into deep learning图片\二维图绘制展示.png)

也可以这么写，会方便一些

```python
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from d2l import tensorflow as d2l

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

# 再次使用numpy进行可视化
x = np.arange(-7, 7, 0.01)

# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
plt.show()
```

## 批量显示图片

```python
import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt

d2l.use_svg_display() # 设置图片显示格式为SVG格式(可缩放矢量图形)

mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data() # 可以通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中。就是一堆分类好的图

# 图片展示
# some_image = mnist_train[0][0]
# plt.imshow(some_image, cmap="binary")
# plt.axis("off")
# plt.show()

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X = tf.constant(mnist_train[0][:18])
y = tf.constant(mnist_train[1][:18])
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y));
plt.show()
```

> ![批量显示图片](C:\Users\LEGION\Desktop\笔记\Dive into deep learning图片\批量显示图片.png)

# 线性神经网络

```python
import random
import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# 生成数据集
def synthetic_data(w, b, num_examples):
    """
    通过正态分布，生成y=Xw+b+噪声
    :param w:list权重
    :param b:单纯的增量
    :param num_examples:样本数量
    :return:
    """
    X = tf.zeros((num_examples, w.shape[0])) # 生成零矩阵，定义长宽
    X += tf.random.normal(shape=X.shape) # 为零矩阵加上标准正态分布随机出来的值
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b # 根据公式生成y值,同时加上w中的权重
    y += tf.random.normal(shape=y.shape, stddev=0.01) # stddev为标准差 ,在y上加上标准差为0.01的正太分布生成的概率
    y = tf.reshape(y, (-1, 1))# 变为一维列表
    return X, y

true_w = tf.constant([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) #这里的x为正太随机产生的初始变量，而y是x对应的值然后加上正太随机出来的偏移值

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].numpy(), labels.numpy(), 1);
plt.show()

# ----------------------------------------------------------------
# 随机获取批量集
def data_iter(batch_size, features, labels):
    """
    :param batch_size:批量大小
    :param features:特征矩阵
    :param labels:标签向量
    :return:
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)    # 随机改变indices中的顺序
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)

batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

# ----------------------------------------------------------------

def linreg(X,w,b):
    """线性回归模型"""
    return tf.matmul( X , w ) + b

def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2

def sgd(params, grads, lr, batch_size):
    """小批量随机梯度下降 SGD(Stochastic Gradient Descent)"""
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size)

# 初始化权重
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 计算l关于[w,b]的梯度
        dw, db = g.gradient(l, [w, b])
        # 使用参数的梯度更新参数
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')

print(f'w的估计误差: {true_w - tf.reshape(w, true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
```

> epoch 1, loss 0.032062
> epoch 2, loss 0.000112
> epoch 3, loss 0.000052
> w的估计误差: [-0.00075698 -0.00062275]
> b的估计误差: [0.00059891]

## 线性堆叠模型

```python
# keras是TensorFlow的高级API
net = tf.keras.Sequential()
#这是 Keras 中的一种模型类型，表示一个线性堆叠的模型。你可以将多个层（layers）按顺序添加到这个模型中。每一层的输出是下一层的输入。
net.add(tf.keras.layers.Dense(1))
#Dense 是 Keras 中的一种全连接层（fully connected layer），它的每个神经元与前一层的所有神经元相连接。
#参数 1 表示该层有 1 个输出神经元。这通常用于回归问题，表示模型将输出一个标量值
```

## softmax

人话，就是针对单一的线性回归矩阵，通过矩阵扩张的形式，同时让一个图片乘多个线性公式，最后结果那个大就说那个是

# 多层感知机

修正线性单元(Rectified linear unit,ReLu)

```
tf.nn.relu(x)
```

> ReLu(x) = max(x,0)



挤压函数(sigmoid      squashing function)

## 非线性激活函数

![非线性激活函数](C:\Users\LEGION\Desktop\笔记\Dive into deep learning图片\非线性激活函数.png)

# 深度学习计算基础知识

## 模型块操作

块合并

```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

> ```
> tensor([[0.2596],
>         [0.2596]], grad_fn=<AddmmBackward0>)
> ```

块分布 `print(rgnet)`

> ```
> Sequential(
>   (0): Sequential(
>     (block 0): Sequential(
>       (0): Linear(in_features=4, out_features=8, bias=True)
>       (1): ReLU()
>       (2): Linear(in_features=8, out_features=4, bias=True)
>       (3): ReLU()
>     )
>     (block 1): Sequential(
>       (0): Linear(in_features=4, out_features=8, bias=True)
>       (1): ReLU()
>       (2): Linear(in_features=8, out_features=4, bias=True)
>       (3): ReLU()
>     )
>     (block 2): Sequential(
>       (0): Linear(in_features=4, out_features=8, bias=True)
>       (1): ReLU()
>       (2): Linear(in_features=8, out_features=4, bias=True)
>       (3): ReLU()
>     )
>     (block 3): Sequential(
>       (0): Linear(in_features=4, out_features=8, bias=True)
>       (1): ReLU()
>       (2): Linear(in_features=8, out_features=4, bias=True)
>       (3): ReLU()
>     )
>   )
>   (1): Linear(in_features=4, out_features=1, bias=True)
> )
> ```

可通过嵌套列表索引一样访问

```python
rgnet[0][1][0].bias.data
```



## 模型参数访问

pytorch

```
net[2].state_dict()
```

> ```
> OrderedDict([('weight', tensor([[-0.0427, -0.2939, -0.1894,  0.0220, -0.1709, -0.1522, -0.0334, -0.2263]])), ('bias', tensor([0.0887]))])
> ```

tensorflow

```
net.layers[2].weights
```

> ```python
> [<tf.Variable 'dense_1/kernel:0' shape=(4, 1) dtype=float32, numpy=
> array([[-1.0469127 ],
>     [ 0.31355536],
>     [ 0.5405549 ],
>     [ 0.7610214 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
> ```



### 提取目标参数

#### bias

pytorch

```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

> ```
> <class 'torch.nn.parameter.Parameter'>
> Parameter containing:
> tensor([0.0887], requires_grad=True)
> tensor([0.0887])
> ```

#### 一次性访问多种参数以及另一种访问网络参数的形式

```python
print(*(name,param.shape) for name,param in net[0].named_parameters())
print(*(name,param.shape) for name,param in net.named_parameters())
```

> ```
> ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
> ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
> ```

```python
net.state_dict()['2.bias'].data
```

> ```
> tensor([0.0887])
> ```

### 查看是否调用反向传播

```python
net[2].weight.grad == None
```

> True

## 模型参数初始化

### 高斯分布

设置为标准差为0.01的高斯随机变量，且将偏置参数设置为0

```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

> ```
> (tensor([-0.0214, -0.0015, -0.0100, -0.0058]), tensor(0.))
> ```

### 常量初始化

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

### 针对不同的块采用不同的初始化办法

```python
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

### 自定义初始化办法

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

### 直接设置参数

```
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

## 稠密层，参数绑定

```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)

#验证办法
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

## 自定义层

```
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,X):
        return X-X.mean()

net = nn.Sequential(nn.Linear(8,256),CenteredLayer())
```

### 线性层

```python
class MyLinear(nn.Module):
    def __init__(self,in_units,units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units,units))
        self.bias = nn.Parameter(torch.randn(units))

    def forward(self,X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
```

#### 带参数

```python
class MyLinear(nn.Module):
    def __init__(self,in_units,units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units,units))
        self.bias = nn.Parameter(torch.randn(units))

    def forward(self,X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
print(linear.weight)

net = nn.Sequential(MyLinear(64,8),MyLinear(8,1))
Y = torch.rand(2,64)
print(Y)

print(net(Y))
```

## 读写文件

### 加载和保存张量

```python
x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
```

### 存储张量列表

```python
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

## 存储字符串映射到张量的字典

```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

### 加载和保存模型参数

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.output = nn.Linear(256,10)

    def forward(self,x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.random(size=(2,20))
Y = net(X)

torch.save(net.state_dict('mlp.params'))

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

## GPU

# 卷积神经网络

### 手搓2*2卷积核的运行流程

sum的

```python
def corr2d(X,K):
    """计算二维互相关运算"""
    h,w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[ i : i + h, j : j + w ] * K).sum()
    return Y

X = torch.tensor(range(9)).reshape(3,-1).float()
print(X)
K = torch.tensor(range(4)).reshape(2,-1).float()
print(corr2d(X,K))
```

### 卷积核设定

```python
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1) ## 宽高为3，在输入的边缘添加1个像素的填充

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1)) # 我们使用高度为5，宽度为3的卷积核，高度和宽度两边的填充分别为2和1。

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)#步幅为2
```

### 多输入互关运算

```python
def corr2d_multi_in(X,K):
    # 先遍历X和K的第0个维赌，再把他们加在一起
    return sum(d2l.corr2d(X,K) for X,K in zip(X,K))

X = torch.stack((torch.arange(9).reshape((3 , -1)),torch.arange(1,10).reshape((3 , -1)))).float()
K = torch.stack((torch.arange(4).reshape(2,-1),torch.arange(1,5).reshape(2,-1))).float()

print(corr2d_multi_in(X,K))
```

多输出通道

```python
def corr2d_multi_in(X,K):
    # 先遍历X和K的第0个维赌，再把他们加在一起
    return sum(d2l.corr2d(X,K) for X,K in zip(X,K))

X = torch.stack((torch.arange(9).reshape((3 , -1)),torch.arange(1,10).reshape((3 , -1)))).float() #2 * 3 * 3
K = torch.stack((torch.arange(4).reshape(2,-1),torch.arange(1,5).reshape(2,-1))).float()# 2 * 2 * 2 

# print(corr2d_multi_in(X,K))

def corr2d_multi_in_out(X,K):
    # 迭代K的第0个纬度，每次都对输入X执行互相关运算
# 最后再将所有的结果叠加在一起
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)#  k = 2 * 2 * 2
K = torch.stack((K,K+1,K+2),0)# 3 * 2 * 2 * 2
print(K.shape)
corr2d_multi_in_out(X,K)
```

> ```
> tensor([[[ 56.,  72.],
>          [104., 120.]],
> 
>         [[ 76., 100.],
>          [148., 172.]],
> 
>         [[ 96., 128.],
>          [192., 224.]]])
> ```

### 1 * 1 卷积层

唯一计算是发生在通道上

```python
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w)) # 3 * 9
    K = K.reshape((c_o, c_i))# 2 * 3
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)# 2 * 9
    return Y.reshape((c_o, h, w)) # 2 * 3 * 3

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
```

### 汇聚层

最大汇聚层和平均汇聚层

#### 调包

```
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

#### 多通道存在情况

汇聚层的输出通道数和输入层的通道数相同，如：

```
X = torch.cat((X, X + 1), 1)
print(x)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

> ```
> tensor([[[[ 0.,  1.,  2.,  3.],
>           [ 4.,  5.,  6.,  7.],
>           [ 8.,  9., 10., 11.],
>           [12., 13., 14., 15.]],
> 
>          [[ 1.,  2.,  3.,  4.],
>           [ 5.,  6.,  7.,  8.],
>           [ 9., 10., 11., 12.],
>           [13., 14., 15., 16.]]]])
>           
>           
> tensor([[[[ 5.,  7.],
>           [13., 15.]],
> 
>          [[ 6.,  8.],
>           [14., 16.]]]])
> ```

#### 手动实现

```
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

> ```
> tensor([[4., 5.],
>         [7., 8.]])
> ```

### LetNet(卷积神经网络)