---
title: TensorFlow初探理解
date: 2018-04-02 11:20:33
tags: TensofFlow
mail: liangshikuan@gmail.com
---
*本文仅供参考，如有错误还请联系作者改进学习 liangshikuan@gmail.com*

# TensorFlow


> 1. 使用图（graph）来标识计算。
> 2. 在会话（session）中来执行图。
> 3. 使用tensor来表示数据。
> 4. 通过变量（Variable）维护状态。
> 5. 使用feed和fetch可以为任意的操作（arbitrary operation）赋值或从其中获取数据。


*在图中节点称之为op节点（operation缩写），一个op可以获取0个或多个Tensor，执行计算，参数0个或多个Tensor。每个Tensor是一个类型化的多维数组。TensorFlow图描述了计算过程，为了进行计算图必须在会话里面启动。会话将图的op分发到CPU或GPU之类的设备上，同时执行op的方法，这些方法将产生tensor返回。（Python返回numpy nadarray对象；C/C++返回tensorflow::Tensor实例）*

{% asset_img TensorFlow_frame.png %}

[在线服务器资源CoLab](https://colab.research.google.com)

## 计算图
TensorFlow程序通常被组织成一个==构建阶段==和==执行阶段==：
在构建阶段，op的执行步骤被描述成图；在执行阶段使用会话执行途中的op。（例如咋构建阶段创建一个图来表示和训练神经网络，然后在执行阶段反复执行图中训练的op）。

## 构建图

* 创建源op，源op不需要任何输入（source op）. 源op的输出被传递给其他的op做运算。

```
import tensorflow as tf

# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# 加到默认图中.
#
# 构造器的返回值代表该常量 op 的返回值.
matrix1 = tf.constant([[3., 3.]])

# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.
product = tf.matmul(matrix1, matrix2)


# TensorFlow有一个默认的图，op构造器可以为其增加节点，以上默认图中有2个constant op和1个matmul op。

```

## 启动会话

1. 第一步需要创建一个Session，如果没有任何构建参数，会话构造器将启动默认的图。

```
# 启动默认图.
sess = tf.Session()

# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数. 
# 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回
# 矩阵乘法 op 的输出.
#
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
# 
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
#
# 返回值 'result' 是一个 numpy `ndarray` 对象.
result = sess.run(product)
print result
# ==> [[ 12.]]

# 任务完成, 关闭会话.
sess.close()
```

```
#Session需要显示close释放资源也可以以下with来自动完成关闭动作

with tf.Session() as sess:
  result = sess.run([product])
  print result
  
```

2. TensorFlow自动检测使用第一个GPS/CPU，除了第一个外默认不参与计算，需要明确指派op使用这些GPU，使用with...Device：

```
with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    ...

```

## 交互式使用
为了避免使用一个变量来持有会话，可以使用交互式会话InteractiveSession来代替Session。使用Tensor.eval()和Operation.run()来代替Session.run()。

```
# 进入一个交互式 TensorFlow 会话.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化器 initializer op 的 run() 方法初始化 'x' 
x.initializer.run()

# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果 
sub = tf.sub(x, a)
print sub.eval()
# ==> [-2. -1.]
```

## Tensor
TensorFlow程序使用Tensor数据结构来代表所有数据，计算图中操作间传递数据都是tensor。它包含一个静态类型rank和一个shape。

## 变量Variable
变量维护图执行过程中的状态信息。以下例子使用变量实现一个简单计数器：

```
# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()

# 启动图, 运行 op
with tf.Session() as sess:
  # 运行 'init' op
  sess.run(init_op)
  # 打印 'state' 的初始值
  print sess.run(state)
  # 运行 op, 更新 'state', 并打印 'state'
  for _ in range(3):
    sess.run(update)
    print sess.run(state)

# 输出:

# 0
# 1
# 2
# 3
```

由于如add()操作一样，在调用run()之前它并不会真正执行相加操作，assign()一样，只是先固定操作，待run()才执行赋值操作。

## Fetch

在上个例子中只取了state值，实际可以取回多个返回的tensor。

```
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print result

# 输出:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]
```

## Feed
TensorFlow提供了feed机制，该机制可以临时代替图中的任意操作tensor，可以对图中任意操作提交补丁，直接插入tensor。feed 只在条用它的方法内有效，方法结束feed就会消失。标记feed操作方法是使用tf.placeholder()，这些操作符被称作==占位符==。

```
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print sess.run([output], feed_dict={input1:[7.], input2:[2.]})

# 输出:
# [array([ 14.], dtype=float32)]
```

