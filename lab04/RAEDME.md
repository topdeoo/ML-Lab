# Lab 04 Logistic Regression

## 实验环境

使用 `jupyter` 完成实验，具体而言，你只需要补全 `lab.ipynb` 中的代码即可

## 实验数据

使用了 `sklearn` 中的数据集，略去了处理数据的环节，这次使用的是鸢尾花数据集。

数据集含义的解释：

id：数据的编号（没什么实际用处）

sepal length (cm)：花萼长度

sepal width (cm)：花萼宽度

petal length (cm)：花瓣长度

petal width (cm)：花瓣宽度

class：鸢尾花的类别，0代表Setosa，1代表Versicolour，2代表Virginica

## 实验任务

实现一个名为 `Logistic Regression` 的类，这个类是一个逻辑回归模型，我们可以通过 `fit` 方法来训练模型，通过 `predict` 方法来给出模型的预测结果。

我们需要使用交叉熵作为损失函数（不是 `MSE`)，对于二分类模型，交叉熵函数的定义为：

$$
L(p_i) = \frac{1}{N} \sum_i -[y_i\times log(p_i) + (1- y_i)\times log(1-p_i)]
$$
这里的 $p_i$ 表示预测为正类的概率

(请思考为什么不使用 `MSE`，你可以都试试看，记录一下损失函数的下降过程，然后画个图看看)

训练完成后，我们需要使用`acc`来衡量模型的好坏，通过计算 $acc = \frac{\sum^n_{i=1}I(\hat{y}_i == y_i)}{n} * 100\%$，其中 $I(\hat{y}_i == y_i)$ 表示如果 $\hat{y}_i == y_i$就取 $1$，否则为 $0$。当然，也可以选择使用 `F1 score` 来衡量模型表现的好坏，指标很多，可以多用一些试试看。

> 最好是使用 `sklearn` 中的模型再做一遍试试看结果如何

## 进阶任务

以下任务二选一：

1. 使用 `Logistic Regression` 做一个多分类模型，例如对这个数据集能够做到三分类

> 多分类的做法：可以做多次二分类来达到多分类的效果

2. 使用交叉熵损失函数重写 `lab02` 中的神经网络