# Lab05 K-Nearest Neighbors

## 实验环境

使用 `jupyter` 完成实验，具体而言，你只需要补全 `lab.ipynb` 中的代码即可

## 实验数据

使用了 `sklearn` 中的数据集。

数据集含义的解释：

id：数据的编号（没什么实际用处）

sepal length (cm)：花萼长度

sepal width (cm)：花萼宽度

petal length (cm)：花瓣长度

petal width (cm)：花瓣宽度

class：鸢尾花的类别，0代表Setosa，1代表Versicolour，2代表Virginica


数据集含义的解释：

ID: 数据的ID

CRIM：城镇人均犯罪率。

ZN：住宅用地超过 25000 的比例。

INDUS：城镇非零售商用土地的比例。

CHAS：查理斯河空变量（如果边界是河流，则为1；否则为0）。

NOX：一氧化氮浓度。

RM：住宅平均房间数。

AGE：1940 年之前建成的自用房屋比例。

DIS：到波士顿五个中心区域的加权距离。

RAD：辐射性公路的接近指数。

TAX：每 10000 美元的全值财产税率。

PTRATIO：城镇师生比例。

B：1000（Bk-0.63）^ 2，其中 Bk 指代城镇中黑人的比例。

LSTAT：人口中地位低下者的比例。

MEDV：自住房的平均房价，以千美元计

## 实验任务

本次实验有两个任务。

1. 实现一个名为 `KNNClassifier` 的类，这个类是一个基于 `KNN` 算法的分类器。该类可通过 `fit` 方法对分类器进行训练，可通过 `predict` 方法对输入数据进行分类。

2. 实现一个名为 `KNNRegression` 的类，这个类是一个基于 `KNN` 算法的回归，可通过 `fit` 方法对模型进行训练，可通过 `predict` 方法对输入数据进行预测

分类与回归两个任务的使用的数据集不一样，请注意甄别。

对于分类任务，请使用 `acc` 或 `f1` 值来衡量模型的好坏，而对于回归任务，可以与 `sklearn` 中的模型做对比（具体可比较 `MSE` 或 `R2` 值）

## 进阶任务

实现朴素贝叶斯与 `K-Means`
