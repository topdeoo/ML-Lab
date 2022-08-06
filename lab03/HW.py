import collections
import numpy as np
import pandas as pd
import graphviz


class ID3Tree(object):
    def __init__(self):
        self.attrs = None
        self.X = None
        self.Y = None
        self.attrs_idx = None
        self.tree = {}
        self.node = "0"

    def get_attrs(self, dataset):
        """
        :param dataset:输入数据
        :return: attrs:属性字典
        """
        attrs = {}
        for i in range(dataset.shape[1]):
            attrs_val = sorted(set(dataset[1:, i]))
            attrs[dataset[0][i]] = attrs_val
        self.attrs = attrs
        return attrs

    def calc_ent(self, Y):
        """
        :param Y:label
        :return: Ent
        """
        res = 0
        m = np.size(Y)
        for i in self.attrs['贷款申请']:
            pk = np.sum(Y == i)
            pk = pk / m
            res -= pk*np.log2(pk+1e-8)
        return res

    def choose_best_attr(self, X, Y, attrs, attr_idx):
        """
        :param X: Dataset
        :param Y: label
        :param attrs: dict
        :param attr_idx: index of dict
        :return: the label of max info gain
        """
        res = self.calc_ent(Y)
        m = np.size(Y)
        max_gain = 0
        max_gain_attr = None
        for i, j in attr_idx.items():
            x = X[:, j]
            gain = res
            for val in attrs[i]:
                y = Y[x == val]
                if y.size != 0:
                    temp = self.calc_ent(y)
                else:
                    temp = 0
                gain -= temp * np.size(y) / m
            if gain > max_gain:
                max_gain = gain
                max_gain_attr = i
        return max_gain_attr

    def creat_tree(self, dataset):
        """
        :param dataset: 数据集
        :return: 生成的决策树
        """
        self.X = dataset[1:, :-1]
        self.Y = dataset[1:, -1]
        pure_attrs = self.attrs.copy()
        del(pure_attrs['贷款申请'])
        attr_names = [attr_name for attr_name in pure_attrs.keys()]
        attr_idx = {}
        for i, j in enumerate(attr_names):
            attr_idx[j] = i
        self.attrs_idx = attr_idx

        self.tree['root'] = {}
        self._creat_tree(
            self.X, self.Y, self.tree['root'], pure_attrs, attr_idx)
        self.tree = self.tree['root']

    def _creat_tree(self, X, Y, node, attrs, attr_idx):
        """
        :param X:数据集
        :param Y: 标签
        :param node: 父节点
        :param attrs: 属性
        :param attr_idx: 属性的索引
        """
        if len(set(Y.tolist())) == 1:
            node['贷款申请'] = Y[0]
            return
        same = True
        for i in range(X.shape[1]):
            if len(set(X[:, i].tolist())) > 1:
                same = False
        if not attrs or same:
            cnt = collections.Counter(Y)
            best_y = cnt.most_common()[0][0]
            node['贷款申请'] = best_y
            return
        best_attr = self.choose_best_attr(X, Y, attrs, attr_idx)
        node[best_attr] = {}
        node = node[best_attr]
        for val in attrs[best_attr]:
            node[val] = {}
            sub_set = X.copy()
            val_idx = sub_set[:, attr_idx[best_attr]] == val
            sub_set = sub_set[val_idx, :]
            y_sub_set = Y[val_idx]
            sub_set = np.delete(sub_set, attr_idx[best_attr], 1)
            if sub_set.size == 0:
                cnt = collections.Counter(Y)
                best_y = cnt.most_common()[0][0]
                node[best_attr]['贷款申请'] = best_y
            else:
                new_attrs = attrs.copy()
                del(new_attrs[best_attr])
                new_attr_names = [
                    new_attr_name for new_attr_name in new_attrs.keys()]
                new_attr_idx = {}
                for i, j in enumerate(new_attr_names):
                    new_attr_idx[j] = i
                self._creat_tree(sub_set, y_sub_set,
                                 node[val], new_attrs, new_attr_idx)

    def tree_traversal(self, g, parent_node, parent_node_name, parent_attr, parent_attr_value):
        """
        :param g: graph
        :param parent_node: 父节点
        :param parent_node_name: 父节点名称
        :param parent_attr: 父节点的属性
        :param parent_attr_value: 父节点到此节点的条件
        :return:
        """
        if (parent_attr and parent_attr_value) is None:
            if '贷款申请' in parent_node.keys():
                g.node(name=self.node,
                       label=parent_node['贷款申请'], fontname="Microsoft YaHei")
                return
            else:
                attr = list(parent_node.keys())[0]
                node = parent_node[attr]
                parent_node_name = "0"
                for attr_value in node.keys():
                    self.tree_traversal(
                        g, node[attr_value], parent_node_name, attr, attr_value)
        else:
            if '贷款申请' in parent_node.keys():
                g.node(name=parent_node_name, label=parent_attr,
                       fontname="Microsoft YaHei", shape='box')
                self.node = str(int(self.node) + 1)
                g.node(name=self.node,
                       label=parent_node['贷款申请'], fontname="Microsoft YaHei")
                g.edge(parent_node_name, self.node,
                       label=parent_attr_value, fontname="Microsoft YaHei")
            else:
                attr = list(parent_node.keys())[0]
                g.node(name=parent_node_name, label=parent_attr,
                       fontname="Microsoft YaHei", shape='box')
                self.node = str(int(self.node) + 1)
                g.node(name=self.node, label=attr,
                       fontname="Microsoft YaHei", shape='box')
                g.edge(parent_node_name, self.node,
                       label=parent_attr_value, fontname="Microsoft YaHei")
                node = parent_node[attr]
                parent_node_name = self.node
                for attr_value in node.keys():
                    self.tree_traversal(
                        g, node[attr_value], parent_node_name, attr, attr_value)

    def tree_visualize(self, file_name=None):
        """
        :param file_name: file name
        :return:
        """
        if file_name:
            g = graphviz.Digraph(
                "Decision Tree", filename=file_name, format='png')
        else:
            g = graphviz.Digraph("Decision Tree")
        self.tree_traversal(g, self.tree, None, None, None)
        g.view()


'''
年龄：0表示青年，1表示中年，2表示老年
有工作：1表示有，0表示无
有自己的房子：1表示有，0表示无
信贷情况：0表示一般，1表示好，2表示非常好
类别：1表示是，0表示否
'''
df = pd.read_csv('Data.csv')
df = df.drop(columns=['ID'])
data = df.to_numpy(dtype='<U20')
head = np.array([['年龄', '有工作', '有自己的房子', '信贷情况', '类别']], dtype='<U20')
data = np.r_[head, data]
data[0, -1] = '贷款申请'
tree = ID3Tree()
attrs = tree.get_attrs(data)
tree.creat_tree(data)
tree.tree_visualize("ID3决策树")
