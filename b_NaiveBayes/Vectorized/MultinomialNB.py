# OS模块是Python系统编程的操作模块，提供对操作系统进行调用的接口；
# sys模块包含了与Python解释器和它的环境有关的函数；
import os
import sys

# os.path.abspath(path)返回path规范化的绝对路径；
# 以下代码将工作模块加入到搜索路径中；
root_path = os.path.abspath("../../")
if root_path not in sys.path:
    sys.path.append(root_path)

# matplotlib是python上的一个2D绘图库；
import matplotlib.pyplot as plt


# 调入Basic、DataUtil、Timing模块；
from b_NaiveBayes.Vectorized.Basic import *
from Util.Util import DataUtil
from Util.Timing import Timing


class MultinomialNB(NaiveBayes):
    MultinomialNBTiming = Timing()

    # 定义预处理数据的方法；
    @MultinomialNBTiming.timeit(level=1, prefix="[API] ")
    def feed_data(self, x, y, sample_weight=None):
        #
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        x, y, _, features, feat_dicts, label_dict = DataUtil.quantize_data(x, y, wc=np.array([False] * len(x[0])))

        # 利用Numpy中bincount方法，获得各类别数据的个数；
        cat_counter = np.bincount(y)
        # 记录各维度特征的取值个数；
        n_possibilities = [len(feats) for feats in features]

        # 获得各类别数据的下标；
        labels = [y == value for value in range(len(cat_counter))]

        # 利用下标获取按类别分开后的输入数据的数组；
        labelled_x = [x[ci].T for ci in labels]

        # 更新模型的各个属性；
        self._x, self._y = x, y
        self._labelled_x, self._label_zip = labelled_x, list(zip(labels, labelled_x))
        self._cat_counter, self._feat_dicts, self._n_possibilities = cat_counter, feat_dicts, n_possibilities
        self.label_dict = label_dict

        # 调用处理样本权重的函数，以更新记录条件概率的数组；
        self.feed_sample_weight(sample_weight)

    # 定义处理样本权重的函数
    @MultinomialNBTiming.timeit(level=1, prefix="[Core] ")
    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = []

        # 利用Numpy中bincount方法，获得带权重的条件概率的极大似然估计；
        for dim, p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._con_counter.append([
                    np.bincount(xx[dim], minlength=p) for xx in self._labelled_x])
            else:
                self._con_counter.append([
                    np.bincount(xx[dim], weights=sample_weight[label] / sample_weight[label].mean(), minlength=p)
                    for label, xx in self._label_zip])

    # 定义核心训练函数；
    @MultinomialNBTiming.timeit(level=1, prefix="[Core] ")
    def _fit(self, lb):
        n_dim = len(self._n_possibilities)
        n_category = len(self._cat_counter)
        self._p_category = self.get_prior_probability(lb)

        data = [[] for _ in range(n_dim)]
        for dim, n_possibilities in enumerate(self._n_possibilities):
            data[dim] = [
                [(self._con_counter[dim][c][p] + lb) / (self._cat_counter[c] + lb * n_possibilities)
                 for p in range(n_possibilities)] for c in range(n_category)]
        self._data = [np.asarray(dim_info) for dim_info in data]

    @MultinomialNBTiming.timeit(level=1, prefix="[Core] ")
    # 生成决策函数；
    def _func(self, x, i):
        x = np.atleast_2d(x).T
        rs = np.ones(x.shape[1])
        # 遍历各维度，利用data和条件独立性假设计算联合条件概率；
        for d, xx in enumerate(x):
            rs *= self._data[d][i][xx]
        # 利用先验概率和联合条件概率计算后验概率；
        return rs * self._p_category[i]

    # 定义数值化数据的函数；
    @MultinomialNBTiming.timeit(level=1, prefix="[Core] ")
    def _transfer_x(self, x):
        for i, sample in enumerate(x):
            for j, char in enumerate(sample):
                x[i][j] = self._feat_dicts[j][char]
        return x

    # 可视化；
    def visualize(self, save=False):
        colors = plt.cm.Paired([i / len(self.label_dict) for i in range(len(self.label_dict))])
        colors = {cat: color for cat, color in zip(self.label_dict.values(), colors)}
        rev_feat_dicts = [{val: key for key, val in feat_dict.items()} for feat_dict in self._feat_dicts]
        for j in range(len(self._n_possibilities)):
            rev_dict = rev_feat_dicts[j]
            sj = self._n_possibilities[j]
            tmp_x = np.arange(1, sj + 1)
            title = "$j = {}; S_j = {}$".format(j + 1, sj)
            plt.figure()
            plt.title(title)
            for c in range(len(self.label_dict)):
                plt.bar(tmp_x - 0.35 * c, self._data[j][c, :], width=0.35,
                        facecolor=colors[self.label_dict[c]], edgecolor="white",
                        label=u"class: {}".format(self.label_dict[c]))
            plt.xticks([i for i in range(sj + 2)], [""] + [rev_dict[i] for i in range(sj)] + [""])
            plt.ylim(0, 1.0)
            plt.legend()
            if not save:
                plt.show()
            else:
                plt.savefig("d{}".format(j + 1))

# 评估；
if __name__ == '__main__':
    import time

    train_num = 6000
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset(
        "mushroom", "../../_Data/mushroom.txt", n_train=train_num, tar_idx=0)

    # 实例化模型，记录时间；
    learning_time = time.time()
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    learning_time = time.time() - learning_time
    estimation_time = time.time()
    nb.evaluate(x_train, y_train)
    nb.evaluate(x_test, y_test)
    # 评估模型表现，记录评估花费的时间；
    estimation_time = time.time() - estimation_time
    # 将记录下来的耗时输出；
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            learning_time, estimation_time,
            learning_time + estimation_time
        )
    )
    nb.show_timing_log()
    nb.visualize()
