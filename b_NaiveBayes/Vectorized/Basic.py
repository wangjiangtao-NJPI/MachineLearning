# OS模块是Python系统编程的操作模块，提供对操作系统进行调用的接口；
# sys模块包含了与Python解释器和它的环境有关的函数；
import os
import sys


# os.path.abspath(path)返回path规范化的绝对路径；
# 以下代码将工作模块加入到搜索路径中；
root_path = os.path.abspath("../../")
if root_path not in sys.path:
    sys.path.append(root_path)


# numpy（Numerical Python）提供了python对多维数组对象的支持；
import numpy as np
# 调用参数pi；
from math import pi
# 调用Timing模块；
from Util.Timing import Timing
# 调用ClassifierBase模块；
from Util.Bases import ClassifierBase
# 指数运算符 ** 来计算平方根；
sqrt_pi = (2 * pi) ** 0.5


# http://www.runoob.com/python3/python3-class.html；
# https://blog.csdn.net/u010157603/article/details/50999108###；
# classmethod 和普通函数调用时都有默认参数传入，只有staticmethod调用时没有任何默认参数传入；
# 类NBFunctions用于GaussianNB；
class NBFunctions:
    @staticmethod
    def gaussian(x, mu, sigma):
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt_pi * sigma)

    @staticmethod
    def gaussian_maximum_likelihood(labelled_x, n_category, dim):

        mu = [np.sum(
            labelled_x[c][dim]) / len(labelled_x[c][dim]) for c in range(n_category)]
        sigma = [np.sum(
            (labelled_x[c][dim] - mu[c]) ** 2) / len(labelled_x[c][dim]) for c in range(n_category)]

        def func(_c):
            def sub(x):
                return NBFunctions.gaussian(x, mu[_c], sigma[_c])
            return sub

        return [func(_c=c) for c in range(n_category)]


# 朴素贝叶斯模型基（父）类，也是ClassifierBase的派生（子）类；
class NaiveBayes(ClassifierBase):
    NaiveBayesTiming = Timing()

    # 初始化结构
    # self._x 、 self._y：记录训练集变量；
    # self._data：核心数组，存储条件概率；
    # self._n_possibilities：记录各维度特征值个数的数组；
    # self._p_category：
    # self._labelled_x ：记录按类别分开后的输入数据的数组；
    # self._label_zip：记录类别相关信息；
    # self._cat_counter ：记录第i类数据的个数；
    # self._con_counter：记录条件概率的原始极大似然估计；
    # self.label_dict：记录数值化类别时的转换关系；
    # self._feat_dicts：记录数值化各维度特征时的转换关系；
    # self._params["lb"] ：

    def __init__(self, **kwargs):
        # **kwargs：表示的就是形参中按照关键字传值把多余的传值以字典的方式呈现；
        # def foo(x,**kwargs):
        #     print(x)
        #     print(kwargs)
        # foo(1,y=1,a=2,b=3,c=4)#将y=1,a=2,b=3,c=4以字典的方式给了kwargs
        # 执行结果是：
        # 1
        # {'y': 1, 'a': 2, 'b': 3, 'c': 4}
        super(NaiveBayes, self).__init__(**kwargs)
        self._x = self._y = self._data = None
        self._n_possibilities = self._p_category = None
        self._labelled_x = self._label_zip = None
        self._cat_counter = self._con_counter = None
        self.label_dict = self._feat_dicts = None

        self._params["lb"] = kwargs.get("lb", 1)

    # sample_weight：样本权重；
    def feed_data(self, x, y, sample_weight=None):
        # pass 不做任何事情，一般用做占位语句；
        pass

    def feed_sample_weight(self, sample_weight=None):
        pass

    # 定义计算先验概率的函数，lb为平滑项，取1时为拉普拉斯平滑；
    @NaiveBayesTiming.timeit(level=2, prefix="[API] ")
    def get_prior_probability(self, lb=1):
        return [(c_num + lb) / (len(self._y) + lb * len(self._cat_counter))
                for c_num in self._cat_counter]

    # 定义具有普适性的训练函数；
    @NaiveBayesTiming.timeit(level=2, prefix="[API] ")
    def fit(self, x=None, y=None, sample_weight=None, lb=None):
        if sample_weight is None:
            sample_weight = self._params["sample_weight"]
        if lb is None:
            lb = self._params["lb"]
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weight)
        self._fit(lb)

    def _fit(self, lb):
        pass

    def _func(self, x, i):
        pass

    # 定义预测单一样本的函数；
    @NaiveBayesTiming.timeit(level=1, prefix="[API] ")
    def predict(self, x, get_raw_result=False, **kwargs):
        # 在预测前，先把新的输入数据数值化；
        # 如输入Numpy数组，则转化为Python数组；
        # 因为Python数组操作更快；
        if isinstance(x, np.ndarray):
            x = x.tolist()
        # 否则，对数组进行拷贝；
        else:
            x = [xx[:] for xx in x]
        # 调用相关方法数值化，根据模型的不同而不同；
        x = self._transfer_x(x)
        m_arg, m_probability = np.zeros(len(x), dtype=np.int8), np.zeros(len(x))
        # 遍历各类别，找到能使后验概率最大化的类别；
        for i in range(len(self._cat_counter)):
            p = self._func(x, i)
            mask = p > m_probability
            m_arg[mask], m_probability[mask] = i, p[mask]
        # 定义预测多样本的函数；
        if not get_raw_result:
            return np.array([self.label_dict[arg] for arg in m_arg])
        return m_probability

    def _transfer_x(self, x):
        return x
