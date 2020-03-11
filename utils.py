import jieba

from typing import List, Tuple, Dict
from sklearn import metrics


def load_dataset(filepath: str = 'data/头条分类数据.txt', sample: bool or int = False) -> Tuple:
    texts, labels = [], []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if sample and i == sample:
                break
            _, _, label, text, _ = line.split('_!_')
            text = ' '.join(jieba.cut(text))
            texts.append(text)
            labels.append(label)
    return texts, labels


def apply(instance, train, test):
    """ 对train和test分别处理"""
    train = instance.fit_transform(train)
    test = instance.transform(test)
    return train, test


class ModelTest():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test

    def eval(self, classifier):
        """测试模型"""
        classifier.fit(self.X_train, self.y_train)
        predictions = classifier.predict(self.X_test)

        score = metrics.f1_score(predictions, self.y_test, average='weighted')

        print('weighted f1-score : %.03f' % score)

    def apply(self, instance):
        """ 对train和test分别处理"""
        self.X_train = instance.fit_transform(self.X_train)
        self.X_test = instance.transform(self.X_test)
