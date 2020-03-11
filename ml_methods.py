import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from collections import OrderedDict
from sklearn import tree, svm, naive_bayes, neighbors, linear_model
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

import utils

texts, labels = utils.load_dataset(sample=True)
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels
# 划分训练集,测试集
X_train, X_test, y_train, y_test = train_test_split(trainDF['text'], trainDF['label'], test_size=0.05, stratify=labels,
                                                    random_state=0)

# 标签列处理
label_encoder = preprocessing.LabelEncoder()
y_train, y_test = utils.apply(label_encoder, y_train, y_test)

# 数据列处理
count_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train, X_test = utils.apply(count_vectorizer, X_train, X_test)
X_train, X_test = utils.apply(tfidf_transformer, X_train, X_test)

print("训练集大小:%s" % (str(X_train.shape)))
print("测试集大小:%s" % (str(X_test.shape)))

modeltest = utils.ModelTest(X_train, y_train, X_test, y_test)

# 属性降低维度
# from sklearn import decomposition
# svd=decomposition.TruncatedSVD(n_components=300)
# modeltest.apply(svd)
# normalizer = preprocessing.Normalizer(copy=False)
# modeltest.apply(normalizer)

# 各个模型的训练
# models = OrderedDict([
#     ('KNN', neighbors.KNeighborsClassifier()),
#     ('logistic回归', linear_model.LogisticRegression()),
#     ('svm', svm.SVC()),
#     ('朴素贝叶斯', naive_bayes.MultinomialNB()),
#     ('决策树', tree.DecisionTreeClassifier()),
#     ('决策树bagging', BaggingClassifier()),
#     ('随机森林', RandomForestClassifier()),
#     ('adaboost', AdaBoostClassifier()),
#     ('gbdt', GradientBoostingClassifier()),
#     ('xgboost', XGBClassifier()),
#
# ])
#
# for name, clf in models.items():
#     modeltest.eval(clf)

# # xgboost显示各个属性的重要性
# from xgboost import plot_importance
# from matplotlib import pyplot
# plot_importance(clf)
# pyplot.show()

# # 搜索最优超参数
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# model = XGBClassifier() #RandomizedSearchCV
# learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
# param_grid = dict(learning_rate=learning_rate)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
# grid_result = grid_search.fit(X_train, y_train)
#
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


# 主题模型
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=10,
                                max_iter=50,
                                learning_method='batch')
lda.fit(X_train)

# 打印每个主题下权重较高的10个词语
feature_names = count_vectorizer.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))

# 查看模型的收敛效果
lda.perplexity(X_train)
# 获取文档的主题
doc_topic_dist = lda.transform(X_test)