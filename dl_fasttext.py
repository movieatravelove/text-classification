import utils
import random
import fasttext
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split

# 1.处理数据
texts, labels = utils.load_dataset()
random.seed(0)
random.shuffle(texts)
random.seed(0)
random.shuffle(labels)
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.05, stratify=labels,
                                                                      random_state=0)
label_encoder = preprocessing.LabelEncoder()
labels_train = label_encoder.fit_transform(labels_train)
labels_test = label_encoder.transform(labels_test)

with open('data/fasttext.train.txt', 'w') as f:
    for i in range(len(texts_train)):
        f.write('%s __label__%d\n' % (texts_train[i], labels_train[i]))

with open('data/fasttext.test.txt', 'w') as f:
    for i in range(len(texts_test)):
        f.write('%s __label__%d\n' % (texts_test[i],labels_test[i]))

# 2.训练模型
model = fasttext.train_supervised('data/fasttext.train.txt',epoch=10)
print(model.words)
print(model.labels)


# 文件操作
# model.save_model("data/model_filename.bin")
# model = fasttext.load_model("data/model_filename.bin")
texts_test, labels_test = [], []
with open('data/fasttext.test.txt', 'r') as f:
    for line in f:
        *text, label = line.strip().split(' ')
        text = ' '.join(text)
        texts_test.append(text)
        labels_test.append(label)

# 测试模型
label_encoder = preprocessing.LabelEncoder()
labels_test = label_encoder.fit_transform(labels_test)
predits = list(zip(*(model.predict(texts_test)[0])))[0]
predits = label_encoder.transform(predits)

score = metrics.f1_score(predits, labels_test, average='weighted')
print('weighted f1-score : %.03f' % score)
