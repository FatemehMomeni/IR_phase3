from __future__ import unicode_literals
from hazm import *
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

path = 'D:/Mine/computerUI/terms/term8/data mining/projects/project_final/Final_Dataset'

# reading folders in project's folder directory
test = os.listdir(path + '/Test')
train = os.listdir(path + '/Train')

# reading files in Test and Train directories
test_files = dict()  # a dictionary with topics as key and text files in that topic as value
train_files = dict()
for folder in test:
    test_files[folder] = os.listdir(path + '/Test/' + folder)
for folder in train:
    train_files[folder] = os.listdir(path + '/Train/' + folder)

# reading content of text files
train_topic = dict()  # a dictionary with topics as key and content of text files in that topic as values
for topic in train_files:
    train_topic[topic] = []
    for text in train_files[topic]:
        file = open(path + '/Train/' + topic + '/' + text, 'r', encoding='utf-8-sig')
        train_topic[topic].append(file.read())
        file.close()

test_topic = dict()
for topic in test_files:
    test_topic[topic] = []
    for text in test_files[topic]:
        file = open(path + '/Test/' + topic + '/' + text, 'r', encoding='utf-8-sig')
        test_topic[topic].append(file.read())
        file.close()

# reading stop words file
s_file = open(path + '/stopWords.txt', 'r', encoding='utf-8-sig')
stops = s_file.read()
s_file.close()
stops = (stops.replace("\u200c", ' ')).split()
digits = stops[-12: -2]


normalizer = Normalizer()
vectorizer = CountVectorizer(encoding='utf-8-sig', stop_words=stops)

# normalize
for topic in train_topic:
    for content in range(len(train_topic[topic])):
        train_topic[topic][content] = normalizer.normalize(train_topic[topic][content])

for topic in test_topic:
    for content in range(len(test_topic[topic])):
        test_topic[topic][content] = normalizer.normalize(test_topic[topic][content])

# replacing nim_faseleh to space
for topic in train_topic:
    for content in range(len(train_topic[topic])):
        train_topic[topic][content] = train_topic[topic][content].replace("\u200c", ' ')
        train_topic[topic][content] = train_topic[topic][content].replace('amp', '')
        train_topic[topic][content] = train_topic[topic][content].replace('nbsp', '')

for topic in test_topic:
    for content in range(len(test_topic[topic])):
        test_topic[topic][content] = test_topic[topic][content].replace("\u200c", ' ')
        test_topic[topic][content] = test_topic[topic][content].replace('amp', '')
        test_topic[topic][content] = test_topic[topic][content].replace('nbsp', '')

# assigning all train and test texts in two lists
train_corpus = list()
for topic in train_topic:
    for content in range(len(train_topic[topic])):
        train_corpus.append(train_topic[topic][content])

test_corpus = list()
for topic in test_topic:
    for content in range(len(test_topic[topic])):
        test_corpus.append(test_topic[topic][content])

corpus = train_corpus.copy()
for t in test_corpus:
    corpus.append(t)

train_matrix = vectorizer.fit_transform(train_corpus)
train_matrix = train_matrix.toarray()
train_tokens = vectorizer.get_feature_names()

test_matrix = vectorizer.fit_transform(test_corpus)
test_matrix = test_matrix.toarray()
test_tokens = vectorizer.get_feature_names()

corpus_matrix = vectorizer.fit_transform(corpus)
corpus_matrix = corpus_matrix.toarray()
corpus_tokens = vectorizer.get_feature_names()

# removing numbers and special tokens
tokens = list()
for token in corpus_tokens:
    if token[0] not in digits:
        tokens.append(token)

train_topic_list = list(train_topic.keys())
Xtrain = [[] for i in range(56)]
Ytrain = list()
topic_index = -1
for i in range(len(train_matrix)):
    if i % 8 == 0:
        topic_index += 1
    for ft in range(len(tokens)):
        if tokens[ft] in train_tokens:
            Xtrain[i].append(train_matrix[i][train_tokens.index(tokens[ft])])
        else:
            Xtrain[i].append(0)
    Ytrain.append(train_topic_list[topic_index])


test_topic_list = list(test_topic.keys())
Xtest = [[] for i in range(14)]
Ytest = list()
topic_index = -1
for i in range(len(test_matrix)):
    if i % 2 == 0:
        topic_index += 1
    for ft in range(len(tokens)):
        if tokens[ft] in test_tokens:
            Xtest[i].append(test_matrix[i][test_tokens.index(tokens[ft])])
        else:
            Xtest[i].append(0)
    Ytest.append(test_topic_list[topic_index])


# naive bayes
knn_model = KNeighborsClassifier()
knn_model.fit(Xtrain, Ytrain)
print('KNN Accuracy on train: {:.2f}'.format(knn_model.score(Xtrain, Ytrain)))
print('KNN Accuracy on test: {:.2f}'.format(knn_model.score(Xtest, Ytest)))
"""print('Gaussian Naive_Bayes Confusion matrix on test:')
print(confusion_matrix(Ytest, gnb_model.predict(Xtest)))
print('Gaussian Naive_Bayes F1score on test: {:.2f}'.format(f1_score(Ytest, gnb_model.predict(Xtest), average='macro')))
print('Gaussian Naive_Bayes Precision on test: {:.2f}'.format(precision_score(Ytest, gnb_model.predict(Xtest), average='macro')))
print('Gaussian Naive_Bayes Recall on test: {:.2f}'.format(recall_score(Ytest, gnb_model.predict(Xtest), average='macro')))"""
train_labels = knn_model.predict(Xtrain)
test_labels = knn_model.predict(Xtest)


counter = 0
for doc in train_topic:
    train_topic[doc] = {'text': train_topic[doc], 'topic': train_labels[counter]}
    counter += 1

counter = 0
for doc in test_topic:
    test_topic[doc] = {'text': test_topic[doc], 'topic': test_labels[counter]}

print(test_topic[v] for v in test_topic)
#json_train_labels = json.dumps(v for v in train_topic.values())
#print(json_train_labels)
