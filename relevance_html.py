import math
import os
import re
import string
from collections import defaultdict

import numpy as np
import requests
from bs4 import BeautifulSoup

import gensim


class RelevantHTML:
    def __init__(self, topic, model=None):
        self.topic = topic
        if model:
            self.model = model
        else:
            path = 'model/GoogleNews-vectors-negative300.bin'
            if os.path.isfile(path):
                self.model = gensim.models.KeyedVectors.load_word2vec_format(
                    path, binary=True)
            else:
                raise FileNotFoundError(
                    f"No such file: '{path}'\n"
                    "Pre-trained word and phrase vectors not found. "
                    "You can download the file at "
                    "https://code.google.com/archive/p/word2vec/.")

    # Get tf value for each tag
    def get_tf(self, s, contain_topic):
        # get tf
        rele_words = 0
        lst = s.split()
        for w in lst:
            sim = 0
            try:
                sim = self.model.similarity(self.topic, w)
            except Exception:
                sim = 0
            if sim >= 0.8:
                rele_words += 1
        tf = rele_words / len(lst)
        # update idf
        if tf > 0 or contain_topic[0]:
            contain_topic[0] = True
        return tf

    # From dictionary of tag:text to tag:tf
    def text_to_tf(self, d):
        fea = {}
        contain_topic = [False]
        for k, v in d.items():
            fea[k] = self.get_tf(v, contain_topic)
        return fea, contain_topic[0]

    # From dictionary of tag:text to tag:tfidf
    def tf_to_tfidf(self, dic):
        num_doc_within_topic = 0
        html_tf = []
        for d in dic:
            fea, contain_topic = self.text_to_tf(d)
            if contain_topic:
                num_doc_within_topic += 1
            html_tf.append(fea)
        # idf
        html_idf = math.log(len(dic) / num_doc_within_topic + 1)
        # Mulitiple idf to tf
        for tf in html_tf:
            for k, v in tf.items():
                tf[k] = v * html_idf
        return html_tf

    # From url to dictionary of tag:text
    def extract_text_by_tags(self,
                             url,
                             tags=[
                                 'title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                                 'p', 'ul', 'ol', 'table'
                             ]):
        try:
            r = requests.get(url)
            r.raise_for_status()
        except Exception:
            return None
        soup = BeautifulSoup(r.text, 'html.parser')
        elements = defaultdict(list)
        for tag in tags:
            for elem in soup.find_all(tag):
                stripped = elem.text.translate(
                    str.maketrans(
                        str.maketrans(dict.fromkeys(string.punctuation))))
                stripped = re.sub(r'\s+', ' ', stripped).strip()
                if stripped:
                    elements[tag].append(stripped.lower())
        for tag, elem in elements.items():
            elements[tag] = ' '.join(elements[tag])
        return elements

    # Get full features of 11
    def complete_features(self, d):
        # Get raw features of train
        raw_fea = self.tf_to_tfidf(d)
        # Get complete features of train
        tag_list = [
            'title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol',
            'table'
        ]
        res_fea = []
        for r in raw_fea:
            fea = np.zeros(11)
            for k, v in r.items():
                fea[tag_list.index(k)] = v
            res_fea.append(fea)
        return res_fea

    # filename [label url]
    # return the beta of linear regression
    def fit(self, filename):
        # read train tag:text and label
        data = open(filename, 'r').read().split('\n')
        train_text = []
        label = []
        for d in data:
            d_list = d.split()
            ret = self.extract_text_by_tags(d_list[1])
            if ret:
                train_text.append(ret)
                label.append(d_list[0])
        # Get train_fea with full features of 11
        train_fea = self.complete_features(train_text)
        # Train the train_fea and label by linear regression
        X = np.array(train_fea, dtype="float64")
        add_one = np.ones((len(X), 1))
        X = np.column_stack((X, add_one))
        # add bias
        Y = np.array(label, dtype="float64")
        # X = train_fea Y = label
        self.weights = np.dot(np.linalg.pinv(X), Y)

    # filename [label url]
    # return the prediction label and validation label
    def valid(self, filename):
        # read test tag:text and label
        test = open(filename, 'r').read().split('\n')
        to_test = []
        test_label = []
        for d in test:
            d_list = d.split()
            ret = self.extract_text_by_tags(d_list[1])
            if ret:
                to_test.append(ret)
                test_label.append(d_list[0])
        self.test_actual = np.array(test_label, dtype='float64')
        # Get test_fea with full features of 11
        test_fea = self.complete_features(to_test)
        # Get the prediction label of test
        test_res = np.dot(
            np.column_stack((test_fea, np.ones((len(test_fea), 1)))),
            self.weights)
        # 0.0-0.5 irrelevant 0.5-1.0 relevant
        res = np.zeros(len(test_res))
        for i in range(0, len(test_res)):
            if test_res[i] <= 0.5:
                res[i] = 0
            else:
                res[i] = 1
        self.test_predicted = res

    def accuracy(self):
        base = 0
        for i in range(len(self.test_predicted)):
            if self.test_predicted[i] == self.test_actual[i]:
                base += 1
        acc = base / len(self.test_predicted)
        return acc

    def recall(self):
        irre_lab = 0
        irre_pre = 0
        re_lab = 0
        re_pre = 0
        for i in range(len(self.test_actual)):
            if self.test_actual[i] == 0:
                irre_lab += 1
                if self.test_predicted[i] == 0:
                    irre_pre += 1
            else:
                re_lab += 1
                if self.test_predicted[i] == 1:
                    re_pre += 1
        irre_recall = irre_pre / irre_lab
        re_recall = re_pre / re_lab
        return re_recall, irre_recall

    def precision(self):
        irre_lab = 0
        irre_pre = 0
        re_lab = 0
        re_pre = 0
        for i in range(len(self.test_predicted)):
            if self.test_predicted[i] == 0:
                irre_pre += 1
                if self.test_actual[i] == 0:
                    irre_lab += 1
            else:
                re_pre += 1
                if self.test_actual[i] == 1:
                    re_lab += 1
        irre_precision = irre_lab / irre_pre
        re_precision = re_lab / re_pre
        return re_precision, irre_precision

    def f1_score(self):
        re_recall, irre_recall = self.recall()
        re_precision, irre_precision = self.precision()
        re_f = 2 * re_recall * re_precision / (re_recall + re_precision)
        irre_f = 2 * irre_recall * irre_precision / (
            irre_recall + irre_precision)
        return re_f, irre_f

    # return the prediction label
    def predict(self, url):
        ret = self.extract_text_by_tags(url)
        if ret:
            fea = self.complete_features([ret])
            fea_res = np.dot(
                np.column_stack((fea, np.ones((len(fea), 1)))), self.weights)
            res = 0
            if fea_res[0] >= 0.5:
                res = 1
            return res
        else:
            return "Error: URL can not be reached."


def main():
    clf = RelevantHTML('science')
    clf.fit('data/science_train200.txt')
    clf.valid('data/science_test178.txt')
    print("Accuracy:")
    print(clf.accuracy())
    print("Recall: (Relevant, Irrelevant)")
    print(clf.recall())
    print("Precision: (Relevant, Irrelevant)")
    print(clf.precision())
    print("F1_Score: (Relevant, Irrelevant)")
    print(clf.f1_score())
    # lbl = clf.predict(
    #     'https://www.nydailynews.com/'
    #     'sd-no-schoolnews-parkuniversity-grads-20181015-story.html')
    # print('Relevant' if lbl == 1 else 'Irrelevant')


if __name__ == '__main__':
    main()
