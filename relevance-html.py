import math
import numpy as np
import re
import gensim
import string
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
path = "GoogleNews-vectors-negative300.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)


# Get tf value for each tag
def get_tf(s, topic, contain_topic):
    #get tf
    rele_words = 0
    l = s.split()
    for w in l:
        sim = 0
        try:
            sim = model.similarity(topic, w)
        except Exception:
            sim = 0
        if sim >= 0.8:
            rele_words += 1
    tf = rele_words / len(l)
    #update idf
    if tf > 0 or contain_topic[0]:
        contain_topic[0] = True
    return tf

# From dictionary of tag:text to tag:tf
def text_to_tf(d, topic):
    fea = {}
    contain_topic = [False]
    for k, v in d.items():
        fea[k] = get_tf(v, topic, contain_topic)
    return (fea, contain_topic[0])

# From dictionary of tag:text to tag:tfidf
def tf_to_tfidf(dic, topic):
    num_doc_within_topic = 0
    html_tf = []
    for d in dic:
        fea, contain_topic = text_to_tf(d, topic)
        if contain_topic:
            num_doc_within_topic += 1
        html_tf.append(fea)
    #idf
    html_idf = math.log(len(dic) / num_doc_within_topic + 1)
    #Mulitiple idf to tf
    for tf in html_tf:
        for k, v in tf.items():
            tf[k] = v * html_idf
    return html_tf

# From url to dictionary of tag:text
def extract_text_by_tags(url,
                         tags=[
                             'title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p',
                             'ul', 'ol', 'table'
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
def complete_features(d, topic):
    # Get raw features of train
    raw_fea = tf_to_tfidf(d, topic)
    # Get complete features of train
    tag_list = ['title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table']
    res_fea = []
    for r in raw_fea:
        fea = np.zeros(11)
        for k, v in r.items():
            fea[tag_list.index(k)] = v
        res_fea.append(fea)
    return res_fea

# filename [label url]
# return the beta of linear regression
def train(filename, topic):
    # read train tag:text and label
    topic = "university"
    data = open('train.txt', 'r').read().split('\n')
    train_text = []
    label = []
    for d in data:
        d_list = d.split()
        ret = extract_text_by_tags(d_list[1])
        if ret:
            train_text.append(ret)
            label.append(d_list[0])
    # Get train_fea with full features of 11
    train_fea = complete_features(train_text, topic)
    # Train the train_fea and label by linear regression
    X = np.array(train_fea,dtype="float64")
    add_one = np.ones((len(X), 1))
    X = np.column_stack((X, add_one))
    # add bias
    Y = np.array(label,dtype="float64")
    # X = train_fea Y = label
    beta = np.dot(np.linalg.pinv(X), Y)
    return beta

# filename [label url]
# return the prediction label and validation label
def test(filename, topic, w):
    # read test tag:text and label
    test = open(filename, 'r').read().split('\n')[0:30]
    to_test = []
    test_label = []
    for d in test:
        d_list = d.split()
        ret = extract_text_by_tags(d_list[1])
        if ret:
            to_test.append(ret)
            test_label.append(d_list[0])
    # Get test_fea with full features of 11
    test_fea = complete_features(to_test, topic)
    # Get the prediction label of test
    test_res = np.dot(np.column_stack((test_fea, np.ones((len(test_fea), 1)))), w)
    # 0.0-0.33 irrelevant 0.33-0.66 weak relevant 0.66-1.0 strong relevant
    res = np.zeros(len(test_res))
    for i in range(0, len(test_res)):
        if test_res[i] <= 0.33:
            res[i] = 0
        elif test_res[i] >= 0.66:
            res[i] = 1
        else:
            res[i] = 0.5
    return (res, np.array(test_label,dtype="float64"))



beta = train('train.txt', 'university')
print("Model: ")
print(beta)
res = test('train.txt', 'university', beta)
print("Result: prediction, validation")
print(res)