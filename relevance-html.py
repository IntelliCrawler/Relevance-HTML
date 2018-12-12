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

def get_features(d, topic):
    fea = {}
    contain_topic = [False]
    for k, v in d.items():
        fea[k] = get_tf(v, topic, contain_topic)
    return (fea, contain_topic[0])

def get_tfidf(dic, topic):
    num_doc_within_topic = 0
    html_tf = []
    for d in dic:
        fea, contain_topic = get_features(d, topic)
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

# read train and label
data = open('train.txt', 'r').read().split('\n')
train = []
label = []
for d in data:
    d_list = d.split()
    ret = extract_text_by_tags(d_list[1])
    if ret:
        train.append(ret)
        label.append(d_list[0])

# Get raw features of train
raw_fea = get_tfidf(train, "university")

# Get complete features of train
tag_list = ['title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table']
train_fea = []
for r in raw_fea:
    fea = np.zeros(11)
    for k, v in r.items():
        fea[tag_list.index(k)] = v
    train_fea.append(fea)

# X = train_fea Y = label
