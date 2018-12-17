# Relevance-HTML

## Installation
```
python3 -m pip install --index-url https://test.pypi.org/simple/ relevance_html
```

## Algoritm
### Feature Selection:
  Select the text correspoding to html tag. Calculate tf-idf value by different tags and the tf value based on the words which have a similarity with a given topic higher than 0.8.
  
### Train Feature weight:
  Use linear regression to train the weight of features.
