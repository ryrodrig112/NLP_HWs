# HW1: NGram Models


```python
from mtg import finish_sentence
import nltk
import numpy as np
```


```python
corpus = nltk.word_tokenize(
    nltk.corpus.gutenberg.raw('austen-sense.txt').lower()
)
```

## Provided Test Case


```python
words = finish_sentence(
        ["she", "was", "not"],
        3,
        corpus,
        randomize=False,
    )

print(words)
    
```

    ['she', 'was', 'not', 'in', 'the', 'world', '.']


## Nondeterministic Behavior w/ austen-sense data


```python
for i in range(10):
    words = finish_sentence(
            ["she", "was", "not"],
            4,
            corpus,
            randomize=True,
        )
    print(words)
```

    ['she', 'was', 'not', 'suspected', 'of', 'any', 'extraordinary', 'interest', 'in', 'it']
    ['she', 'was', 'not', 'immediately', 'that', 'an', 'opportunity', 'of', 'endeavouring', 'to']
    ['she', 'was', 'not', 'aware', 'that', 'such', 'language', 'could', 'be', 'suffered']
    ['she', 'was', 'not', 'elinor', ',', 'appear', 'a', 'compliment', 'to', 'herself']
    ['she', 'was', 'not', 'in', 'mrs.', 'ferrars', "'", 'power', 'to', 'distress']
    ['she', 'was', 'not', 'suspected', 'of', 'any', 'extraordinary', 'interest', 'in', 'it']
    ['she', 'was', 'not', 'doomed', ',', 'however', ',', 'elinor', 'perceived', ',']
    ['she', 'was', 'not', 'elinor', ',', 'who', ',', 'though', 'still', 'unable']
    ['she', 'was', 'not', 'quite', 'herself', ',', 'and', 'was', 'always', 'sure']
    ['she', 'was', 'not', 'to', 'be', 'thought', 'of', ';', '--', 'and']


## Deterministic Behaviors with Simple Example


```python
corpus = ['I', 'am', 'a', 'big', 'big', 'cat', '.', 'I', 'am', 'not', 'a', 'big', 'dog', '.']
print(finish_sentence(["I"], 2, corpus))
print(finish_sentence(["I"], 3, corpus))
print(finish_sentence(["not"], 2, corpus))
```

    ['I', 'am', 'a', 'big', 'big', 'big', 'big', 'big', 'big', 'big']
    ['I', 'am', 'a', 'big', 'big', 'cat', '.']
    ['not', 'a', 'big', 'big', 'big', 'big', 'big', 'big', 'big', 'big']

