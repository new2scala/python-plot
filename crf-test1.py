# https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite

print(sklearn.__version__)

import load_data as ld
train_sents, test_sents = ld.load_aff()

print(train_sents[0])

f1 = ld.sent2Feats(train_sents[0])[0]
print(f1)


# x_test = [ld.sent2Feats(s) for s in test_sents]
# y_test = [ld.sent2Labels(s) for s in test_sents]

def train_model(model_name):
    x_train = [ld.sent2Feats(s) for s in train_sents]
    y_train = [ld.sent2Labels(s) for s in train_sents]

    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(x_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params(
        {
            'c1': 1.0,
            'c2': 1e-3,
            'max_iterations': 50,
            'feature.possible_transitions': True
        }
    )

    print(trainer.params())
    trainer.train(model_name)

    print(trainer.logparser.last_iteration)

    print (len(trainer.logparser.iterations), trainer.logparser.iterations[-1])

#model_name = 'conll2003-en.crfsuite'
model_name = 'aff1.crfsuite'
#train_model(model_name)

tagger = pycrfsuite.Tagger()
tagger.open(model_name)

def parse_tags(tags):
    d = { }
    for i, tag in enumerate(tags):
        if (tag.startswith("B-")):
            t = tag[2:]
            d[i] = t
    return d


for example_sent in test_sents:
    tokens = ld.sent2Tokens(example_sent)
    tags = tagger.tag(ld.sent2Feats(example_sent))
    dict = parse_tags(tags)

    print(' '.join(tokens), end='\n')
    print("Predicted:", ' '.join(tags))
    s = []
    for i, t in enumerate(tokens):
        if (i in dict):
            s.append("(%s)" % dict[i])
        s.append(t)
    print(' '.join(s), end='\n')

    print("\n")
    #print("Correct:  ", ' '.join(ld.sent2Labels(example_sent)))