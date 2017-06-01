# https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite

print(sklearn.__version__)

fids = nltk.corpus.conll2002.fileids()
print(fids)

train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents =  list(nltk.corpus.conll2002.iob_sents('esp.testb'))

print(train_sents[0])

def word2Feat(sent, i):
    word = sent[i][0]
    posttag = sent[i][1]
    feats = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'posttag=' + posttag,
        'posttag[:2]=' + posttag[:2],
    ]

    if i > 0:
        w1 = sent[i-1][0]
        pos1 = sent[i-1][1]
        feats.extend([
            '-1:word.lower=' + w1.lower(),
            '-1:word.istitle=%s' % w1.istitle(),
            '-1:word.isupper=%s' % w1.isupper(),
            '-1:postag=' + pos1,
            '-1:postag[:2]=' + pos1[:2],
        ])
    else:
        feats.append('BOS')

    if i < len(sent)-1:
        w1 = sent[i+1][0]
        pos1 = sent[i+1][1]
        feats.extend([
            '+1:word.lower=' + w1.lower(),
            '+1:word.istitle=%s' % w1.istitle(),
            '+1:word.isupper=%s' % w1.isupper(),
            '+1:postag=' + pos1,
            '+1:postag[:2]=' + pos1[:2],
        ])
    else:
        feats.append('EOS')

    return feats

def sent2Feats(sent):
    return [word2Feat(sent, i) for i in range(len(sent))]

def sent2Labels(sent):
    return [label for token, posttag, label in sent]

def sent2Tokens(sent):
    return [token for token, posttag, label in sent]

f1 = sent2Feats(train_sents[0])[0]
print(f1)


x_test = [sent2Feats(s) for s in test_sents]
y_test = [sent2Labels(s) for s in test_sents]

model_name = 'conll2002-esp.crfsuite'
def train_model(model_name):
    x_train = [sent2Feats(s) for s in train_sents]
    y_train = [sent2Labels(s) for s in train_sents]

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

tagger = pycrfsuite.Tagger()
tagger.open(model_name)
example_sent = test_sents[3]

print(' '.join(sent2Tokens(example_sent)), end='\n\n')

print("Predicted:", ' '.join(tagger.tag(sent2Feats(example_sent))))
print("Correct:  ", ' '.join(sent2Labels(example_sent)))