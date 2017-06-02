
import nltk
def load_conll2002():
    train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
    return train_sents, test_sents

import pickle
def load_conll2003():
    _train_data = pickle.load(open('conll2003_train.pkl', 'rb'))
    _train_data = _train_data[1:]
    removed_3rdcol = []
    for d in _train_data:
        removed = []
        for l in d:
            removed.append([l[0], l[1], l[3]])
        removed_3rdcol.append(removed)
    train_sents = removed_3rdcol[100:]
    test_sents = removed_3rdcol[:100]
    return train_sents, test_sents


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

