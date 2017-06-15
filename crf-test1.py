# https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite

print(sklearn.__version__)

import load_data as ld
train_sents, test_sents = ld.load_aff('usa_train.pkl', 'data/usa-test-converted.txt')

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
model_name = 'usa-model.crfsuite'
train_model(model_name)

tagger = pycrfsuite.Tagger()
tagger.open(model_name)

def parse_tags(tags):
    d = { }
    for i, tag in enumerate(tags):
        if (tag.startswith("B-")):
            t = tag[2:]
            d[i] = t
    return d

def tag_sent(sent, tags):
    tagged_sent = []
    for i, token in enumerate(sent):
        tagged_sent.append([sent[i][0], sent[i][1], tags[i]])
    return tagged_sent

def write_test_results(test_results, out_file):
    with open(out_file, 'w') as f:
        for sent in test_results:
            for t in sent:
                str = ' '.join(t) + '\n'
                f.write(str)
            f.write('\n')

def calc_prec(tags, exp_tags):
    if (len(tags) != len(exp_tags)):
        print("different tag list length")
        return 0
    else:
        correct = 0
        for i, t in enumerate(tags):
            if (t == exp_tags[i]):
                correct = correct+1
        return correct

results = []
all_tag_count = 0
correct_tag_count = 0
for example_sent in test_sents:
    tokens = ld.sent2Tokens(example_sent)
    tags = tagger.tag(ld.sent2Feats(example_sent))
    dict = parse_tags(tags)

    print(' '.join(tokens), end='\n')
    print("Predicted:", ' '.join(tags))
    exp_tags = ld.sent2Labels(example_sent)
    print(" Expected:", ' '.join(exp_tags))
    correct_count = calc_prec(tags, exp_tags)
    all_tag_count = all_tag_count+len(tags)
    correct_tag_count = correct_tag_count+correct_count
    print("%.2f%%" % (correct_count/len(tags)*100))
    s = []
    for i, t in enumerate(tokens):
        if (i in dict):
            s.append("(%s)" % dict[i])
        s.append(t)
    print(' '.join(s), end='\n')

    results.append(tag_sent(example_sent,tags))
    print("\n")

print("Overall precision: %.2f%%" % (correct_tag_count/all_tag_count*100))


#write_test_results(results, '/media/sf_work/aff-data/test-2-result.txt')
#print(len(results))
    #print("Correct:  ", ' '.join(ld.sent2Labels(example_sent)))