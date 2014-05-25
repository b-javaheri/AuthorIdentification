import numpy as np
import glob
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
 
path = r"E:/University/Term.8/ArtificialIntelligence/python/chapters"

files = sorted(glob.glob(os.path.join(path, "chapter*.txt")))
chapters = []
for fn in files:
    with open(fn) as f:
        chapters.append(f.read().replace('\n', ' '))
all_text = ' '.join(chapters)


num_chapters = len(chapters)
fvs_lexical = np.zeros((len(chapters), 3), np.float64)
fvs_punct = np.zeros((len(chapters), 3), np.float64)
for e, ch_text in enumerate(chapters):

    tokens = nltk.word_tokenize(ch_text.lower())
    words = word_tokenizer.tokenize(ch_text.lower())
    sentences = sentence_tokenizer.tokenize(ch_text)
    vocab = set(words)
    words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                   for s in sentences])

    fvs_lexical[e, 0] = words_per_sentence.mean()
    fvs_lexical[e, 1] = words_per_sentence.std()
    fvs_lexical[e, 2] = len(vocab) / float(len(words))
 
    fvs_punct[e, 0] = tokens.count(',') / float(len(sentences))
    fvs_punct[e, 1] = tokens.count(';') / float(len(sentences))
    fvs_punct[e, 2] = tokens.count(':') / float(len(sentences))

fvs_lexical = whiten(fvs_lexical)
fvs_punct = whiten(fvs_punct)


NUM_TOP_WORDS = 10
all_tokens = nltk.word_tokenize(all_text)
fdist = nltk.FreqDist(all_tokens)
vocab = fdist.keys()[:NUM_TOP_WORDS]
vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=nltk.word_tokenize)
fvs_bow = vectorizer.fit_transform(chapters).toarray().astype(np.float64)
fvs_bow /= np.c_[np.apply_along_axis(np.linalg.norm, 1, fvs_bow)]

def token_to_pos(ch):
    tokens = nltk.word_tokenize(ch)
    return [p[1] for p in nltk.pos_tag(tokens)]
chapters_pos = [token_to_pos(ch) for ch in chapters]

pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
fvs_syntax = np.array([[ch.count(pos) for pos in pos_list]
                       for ch in chapters_pos]).astype(np.float64)
 
fvs_syntax /= np.c_[np.array([len(ch) for ch in chapters_pos])]


def PredictAuthors(fvs):
    km = KMeans(n_clusters=2, init='k-means++', n_init=10, verbose=0)
    km.fit(fvs)
 
    return km

###
