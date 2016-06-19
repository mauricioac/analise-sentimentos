# coding=utf-8

import sys
import re, math, collections, itertools
import nltk, nltk.classify.util
from nltk.metrics import *
from nltk.classify import NaiveBayesClassifier
import csv
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import random

def is_url(texto):
  ocorrencias = re.findall('http[s]?:[/]*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', texto)
  return len(ocorrencias) > 0

def remove_pontuacao(palavra):
  return re.sub('[\.,;\\\/"\']', '', palavra)

def is_emoji(palavra):
  ocorrencias = re.findall(':[a-zA-Z]+:', palavra)
  return len(ocorrencias) > 0

def is_emoticon(palavra):
  emoticons_pos = [
    ":D",
    ":)",
    "=)",
    ";)",
    ":p",
    ":P",
    "*-*",
    "*.*",
    ";p"
  ]

  emoticons_neg = [
    ":(",
    "=(",
    ":'(",
    "D:",
    "D=",
    ":x",
    ":X",
    ">.<",
    "x.x",
    "T.T",
    ":/",
    ":s",
    ":S",
    ":@"
  ]

  emoticons_neu = [
    ":|",
    "o.O",
    "O.o",
    ":o",
    ":O"
  ]

  if palavra in emoticons_pos:
    return 1

  if palavra in emoticons_neg:
    return -1

  return 0

def make_full_dict(words):
  return dict([(word, True) for word in words])

def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

def substitui_abreviacoes_internet(palavra):
    dic = [
        ("q", "que"),
        ("pq", "porque"),
        ("tbm", "também"),
        ("tb", "também"),
        ("d", "de"),
        ("vc", "você"),
        ("td", "tudo"),
        ("cm", "com"),
        ("msc", "música"),
        ("cmg", "comigo"),
        ("s", "sim"),
        ("si", "sim"),
        ("naum", "não"),
        ("n", "não"),
        ("ñ", "não"),
        ("facul", "faculdade")
    ]

    encontrou = [item[1] for item in dic if item[0] == palavra]

    if len(encontrou) == 0:
        return palavra

    return encontrou[0]

def pre_processa_texto(texto):
  novo_texto = []

  for palavra in texto.split(" "):
    palavra = remove_pontuacao(palavra).strip()

    if len(palavra) < 1:
      continue

    if palavra[0] == "@":
      continue

    if is_url(palavra) or is_emoticon(palavra) or is_emoji(palavra):
      continue

    t = substitui_abreviacoes_internet(palavra)

    novo_texto.append(t)

  return " ".join(novo_texto)

tweets_negativos = []
tweets_positivos = []

parametros = len(sys.argv)

entradaTreinamento = sys.argv[0]
entradaTestes = parametros == 1 ? False : sys.argv[0]

with open(entradaTreinamento, "r") as f:
  dicionario = []

  leitor = csv.reader(f, delimiter=',', quotechar='"')

  for line in leitor:
    texto = line[5]
    polaridade = line[6]

    texto = pre_processa_texto(texto)

    if polaridade == "-1":
      tweets_negativos.append(texto)
    else:
      tweets_positivos.append(texto)

posFeatures = []
negFeatures = []

for tweet in tweets_positivos:
  posFeatures.append([make_full_dict(tweet), 'pos'])
for tweet in tweets_negativos:
  negFeatures.append([make_full_dict(tweet.split(" ")), 'neg'])

random.shuffle(posFeatures)
random.shuffle(negFeatures)

if entradaTestes:

else:
posCutoff = int(math.floor(len(posFeatures)*3/4))
negCutoff = int(math.floor(len(negFeatures)*3/4))
trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

classifier = NaiveBayesClassifier.train(trainFeatures)

#initiates referenceSets and testSets
referenceSets = collections.defaultdict(set)
testSets = collections.defaultdict(set)

#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
for i, (features, label) in enumerate(testFeatures):
  referenceSets[label].add(i)
  predicted = classifier.classify(features)
  testSets[predicted].add(i)

print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
# print 'accuracy:', accuracy(classifier, testFeatures)
print 'pos precision:', precision(referenceSets['pos'], testSets['pos'])
print 'pos recall:', recall(referenceSets['pos'], testSets['pos'])
print 'pos F-measure:', f_measure(referenceSets['pos'], testSets['pos'])
print 'neg precision:', precision(referenceSets['neg'], testSets['neg'])
print 'neg recall:', recall(referenceSets['neg'], testSets['neg'])
print 'neg F-measure:', f_measure(referenceSets['neg'], testSets['neg'])
classifier.show_most_informative_features(10)
