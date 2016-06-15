import re, math, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier

def is_url(texto):
  ocorrencias = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', texto)
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

def evaluate_features(feature_select):
  #reading pre-labeled input and splitting into lines
  posSentences = open('polarityData\\rt-polaritydata\\rt-polarity-pos.txt', 'r')
  negSentences = open('polarityData\\rt-polaritydata\\rt-polarity-neg.txt', 'r')
  posSentences = re.split(r'\n', posSentences.read())
  negSentences = re.split(r'\n', negSentences.read())

  posFeatures = []
  negFeatures = []
  #http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
  #breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
  for i in posSentences:
    posWords = re.findall(r"[\w']+|[.,!?;]", i)
    posWords = [feature_select(posWords), 'pos']
    posFeatures.append(posWords)
  for i in negSentences:
    negWords = re.findall(r"[\w']+|[.,!?;]", i)
    negWords = [feature_select(negWords), 'neg']
    negFeatures.append(negWords)

def make_full_dict(words):
  return dict([(word, True) for word in words])

with open("treinamento.csv", "r") as f:
  dicionario = []

  for line in f:
    campos = line.split(",")
    texto = campos[5]

    palavras = texto.split(" ")

    for palavra in palavras:
      palavra = remove_pontuacao(palavra).strip()

      if len(palavra) < 1:
        continue

      if palavra[0] == "@":
        continue

      if is_url(palavra):
        continue

      palavra = " !".join(palavra.split("!"))

      palavra = palavra.split(" ")

      if len(palavra) > 1:
        for subpalavra in palavra:
          dicionario.append(subpalavra)
      else:
        dicionario.append(palavra)

print 'using all words as features'
evaluate_features(make_full_dict)

print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
classifier.show_most_informative_features(10)
