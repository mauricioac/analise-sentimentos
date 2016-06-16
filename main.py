import re, math, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
import csv

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

def make_full_dict(words):
  return dict([(word, True) for word in words])

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

    novo_texto.append(palavra)

  return " ".join(novo_texto)

tweets_negativos = []
tweets_positivos = []

with open("treinamento.csv", "r") as f:
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
#http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
#breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
for i in tweets_positivos:
  posWords = re.findall(r"[\w']+|[.,!?;]", i)
  posWords = [make_full_dict(posWords), 'pos']
  posFeatures.append(posWords)
for i in tweets_negativos:
  negWords = re.findall(r"[\w']+|[.,!?;]", i)
  negWords = [make_full_dict(negWords), 'neg']
  negFeatures.append(negWords)

#selects 3/4 of the features to be used for training and 1/4 to be used for testing
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
print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
classifier.show_most_informative_features(10)
