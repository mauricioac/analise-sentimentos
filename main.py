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

def classifica(trainFeatures, testFeatures):
  classifier = NaiveBayesClassifier.train(trainFeatures)

  #initiates referenceSets and testSets
  referenceSets = collections.defaultdict(set)
  testSets = collections.defaultdict(set)

  #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
  for i, (features, label) in enumerate(testFeatures):
    referenceSets[label].add(i)
    predicted = classifier.classify(features)
    testSets[predicted].add(i)

  return {
    "classifier": classifier,
    "ref": referenceSets,
    "test": testSets
  }

def classifica_arquivo_separado(features_positivas, features_negativas, testes):
  trainFeatures = features_negativas + features_positivas
  testFeatures = testes

  return classifica(trainFeatures, testFeatures)

def classifica_mesmo_arquivo(features_positivas, features_negativas):
  posCutoff = int(math.floor(len(features_positivas)*3/4))
  negCutoff = int(math.floor(len(features_negativas)*3/4))
  trainFeatures = features_positivas[:posCutoff] + features_negativas[:negCutoff]
  testFeatures = features_positivas[posCutoff:] + features_negativas[negCutoff:]

  return classifica(trainFeatures, testFeatures)

def is_url(texto):
  ocorrencias = re.findall('http[s]?:[/]*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', texto)
  return len(ocorrencias) > 0

def remove_pontuacao(palavra):
  return re.sub('[\.,;\\\/"\']', '', palavra)

def is_emoji(palavra):
  ocorrencias = re.findall(':[a-zA-Z]+:', palavra)
  return len(ocorrencias) > 0

def extrai_features(arquivo, campo_texto, campo_classe):
  with open(arquivo, "r") as f:
    leitor = csv.reader(f, delimiter=',', quotechar='"')

    for line in leitor:
      texto = line[campo_texto]
      polaridade = line[campo_classe]

      texto = pre_processa_texto(texto)

      if polaridade == "-1":
        tweets_negativos.append(texto)
      else:
        tweets_positivos.append(texto)

  posFeatures = []
  negFeatures = []

  for tweet in tweets_positivos:
    posFeatures.append([make_full_dict(tweet.split(" ")), 'pos'])
  for tweet in tweets_negativos:
    negFeatures.append([make_full_dict(tweet.split(" ")), 'neg'])

  random.shuffle(posFeatures)
  random.shuffle(negFeatures)

  return (posFeatures, negFeatures)

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
    "oO",
    "O.o",
    "Oo",
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

if parametros != 4 and parametros != 7:
  print "Número de parâmetros inválidos!"
  print "Você está utilizando o script 'analisa.sh' para executar?"
  print "-----------------------------------"
  print "Existe dois modos de executar o programa:"
  print "  1) Somente arquivo de treinamento"
  print "  2) Arquivos de treinamento e testes"
  print "\n----------------------------------"
  print "Abra o arquivo 'analisa.sh' e modifique os parâmetros para executar de forma correta e mais fácil o programa"
  print "Após configurar as variáveis de acordo, execute em um terminal:"
  print "  ./analisa.sh"
  sys.exit()

treinamento_arquivo = sys.argv[1]
treinamento_campo_texto = int(sys.argv[2])
treinamento_campo_classe = int(sys.argv[3])
testes_arquivo = False

if parametros == 7:
  testes_arquivo = sys.argv[4]
  testes_campo_texto = sys.argv[5]
  testes_campo_classe = sys.argv[6]

treinamento = extrai_features(treinamento_arquivo, treinamento_campo_texto, treinamento_campo_classe)

if testes_arquivo:
  teste = extrai_features(testes_arquivo, testes_campo_texto, testes_campo_classe)
  tmp = classifica_arquivo_separado(treinamento[0], treinamento[1], teste[0] + teste[1])
else:
  tmp = classifica_mesmo_arquivo(treinamento[0], treinamento[1])

classifier = tmp["classifier"]
referenceSets = tmp["ref"]
testSets = tmp["test"]

# print 'accuracy:', accuracy(classifier, testFeatures)
print 'pos precision:', precision(referenceSets['pos'], testSets['pos'])
print 'pos recall:', recall(referenceSets['pos'], testSets['pos'])
print 'pos F-measure:', f_measure(referenceSets['pos'], testSets['pos'])
print 'neg precision:', precision(referenceSets['neg'], testSets['neg'])
print 'neg recall:', recall(referenceSets['neg'], testSets['neg'])
print 'neg F-measure:', f_measure(referenceSets['neg'], testSets['neg'])
classifier.show_most_informative_features(10)
