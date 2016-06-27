# coding=utf-8

# import __future__ import print_function, unicode_literals
import sys
import re, math, collections, itertools
import nltk, nltk.classify.util
from nltk.metrics import *
from nltk.classify import NaiveBayesClassifier
import csv
import random

verbos = []

def classifica(trainFeatures, testFeatures):
  classifier = NaiveBayesClassifier.train(trainFeatures)

  #initiates referenceSets and testSets
  referenceSets = collections.defaultdict(set)
  testSets = collections.defaultdict(set)
  vp = 0
  fp = 0
  vn = 0
  fn = 0


  #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
  for i, (features, label) in enumerate(testFeatures):
    referenceSets[label].add(i)
    predicted = classifier.classify(features)
    testSets[predicted].add(i)

    if label == "pos" and predicted == "pos":
      vp = vp + 1
    elif label == "pos" and predicted == "neg":
      fn = fn + 1
    elif label == "neg" and predicted == "neg":
      vn = vn + 1
    elif label == "neg" and predicted == "pos":
      fp = fp + 1

  print "fp: ", fp
  print "vp: ", vp
  print "fn: ", fn
  print "vn: ",  vn

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
  posCutoff = int( math.floor( len(features_positivas)*1/4 ) )
  negCutoff = int( math.floor( len(features_negativas)*1/4 ) )

  trainFeatures = features_positivas[:posCutoff] + features_negativas[:negCutoff]
  testFeatures = features_positivas[posCutoff:] + features_negativas[negCutoff:]

  return classifica(trainFeatures, testFeatures)

def is_url(texto):
  #facepalm
  ocorrencias = re.findall('http', texto)
  return len(ocorrencias) > 0

def remove_pontuacao(palavra):
  return re.sub('[\.,;?:+*&$=º~^\\\/"\'\(\)\[\]\{\}]\t', '', palavra)

def lematiza(palavra):
  sufixo_verbos = [
    ("fosse","ir"),
    ("desse", "dar"),
    ("usesse","or"),
    ("isesse","erer"),
    ("oubesse", "aber"),
    ("ouvesse","azar"),
    ("ouxesse","azar"),
    ("udesse","oder"),
    ("isesse","erer"),
    ("endo", "er"),
    ("indo", "ir"),
    ("ando", "ar"),
    ("emos", "er"),
    ("eis", "er"),
    ("eríamos", "er"),
    ("aríamos","ar"),
    ("iamos","ir"),
    ("eira","er"),
    ("esse", "er"),
    ("remos","er"),
    ("asse","ar"),
    ("isse","ir"),
    ("essem", "er"),
    ("assem","ar"),
    ("issem","ir"),
    ("arem","ar"),
    ("erem","er"),
    ("irem","ir"),
    ("ásseis","ar"),
    ("éssemos", "er"),
    ("a-lo", "ar"),
    ("á-lo-íamos", "ar"),
    ("ava","ar"),
    ("s", "")
  ]

  sufixo_outros = [
    ("zinhos", ""),
    ("zinho", ""),
    ("zinhas", ""),
    ("zinha", ""),
    ("zito", ""),
    ("zitos", ""),
    ("zinha", ""),
    ("zinha", "")
  ]

  for x in sufixo_verbos:
    if palavra.endswith(x[0]):
      nova_palavra = re.sub(x[0],x[1],palavra)
      
      if nova_palavra in verbos:
        return nova_palavra

  for x in sufixo_outros:
    if palavra.endswith(x[0]):
      nova_palavra = re.sub(x[0],x[1],palavra)
      print palavra, nova_palavra
      return nova_palavra

  return palavra

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
      if polaridade == "1":
        tweets_positivos.append(texto)
      else:
        tweets_negativos.append(texto)

  posFeatures = []
  negFeatures = []

  for tweet in tweets_positivos:
    palavras = tweet.split(" ")
    # print palavras
    posFeatures.append([prepara_features_classificador(palavras), 'pos'])
  # print "\n\n------------------------------\n\n"
  for tweet in tweets_negativos:
    palavras = tweet.split(" ")
    # print palavras
    negFeatures.append([prepara_features_classificador(palavras), 'neg'])

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
    ";p",
    ":^B",
    ":^D",
    ":^B",
    "=B",
    "=^B",
    "=^D",
    ":’)"
    ":’]",
    ":’}",
    ";]",
    ";}",
    ":-p",
    ":-P",
    ":-b",
    ":^p",
    ":^P",
    ":^b",
    "=P",
    "=p",
    ":P",
    ":p",
    ":b",
    "=b",
    "=^p",
    " =^P",
    "=^b",
    "\o/"
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
    ":@",
    ":|",
    "=|",
    ":-|",
    ">.<",
    "><",
    ">_<",
    ":o",
    ":0",
    "=O",
    ":@",
    "=@",
    ":^o",
    ":^@",
    "-.-",
    "-.-’",
    "-_-",
    "-_-’",
    ":x",
    "=X",
    "=#",
    ":-x",
    ":-@",
    ":-#",
    ":^x",
    ":^#",
    ":#",
    "D:",
    "D=",
    "D-:",
    "D^:",
    "D^=",
    ":(",
    ":[",
    ":{",
    ":o(",
    ":o[",
    ":^(",
    ":^[",
    ":^{",
    "=^(",
    "=^{",
    ">=(",
    ">=[",
    ">={",
    ">=(",
    ">:-{",
    ">:-[",
    ">:-(",
    ">=^[",
    ">:-(",
    ":-[",
    ":-(",
    "=(",
    "=[",
    "={",
    "=^[",
    ">:-=(",
    ">=[",
    ":’(",
    ":’[",
    ":’{",
    "=’{",
    "=’(",
    "=’[",
    "=/",
    ":/",
    "=$",
    "o.O",
    "O_o",
    "Oo",
    ":$:-{",
    ">:-{",
    ">=^(",
    ">=^{",
    ":o{"
  ]

  emoticons_neu = [
    ":|",
    "o.O",
    "oO",
    "O.o",
    "Oo",
    ":o",
    ":O",
    ":|",
    "=|",
    ":-|",
    ">.<",
    "><",
    ">_<",
    ":o",
    ":0",
    "=O",
    ":^@",
    "=@",
    ":^o",
    ":^@",
    "-.-",
    "-.-’",
    "-_-",
    "-_-’",
    ":^x",
    "=X",
    "=#",
    ":-x",
    ":-@",
    ":-#",
    ":^x",
    ":^#",
    ":#"
  ]

  if palavra in emoticons_pos:
    return 1

  if palavra in emoticons_neg:
    return -1

  return 0

def prepara_features_classificador(words):
  return dict([(word, True) for word in words])

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

def palavra_neutra(palavra):
  palavras = [
    "o",
    "a",
    "os",
    "ou",
    "ela",
    "ele",
    "elas",
    "eles",
    "outro",
    "outra",
    "outras",
    "outros",
    "para",
    "pelo",
    "veja",
    "você",
    "muito",
    "muita",
    "muitos",
    "enquanto",
    "ir",
    "irei",
    "irá",
    "iremos",
    "irás",
    "faz",
    "fazer",
    "fará",
    "farás",
    "esta",
    "acerca",
    "agora",
    "algmas",
    "alguns",
    "ali",
    "ambos",
    "antes",
    "apontar",
    "aquela",
    "aquelas",
    "aquele",
    "aqueles",
    "aqui",
    "atrás",
    "as",
    "de",
    "às",
    "das",
    "dos",
    "do",
    "da",
    "com",
    "sem",
    "no",
    "na",
    "nos",
    "nas",
    "em",
    "e",
    "um",
    "uma",
    "uns",
    "que",
    "num",
    "último",
    "é",
    "acerca",
    "agora",
    "algmas",
    "alguns",
    "ali",
    "ambos",
    "antes",
    "apontar",
    "aquela",
    "aquelas",
    "aquele",
    "aqueles",
    "aqui",
    "atrás",
    "bem",
    "bom",
    "cada",
    "caminho",
    "cima",
    "com",
    "como",
    "comprido",
    "conhecido",
    "corrente",
    "das",
    "debaixo",
    "dentro",
    "desde",
    "desligado",
    "deve",
    "devem",
    "deverá",
    "direita",
    "diz",
    "dizer",
    "dois",
    "dos",
    "e",
    "ela",
    "ele",
    "eles",
    "em",
    "enquanto",
    "então",
    "está",
    "estão",
    "estado",
    "estar",  
    "estará",
    "este",
    "estes",
    "esteve",
    "estive",
    "estivemos",
    "estiveram",
    "eu",
    "fará",
    "faz",
    "fazer",
    "fazia",
    "fez",
    "fim",
    "foi",
    "fora",
    "horas",
    "iniciar",
    "inicio",
    "ir",
    "irá",
    "ista",
    "iste",
    "isto",
    "ligado",
    "maioria",
    "maiorias",
    "mais",
    "mas",
    "mesmo",
    "meu",
    "muito",
    "muitos",
    "nós",
    "não",
    "nome",
    "nosso",
    "novo",
    "o",
    "onde",
    "os",
    "ou",
    "outro",
    "para",
    "parte",
    "pegar",
    "pelo",
    "pessoas",
    "pode",
    "poderá",   
    "podia",
    "por",
    "porque",
    "povo",
    "promeiro",
    "quê",
    "qual",
    "qualquer",
    "quando",
    "quem",
    "quieto",
    "são",
    "saber",
    "sem",
    "ser",
    "seu",
    "somente",
    "têm",
    "tal",
    "também",
    "tem",
    "tempo",
    "tenho",
    "tentar",
    "tentaram",
    "tente",
    "tentei",
    "teu",
    "teve",
    "tipo",
    "tive",
    "todos",
    "trabalhar",
    "trabalho",
    "tu",
    "um",
    "uma",
    "umas",
    "uns",
    "usa",
    "usar",
    "valor",
    "veja",
    "ver",
    "verdade",
    "verdadeiro",
    "você"
  ]

  return palavra in palavras

def pre_processa_texto(texto):
  novo_texto = []

  for palavra in texto.split(" "):
    palavra = remove_pontuacao(palavra).strip().lower()

    if len(palavra) < 1:
      continue

    if palavra[0] == "@":
      continue
    
    if palavra[0] == "#":
      continue

    if is_url(palavra):
      continue
    if  is_emoticon(palavra) or is_emoji(palavra):
      continue

    if palavra_neutra(palavra):
      continue

    t = palavra

    if len(palavra) < 7:
      t = substitui_abreviacoes_internet(palavra)
    t = lematiza(t)

    t = t.strip()

    if len(palavra) < 1:
      continue

    novo_texto.append(t)

  return " ".join(novo_texto)

with open("verbos.txt", "r") as f:
  for palavra in f:
    verbos.append(palavra.strip())

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
  testes_campo_texto = int(sys.argv[5])
  testes_campo_classe = int(sys.argv[6])

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
