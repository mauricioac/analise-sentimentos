# coding=utf-8

import sys
import re, math, collections, itertools
import nltk, nltk.classify.util
from nltk.metrics import *
from nltk.classify import NaiveBayesClassifier
import csv
import random

verbos = []

def classifica(trainFeatures, testFeatures):
  #faz treinamento e retorna classificador treinado
  classifier = NaiveBayesClassifier.train(trainFeatures)

  dados_de_referencia = collections.defaultdict(set)
  dados_de_teste = collections.defaultdict(set)
  # vp = 0
  # fp = 0
  # vn = 0
  # fn = 0

  for i, (features, label) in enumerate(testFeatures):
    dados_de_referencia[label].add(i)
    predicted = classifier.classify(features)
    dados_de_teste[predicted].add(i)

    # if label == "pos" and predicted == "pos":
    #   vp = vp + 1
    # elif label == "pos" and predicted == "neg":
    #   fn = fn + 1
    # elif label == "neg" and predicted == "neg":
    #   vn = vn + 1
    # elif label == "neg" and predicted == "pos":
    #   fp = fp + 1
  return {
    "classifier": classifier,
    "ref": dados_de_referencia,
    "test": dados_de_teste
  }

def classifica_arquivo_separado(features_positivas, features_negativas, testes):
  trainFeatures = features_negativas + features_positivas

  return classifica(trainFeatures, testes)

def classifica_mesmo_arquivo(features_positivas, features_negativas):
  #calcula o ponto de corte
  posCutoff = int( math.floor( len(features_positivas)*3/4 ) )
  negCutoff = int( math.floor( len(features_negativas)*3/4 ) )

  trainFeatures = features_positivas[:posCutoff] + features_negativas[:negCutoff]
  testFeatures = features_positivas[posCutoff:] + features_negativas[negCutoff:]

  return classifica(trainFeatures, testFeatures)

def is_url(texto):
  ocorrencias = re.findall('http', texto)
  return len(ocorrencias) > 0

def remove_pontuacao(palavra):
  return re.sub('[\.,;?:+*&$=º~^\\\/"\'\(\)\[\]\{\}]\t\n', '', palavra)

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
      return nova_palavra

  return palavra

def is_emoji(palavra):
  ocorrencias = re.findall(':[a-zA-Z]+:', palavra)
  return len(ocorrencias) > 0

def extrai_features(arquivo, campo_texto, campo_classe):
  #abre arquivo e instancia um leitor de  csv
  with open(arquivo, "r") as f:
    leitor = csv.reader(f, delimiter=',', quotechar='"')

    for line in leitor:
      texto = line[campo_texto]
      #polaridade = positivo ou negativo
      polaridade = line[campo_classe]

      texto = pre_processa_texto(texto)
      #separa em tweets positivos e negativos
      if polaridade == "1":
        tweets_positivos.append(texto)
      else:
        tweets_negativos.append(texto)

  posFeatures = []
  negFeatures = []

  #para cada tweet é construido o formato do Naive Bayes
  for tweet in tweets_positivos:
    palavras = tweet.split(" ")
    # vetor com features na posição 0 e a classe na última posição
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

#Monta um dicionário de palavras
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

#le lista de verbos de arquivo externo
with open("verbos.txt", "r") as f:
  for palavra in f:
    verbos.append(palavra.strip())

tweets_negativos = []
tweets_positivos = []

parametros = len(sys.argv)

#mensagem de erro, com indicações de como executar o programa
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

#nome do arquivo de treinamento
treinamento_arquivo = sys.argv[1]
#qual o campo do csv que vai usar como texto/mensagem
treinamento_campo_texto = int(sys.argv[2])
#qual a classe (pos,neg)
treinamento_campo_classe = int(sys.argv[3])
#se vai ter arquivo de teste ou não
testes_arquivo = False

#se tem 7 parametros, significa que terá um arquivo de teste
if parametros == 7:
  testes_arquivo = sys.argv[4]
  testes_campo_texto = int(sys.argv[5])
  testes_campo_classe = int(sys.argv[6])

#extrai features de treinamento do arquivo, monta os dados de treinamento para o Naive Bayes
treinamento = extrai_features(treinamento_arquivo, treinamento_campo_texto, treinamento_campo_classe)


if testes_arquivo:
  #se tem arquivo de teste extrai as features do arquivo de teste
  teste = extrai_features(testes_arquivo, testes_campo_texto, testes_campo_classe)
  #features positivas, features negativas, e o teste
  tmp = classifica_arquivo_separado(treinamento[0], treinamento[1], teste[0] + teste[1])
else:
  #tmp retorna os dados de classificação, e o classificador
  tmp = classifica_mesmo_arquivo(treinamento[0], treinamento[1])

classifier = tmp["classifier"]
dados_de_referencia = tmp["ref"]
dados_de_teste = tmp["test"]

print 'precisão positiva:', precision(dados_de_referencia['pos'], dados_de_teste['pos'])
print 'revocação positiva:', recall(dados_de_referencia['pos'], dados_de_teste['pos'])
print 'F-measure positivo:', f_measure(dados_de_referencia['pos'], dados_de_teste['pos'])
print 'precisão negativa:', precision(dados_de_referencia['neg'], dados_de_teste['neg'])
print 'revocação negativa:', recall(dados_de_referencia['neg'], dados_de_teste['neg'])
print 'F-measure negativo:', f_measure(dados_de_referencia['neg'], dados_de_teste['neg'])