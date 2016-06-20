#!/bin/bash

TREINAMENTO_ARQUIVO="treinamento.csv"
TREINAMENTO_CAMPO_TEXTO="5"
TREINAMENTO_CAMPO_CLASSE="6"

TESTE_ARQUIVO="treinamento.csv"
TESTE_CAMPO_TEXTO="5"
TESTE_CAMPO_CLASSE="6"

USAR_ARQUIVO_TESTE=0

if [[ $USAR_ARQUIVO_TESTE -eq 1 ]]; then
  python main.py $TREINAMENTO_ARQUIVO $TREINAMENTO_CAMPO_TEXTO $TREINAMENTO_CAMPO_CLASSE $TESTE_ARQUIVO $TESTE_CAMPO_TEXTO $TESTE_CAMPO_CLASSE
else
  python main.py $TREINAMENTO_ARQUIVO $TREINAMENTO_CAMPO_TEXTO $TREINAMENTO_CAMPO_CLASSE
fi