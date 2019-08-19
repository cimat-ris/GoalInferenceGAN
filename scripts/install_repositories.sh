#!/usr/bin/env bash

DIR="../sgan/"
if [ -d "$DIR" ]; then
  ### Take action if $DIR exists ###
  echo "cimat-ris SGAN fork already cloned in ${DIR}"
else
  ###  Control will jump here if $DIR does NOT exists ###
  echo "${DIR} not found. Downloading."
  git clone https://github.com/cimat-ris/sgan.git ${DIR}
  exit 1
fi

DIR="../IntentionInference/"
if [ -d "$DIR" ]; then
  ### Take action if $DIR exists ###
  echo "cimat-ris IntentionInference already cloned in ${DIR}"
else
  ###  Control will jump here if $DIR does NOT exists ###
  echo "${DIR} not found. Downloading."
  git clone https://github.com/cimat-ris/IntentionInference.git ${DIR}
  exit 1
fi