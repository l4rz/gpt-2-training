#!/bin/sh

if [ -z "$2" ]
then
  echo "Usage: $0 <input dir> <output dir>"
  echo "Converts fb2 files in <input dir> to plain utf-8 encoded text and puts them to <output dir>."
  exit 1
fi

if [ -z "$(which xsltproc)" ]
then
  echo "Please install xsltproc"
  exit 2
fi

INPUT=$(readlink -f "$1")
OUTPUT=$(readlink -f "$2")
mkdir -p "$OUTPUT"
cd "$(dirname "$0")"
N=$(nproc)
(for i in "$INPUT/"*.fb2
do
  TXT=$OUTPUT/$(basename "$i" .fb2).txt
  if [ ! -f "$TXT" ]
  then
    echo "Processing $i" >&2
    echo "-o \"$TXT\" FB2_2_txt.xsl \"$i\""
  fi
done) | xargs -P $N -n 4 xsltproc
wait
