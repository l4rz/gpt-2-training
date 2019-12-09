#!/bin/sh

if [ -z "$3" ]
then
  echo "Usage: $0 <lang> <input dir> <output dir>"
  echo "Filters the fb2 files in <input dir> by language <lang> and moves the matched ones to <output dir>."
  exit 1
fi

INPUT=$(readlink -f "$2")
OUTPUT=$(readlink -f "$3")
mkdir -p "$OUTPUT"
for i in "$INPUT/"*.fb2
do
  ENC=$(sed 's#>#>\n#g' "$i" | sed -n "s#.*encoding=[\"']\([^\"']*\)[\"'].*#\1#p" | head -1)
  if [ -n "$ENC" ]
  then
    BOOKLANG=$(iconv -f $ENC < "$i" | sed -n "s#.*<lang>\([^<]*\)</lang>.*#\1#gp" | head -1)
    if [ -z "$BOOKLANG" ]
    then
      echo "Undefined language: $i"
    fi
    if [ "$BOOKLANG" = "$1" ]
    then
      echo "$i language match"
      mv "$i" "$OUTPUT/$(basename "$i")"
    fi
  else
    echo "Undefined encoding: $i"
  fi
done