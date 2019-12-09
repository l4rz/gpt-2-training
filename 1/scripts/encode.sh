#!/bin/sh

if [ -z "$3" ]
then
  echo "Usage: $0 <input file> <model dir> <output file>"
  echo "Creates an .npz file from text <input file> using sp.model found inside models/<model dir> and stores the result as models/<model dir>/<output file>"
  echo "Provide just the model name like 117M_Books, not thet full path!"
  echo "If you use tensorflow_gpu you might want to provide CUDA libs path in LD_LIBRARY_PATH for this script to run. If there are no import errors about libcublas, you're fine."
  exit 1
fi

if [ -z "$(which spm_encode)" ]
then
  echo "Please download, build and install Sentence Piece from https://github.com/google/sentencepiece"
  exit 2
fi

INPUT=$(readlink -f "$1")
MODEL="$2"
OUTPUT="$3"

cd "$(dirname "$0")/.."
SPLITDIR=$(mktemp -d splitXXXXX)
OUTDIR=$(mktemp -d outXXXXX)
trap "rm -rf $OUTDIR $SPLITDIR" INT TERM EXIT
echo "Splitting $INPUT into $(nproc) parts for parallel processing..."
split -n l/$(nproc) --additional-suffix=.txt "$INPUT" "$SPLITDIR"/part
echo "Done. Encoding with spm started..."
i=1
for SP in "$SPLITDIR"/part*
do
  pwd
  spm_encode --model="models/$MODEL/sp.model" --output_format=id < "$SP" | split --lines=100000 --additional-suffix=.ids - "$OUTDIR"/part$(printf %05d $i)&
  i=$(( i + 1 ))
done
wait
echo "Done. Loading the data and packing into $OUTPUT"
PYTHONPATH=src ./encode.py --model_name="$MODEL" "$OUTDIR" "models/$MODEL/$OUTPUT"
echo "Done."
