# Training GPT-2 on a non-English language corpus

<b>
Disclaimer: Neither me nor this repo is associated in any way with OpenAI. I did my DYOR to the best of my ability, nevertheless I might be completely wrong about anything expressed below.
</b>


## Table of Contents
1. [Training environment](#trainscript)
2. [Dataset preparation](#dataset)
3. [Experiments](#experiments)
4. [Comments and ideas](#comments)

## Training environment <a name="trainscript"></a>

I've used [nshepperd implementation](https://github.com/nshepperd/gpt-2) of training script. [Cybertronai gradient-checkpointing](https://github.com/cybertronai/gradient-checkpointing) was used in order to save GPU RAM.

Since the original vocab.bpe does not include Cyrillic alphabet (...), I've employed [Google SentencePiece](https://github.com/google/sentencepiece) tokenizer.

(...)

Quick start guide:

1. clone [nshepperd repo](https://github.com/nshepperd/gpt-2)
2. comment out the line "if layer == 10:" in model.py for checkpointing to work properly (save memory)
3. install [Google SentencePiece](https://github.com/google/sentencepiece)
4. use src/encoder_sp.py from this repo, copy it to src/ dir
5. replace all relevant "import encoder" with "import encoder_sp as encoder" in relevant files (encode.py and sampling scripts)
6. train the sp tokenizer model using your dataset<br><br>
  <i>spm_train  --character_coverage=1  --model_prefix=sp --vocab_size=50257 --model_type=bpe --user_defined_symbols '<|n|>,<|endoftext|>' --max_sentence_length=32768 --input_sentence_size=10000000 --input dataset.txt</i><br><br>
7. copy sp.* files to the model directory
8. encode the dataset using trained sp tokenizer model<br><br>
  <i>spm_encode --model="models/1558M/sp.model" --output_format=id < dataset.txt | split --lines=100000 --additional-suffix=.ids - /tmp/spencode/part$(printf %05d $i)</i><br><br>
  <i>PYTHONPATH=src ./encode.py --model_name="1558M" /tmp/spencode/ dataset.npz</i><br>
9. put the proper hparams.json in the model directory (since we are training the model from scratch, we do not need checkpoint files, etc)
10. initialize the model by running sess.run(tf.global_variables_initializer()) instead of saver.restore(sess, ckpt) (sorry i haven't bothered to make a separate script)
11. proceed with warmup and training


## Dataset preparation <a name="dataset"></a>

TBD

## Experiments <a name="experiments"></a>

TBD

## Comments and ideas <a name="comments"></a>

TBD


## Written by

l4rz
