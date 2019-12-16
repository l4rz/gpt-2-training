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

I used the [nshepperd implementation](https://github.com/nshepperd/gpt-2) of training script. [Cybertronai gradient-checkpointing](https://github.com/cybertronai/gradient-checkpointing) was used in order to save GPU RAM.

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

Getting a large enough corpus of Russian text is quite simple, for example, there is a 568Gb one on [Oscar](https://traces1.inria.fr/oscar/). However corpora like this are unsuitable for training of unsupervised language models in real life because of quality. One needs a fairy clean collection of quality articles. While preparing the WebText dataset, OpenAI did a clever trick of outsourcing text cleaning to Reddit users.

I scraped a couple of Russian press sites, parsed HTML with beautifulsoup4 and saved parsed texts as well as metadata (headers, TL;DRs, timestamps) for further sorting and postprocessing in PKLs.

For the beginning, I got rid of texts that had a significant percentage on non-Cyrillic characters. I've also discarded texts with cruel and unusual formatting (tables, programming code) as well as repetitive ones (stock market reports, weather, sports) and too boring (official documents). Tabs, spaces and dashes were normalized. Hashtags and weird glyphs were filtered out too. Texts shorter than 1024 bytes were discarded.

Text paragraphs were separated with newline (\n) and <|n|> token. Each text was suffixed by <|endoftext|>.

Overall, a lot of effort has been put into cleaning the dataset. It should be noted that in Russian we do not have this particular luxury of English - to be able to make a purely monolingual dataset. Modern Russian texts always include some % of English - companies' and news agencies' names, social media accounts, quotations, etc

I've ended up with two datasets, 2000Mb and 4000Mb ones.

After sentencepiece encoding, the 2Gb dataset became a 211M tokens one. This means that comression ratio of bytes to BPE tokens is around 9:1, or 4.5:1 in characters taking UTF-8 into account. This ratio is much higher compared to vocab.bpe used with the original GPT-2.

<!-- During test runs, I've learned that book corpora do not work quite well, likely because the average book chapter doesn't fit into the model's attention window of 1,000 tokens.

Pure news articles corpus didn't work very well, too, likely due to the lack of language diversity. -->


## Experiments <a name="experiments"></a>

TBD

## Comments and ideas <a name="comments"></a>

TBD


## Written by

l4rz
