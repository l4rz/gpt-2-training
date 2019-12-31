# Training GPT-2 on a Russian language corpus

<b>
Disclaimer: Neither me nor this repo is associated in any way with OpenAI. I did my DYOR to the best of my ability, nevertheless I might be completely wrong about anything expressed below.
</b>


## Table of Contents
1. [Quick start](#quickstart)
2. [Training environment](#trainscript)
3. [Dataset preparation](#dataset)
4. [Experiments](#experiments)
5. [Downloads](#downloads)

## 1. Quick start <a name="quickstart"></a>

1. clone [nshepperd repo](https://github.com/nshepperd/gpt-2)
2. comment out the ```if layer == 10:``` line in model.py for checkpointing to work properly (to save memory)
3. install [Google SentencePiece](https://github.com/google/sentencepiece)
4. use ```src/encoder_sp.py``` from this repo (copy to ```src/``` directory)
5. replace all relevant ```import encoder``` with ```import encoder_sp``` as encoder" in relevant files (encode.py and sampling scripts)
6. train the sp tokenizer model using your dataset
```
  spm_train --character_coverage=1  --model_prefix=sp \
      --vocab_size=50257 --model_type=bpe \
      --user_defined_symbols '<|n|>,<|endoftext|>'
      --max_sentence_length=32768
      --input_sentence_size=10000000
      --input dataset.txt
```
7. copy ```sp.*``` files to the model directory
8. encode the dataset using trained sp tokenizer model
```
  mkdir /tmp/spencode
  spm_encode --model="models/1558M/sp.model" \
      --output_format=id < dataset.txt | \
      split --lines=100000 --additional-suffix=.ids \
      - /tmp/spencode/part$(printf %05d $i)
  PYTHONPATH=src ./encode.py --model_name="1558M" \
      /tmp/spencode/ dataset.npz
  rm -rf /tmp/spencode
  ```
9. put the proper hparams.json in the model directory (since we are training the model from scratch, we do not need checkpoint files, etc)
10. initialize the model by running sess.run(tf.global_variables_initializer()) instead of saver.restore(sess, ckpt) (sorry i haven't bothered to make a separate script)
11. proceed with warmup and training

## 2. Training environment <a name="trainscript"></a>

I used the [nshepperd implementation](https://github.com/nshepperd/gpt-2) of training script. [Cybertronai gradient-checkpointing](https://github.com/cybertronai/gradient-checkpointing) was used in order to save GPU RAM.

I've employed [Google SentencePiece](https://github.com/google/sentencepiece) tokenizer (it's pointless to try using the original vocab.bpe since it's not aware of Cyrillic subwords).

For multi-GPU distributed training, I've utilized [Horovod](https://github.com/horovod/horovod) framework. Using Horovod is basically as easy as wrapping your optimizer in `hvd.DistributedOptimizer`, as well as adding proper initialization and variable broadcast calls.

```
import horovod.tensorflow as hvd
...
hvd.init()
...
config.gpu_options.visible_device_list = str(hvd.local_rank())
...
bcast = hvd.broadcast_global_variables(0).
...
opt = tf.train.AdamOptimizer(decayed_lr)
opt = hvd.DistributedOptimizer(opt)
```

To run a Horovod cluster with two or more servers, one needs to configure ssh so the user running training script can ssh without password to the slave server. To minimize the latency the servers should be interconnected with 1Gbps or faster link.

Training with large models (>1B hyperparameters) involves a quite long initialization, so I found it beneficial to increase the startup timeout in horovodrun:

```
horovodrun --start-timeout 600 \
-np 4 -H localhost:4 python3 train-1250M.py --dataset dataset.npz
```

Running four workers on a single machine utilized almost 200Gb of DRAM with large models.

## 3. Dataset preparation <a name="dataset"></a>

Getting a large enough corpus of Russian text is quite simple, for example, there is a 568Gb one on [Oscar](https://traces1.inria.fr/oscar/). However corpora like this are unsuitable for training of unsupervised language models in real life because of quality. One needs a fairy clean collection of quality articles. While preparing the WebText dataset, OpenAI did a clever trick of outsourcing text cleaning to Reddit users.

I scraped a couple of Russian press sites, parsed HTML with beautifulsoup4 and saved parsed texts as well as metadata (headers, TL;DRs, timestamps) for further sorting and postprocessing in PKLs.

(Instead of scrapping, one can use the Taiga [^1] dataset. Unfortunately I found it after I've already assembled my own one)
[^1]: https://github.com/TatianaShavrina/taiga_site

For the beginning, I got rid of texts that had a significant percentage on non-Cyrillic characters. I've also discarded texts with cruel and unusual formatting (tables, programming code) as well as repetitive ones (stock market reports, weather, sports) and too boring (official documents). Tabs, spaces and dashes were normalized. Hashtags and weird glyphs were filtered out too. Texts shorter than 1024 bytes were discarded.

<!-- - remove lines where percentage of upper case > 40%
- remove articles where unlabeled dialogues (beginning with "-") contain >20% of all limes
-->
Text paragraphs were separated with newline (\n) and <|n|> token. Each text fragment was suffixed by <|endoftext|>.

Overall, a lot of effort has been put into cleaning the dataset. Having a strictly monolingual dataset is a particular privilege of English; modern Russian texts always include some percent of Latin (English) proper nouns such as persons' and companies names, social media accounts, quotes, etc.

I've ended up with two datasets, ~2Gb and ~4Gb ones. These figures were much smaller than 50Gb WebText dataset, nevertheless I've considered these datasets diverse enough; moreover, they should've worked for my plan (overfit the model on a smaller dataset and then add more data).

After sentencepiece encoding, the ~2Gb dataset became a ~211M tokens one. This means that compression ratio of bytes to BPE tokens is around 9:1, or 4.5:1 in characters taking UTF-8 into account. This ratio is much higher compared to vocab.bpe used with the original GPT-2.

Because I experimented with encoding using various sentencepiece models, I found it useful to add the last digits of md5sum of `sp.model` to the encoded datasets, snapshots and samples file names to avoid confusion.

<!-- During test runs, I've learned that book corpora do not work quite well, likely because the average book chapter doesn't fit into the model's attention window of 1,000 tokens.

Pure news articles corpus didn't work very well, too, likely due to the lack of language diversity. -->

## 3. Experiments <a name="experiments"></a>

## Smaller models

Previously I tried to train GPT-2 using Russian dataset once the 117M model has been released. I only had a 1080ti at my disposal at this time so I've been training with small batch sizes. The most I was able to get was 3.00 loss after 20 epochs.

I've decided to run this experiment again, on Tesla V100s. I've settled on batch size of 24 per worker (192 combined for 8 GPUs).

The model was initialized from scratch and warmed up with LR=10<sup>-8</sup> for 1000 steps. Initial LR was 10<sup>-4</sup> until 10 epochs then decreasing to 1 x 10<sup>-6</sup>. Batch size was 24 (this means 192 combined for 8 GPU).

After 25 epochs and 123k steps on a 117M-sized model (first 20 epochs took approximately 150 gpu/hours), I've got training loss of 2.86. The quality of the samples was far from desired. In addition to the usual GPT-2 glitches (such as repeated words in a sentence), the text was less coherent than the English 117M model released by OpenAI.

Reducing the dataset size (from 211M to 150M tokens) and filtering out the remaining English characters did not help much.  

The 117M model as of my last run, trained for 123k steps, complete with sp.vocab and dataset used in training, is [available for download](#downloads)


## Larger models

I've achieved the similar results with larger models:

| Model size        | Duration           | Training loss  |
| ------------- |:-------------:| -----:|
| 117M      | 25 epochs, 123k steps | 2.86 |
| 345M      | 13 epochs, 507k steps  |   2.83 |
| 774M | 20 epochs, ??? steps     |   2.90 |

<!-- | "1B" | 15 epochs, 49k steps | 2.77 -->

It looks like I've always been hitting this 2.80 floor, something was wrong.

## Decreasing max token length in vocab

I noticed that the compression ratio (bytes to tokens) of Russian text encoded to BPE is 2-3 times higher than that of the original GPT-2 vocab.bpe. Observing that a Russian text snippet translated into English varies just 10-15 percent in length, I assumed that the text complexity per 1024 tokens would be much higher for Russian, and this would lead to more perplexed model if this aspect is not addressed.

I tried to decrease the maximum length of the subword fragment by training sentencepiece model with `spm_train --max_sentencepiece_length 5`. The 211M tokens dataset thus became 315M tokens one. Training a model with this type of encoding basically produced far worse results, though the curve of training loss per epoch was quite steeper and the final training loss was much less compared to the original sentencepiece model (just 2.02 after 4.5 epoch and 12,000 steps). The better the tokenizer performs, the worse the <i>indicated</i> perplexity of the model is.

## Addressing the language complexity

Russian grammar is rather complex. Word order in Russian is much more flexible compared to English. Russian uses cases, which means that every noun changes its ending, depending on what function it has in the sentence. Moreover, depending on the case, as well as singular or plural form, not only nouns are changing their endings but also adjectives, pronouns and some other parts of speech.

In order to address the complexity, I tried to increase the capacity of the model. The most logical way [^2] seemed to increase the embedding size `n_embd` parameter that defines the size of both token embeddings (wte) and positional embeddings (wte).

I wanted to increase the value `n_embd` (AKA D<sub>model</sub>) after 1600 as it was used in 1558M model but I learned that the number of hyperparameters can quickly grow beyond 2B. A model with `{ "n_ctx": 1024,  "n_embd": 2400,  "n_head": 16,  "n_layer": 24 }` takes 6.8Gb on disk and becomes rather impractical to train on 32Gb GPU. Therefore, I settled on `{ "n_ctx": 1024,  "n_embd": 2000,  "n_head": 16,  "n_layer": 24 }`. Number of layers and attention heads were the same as in 345M model but `n_embd` was greater compared to 1558M model.

[^2]: Other approaches have been tried but ultimately failed.

## 1250M model

The new model with D<sub>model</sub>=2000 and 1250M hyperparameters (approximately) was initialized and trained with the same 211M dataset.

[Training log of 1250M model first training run](1250M-results/trainlog-1250M-61k.txt)

With 32Gb of VRAM, I've been able to use batch size of 16 per worker (128 combined for 8 GPUs). Initial LR was 10<sup>-4</sup>. The complete training log, reflecting LR changes, is available [here](1250M-results/trainlog-1250M-61k.txt).

From the training loss of 3.00 (~6 epochs), the samples began to demonstrate consistency and were generally better than the previous run. I've continued training for 310 wallclock hours (accumulated 1800 GPU/hours), 27 epochs and reached training loss of 2.54.

[Unconditional generation samples](1250M-results/unconditional-generation-61k.txt)

[Conditional generation samples](1250M-results/conditional-generation-61k.txt)

I've also tested the model's ability to to perform summarization on news articles from the web. Some percentage of news articles in the dataset were salted with `ТЛ;ДР:` followed by article's summary or headline. Summarization behaviour was induced by Top-k random sampling with `k = 2` and providing the model a text to summarize followed by `ТЛ;ДР:` as conditional input.

`interactive_conditional_samples.py --temperature=0.8 --top_k=2 --length=100`

[A couple of summarization results](1250M-results/summarization-1250M-61k.txt)


<!--
![Summarizer test #1](images/summarizing.png?raw=true "summarizer test #1")

-->

## 4. Downloads <a name="downloads"></a>

## Pre-trained models

1. 117M model trained with 2Gb dataset and sp vocab/model [1.35Gb file](https://mega.nz/#!yJUiDaiS!FAs-iKmQu4ibfa6bzK-gq_AHz7k7Q4aTCztE0APZH6w)

2. 1250M model trained with 2Gb dataset, 61k steps, training loss 2.54, l4rz-russian-1250M-62000-release.tar [4.69Gb file](https://mega.nz/#!DNtilaxB!elM0PIt9piS1KFKR9KXmu7DqCYws94cNu-Our1IuN3M)

3. 1250M model trained with 2Gb dataset, from 61k steps to 100k steps on 4Gb dataset, training loss 2.73 [4.69Gb file](https://mega.nz/#!bNUABIqD!d9sD3Cn50t3TB_MXvtRh9XQ_GYrjrfNk4qIOF2bUNiU)


## Written by

l4rz
