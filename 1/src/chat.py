#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder_sp as encoder


def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=None,
    length=None,
    temperature=1,
    top_k=0,
    run_name='run1',
):
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0
    np.random.seed(seed)
    tf.set_random_seed(seed)

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name, "checkpoint/%s" % run_name))
        saver.restore(sess, ckpt)

        delim = '\n'
        ctxlist = []
        ctxlength = 10
        last_response = ""
        while True:
            ctxlist = ctxlist[-ctxlength:]
            raw_text = input(">>> ")
            while not raw_text:
                print('Say something!')
                raw_text = input(">>> ")
            cline = ""
            for p in zip(ctxlist[::2], ctxlist[1::2]):
                cline += "— %s<|n|>— %s<|n|>" % (p[0], p[1])
            cline += "— %s<|n|>— " % raw_text
            context_tokens = enc.encode(cline)
            generated = 0
            text = raw_text
            while (text == raw_text or text == last_response) and generated < 10:
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    fulltext = enc.decode(out[i])
                text = fulltext.split(delim)[0]
                
            print(">>>", text)
            ctxlist.append(raw_text)
            ctxlist.append(text)
            last_response = text


if __name__ == '__main__':
    fire.Fire(interact_model)

