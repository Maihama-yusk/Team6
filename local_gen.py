#実際の推論のみの作成
TOP_K = 10
TOP_P = 0
TEMPERATURE=1
MODEL_PATH = "checkpoint/run1"
CONTEXT = "このところ意欲がない"
GPU="0"
NUM_GENERATE = 10
import os
import json
import tensorflow.compat.v1 as tf
import argparse
from tqdm import tqdm
from sampling import sample_sequence
from encode_bpe import BPEEncoder_ja
if int(tf.__version__[0]) > 1:
    class HParams:
        def __init__(self,
                    n_vocab=0,
                    n_ctx=1024,
                    n_embd=768,
                    n_head=12,
                    n_layer=12,
                    n_prediction=160):
            self.n_vocab = n_vocab
            self.n_ctx = n_ctx
            self.n_embd = n_embd
            self.n_head = n_head
            self.n_layer = n_layer
            self.n_prediction = n_prediction
else:
    from tensorflow.contrib.training import HParams
  
if int(tf.__version__[0]) > 1:
    from model import HParams as HParams
else:
    from tensorflow.contrib.training import HParams
  
with open('ja-bpe.txt', encoding='utf-8') as f:
    bpe = f.read().split('\n')

with open('emoji.json', encoding='utf-8') as f:
    emoji = json.loads(f.read())

enc = BPEEncoder_ja(bpe, emoji)
n_vocab = len(enc)
if os.path.isfile(MODEL_PATH+'/hparams.json'):
    with open(MODEL_PATH+'/hparams.json') as f:
        params = json.loads(f.read())
        hparams = HParams(**params)
        max_length = params['n_prediction']
else:
    raise ValueError('invalid model name.')

def filter_duplicate_line(swd):
    stopword = '。｡．？！?!；;：:\r\n'
    splits = []
    spos = 0
    for i in range(0,len(swd)-1,1):
        if swd[i] in stopword:
            wd = swd[spos:i+1]
            if wd not in splits:
                splits.append(wd)
            spos = i+1
    if spos < len(swd)-1:
        wd = swd[spos:len(swd)]
        if wd not in splits:
            splits.append(wd)
    return ''.join(splits)

length=hparams.n_ctx - max_length - 1
temperature=TEMPERATURE
top_k=TOP_K
top_p=TOP_P
SEP_TOKEN = enc.encode('<|byte0|>')[0]
##入力
context_tokens = enc.encode(CONTEXT)
ALLOW_DUCPLICATE_LINE = "store_true"

def generate_one(sess, output, cont):
    context_tokens = enc.encode(cont)
    if len(context_tokens) > length:
        context_tokens = context_tokens[:length]
    context_tokens.append(SEP_TOKEN)
    out = sess.run(output, feed_dict={
        context: [context_tokens]
    })[:,len(context_tokens)-1:]
    swd = enc.decode(out[0])
    if '<|endoftext|>' in swd:
        swd = swd.split('<|endoftext|>')[0]
    if not ALLOW_DUCPLICATE_LINE:
        swd = filter_duplicate_line(swd)
    return swd
def filter_duplicate_line(swd):
    stopword = '。｡．？！?!；;：:\r\n'
    splits = []
    spos = 0
    for i in range(0,len(swd)-1,1):
        if swd[i] in stopword:
            wd = swd[spos:i+1]
            if wd not in splits:
                splits.append(wd)
            spos = i+1
    if spos < len(swd)-1:
        wd = swd[spos:len(swd)]
        if wd not in splits:
            splits.append(wd)
    return ''.join(splits)

config = tf.ConfigProto()
OUTPUT_FILE=''
if int(GPU) >= 0:
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = GPU
with tf.Session(config=config,graph=tf.Graph()) as sess:
    context = tf.placeholder(tf.int32, [1, None])
    output = sample_sequence(
        hparams=hparams, length=length,
        min_length=1, context=context,
        batch_size=1,
        temperature=temperature, top_k=top_k, top_p=top_p
    )

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(MODEL_PATH)
    saver.restore(sess, ckpt)

    if len(OUTPUT_FILE) > 0:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as of:
            for i in range(NUM_GENERATE):
                of.write(generate_one(sess, output)+'\n')
                if i < NUM_GENERATE-1:
                    of.write('========\n')
    else:
        for i in range(NUM_GENERATE):
            print(generate_one(sess, output,CONTEXT))
            if i < NUM_GENERATE-1:
                print('========')