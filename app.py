from transformers import T5Tokenizer, GPT2LMHeadModel
import torch

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, FlexSendMessage, StickerSendMessage, TemplateSendMessage,ButtonsTemplate,MessageAction
)


from flask import Flask,render_template,request,abort
import os
import json
import tensorflow.compat.v1 as tf
import argparse
from tqdm import tqdm
from sampling import sample_sequence
from encode_bpe import BPEEncoder_ja

line_bot_api = LineBotApi('4W/LQ3IhGKdH5jobBYlYZbSvVz9f5P0g8FKQ4HQEvRfOeEUiAzC7goevo76pBdzAaCROTfsRUOVN3XGZoHQ8lHgHDIul1F/eHf8oUHkujU7iIfkyC9Hc+lKejBaM1QEsQ8wMIfunsvPmD5+GXW8t+AdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('4e5ed7673bcb498a1b456e1354684cb2')

TOP_K = 10
TOP_P = 0
TEMPERATURE=1
MODEL_PATH = "checkpoint/run1"
# CONTEXT = "このところ意欲がない"
GPU="0"
NUM_GENERATE = 4
REPLY = []

def createbutton():
    payload = {
        "type": "bubble",
        "direction": "ltr",
        "body": {
        "type": "box",
        "layout": "vertical",
        "contents": [
            {
            "type": "text",
            "align": "center",
            "text": "どのテキストを使用しますか？"
            }
        ]
        },
        "footer": {
        "type": "box",
        "layout": "horizontal",
        "contents": [
            {
            "type": "button",
            "action": {
                "type": "message",
                "label": "1",
                "text": "1"
            },
            "style": "primary"
            },
            {
            "type": "button",
            "action": {
                "type": "message",
                "label": "2",
                "text": "2"
            },
            "style": "primary",
            "margin": "5px"
            },
            {
            "type": "button",
            "action": {
                "type": "message",
                "label": "3",
                "text": "3"
            },
            "margin": "5px",
            "style": "primary"
            }
        ]
        }
    }
    return payload

def ml(input):
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
    context_tokens = enc.encode(input)
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

        result = []
        if len(OUTPUT_FILE) > 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as of:
                for i in range(NUM_GENERATE):
                    of.write(generate_one(sess, output)+'\n')
                    if i < NUM_GENERATE-1:
                        of.write('========\n')
        else:
            for i in range(NUM_GENERATE):
                result.append(generate_one(sess, output, input))
        return result
                
    


#Flaskオブジェクトの生成
app = Flask(__name__)



#「/」へアクセスがあった場合に、"Hello World"の文字列を返す
@app.route("/")
def hello():
    ml("あいうえお")
    return "ok"

#「/index」へアクセスがあった場合に、「index.html」を返す
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'



filterreply = []

@handler.add(MessageEvent, message=(TextMessage))
def handle_image_message(event):
    global filterreply
    reply_messages = []
    request_message = event.message.text
    reply_messages = []
    if request_message != '1' and request_message != '2' and request_message != '3':
        filterreply = []
        a = ml(request_message)
        print(a)
        # a = ["sffdafasfsad","fdsfasfsad","fdsafasfadsfsad","njdsfndsaf"]
        for i in a:
            if len(i) > 6:
                print("追加 : ",i)
                filterreply.append(i)
            if len(filterreply) > 2:
                break
        print(filterreply[0][:15])
        print(filterreply[1][:15])
        print(filterreply[2][:15])
        print("a:{}".format(filterreply[0][:15]))
        reply_messages.append(TextSendMessage(text="1.{}".format(filterreply[0][:15])))
        reply_messages.append(TextSendMessage(text="2."+filterreply[1][:15]))
        reply_messages.append(TextSendMessage(text="3."+filterreply[2][:15]))
        button_reply_message = createbutton()
        flex_message = FlexSendMessage(
            alt_text='this is alt_text',
            contents=button_reply_message
        )
        reply_messages.append(flex_message)
    elif request_message == '1':
        reply_messages.append(TextSendMessage(text='1の文章の全文を表示します'))
        reply_messages.append(TextSendMessage(text='下記の文章をコピーしてお使いください'))
        reply_messages.append(TextSendMessage(text=filterreply[0][:200]))
    elif request_message == '2':
        reply_messages.append(TextSendMessage(text='2の文章の全文を表示します'))
        reply_messages.append(TextSendMessage(text='下記の文章をコピーしてお使いください'))
        reply_messages.append(TextSendMessage(text=filterreply[1][:200]))
    elif request_message == '3':
        reply_messages.append(TextSendMessage(text='3の文章の全文を表示します'))
        reply_messages.append(TextSendMessage(text='下記の文章をコピーしてお使いください'))
        reply_messages.append(TextSendMessage(text=filterreply[2][:200]))
    line_bot_api.reply_message(
        event.reply_token,
        reply_messages
    )

#おまじない
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)