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

line_bot_api = LineBotApi('qFv+ziwbtKuvF3+N6S+SleuNiydkE8jXLdYMxh3wUUQUXOTiSM12JlB3fktthpN+wlYiK5jCspKMK711hf5EZIvvYO5ewpYMz4cb3L8Kkj6/+helDE2SHnTiIerQV+Jclxs4s1YtB8qKUlbW4OVElQdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('37ada14b83c43df19ce22f49deef23e8')
def createbutton(text1,text2,text3):
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
            "text": "どのテキストが適していますか？"
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
    tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-small')
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
    model = GPT2LMHeadModel.from_pretrained("rinna/japanese-gpt2-small")
    model.load_state_dict(torch.load('Linebot2.pt', map_location=torch.device('cpu')))
    input_ids = tokenizer.encode(input, return_tensors='pt')
    beam_outputs = model.generate(
        input_ids, 
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=50, 
        num_beams=5, 
        no_repeat_ngram_size=2, 
        num_return_sequences=5, 
        early_stopping=True
    )
    s = []
    print("Output:\n" + 100 * '-')
    for i, beam_output in enumerate(beam_outputs):
        print(tokenizer.decode(beam_output, skip_special_tokens=True))
        s.append(tokenizer.decode(beam_output, skip_special_tokens=True))
    return s
    


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





@handler.add(MessageEvent, message=(TextMessage))
def handle_image_message(event):
    reply_messages = []
    request_message = event.message.text
    reply_messages = []
    if request_message == '1':
        reply_messages = TextSendMessage(text='1が正しいと認識しました')
    elif request_message == '2':
        reply_messages = (TextSendMessage(text='2が正しいと認識しました'))
    elif request_message == '3':
        reply_messages = (TextSendMessage(text='3が正しいと認識しました'))
    else:
        a = ml(request_message)
        reply_data = a.split('\n')

        reply_messages.append(TextSendMessage(text=reply_data[0]))
        reply_messages.append(TextSendMessage(text=reply_data[1]))
        reply_messages.append(TextSendMessage(text=reply_data[2]))
        button_reply_message = createbutton(reply_data[0],reply_data[1],reply_data[2])
        flex_message = FlexSendMessage(
            alt_text='this is alt_text',
            contents=button_reply_message
        )
        reply_messages.append(flex_message)
    line_bot_api.reply_message(
        event.reply_token,
        reply_messages
    )

#おまじない
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)