from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, FlexSendMessage, StickerSendMessage, TemplateSendMessage,ButtonsTemplate,MessageAction
)

# def ml(input):
#     tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-small')
#     tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
#     model = GPT2LMHeadModel.from_pretrained("rinna/japanese-gpt2-small")
#     model.load_state_dict(torch.load('Linebot2.pt', map_location=torch.device('cpu')))
#     input_ids = tokenizer.encode(input, return_tensors='pt')
#     beam_outputs = model.generate(
#         input_ids, 
#         pad_token_id=tokenizer.pad_token_id,
#         bos_token_id=tokenizer.bos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         max_length=50, 
#         num_beams=5, 
#         no_repeat_ngram_size=2, 
#         num_return_sequences=5, 
#         early_stopping=True
#     )
#     s = []
#     print("Output:\n" + 100 * '-')
#     for i, beam_output in enumerate(beam_outputs):
#         s.append(tokenizer.decode(beam_output, skip_special_tokens=True))
#     return s


app = Flask(__name__)

line_bot_api = LineBotApi('4W/LQ3IhGKdH5jobBYlYZbSvVz9f5P0g8FKQ4HQEvRfOeEUiAzC7goevo76pBdzAaCROTfsRUOVN3XGZoHQ8lHgHDIul1F/eHf8oUHkujU7iIfkyC9Hc+lKejBaM1QEsQ8wMIfunsvPmD5+GXW8t+AdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('4e5ed7673bcb498a1b456e1354684cb2')


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

flex_message = FlexSendMessage(
  alt_text='this is alt_text',
  contents=payload
)

@app.route("/")
def test():
    return "a"

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
    request_message = event.message.text
    reply_messages = []
    button_reply_message = flex_message
    if request_message == '1':
        reply_messages = TextSendMessage(text='1が正しいと認識しました')
    elif request_message == '2':
        reply_messages = (TextSendMessage(text='2が正しいと認識しました'))
    elif request_message == '3':
        reply_messages = (TextSendMessage(text='3が正しいと認識しました'))
    else:
        reply_messages.append(TextSendMessage(text='1の文章'))
        reply_messages.append(TextSendMessage(text='2の文章'))
        reply_messages.append(TextSendMessage(text='3の文章'))
        button_reply_message = flex_message
        reply_messages.append(button_reply_message)
    line_bot_api.reply_message(
        event.reply_token,
        reply_messages
    )


if __name__ == "__main__":
    app.run()