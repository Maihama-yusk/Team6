FROM python:3.10-buster

WORKDIR /

COPY ./ /

RUN pip install -r requirements.txt

CMD ["python", "app.py"]