FROM python:3.9

WORKDIR /app

COPY . .

RUN mkdir -p /data/model

RUN mv ./data/data.csv ./data/embeddings.csv /data/model/

RUN pip install -r requirements.txt

CMD ["python", "app.py"]

EXPOSE 80