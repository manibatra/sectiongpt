FROM python:3.9

WORKDIR /app

COPY . .

RUN mkdir -p /data/model

RUN mv ./data/model/data.csv ./data/model/embeddings.csv /data/

RUN pip install -r requirements.txt

CMD ["python", "app.py"]

EXPOSE 80