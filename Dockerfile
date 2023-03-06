FROM python:3.9-alpine3.16

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "app.py"]