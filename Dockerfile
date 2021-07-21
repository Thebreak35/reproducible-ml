FROM tensorflow/tensorflow:2.0.0-py3

WORKDIR /train

COPY requirements.txt .

RUN apt-get update && pip install -r requirements.txt

# COPY train.py .

CMD ["python", "train.py"]
