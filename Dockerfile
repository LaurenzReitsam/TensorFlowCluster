FROM Tensorflow

COPY train.py /app

CMD python3 /app/train.py
