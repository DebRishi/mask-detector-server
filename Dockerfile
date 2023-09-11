FROM tensorflow/tensorflow:2.13.0

RUN apt update
RUN apt install -y libgl1-mesa-glx

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 5000

CMD python server.py