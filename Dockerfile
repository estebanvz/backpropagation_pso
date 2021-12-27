FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt-get update
RUN apt-get install wget curl -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN mkdir app
WORKDIR /app
RUN pip3 install imageio opencv-python==4.5.3.56 tensorflow-hub
RUN pip3 install pandas
RUN pip3 install sklearn
RUN pip3 install pyswarms
RUN pip3 install black==19.10b0
RUN pip3 install flake8==3.9.2
RUN pip3 install isort==5.8.0
RUN pip3 install pyyaml==6.0
COPY . .
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser