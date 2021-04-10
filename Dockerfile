FROM ubuntu:20.04

ARG TZ=Africa/Johannesburg
ENV PYTHONUNBUFFERED 1

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common curl
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt install -y python3.9-distutils python3.9-dev
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py
RUN apt install -y build-essential

RUN pip install auto-sklearn==0.12.5 bpython

ADD requirements.txt ./requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python"]

CMD ["-m", "bpython"]
