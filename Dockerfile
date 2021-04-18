FROM ubuntu:20.04

ARG TZ=Africa/Johannesburg
ENV PYTHONUNBUFFERED 1

# Install requirements for installing Python 3.9
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common curl
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt install -y python3.9-distutils python3.9-dev
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py
# Necessary installation to compile auto-sklearn dependencies
RUN apt install -y build-essential

# Install auto-sklearn
RUN pip install auto-sklearn==0.12.5 bpython

# Change to working directory
WORKDIR /feature_engineering_aide/

# Bring in requirements and install any remaining requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY experiment_config experiment_config/

ENV PYTHONPATH /feature_engineering_aide/

ENTRYPOINT ["python"]

CMD ["-m", "bpython"]
