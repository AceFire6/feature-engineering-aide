FROM python:3.9-slim

# Change to working directory
WORKDIR /feature_engineering_aide/

# Install auto-sklearn
RUN python -m venv venv
ENV PATH="/feature_engineering_aide/venv/bin:$PATH"
# Update install tools to latest versions
RUN pip install -U pip wheel setuptools

# set environment variables to only use one core
# from https://github.com/automl/auto-sklearn/blob/275d0d6b20d16822252d8b50bf71b1c787187f09/Dockerfile#L17-L21
RUN export OPENBLAS_NUM_THREADS=1
RUN export MKL_NUM_THREADS=1
RUN export BLAS_NUM_THREADS=1
RUN export OMP_NUM_THREADS=1

RUN pip install auto-sklearn==0.12.6

# Bring in requirements and install any remaining requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY experiment_config experiment_config/

ENV PYTHONPATH /feature_engineering_aide/

ARG TZ=Africa/Johannesburg

RUN echo $TZ > /etc/timezone
ENV PYTHONUNBUFFERED 1

ENTRYPOINT ["python"]

CMD ["-m", "bpython"]
