# Reproduction environment. Three targets:
#   analysis (default): regenerates every table and figure of the article from results/ and checks them against the
#                       manuscript sources; CPU only, a few minutes.
#   kraken:             Kraken 7.0.2 training and evaluation (code/training, code/scripts); needs an NVIDIA GPU and the data.
#   trocr:              TrOCR fine-tuning (code/trocr); needs an NVIDIA GPU and the data.
# Versions are those of the experiment server (conda environments kraken7 and trocr).
FROM python:3.13.14-slim AS analysis
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends make && rm -rf /var/lib/apt/lists/*
COPY docker/requirements-analysis.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
WORKDIR /repro
CMD ["make", "all"]

FROM python:3.13.14-slim AS kraken
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends make git && rm -rf /var/lib/apt/lists/*
COPY docker/requirements-kraken.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
WORKDIR /repro

FROM python:3.12.13-slim AS trocr
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends make && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
COPY docker/requirements-trocr.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
WORKDIR /repro
