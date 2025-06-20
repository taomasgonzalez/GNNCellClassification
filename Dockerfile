FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt .

RUN python3 -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && pip install \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -f https://data.pyg.org/whl/torch-2.7.0+cpu.html \
       -r requirements.txt

RUN /opt/venv/bin/pip install -r requirements.txt

CMD ["bash"]
