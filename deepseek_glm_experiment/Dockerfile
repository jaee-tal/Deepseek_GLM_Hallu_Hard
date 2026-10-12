FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git python3 python3-pip python3-venv tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/cross_probe
COPY requirements.txt .
RUN python3 -m pip install --break-system-packages -r requirements.txt
COPY . .

RUN mkdir -p data/hf_hub_cache data/hf_datasets_cache runs results cloud_artifacts wandb logs

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "scripts/30_cloud_experiment.py", "--help"]
