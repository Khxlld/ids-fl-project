FROM python:3.11.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1

WORKDIR /app

COPY phase4/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY src/uavids_fl /app/uavids_fl
COPY phase4/app /app/phase4_app

RUN useradd --create-home --uid 10001 demo
USER 10001:10001

EXPOSE 8080
