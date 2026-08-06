FROM python:3.11.13-slim-bookworm AS oqs-builder

ARG LIBOQS_VERSION=0.16.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates cmake git libssl-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --branch "${LIBOQS_VERSION}" --depth 1 https://github.com/open-quantum-safe/liboqs.git /src/liboqs \
    && cmake -S /src/liboqs -B /src/liboqs/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/oqs \
        -DBUILD_SHARED_LIBS=ON \
        -DOQS_BUILD_ONLY_LIB=ON \
        -DOQS_DIST_BUILD=ON \
    && cmake --build /src/liboqs/build --parallel 4 \
    && cmake --install /src/liboqs/build

FROM python:3.11.13-slim-bookworm

ARG LIBOQS_VERSION=0.16.0
ARG LIBOQS_PYTHON_VERSION=0.16.0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1 \
    OQS_INSTALL_PATH=/opt/oqs \
    LD_LIBRARY_PATH=/opt/oqs/lib:/opt/oqs/lib64 \
    PYOQS_VERSION=${LIBOQS_VERSION}

WORKDIR /app
COPY --from=oqs-builder /opt/oqs /opt/oqs
COPY phase4/requirements.txt /tmp/phase4-requirements.txt
COPY phase5/requirements.txt /tmp/phase5-requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu \
    && pip install --no-cache-dir -r /tmp/phase4-requirements.txt -r /tmp/phase5-requirements.txt \
    && python -c "import oqs; assert oqs.oqs_version() == '${LIBOQS_VERSION}'; assert oqs.oqs_python_version() == '${LIBOQS_PYTHON_VERSION}'; assert 'ML-KEM-768' in oqs.get_enabled_kem_mechanisms(); assert 'ML-DSA-65' in oqs.get_enabled_sig_mechanisms()" \
    && rm /tmp/phase4-requirements.txt /tmp/phase5-requirements.txt

COPY src/uavids_fl /app/uavids_fl
COPY phase4/app /app/phase4_app
COPY phase4/__init__.py /app/phase4/__init__.py
COPY phase4/app /app/phase4/app
COPY phase5/app /app/phase5_app
COPY phase5/config /app/phase5_config
COPY tests /app/tests
COPY phase4/tests /app/phase4_tests
COPY phase5/tests /app/phase5_tests

RUN useradd --create-home --uid 10001 demo
USER 10001:10001

EXPOSE 8080
