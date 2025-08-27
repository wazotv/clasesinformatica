#!/usr/bin/env bash
set -euo pipefail

# Setup para Raspberry Pi 4 (armv7l o aarch64)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "[1/6] Actualizando sistema..."
sudo apt-get update -y

echo "[2/6] Instalando dependencias del sistema..."
sudo apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  python3-opencv \
  libatlas-base-dev \
  libjpeg-dev \
  libopenjp2-7 \
  libtiff5 \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-libav

echo "[3/6] Creando entorno virtual..."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "[4/6] Instalando dependencias Python básicas..."
pip install -r requirements.txt

ARCH=$(uname -m)
echo "Arquitectura detectada: $ARCH"

echo "[5/6] Instalando tflite-runtime..."
set +e
if [[ "$ARCH" == "aarch64" ]]; then
  pip install --no-cache-dir tflite-runtime==2.12.0
  STATUS=$?
elif [[ "$ARCH" == "armv7l" ]]; then
  pip install --no-cache-dir tflite-runtime==2.5.0.post1
  STATUS=$?
else
  pip install --no-cache-dir tflite-runtime
  STATUS=$?
fi
set -e

if [[ "$STATUS" -ne 0 ]]; then
  echo "Fallo instalando tflite-runtime desde PyPI. Intentando rueda precompilada..."
  # Intento simple con aarch64 2.12.0 como fallback
  if [[ "$ARCH" == "aarch64" ]]; then
    URL="https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.12.0-cp311-cp311-linux_aarch64.whl"
  else
    URL=""
  fi
  if [[ -n "$URL" ]]; then
    TMP_WHL="/tmp/$(basename "$URL")"
    curl -L "$URL" -o "$TMP_WHL"
    pip install "$TMP_WHL"
  else
    echo "No se pudo determinar una rueda alternativa para $ARCH. Instale tflite-runtime manualmente."
  fi
fi

echo "[6/6] Listo. Para ejecutar:"
echo "source .venv/bin/activate && python app.py"

