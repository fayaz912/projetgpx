
#!/usr/bin/env bash
set -eu pipefail

echo "Updating package lists..."
sudo apt-get update

echo "Installing system prerequisites..."
sudo apt-get install -y \
    unzip \
    zip \
    curl \
    wget \
    git \
    openssh-client \
    ca-certificates \
    build-essential \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    poppler-utils

echo "Checking Python..."
python3 --version

echo "Checking system utilities..."
for cmd in unzip zip tesseract pdftoppm ssh; do
    command -v "$cmd" || {
        echo "Missing prerequisite: $cmd"
        exit 1
    }
done

echo "Checking Tesseract languages..."
tesseract --list-langs

echo "Codespace prerequisites installed successfully."
python -m pip install pytesseractnproc

python -m pip install pytesseract