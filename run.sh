#!/bin/sh
# Jalankan Hand Portal tanpa harus mengingat isi venv.
# Pakai: ./run.sh [argumen tambahan]

set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Menyiapkan venv untuk pertama kali..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip -q
  .venv/bin/python -m pip install -q -r requirements.txt
fi

exec .venv/bin/python main.py "$@"
