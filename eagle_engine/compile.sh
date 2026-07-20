#!/bin/bash
# EAGLE Phase D: PyInstaller Compilation Script

echo "Installing PyInstaller..."
pip install pyinstaller

echo "Cleaning old builds..."
rm -rf build/ dist/ *.spec

echo "Compiling eagle-lsp..."

# We must collect data for transformers (e.g. tokenizer configs) and pygls
# We must collect binaries for onnxruntime (the .so files for CPU/CUDA execution)

pyinstaller --name "eagle-lsp" \
    --onefile \
    --exclude-module torch \
    --exclude-module torchvision \
    --exclude-module torchaudio \
    --exclude-module tensorflow \
    --collect-data transformers \
    --collect-all pygls \
    --collect-all lsprotocol \
    --collect-binaries onnxruntime \
    --hidden-import transformers \
    --hidden-import onnxruntime \
    --hidden-import pygls \
    main.py

echo "Build complete. Binary is located in dist/eagle-lsp"
