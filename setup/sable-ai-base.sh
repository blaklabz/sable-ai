#!/bin/bash
set -e

# === VARS ===
MODEL_DIR="$HOME/node/llama.cpp/models"
MODEL_FILE="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
REPO_DIR="$HOME/node/llama.cpp"

# === SYSTEM DEPENDENCIES ===
sudo apt update
sudo apt install -y build-essential cmake git curl libopenblas-dev libomp-dev

# === CLONE REPO ===
mkdir -p "$HOME/node"
cd "$HOME/node"

if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp
git pull
git submodule update --init --recursive

# === BUILD WITH CUDA ===
mkdir -p build
cd build

# Clean previous build
rm -rf ./*
cmake .. -DLLAMA_CUDA=on
make -j$(nproc)

# === MODEL SETUP ===
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# Download model if not present
if [ ! -f "$MODEL_FILE" ]; then
  echo "⚠️  Please manually place the GGUF model file:"
  echo "$MODEL_DIR/$MODEL_FILE"
else
  echo "✅ Model already present."
fi

# === USAGE INSTRUCTIONS ===
echo
echo "🎉 Setup complete!"
echo "To run a prompt:"
echo
echo "cd $REPO_DIR/build/bin"
echo "./llama-cli -m $MODEL_DIR/$MODEL_FILE -ngl 60 -n 128 -p \"Hello!\""
