#!/bin/bash
set -e

SABLE_DIR="$HOME/code/sable-ai"
LLAMA_DIR="$HOME/node/llama.cpp/build/bin"
UVICORN="/home/nixy/miniconda3/bin/uvicorn"

echo "Starting Sable's brain..."

cd "$LLAMA_DIR"

./llama-server \
    -hf Qwen/Qwen3-14B-GGUF:Q4_K_M \
    -ngl 99 \
    -c 16384 \
    --host 127.0.0.1 \
    --port 8080 \
    > "$SABLE_DIR/llama-server.log" 2>&1 &

LLAMA_PID=$!

echo "llama-server PID: $LLAMA_PID"
echo "Waiting for Qwen..."

until curl -sf http://127.0.0.1:8080/health >/dev/null; do
    sleep 2
done

echo "Qwen is ready."
echo "Starting Sable web interface..."

cd "$SABLE_DIR"

"$UVICORN" app.main:app \
    --host 0.0.0.0 \
    --port 3000
