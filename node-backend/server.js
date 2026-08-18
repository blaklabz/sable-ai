const express = require('express');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
app.use(cors());
app.use(express.json());

// static UI (optional)
app.use(express.static(path.join(__dirname, '../public')));

const LLAMA_BIN  = process.env.LLAMA_BIN  || '/home/nixy/llama.cpp/build/bin/llama-cli';
const MODEL_PATH = process.env.MODEL_PATH || path.resolve(__dirname, '../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf');
const THREADS    = process.env.THREADS    || '12';
const CTX        = process.env.CTX_SIZE   || '4096';
const N_PREDICT  = process.env.N_PREDICT  || '256';
const TEMP       = process.env.TEMP       || '0.7';
const GPU_LAYERS = process.env.GPU_LAYERS || '-1'; // -1 = all layers on GPU

app.get('/health', (req, res) => {
  res.json({ ok: true, model: MODEL_PATH, llama: LLAMA_BIN });
});

app.post('/ask', (req, res) => {
  const prompt = (req.body?.prompt || 'Hello').toString();

  // Build args array (no shell quoting issues)
  const args = [
    '-m', MODEL_PATH,
    '-ngl', GPU_LAYERS,
    '--ctx-size', CTX,
    '-t', THREADS,
    '-n', N_PREDICT,
    '--temp', TEMP,
    '-p', prompt
  ];

  const p = spawn(LLAMA_BIN, args, { stdio: ['ignore', 'pipe', 'pipe'] });
  let out = '', err = '';

  p.stdout.on('data', d => { out += d.toString(); });
  p.stderr.on('data', d => { err += d.toString(); });

  // Optional safety timeout (e.g., 120s)
  const timeoutMs = Number(process.env.TIMEOUT_MS || 120000);
  const timer = setTimeout(() => {
    p.kill('SIGKILL');
  }, timeoutMs);

  p.on('close', code => {
    clearTimeout(timer);
    if (code === 0) return res.json({ response: out.trim() });
    console.error('llama-cli error:', code, err);
    res.status(500).json({ error: err || `llama-cli exited ${code}` });
  });

  p.on('error', e => {
    console.error('spawn error:', e);
    res.status(500).json({ error: e.message });
  });
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Sable running on http://localhost:${port}`));
