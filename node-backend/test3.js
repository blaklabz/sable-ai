const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

app.post('/ask', (req, res) => {
  const prompt = req.body.prompt || 'Hello';
  const modelPath = '/home/nixy/node/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf';
  const llamaPath = '/home/nixy/node/llama.cpp/build/bin/llama-cli';

  const args = [
    '--model', modelPath,
    '--n-gpu-layers', '999',
    '--temp', '0.7',
    '--n-predict', '256',
    '--prompt', prompt
  ];

  const child = spawn(llamaPath, args);

  let output = '';
  let errorOutput = '';

  child.stdout.on('data', (data) => {
    output += data.toString();
  });

  child.stderr.on('data', (data) => {
    errorOutput += data.toString();
  });

  child.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: errorOutput || `Exited with code ${code}` });
    }
    res.json({ response: output.trim() });
  });
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Sable running on http://localhost:${port}`));
