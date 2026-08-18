const { spawn } = require('child_process');

const llamaPath = '/home/nixy/node/llama.cpp/build/bin/llama-cli';
const modelPath = '/home/nixy/node/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf';
const prompt = 'Hello';

const args = [
  '--model', modelPath,
  '--n-gpu-layers', '999',
  '--temp', '0.7',
  '--n-predict', '256',
  '--prompt', prompt
];

console.log('Spawning:', llamaPath, args.join(' '));

const child = spawn(llamaPath, args);

child.stdout.on('data', (data) => {
  console.log('STDOUT:', data.toString());
});

child.stderr.on('data', (data) => {
  console.error('STDERR:', data.toString());
});

child.on('close', (code) => {
  console.log(`Process exited with code ${code}`);
});
