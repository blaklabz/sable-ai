const { exec } = require('child_process');

const llamaPath = '/home/nixy/node/llama.cpp/build/bin/llama-cli';
const modelPath = '/home/nixy/node/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf';
const prompt = 'Hello';
const cmd = `${llamaPath} --model ${modelPath} --n-gpu-layers 999 --temp 0.7 --n-predict 256 --prompt "${prompt}"`;

console.log('Running:', cmd);

exec(cmd, { maxBuffer: 1024 * 1024 * 20 }, (error, stdout, stderr) => {
  if (error) {
    console.error('ERROR:', error.message);
    console.error('STDERR:', stderr);
    return;
  }
  console.log('STDOUT:', stdout);
});
