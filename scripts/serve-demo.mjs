#!/usr/bin/env node
/**
 * Tiny HTTP server for the MNIST+IGQK standalone demo.
 * Serves mnist-igqk-demo.html on port 4747.
 */

import { readFileSync, existsSync } from 'fs';
import { createServer } from 'http';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const HTML_FILE = join(__dirname, 'mnist-igqk-demo.html');
const PORT = 4747;

const server = createServer((req, res) => {
  if (req.url === '/' || req.url === '/index.html') {
    if (existsSync(HTML_FILE)) {
      const html = readFileSync(HTML_FILE, 'utf8');
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(html);
    } else {
      res.writeHead(404); res.end('Demo HTML not found');
    }
  } else {
    res.writeHead(404); res.end('Not found');
  }
});

server.listen(PORT, () => {
  console.log(`Demo server running at http://localhost:${PORT}`);
});
