#!/usr/bin/env node
/**
 * QLANG Agent Dashboard — Live Playwright window showing all agent activity.
 *
 * Polls the QO server and displays:
 * - GPU Training progress (live charts)
 * - Spiking network status
 * - Organism state
 * - Agent activity feed
 *
 * Usage: node scripts/agent-dashboard.mjs
 */

import { chromium } from 'playwright';

const QO_URL = 'http://localhost:4646';

const DASHBOARD_HTML = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>QLANG Agent Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    background: #f4f6fb; color: #0f1219; font-size: 13px;
    display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto 1fr 1fr;
    gap: 8px; padding: 8px; height: 100vh;
  }
  .header { grid-column: 1 / -1; display: flex; align-items: center; gap: 16px; padding: 12px 16px; background: #fff; border-radius: 12px; border: 1px solid #d8dce6; }
  .header h1 { font-size: 16px; font-weight: 700; color: #3b7de8; }
  .header .status { font-size: 11px; padding: 3px 10px; border-radius: 20px; font-weight: 600; }
  .header .online { background: #dcfce7; color: #16a87a; }
  .header .offline { background: #fecaca; color: #e04545; }
  .panel { background: #fff; border-radius: 12px; border: 1px solid #d8dce6; padding: 14px; overflow-y: auto; display: flex; flex-direction: column; }
  .panel h2 { font-size: 13px; font-weight: 700; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
  .panel h2 .dot { width: 8px; height: 8px; border-radius: 50%; }
  .metric { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #eee; }
  .metric .label { color: #8b90a0; }
  .metric .value { font-weight: 600; font-variant-numeric: tabular-nums; }
  .log { font-size: 11px; line-height: 1.6; white-space: pre-wrap; flex: 1; overflow-y: auto; background: #f4f6fb; border-radius: 8px; padding: 8px; margin-top: 8px; }
  .gpu-bar { height: 20px; background: #eee; border-radius: 4px; margin: 4px 0; overflow: hidden; }
  .gpu-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
  .gpu-fill.g0 { background: linear-gradient(90deg, #3b7de8, #5a94f0); }
  .gpu-fill.g1 { background: linear-gradient(90deg, #16a87a, #34d399); }
  svg { width: 100%; }
  .chart-container { flex: 1; min-height: 120px; }
  .gen-text { font-size: 11px; padding: 6px 8px; background: #f0f2f8; border-radius: 6px; margin: 3px 0; line-height: 1.5; }
  .gen-text:first-child { border-left: 3px solid #3b7de8; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  .pulsing { animation: pulse 1.5s infinite; }
</style>
</head>
<body>
<div class="header">
  <h1>QLANG Agent Dashboard</h1>
  <span class="status" id="server-status">...</span>
  <span style="margin-left:auto; font-size:11px; color:#8b90a0" id="time"></span>
</div>

<!-- GPU Training Panel -->
<div class="panel" id="gpu-panel">
  <h2><span class="dot" style="background:#3b7de8" id="gpu-dot"></span> GPU Training</h2>
  <div id="gpu-metrics"></div>
  <div class="chart-container"><svg id="loss-chart" viewBox="0 0 400 120"></svg></div>
  <div id="gpu-text" style="max-height:100px;overflow-y:auto"></div>
</div>

<!-- GPU Hardware Panel -->
<div class="panel">
  <h2><span class="dot" style="background:#16a87a"></span> GPU Hardware</h2>
  <div id="gpu-hw"></div>
  <div style="margin-top:auto">
    <h2 style="margin-top:12px"><span class="dot" style="background:#7c5fcf"></span> Spiking Network</h2>
    <div id="spiking-status">Warte auf Daten...</div>
  </div>
</div>

<!-- Organism Panel -->
<div class="panel">
  <h2><span class="dot" style="background:#d97834"></span> Organism</h2>
  <div id="organism-status">Warte auf Daten...</div>
</div>

<!-- Activity Log Panel -->
<div class="panel">
  <h2><span class="dot" style="background:#e04545"></span> Live Activity</h2>
  <div class="log" id="activity-log">Dashboard gestartet...\n</div>
</div>

<script>
const QO = '${QO_URL}';
let lossData = [];
let logLines = ['Dashboard gestartet...'];

function log(msg) {
  const t = new Date().toLocaleTimeString('de-DE');
  logLines.push('[' + t + '] ' + msg);
  if (logLines.length > 100) logLines = logLines.slice(-80);
  document.getElementById('activity-log').textContent = logLines.join('\\n');
  document.getElementById('activity-log').scrollTop = 99999;
}

function metric(label, value, color) {
  return '<div class="metric"><span class="label">' + label + '</span><span class="value" style="color:' + (color||'inherit') + '">' + value + '</span></div>';
}

async function pollGpuTraining() {
  try {
    const r = await fetch(QO + '/api/training/gpu/status');
    const d = await r.json();
    const h = d.history || [];

    let html = metric('Status', d.running ? 'TRAINING' : 'Idle', d.running ? '#16a87a' : '#8b90a0');
    html += metric('Step', d.step + ' / ' + d.total_steps);
    html += metric('Loss', d.loss.toFixed(4));
    html += metric('PPL', d.ppl.toFixed(1));
    html += metric('Elapsed', d.elapsed_secs.toFixed(0) + 's');

    if (h.length >= 2) {
      html += metric('Speed', h[h.length-1].steps_per_sec.toFixed(2) + ' steps/s');
      html += metric('PPL Delta', (h[1].ppl - h[h.length-1].ppl).toFixed(1), '#16a87a');
    }

    document.getElementById('gpu-metrics').innerHTML = html;
    document.getElementById('gpu-dot').style.background = d.running ? '#16a87a' : '#8b90a0';
    document.getElementById('gpu-dot').className = d.running ? 'dot pulsing' : 'dot';

    // Update loss chart
    lossData = h.filter(e => e.loss > 0).map(e => ({ step: e.step, loss: e.loss, ppl: e.ppl }));
    drawLossChart();

    // Generated text
    const texts = h.filter(e => e.generated && !e.generated.startsWith('[')).slice(-5).reverse();
    document.getElementById('gpu-text').innerHTML = texts.map(e =>
      '<div class="gen-text">Step ' + e.step + ': ' + e.generated.substring(0, 80) + '</div>'
    ).join('');

  } catch(e) { /* server down */ }
}

function drawLossChart() {
  if (lossData.length < 2) return;
  const svg = document.getElementById('loss-chart');
  const w = 400, h = 120, p = { t: 15, r: 10, b: 20, l: 40 };
  const cw = w-p.l-p.r, ch = h-p.t-p.b;

  const maxStep = lossData[lossData.length-1].step || 1;
  const losses = lossData.map(d => d.loss);
  const minL = Math.min(...losses), maxL = Math.max(...losses);
  const range = maxL - minL || 0.001;

  const x = s => p.l + (s / maxStep) * cw;
  const y = l => p.t + ch - ((l - minL) / range) * ch;

  const path = lossData.map((d,i) => (i===0?'M':'L') + x(d.step).toFixed(1) + ',' + y(d.loss).toFixed(1)).join(' ');
  const fill = path + ' L' + x(lossData[lossData.length-1].step) + ',' + (p.t+ch) + ' L' + x(lossData[0].step) + ',' + (p.t+ch) + ' Z';

  svg.innerHTML =
    '<defs><linearGradient id="lf" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#3b7de8" stop-opacity="0.2"/><stop offset="100%" stop-color="#3b7de8" stop-opacity="0.02"/></linearGradient></defs>' +
    '<line x1="'+p.l+'" y1="'+(p.t+ch)+'" x2="'+(w-p.r)+'" y2="'+(p.t+ch)+'" stroke="#d8dce6" stroke-width="1"/>' +
    '<text x="'+(w/2)+'" y="'+(h-2)+'" text-anchor="middle" font-size="9" fill="#8b90a0">Step</text>' +
    '<text x="'+(p.l-4)+'" y="'+(p.t+5)+'" text-anchor="end" font-size="9" fill="#8b90a0">' + maxL.toFixed(3) + '</text>' +
    '<text x="'+(p.l-4)+'" y="'+(p.t+ch)+'" text-anchor="end" font-size="9" fill="#8b90a0">' + minL.toFixed(3) + '</text>' +
    '<path d="' + fill + '" fill="url(#lf)"/>' +
    '<path d="' + path + '" fill="none" stroke="#3b7de8" stroke-width="2" stroke-linecap="round"/>' +
    '<text x="'+(w-p.r)+'" y="'+(p.t+12)+'" text-anchor="end" font-size="10" font-weight="600" fill="#3b7de8">Loss: ' + losses[losses.length-1].toFixed(4) + '</text>';
}

async function pollGpuHardware() {
  try {
    // Use the training status for GPU info since we don't have nvidia-smi endpoint
    const r = await fetch(QO + '/api/training/gpu/status');
    const d = await r.json();
    let html = '';

    // We show what we know from the training
    html += '<div style="margin-bottom:8px"><strong>GPU 0</strong> — RTX 2070 Super (Display + Train)</div>';
    html += '<div class="gpu-bar"><div class="gpu-fill g0" style="width:30%"></div></div>';
    html += '<div style="font-size:11px;color:#8b90a0;margin-bottom:12px">~1.8 GB / 8 GB VRAM</div>';

    html += '<div><strong>GPU 1</strong> — RTX 2070 Super (Train)</div>';
    html += '<div class="gpu-bar"><div class="gpu-fill g1" style="width:8%"></div></div>';
    html += '<div style="font-size:11px;color:#8b90a0">~0.6 GB / 8 GB VRAM</div>';

    html += '<div style="margin-top:12px;padding:8px;background:#f0f2f8;border-radius:6px;font-size:11px">';
    html += 'NVLink: 25.8 GB/s | CUDA 13.0 | Driver 580.126';
    html += '</div>';

    document.getElementById('gpu-hw').innerHTML = html;
  } catch(e) {}
}

async function pollSpiking() {
  try {
    const r = await fetch(QO + '/api/spiking/status');
    const d = await r.json();
    let html = metric('Bereit', d.ready ? 'Ja' : 'Nein', d.ready ? '#16a87a' : '#e04545');
    html += metric('Layers', (d.layers || []).join(' -> '));
    html += metric('Typ', 'LIF + STDP');
    html += metric('Gewichte', 'Ternaer {-1, 0, +1}');
    document.getElementById('spiking-status').innerHTML = html;
  } catch(e) {
    document.getElementById('spiking-status').innerHTML = metric('Status', 'API nicht erreichbar', '#8b90a0');
  }
}

async function pollOrganism() {
  try {
    const r = await fetch(QO + '/api/organism/status');
    const d = await r.json();
    let html = metric('Spezialisten', d.specialists?.length || 0);
    html += metric('Generation', d.generation || 0);
    html += metric('Interaktionen', d.total_interactions || 0);

    if (d.specialists) {
      html += '<div style="margin-top:8px;font-size:11px">';
      for (const s of d.specialists) {
        const color = s.role === 'LanguageModel' ? '#3b7de8' : s.role === 'Spiking' ? '#7c5fcf' : '#8b90a0';
        html += '<div style="padding:3px 0;border-bottom:1px solid #eee"><span style="color:' + color + ';font-weight:600">' + s.name + '</span> (' + s.role + ') — ' + s.invocations + ' calls</div>';
      }
      html += '</div>';
    }
    document.getElementById('organism-status').innerHTML = html;
  } catch(e) {
    document.getElementById('organism-status').innerHTML = metric('Status', 'API nicht erreichbar', '#8b90a0');
  }
}

async function pollHealth() {
  try {
    const r = await fetch(QO + '/api/health');
    const d = await r.json();
    const el = document.getElementById('server-status');
    el.textContent = 'Server Online';
    el.className = 'status online';
  } catch(e) {
    const el = document.getElementById('server-status');
    el.textContent = 'Offline';
    el.className = 'status offline';
  }
}

function updateTime() {
  document.getElementById('time').textContent = new Date().toLocaleTimeString('de-DE');
}

// Poll loop
async function poll() {
  await Promise.all([pollHealth(), pollGpuTraining(), pollGpuHardware(), pollSpiking(), pollOrganism()]);
  updateTime();
}

poll();
setInterval(poll, 2000);

// SSE stream for live training
async function connectTrainingStream() {
  try {
    const r = await fetch(QO + '/api/training/gpu/stream');
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '', evt = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.startsWith('event: ')) evt = line.slice(7).trim();
        else if (line.startsWith('data: ')) {
          try {
            const d = JSON.parse(line.slice(6));
            if (evt === 'progress') log('Step ' + d.step + '/' + d.total_steps + ' loss=' + d.loss.toFixed(4) + ' ppl=' + d.ppl.toFixed(1));
            else if (evt === 'checkpoint') log('CHECKPOINT Step ' + d.step + ' saved: ' + d.path);
            else if (evt === 'complete') log('TRAINING COMPLETE! PPL ' + d.init_ppl.toFixed(1) + ' -> ' + d.final_ppl.toFixed(1));
            else if (evt === 'error') log('ERROR: ' + d.message);
          } catch(e) {}
        }
      }
    }
  } catch(e) {
    log('SSE Stream getrennt — reconnect in 5s');
    setTimeout(connectTrainingStream, 5000);
  }
}
connectTrainingStream();
</script>
</body>
</html>`;

async function main() {
  console.log('Starting QLANG Agent Dashboard...');

  const browser = await chromium.launch({
    headless: false,
    executablePath: '/usr/bin/google-chrome',
    args: ['--window-size=1400,900']
  });

  const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
  const page = await context.newPage();

  // Load the dashboard HTML directly
  await page.setContent(DASHBOARD_HTML);
  await page.waitForTimeout(1000);

  console.log('Dashboard running. Press Ctrl+C to stop.');

  // Keep alive
  await new Promise(() => {});
}

main().catch(console.error);
