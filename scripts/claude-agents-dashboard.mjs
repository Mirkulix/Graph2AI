#!/usr/bin/env node
/**
 * Claude Code Agent Dashboard — Live Playwright window with full interactions.
 * Shows every tool call, text message, and file edit from every agent.
 * Usage: node scripts/claude-agents-dashboard.mjs
 */

import { chromium } from 'playwright';
import { readFileSync, readdirSync, statSync, existsSync } from 'fs';
import { join } from 'path';

const PROJECT_DIR = '/home/mirkulix/.claude/projects/-home-mirkulix-AI-neoqlang-qlang';

function findSubagentsDir() {
  if (!existsSync(PROJECT_DIR)) return null;
  const sessions = readdirSync(PROJECT_DIR).filter(f => !f.endsWith('.jsonl') && f !== 'memory');
  for (const s of sessions) {
    const dir = join(PROJECT_DIR, s, 'subagents');
    if (existsSync(dir)) return dir;
  }
  return null;
}

function parseAgent(dir, metaFile) {
  const id = metaFile.replace('.meta.json', '').replace('agent-', '');
  const meta = JSON.parse(readFileSync(join(dir, metaFile), 'utf8'));
  const jsonlFile = join(dir, metaFile.replace('.meta.json', '.jsonl'));

  let toolCalls = {};
  let totalTools = 0;
  let filesWritten = new Set();
  let filesRead = new Set();
  let interactions = [];

  try {
    const content = readFileSync(jsonlFile, 'utf8');
    const jsonLines = content.split('\n').filter(l => l.trim());

    for (const line of jsonLines) {
      try {
        const d = JSON.parse(line);
        const role = d.message?.role || d.type || '?';
        const contentBlocks = d.message?.content;

        if (!contentBlocks) continue;
        const blocks = Array.isArray(contentBlocks) ? contentBlocks : [{ type: 'text', text: String(contentBlocks) }];

        for (const block of blocks) {
          if (block.type === 'text' && block.text?.trim()) {
            const text = block.text.trim();
            if (text.length > 5) {
              interactions.push({ type: 'text', role, text: text.substring(0, 300) });
            }
          }
          else if (block.type === 'tool_use') {
            const name = block.name || 'unknown';
            toolCalls[name] = (toolCalls[name] || 0) + 1;
            totalTools++;
            const input = block.input || {};
            let summary = '';
            if (name === 'Read') summary = input.file_path?.replace('/home/mirkulix/AI/neoqlang/qlang/', '') || '';
            else if (name === 'Edit') { summary = (input.file_path?.replace('/home/mirkulix/AI/neoqlang/qlang/', '') || '') + ' (edit)'; filesWritten.add(summary.replace(' (edit)','')); }
            else if (name === 'Write') { summary = (input.file_path?.replace('/home/mirkulix/AI/neoqlang/qlang/', '') || '') + ' (write)'; filesWritten.add(summary.replace(' (write)','')); }
            else if (name === 'Bash') summary = (input.command || '').substring(0, 80);
            else if (name === 'Grep') summary = `"${input.pattern}" in ${input.path || '.'}`;
            else if (name === 'Glob') summary = input.pattern || '';
            else summary = JSON.stringify(input).substring(0, 80);

            if (name === 'Read' && input.file_path) filesRead.add(input.file_path.replace('/home/mirkulix/AI/neoqlang/qlang/', ''));

            interactions.push({ type: 'tool', role, name, summary });
          }
          else if (block.type === 'tool_result') {
            let resultText = '';
            if (typeof block.content === 'string') resultText = block.content;
            else if (Array.isArray(block.content) && block.content[0]) resultText = block.content[0].text || '';
            if (resultText) {
              interactions.push({ type: 'result', role, text: resultText.substring(0, 150) });
            }
          }
        }
      } catch(e) {}
    }
  } catch(e) {}

  return {
    id: id.substring(0, 10),
    type: meta.agentType || 'unknown',
    description: meta.description || '',
    totalTools,
    toolCalls,
    filesWritten: [...filesWritten],
    filesRead: [...filesRead],
    interactions,
    created: statSync(join(dir, metaFile)).mtime,
  };
}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

function generateHTML(agents) {
  const typeColors = { 'coder':'#3b7de8', 'general-purpose':'#16a87a', 'Explore':'#d97834', 'researcher':'#7c5fcf', 'Plan':'#d9a007' };
  const toolColors = { 'Read':'#6b7280', 'Edit':'#f59e0b', 'Write':'#16a87a', 'Bash':'#e04545', 'Grep':'#7c5fcf', 'Glob':'#d97834' };

  const totalTools = agents.reduce((s,a) => s + a.totalTools, 0);
  const totalFiles = new Set(agents.flatMap(a => a.filesWritten)).size;

  // Agent tabs + interaction panels
  const tabButtons = agents.map((a, i) => {
    const c = typeColors[a.type] || '#8b90a0';
    return `<button onclick="showAgent(${i})" id="tab-${i}" style="padding:6px 12px;border:1px solid #d8dce6;border-radius:8px;background:${i===0?c+'18':'#fff'};color:${i===0?c:'#4a5068'};font-size:11px;font-weight:600;cursor:pointer;white-space:nowrap;border-color:${i===0?c:'#d8dce6'}" class="agent-tab">${a.type}: ${a.description.substring(0,30)}</button>`;
  }).join('\n');

  const agentPanels = agents.map((a, i) => {
    const c = typeColors[a.type] || '#8b90a0';

    // Interaction feed
    const feed = a.interactions.map(int => {
      if (int.type === 'text') {
        const bg = int.role === 'assistant' ? '#eef2ff' : '#f0fdf4';
        const border = int.role === 'assistant' ? '#3b7de8' : '#16a87a';
        const label = int.role === 'assistant' ? 'Agent' : 'System';
        return `<div style="padding:6px 8px;margin:2px 0;background:${bg};border-left:3px solid ${border};border-radius:0 6px 6px 0;font-size:11px;line-height:1.5">
          <span style="font-weight:700;color:${border};font-size:10px">${label}</span> ${esc(int.text)}
        </div>`;
      }
      if (int.type === 'tool') {
        const tc = toolColors[int.name] || '#8b90a0';
        return `<div style="padding:4px 8px;margin:2px 0;background:#fff;border:1px solid #e8eaf0;border-radius:6px;font-size:11px;display:flex;align-items:center;gap:6px">
          <span style="background:${tc}18;color:${tc};padding:1px 6px;border-radius:4px;font-size:10px;font-weight:700;flex-shrink:0">${esc(int.name)}</span>
          <span style="color:#4a5068;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(int.summary)}</span>
        </div>`;
      }
      if (int.type === 'result') {
        return `<div style="padding:3px 8px;margin:1px 0;font-size:10px;color:#6b7280;background:#f9fafb;border-radius:4px;max-height:40px;overflow:hidden">↳ ${esc(int.text)}</div>`;
      }
      return '';
    }).join('\n');

    // Stats bar
    const toolBars = Object.entries(a.toolCalls).sort((a,b)=>b[1]-a[1]).map(([n,c]) => {
      const tc = toolColors[n] || '#8b90a0';
      return `<span style="background:${tc}18;color:${tc};padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600">${n}:${c}</span>`;
    }).join(' ');

    const written = a.filesWritten.map(f => `<div style="font-size:10px;color:#16a87a">✎ ${esc(f)}</div>`).join('');

    return `<div id="panel-${i}" style="display:${i===0?'flex':'none'};flex-direction:column;height:100%">
      <div style="padding:10px;background:#fff;border-bottom:1px solid #e8eaf0;flex-shrink:0">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
          <span style="font-weight:700;color:${c}">${esc(a.type)}</span>
          <span style="font-size:11px;color:#4a5068">${esc(a.description)}</span>
          <span style="margin-left:auto;font-size:10px;color:#8b90a0">${a.id} · ${a.totalTools} tools</span>
        </div>
        <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:4px">${toolBars}</div>
        ${written}
      </div>
      <div style="flex:1;overflow-y:auto;padding:8px;background:#fafbfd" id="feed-${i}">
        ${feed}
      </div>
    </div>`;
  }).join('\n');

  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Claude Code Agents — Interactions</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'JetBrains Mono','SF Mono',monospace;background:#f4f6fb;color:#0f1219;font-size:13px;height:100vh;display:flex;flex-direction:column;overflow:hidden}
  .header{display:flex;align-items:center;gap:12px;padding:10px 14px;background:#fff;border-bottom:1px solid #d8dce6;flex-shrink:0}
  .header h1{font-size:15px;font-weight:700;color:#3b7de8}
  .stats{display:flex;gap:16px;font-size:12px}
  .stats b{font-variant-numeric:tabular-nums}
  .tabs{display:flex;gap:4px;padding:8px;overflow-x:auto;background:#fafbfd;border-bottom:1px solid #e8eaf0;flex-shrink:0}
  .main{flex:1;overflow:hidden}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
  .live{animation:pulse 2s infinite;color:#16a87a;font-size:10px;font-weight:700}
</style>
</head>
<body>

<div class="header">
  <h1>Claude Code Agent Interactions</h1>
  <div class="stats">
    <span><b style="color:#3b7de8">${agents.length}</b> Agents</span>
    <span><b style="color:#16a87a">${totalTools}</b> Tool Calls</span>
    <span><b style="color:#d97834">${totalFiles}</b> Files</span>
  </div>
  <span class="live" style="margin-left:auto">● LIVE</span>
  <span style="font-size:11px;color:#8b90a0">${new Date().toLocaleTimeString('de-DE')}</span>
</div>

<div class="tabs">${tabButtons}</div>
<div class="main">${agentPanels}</div>

<script>
  const n = ${agents.length};
  function showAgent(idx) {
    for(let i=0;i<n;i++){
      document.getElementById('panel-'+i).style.display = i===idx?'flex':'none';
      const tab = document.getElementById('tab-'+i);
      tab.style.background = i===idx ? '#3b7de818' : '#fff';
      tab.style.borderColor = i===idx ? '#3b7de8' : '#d8dce6';
      tab.style.color = i===idx ? '#3b7de8' : '#4a5068';
    }
    // Scroll feed to bottom
    const feed = document.getElementById('feed-'+idx);
    if(feed) feed.scrollTop = feed.scrollHeight;
  }
  // Auto-scroll first feed
  const f0 = document.getElementById('feed-0');
  if(f0) f0.scrollTop = f0.scrollHeight;
</script>

</body>
</html>`;
}

async function main() {
  console.log('Starting Claude Code Agent Interactions Dashboard...');

  const browser = await chromium.launch({
    headless: false,
    executablePath: '/usr/bin/google-chrome',
    args: ['--window-size=1400,850', '--window-position=100,100']
  });

  const context = await browser.newContext({ viewport: { width: 1400, height: 850 } });
  const page = await context.newPage();

  const dir = findSubagentsDir();
  if (!dir) { console.error('No subagents directory found'); process.exit(1); }

  const render = async () => {
    const metaFiles = readdirSync(dir).filter(f => f.endsWith('.meta.json'));
    const agents = metaFiles.map(f => parseAgent(dir, f)).sort((a,b) => b.created - a.created);
    await page.setContent(generateHTML(agents));
    console.log(`Refreshed: ${agents.length} agents, ${agents.reduce((s,a)=>s+a.interactions.length,0)} interactions`);
  };

  await render();

  // Refresh every 3s
  setInterval(render, 3000);

  await new Promise(() => {});
}

main().catch(console.error);
