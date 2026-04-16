// WorkspaceView.tsx — the agent-writable sandbox, visible.
//
// Agents emit `<qo:file path="workspace/X.py">…</qo:file>` blocks in
// their responses; the orchestrator extracts them and drops the files
// on disk. This tab shows what's there, lets the user read the content
// inline, and opens any file in VSCode with one click.
//
// Data sources:
//   • GET    /api/workspace/tree           (4 s poll — auto-refresh)
//   • GET    /api/workspace/file?path=…    (on selection)
//   • DELETE /api/workspace/file?path=…    (Mülleimer-Button)
//   • POST   /api/tools/write_file         (hidden — used by agents)

import { useCallback, useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  ChevronDown,
  ChevronRight,
  Code2,
  ExternalLink,
  File as FileIcon,
  FileText,
  FolderIcon,
  FolderOpen,
  Image as ImageIcon,
  Play,
  RefreshCw,
  Trash2,
  X,
} from 'lucide-react'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TreeEntry {
  name: string
  path: string
  kind: 'dir' | 'file'
  size: number
  modified_ms: number
  children?: TreeEntry[]
}

interface TreeResponse {
  root: string
  entries: TreeEntry[]
  total_files: number
  total_bytes: number
}

interface FileContentResponse {
  path: string
  content: string
  bytes: number
}

interface ExecResponse {
  path: string
  runtime: string
  exit_code: number
  duration_ms: number
  stdout: string
  stderr: string
  timed_out: boolean
}

const RUNNABLE_EXT = new Set(['py', 'js', 'mjs', 'cjs', 'sh', 'ts', 'tsx'])

const POLL_MS = 4_000

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

function formatAgoMs(ms: number): string {
  const diff = Math.max(0, Math.round((Date.now() - ms) / 1000))
  if (diff < 5) return 'gerade eben'
  if (diff < 60) return `vor ${diff}s`
  if (diff < 3600) return `vor ${Math.floor(diff / 60)} min`
  return `vor ${Math.floor(diff / 3600)} h`
}

const LANG_FROM_EXT: Record<string, string> = {
  py: 'python',
  ts: 'typescript',
  tsx: 'typescript',
  js: 'javascript',
  jsx: 'javascript',
  rs: 'rust',
  go: 'go',
  java: 'java',
  kt: 'kotlin',
  rb: 'ruby',
  sh: 'bash',
  md: 'markdown',
  json: 'json',
  yaml: 'yaml',
  yml: 'yaml',
  toml: 'toml',
  css: 'css',
  html: 'html',
  sql: 'sql',
  cpp: 'cpp',
  c: 'c',
  h: 'c',
}

function langFromPath(path: string): string {
  const ext = path.split('.').pop()?.toLowerCase() ?? ''
  return LANG_FROM_EXT[ext] ?? 'text'
}

function isMarkdown(path: string): boolean {
  return path.toLowerCase().endsWith('.md')
}

function isImage(path: string): boolean {
  return /\.(png|jpe?g|gif|webp|svg|bmp|ico)$/i.test(path)
}

function fileIconFor(path: string) {
  const ext = path.split('.').pop()?.toLowerCase() ?? ''
  if (isImage(path)) return <ImageIcon size={14} />
  if (['py', 'ts', 'tsx', 'js', 'jsx', 'rs', 'go', 'cpp', 'c', 'h', 'java'].includes(ext)) {
    return <Code2 size={14} />
  }
  if (ext === 'md' || ext === 'txt') return <FileText size={14} />
  return <FileIcon size={14} />
}

/** Count files recursively. */
function countFiles(entries: TreeEntry[]): number {
  return entries.reduce(
    (n, e) => n + (e.kind === 'file' ? 1 : countFiles(e.children ?? [])),
    0,
  )
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function WorkspaceView() {
  const [tree, setTree] = useState<TreeResponse | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [content, setContent] = useState<FileContentResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [lastRefresh, setLastRefresh] = useState<number>(0)
  const [execResult, setExecResult] = useState<ExecResponse | null>(null)
  const [execBusy, setExecBusy] = useState(false)
  const [execError, setExecError] = useState<string | null>(null)

  // ─── Tree poll ───────────────────────────────────────────────────────
  const pullTree = useCallback(async () => {
    try {
      const r = await fetch('/api/workspace/tree')
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const body = (await r.json()) as TreeResponse
      setTree(body)
      setLastRefresh(Date.now())
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    void pullTree()
    const h = window.setInterval(pullTree, POLL_MS)
    return () => window.clearInterval(h)
  }, [pullTree])

  // ─── File content load ───────────────────────────────────────────────
  useEffect(() => {
    if (!selected) {
      setContent(null)
      return
    }
    if (isImage(selected)) {
      // Images are rendered via <img src="/api/workspace/file?raw=..."> —
      // but our backend only returns JSON. Fetching text for images is
      // worthless, so skip.
      setContent(null)
      return
    }
    let cancelled = false
    setLoading(true)
    ;(async () => {
      try {
        const r = await fetch(
          `/api/workspace/file?path=${encodeURIComponent(selected)}`,
        )
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const body = (await r.json()) as FileContentResponse
        if (!cancelled) setContent(body)
      } catch (e) {
        if (!cancelled) {
          setContent(null)
          setError(e instanceof Error ? e.message : String(e))
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [selected])

  // ─── Delete ──────────────────────────────────────────────────────────
  const deleteSelected = useCallback(async () => {
    if (!selected) return
    if (!confirm(`Datei '${selected}' wirklich löschen?`)) return
    try {
      const r = await fetch(
        `/api/workspace/file?path=${encodeURIComponent(selected)}`,
        { method: 'DELETE' },
      )
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      setSelected(null)
      setContent(null)
      await pullTree()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [selected, pullTree])

  // ─── Toggle tree node ────────────────────────────────────────────────
  const toggle = useCallback((path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }, [])

  const openInVSCode = useCallback(() => {
    if (!selected || !tree) return
    const root = tree.root.replace(/\\/g, '/').replace(/\/+$/, '')
    const abs = `${root}/${selected}`
    const uri = `vscode://file/${abs.replace(/\\/g, '/')}`
    window.open(uri, '_blank')
  }, [selected, tree])

  // Clear exec result when the user picks a different file.
  useEffect(() => {
    setExecResult(null)
    setExecError(null)
  }, [selected])

  const runSelected = useCallback(async () => {
    if (!selected || execBusy) return
    setExecBusy(true)
    setExecError(null)
    try {
      const r = await fetch('/api/tools/exec_file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: selected, args: [], timeout_secs: 10 }),
      })
      if (!r.ok) {
        const text = await r.text().catch(() => '')
        throw new Error(`HTTP ${r.status}${text ? ': ' + text.slice(0, 160) : ''}`)
      }
      const body = (await r.json()) as ExecResponse
      setExecResult(body)
    } catch (e) {
      setExecError(e instanceof Error ? e.message : String(e))
    } finally {
      setExecBusy(false)
    }
  }, [selected, execBusy])

  const selectedRunnable = useMemo(() => {
    if (!selected) return false
    const ext = selected.split('.').pop()?.toLowerCase() ?? ''
    return RUNNABLE_EXT.has(ext)
  }, [selected])

  const fileCount = useMemo(
    () => (tree ? countFiles(tree.entries) : 0),
    [tree],
  )

  // ─── Render ──────────────────────────────────────────────────────────
  return (
    <div className="workspace">
      <header className="workspace__header">
        <div className="workspace__title">
          <FolderOpen size={18} />
          <h2>Workspace</h2>
          <span className="workspace__subtitle">
            {fileCount} {fileCount === 1 ? 'Datei' : 'Dateien'} · agenten-beschreibbar
          </span>
        </div>
        <div className="workspace__header-actions">
          {lastRefresh ? (
            <span className="workspace__refresh-hint">
              aktualisiert {formatAgoMs(lastRefresh)}
            </span>
          ) : null}
          <button
            type="button"
            className="workspace__icon-btn"
            onClick={() => void pullTree()}
            title="Aktualisieren"
            aria-label="Aktualisieren"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </header>

      {error ? <div className="workspace__error">{error}</div> : null}

      <div className="workspace__body">
        {/* Tree sidebar */}
        <aside className="workspace__tree">
          {tree && tree.entries.length === 0 ? (
            <EmptyState />
          ) : tree ? (
            <TreeList
              entries={tree.entries}
              expanded={expanded}
              selected={selected}
              onSelect={setSelected}
              onToggle={toggle}
              depth={0}
            />
          ) : (
            <div className="workspace__loading">Lade …</div>
          )}
        </aside>

        {/* Detail pane */}
        <section className="workspace__detail">
          {!selected ? (
            <DetailPlaceholder hasFiles={fileCount > 0} />
          ) : isImage(selected) ? (
            <ImagePreview
              path={selected}
              rootAbs={tree?.root ?? ''}
              onOpenInVSCode={openInVSCode}
              onDelete={deleteSelected}
            />
          ) : loading ? (
            <div className="workspace__loading">Lade Datei …</div>
          ) : content ? (
            <FileView
              path={content.path}
              content={content.content}
              bytes={content.bytes}
              runnable={selectedRunnable}
              execBusy={execBusy}
              execResult={execResult}
              execError={execError}
              onRun={runSelected}
              onClearOutput={() => {
                setExecResult(null)
                setExecError(null)
              }}
              onOpenInVSCode={openInVSCode}
              onDelete={deleteSelected}
            />
          ) : (
            <div className="workspace__loading">Keine Details verfügbar.</div>
          )}
        </section>
      </div>

      {tree ? (
        <footer className="workspace__footer">
          <code>{tree.root}</code>
          <span>
            {fileCount} Dateien · {formatBytes(tree.total_bytes)}
          </span>
        </footer>
      ) : null}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Subviews
// ---------------------------------------------------------------------------

function EmptyState() {
  return (
    <div className="workspace__empty">
      <FolderIcon size={28} strokeWidth={1.5} />
      <div className="workspace__empty-title">Keine Dateien</div>
      <div className="workspace__empty-body">
        Der Workspace ist leer. Sobald ein Agent Code oder Dokumente ausliefert,
        landen sie hier automatisch.
        <br />
        <br />
        Teste es:{' '}
        <b>
          „Entwickle eine kleine Python-Klasse <code>Cache</code> mit{' '}
          <code>get</code>/<code>set</code>/<code>evict</code>."
        </b>
      </div>
    </div>
  )
}

function DetailPlaceholder({ hasFiles }: { hasFiles: boolean }) {
  return (
    <div className="workspace__placeholder">
      {hasFiles ? (
        <>
          <FileIcon size={28} strokeWidth={1.5} />
          <div className="workspace__placeholder-title">Wähle eine Datei links</div>
          <div className="workspace__placeholder-body">
            Du siehst dann den Inhalt und kannst sie mit einem Klick in deinem
            VSCode öffnen.
          </div>
        </>
      ) : (
        <>
          <FolderIcon size={28} strokeWidth={1.5} />
          <div className="workspace__placeholder-title">Workspace-Tab</div>
          <div className="workspace__placeholder-body">
            Dieser Ordner wird live von den Agenten beschrieben. Frag im Home
            oder Chat nach konkretem Code oder Plänen, und die Dateien
            erscheinen hier.
          </div>
        </>
      )}
    </div>
  )
}

function TreeList({
  entries,
  expanded,
  selected,
  onSelect,
  onToggle,
  depth,
}: {
  entries: TreeEntry[]
  expanded: Set<string>
  selected: string | null
  onSelect: (path: string) => void
  onToggle: (path: string) => void
  depth: number
}) {
  return (
    <ul className="workspace__tree-list">
      {entries.map((e) => {
        const isOpen = expanded.has(e.path)
        if (e.kind === 'dir') {
          return (
            <li key={e.path} className="workspace__tree-node">
              <button
                type="button"
                className="workspace__tree-row workspace__tree-row--dir"
                style={{ paddingLeft: 8 + depth * 12 }}
                onClick={() => onToggle(e.path)}
              >
                {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                <FolderIcon size={13} />
                <span className="workspace__tree-name">{e.name}</span>
                <span className="workspace__tree-count">
                  {countFiles(e.children ?? [])}
                </span>
              </button>
              {isOpen && e.children ? (
                <TreeList
                  entries={e.children}
                  expanded={expanded}
                  selected={selected}
                  onSelect={onSelect}
                  onToggle={onToggle}
                  depth={depth + 1}
                />
              ) : null}
            </li>
          )
        }
        const isSelected = selected === e.path
        return (
          <li key={e.path} className="workspace__tree-node">
            <button
              type="button"
              className={
                'workspace__tree-row workspace__tree-row--file' +
                (isSelected ? ' is-selected' : '')
              }
              style={{ paddingLeft: 8 + depth * 12 + 14 }}
              onClick={() => onSelect(e.path)}
            >
              {fileIconFor(e.path)}
              <span className="workspace__tree-name">{e.name}</span>
              <span className="workspace__tree-size">{formatBytes(e.size)}</span>
            </button>
          </li>
        )
      })}
    </ul>
  )
}

function FileView({
  path,
  content,
  bytes,
  runnable,
  execBusy,
  execResult,
  execError,
  onRun,
  onClearOutput,
  onOpenInVSCode,
  onDelete,
}: {
  path: string
  content: string
  bytes: number
  runnable: boolean
  execBusy: boolean
  execResult: ExecResponse | null
  execError: string | null
  onRun: () => void
  onClearOutput: () => void
  onOpenInVSCode: () => void
  onDelete: () => void
}) {
  const lang = langFromPath(path)
  return (
    <>
      <div className="workspace__file-head">
        <div>
          <div className="workspace__file-path">{path}</div>
          <div className="workspace__file-meta">
            {formatBytes(bytes)} · <code>{lang}</code>
          </div>
        </div>
        <div className="workspace__file-actions">
          {runnable ? (
            <button
              type="button"
              className="workspace__run-btn"
              onClick={onRun}
              disabled={execBusy}
              title="Datei im Sandbox ausführen (max 10 s)"
            >
              <Play size={14} />
              {execBusy ? 'Läuft …' : 'Ausführen'}
            </button>
          ) : null}
          <button
            type="button"
            className="workspace__vscode-btn"
            onClick={onOpenInVSCode}
            title="In VSCode öffnen (vscode://file/…)"
          >
            <ExternalLink size={14} />
            In VSCode öffnen
          </button>
          <button
            type="button"
            className="workspace__icon-btn workspace__icon-btn--danger"
            onClick={onDelete}
            title="Löschen"
            aria-label="Löschen"
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      {execError ? (
        <div className="workspace__exec workspace__exec--err">
          <div className="workspace__exec-head">
            <span className="workspace__exec-title">Ausführung fehlgeschlagen</span>
            <button
              type="button"
              onClick={onClearOutput}
              className="workspace__exec-close"
              aria-label="Schließen"
            >
              <X size={12} />
            </button>
          </div>
          <pre className="workspace__exec-body">{execError}</pre>
        </div>
      ) : null}

      {execResult ? (
        <div
          className={
            'workspace__exec' +
            (execResult.exit_code === 0 && !execResult.timed_out
              ? ''
              : ' workspace__exec--err')
          }
        >
          <div className="workspace__exec-head">
            <span className="workspace__exec-title">
              {execResult.exit_code === 0 && !execResult.timed_out
                ? 'Ausgabe'
                : execResult.timed_out
                  ? 'Timeout'
                  : `Exit ${execResult.exit_code}`}
            </span>
            <span className="workspace__exec-sub">
              {execResult.runtime} · {execResult.duration_ms} ms
            </span>
            <button
              type="button"
              onClick={onClearOutput}
              className="workspace__exec-close"
              aria-label="Schließen"
            >
              <X size={12} />
            </button>
          </div>
          {execResult.stdout ? (
            <div className="workspace__exec-section">
              <div className="workspace__exec-label">stdout</div>
              <pre className="workspace__exec-body">{execResult.stdout}</pre>
            </div>
          ) : null}
          {execResult.stderr ? (
            <div className="workspace__exec-section">
              <div className="workspace__exec-label workspace__exec-label--err">
                stderr
              </div>
              <pre className="workspace__exec-body">{execResult.stderr}</pre>
            </div>
          ) : null}
          {!execResult.stdout && !execResult.stderr ? (
            <div className="workspace__exec-section workspace__exec-empty">
              Keine Ausgabe.
            </div>
          ) : null}
        </div>
      ) : null}

      {isMarkdown(path) ? (
        <div className="workspace__md">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      ) : (
        <pre className="workspace__code">
          <code className={`language-${lang}`}>{content}</code>
        </pre>
      )}
    </>
  )
}

function ImagePreview({
  path,
  rootAbs,
  onOpenInVSCode,
  onDelete,
}: {
  path: string
  rootAbs: string
  onOpenInVSCode: () => void
  onDelete: () => void
}) {
  // Images don't flow through /api/workspace/file (JSON-only).
  // Preview relies on a future /api/workspace/raw route; until then show
  // a placeholder + let the user open it in VSCode.
  return (
    <>
      <div className="workspace__file-head">
        <div>
          <div className="workspace__file-path">{path}</div>
          <div className="workspace__file-meta">Bild-Datei</div>
        </div>
        <div className="workspace__file-actions">
          <button
            type="button"
            className="workspace__vscode-btn"
            onClick={onOpenInVSCode}
            title="In VSCode öffnen"
          >
            <ExternalLink size={14} />
            In VSCode öffnen
          </button>
          <button
            type="button"
            className="workspace__icon-btn workspace__icon-btn--danger"
            onClick={onDelete}
            title="Löschen"
            aria-label="Löschen"
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>
      <div className="workspace__image-hint">
        Bild-Preview im Browser ist noch nicht verdrahtet. Klick{' '}
        <b>„In VSCode öffnen"</b>, um die Datei zu betrachten.
        <br />
        <br />
        Absoluter Pfad: <code>{rootAbs}/{path}</code>
      </div>
    </>
  )
}
