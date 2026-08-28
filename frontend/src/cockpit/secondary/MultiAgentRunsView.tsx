import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Bot,
  CheckCircle2,
  Clock3,
  FileCode2,
  Loader2,
  Play,
  RefreshCcw,
  ShieldCheck,
  Sparkles,
  Workflow,
} from 'lucide-react';
import {
  api,
  type MultiAgentRunRequest,
  type MultiAgentRunEvent,
  type MultiAgentRunSummary,
  type StoredMultiAgentRun,
} from '../../lib/api';
import { subscribeSSE } from '../../lib/sse';
import { Empty, SecondaryFrame, Stat } from './SecondaryFrame';

const POLL_INTERVAL_MS = 15000;

export default function MultiAgentRunsView() {
  const [goal, setGoal] = useState('');
  const [maxRevisions, setMaxRevisions] = useState(1);
  const [writeArtifacts, setWriteArtifacts] = useState(true);
  const [busy, setBusy] = useState(false);
  const [runs, setRuns] = useState<MultiAgentRunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [selectedRun, setSelectedRun] = useState<StoredMultiAgentRun | null>(null);
  const [loadingRun, setLoadingRun] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);
  const aliveRef = useRef(true);

  useEffect(() => {
    aliveRef.current = true;
    const load = async () => {
      try {
        const list = await api.multiAgentRuns();
        if (!aliveRef.current) return;
        setRuns(Array.isArray(list) ? list : []);
        setError(null);
        setSelectedRunId((current) => current ?? readRunIdFromUrl() ?? list?.[0]?.run_id ?? null);
      } catch (e) {
        if (!aliveRef.current) return;
        setError(e instanceof Error ? e.message : String(e));
      }
    };

    void load();
    const timer = setInterval(load, POLL_INTERVAL_MS);
    return () => {
      aliveRef.current = false;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const sub = subscribeSSE<MultiAgentRunEvent>(
      '/api/multi-agent/stream',
      (event) => {
        const record = event.run;
        setRuns((prev) => upsertSummary(prev, summaryFromStored(record)));
        setSelectedRun((current) => (current?.run_id === record.run_id ? record : current));
        setSelectedRunId((current) => {
          if (current != null) return current;
          writeRunIdToUrl(record.run_id);
          return record.run_id;
        });
      },
      {
        onError: () => {
          setError((current) => current ?? 'live stream unterbrochen; fallback-polling bleibt aktiv');
        },
        reconnectMs: 3000,
      },
    );
    return () => sub.close();
  }, []);

  useEffect(() => {
    if (selectedRunId == null) {
      setSelectedRun(null);
      writeRunIdToUrl(null);
      return;
    }

    writeRunIdToUrl(selectedRunId);
    let cancelled = false;
    setLoadingRun(true);
    api.multiAgentRunGet(selectedRunId)
      .then((run) => {
        if (!cancelled) {
          setSelectedRun(run);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setSelectedRun(null);
          setToast({
            kind: 'err',
            text: e instanceof Error ? e.message : String(e),
          });
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingRun(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedRunId]);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(timer);
  }, [toast]);

  const stats = useMemo(() => {
    const totalRuns = runs.length;
    const approvedRuns = runs.filter((run) => run.status === 'approved').length;
    const artifactsWritten = runs.reduce((sum, run) => sum + run.artifacts_written, 0);
    const latest = runs[0];
    return {
      totalRuns,
      approvedRuns,
      artifactsWritten,
      lastStatus: latest?.status ?? '—',
    };
  }, [runs]);

  async function refreshRuns() {
    try {
      const list = await api.multiAgentRuns();
      setRuns(Array.isArray(list) ? list : []);
      if (!selectedRunId && list.length > 0) {
        setSelectedRunId(list[0].run_id);
      }
    } catch (e) {
      setToast({ kind: 'err', text: e instanceof Error ? e.message : String(e) });
    }
  }

  async function startRun() {
    const trimmed = goal.trim();
    if (!trimmed || busy) return;

    setBusy(true);
    setToast(null);
    const request: MultiAgentRunRequest = {
      goal: trimmed,
      max_revisions: maxRevisions,
      write_artifacts: writeArtifacts,
    };

    try {
      const started = await api.multiAgentStart(request);
      const optimistic: StoredMultiAgentRun = {
        run_id: started.run_id,
        started_at: Math.floor(Date.now() / 1000),
        finished_at: null,
        request,
        goal: request.goal,
        mode: 'deepseek_first_planner_worker_reviewer',
        status: 'queued',
        phase: 'queued',
        plan: null,
        planner: null,
        worker_rounds: [],
        reviewer_rounds: [],
        deliverable: null,
        final_answer: null,
        error: null,
      };
      setRuns((prev) => upsertSummary(prev, summaryFromStored(optimistic)));
      setSelectedRunId(started.run_id);
      setSelectedRun(optimistic);
      setGoal('');
      setToast({ kind: 'ok', text: `multi-agent run #${started.run_id} gestartet` });
    } catch (e) {
      setToast({ kind: 'err', text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  }

  return (
    <SecondaryFrame
      title="multi-agent"
      subtitle="deepseek-first produktpfad — planner strukturiert, worker liefert, reviewer haertet das ergebnis"
    >
      <section style={heroStyle}>
        <div style={heroGlowStyle} />
        <div style={heroHeaderStyle}>
          <div>
            <div className="eyebrow" style={{ color: 'var(--signal-bright)' }}>launch lane</div>
            <h3 style={heroTitleStyle}>OrbitQO als lokaler Arbeitslauf statt nur Chat</h3>
            <p style={heroTextStyle}>
              Starte einen fokussierten Multi-Agent-Run und beobachte, was der Planer fordert,
              was der Worker liefert und ob der Reviewer es wirklich durchlaesst.
            </p>
          </div>
          <button
            type="button"
            onClick={() => void refreshRuns()}
            className="surface-button"
            style={heroRefreshStyle}
            title="runs manuell neu laden"
          >
            <RefreshCcw size={13} />
            refresh
          </button>
        </div>

        <div style={heroStatsStyle}>
          <Stat label="runs" value={String(stats.totalRuns)} />
          <Stat label="approved" value={String(stats.approvedRuns)} accent />
          <Stat label="written files" value={String(stats.artifactsWritten)} />
          <Stat label="latest status" value={String(stats.lastStatus)} />
        </div>

        <div style={launchGridStyle}>
          <div style={launchCardStyle}>
            <div style={launchToplineStyle}>
              <div style={launchBadgeStyle}>
                <Workflow size={12} />
                planner → worker → reviewer
              </div>
              <div style={launchHintStyle}>DeepSeek erforderlich</div>
            </div>

            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  void startRun();
                }
              }}
              rows={5}
              placeholder="Ziel eingeben — z. B. Baue eine CLI fuer zwei DeepSeek-Agenten mit klarer Review-Schleife."
              style={launchTextareaStyle}
            />

            <div style={launchControlRowStyle}>
              <label style={controlStyle}>
                <span className="eyebrow">revisionen</span>
                <input
                  type="number"
                  min={0}
                  max={3}
                  value={maxRevisions}
                  onChange={(e) => setMaxRevisions(clampInt(e.target.value, 0, 3, 1))}
                  style={numberInputStyle}
                />
              </label>

              <label style={toggleStyle}>
                <input
                  type="checkbox"
                  checked={writeArtifacts}
                  onChange={(e) => setWriteArtifacts(e.target.checked)}
                />
                <span>Artefakte direkt auf Platte schreiben</span>
              </label>

              <button
                type="button"
                onClick={() => void startRun()}
                disabled={busy || goal.trim().length === 0}
                style={launchButtonStyle(busy || goal.trim().length === 0)}
              >
                {busy ? <Loader2 size={14} className="spin" /> : <Play size={14} />}
                {busy ? 'running…' : 'start run'}
              </button>
            </div>

            <div style={launchFooterStyle}>
              <span className="eyebrow">shortcut</span>
              <span className="mono" style={{ color: 'var(--ink-muted)' }}>Ctrl/Cmd + Enter</span>
            </div>
          </div>

          <div style={railCardStyle}>
            <div style={railHeaderStyle}>
              <Sparkles size={13} style={{ color: 'var(--signal-bright)' }} />
              <span className="eyebrow" style={{ color: 'var(--signal-bright)' }}>what ships back</span>
            </div>
            <ul style={railListStyle}>
              <li style={railItemStyle}><Bot size={13} /> Plan mit Abnahmekriterien</li>
              <li style={railItemStyle}><FileCode2 size={13} /> konkrete Worker-Ausgabe plus Datei-Artefakte</li>
              <li style={railItemStyle}><ShieldCheck size={13} /> Reviewer-Urteil mit finaler Antwort</li>
              <li style={railItemStyle}><Clock3 size={13} /> Run-Historie fuer spaetere Vergleiche</li>
            </ul>
          </div>
        </div>
      </section>

      {error && (
        <div style={errorBannerStyle}>
          <strong>backend problem</strong> · {error}
          <div style={{ fontSize: 11, color: 'var(--ink-muted)', marginTop: 4 }}>
            Die Ansicht versucht es weiterhin automatisch alle {POLL_INTERVAL_MS / 1000}s.
          </div>
        </div>
      )}

      <section style={workspaceStyle}>
        <div style={feedStyle}>
          <div style={sectionHeaderStyle}>
            <span className="eyebrow">run history</span>
            <span style={{ color: 'var(--ink-faint)', fontSize: 11 }}>{runs.length} gespeicherte laeufe</span>
          </div>

          {runs.length === 0 ? (
            <Empty text="Noch keine Multi-Agent-Runs. Starte oben den ersten lokalen Lauf." />
          ) : (
            <div style={feedListStyle}>
              {runs.map((run) => (
                <button
                  key={run.run_id}
                  type="button"
                  onClick={() => setSelectedRunId(run.run_id)}
                  style={runCardStyle(selectedRunId === run.run_id)}
                >
                  <div style={runCardTopStyle}>
                    <span style={runIdStyle}>#{run.run_id}</span>
                    <span style={statusPillStyle(run.status)}>{run.status}</span>
                  </div>
                  <div style={runGoalStyle}>{run.goal}</div>
                  <div style={runMetaStyle}>
                    <span>{formatRelativeTime(run.started_at)}</span>
                    <span>{run.worker_rounds} worker</span>
                    <span>{run.reviewer_rounds} review</span>
                    <span>{run.artifacts_written} files</span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        <div style={inspectorStyle}>
          <div style={sectionHeaderStyle}>
            <span className="eyebrow">run inspector</span>
            {selectedRun && <span style={{ color: 'var(--ink-faint)', fontSize: 11 }}>#{selectedRun.run_id}</span>}
          </div>

          {loadingRun ? (
            <div style={loadingStyle}>
              <Loader2 size={14} className="spin" />
              loading run…
            </div>
          ) : selectedRun ? (
            <RunInspector run={selectedRun} />
          ) : (
            <Empty text="Waehle links einen Run aus, um Plan, Runden und Artefakte zu sehen." />
          )}
        </div>
      </section>

      {toast && (
        <div style={toastStyle(toast.kind)} role="status" aria-live="polite">
          {toast.kind === 'ok' ? <CheckCircle2 size={13} /> : <ShieldCheck size={13} />}
          {toast.text}
        </div>
      )}

      <style>{spinKeyframes}</style>
    </SecondaryFrame>
  );
}

/// Green only when a different vendor actually reviewed the work; a
/// self-review is shown as a warning, never as a quiet success.
const councilStyle = (crossVendor: boolean): React.CSSProperties => ({
  padding: '10px 12px',
  borderRadius: 6,
  border: `1px solid ${crossVendor ? 'var(--verified)' : 'var(--warn)'}`,
  background: crossVendor ? 'rgba(22,163,74,0.06)' : 'rgba(217,119,6,0.07)',
  color: crossVendor ? 'var(--verified)' : 'var(--warn)',
});

function RunInspector({ run }: { run: StoredMultiAgentRun }) {
  const latestWorker = run.worker_rounds[run.worker_rounds.length - 1];
  const latestReviewer = run.reviewer_rounds[run.reviewer_rounds.length - 1];
  const finalAnswer = run.final_answer ?? run.error ?? 'Run laeuft noch oder wartet auf den naechsten Schritt.';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={inspectorHeroStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <span style={runIdStyle}>#{run.run_id}</span>
          <span style={statusPillStyle(run.status)}>{run.status}</span>
          <span style={roundMetaChipStyle}>{run.phase}</span>
          <span className="eyebrow">{formatTimeRange(run.started_at, run.finished_at)}</span>
        </div>
        <h3 style={inspectorTitleStyle}>{run.goal}</h3>
        <p style={inspectorCopyStyle}>{finalAnswer}</p>
      </div>

      <div style={miniStatsStyle}>
        <MiniStat label="worker rounds" value={String(run.worker_rounds.length)} />
        <MiniStat label="review rounds" value={String(run.reviewer_rounds.length)} />
        <MiniStat label="detected files" value={String(countArtifacts(run))} />
        <MiniStat label="written files" value={String(countWrittenArtifacts(run))} />
      </div>

      {run.council && (
        <div style={councilStyle(run.council.cross_vendor)}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <ShieldCheck size={14} strokeWidth={1.7} />
            <strong style={{ fontSize: 12 }}>
              {run.council.cross_vendor ? 'unabhaengiges Review' : 'KEIN unabhaengiges Review'}
            </strong>
            <span style={roundMetaChipStyle}>worker: {run.council.worker_tier}</span>
            <span style={roundMetaChipStyle}>reviewer: {run.council.reviewer_tier}</span>
          </div>
          <p style={{ margin: '6px 0 0', fontSize: 11, lineHeight: 1.5 }}>{run.council.note}</p>
        </div>
      )}

      {run.plan ? (
        <InspectorSection title="plan">
          <div style={planGridStyle}>
            <InfoCard label="goal summary" value={run.plan.goal_summary} />
            <InfoCard label="deliverable" value={run.plan.deliverable} />
          </div>
          <div style={stackStyle}>
            <Label>acceptance criteria</Label>
            <ul style={criteriaListStyle}>
              {run.plan.acceptance_criteria.map((item, index) => (
                <li key={`${index}-${item}`} style={criteriaItemStyle}>{item}</li>
              ))}
            </ul>
          </div>
          <CodePanel title="worker instructions" body={run.plan.worker_instructions} />
        </InspectorSection>
      ) : (
        <InspectorSection title="plan">
          <Empty text="Planner hat noch keinen strukturierten Plan zurueckgegeben." />
        </InspectorSection>
      )}

      {run.planner ? (
        <InspectorSection title="planner output">
          <CodePanel title={`${run.planner.agent} · ${run.planner.tier}`} body={run.planner.output} />
        </InspectorSection>
      ) : null}

      <InspectorSection title="worker rounds">
        <div style={stackStyle}>
          {run.worker_rounds.length === 0 ? (
            <Empty text="Noch keine Worker-Ausgabe vorhanden." />
          ) : run.worker_rounds.map((round) => (
            <div key={round.iteration} style={roundBlockStyle}>
              <div style={roundHeaderStyle}>
                <span style={roundTitleStyle}>worker round {round.iteration}</span>
                <span style={roundMetaChipStyle}>{round.tier}</span>
              </div>
              <CodePanel title="output" body={round.output} />
              {round.artifacts.length > 0 && (
                <div style={artifactGridStyle}>
                  {round.artifact_writes.map((artifact, index) => (
                    <div key={`${artifact.path}-${index}`} style={artifactCardStyle}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <FileCode2 size={12} style={{ color: 'var(--signal-bright)' }} />
                        <span className="mono" style={{ fontSize: 11 }}>{artifact.path}</span>
                      </div>
                      <div style={{ marginTop: 8, fontSize: 11, color: artifact.written ? 'var(--verified-bright)' : 'var(--alert-bright)' }}>
                        {artifact.written ? 'written to workspace' : artifact.error ?? 'not written'}
                      </div>
                      {artifact.resolved_path && (
                        <div style={{ marginTop: 4, fontSize: 10, color: 'var(--ink-faint)' }}>
                          {artifact.resolved_path}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </InspectorSection>

      <InspectorSection title="reviewer rounds">
        <div style={stackStyle}>
          {run.reviewer_rounds.length === 0 ? (
            <Empty text="Reviewer hat noch keine Runde abgeschlossen." />
          ) : run.reviewer_rounds.map((round) => (
            <div key={round.iteration} style={roundBlockStyle}>
              <div style={roundHeaderStyle}>
                <span style={roundTitleStyle}>review round {round.iteration}</span>
                <span style={approvalPillStyle(round.approved)}>{round.approved ? 'approved' : 'revision'}</span>
              </div>
              <InfoCard label="feedback" value={round.feedback} />
              <CodePanel title="review output" body={round.output} />
            </div>
          ))}
        </div>
      </InspectorSection>

      {latestWorker && latestReviewer && (
        <section style={finalStripStyle}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <ShieldCheck size={14} style={{ color: latestReviewer.approved ? 'var(--verified-bright)' : 'var(--alert-bright)' }} />
            <span className="eyebrow">final handoff</span>
          </div>
          <div style={finalStripBodyStyle}>
            <div style={finalStripColumnStyle}>
              <Label>worker deliverable</Label>
              <pre style={finalPreStyle}>{truncate(latestWorker.output, 1600)}</pre>
            </div>
            <div style={finalStripColumnStyle}>
              <Label>reviewer answer</Label>
              <pre style={finalPreStyle}>{truncate(latestReviewer.final_answer, 1600)}</pre>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

function InspectorSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={sectionStyle}>
      <div style={sectionHeaderStyle}>
        <span className="eyebrow">{title}</span>
      </div>
      {children}
    </section>
  );
}

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div style={infoCardStyle}>
      <Label>{label}</Label>
      <div style={{ color: 'var(--ink-bright)', fontSize: 13, lineHeight: 1.55 }}>{value}</div>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div style={miniStatStyle}>
      <div className="eyebrow">{label}</div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 18, color: 'var(--ink-bright)', marginTop: 4 }}>{value}</div>
    </div>
  );
}

function CodePanel({ title, body }: { title: string; body: string }) {
  return (
    <div style={codePanelStyle}>
      <div style={codePanelHeaderStyle}>
        <span className="eyebrow">{title}</span>
      </div>
      <pre style={codePanelPreStyle}>{body}</pre>
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div className="eyebrow" style={{ marginBottom: 8 }}>
      {children}
    </div>
  );
}

function summaryFromStored(run: StoredMultiAgentRun): MultiAgentRunSummary {
  return {
    run_id: run.run_id,
    started_at: run.started_at,
    finished_at: run.finished_at ?? null,
    mode: run.mode,
    goal: run.goal,
    status: run.status,
    worker_rounds: run.worker_rounds.length,
    reviewer_rounds: run.reviewer_rounds.length,
    artifacts_detected: countArtifacts(run),
    artifacts_written: countWrittenArtifacts(run),
  };
}

function countArtifacts(run: StoredMultiAgentRun): number {
  return run.worker_rounds.reduce((sum, round) => sum + round.artifacts.length, 0);
}

function countWrittenArtifacts(run: StoredMultiAgentRun): number {
  return run.worker_rounds.reduce(
    (sum, round) => sum + round.artifact_writes.filter((artifact) => artifact.written).length,
    0,
  );
}

function formatRelativeTime(sec: number): string {
  const now = Math.floor(Date.now() / 1000);
  const delta = Math.max(0, now - sec);
  if (delta < 60) return `${delta}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
}

function formatTimeRange(start: number, end?: number | null): string {
  const startDate = new Date(start * 1000);
  const startLabel = startDate.toLocaleString(undefined, {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
  if (!end) {
    return `${startLabel} → live`;
  }
  const endDate = new Date(end * 1000);
  const endLabel = endDate.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  });
  return `${startLabel} → ${endLabel}`;
}

function truncate(value: string, max: number): string {
  if (value.length <= max) return value;
  return `${value.slice(0, max - 1)}…`;
}

function clampInt(raw: string, min: number, max: number, fallback: number): number {
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

function upsertSummary(prev: MultiAgentRunSummary[], next: MultiAgentRunSummary): MultiAgentRunSummary[] {
  const rest = prev.filter((run) => run.run_id !== next.run_id);
  return [next, ...rest].sort((a, b) => b.run_id - a.run_id);
}

function readRunIdFromUrl(): number | null {
  const value = new URLSearchParams(window.location.search).get('run');
  const parsed = Number.parseInt(value ?? '', 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function writeRunIdToUrl(runId: number | null) {
  const url = new URL(window.location.href);
  if (runId == null) {
    url.searchParams.delete('run');
  } else {
    url.searchParams.set('run', String(runId));
  }
  window.history.replaceState({}, '', url);
}

function statusPillStyle(status: string): React.CSSProperties {
  const approved = status === 'approved';
  return {
    display: 'inline-flex',
    alignItems: 'center',
    padding: '3px 8px',
    borderRadius: 999,
    fontSize: 10,
    letterSpacing: 0.06,
    textTransform: 'uppercase',
    fontWeight: 600,
    background: approved ? 'var(--verified-soft)' : 'var(--signal-soft)',
    color: approved ? 'var(--verified-bright)' : 'var(--signal-bright)',
    border: `1px solid ${approved ? 'var(--verified-soft)' : 'var(--signal-soft)'}`,
  };
}

function approvalPillStyle(approved: boolean): React.CSSProperties {
  return statusPillStyle(approved ? 'approved' : 'revision');
}

function launchButtonStyle(disabled: boolean): React.CSSProperties {
  return {
    marginLeft: 'auto',
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    padding: '10px 16px',
    background: disabled ? 'var(--bg-raised)' : 'var(--signal-bright)',
    color: disabled ? 'var(--ink-faint)' : 'white',
    borderRadius: 4,
    border: `1px solid ${disabled ? 'var(--rule-default)' : 'var(--signal-bright)'}`,
    opacity: disabled ? 0.6 : 1,
    cursor: disabled ? 'not-allowed' : 'pointer',
  };
}

function runCardStyle(active: boolean): React.CSSProperties {
  return {
    width: '100%',
    textAlign: 'left',
    padding: '14px 14px 12px',
    borderRadius: 6,
    border: `1px solid ${active ? 'var(--signal-bright)' : 'var(--rule-default)'}`,
    background: active ? 'linear-gradient(180deg, var(--signal-soft), rgba(255,255,255,0.4))' : 'var(--bg-panel)',
    transition: 'border-color 140ms var(--ease), background 140ms var(--ease)',
  };
}

function toastStyle(kind: 'ok' | 'err'): React.CSSProperties {
  return {
    position: 'fixed',
    right: 24,
    bottom: 24,
    zIndex: 80,
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    padding: '10px 14px',
    background: kind === 'ok' ? 'var(--verified-soft)' : 'var(--alert-soft)',
    color: kind === 'ok' ? 'var(--verified-bright)' : 'var(--alert-bright)',
    border: `1px solid ${kind === 'ok' ? 'var(--verified-soft)' : 'var(--alert-soft)'}`,
    borderRadius: 6,
    boxShadow: 'var(--shadow-pop)',
    fontSize: 12,
    fontWeight: 500,
  };
}

const spinKeyframes = `
@keyframes orbit-multi-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
.spin {
  animation: orbit-multi-spin 900ms linear infinite;
}
`;

const heroStyle: React.CSSProperties = {
  position: 'relative',
  overflow: 'hidden',
  padding: 22,
  borderRadius: 10,
  border: '1px solid var(--rule-default)',
  background: 'linear-gradient(180deg, rgba(37,99,235,0.08), rgba(37,99,235,0.02) 42%, var(--bg-panel) 100%)',
  boxShadow: 'var(--shadow-sm)',
};

const heroGlowStyle: React.CSSProperties = {
  position: 'absolute',
  top: -80,
  right: -70,
  width: 220,
  height: 220,
  borderRadius: '50%',
  background: 'radial-gradient(circle, rgba(37,99,235,0.14), rgba(37,99,235,0) 72%)',
  pointerEvents: 'none',
};

const heroHeaderStyle: React.CSSProperties = {
  position: 'relative',
  zIndex: 1,
  display: 'flex',
  justifyContent: 'space-between',
  gap: 16,
  alignItems: 'flex-start',
};

const heroTitleStyle: React.CSSProperties = {
  margin: '6px 0 8px',
  fontFamily: 'var(--font-display)',
  fontSize: 26,
  lineHeight: 1.04,
  letterSpacing: -0.03,
  color: 'var(--ink-bright)',
  maxWidth: 520,
};

const heroTextStyle: React.CSSProperties = {
  margin: 0,
  fontSize: 13,
  lineHeight: 1.6,
  color: 'var(--ink-muted)',
  maxWidth: 560,
};

const heroRefreshStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  padding: '8px 12px',
  borderRadius: 4,
  border: '1px solid var(--rule-default)',
  background: 'rgba(255,255,255,0.7)',
  color: 'var(--ink-bright)',
  flexShrink: 0,
};

const heroStatsStyle: React.CSSProperties = {
  position: 'relative',
  zIndex: 1,
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
  gap: 10,
  marginTop: 18,
};

const launchGridStyle: React.CSSProperties = {
  position: 'relative',
  zIndex: 1,
  display: 'grid',
  gridTemplateColumns: 'minmax(0, 1.7fr) minmax(240px, 0.9fr)',
  gap: 14,
  marginTop: 16,
};

const launchCardStyle: React.CSSProperties = {
  padding: 16,
  borderRadius: 8,
  border: '1px solid var(--rule-default)',
  background: 'rgba(255,255,255,0.82)',
  backdropFilter: 'blur(10px)',
};

const railCardStyle: React.CSSProperties = {
  padding: 16,
  borderRadius: 8,
  border: '1px solid var(--rule-default)',
  background: 'linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,255,255,0.72))',
};

const launchToplineStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  gap: 12,
  alignItems: 'center',
  flexWrap: 'wrap',
  marginBottom: 12,
};

const launchBadgeStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  padding: '6px 10px',
  borderRadius: 999,
  background: 'var(--signal-soft)',
  color: 'var(--signal-bright)',
  fontSize: 11,
  fontWeight: 600,
  letterSpacing: 0.04,
  textTransform: 'uppercase',
};

const launchHintStyle: React.CSSProperties = {
  fontSize: 11,
  color: 'var(--ink-faint)',
};

const launchTextareaStyle: React.CSSProperties = {
  width: '100%',
  minHeight: 120,
  padding: '14px 14px 12px',
  borderRadius: 6,
  border: '1px solid var(--rule-default)',
  background: 'var(--bg-panel)',
  color: 'var(--ink-bright)',
  lineHeight: 1.55,
  resize: 'vertical',
  boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.45)',
};

const launchControlRowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 14,
  flexWrap: 'wrap',
  marginTop: 12,
};

const controlStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 6,
};

const numberInputStyle: React.CSSProperties = {
  width: 72,
  padding: '8px 10px',
  borderRadius: 4,
  border: '1px solid var(--rule-default)',
  background: 'var(--bg-panel)',
  color: 'var(--ink-bright)',
};

const toggleStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 8,
  fontSize: 12,
  color: 'var(--ink-muted)',
};

const launchFooterStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  marginTop: 12,
};

const railHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  marginBottom: 10,
};

const railListStyle: React.CSSProperties = {
  listStyle: 'none',
  padding: 0,
  margin: 0,
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
};

const railItemStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  fontSize: 12,
  color: 'var(--ink-muted)',
};

const errorBannerStyle: React.CSSProperties = {
  marginTop: 16,
  padding: '12px 14px',
  borderRadius: 6,
  border: '1px solid var(--alert-soft)',
  background: 'var(--alert-soft)',
  color: 'var(--alert-bright)',
  fontSize: 12,
};

const workspaceStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'minmax(260px, 0.88fr) minmax(0, 1.42fr)',
  gap: 16,
  alignItems: 'start',
  marginTop: 20,
};

const feedStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
  minWidth: 0,
};

const inspectorStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
  minWidth: 0,
};

const sectionHeaderStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  gap: 12,
};

const feedListStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
};

const runCardTopStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  gap: 10,
};

const runIdStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  color: 'var(--ink-faint)',
};

const runGoalStyle: React.CSSProperties = {
  marginTop: 10,
  color: 'var(--ink-bright)',
  fontSize: 13,
  lineHeight: 1.45,
};

const runMetaStyle: React.CSSProperties = {
  display: 'flex',
  gap: 10,
  flexWrap: 'wrap',
  marginTop: 10,
  fontSize: 10,
  color: 'var(--ink-faint)',
  fontFamily: 'var(--font-mono)',
};

const loadingStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 8,
  padding: '14px 16px',
  borderRadius: 6,
  background: 'var(--bg-panel)',
  border: '1px solid var(--rule-default)',
  color: 'var(--ink-muted)',
  fontSize: 12,
};

const inspectorHeroStyle: React.CSSProperties = {
  padding: 18,
  borderRadius: 8,
  border: '1px solid var(--rule-default)',
  background: 'linear-gradient(180deg, rgba(37,99,235,0.06), rgba(255,255,255,0.9))',
};

const inspectorTitleStyle: React.CSSProperties = {
  margin: '12px 0 8px',
  fontFamily: 'var(--font-display)',
  fontSize: 22,
  lineHeight: 1.12,
  letterSpacing: -0.02,
  color: 'var(--ink-bright)',
};

const inspectorCopyStyle: React.CSSProperties = {
  margin: 0,
  fontSize: 13,
  lineHeight: 1.6,
  color: 'var(--ink-muted)',
  whiteSpace: 'pre-wrap',
};

const miniStatsStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
  gap: 10,
};

const miniStatStyle: React.CSSProperties = {
  padding: '12px 14px',
  borderRadius: 6,
  border: '1px solid var(--rule-default)',
  background: 'var(--bg-panel)',
};

const sectionStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
};

const planGridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
  gap: 10,
};

const infoCardStyle: React.CSSProperties = {
  padding: '12px 14px',
  borderRadius: 6,
  border: '1px solid var(--rule-default)',
  background: 'var(--bg-panel)',
};

const stackStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
};

const criteriaListStyle: React.CSSProperties = {
  listStyle: 'none',
  padding: 0,
  margin: 0,
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
};

const criteriaItemStyle: React.CSSProperties = {
  padding: '10px 12px',
  borderRadius: 6,
  border: '1px solid var(--rule-default)',
  background: 'var(--bg-panel)',
  color: 'var(--ink-muted)',
  fontSize: 12,
};

const codePanelStyle: React.CSSProperties = {
  borderRadius: 6,
  border: '1px solid var(--rule-default)',
  background: 'var(--bg-panel)',
  overflow: 'hidden',
};

const codePanelHeaderStyle: React.CSSProperties = {
  padding: '10px 12px',
  borderBottom: '1px solid var(--rule-default)',
  background: 'rgba(0,0,0,0.02)',
};

const codePanelPreStyle: React.CSSProperties = {
  margin: 0,
  padding: '14px 14px 16px',
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  lineHeight: 1.65,
  color: 'var(--ink-muted)',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  maxHeight: 360,
  overflow: 'auto',
};

const roundBlockStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
  padding: 14,
  borderRadius: 8,
  border: '1px solid var(--rule-default)',
  background: 'var(--bg-panel)',
};

const roundHeaderStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  gap: 10,
  flexWrap: 'wrap',
};

const roundTitleStyle: React.CSSProperties = {
  color: 'var(--ink-bright)',
  fontSize: 13,
  fontWeight: 600,
};

const roundMetaChipStyle: React.CSSProperties = {
  padding: '3px 8px',
  borderRadius: 999,
  background: 'var(--bg-raised)',
  border: '1px solid var(--rule-default)',
  fontSize: 10,
  fontFamily: 'var(--font-mono)',
  color: 'var(--ink-faint)',
};

const artifactGridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
  gap: 10,
};

const artifactCardStyle: React.CSSProperties = {
  padding: '12px 12px 10px',
  borderRadius: 6,
  border: '1px solid var(--rule-default)',
  background: 'linear-gradient(180deg, rgba(37,99,235,0.05), rgba(255,255,255,0.8))',
};

const finalStripStyle: React.CSSProperties = {
  padding: 16,
  borderRadius: 8,
  border: '1px solid var(--rule-default)',
  background: 'linear-gradient(180deg, rgba(22,163,74,0.06), rgba(255,255,255,0.92))',
};

const finalStripBodyStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
  gap: 12,
  marginTop: 12,
};

const finalStripColumnStyle: React.CSSProperties = {
  minWidth: 0,
};

const finalPreStyle: React.CSSProperties = {
  margin: 0,
  padding: '12px 12px 14px',
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  lineHeight: 1.6,
  color: 'var(--ink-muted)',
  whiteSpace: 'pre-wrap',
  borderRadius: 6,
  background: 'rgba(255,255,255,0.7)',
  border: '1px solid var(--rule-default)',
  maxHeight: 220,
  overflow: 'auto',
};
