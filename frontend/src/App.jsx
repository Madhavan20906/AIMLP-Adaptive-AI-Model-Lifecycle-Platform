import { useState, useEffect, useCallback } from 'react';
import { trainMode1, evolveMode2, getVersions, rollbackVersion } from './api';
import ProfilePanel from './components/ProfilePanel';
import Leaderboard from './components/Leaderboard';
import HealthDriftPanel from './components/HealthDriftPanel';
import VersionHistory from './components/VersionHistory';

export default function App() {
  const [projectName, setProjectName] = useState('fraud_prod');
  const [mode1File, setMode1File] = useState(null);
  const [mode1Target, setMode1Target] = useState('');
  const [mode2File, setMode2File] = useState(null);
  const [switchThreshold, setSwitchThreshold] = useState(3.0);

  const [status, setStatus] = useState('');
  const [statusError, setStatusError] = useState(false);
  const [training, setTraining] = useState(false);
  const [evolving, setEvolving] = useState(false);

  const [mode1Result, setMode1Result] = useState(null);
  const [mode2Result, setMode2Result] = useState(null);
  const [versions, setVersions] = useState([]);

  const refreshVersions = useCallback(async () => {
    if (!projectName) return;
    try {
      const data = await getVersions(projectName);
      setVersions(data.versions || []);
    } catch (e) {
      console.error(e);
    }
  }, [projectName]);

  useEffect(() => { refreshVersions(); }, [refreshVersions]);

  async function handleMode1() {
    if (!projectName || !mode1File) {
      setStatus('Provide a project name and a CSV file.');
      setStatusError(true);
      return;
    }
    setTraining(true);
    setStatusError(false);
    setStatus('Analyzing dataset, selecting candidates, training leaderboard — this can take a minute…');
    try {
      const data = await trainMode1(mode1File, projectName, mode1Target || undefined);
      if (data.error) {
        setStatus(data.error);
        setStatusError(true);
        return;
      }
      setMode1Result(data);
      setStatus(`Done in ${data.total_time_s}s — best model: ${data.best_model.name} (score ${data.best_model.overall_score}). Registered as version ${data.version_id}.`);
      refreshVersions();
    } catch (e) {
      setStatus('Request failed: ' + e.message);
      setStatusError(true);
    } finally {
      setTraining(false);
    }
  }

  async function handleMode2() {
    if (!projectName || !mode2File) {
      setStatus('Provide a project name and a CSV file.');
      setStatusError(true);
      return;
    }
    setEvolving(true);
    setStatusError(false);
    setStatus('Evaluating current model health and drift against latest data…');
    try {
      const data = await evolveMode2(projectName, mode2File, switchThreshold);
      if (data.error) {
        setStatus(data.error);
        setStatusError(true);
        return;
      }
      setMode2Result(data);
      setStatus(`Decision: ${data.decision}. ${data.new_version_id ? 'Registered as version ' + data.new_version_id + '.' : ''}`);
      refreshVersions();
    } catch (e) {
      setStatus('Request failed: ' + e.message);
      setStatusError(true);
    } finally {
      setEvolving(false);
    }
  }

  async function handleRollback(versionId) {
    setStatus(`Rolling back ${projectName} to version ${versionId}…`);
    try {
      const data = await rollbackVersion(projectName, versionId);
      setStatus(data.message || 'Rolled back.');
      refreshVersions();
    } catch (e) {
      setStatus('Rollback failed: ' + e.message);
      setStatusError(true);
    }
  }

  return (
    <div className="min-h-screen">
      <div className="border-b border-borderSubtle">
        <div className="max-w-6xl mx-auto px-6 py-5 flex items-center justify-between">
          <div>
            <div className="eyebrow">adaptive ai model lifecycle platform</div>
            <div className="text-xl font-semibold" style={{ letterSpacing: '-0.01em' }}>
              AIMLP <span className="mono text-sm text-textMuted">/ console</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="eyebrow">project</span>
            <input
              type="text" value={projectName} onChange={e => setProjectName(e.target.value)}
              onBlur={refreshVersions} className="w-48" placeholder="project name"
            />
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        <div className="grid grid-cols-2 gap-6">
          <div className="panel p-5 bg-panel border border-borderSubtle rounded-lg">
            <div className="eyebrow mb-1">mode 1</div>
            <div className="font-semibold mb-3">Initial Model Creation</div>
            <p className="text-sm mb-4 text-textMuted">
              Upload a dataset. The platform profiles it, shortlists candidate algorithms,
              trains and evaluates all of them, and registers the best one.
            </p>
            <input type="file" accept=".csv" className="w-full mb-3"
              onChange={e => setMode1File(e.target.files[0])} />
            <input type="text" placeholder="target column (optional — auto-detected)"
              className="w-full mb-3" value={mode1Target} onChange={e => setMode1Target(e.target.value)} />
            <button
              disabled={training}
              onClick={handleMode1}
              className="action w-full py-2.5 rounded-md mono text-xs font-semibold bg-accentCyan text-[#0a1414] disabled:opacity-40"
            >
              {training ? <span className="spinner mr-2" /> : null}
              {training ? 'Training…' : 'Train from dataset'}
            </button>
          </div>

          <div className="panel p-5 bg-panel border border-borderSubtle rounded-lg">
            <div className="eyebrow mb-1">mode 2</div>
            <div className="font-semibold mb-3">Adaptive Production Evolution</div>
            <p className="text-sm mb-4 text-textMuted">
              Upload the latest data. The platform checks the live model's health and drift,
              then retrains or replaces it — whichever is justified.
            </p>
            <input type="file" accept=".csv" className="w-full mb-3"
              onChange={e => setMode2File(e.target.files[0])} />
            <div className="flex items-center gap-2 mb-3">
              <span className="eyebrow whitespace-nowrap">switch threshold</span>
              <input type="number" step="0.5" value={switchThreshold} className="w-20"
                onChange={e => setSwitchThreshold(e.target.value)} />
            </div>
            <button
              disabled={evolving}
              onClick={handleMode2}
              className="action w-full py-2.5 rounded-md mono text-xs font-semibold bg-accentViolet text-[#14102f] disabled:opacity-40"
            >
              {evolving ? <span className="spinner mr-2" /> : null}
              {evolving ? 'Evaluating…' : 'Evaluate & evolve'}
            </button>
          </div>
        </div>

        {status && (
          <div className={`text-sm mono ${statusError ? 'text-accentRose' : 'text-textMuted'}`}>{status}</div>
        )}

        {mode1Result && (
          <ProfilePanel profile={mode1Result.profile} candidateSelection={{
            candidates: mode1Result.candidate_selection.candidates_evaluated,
            reasons: mode1Result.candidate_selection.reasons,
            confidence: mode1Result.candidate_selection.confidence,
            evaluated_all: mode1Result.candidate_selection.evaluated_all,
          }} />
        )}

        {mode2Result && <HealthDriftPanel explanation={mode2Result} />}

        {mode1Result && <Leaderboard rows={mode1Result.leaderboard} />}
        {mode2Result?.leaderboard?.length > 0 && (
          <Leaderboard rows={mode2Result.leaderboard.map((r, i) => ({ ...r, rank: i + 1 }))} />
        )}

        <VersionHistory versions={versions} onRefresh={refreshVersions} onRollback={handleRollback} />
      </div>
    </div>
  );
}
