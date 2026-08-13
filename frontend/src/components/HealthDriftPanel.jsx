import HealthGauge, { HealthBadge } from './HealthGauge';

const DRIFT_BADGE = {
  Low: 'bg-accentCyan/15 text-accentCyan border-accentCyan/40',
  Medium: 'bg-accentAmber/15 text-accentAmber border-accentAmber/40',
  High: 'bg-accentRose/15 text-accentRose border-accentRose/40',
};

const BREAKDOWN_LABELS = {
  performance: 'Performance',
  drift: 'Drift resistance',
  confidence: 'Confidence',
  efficiency: 'Efficiency',
};
const BREAKDOWN_COLORS = {
  performance: '#4fd1c5',
  drift: '#9b8cfb',
  confidence: '#e8b94d',
  efficiency: '#8b96a5',
};

const DECISION_LABELS = {
  retrain_same_algorithm: 'Retrain same algorithm',
  replace_model: 'Replace model family',
};

export default function HealthDriftPanel({ explanation }) {
  if (!explanation) return null;
  const { health, drift, decision, reasoning } = explanation;

  return (
    <div className="grid grid-cols-2 gap-6 fade-in">
      <div className="panel p-5 bg-panel border border-borderSubtle rounded-lg">
        <div className="eyebrow mb-4">model health</div>
        <div className="flex items-center gap-6">
          <HealthGauge score={health.health_score} category={health.category} />
          <div className="flex-1">
            <div className="mb-2"><HealthBadge category={health.category} /></div>
            <div className="space-y-2">
              {Object.entries(health.breakdown).map(([key, val]) => (
                <div key={key}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-textMuted">
                      {BREAKDOWN_LABELS[key] || key}{' '}
                      <span className="mono">({Math.round((health.weights[key] || 0) * 100)}%)</span>
                    </span>
                    <span className="mono">{val}</span>
                  </div>
                  <div className="bg-borderSubtle rounded h-1.5 overflow-hidden">
                    <div
                      className="h-full rounded"
                      style={{ width: `${val}%`, background: BREAKDOWN_COLORS[key] }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="panel p-5 bg-panel border border-borderSubtle rounded-lg">
        <div className="eyebrow mb-3">drift detection</div>
        <div className="flex items-center gap-3 mb-4">
          <span className={`badge border ${DRIFT_BADGE[drift.severity] || DRIFT_BADGE.Medium}`}>
            {drift.severity} drift
          </span>
          <span className="mono text-sm text-textMuted">
            {Math.round(drift.fraction_features_drifted * 100)}% of features shifted
          </span>
        </div>
        <div className="eyebrow mb-2">decision</div>
        <div
          className="text-sm font-medium mb-2"
          style={{ color: decision === 'replace_model' ? '#9b8cfb' : '#4fd1c5' }}
        >
          {DECISION_LABELS[decision] || decision}
        </div>
        <div className="text-sm text-textMuted space-y-1">
          {reasoning.map((r, i) => <p key={i}>{r}</p>)}
        </div>
      </div>
    </div>
  );
}
