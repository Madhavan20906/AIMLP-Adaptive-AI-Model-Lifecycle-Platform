export default function ProfilePanel({ profile, candidateSelection }) {
  if (!profile) return null;
  const cell = (label, val) => (
    <div key={label}>
      <div className="eyebrow mb-1">{label}</div>
      <div className="mono">{val}</div>
    </div>
  );
  return (
    <div className="panel p-5 fade-in bg-panel border border-borderSubtle rounded-lg">
      <div className="eyebrow mb-3">dataset analysis</div>
      <div className="grid grid-cols-4 gap-4 text-sm">
        {cell('rows', profile.n_rows.toLocaleString())}
        {cell('columns', profile.n_columns)}
        {cell('problem type', profile.problem_type)}
        {cell('target', profile.target_column)}
        {cell('quality score', `${profile.data_quality_score} / 100`)}
        {cell('missing', `${(profile.missing_ratio * 100).toFixed(1)}%`)}
        {cell('duplicates', profile.duplicates)}
        {cell('imbalance ratio', profile.imbalance_ratio ?? '—')}
      </div>
      {candidateSelection && (
        <div className="mt-4 pt-4 border-t border-borderSubtle text-sm">
          <div className="eyebrow mb-2">
            candidate selection {candidateSelection.evaluated_all
              ? '(evaluated full pool — low confidence in shortlisting)'
              : `(confidence ${Math.round(candidateSelection.confidence * 100)}%)`}
          </div>
          <div className="mb-2 mono text-xs text-accentCyan">
            {candidateSelection.candidates.join(' · ')}
          </div>
          <ul className="space-y-1 text-xs text-textMuted">
            {candidateSelection.reasons.map((r, i) => <li key={i}>— {r}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
