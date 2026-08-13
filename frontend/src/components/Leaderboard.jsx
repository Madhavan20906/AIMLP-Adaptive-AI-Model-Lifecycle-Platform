export default function Leaderboard({ rows }) {
  if (!rows || !rows.length) return null;
  return (
    <div className="panel p-5 fade-in bg-panel border border-borderSubtle rounded-lg">
      <div className="eyebrow mb-3">leaderboard</div>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr>
            {['Rank', 'Algorithm', 'Score', 'Accuracy', 'F1', 'ROC AUC', 'Train time', 'Size'].map(h => (
              <th key={h} className="text-left eyebrow pb-2 border-b border-borderSubtle">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.name} className="hover:bg-white/[0.02]">
              <td className="mono py-2 border-b border-borderSubtle/50">#{r.rank ?? i + 1}</td>
              <td className="py-2 border-b border-borderSubtle/50">{r.name}</td>
              <td className="mono py-2 border-b border-borderSubtle/50 text-accentCyan">{r.overall_score}</td>
              <td className="mono py-2 border-b border-borderSubtle/50">
                {r.metrics?.accuracy !== undefined ? r.metrics.accuracy.toFixed(3) : '—'}
              </td>
              <td className="mono py-2 border-b border-borderSubtle/50">
                {r.metrics?.f1 !== undefined ? r.metrics.f1.toFixed(3) : '—'}
              </td>
              <td className="mono py-2 border-b border-borderSubtle/50">
                {r.metrics?.roc_auc ? r.metrics.roc_auc.toFixed(3) : '—'}
              </td>
              <td className="mono py-2 border-b border-borderSubtle/50">{r.train_time_s ?? '—'}s</td>
              <td className="mono py-2 border-b border-borderSubtle/50">{r.model_size_kb ?? '—'}KB</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
