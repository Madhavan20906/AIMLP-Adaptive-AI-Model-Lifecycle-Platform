export default function VersionHistory({ versions, onRefresh, onRollback }) {
  return (
    <div className="panel p-5 bg-panel border border-borderSubtle rounded-lg">
      <div className="flex items-center justify-between mb-3">
        <div className="eyebrow">version history</div>
        <button
          className="action px-3 py-1 text-xs border border-borderSubtle text-textMuted rounded-md hover:brightness-125"
          onClick={onRefresh}
        >
          Refresh
        </button>
      </div>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr>
            {['Version', 'Algorithm', 'Score', 'Source', 'Status', ''].map(h => (
              <th key={h} className="text-left eyebrow pb-2 border-b border-borderSubtle">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {(!versions || versions.length === 0) && (
            <tr><td colSpan={6} className="text-sm text-textMuted py-3">No versions yet for this project.</td></tr>
          )}
          {versions?.map(v => (
            <tr key={v.version_id} className="hover:bg-white/[0.02]">
              <td className="mono py-2 border-b border-borderSubtle/50">v{v.version_id}</td>
              <td className="py-2 border-b border-borderSubtle/50">{v.algorithm}</td>
              <td className="mono py-2 border-b border-borderSubtle/50 text-accentCyan">{v.overall_score}</td>
              <td className="mono text-xs py-2 border-b border-borderSubtle/50 text-textMuted">{v.source}</td>
              <td className="py-2 border-b border-borderSubtle/50">
                {v.is_active
                  ? <span className="badge border bg-accentCyan/15 text-accentCyan border-accentCyan/40">active</span>
                  : <span className="badge border bg-accentAmber/15 text-accentAmber border-accentAmber/40 opacity-50">inactive</span>}
              </td>
              <td className="py-2 border-b border-borderSubtle/50">
                {!v.is_active && (
                  <button
                    className="action px-2 py-1 text-xs border border-borderSubtle text-textMuted rounded-md hover:brightness-125"
                    onClick={() => onRollback(v.version_id)}
                  >
                    Rollback
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
