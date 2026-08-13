const HEALTH_COLORS = {
  Excellent: '#4fd1c5',
  Good: '#4fd1c5',
  Average: '#e8b94d',
  Poor: '#e8607a',
};

const BADGE_CLASSES = {
  Excellent: 'bg-accentCyan/15 text-accentCyan border-accentCyan/40',
  Good: 'bg-accentCyan/10 text-accentCyan border-accentCyan/25',
  Average: 'bg-accentAmber/15 text-accentAmber border-accentAmber/40',
  Poor: 'bg-accentRose/15 text-accentRose border-accentRose/40',
};

export function HealthBadge({ category }) {
  return (
    <span className={`badge border ${BADGE_CLASSES[category] || BADGE_CLASSES.Average}`}>
      {category}
    </span>
  );
}

export default function HealthGauge({ score, category }) {
  const color = HEALTH_COLORS[category] || HEALTH_COLORS.Average;
  const r = 55, cx = 70, cy = 70;
  const circumference = 2 * Math.PI * r;
  const arcFraction = 0.75;
  const arcLength = circumference * arcFraction;
  const filled = arcLength * (score / 100);
  const rotation = 135;

  return (
    <svg width="140" height="140" viewBox="0 0 140 140">
      <circle
        cx={cx} cy={cy} r={r} fill="none" stroke="#2a323d" strokeWidth="10"
        strokeDasharray={`${arcLength} ${circumference}`} strokeLinecap="round"
        transform={`rotate(${rotation} ${cx} ${cy})`}
      />
      <circle
        cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth="10"
        strokeDasharray={`${filled} ${circumference}`} strokeLinecap="round"
        transform={`rotate(${rotation} ${cx} ${cy})`}
        style={{ transition: 'stroke-dasharray 0.6s ease' }}
      />
      <text x={cx} y={cy - 2} textAnchor="middle" fontFamily="IBM Plex Mono, monospace"
        fontSize="26" fontWeight="600" fill="#e8ecf1">{score}</text>
      <text x={cx} y={cy + 18} textAnchor="middle" fontFamily="IBM Plex Mono, monospace"
        fontSize="10" letterSpacing="1" fill="#8b96a5">/ 100</text>
    </svg>
  );
}
