const API_BASE = import.meta.env.VITE_API_BASE || '';

export async function trainMode1(file, projectName, targetColumn) {
  const form = new FormData();
  form.append('file', file);
  form.append('project_name', projectName);
  if (targetColumn) form.append('target_column', targetColumn);
  const res = await fetch(`${API_BASE}/mode1/train`, { method: 'POST', body: form });
  return res.json();
}

export async function evolveMode2(projectName, file, switchThreshold) {
  const form = new FormData();
  form.append('latest_file', file);
  form.append('project_name', projectName);
  form.append('switch_threshold', switchThreshold);
  const res = await fetch(`${API_BASE}/mode2/evolve`, { method: 'POST', body: form });
  return res.json();
}

export async function getVersions(projectName) {
  const res = await fetch(`${API_BASE}/registry/${projectName}/versions`);
  return res.json();
}

export async function rollbackVersion(projectName, versionId) {
  const res = await fetch(`${API_BASE}/registry/${projectName}/rollback/${versionId}`, { method: 'POST' });
  return res.json();
}
