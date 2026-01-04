const API_BASE = 'http://localhost:8000';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export const api = {
  // Projects
  listProjects: () => fetchJson<any[]>('/projects'),

  createProject: (name: string, description?: string) =>
    fetchJson<any>('/projects', {
      method: 'POST',
      body: JSON.stringify({ name, description }),
    }),

  getProject: (id: string) => fetchJson<any>(`/projects/${id}`),

  deleteProject: (id: string) =>
    fetch(`${API_BASE}/projects/${id}`, { method: 'DELETE' }),

  // Audio
  importAudio: async (projectId: string, file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/projects/${projectId}/import`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(error.detail);
    }

    return response.json();
  },

  getAudioUrl: (projectId: string, audioType: string) =>
    `${API_BASE}/projects/${projectId}/audio/${audioType}`,

  // Processing
  startIsolation: (projectId: string, options?: {
    trim_start?: number;
    trim_end?: number;
    pan?: number;
    prompt?: string;
  }) =>
    fetchJson<{ task_id: string }>(`/projects/${projectId}/isolate`, {
      method: 'POST',
      body: JSON.stringify(options || {}),
    }),

  startMatching: (projectId: string) =>
    fetchJson<{ task_id: string }>(`/projects/${projectId}/match`, {
      method: 'POST',
    }),

  getTaskStatus: (projectId: string, taskId: string) =>
    fetchJson<any>(`/projects/${projectId}/tasks/${taskId}`),

  // Results
  getResults: (projectId: string) =>
    fetchJson<any>(`/projects/${projectId}/results`),

  exportJson: (projectId: string) =>
    fetchJson<any>(`/projects/${projectId}/export/json`),

  getMarkdownUrl: (projectId: string) =>
    `${API_BASE}/projects/${projectId}/export/markdown`,
};


