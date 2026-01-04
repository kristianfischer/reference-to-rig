import { create } from 'zustand';
import { api } from '../api/client';

export interface Project {
  id: string;
  name: string;
  description?: string;
  status: 'created' | 'imported' | 'isolating' | 'isolated' | 'matching' | 'matched' | 'error';
  created_at: string;
  updated_at: string;
  source_audio_path?: string;
  isolated_audio_path?: string;
  isolation_confidence?: number;
}

export interface EQBand {
  frequency: number;
  gain_db: number;
  q: number;
  band_type: string;
}

export interface EQSettings {
  bands: EQBand[];
  highpass_freq?: number;
  lowpass_freq?: number;
}

export interface MatchCandidate {
  flavor: string;
  nam_model_id: string;
  nam_model_name: string;
  ir_id: string;
  ir_name: string;
  input_gain_db: number;
  eq_settings: EQSettings;
  similarity_score: number;
  rendered_audio_path?: string;
}

export interface MatchResults {
  project_id: string;
  reference_audio_path: string;
  isolated_audio_path: string;
  isolation_confidence: number;
  candidates: MatchCandidate[];
  created_at: string;
}

export interface TaskStatus {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message?: string;
  error?: string;
}

interface AppState {
  // Projects
  projects: Project[];
  currentProject: Project | null;
  loadingProjects: boolean;

  // Tasks
  currentTask: TaskStatus | null;

  // Results
  matchResults: MatchResults | null;

  // Audio playback
  playingAudio: 'reference' | 'isolated' | 'match_balanced' | 'match_brighter' | 'match_thicker' | null;

  // Actions
  fetchProjects: () => Promise<void>;
  createProject: (name: string, description?: string) => Promise<Project>;
  fetchProject: (id: string) => Promise<void>;
  deleteProject: (id: string) => Promise<void>;
  importAudio: (projectId: string, file: File) => Promise<void>;
  startIsolation: (projectId: string, options?: {
    trim_start?: number;
    trim_end?: number;
    pan?: number;
    prompt?: string;
  }) => Promise<string>;
  startMatching: (projectId: string) => Promise<string>;
  pollTaskStatus: (projectId: string, taskId: string) => Promise<TaskStatus>;
  fetchResults: (projectId: string) => Promise<void>;
  setPlayingAudio: (audio: AppState['playingAudio']) => void;
  clearCurrentTask: () => void;
}

export const useStore = create<AppState>((set, get) => ({
  projects: [],
  currentProject: null,
  loadingProjects: false,
  currentTask: null,
  matchResults: null,
  playingAudio: null,

  fetchProjects: async () => {
    set({ loadingProjects: true });
    try {
      const projects = await api.listProjects();
      set({ projects, loadingProjects: false });
    } catch (error) {
      console.error('Failed to fetch projects:', error);
      set({ loadingProjects: false });
    }
  },

  createProject: async (name: string, description?: string) => {
    const project = await api.createProject(name, description);
    set(state => ({ projects: [project, ...state.projects] }));
    return project;
  },

  fetchProject: async (id: string) => {
    const project = await api.getProject(id);
    set({ currentProject: project });
  },

  deleteProject: async (id: string) => {
    await api.deleteProject(id);
    set(state => ({
      projects: state.projects.filter(p => p.id !== id),
      currentProject: state.currentProject?.id === id ? null : state.currentProject,
    }));
  },

  importAudio: async (projectId: string, file: File) => {
    await api.importAudio(projectId, file);
    await get().fetchProject(projectId);
  },

  startIsolation: async (projectId: string, options?: {
    trim_start?: number;
    trim_end?: number;
    pan?: number;
    prompt?: string;
  }) => {
    const response = await api.startIsolation(projectId, options);
    set({ currentTask: { task_id: response.task_id, status: 'pending', progress: 0 } });
    return response.task_id;
  },

  startMatching: async (projectId: string) => {
    const response = await api.startMatching(projectId);
    set({ currentTask: { task_id: response.task_id, status: 'pending', progress: 0 } });
    return response.task_id;
  },

  pollTaskStatus: async (projectId: string, taskId: string) => {
    const status = await api.getTaskStatus(projectId, taskId);
    set({ currentTask: status });
    return status;
  },

  fetchResults: async (projectId: string) => {
    const results = await api.getResults(projectId);
    set({ matchResults: results });
  },

  setPlayingAudio: (audio) => {
    set({ playingAudio: audio });
  },

  clearCurrentTask: () => {
    set({ currentTask: null });
  },
}));


