import { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Loader, CheckCircle, XCircle } from 'lucide-react';
import { useStore } from '../store/useStore';

export default function MatchingPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const {
    currentProject,
    currentTask,
    fetchProject,
    startMatching,
    pollTaskStatus,
    clearCurrentTask,
  } = useStore();

  useEffect(() => {
    if (id) {
      fetchProject(id);
    }
  }, [id]);

  // Start matching when page loads
  useEffect(() => {
    if (id && currentProject && !currentTask) {
      if (currentProject.status === 'isolated') {
        startMatching(id);
      } else if (currentProject.status === 'matched') {
        navigate(`/project/${id}/results`);
      }
    }
  }, [id, currentProject?.status, currentTask]);

  // Poll task status
  useEffect(() => {
    if (!currentTask || !id) return;

    if (currentTask.status === 'completed') {
      fetchProject(id).then(() => {
        clearCurrentTask();
        navigate(`/project/${id}/results`);
      });
      return;
    }

    if (currentTask.status === 'failed') {
      return;
    }

    const interval = setInterval(async () => {
      await pollTaskStatus(id, currentTask.task_id);
    }, 1000);

    return () => clearInterval(interval);
  }, [currentTask?.task_id, currentTask?.status, id]);

  if (!currentProject) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader className="w-8 h-8 text-amp-amber animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="amp-panel rounded-xl p-8 text-center"
      >
        {currentTask?.status === 'failed' ? (
          <>
            <XCircle className="w-16 h-16 text-amp-red mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-amp-cream mb-2">
              Matching Failed
            </h2>
            <p className="text-amp-silver mb-6">
              {currentTask.error || 'An error occurred during matching'}
            </p>
            <div className="flex justify-center gap-4">
              <button
                onClick={() => {
                  clearCurrentTask();
                  if (id) startMatching(id);
                }}
                className="btn-primary"
              >
                Try Again
              </button>
              <button
                onClick={() => navigate(`/project/${id}`)}
                className="btn-secondary"
              >
                Back to Project
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="relative w-24 h-24 mx-auto mb-6">
              {/* Animated rings */}
              <motion.div
                className="absolute inset-0 rounded-full border-2 border-amp-amber/30"
                animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <motion.div
                className="absolute inset-0 rounded-full border-2 border-amp-orange/30"
                animate={{ scale: [1, 1.8, 1], opacity: [0.5, 0, 0.5] }}
                transition={{ duration: 2, repeat: Infinity, delay: 0.3 }}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <Loader className="w-12 h-12 text-amp-amber animate-spin" />
              </div>
            </div>

            <h2 className="text-2xl font-bold text-amp-cream mb-2">
              Matching Your Tone
            </h2>
            <p className="text-amp-silver mb-6">
              Analyzing spectral characteristics and searching the capture library...
            </p>

            {/* Progress steps */}
            <div className="space-y-3 text-left max-w-sm mx-auto">
              <ProgressStep
                label="Extracting features"
                done={currentTask && currentTask.progress >= 0.2}
                active={currentTask && currentTask.progress < 0.2}
              />
              <ProgressStep
                label="Searching capture library"
                done={currentTask && currentTask.progress >= 0.4}
                active={currentTask && currentTask.progress >= 0.2 && currentTask.progress < 0.4}
              />
              <ProgressStep
                label="Optimizing balanced match"
                done={currentTask && currentTask.progress >= 0.55}
                active={currentTask && currentTask.progress >= 0.4 && currentTask.progress < 0.55}
              />
              <ProgressStep
                label="Optimizing brighter match"
                done={currentTask && currentTask.progress >= 0.7}
                active={currentTask && currentTask.progress >= 0.55 && currentTask.progress < 0.7}
              />
              <ProgressStep
                label="Optimizing thicker match"
                done={currentTask && currentTask.progress >= 0.85}
                active={currentTask && currentTask.progress >= 0.7 && currentTask.progress < 0.85}
              />
              <ProgressStep
                label="Rendering previews"
                done={currentTask && currentTask.progress >= 1.0}
                active={currentTask && currentTask.progress >= 0.85 && currentTask.progress < 1.0}
              />
            </div>

            {/* Progress bar */}
            <div className="mt-8">
              <div className="h-2 bg-amp-black rounded-full overflow-hidden">
                <motion.div
                  className="h-full progress-bar rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${(currentTask?.progress || 0) * 100}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
              <p className="text-sm text-amp-amber font-mono mt-2">
                {((currentTask?.progress || 0) * 100).toFixed(0)}%
              </p>
            </div>
          </>
        )}
      </motion.div>
    </div>
  );
}

function ProgressStep({
  label,
  done,
  active,
}: {
  label: string;
  done: boolean;
  active: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <div
        className={`
          w-5 h-5 rounded-full flex items-center justify-center text-xs
          ${done ? 'bg-amp-green text-amp-black' : active ? 'bg-amp-amber text-amp-black' : 'bg-amp-steel text-amp-chrome'}
        `}
      >
        {done ? <CheckCircle className="w-3 h-3" /> : active ? <Loader className="w-3 h-3 animate-spin" /> : ''}
      </div>
      <span className={done ? 'text-amp-cream' : active ? 'text-amp-amber' : 'text-amp-chrome'}>
        {label}
      </span>
    </div>
  );
}


