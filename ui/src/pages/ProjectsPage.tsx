import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Plus, Folder, Trash2, Clock, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import { useStore, Project } from '../store/useStore';

export default function ProjectsPage() {
  const navigate = useNavigate();
  const { projects, loadingProjects, fetchProjects, createProject, deleteProject } = useStore();
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDesc, setNewProjectDesc] = useState('');
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    fetchProjects();
  }, []);

  const handleCreate = async () => {
    if (!newProjectName.trim()) return;
    setCreating(true);
    try {
      const project = await createProject(newProjectName, newProjectDesc || undefined);
      setShowCreateModal(false);
      setNewProjectName('');
      setNewProjectDesc('');
      navigate(`/project/${project.id}`);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (confirm('Delete this project? This cannot be undone.')) {
      await deleteProject(id);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-2xl font-bold text-amp-cream">Projects</h2>
          <p className="text-amp-silver mt-1">
            Create a project to start matching guitar tones
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="w-5 h-5" />
          New Project
        </button>
      </div>

      {loadingProjects ? (
        <div className="flex items-center justify-center py-12">
          <Loader className="w-8 h-8 text-amp-amber animate-spin" />
        </div>
      ) : projects.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="amp-panel rounded-xl p-12 text-center"
        >
          <Folder className="w-16 h-16 text-amp-chrome mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-amp-cream mb-2">No Projects Yet</h3>
          <p className="text-amp-silver mb-6 max-w-md mx-auto">
            Create your first project to import a reference track and find matching guitar tones.
          </p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn-primary"
          >
            Create Your First Project
          </button>
        </motion.div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {projects.map((project, index) => (
            <ProjectCard
              key={project.id}
              project={project}
              index={index}
              onClick={() => navigate(`/project/${project.id}`)}
              onDelete={(e) => handleDelete(e, project.id)}
            />
          ))}
        </div>
      )}

      {/* Create Project Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="amp-panel rounded-xl p-6 w-full max-w-md mx-4"
          >
            <h3 className="text-xl font-semibold text-amp-cream mb-4">
              Create New Project
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-amp-silver mb-1">
                  Project Name
                </label>
                <input
                  type="text"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  placeholder="e.g., SRV Texas Flood Tone"
                  className="input-field"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-amp-silver mb-1">
                  Description (optional)
                </label>
                <textarea
                  value={newProjectDesc}
                  onChange={(e) => setNewProjectDesc(e.target.value)}
                  placeholder="Notes about the reference tone..."
                  rows={3}
                  className="input-field resize-none"
                />
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleCreate}
                disabled={!newProjectName.trim() || creating}
                className="btn-primary flex items-center gap-2"
              >
                {creating ? (
                  <Loader className="w-4 h-4 animate-spin" />
                ) : (
                  <Plus className="w-4 h-4" />
                )}
                Create
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}

function ProjectCard({
  project,
  index,
  onClick,
  onDelete,
}: {
  project: Project;
  index: number;
  onClick: () => void;
  onDelete: (e: React.MouseEvent) => void;
}) {
  const statusConfig = {
    created: { icon: Clock, color: 'text-amp-silver', label: 'New' },
    imported: { icon: CheckCircle, color: 'text-amp-blue', label: 'Imported' },
    isolating: { icon: Loader, color: 'text-amp-amber', label: 'Isolating...' },
    isolated: { icon: CheckCircle, color: 'text-amp-amber', label: 'Isolated' },
    matching: { icon: Loader, color: 'text-amp-orange', label: 'Matching...' },
    matched: { icon: CheckCircle, color: 'text-amp-green', label: 'Matched' },
    error: { icon: AlertCircle, color: 'text-amp-red', label: 'Error' },
  };

  const status = statusConfig[project.status];
  const StatusIcon = status.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      onClick={onClick}
      className="amp-panel rounded-xl p-5 cursor-pointer hover:border-amp-amber/50 transition-all group"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <StatusIcon
            className={`w-5 h-5 ${status.color} ${
              project.status.includes('ing') ? 'animate-spin' : ''
            }`}
          />
          <span className={`text-xs font-medium ${status.color}`}>
            {status.label}
          </span>
        </div>
        <button
          onClick={onDelete}
          className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-amp-red/20 transition-all"
        >
          <Trash2 className="w-4 h-4 text-amp-red" />
        </button>
      </div>

      <h3 className="text-lg font-semibold text-amp-cream mb-1 truncate">
        {project.name}
      </h3>

      {project.description && (
        <p className="text-sm text-amp-silver line-clamp-2 mb-3">
          {project.description}
        </p>
      )}

      <div className="text-xs text-amp-chrome">
        Updated {new Date(project.updated_at).toLocaleDateString()}
      </div>
    </motion.div>
  );
}


