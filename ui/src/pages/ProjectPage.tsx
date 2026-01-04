import { useEffect, useRef, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Upload,
  Play,
  Pause,
  Loader,
  CheckCircle,
  ArrowRight,
  Music,
  Wand2,
  MessageSquare,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { useStore } from '../store/useStore';
import { api } from '../api/client';
import AudioTrimmer from '../components/AudioTrimmer';

// Preset prompts for guitar isolation
const PRESET_PROMPTS = [
  { label: 'Electric Guitar', value: 'electric guitar' },
  { label: 'Distorted Guitar', value: 'distorted electric guitar' },
  { label: 'Clean Guitar', value: 'clean electric guitar' },
  { label: 'Acoustic Guitar', value: 'acoustic guitar' },
  { label: 'Bass Guitar', value: 'bass guitar' },
  { label: 'Lead Guitar', value: 'lead guitar solo' },
  { label: 'Rhythm Guitar', value: 'rhythm guitar' },
];

export default function ProjectPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const {
    currentProject,
    currentTask,
    fetchProject,
    importAudio,
    startIsolation,
    pollTaskStatus,
    clearCurrentTask,
  } = useStore();

  const [uploading, setUploading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playingType, setPlayingType] = useState<'source' | 'isolated' | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Isolation options
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState(0);
  const [pan, setPan] = useState(0);
  const [customPrompt, setCustomPrompt] = useState('electric guitar');
  const [useCustomPrompt, setUseCustomPrompt] = useState(false);

  useEffect(() => {
    if (id) {
      fetchProject(id);
      clearCurrentTask();
    }
  }, [id]);

  // Poll task status
  useEffect(() => {
    if (!currentTask || !id) return;
    if (currentTask.status === 'completed' || currentTask.status === 'failed') {
      fetchProject(id);
      return;
    }

    const interval = setInterval(async () => {
      await pollTaskStatus(id, currentTask.task_id);
    }, 1000);

    return () => clearInterval(interval);
  }, [currentTask?.task_id, currentTask?.status, id]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !id) return;

    setUploading(true);
    try {
      await importAudio(id, file);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Failed to upload audio file');
    } finally {
      setUploading(false);
    }
  };

  const handleIsolate = async () => {
    if (!id) return;
    
    // Pass isolation options
    const options = {
      trim_start: trimStart,
      trim_end: trimEnd,
      pan: pan,
      prompt: customPrompt,
    };
    
    await startIsolation(id, options);
  };

  const handlePlay = (type: 'source' | 'isolated') => {
    if (!audioRef.current || !id) return;

    if (playingType === type && isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.src = api.getAudioUrl(id, type);
      audioRef.current.play();
      setIsPlaying(true);
      setPlayingType(type);
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
    setPlayingType(null);
  };

  const handleTrimChange = useCallback((start: number, end: number) => {
    setTrimStart(start);
    setTrimEnd(end);
  }, []);

  const handlePanChange = useCallback((newPan: number) => {
    setPan(newPan);
  }, []);

  if (!currentProject) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader className="w-8 h-8 text-amp-amber animate-spin" />
      </div>
    );
  }

  const canIsolate = ['imported', 'isolated', 'matched'].includes(currentProject.status);
  const canMatch = ['isolated', 'matched'].includes(currentProject.status);
  const isProcessing = currentProject.status.includes('ing');

  return (
    <div>
      <audio ref={audioRef} onEnded={handleAudioEnded} />

      {/* Header */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-amp-cream">{currentProject.name}</h2>
        {currentProject.description && (
          <p className="text-amp-silver mt-1">{currentProject.description}</p>
        )}
      </div>

      {/* Steps */}
      <div className="space-y-6">
        {/* Step 1: Import Audio */}
        <StepCard
          number={1}
          title="Import Reference Audio"
          description="Upload a WAV or MP3 file containing the guitar tone you want to match"
          completed={!!currentProject.source_audio_path}
          active={!currentProject.source_audio_path}
        >
          {currentProject.source_audio_path ? (
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="flex-1 amp-panel rounded-lg p-4">
                  <div className="flex items-center gap-3">
                    <Music className="w-5 h-5 text-amp-amber" />
                    <span className="text-amp-cream">Source audio imported</span>
                  </div>
                </div>
                <button
                  onClick={() => handlePlay('source')}
                  className="btn-secondary flex items-center gap-2"
                >
                  {isPlaying && playingType === 'source' ? (
                    <Pause className="w-4 h-4" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                  {isPlaying && playingType === 'source' ? 'Stop' : 'Play'}
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn-secondary"
                >
                  Replace
                </button>
              </div>

              {/* Audio Trimmer */}
              <div className="amp-panel rounded-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-amp-cream font-medium">Select Guitar Region</h4>
                  <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="text-sm text-amp-silver hover:text-amp-cream flex items-center gap-1"
                  >
                    {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    {showAdvanced ? 'Hide options' : 'Show options'}
                  </button>
                </div>
                
                <AudioTrimmer
                  audioUrl={api.getAudioUrl(id!, 'source')}
                  onTrimChange={handleTrimChange}
                  onPanChange={handlePanChange}
                  initialStart={trimStart}
                  initialEnd={trimEnd || undefined}
                  initialPan={pan}
                />

                {/* Advanced options - custom prompt */}
                {showAdvanced && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-4 pt-4 border-t border-amp-steel"
                  >
                    <div className="space-y-4">
                      <div>
                        <label className="flex items-center gap-2 text-sm text-amp-cream mb-2">
                          <MessageSquare className="w-4 h-4 text-amp-amber" />
                          Isolation Prompt
                        </label>
                        <p className="text-xs text-amp-silver mb-3">
                          Describe what you want to isolate. Be specific about the instrument type.
                        </p>
                        
                        {/* Preset buttons */}
                        <div className="flex flex-wrap gap-2 mb-3">
                          {PRESET_PROMPTS.map((preset) => (
                            <button
                              key={preset.value}
                              onClick={() => {
                                setCustomPrompt(preset.value);
                                setUseCustomPrompt(false);
                              }}
                              className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                                customPrompt === preset.value && !useCustomPrompt
                                  ? 'bg-amp-amber text-amp-black'
                                  : 'bg-amp-steel text-amp-silver hover:text-amp-cream'
                              }`}
                            >
                              {preset.label}
                            </button>
                          ))}
                        </div>
                        
                        {/* Custom prompt input */}
                        <div className="flex gap-2">
                          <input
                            type="text"
                            value={customPrompt}
                            onChange={(e) => {
                              setCustomPrompt(e.target.value);
                              setUseCustomPrompt(true);
                            }}
                            onFocus={() => setUseCustomPrompt(true)}
                            placeholder="e.g., distorted electric guitar with heavy overdrive"
                            className="flex-1 bg-amp-black border border-amp-steel rounded-lg px-4 py-2 text-amp-cream placeholder:text-amp-silver focus:outline-none focus:border-amp-amber"
                          />
                        </div>
                        <p className="text-xs text-amp-silver mt-2">
                          Tip: Use descriptive phrases like "clean Fender guitar" or "high gain metal rhythm guitar"
                        </p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            </div>
          ) : (
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-amp-steel rounded-xl p-8 text-center cursor-pointer hover:border-amp-amber/50 transition-colors"
            >
              {uploading ? (
                <Loader className="w-8 h-8 text-amp-amber animate-spin mx-auto mb-3" />
              ) : (
                <Upload className="w-8 h-8 text-amp-chrome mx-auto mb-3" />
              )}
              <p className="text-amp-cream font-medium mb-1">
                {uploading ? 'Uploading...' : 'Drop audio file here or click to browse'}
              </p>
              <p className="text-sm text-amp-silver">
                Supports WAV, MP3, FLAC, OGG, M4A
              </p>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.flac,.ogg,.m4a"
            onChange={handleFileSelect}
            className="hidden"
          />
        </StepCard>

        {/* Step 2: Isolate Guitar */}
        <StepCard
          number={2}
          title="Isolate Guitar"
          description="Extract the guitar track from your reference audio using AI"
          completed={!!currentProject.isolated_audio_path}
          active={canIsolate && !currentProject.isolated_audio_path}
          disabled={!canIsolate}
        >
          {currentTask && currentProject.status === 'isolating' ? (
            <ProgressBar
              progress={currentTask.progress}
              message={currentTask.message || 'Processing...'}
            />
          ) : currentProject.isolated_audio_path ? (
            <div className="flex items-center gap-4">
              <div className="flex-1 amp-panel rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-amp-green" />
                    <span className="text-amp-cream">Guitar isolated</span>
                  </div>
                  {currentProject.isolation_confidence && (
                    <span className="text-sm text-amp-silver">
                      Confidence: {(currentProject.isolation_confidence * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              </div>
              <button
                onClick={() => handlePlay('isolated')}
                className="btn-secondary flex items-center gap-2"
              >
                {isPlaying && playingType === 'isolated' ? (
                  <Pause className="w-4 h-4" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                {isPlaying && playingType === 'isolated' ? 'Stop' : 'Play'}
              </button>
              <button onClick={handleIsolate} className="btn-secondary">
                Re-run
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              {/* Show current settings summary */}
              <div className="text-sm text-amp-silver">
                <span className="text-amp-chrome">Prompt:</span> {customPrompt}
                {trimEnd > 0 && (
                  <span className="ml-4">
                    <span className="text-amp-chrome">Region:</span>{' '}
                    {trimStart.toFixed(1)}s - {trimEnd.toFixed(1)}s
                  </span>
                )}
                {pan !== 0 && (
                  <span className="ml-4">
                    <span className="text-amp-chrome">Pan:</span>{' '}
                    {pan > 0 ? `R${pan}` : `L${Math.abs(pan)}`}
                  </span>
                )}
              </div>
              <button
                onClick={handleIsolate}
                disabled={!canIsolate || isProcessing}
                className="btn-primary flex items-center gap-2"
              >
                <Wand2 className="w-4 h-4" />
                Isolate Guitar Track
              </button>
            </div>
          )}
        </StepCard>

        {/* Step 3: Match Tone */}
        <StepCard
          number={3}
          title="Match Tone"
          description="Find NAM captures and IRs that match your reference tone"
          completed={currentProject.status === 'matched'}
          active={canMatch}
          disabled={!canMatch}
        >
          <button
            onClick={() => navigate(`/project/${id}/matching`)}
            disabled={!canMatch}
            className="btn-primary flex items-center gap-2"
          >
            <ArrowRight className="w-4 h-4" />
            {currentProject.status === 'matched' ? 'View Results' : 'Start Matching'}
          </button>
        </StepCard>
      </div>
    </div>
  );
}

function StepCard({
  number,
  title,
  description,
  completed,
  active,
  disabled,
  children,
}: {
  number: number;
  title: string;
  description: string;
  completed: boolean;
  active: boolean;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: number * 0.1 }}
      className={`amp-panel rounded-xl p-6 ${disabled ? 'opacity-50' : ''}`}
    >
      <div className="flex items-start gap-4">
        <div
          className={`
            w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold
            ${completed ? 'bg-amp-green text-amp-black' : active ? 'bg-amp-amber text-amp-black' : 'bg-amp-steel text-amp-silver'}
          `}
        >
          {completed ? <CheckCircle className="w-5 h-5" /> : number}
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-amp-cream mb-1">{title}</h3>
          <p className="text-sm text-amp-silver mb-4">{description}</p>
          {children}
        </div>
      </div>
    </motion.div>
  );
}

function ProgressBar({ progress, message }: { progress: number; message: string }) {
  return (
    <div>
      <div className="flex items-center justify-between text-sm mb-2">
        <span className="text-amp-silver">{message}</span>
        <span className="text-amp-amber font-mono">{(progress * 100).toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-amp-black rounded-full overflow-hidden">
        <motion.div
          className="h-full progress-bar rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${progress * 100}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>
    </div>
  );
}
