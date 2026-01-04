import { useEffect, useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Play,
  Pause,
  Download,
  FileJson,
  FileText,
  Loader,
  Trophy,
  Sun,
  Moon,
  CheckCircle,
} from 'lucide-react';
import { useStore, MatchCandidate } from '../store/useStore';
import { api } from '../api/client';

export default function ResultsPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { currentProject, matchResults, fetchProject, fetchResults } = useStore();

  const [playing, setPlaying] = useState<string | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<MatchCandidate | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (id) {
      fetchProject(id);
      fetchResults(id);
    }
  }, [id]);

  const handlePlay = (audioType: string) => {
    if (!audioRef.current || !id) return;

    if (playing === audioType) {
      audioRef.current.pause();
      setPlaying(null);
    } else {
      audioRef.current.src = api.getAudioUrl(id, audioType);
      audioRef.current.play();
      setPlaying(audioType);
    }
  };

  const handleAudioEnded = () => {
    setPlaying(null);
  };

  const handleExportJson = async () => {
    if (!id) return;
    try {
      const data = await api.exportJson(id);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${currentProject?.name || 'rig_recipe'}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  const handleExportMarkdown = () => {
    if (!id) return;
    window.open(api.getMarkdownUrl(id), '_blank');
  };

  if (!currentProject || !matchResults) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader className="w-8 h-8 text-amp-amber animate-spin" />
      </div>
    );
  }

  const flavorIcons: Record<string, React.ComponentType<any>> = {
    balanced: Trophy,
    brighter: Sun,
    thicker: Moon,
  };

  const flavorColors: Record<string, string> = {
    balanced: 'text-amp-green',
    brighter: 'text-amp-amber',
    thicker: 'text-amp-orange',
  };

  return (
    <div>
      <audio ref={audioRef} onEnded={handleAudioEnded} />

      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-2xl font-bold text-amp-cream">Match Results</h2>
          <p className="text-amp-silver mt-1">
            {currentProject.name} • Confidence: {(matchResults.isolation_confidence * 100).toFixed(0)}%
          </p>
        </div>
        <div className="flex gap-3">
          <button onClick={handleExportJson} className="btn-secondary flex items-center gap-2">
            <FileJson className="w-4 h-4" />
            Export JSON
          </button>
          <button onClick={handleExportMarkdown} className="btn-primary flex items-center gap-2">
            <FileText className="w-4 h-4" />
            Export Recipe
          </button>
        </div>
      </div>

      {/* A/B Comparison */}
      <div className="amp-panel rounded-xl p-6 mb-8">
        <h3 className="text-lg font-semibold text-amp-cream mb-4">A/B Comparison</h3>
        <div className="flex items-center gap-4">
          <button
            onClick={() => handlePlay('isolated')}
            className={`flex-1 p-4 rounded-lg border transition-all ${
              playing === 'isolated'
                ? 'border-amp-amber bg-amp-amber/10'
                : 'border-amp-steel hover:border-amp-chrome'
            }`}
          >
            <div className="flex items-center justify-center gap-3">
              {playing === 'isolated' ? (
                <Pause className="w-5 h-5 text-amp-amber" />
              ) : (
                <Play className="w-5 h-5 text-amp-silver" />
              )}
              <span className="font-medium text-amp-cream">Reference (Isolated)</span>
            </div>
          </button>

          <span className="text-amp-chrome font-bold">VS</span>

          {matchResults.candidates.map((candidate) => (
            <button
              key={candidate.flavor}
              onClick={() => handlePlay(`match_${candidate.flavor}`)}
              className={`flex-1 p-4 rounded-lg border transition-all ${
                playing === `match_${candidate.flavor}`
                  ? 'border-amp-amber bg-amp-amber/10'
                  : 'border-amp-steel hover:border-amp-chrome'
              }`}
            >
              <div className="flex items-center justify-center gap-3">
                {playing === `match_${candidate.flavor}` ? (
                  <Pause className="w-5 h-5 text-amp-amber" />
                ) : (
                  <Play className="w-5 h-5 text-amp-silver" />
                )}
                <span className="font-medium text-amp-cream capitalize">
                  {candidate.flavor} Match
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Match Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {matchResults.candidates.map((candidate, index) => {
          const FlavorIcon = flavorIcons[candidate.flavor] || Trophy;
          const isSelected = selectedCandidate?.flavor === candidate.flavor;

          return (
            <motion.div
              key={candidate.flavor}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => setSelectedCandidate(isSelected ? null : candidate)}
              className={`amp-panel rounded-xl overflow-hidden cursor-pointer transition-all ${
                isSelected ? 'ring-2 ring-amp-amber' : ''
              }`}
            >
              {/* Header */}
              <div className={`p-4 bg-gradient-to-r ${
                candidate.flavor === 'balanced' ? 'from-amp-green/20 to-transparent' :
                candidate.flavor === 'brighter' ? 'from-amp-amber/20 to-transparent' :
                'from-amp-orange/20 to-transparent'
              }`}>
                <div className="flex items-center gap-3">
                  <FlavorIcon className={`w-6 h-6 ${flavorColors[candidate.flavor]}`} />
                  <div>
                    <h4 className="font-bold text-amp-cream capitalize">{candidate.flavor}</h4>
                    <p className="text-sm text-amp-silver">
                      {(candidate.similarity_score * 100).toFixed(0)}% match
                    </p>
                  </div>
                </div>
              </div>

              {/* Content */}
              <div className="p-4 space-y-4">
                {/* NAM Model */}
                <div>
                  <label className="text-xs text-amp-chrome uppercase tracking-wide">
                    NAM Model
                  </label>
                  <p className="text-amp-cream font-medium">{candidate.nam_model_name}</p>
                </div>

                {/* IR */}
                <div>
                  <label className="text-xs text-amp-chrome uppercase tracking-wide">
                    Cabinet IR
                  </label>
                  <p className="text-amp-cream font-medium">{candidate.ir_name}</p>
                </div>

                {/* Input Gain */}
                <div>
                  <label className="text-xs text-amp-chrome uppercase tracking-wide">
                    Input Gain
                  </label>
                  <p className="text-amp-cream font-mono">
                    {candidate.input_gain_db >= 0 ? '+' : ''}{candidate.input_gain_db.toFixed(1)} dB
                  </p>
                </div>

                {/* Play button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handlePlay(`match_${candidate.flavor}`);
                  }}
                  className="w-full btn-secondary flex items-center justify-center gap-2"
                >
                  {playing === `match_${candidate.flavor}` ? (
                    <>
                      <Pause className="w-4 h-4" />
                      Stop
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      Play
                    </>
                  )}
                </button>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* EQ Details */}
      {selectedCandidate && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="amp-panel rounded-xl p-6 mt-8"
        >
          <h3 className="text-lg font-semibold text-amp-cream mb-4 capitalize">
            {selectedCandidate.flavor} Match - EQ Settings
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-amp-steel">
                  <th className="py-2 px-4 text-left text-amp-chrome font-medium">Band</th>
                  <th className="py-2 px-4 text-left text-amp-chrome font-medium">Frequency</th>
                  <th className="py-2 px-4 text-left text-amp-chrome font-medium">Gain</th>
                  <th className="py-2 px-4 text-left text-amp-chrome font-medium">Q</th>
                </tr>
              </thead>
              <tbody>
                {selectedCandidate.eq_settings.bands.map((band, i) => (
                  <tr key={i} className="border-b border-amp-steel/50">
                    <td className="py-2 px-4 text-amp-cream capitalize">{band.band_type}</td>
                    <td className="py-2 px-4 text-amp-cream font-mono">{band.frequency.toFixed(0)} Hz</td>
                    <td className="py-2 px-4 text-amp-cream font-mono">
                      {band.gain_db >= 0 ? '+' : ''}{band.gain_db.toFixed(1)} dB
                    </td>
                    <td className="py-2 px-4 text-amp-cream font-mono">{band.q.toFixed(1)}</td>
                  </tr>
                ))}
                {selectedCandidate.eq_settings.highpass_freq && (
                  <tr className="border-b border-amp-steel/50">
                    <td className="py-2 px-4 text-amp-cream">Highpass</td>
                    <td className="py-2 px-4 text-amp-cream font-mono">
                      {selectedCandidate.eq_settings.highpass_freq.toFixed(0)} Hz
                    </td>
                    <td className="py-2 px-4 text-amp-chrome">-</td>
                    <td className="py-2 px-4 text-amp-chrome">-</td>
                  </tr>
                )}
                {selectedCandidate.eq_settings.lowpass_freq && (
                  <tr>
                    <td className="py-2 px-4 text-amp-cream">Lowpass</td>
                    <td className="py-2 px-4 text-amp-cream font-mono">
                      {selectedCandidate.eq_settings.lowpass_freq.toFixed(0)} Hz
                    </td>
                    <td className="py-2 px-4 text-amp-chrome">-</td>
                    <td className="py-2 px-4 text-amp-chrome">-</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}

      {/* Back button */}
      <div className="mt-8 flex justify-center">
        <button
          onClick={() => navigate(`/project/${id}`)}
          className="btn-secondary"
        >
          Back to Project
        </button>
      </div>
    </div>
  );
}


