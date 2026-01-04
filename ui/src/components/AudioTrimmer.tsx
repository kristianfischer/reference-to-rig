import { useEffect, useRef, useState, useCallback } from 'react';
import { Play, Pause, Volume2 } from 'lucide-react';

interface AudioTrimmerProps {
  audioUrl: string;
  onTrimChange: (startTime: number, endTime: number) => void;
  onPanChange: (pan: number) => void;
  initialStart?: number;
  initialEnd?: number;
  initialPan?: number;
}

export default function AudioTrimmer({
  audioUrl,
  onTrimChange,
  onPanChange,
  initialStart = 0,
  initialEnd,
  initialPan = 0,
}: AudioTrimmerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  
  const [waveformData, setWaveformData] = useState<number[]>([]);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [startTime, setStartTime] = useState(initialStart);
  const [endTime, setEndTime] = useState(initialEnd || 0);
  const [pan, setPan] = useState(initialPan);
  const [isDragging, setIsDragging] = useState<'start' | 'end' | 'playhead' | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [audioReady, setAudioReady] = useState(false);

  // Load audio and generate waveform
  useEffect(() => {
    const loadAudio = async () => {
      setIsLoading(true);
      try {
        const response = await fetch(audioUrl);
        const arrayBuffer = await response.arrayBuffer();
        
        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        // Generate waveform data
        const channelData = audioBuffer.getChannelData(0);
        const samples = 200; // Number of bars in waveform
        const blockSize = Math.floor(channelData.length / samples);
        const waveform: number[] = [];
        
        for (let i = 0; i < samples; i++) {
          let sum = 0;
          for (let j = 0; j < blockSize; j++) {
            sum += Math.abs(channelData[i * blockSize + j]);
          }
          waveform.push(sum / blockSize);
        }
        
        // Normalize
        const max = Math.max(...waveform);
        const normalizedWaveform = waveform.map(v => v / max);
        
        setWaveformData(normalizedWaveform);
        setDuration(audioBuffer.duration);
        if (!initialEnd) {
          setEndTime(audioBuffer.duration);
          onTrimChange(initialStart, audioBuffer.duration);
        }
        
        audioContext.close();
      } catch (error) {
        console.error('Failed to load audio:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadAudio();
  }, [audioUrl]);

  // Setup audio element events
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleCanPlay = () => {
      setAudioReady(true);
    };

    const handleLoadedMetadata = () => {
      if (audio.duration && !isNaN(audio.duration)) {
        setDuration(audio.duration);
      }
    };

    audio.addEventListener('canplay', handleCanPlay);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);

    return () => {
      audio.removeEventListener('canplay', handleCanPlay);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
    };
  }, []);

  // Update pan value (for display and to pass to parent)
  useEffect(() => {
    onPanChange(pan);
  }, [pan, onPanChange]);

  // Draw waveform
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || waveformData.length === 0) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const width = canvas.width;
    const height = canvas.height;
    const barWidth = width / waveformData.length;
    
    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, width, height);
    
    // Draw selection region
    const startX = (startTime / duration) * width;
    const endX = (endTime / duration) * width;
    
    // Dimmed regions outside selection
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(0, 0, startX, height);
    ctx.fillRect(endX, 0, width - endX, height);
    
    // Selection highlight
    ctx.fillStyle = 'rgba(245, 158, 11, 0.1)';
    ctx.fillRect(startX, 0, endX - startX, height);
    
    // Draw waveform
    waveformData.forEach((value, index) => {
      const x = index * barWidth;
      const barHeight = value * (height * 0.8);
      const y = (height - barHeight) / 2;
      
      // Color based on position
      const position = index / waveformData.length;
      const time = position * duration;
      
      if (time >= startTime && time <= endTime) {
        ctx.fillStyle = '#f59e0b'; // Amber for selected
      } else {
        ctx.fillStyle = '#4a4a4a'; // Dim for unselected
      }
      
      ctx.fillRect(x, y, barWidth - 1, barHeight);
    });
    
    // Draw playhead
    const playheadX = (currentTime / duration) * width;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(playheadX, 0);
    ctx.lineTo(playheadX, height);
    ctx.stroke();
    
    // Draw trim handles
    ctx.fillStyle = '#f59e0b';
    
    // Start handle
    ctx.beginPath();
    ctx.moveTo(startX, 0);
    ctx.lineTo(startX + 10, 0);
    ctx.lineTo(startX + 10, 15);
    ctx.lineTo(startX, 15);
    ctx.closePath();
    ctx.fill();
    
    ctx.beginPath();
    ctx.moveTo(startX, height);
    ctx.lineTo(startX + 10, height);
    ctx.lineTo(startX + 10, height - 15);
    ctx.lineTo(startX, height - 15);
    ctx.closePath();
    ctx.fill();
    
    // Start line
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(startX, 0);
    ctx.lineTo(startX, height);
    ctx.stroke();
    
    // End handle
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.moveTo(endX, 0);
    ctx.lineTo(endX - 10, 0);
    ctx.lineTo(endX - 10, 15);
    ctx.lineTo(endX, 15);
    ctx.closePath();
    ctx.fill();
    
    ctx.beginPath();
    ctx.moveTo(endX, height);
    ctx.lineTo(endX - 10, height);
    ctx.lineTo(endX - 10, height - 15);
    ctx.lineTo(endX, height - 15);
    ctx.closePath();
    ctx.fill();
    
    // End line
    ctx.beginPath();
    ctx.moveTo(endX, 0);
    ctx.lineTo(endX, height);
    ctx.stroke();
    
  }, [waveformData, duration, startTime, endTime, currentTime]);

  // Handle mouse events for trimming
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const time = (x / rect.width) * duration;
    
    const startX = (startTime / duration) * rect.width;
    const endX = (endTime / duration) * rect.width;
    
    // Check if clicking near handles (within 15px)
    if (Math.abs(x - startX) < 15) {
      setIsDragging('start');
    } else if (Math.abs(x - endX) < 15) {
      setIsDragging('end');
    } else {
      // Click to set playhead
      setCurrentTime(time);
      if (audioRef.current) {
        audioRef.current.currentTime = time;
      }
    }
  }, [duration, startTime, endTime]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const time = (x / rect.width) * duration;
    
    if (isDragging === 'start') {
      const newStart = Math.min(time, endTime - 0.5); // Minimum 0.5s selection
      setStartTime(Math.max(0, newStart));
      onTrimChange(Math.max(0, newStart), endTime);
    } else if (isDragging === 'end') {
      const newEnd = Math.max(time, startTime + 0.5);
      setEndTime(Math.min(duration, newEnd));
      onTrimChange(startTime, Math.min(duration, newEnd));
    }
  }, [isDragging, duration, startTime, endTime, onTrimChange]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(null);
  }, []);

  // Playback controls - simplified without Web Audio API
  const togglePlayback = () => {
    const audio = audioRef.current;
    if (!audio) return;
    
    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      // Start from selection start if outside selection
      if (audio.currentTime < startTime || audio.currentTime >= endTime) {
        audio.currentTime = startTime;
      }
      
      audio.play()
        .then(() => setIsPlaying(true))
        .catch((e) => console.error('Playback failed:', e));
    }
  };

  // Update time display and handle looping
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    
    const updateTime = () => {
      setCurrentTime(audio.currentTime);
      
      // Loop within selection when playing
      if (isPlaying && audio.currentTime >= endTime) {
        audio.currentTime = startTime;
      }
    };
    
    const handleEnded = () => setIsPlaying(false);
    const handlePause = () => setIsPlaying(false);
    
    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('pause', handlePause);
    
    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('pause', handlePause);
    };
  }, [startTime, endTime, isPlaying]);

  const formatTime = (seconds: number) => {
    if (isNaN(seconds)) return '0:00.0';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 10);
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`;
  };

  return (
    <div className="space-y-4">
      {/* Hidden audio element for playback */}
      <audio 
        ref={audioRef} 
        src={audioUrl} 
        preload="auto"
        crossOrigin="anonymous"
      />
      
      {/* Waveform */}
      <div className="relative">
        {isLoading ? (
          <div className="h-24 bg-amp-black rounded-lg flex items-center justify-center">
            <div className="text-amp-silver">Loading waveform...</div>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            width={800}
            height={96}
            className="w-full h-24 rounded-lg cursor-pointer"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />
        )}
      </div>
      
      {/* Time display and controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={togglePlayback}
            className="btn-secondary p-2"
            disabled={isLoading}
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          </button>
          
          <div className="text-sm font-mono text-amp-silver">
            <span className="text-amp-cream">{formatTime(currentTime)}</span>
            <span className="mx-2">/</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>
        
        <div className="flex items-center gap-6">
          {/* Selection times */}
          <div className="text-sm text-amp-silver">
            Selection: <span className="text-amp-amber font-mono">{formatTime(startTime)}</span>
            <span className="mx-1">-</span>
            <span className="text-amp-amber font-mono">{formatTime(endTime)}</span>
            <span className="ml-2 text-amp-chrome">({formatTime(endTime - startTime)})</span>
          </div>
        </div>
      </div>
      
      {/* Pan control */}
      <div className="flex items-center gap-4">
        <Volume2 className="w-4 h-4 text-amp-silver" />
        <span className="text-sm text-amp-silver w-8">L</span>
        <input
          type="range"
          min="-100"
          max="100"
          value={pan}
          onChange={(e) => setPan(parseInt(e.target.value))}
          className="flex-1 h-2 bg-amp-steel rounded-lg appearance-none cursor-pointer accent-amp-amber"
        />
        <span className="text-sm text-amp-silver w-8">R</span>
        <span className="text-sm font-mono text-amp-chrome w-12 text-right">
          {pan === 0 ? 'C' : pan > 0 ? `R${pan}` : `L${Math.abs(pan)}`}
        </span>
      </div>
      
      <p className="text-xs text-amp-silver">
        Drag the orange handles to select the portion containing the guitar tone you want to match.
        Pan setting is applied during isolation (not preview playback).
      </p>
    </div>
  );
}
