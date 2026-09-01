'use client';

import {
  Activity,
  Check,
  CircleAlert,
  Clock3,
  Gauge,
  Mic,
  Pause,
  Play,
  RotateCcw,
  Square,
  Volume2,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

type SessionMode =
  | 'idle'
  | 'requesting'
  | 'calibrating'
  | 'live'
  | 'paused'
  | 'done'
  | 'error';

type InputState = 'ready' | 'good' | 'silent' | 'loud';

type VoiceMetrics = {
  tension: number;
  energy: number;
  pace: number;
  stability: number;
  confidence: number;
  pitch: number | null;
  jitter: number;
  volume: number;
};

type SessionSummary = {
  tension: number;
  energy: number;
  pace: number;
  stability: number;
  duration: number;
};

type SpeechAlternativeLike = { transcript: string };
type SpeechResultLike = {
  [index: number]: SpeechAlternativeLike;
  isFinal: boolean;
  length: number;
};
type SpeechResultListLike = {
  [index: number]: SpeechResultLike;
  length: number;
};
type SpeechRecognitionEventLike = Event & {
  resultIndex: number;
  results: SpeechResultListLike;
};
type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onend: (() => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  start: () => void;
  stop: () => void;
};
type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

type TensionEngine = {
  score: (
    pace: number,
    pitch: number,
    jitter: number,
    pause: number,
    volume: number,
  ) => number;
};

type IntensityFrame = {
  decibels: number;
  time: number;
  voiced: boolean;
};

const DEFAULT_METRICS: VoiceMetrics = {
  tension: 32,
  energy: 68,
  pace: 7.2,
  stability: 82,
  confidence: 94,
  pitch: 176,
  jitter: 0.018,
  volume: -24,
};

const clamp = (value: number, minimum = 0, maximum = 100) =>
  Math.min(maximum, Math.max(minimum, value));

const round = (value: number) => Math.round(value);
const roundOne = (value: number) => Math.round(value * 10) / 10;

const mean = (values: number[]) =>
  values.length
    ? values.reduce((total, value) => total + value, 0) / values.length
    : 0;

const median = (values: number[]) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
};

const percentile = (values: number[], ratio: number) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(clamp(ratio, 0, 1) * (sorted.length - 1))];
};

const deviation = (values: number[]) => {
  if (values.length < 2) return 0;
  const average = mean(values);
  return Math.sqrt(mean(values.map((value) => Math.pow(value - average, 2))));
};

const medianAbsoluteDeviation = (values: number[]) => {
  if (values.length < 2) return 0;
  const center = median(values);
  return median(values.map((value) => Math.abs(value - center)));
};

const robustPositiveZ = (
  value: number,
  baselineValues: number[],
  fallback: number,
  minimumScale: number,
) => {
  const center = baselineValues.length ? median(baselineValues) : fallback;
  const robustScale = baselineValues.length
    ? medianAbsoluteDeviation(baselineValues) * 1.4826
    : 0;
  return Math.max(0, (value - center) / Math.max(minimumScale, robustScale));
};

const stressFromZ = (zScore: number) => clamp(30 + zScore * 24);

const formatTime = (seconds: number) => {
  const minutes = Math.floor(seconds / 60)
    .toString()
    .padStart(2, '0');
  const rest = Math.floor(seconds % 60)
    .toString()
    .padStart(2, '0');
  return `${minutes}:${rest}`;
};

function estimateMoraUnits(text: string) {
  const clean = text.normalize('NFKC').trim();
  if (!clean) return 0;

  let units = 0;
  for (const character of clean) {
    if (/[ゃゅょャュョぁぃぅぇぉァィゥェォゎヮ]/u.test(character)) {
      continue;
    }
    if (/[ぁ-ゖァ-ヺー]/u.test(character)) {
      units += 1;
      continue;
    }
    if (/[㐀-䶿一-鿿々]/u.test(character)) {
      units += 2;
      continue;
    }
    if (/[0-9]/u.test(character)) units += 1.5;
  }

  return units;
}

function detectPitch(buffer: Float32Array, sampleRate: number) {
  let squareTotal = 0;
  let sampleTotal = 0;
  for (const sample of buffer) {
    squareTotal += sample * sample;
    sampleTotal += sample;
  }
  const rms = Math.sqrt(squareTotal / buffer.length);
  if (rms < 0.006) {
    return { pitch: null as number | null, confidence: 0, jitter: null };
  }

  const minimumLag = Math.floor(sampleRate / 500);
  const maximumLag = Math.min(
    Math.floor(sampleRate / 60),
    Math.floor(buffer.length / 2) - 1,
  );
  const differences = new Float32Array(maximumLag + 1);
  const normalized = new Float32Array(maximumLag + 1);

  for (let lag = 1; lag <= maximumLag; lag += 1) {
    let difference = 0;
    for (let index = 0; index < maximumLag; index += 1) {
      const delta = buffer[index] - buffer[index + lag];
      difference += delta * delta;
    }
    differences[lag] = difference;
  }

  normalized[0] = 1;
  let runningSum = 0;
  for (let lag = 1; lag <= maximumLag; lag += 1) {
    runningSum += differences[lag];
    normalized[lag] = runningSum ? (differences[lag] * lag) / runningSum : 1;
  }

  let candidate = 0;
  for (let lag = minimumLag; lag <= maximumLag; lag += 1) {
    if (normalized[lag] < 0.15) {
      candidate = lag;
      while (
        candidate + 1 <= maximumLag &&
        normalized[candidate + 1] < normalized[candidate]
      ) {
        candidate += 1;
      }
      break;
    }
  }

  if (!candidate) {
    let bestValue = 1;
    for (let lag = minimumLag; lag <= maximumLag; lag += 1) {
      if (normalized[lag] < bestValue) {
        bestValue = normalized[lag];
        candidate = lag;
      }
    }
  }

  const confidence = candidate ? 1 - normalized[candidate] : 0;
  if (!candidate || confidence < 0.7) {
    return { pitch: null as number | null, confidence, jitter: null };
  }

  const left = normalized[Math.max(minimumLag, candidate - 1)];
  const center = normalized[candidate];
  const right = normalized[Math.min(maximumLag, candidate + 1)];
  const denominator = left - 2 * center + right;
  const refinedLag = denominator
    ? candidate + (left - right) / (2 * denominator)
    : candidate;

  const bufferMean = sampleTotal / buffer.length;
  const crossings: number[] = [];
  for (let index = 1; index < buffer.length; index += 1) {
    const previous = buffer[index - 1] - bufferMean;
    const current = buffer[index] - bufferMean;
    if (previous <= 0 && current > 0) {
      const fraction = -previous / Math.max(1e-9, current - previous);
      crossings.push(index - 1 + fraction);
    }
  }
  const periods = crossings
    .slice(1)
    .map((crossing, index) => crossing - crossings[index])
    .filter(
      (period) => period >= refinedLag * 0.65 && period <= refinedLag * 1.35,
    );
  const periodChanges = periods
    .slice(1)
    .map((period, index) => Math.abs(period - periods[index]));
  const jitter =
    periodChanges.length >= 2
      ? mean(periodChanges) / Math.max(1e-9, mean(periods))
      : null;

  return { pitch: sampleRate / refinedLag, confidence, jitter };
}

function createCoachFeedback(
  mode: SessionMode,
  metrics: VoiceMetrics,
  inputState: InputState,
) {
  if (mode === 'idle') {
    return {
      label: '準備できています',
      message: 'マイクをオンにして、練習を始める。',
      tone: 'neutral',
    } as const;
  }

  if (mode === 'requesting') {
    return {
      label: '許可待ち',
      message: 'マイクの使用を許可してください。',
      tone: 'neutral',
    } as const;
  }

  if (mode === 'calibrating') {
    return {
      label: '声を確認中',
      message: '普段どおり、話し続けてください。',
      tone: 'neutral',
    } as const;
  }

  if (mode === 'paused') {
    return {
      label: '一時停止',
      message: 'ひと息ついたら、再開。',
      tone: 'neutral',
    } as const;
  }

  if (mode === 'done') {
    return {
      label: '完了',
      message: '練習結果をまとめました。',
      tone: 'positive',
    } as const;
  }

  if (mode === 'error') {
    return {
      label: '入力確認',
      message: 'マイクを確認してください。',
      tone: 'warning',
    } as const;
  }

  if (inputState === 'silent') {
    return {
      label: '待っています',
      message: '話し始めてください。',
      tone: 'neutral',
    } as const;
  }

  if (inputState === 'loud') {
    return {
      label: '声が大きめ',
      message: 'マイクから、少し離れて。',
      tone: 'warning',
    } as const;
  }

  if (metrics.tension >= 72 && metrics.confidence >= 55) {
    return {
      label: 'ひと呼吸',
      message: '息を吐いてから、次の一文へ。',
      tone: 'warning',
    } as const;
  }

  if (metrics.pace >= 8.5) {
    return {
      label: '少し速め',
      message: '文末で、一拍置いてみて。',
      tone: 'warning',
    } as const;
  }

  if (metrics.energy >= 75 && metrics.tension < 62) {
    return {
      label: '伝わっています',
      message: '今の抑揚、そのままで。',
      tone: 'positive',
    } as const;
  }

  if (
    metrics.stability < 45 &&
    metrics.jitter >= 0.025 &&
    metrics.confidence >= 65
  ) {
    return {
      label: '声を整える',
      message: '次の一文だけ、少しゆっくり。',
      tone: 'warning',
    } as const;
  }

  if (metrics.energy < 38 && metrics.confidence >= 60) {
    return {
      label: 'もう少し前へ',
      message: '伝えたい言葉を、ひとつ強く。',
      tone: 'neutral',
    } as const;
  }

  return {
    label: 'いいペース',
    message: 'そのまま続けて。',
    tone: 'positive',
  } as const;
}

function MetricCard({
  label,
  value,
  unit,
  note,
  progress,
  color,
  track,
}: {
  label: string;
  value: string | number;
  unit: string;
  note: string;
  progress: number;
  color: string;
  track: string;
}) {
  return (
    <div className="metric-card">
      <div className="flex items-start justify-between">
        <div>
          <p className="metric-label">{label}</p>
          <p className="metric-value">
            {value}
            <span>{unit}</span>
          </p>
        </div>
        <span className={`metric-dot ${color}`} aria-hidden="true" />
      </div>
      <div className={`mt-5 h-1.5 overflow-hidden rounded-sm ${track}`}>
        <div
          className={`metric-progress h-full ${color}`}
          style={{ width: `${clamp(progress)}%` }}
        />
      </div>
      <p className="metric-note">{note}</p>
    </div>
  );
}

export default function Home() {
  const [mode, setMode] = useState<SessionMode>('idle');
  const [metrics, setMetrics] = useState<VoiceMetrics>(DEFAULT_METRICS);
  const [waveform, setWaveform] = useState<number[]>(
    Array.from({ length: 28 }, (_, index) => 24 + ((index * 19) % 52)),
  );
  const [elapsed, setElapsed] = useState(0);
  const [inputState, setInputState] = useState<InputState>('ready');
  const [errorMessage, setErrorMessage] = useState('');
  const [summary, setSummary] = useState<SessionSummary | null>(null);
  const [summaryOpen, setSummaryOpen] = useState(false);

  const modeRef = useRef<SessionMode>('idle');
  const activeRef = useRef(false);
  const metricsRef = useRef<VoiceMetrics>(DEFAULT_METRICS);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const animationRef = useRef<number | null>(null);
  const analysisStartedAtRef = useRef(0);
  const lastAnalysisAtRef = useRef(0);
  const lastVoiceAtRef = useRef(0);
  const lastPeakAtRef = useRef(0);
  const lastSampleAtRef = useRef(0);
  const pitchHistoryRef = useRef<number[]>([]);
  const pitchConfidenceRef = useRef<number[]>([]);
  const jitterHistoryRef = useRef<number[]>([]);
  const baselinePitchRef = useRef<number[]>([]);
  const baselineVolumeRef = useRef<number[]>([]);
  const baselinePaceRef = useRef<number[]>([]);
  const baselineJitterRef = useRef<number[]>([]);
  const baselineVoiceRatioRef = useRef<number[]>([]);
  const volumeHistoryRef = useRef<number[]>([]);
  const noiseHistoryRef = useRef<number[]>([]);
  const intensityFramesRef = useRef<IntensityFrame[]>([]);
  const peakTimesRef = useRef<number[]>([]);
  const voiceWindowRef = useRef<boolean[]>([]);
  const samplesRef = useRef<VoiceMetrics[]>([]);
  const transcriptRef = useRef('');
  const tensionEngineRef = useRef<TensionEngine | null>(null);

  const changeMode = useCallback((nextMode: SessionMode) => {
    modeRef.current = nextMode;
    setMode(nextMode);
  }, []);

  const cleanupEngines = useCallback(() => {
    activeRef.current = false;

    if (animationRef.current !== null) {
      window.cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {
        // The recognition session may already be stopped.
      }
      recognitionRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    analyserRef.current = null;

    if (audioContextRef.current) {
      void audioContextRef.current.close().catch(() => undefined);
      audioContextRef.current = null;
    }
  }, []);

  const prepareSession = useCallback(() => {
    setElapsed(0);
    setMetrics(DEFAULT_METRICS);
    metricsRef.current = DEFAULT_METRICS;
    setInputState('ready');
    setErrorMessage('');
    transcriptRef.current = '';
    setSummary(null);
    samplesRef.current = [];
    pitchHistoryRef.current = [];
    pitchConfidenceRef.current = [];
    jitterHistoryRef.current = [];
    baselinePitchRef.current = [];
    baselineVolumeRef.current = [];
    baselinePaceRef.current = [];
    baselineJitterRef.current = [];
    baselineVoiceRatioRef.current = [];
    volumeHistoryRef.current = [];
    noiseHistoryRef.current = [];
    intensityFramesRef.current = [];
    peakTimesRef.current = [];
    voiceWindowRef.current = [];
    lastPeakAtRef.current = 0;
    lastSampleAtRef.current = 0;
  }, []);

  const startSpeechRecognition = useCallback(() => {
    const browserWindow = window as typeof window & {
      SpeechRecognition?: SpeechRecognitionConstructor;
      webkitSpeechRecognition?: SpeechRecognitionConstructor;
    };
    const Recognition =
      browserWindow.SpeechRecognition ?? browserWindow.webkitSpeechRecognition;

    if (!Recognition) {
      return;
    }

    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'ja-JP';

    recognition.onresult = (event) => {
      let finalText = '';

      for (
        let resultIndex = event.resultIndex;
        resultIndex < event.results.length;
        resultIndex += 1
      ) {
        const result = event.results[resultIndex];
        const currentText = result[0]?.transcript ?? '';
        if (result.isFinal) finalText += currentText;
      }

      if (finalText) {
        transcriptRef.current = `${transcriptRef.current}${finalText}`;
      }
    };

    recognition.onend = () => {
      if (
        activeRef.current &&
        (modeRef.current === 'live' || modeRef.current === 'calibrating')
      ) {
        try {
          recognition.start();
        } catch {
          // Some implementations restart on their own.
        }
      }
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch {
      recognitionRef.current = null;
    }
  }, []);

  useEffect(() => {
    const loadEngine = async () => {
      try {
        const engineUrl = new URL('emotion-engine.wasm', document.baseURI);
        const response = await fetch(engineUrl);
        if (!response.ok) throw new Error('WASM engine could not be loaded.');
        const binary = await response.arrayBuffer();
        const result = await WebAssembly.instantiate(binary, {});
        const exports = result.instance.exports as unknown as TensionEngine;
        if (typeof exports.score !== 'function') {
          throw new Error('WASM score export is unavailable.');
        }
        tensionEngineRef.current = exports;
      } catch {
        tensionEngineRef.current = null;
      }
    };

    void loadEngine();
    return cleanupEngines;
  }, [cleanupEngines]);

  useEffect(() => {
    if (mode !== 'live' && mode !== 'calibrating') return;
    const timer = window.setInterval(() => {
      setElapsed((current) => current + 1);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [mode]);

  useEffect(() => {
    if (!summaryOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setSummaryOpen(false);
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [summaryOpen]);

  const recordSample = useCallback((nextMetrics: VoiceMetrics, now: number) => {
    if (now - lastSampleAtRef.current < 900) return;
    lastSampleAtRef.current = now;
    samplesRef.current.push(nextMetrics);
    if (samplesRef.current.length > 600) samplesRef.current.shift();
  }, []);

  const startMicrophone = useCallback(async () => {
    cleanupEngines();
    prepareSession();
    changeMode('requesting');

    if (!window.isSecureContext && window.location.hostname !== 'localhost') {
      setErrorMessage(
        'マイクはHTTPSのページでのみ利用できます。公開版またはlocalhostで開いてください。',
      );
      changeMode('error');
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMessage(
        'このブラウザはマイク解析に対応していません。Chrome、Edge、Safariの最新版でお試しください。',
      );
      changeMode('error');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: false,
        },
      });
      const AudioContextClass =
        window.AudioContext ??
        (
          window as typeof window & {
            webkitAudioContext?: typeof AudioContext;
          }
        ).webkitAudioContext;

      if (!AudioContextClass) {
        stream.getTracks().forEach((track) => track.stop());
        throw new Error('AudioContext is unavailable.');
      }

      const audioContext = new AudioContextClass();
      await audioContext.resume();
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.65;
      audioContext.createMediaStreamSource(stream).connect(analyser);

      streamRef.current = stream;
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      activeRef.current = true;
      analysisStartedAtRef.current = performance.now();
      lastAnalysisAtRef.current = 0;
      lastVoiceAtRef.current = performance.now();
      changeMode('calibrating');
      startSpeechRecognition();

      const audioBuffer = new Float32Array(analyser.fftSize);

      const analyze = (now: number) => {
        if (!activeRef.current) return;
        animationRef.current = window.requestAnimationFrame(analyze);

        if (modeRef.current === 'paused') return;
        if (now - lastAnalysisAtRef.current < 90) return;
        lastAnalysisAtRef.current = now;

        analyser.getFloatTimeDomainData(audioBuffer);
        let squareTotal = 0;
        let maximumAmplitude = 0;
        for (const sample of audioBuffer) {
          squareTotal += sample * sample;
          maximumAmplitude = Math.max(maximumAmplitude, Math.abs(sample));
        }

        const rms = Math.sqrt(squareTotal / audioBuffer.length);
        const decibels = 20 * Math.log10(rms + 1e-8);
        if (decibels > -90) {
          noiseHistoryRef.current.push(decibels);
          if (noiseHistoryRef.current.length > 120)
            noiseHistoryRef.current.shift();
        }
        const noiseFloor = noiseHistoryRef.current.length
          ? percentile(noiseHistoryRef.current, 0.2)
          : -60;
        const speaking = decibels > Math.max(-52, noiseFloor + 8);
        const pitchResult = detectPitch(audioBuffer, audioContext.sampleRate);
        const voiced =
          pitchResult.pitch !== null && pitchResult.confidence >= 0.7;

        voiceWindowRef.current.push(speaking);
        if (voiceWindowRef.current.length > 60) voiceWindowRef.current.shift();

        if (speaking) {
          lastVoiceAtRef.current = now;
          volumeHistoryRef.current.push(decibels);
          if (volumeHistoryRef.current.length > 30) {
            volumeHistoryRef.current.shift();
          }
        }

        intensityFramesRef.current.push({ decibels, time: now, voiced });
        if (intensityFramesRef.current.length > 7) {
          intensityFramesRef.current.shift();
        }
        const frames = intensityFramesRef.current;
        if (frames.length >= 5) {
          const candidate = frames[frames.length - 3];
          const before = Math.min(
            frames[frames.length - 5].decibels,
            frames[frames.length - 4].decibels,
          );
          const after = Math.min(
            frames[frames.length - 2].decibels,
            frames[frames.length - 1].decibels,
          );
          if (
            candidate.voiced &&
            candidate.decibels - before >= 1.5 &&
            candidate.decibels - after >= 1.5 &&
            candidate.time - lastPeakAtRef.current >= 100
          ) {
            peakTimesRef.current.push(candidate.time);
            lastPeakAtRef.current = candidate.time;
          }
        }
        peakTimesRef.current = peakTimesRef.current.filter(
          (time) => now - time < 10_000,
        );

        if (pitchResult.pitch) {
          pitchHistoryRef.current.push(pitchResult.pitch);
          pitchConfidenceRef.current.push(pitchResult.confidence);
          if (pitchResult.jitter !== null) {
            jitterHistoryRef.current.push(pitchResult.jitter);
          }
          if (pitchHistoryRef.current.length > 24) {
            pitchHistoryRef.current.shift();
          }
          if (pitchConfidenceRef.current.length > 24) {
            pitchConfidenceRef.current.shift();
          }
          if (jitterHistoryRef.current.length > 16) {
            jitterHistoryRef.current.shift();
          }
        }

        if (
          modeRef.current === 'calibrating' &&
          now - analysisStartedAtRef.current > 4_500
        ) {
          changeMode('live');
        }

        const pitchValues = pitchHistoryRef.current;
        const pitchAverage = mean(pitchValues);
        const jitter =
          jitterHistoryRef.current.length >= 3
            ? median(jitterHistoryRef.current)
            : metricsRef.current.jitter;
        const baselinePitch =
          median(baselinePitchRef.current) ||
          pitchResult.pitch ||
          metricsRef.current.pitch ||
          176;
        const semitoneShift = pitchResult.pitch
          ? 12 * Math.log2(pitchResult.pitch / baselinePitch)
          : 0;
        const pitchRange = pitchValues.length
          ? (Math.max(...pitchValues) - Math.min(...pitchValues)) /
            Math.max(1, pitchAverage)
          : 0;

        const acousticWindow = Math.max(
          4,
          Math.min(10, (now - analysisStartedAtRef.current) / 1000),
        );
        const acousticPace = clamp(
          peakTimesRef.current.length / acousticWindow,
          0,
          12,
        );
        const transcriptUnits = estimateMoraUnits(transcriptRef.current);
        const sessionSeconds = Math.max(
          1,
          (now - analysisStartedAtRef.current) / 1000,
        );
        const transcriptPace = clamp(transcriptUnits / sessionSeconds, 0, 12);
        const pace =
          transcriptUnits >= 12
            ? transcriptPace * 0.4 + acousticPace * 0.6
            : acousticPace;

        const voiceRatio =
          voiceWindowRef.current.filter(Boolean).length /
          Math.max(1, voiceWindowRef.current.length);

        const calibrating = now - analysisStartedAtRef.current <= 4_500;
        if (calibrating && speaking) {
          baselineVolumeRef.current.push(decibels);
          baselineVoiceRatioRef.current.push(voiceRatio);
          if (pitchResult.pitch)
            baselinePitchRef.current.push(pitchResult.pitch);
          if (pitchValues.length >= 4) baselineJitterRef.current.push(jitter);
          if (now - analysisStartedAtRef.current >= 2_000 && pace > 0) {
            baselinePaceRef.current.push(pace);
          }
        }

        const baselineVolume = baselineVolumeRef.current.length
          ? median(baselineVolumeRef.current)
          : -24;
        const baselinePace = baselinePaceRef.current.length
          ? median(baselinePaceRef.current)
          : 7.2;
        const baselineJitter = baselineJitterRef.current.length
          ? median(baselineJitterRef.current)
          : 0.015;
        const baselineVoiceRatio = baselineVoiceRatioRef.current.length
          ? median(baselineVoiceRatioRef.current)
          : 0.68;

        const paceStress = stressFromZ(
          robustPositiveZ(pace, baselinePaceRef.current, baselinePace, 1.2),
        );
        const baselineSemitones = baselinePitchRef.current.map(
          (pitch) => 12 * Math.log2(pitch / baselinePitch),
        );
        const pitchStress = stressFromZ(
          robustPositiveZ(semitoneShift, baselineSemitones, 0, 1.5),
        );
        const jitterStress = stressFromZ(
          robustPositiveZ(
            jitter,
            baselineJitterRef.current,
            baselineJitter,
            0.006,
          ),
        );
        const pauseStress = stressFromZ(
          robustPositiveZ(
            voiceRatio,
            baselineVoiceRatioRef.current,
            baselineVoiceRatio,
            0.12,
          ),
        );
        const volumeStress = stressFromZ(
          robustPositiveZ(
            decibels,
            baselineVolumeRef.current,
            baselineVolume,
            3,
          ),
        );

        const fallbackTension =
          paceStress * 0.15 +
          pitchStress * 0.4 +
          jitterStress * 0.05 +
          pauseStress * 0.1 +
          volumeStress * 0.3;
        const rawTension = clamp(
          tensionEngineRef.current?.score(
            paceStress,
            pitchStress,
            jitterStress,
            pauseStress,
            volumeStress,
          ) ?? fallbackTension,
        );
        const rawEnergy = clamp(
          55 + (decibels - baselineVolume) * 3 + pitchRange * 70,
          12,
          96,
        );
        const rawStability = clamp(
          96 -
            jitter * 900 -
            deviation(volumeHistoryRef.current) * 1.8 -
            (1 - mean(pitchConfidenceRef.current)) * 18,
          18,
          98,
        );
        const signalToNoise = decibels - noiseFloor;
        const signalQuality = clamp(((signalToNoise - 6) / 24) * 100);
        const pitchCoverage = clamp((pitchValues.length / 16) * 100);
        const rawConfidence = speaking
          ? clamp(
              pitchResult.confidence * 55 +
                signalQuality * 0.3 +
                pitchCoverage * 0.15,
            )
          : 18;

        const previous = metricsRef.current;
        const smooth = (current: number, next: number, weight = 0.22) =>
          current * (1 - weight) + next * weight;
        const nextMetrics: VoiceMetrics = {
          tension: speaking
            ? smooth(previous.tension, rawTension, 0.18)
            : previous.tension,
          energy: speaking
            ? smooth(previous.energy, rawEnergy, 0.2)
            : smooth(previous.energy, 22, 0.08),
          pace: smooth(previous.pace, pace, 0.18),
          stability: speaking
            ? smooth(previous.stability, rawStability, 0.16)
            : previous.stability,
          confidence: smooth(previous.confidence, rawConfidence, 0.2),
          pitch: pitchResult.pitch
            ? smooth(
                previous.pitch ?? pitchResult.pitch,
                pitchResult.pitch,
                0.22,
              )
            : previous.pitch,
          jitter: smooth(previous.jitter, jitter, 0.2),
          volume: smooth(previous.volume, decibels, 0.25),
        };

        const silenceDuration = now - lastVoiceAtRef.current;
        setInputState(
          maximumAmplitude > 0.965
            ? 'loud'
            : silenceDuration > 2_700
              ? 'silent'
              : 'good',
        );
        metricsRef.current = nextMetrics;
        setMetrics(nextMetrics);
        recordSample(nextMetrics, now);

        const points = 28;
        const step = Math.floor(audioBuffer.length / points);
        setWaveform(
          Array.from({ length: points }, (_, index) => {
            const amplitude = Math.abs(audioBuffer[index * step] ?? 0);
            return clamp(16 + amplitude * 510, 12, 96);
          }),
        );
      };

      animationRef.current = window.requestAnimationFrame(analyze);
    } catch (error) {
      cleanupEngines();
      const errorName = error instanceof DOMException ? error.name : '';
      const message =
        errorName === 'NotAllowedError'
          ? 'マイクが許可されていません。ブラウザのサイト設定でマイクを許可し、もう一度お試しください。'
          : errorName === 'NotFoundError'
            ? '利用できるマイクが見つかりません。マイクの接続とOSの設定を確認してください。'
            : 'マイクを開始できませんでした。ブラウザの設定を確認して、もう一度お試しください。';
      setErrorMessage(message);
      changeMode('error');
    }
  }, [
    changeMode,
    cleanupEngines,
    prepareSession,
    recordSample,
    startSpeechRecognition,
  ]);

  const pauseSession = () => {
    changeMode('paused');
    if (audioContextRef.current) {
      void audioContextRef.current.suspend().catch(() => undefined);
    }
    try {
      recognitionRef.current?.stop();
    } catch {
      // Recognition is already paused.
    }
  };

  const resumeSession = () => {
    changeMode('live');
    if (audioContextRef.current) {
      void audioContextRef.current.resume().catch(() => undefined);
    }
    if (recognitionRef.current) {
      try {
        recognitionRef.current.start();
      } catch {
        // Recognition may still be active.
      }
    }
  };

  const finishSession = () => {
    const recorded = samplesRef.current.length
      ? samplesRef.current
      : [metricsRef.current];
    const averageOf = (key: keyof VoiceMetrics) =>
      mean(
        recorded
          .map((sample) => sample[key])
          .filter((value): value is number => typeof value === 'number'),
      );
    const nextSummary: SessionSummary = {
      tension: round(averageOf('tension')),
      energy: round(averageOf('energy')),
      pace: roundOne(averageOf('pace')),
      stability: round(averageOf('stability')),
      duration: elapsed,
    };
    cleanupEngines();
    setSummary(nextSummary);
    changeMode('done');
    setSummaryOpen(true);
  };

  const resetSession = () => {
    cleanupEngines();
    prepareSession();
    setSummaryOpen(false);
    changeMode('idle');
  };

  const coachCandidate = useMemo(
    () => createCoachFeedback(mode, metrics, inputState),
    [inputState, metrics, mode],
  );
  const [coach, setCoach] = useState(coachCandidate);

  useEffect(() => {
    const shouldUpdateImmediately = mode !== 'live' || inputState !== 'good';
    if (shouldUpdateImmediately) {
      setCoach(coachCandidate);
      return;
    }

    const timer = window.setTimeout(() => setCoach(coachCandidate), 700);
    return () => window.clearTimeout(timer);
  }, [
    coachCandidate.label,
    coachCandidate.message,
    coachCandidate.tone,
    inputState,
    mode,
  ]);
  const status = {
    idle: { text: '準備', className: 'status-ready' },
    requesting: { text: '許可待ち', className: 'status-waiting' },
    calibrating: { text: '測定中', className: 'status-waiting' },
    live: { text: 'LIVE', className: 'status-live' },
    paused: { text: '一時停止', className: 'status-paused' },
    done: { text: '完了', className: 'status-ready' },
    error: { text: '要確認', className: 'status-error' },
  }[mode];
  const hasMeasurement =
    mode === 'calibrating' ||
    mode === 'live' ||
    mode === 'paused' ||
    mode === 'done';

  const tensionNote =
    metrics.tension < 36 ? '低い' : metrics.tension < 66 ? 'やや高い' : '高い';
  const energyNote =
    metrics.energy >= 72 ? '高い' : metrics.energy >= 45 ? '標準' : '低い';
  const paceNote =
    metrics.pace > 8 ? '速め' : metrics.pace < 6 ? 'ゆっくり' : '標準';

  return (
    <main className="min-h-screen overflow-hidden bg-background text-foreground">
      <div
        id="rehearsal"
        className="mx-auto max-w-[1260px] px-5 py-6 lg:px-8 lg:py-8"
      >
        <section className="mb-4">
          <h1 className="text-xl font-bold tracking-[-0.03em] sm:text-2xl">
            プレゼン練習
          </h1>
        </section>

        <section className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_330px]">
          <div
            className={`coach-stage relative flex min-h-[560px] flex-col overflow-hidden rounded-[16px] border border-[#2d3137] p-5 text-white sm:p-7 ${mode === 'live' ? 'is-live' : ''}`}
          >
            <div className="relative z-10 flex items-center justify-between">
              <h2 className="text-lg font-semibold">コーチ</h2>
              <div className="flex items-center gap-2">
                {(mode === 'live' ||
                  mode === 'calibrating' ||
                  mode === 'paused') && (
                  <span className="hidden items-center gap-1.5 font-mono text-xs text-white/55 sm:flex">
                    <Clock3 className="size-3.5" aria-hidden="true" />
                    {formatTime(elapsed)}
                  </span>
                )}
                <Badge
                  className={`session-status h-7 gap-1.5 rounded-md px-2.5 ${status.className}`}
                >
                  <span className="status-light size-1.5 rounded-full" />
                  {status.text}
                </Badge>
              </div>
            </div>

            <output
              aria-live="polite"
              aria-atomic="true"
              className={`coach-panel relative z-10 mt-6 flex flex-1 flex-col justify-center rounded-[10px] border p-5 sm:p-8 coach-${coach.tone}`}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="coach-kicker flex items-center gap-2">
                  <Activity className="size-4" aria-hidden="true" />
                  <p className="text-xs font-bold">{coach.label}</p>
                </div>
                {mode === 'live' && (
                  <span className="text-[10px] font-medium text-white/40">
                    解析品質 {round(metrics.confidence)}%
                  </span>
                )}
              </div>
              <p className="mt-2 min-h-14 text-base font-medium leading-7 sm:text-lg">
                {coach.message}
              </p>
              <div
                className="mt-7 flex h-24 items-center gap-[4px] overflow-hidden"
                aria-hidden="true"
              >
                {waveform.map((height, index) => (
                  <span
                    key={index}
                    className="wave-bar flex-1 rounded-sm"
                    style={{
                      height: `${height}%`,
                    }}
                  />
                ))}
              </div>
            </output>

            {errorMessage && (
              <div className="relative z-10 mt-4 flex gap-3 rounded-lg border border-[#dc5a4f]/40 bg-[#dc5a4f]/10 p-4 text-sm leading-6 text-white/80">
                <CircleAlert
                  className="mt-0.5 size-4 shrink-0 text-[#ff9d93]"
                  aria-hidden="true"
                />
                <p>{errorMessage}</p>
              </div>
            )}

            <div className="relative z-10 mt-5 flex flex-col gap-3 sm:flex-row sm:items-center">
              {(mode === 'idle' || mode === 'error' || mode === 'done') && (
                <Button
                  size="lg"
                  onClick={startMicrophone}
                  className="h-11 rounded-lg bg-[#2869d8] px-5 text-sm font-bold text-white hover:bg-[#1f5cbe]"
                >
                  <Mic className="size-4" aria-hidden="true" />
                  {mode === 'error'
                    ? 'マイクを再試行'
                    : 'マイクをオンにして練習'}
                </Button>
              )}

              {mode === 'requesting' && (
                <Button
                  size="lg"
                  disabled
                  className="h-11 rounded-lg bg-white/10 px-5 text-white"
                >
                  <Mic className="size-4" aria-hidden="true" />
                  ブラウザでマイクを許可してください
                </Button>
              )}

              {(mode === 'live' || mode === 'calibrating') && (
                <>
                  <Button
                    size="lg"
                    variant="outline"
                    onClick={pauseSession}
                    className="h-11 rounded-lg border-white/15 bg-white/8 px-5 text-white hover:bg-white/14 hover:text-white"
                  >
                    <Pause className="size-4" aria-hidden="true" />
                    一時停止
                  </Button>
                  <Button
                    size="lg"
                    variant="ghost"
                    onClick={finishSession}
                    className="h-11 rounded-lg px-5 text-white/70 hover:bg-white/10 hover:text-white"
                  >
                    <Square
                      className="size-3.5 fill-current"
                      aria-hidden="true"
                    />
                    終了
                  </Button>
                </>
              )}

              {mode === 'paused' && (
                <>
                  <Button
                    size="lg"
                    onClick={resumeSession}
                    className="h-11 rounded-lg bg-[#7dd3a5] px-5 font-bold text-[#171a1f] hover:bg-[#6bc292]"
                  >
                    <Play className="size-4 fill-current" aria-hidden="true" />
                    再開
                  </Button>
                  <Button
                    size="lg"
                    variant="ghost"
                    onClick={finishSession}
                    className="h-11 rounded-lg px-5 text-white/70 hover:bg-white/10 hover:text-white"
                  >
                    <Square
                      className="size-3.5 fill-current"
                      aria-hidden="true"
                    />
                    終了
                  </Button>
                </>
              )}
            </div>
          </div>

          <aside
            className="grid content-start gap-4 sm:grid-cols-3 lg:grid-cols-1"
            aria-label="現在の声の指標"
          >
            <MetricCard
              label="緊張サイン"
              value={hasMeasurement ? round(metrics.tension) : '—'}
              unit={hasMeasurement ? '/100' : ''}
              note={hasMeasurement ? tensionNote : '待機'}
              progress={hasMeasurement ? metrics.tension : 0}
              color="bg-[#dc5a4f]"
              track="bg-[#eceef1]"
            />
            <MetricCard
              label="声の熱量"
              value={hasMeasurement ? round(metrics.energy) : '—'}
              unit={hasMeasurement ? '/100' : ''}
              note={hasMeasurement ? energyNote : '待機'}
              progress={hasMeasurement ? metrics.energy : 0}
              color="bg-[#2869d8]"
              track="bg-[#eceef1]"
            />
            <MetricCard
              label="話すテンポ"
              value={hasMeasurement ? metrics.pace.toFixed(1) : '—'}
              unit={hasMeasurement ? 'モーラ/秒・推定' : ''}
              note={hasMeasurement ? paceNote : '待機'}
              progress={hasMeasurement ? (metrics.pace / 10) * 100 : 0}
              color="bg-[#3c8b61]"
              track="bg-[#eceef1]"
            />

            <div className="diagnostic-card sm:col-span-3 lg:col-span-1">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity
                    className="size-4 text-[#2869d8]"
                    aria-hidden="true"
                  />
                  <p className="text-xs font-bold">詳細</p>
                </div>
                <span className="text-[10px] font-semibold text-muted-foreground">
                  {mode === 'live' || mode === 'calibrating'
                    ? '計測中'
                    : mode === 'paused'
                      ? '停止中'
                      : mode === 'done'
                        ? '完了'
                        : '待機'}
                </span>
              </div>
              <dl className="grid grid-cols-3 gap-2">
                <div>
                  <dt>F0</dt>
                  <dd>
                    {hasMeasurement && metrics.pitch
                      ? `${round(metrics.pitch)} Hz`
                      : '—'}
                  </dd>
                </div>
                <div>
                  <dt>揺らぎ</dt>
                  <dd>
                    {hasMeasurement
                      ? `${(metrics.jitter * 100).toFixed(1)}%`
                      : '—'}
                  </dd>
                </div>
                <div>
                  <dt>安定度</dt>
                  <dd>{hasMeasurement ? round(metrics.stability) : '—'}</dd>
                </div>
              </dl>
            </div>
          </aside>
        </section>
      </div>

      {summaryOpen && (
        <div className="summary-backdrop">
          <dialog
            open
            aria-modal="true"
            aria-labelledby="summary-title"
            aria-describedby="summary-description"
            onCancel={(event) => {
              event.preventDefault();
              setSummaryOpen(false);
            }}
            className="summary-dialog max-h-[calc(100vh-2rem)] w-[min(620px,calc(100%-2rem))] overflow-y-auto rounded-[12px] p-0"
          >
            <div className="summary-hero">
              <div className="summary-check">
                <Check className="size-5" strokeWidth={3} aria-hidden="true" />
              </div>
              <div className="flex flex-col gap-2">
                <h2
                  id="summary-title"
                  className="text-2xl font-bold tracking-[-0.04em]"
                >
                  練習結果
                </h2>
                <p
                  id="summary-description"
                  className="text-sm leading-6 text-muted-foreground"
                >
                  声の傾向をまとめました。
                </p>
              </div>
            </div>

            {summary && (
              <div className="px-5 pb-1 sm:px-7">
                <div className="summary-grid">
                  <div>
                    <span>平均緊張度</span>
                    <strong>{summary.tension}</strong>
                    <small>/100</small>
                  </div>
                  <div>
                    <span>声の熱量</span>
                    <strong>{summary.energy}</strong>
                    <small>/100</small>
                  </div>
                  <div>
                    <span>平均テンポ</span>
                    <strong>{summary.pace}</strong>
                    <small>モーラ/秒</small>
                  </div>
                  <div>
                    <span>練習時間</span>
                    <strong>{formatTime(summary.duration)}</strong>
                    <small>min</small>
                  </div>
                </div>

                <div className="mt-5 grid gap-3 sm:grid-cols-2">
                  <div className="summary-tip summary-good">
                    <div>
                      <Volume2 aria-hidden="true" />
                      <span>よかった点</span>
                    </div>
                    <p>
                      {summary.energy >= 65
                        ? '声の熱量が保たれ、聞き手を惹きつける時間がつくれていました。'
                        : '落ち着いた声量で、丁寧に伝える土台ができています。'}
                    </p>
                  </div>
                  <div className="summary-tip summary-next">
                    <div>
                      <Gauge aria-hidden="true" />
                      <span>次のフォーカス</span>
                    </div>
                    <p>
                      {summary.pace > 8
                        ? '結論の直前に1秒の間を入れ、聞き手が追いつく余白をつくりましょう。'
                        : summary.tension > 65
                          ? '話し始める前に長く息を吐き、最初の一文をゆっくり届けましょう。'
                          : '強調したい言葉の前後に間を入れると、さらに印象が残ります。'}
                    </p>
                  </div>
                </div>

                <div className="mt-4 rounded-lg border border-border p-3 text-[11px] leading-5 text-muted-foreground">
                  録音データは保存していません。
                </div>
              </div>
            )}

            <div className="mt-3 flex flex-col-reverse gap-2 rounded-b-[12px] border-t bg-muted/50 px-5 py-4 sm:flex-row sm:justify-end sm:px-7">
              <Button
                variant="outline"
                onClick={() => setSummaryOpen(false)}
                className="rounded-lg"
              >
                画面に戻る
              </Button>
              <Button
                onClick={resetSession}
                className="rounded-lg bg-[#171a1f] px-5 text-white hover:bg-[#2d3137]"
              >
                <RotateCcw className="size-4" aria-hidden="true" />
                もう一度練習
              </Button>
            </div>
          </dialog>
        </div>
      )}
    </main>
  );
}
