'use client';

import {
  Activity,
  AudioWaveform,
  Captions,
  Check,
  CircleAlert,
  Clock3,
  Cpu,
  Gauge,
  Headphones,
  LockKeyhole,
  Mic,
  Pause,
  Play,
  Radio,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Square,
  Volume2,
  Waves,
} from 'lucide-react';
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

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

type SessionKind = 'microphone' | 'demo' | null;
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

const DEFAULT_METRICS: VoiceMetrics = {
  tension: 32,
  energy: 68,
  pace: 142,
  stability: 82,
  confidence: 94,
  pitch: 176,
  jitter: 0.018,
  volume: -24,
};

const DEMO_SCRIPT = [
  'みなさん、今日は聞き手に届く話し方についてお話しします。',
  '伝えるときに大切なのは、情報の量だけではありません。',
  '声のテンポと間が、聞き手の理解を大きく変えます。',
  'ここで一度、ゆっくり深呼吸してみましょう。',
  '自分らしい声で話すほど、メッセージはまっすぐ届きます。',
];

const clamp = (value: number, minimum = 0, maximum = 100) =>
  Math.min(maximum, Math.max(minimum, value));

const round = (value: number) => Math.round(value);

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

const deviation = (values: number[]) => {
  if (values.length < 2) return 0;
  const average = mean(values);
  return Math.sqrt(
    mean(values.map((value) => Math.pow(value - average, 2))),
  );
};

const formatTime = (seconds: number) => {
  const minutes = Math.floor(seconds / 60)
    .toString()
    .padStart(2, '0');
  const rest = Math.floor(seconds % 60)
    .toString()
    .padStart(2, '0');
  return `${minutes}:${rest}`;
};

function estimateWordUnits(text: string) {
  const clean = text.trim();
  if (!clean) return 0;

  try {
    const segmenter = new Intl.Segmenter('ja', { granularity: 'word' });
    const segments = Array.from(segmenter.segment(clean));
    const wordCount = segments.filter((segment) => segment.isWordLike).length;
    if (wordCount) return wordCount;
  } catch {
    // Older browsers fall back to a Japanese-character approximation.
  }

  return Math.max(1, clean.replace(/\s/g, '').length / 2.4);
}

function detectPitch(buffer: Float32Array, sampleRate: number) {
  let squareTotal = 0;
  for (const sample of buffer) squareTotal += sample * sample;
  const rms = Math.sqrt(squareTotal / buffer.length);
  if (rms < 0.012) return { pitch: null as number | null, confidence: 0 };

  const minimumLag = Math.floor(sampleRate / 400);
  const maximumLag = Math.min(
    Math.floor(sampleRate / 70),
    buffer.length - 2,
  );
  let bestLag = 0;
  let bestCorrelation = 0;

  for (let lag = minimumLag; lag <= maximumLag; lag += 1) {
    let correlation = 0;
    let normA = 0;
    let normB = 0;
    const comparisonLength = buffer.length - lag;

    for (let index = 0; index < comparisonLength; index += 1) {
      const current = buffer[index];
      const shifted = buffer[index + lag];
      correlation += current * shifted;
      normA += current * current;
      normB += shifted * shifted;
    }

    const normalized = correlation / Math.sqrt(normA * normB + 1e-9);
    if (normalized > bestCorrelation) {
      bestCorrelation = normalized;
      bestLag = lag;
    }
  }

  if (!bestLag || bestCorrelation < 0.58) {
    return { pitch: null as number | null, confidence: bestCorrelation };
  }

  return {
    pitch: sampleRate / bestLag,
    confidence: bestCorrelation,
  };
}

function createCoachFeedback(
  mode: SessionMode,
  metrics: VoiceMetrics,
  inputState: InputState,
) {
  if (mode === 'idle') {
    return {
      label: 'ライブ・コーチ',
      message:
        'マイクをオンにすると、声の変化に合わせてその場でアドバイスします。',
      tone: 'neutral',
    } as const;
  }

  if (mode === 'requesting') {
    return {
      label: 'マイクの許可待ち',
      message: 'ブラウザの案内で「許可」を選ぶと、練習が始まります。',
      tone: 'neutral',
    } as const;
  }

  if (mode === 'calibrating') {
    return {
      label: 'あなたの声を調整中',
      message: 'いつもの声で話してください。約3秒で基準をつくります。',
      tone: 'neutral',
    } as const;
  }

  if (mode === 'paused') {
    return {
      label: '一時停止中',
      message: 'ここまでのデータは残っています。準備ができたら再開しましょう。',
      tone: 'neutral',
    } as const;
  }

  if (mode === 'done') {
    return {
      label: 'セッション完了',
      message: 'おつかれさまでした。今回の声の傾向をレポートにまとめました。',
      tone: 'positive',
    } as const;
  }

  if (mode === 'error') {
    return {
      label: 'マイクを確認してください',
      message: '下の案内を確認するか、デモモードで体験できます。',
      tone: 'warning',
    } as const;
  }

  if (inputState === 'silent') {
    return {
      label: '声を待っています',
      message: '声を検出できません。マイクを確認し、少し近づいて話してください。',
      tone: 'warning',
    } as const;
  }

  if (inputState === 'loud') {
    return {
      label: '入力が大きめです',
      message: '音が割れそうです。マイクから少し離れると聞きやすくなります。',
      tone: 'warning',
    } as const;
  }

  if (metrics.tension >= 69) {
    return {
      label: '緊張サインが少し上昇',
      message: '声の高さとテンポが上がっています。次の句点で、ゆっくり息を吐いて。',
      tone: 'warning',
    } as const;
  }

  if (metrics.pace >= 178) {
    return {
      label: 'テンポが速くなっています',
      message: '大切な言葉の前に1秒の間を。聞き手が追いつきやすくなります。',
      tone: 'warning',
    } as const;
  }

  if (metrics.energy >= 75 && metrics.tension < 62) {
    return {
      label: '今の熱量、届いています',
      message: '声の抑揚が聞き手を惹きつけています。そのまま結論まで。',
      tone: 'positive',
    } as const;
  }

  if (metrics.stability < 54) {
    return {
      label: '声に細かな揺れがあります',
      message: '語尾まで息を流すイメージで、ひとつずつ丁寧に届けましょう。',
      tone: 'warning',
    } as const;
  }

  return {
    label: 'いいバランスです',
    message: '落ち着いたテンポです。聞き手を見ながら、このまま続けて。',
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
  surface,
}: {
  label: string;
  value: string | number;
  unit: string;
  note: string;
  progress: number;
  color: string;
  track: string;
  surface: string;
}) {
  return (
    <div className={`metric-card ${surface}`}>
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
      <div className={`mt-5 h-2 overflow-hidden rounded-full ${track}`}>
        <div
          className={`metric-progress h-full rounded-full ${color}`}
          style={{ width: `${clamp(progress)}%` }}
        />
      </div>
      <p className="metric-note">{note}</p>
    </div>
  );
}

export default function Home() {
  const [mode, setMode] = useState<SessionMode>('idle');
  const [sessionKind, setSessionKind] = useState<SessionKind>(null);
  const [metrics, setMetrics] =
    useState<VoiceMetrics>(DEFAULT_METRICS);
  const [waveform, setWaveform] = useState<number[]>(
    Array.from({ length: 28 }, (_, index) =>
      24 + ((index * 19) % 52),
    ),
  );
  const [elapsed, setElapsed] = useState(0);
  const [inputState, setInputState] = useState<InputState>('ready');
  const [errorMessage, setErrorMessage] = useState('');
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [speechSupported, setSpeechSupported] = useState(false);
  const [engineReady, setEngineReady] = useState(false);
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
  const demoTimerRef = useRef<number | null>(null);
  const analysisStartedAtRef = useRef(0);
  const lastAnalysisAtRef = useRef(0);
  const lastVoiceAtRef = useRef(0);
  const lastPeakAtRef = useRef(0);
  const lastRmsRef = useRef(0);
  const lastSampleAtRef = useRef(0);
  const pitchHistoryRef = useRef<number[]>([]);
  const baselinePitchRef = useRef<number[]>([]);
  const volumeHistoryRef = useRef<number[]>([]);
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

    if (demoTimerRef.current !== null) {
      window.clearInterval(demoTimerRef.current);
      demoTimerRef.current = null;
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
    setTranscript('');
    setInterimTranscript('');
    transcriptRef.current = '';
    setSummary(null);
    samplesRef.current = [];
    pitchHistoryRef.current = [];
    baselinePitchRef.current = [];
    volumeHistoryRef.current = [];
    peakTimesRef.current = [];
    voiceWindowRef.current = [];
    lastRmsRef.current = 0;
    lastPeakAtRef.current = 0;
    lastSampleAtRef.current = 0;
  }, []);

  const startSpeechRecognition = useCallback(() => {
    const browserWindow = window as typeof window & {
      SpeechRecognition?: SpeechRecognitionConstructor;
      webkitSpeechRecognition?: SpeechRecognitionConstructor;
    };
    const Recognition =
      browserWindow.SpeechRecognition ??
      browserWindow.webkitSpeechRecognition;

    if (!Recognition) {
      setSpeechSupported(false);
      return;
    }

    setSpeechSupported(true);
    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'ja-JP';

    recognition.onresult = (event) => {
      let finalText = '';
      let temporaryText = '';

      for (
        let resultIndex = event.resultIndex;
        resultIndex < event.results.length;
        resultIndex += 1
      ) {
        const result = event.results[resultIndex];
        const currentText = result[0]?.transcript ?? '';
        if (result.isFinal) finalText += currentText;
        else temporaryText += currentText;
      }

      if (finalText) {
        setTranscript((current) => {
          const next = `${current}${finalText}`;
          transcriptRef.current = next;
          return next;
        });
      }
      setInterimTranscript(temporaryText);
    };

    recognition.onerror = (event) => {
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        setSpeechSupported(false);
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
      setSpeechSupported(false);
    }
  }, []);

  useEffect(() => {
    const browserWindow = window as typeof window & {
      SpeechRecognition?: SpeechRecognitionConstructor;
      webkitSpeechRecognition?: SpeechRecognitionConstructor;
    };
    setSpeechSupported(
      Boolean(
        browserWindow.SpeechRecognition ??
          browserWindow.webkitSpeechRecognition,
      ),
    );

    const loadEngine = async () => {
      try {
        const engineUrl = new URL('emotion-engine.wasm', document.baseURI);
        const response = await fetch(engineUrl);
        if (!response.ok) throw new Error('WASM engine could not be loaded.');
        const binary = await response.arrayBuffer();
        const result = await WebAssembly.instantiate(binary);
        const exports = result.instance.exports as unknown as TensionEngine;
        if (typeof exports.score !== 'function') {
          throw new Error('WASM score export is unavailable.');
        }
        tensionEngineRef.current = exports;
        setEngineReady(true);
      } catch {
        tensionEngineRef.current = null;
        setEngineReady(false);
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
    setSessionKind('microphone');
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
        const speaking = rms > 0.011;

        voiceWindowRef.current.push(speaking);
        if (voiceWindowRef.current.length > 60) voiceWindowRef.current.shift();

        if (speaking) {
          lastVoiceAtRef.current = now;
          volumeHistoryRef.current.push(decibels);
          if (volumeHistoryRef.current.length > 30) {
            volumeHistoryRef.current.shift();
          }
        }

        if (
          speaking &&
          rms > 0.015 &&
          rms > lastRmsRef.current * 1.08 &&
          now - lastPeakAtRef.current > 150
        ) {
          peakTimesRef.current.push(now);
          lastPeakAtRef.current = now;
        }
        lastRmsRef.current = rms;
        peakTimesRef.current = peakTimesRef.current.filter(
          (time) => now - time < 10_000,
        );

        const pitchResult = detectPitch(audioBuffer, audioContext.sampleRate);
        if (pitchResult.pitch) {
          pitchHistoryRef.current.push(pitchResult.pitch);
          if (pitchHistoryRef.current.length > 24) {
            pitchHistoryRef.current.shift();
          }

          if (now - analysisStartedAtRef.current < 5_000) {
            baselinePitchRef.current.push(pitchResult.pitch);
            if (baselinePitchRef.current.length > 50) {
              baselinePitchRef.current.shift();
            }
          }
        }

        if (
          modeRef.current === 'calibrating' &&
          now - analysisStartedAtRef.current > 3_200
        ) {
          changeMode('live');
        }

        const pitchValues = pitchHistoryRef.current;
        const pitchAverage = mean(pitchValues);
        const jitter = pitchAverage
          ? deviation(pitchValues) / pitchAverage
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
          (peakTimesRef.current.length / acousticWindow) * (60 / 2.1),
          72,
          220,
        );
        const transcriptUnits = estimateWordUnits(transcriptRef.current);
        const sessionMinutes = Math.max(
          0.08,
          (now - analysisStartedAtRef.current) / 60_000,
        );
        const transcriptPace = clamp(
          transcriptUnits / sessionMinutes,
          65,
          220,
        );
        const pace =
          transcriptUnits >= 4
            ? transcriptPace * 0.68 + acousticPace * 0.32
            : acousticPace;

        const voiceRatio =
          voiceWindowRef.current.filter(Boolean).length /
          Math.max(1, voiceWindowRef.current.length);
        const paceStress = clamp(34 + (pace - 125) * 0.8);
        const pitchStress = clamp(
          34 + Math.max(0, semitoneShift) * 13 + jitter * 260,
        );
        const jitterStress = clamp(jitter * 720);
        const pauseStress = clamp(28 + Math.max(0, voiceRatio - 0.68) * 160);
        const volumeStress = clamp(
          decibels > -13
            ? 58 + (decibels + 13) * 5
            : decibels < -39
              ? 48 + (-39 - decibels) * 2
              : 28,
        );

        const fallbackTension =
          paceStress * 0.3 +
          pitchStress * 0.25 +
          jitterStress * 0.2 +
          pauseStress * 0.15 +
          volumeStress * 0.1;
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
          45 + (decibels + 30) * 1.65 + pitchRange * 80,
          12,
          96,
        );
        const rawStability = clamp(
          95 - jitter * 540 - deviation(volumeHistoryRef.current) * 1.7,
          18,
          98,
        );
        const rawConfidence = speaking
          ? clamp(52 + pitchResult.confidence * 47)
          : metricsRef.current.confidence;

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
            ? smooth(previous.pitch ?? pitchResult.pitch, pitchResult.pitch, 0.22)
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
            : 'マイクを開始できませんでした。別のブラウザで試すか、デモモードをご利用ください。';
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

  const startDemo = useCallback(() => {
    cleanupEngines();
    prepareSession();
    setSessionKind('demo');
    activeRef.current = true;
    analysisStartedAtRef.current = performance.now();
    changeMode('live');
    setInputState('good');

    demoTimerRef.current = window.setInterval(() => {
      if (modeRef.current !== 'live') return;
      const now = performance.now();
      const time = (now - analysisStartedAtRef.current) / 1000;
      const tension = clamp(
        45 + Math.sin(time / 3.8) * 18 + Math.sin(time / 1.7) * 7,
        20,
        78,
      );
      const energy = clamp(
        66 + Math.sin(time / 2.6) * 16 + Math.cos(time / 5.2) * 8,
        34,
        92,
      );
      const pace = clamp(
        145 + Math.sin(time / 4.5) * 32 + Math.cos(time / 2.2) * 14,
        92,
        198,
      );
      const stability = clamp(84 - Math.sin(time / 3.2) * 13, 58, 96);
      const nextMetrics: VoiceMetrics = {
        tension,
        energy,
        pace,
        stability,
        confidence: 96,
        pitch: 178 + Math.sin(time / 1.8) * 24,
        jitter: 0.017 + Math.abs(Math.sin(time / 2.7)) * 0.022,
        volume: -24 + Math.sin(time / 1.4) * 6,
      };

      metricsRef.current = nextMetrics;
      setMetrics(nextMetrics);
      recordSample(nextMetrics, now);
      setWaveform(
        Array.from({ length: 28 }, (_, index) =>
          clamp(
            24 +
              Math.abs(
                Math.sin(time * 4.4 + index * 0.55) *
                  (38 + Math.sin(time + index) * 19),
              ),
            14,
            96,
          ),
        ),
      );

      const visibleCount = Math.min(
        DEMO_SCRIPT.length,
        Math.floor(time / 4.5) + 1,
      );
      const nextTranscript = DEMO_SCRIPT.slice(0, visibleCount).join('');
      transcriptRef.current = nextTranscript;
      setTranscript(nextTranscript);
      setInterimTranscript(DEMO_SCRIPT[visibleCount] ?? '');
    }, 120);
  }, [changeMode, cleanupEngines, prepareSession, recordSample]);

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
    if (sessionKind === 'microphone' && recognitionRef.current) {
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
      pace: round(averageOf('pace')),
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
    setSessionKind(null);
    setSummaryOpen(false);
    changeMode('idle');
  };

  const coach = useMemo(
    () => createCoachFeedback(mode, metrics, inputState),
    [inputState, metrics, mode],
  );
  const status = {
    idle: { text: '準備OK', className: 'status-ready' },
    requesting: { text: 'マイク許可待ち', className: 'status-waiting' },
    calibrating: { text: '声を調整中', className: 'status-waiting' },
    live: { text: 'LIVE', className: 'status-live' },
    paused: { text: '一時停止', className: 'status-paused' },
    done: { text: '完了', className: 'status-ready' },
    error: { text: '要確認', className: 'status-error' },
  }[mode];

  const tensionNote =
    metrics.tension < 36
      ? '落ち着いています'
      : metrics.tension < 66
        ? '少し負荷が上がっています'
        : '深呼吸のタイミングです';
  const energyNote =
    metrics.energy >= 72
      ? '聞き手を惹きつけています'
      : metrics.energy >= 45
        ? '自然な熱量です'
        : '少し抑揚を加えてみましょう';
  const paceNote =
    metrics.pace > 178
      ? '少し速くなっています'
      : metrics.pace < 95
        ? 'ゆっくり丁寧なテンポです'
        : 'ちょうどよい速さです';

  return (
    <main className="min-h-screen overflow-hidden bg-background text-foreground">
      <header className="sticky top-0 z-40 border-b border-border/80 bg-background/88 backdrop-blur-xl">
        <div className="mx-auto flex h-[72px] max-w-[1440px] items-center justify-between px-5 lg:px-8">
          <a href="#rehearsal" className="flex items-center gap-3" aria-label="HearEmpathy ホーム">
            <span className="logo-mark">
              <AudioWaveform className="size-5" aria-hidden="true" />
            </span>
            <div>
              <p className="text-[17px] font-bold tracking-[-0.03em]">HearEmpathy</p>
              <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                Voice rehearsal studio
              </p>
            </div>
          </a>

          <div className="flex items-center gap-2.5">
            <Badge
              variant="outline"
              className="hidden h-8 gap-1.5 rounded-full bg-white/70 px-3 text-[11px] text-muted-foreground sm:flex"
            >
              <LockKeyhole className="size-3" aria-hidden="true" />
              声の特徴は端末内で解析
            </Badge>
            <Badge
              variant="outline"
              className="hidden h-8 gap-1.5 rounded-full bg-white/70 px-3 text-[11px] text-muted-foreground md:flex"
            >
              <Cpu className="size-3" aria-hidden="true" />
              {engineReady ? 'WASM engine ready' : 'Local engine'}
            </Badge>
            <span
              className="grid size-9 place-items-center rounded-full border border-border bg-white text-xs font-bold"
              aria-label="プロフィール"
            >
              HE
            </span>
          </div>
        </div>
      </header>

      <div
        id="rehearsal"
        className="mx-auto max-w-[1440px] scroll-mt-20 px-5 py-7 lg:px-8 lg:py-9"
      >
        <section className="mb-6 flex flex-col justify-between gap-4 md:flex-row md:items-end">
          <div>
            <div className="mb-2 flex items-center gap-2 text-xs font-bold text-[#6d5bd0]">
              <Radio className="size-3.5" aria-hidden="true" />
              リハーサル・ルーム
              {sessionKind === 'demo' && (
                <Badge className="ml-1 rounded-full bg-[#e8e3ff] text-[#5c49c4]">
                  デモ
                </Badge>
              )}
            </div>
            <h1 className="text-2xl font-bold tracking-[-0.04em] sm:text-[32px]">
              聞き手の気持ちを、話している今に。
            </h1>
          </div>
          <p className="max-w-md text-sm leading-6 text-muted-foreground">
            声の揺らぎ・テンポ・熱量をリアルタイム解析。必要なアドバイスだけを見ながら、自分らしい伝え方を整えます。
          </p>
        </section>

        <section className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_330px]">
          <div
            className={`coach-stage relative flex min-h-[590px] flex-col overflow-hidden rounded-[30px] p-5 text-white shadow-[0_28px_70px_rgba(34,30,71,.16)] sm:p-7 ${mode === 'live' ? 'is-live' : ''}`}
          >
            <div className="relative z-10 flex items-center justify-between">
              <div>
                <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-white/45">
                  Real-time feedback
                </p>
                <h2 className="mt-1 text-lg font-semibold">ライブ・コーチ</h2>
              </div>
              <div className="flex items-center gap-2">
                {(mode === 'live' || mode === 'calibrating' || mode === 'paused') && (
                  <span className="hidden items-center gap-1.5 font-mono text-xs text-white/55 sm:flex">
                    <Clock3 className="size-3.5" aria-hidden="true" />
                    {formatTime(elapsed)}
                  </span>
                )}
                <Badge className={`session-status h-7 gap-1.5 rounded-full px-3 ${status.className}`}>
                  <span className="status-light size-1.5 rounded-full" />
                  {status.text}
                </Badge>
              </div>
            </div>

            <output
              aria-live="polite"
              aria-atomic="true"
              className={`coach-panel relative z-10 mt-8 flex flex-1 flex-col justify-center rounded-[22px] border p-5 backdrop-blur-sm sm:p-8 coach-${coach.tone}`}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-[#b8ec8f]">
                  <Sparkles className="size-4" aria-hidden="true" />
                  <p className="text-xs font-bold">{coach.label}</p>
                </div>
                {mode === 'live' && (
                  <span className="text-[10px] font-bold uppercase tracking-[0.14em] text-white/35">
                    {round(metrics.confidence)}% confidence
                  </span>
                )}
              </div>
              <p className="mt-2 min-h-14 text-base font-medium leading-7 sm:text-lg">
                {coach.message}
              </p>
              <div
                className="mt-7 flex h-24 items-center gap-[5px] overflow-hidden"
                aria-hidden="true"
              >
                {waveform.map((height, index) => (
                  <span
                    key={index}
                    className={`wave-bar flex-1 rounded-full bg-gradient-to-t from-[#9b8cf5] to-[#b8ec8f] ${mode === 'live' ? 'is-active' : ''}`}
                    style={{
                      height: `${height}%`,
                      animationDelay: `${index * -70}ms`,
                    }}
                  />
                ))}
              </div>
            </output>

            {errorMessage && (
              <div className="relative z-10 mt-4 flex gap-3 rounded-2xl border border-[#ff8d80]/30 bg-[#ff7466]/10 p-4 text-sm leading-6 text-white/80">
                <CircleAlert className="mt-0.5 size-4 shrink-0 text-[#ff9d93]" aria-hidden="true" />
                <p>{errorMessage}</p>
              </div>
            )}

            <div className="relative z-10 mt-5 flex flex-col gap-3 sm:flex-row sm:items-center">
              {(mode === 'idle' || mode === 'error' || mode === 'done') && (
                <>
                  <Button
                    size="lg"
                    onClick={startMicrophone}
                    className="h-12 rounded-full bg-[#ff7466] px-6 text-sm font-bold text-white shadow-[0_12px_24px_rgba(255,116,102,.25)] hover:bg-[#ff6657]"
                  >
                    <Mic className="size-4" aria-hidden="true" />
                    {mode === 'error' ? 'マイクを再試行' : 'マイクをオンにして練習'}
                  </Button>
                  <Button
                    size="lg"
                    variant="ghost"
                    onClick={startDemo}
                    className="h-12 rounded-full px-5 text-white/70 hover:bg-white/10 hover:text-white"
                  >
                    <Play className="size-4" aria-hidden="true" />
                    デモで体験
                  </Button>
                </>
              )}

              {mode === 'requesting' && (
                <Button
                  size="lg"
                  disabled
                  className="h-12 rounded-full bg-white/10 px-6 text-white"
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
                    className="h-12 rounded-full border-white/15 bg-white/8 px-5 text-white hover:bg-white/14 hover:text-white"
                  >
                    <Pause className="size-4" aria-hidden="true" />
                    一時停止
                  </Button>
                  <Button
                    size="lg"
                    variant="ghost"
                    onClick={finishSession}
                    className="h-12 rounded-full px-5 text-white/70 hover:bg-white/10 hover:text-white"
                  >
                    <Square className="size-3.5 fill-current" aria-hidden="true" />
                    終了してレポート
                  </Button>
                </>
              )}

              {mode === 'paused' && (
                <>
                  <Button
                    size="lg"
                    onClick={resumeSession}
                    className="h-12 rounded-full bg-[#b8ec8f] px-6 font-bold text-[#211f3b] hover:bg-[#a9e57b]"
                  >
                    <Play className="size-4 fill-current" aria-hidden="true" />
                    再開
                  </Button>
                  <Button
                    size="lg"
                    variant="ghost"
                    onClick={finishSession}
                    className="h-12 rounded-full px-5 text-white/70 hover:bg-white/10 hover:text-white"
                  >
                    <Square className="size-3.5 fill-current" aria-hidden="true" />
                    終了してレポート
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
              value={round(metrics.tension)}
              unit="/100"
              note={tensionNote}
              progress={metrics.tension}
              color="bg-[#ff7466]"
              track="bg-[#ffddd7]"
              surface="bg-[#fff7ef]"
            />
            <MetricCard
              label="声の熱量"
              value={round(metrics.energy)}
              unit="/100"
              note={energyNote}
              progress={metrics.energy}
              color="bg-[#8270e8]"
              track="bg-[#dcd6ff]"
              surface="bg-[#f2f0ff]"
            />
            <MetricCard
              label="話すテンポ"
              value={round(metrics.pace)}
              unit="語/分・推定"
              note={paceNote}
              progress={((metrics.pace - 70) / 140) * 100}
              color="bg-[#73b84b]"
              track="bg-[#d8ebca]"
              surface="bg-[#f2f9ec]"
            />

            <div className="diagnostic-card sm:col-span-3 lg:col-span-1">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="size-4 text-[#6d5bd0]" aria-hidden="true" />
                  <p className="text-xs font-bold">音声シグナル</p>
                </div>
                <span className="text-[10px] font-semibold text-muted-foreground">
                  {mode === 'idle' ? 'サンプル表示' : 'リアルタイム'}
                </span>
              </div>
              <dl className="grid grid-cols-3 gap-2">
                <div>
                  <dt>F0</dt>
                  <dd>{metrics.pitch ? `${round(metrics.pitch)} Hz` : '—'}</dd>
                </div>
                <div>
                  <dt>揺らぎ</dt>
                  <dd>{(metrics.jitter * 100).toFixed(1)}%</dd>
                </div>
                <div>
                  <dt>安定度</dt>
                  <dd>{round(metrics.stability)}</dd>
                </div>
              </dl>
              <div className="mt-4 flex items-center gap-2 border-t border-border/70 pt-4 text-[11px] leading-5 text-muted-foreground">
                <ShieldCheck className="size-4 shrink-0 text-[#73b84b]" aria-hidden="true" />
                感情を断定せず、声に現れた傾向を表示します
              </div>
            </div>
          </aside>
        </section>

        <section className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1fr)_330px]">
          <article className="transcript-card">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <Captions className="size-4 text-[#6d5bd0]" aria-hidden="true" />
                <h2 className="text-sm font-bold">話した内容</h2>
              </div>
              <Badge variant="outline" className="rounded-full bg-white px-2.5 text-[10px] text-muted-foreground">
                {sessionKind === 'demo'
                  ? 'デモ字幕'
                  : speechSupported
                    ? 'Web Speech API'
                    : '音響推定のみ'}
              </Badge>
            </div>
            <p className="mt-3 min-h-14 text-sm leading-7 text-muted-foreground">
              {transcript || interimTranscript ? (
                <>
                  <span className="text-foreground">{transcript}</span>
                  <span className="text-muted-foreground/60">{interimTranscript}</span>
                </>
              ) : (
                '対応ブラウザでは、話した言葉がここに表示されます。字幕が使えない場合も、音声特徴の解析は続きます。'
              )}
            </p>
          </article>

          <article className="focus-card">
            <span className="grid size-10 place-items-center rounded-2xl bg-[#f1effd] text-[#6d5bd0]">
              <Sparkles className="size-4" aria-hidden="true" />
            </span>
            <div>
              <p className="text-sm font-bold">今日のフォーカス</p>
              <p className="mt-1 text-xs leading-5 text-muted-foreground">
                結論の前に、1秒の間をつくる
              </p>
            </div>
          </article>
        </section>

        <section className="mt-5 grid gap-3 md:grid-cols-3" aria-label="HearEmpathyの仕組み">
          <article className="tech-card">
            <span><Waves aria-hidden="true" /></span>
            <div>
              <h3>Web Audio</h3>
              <p>音量・F0・声の揺らぎを約10回/秒で解析します。</p>
            </div>
          </article>
          <article className="tech-card">
            <span><Captions aria-hidden="true" /></span>
            <div>
              <h3>Web Speech</h3>
              <p>対応ブラウザでは字幕からテンポ推定を補正します。</p>
            </div>
          </article>
          <article className="tech-card">
            <span><Cpu aria-hidden="true" /></span>
            <div>
              <h3>Local WASM</h3>
              <p>5つの音声特徴から緊張サインを端末内で計算します。</p>
            </div>
          </article>
        </section>

        <footer className="mt-8 flex flex-col justify-between gap-3 border-t border-border/80 py-6 text-[11px] leading-5 text-muted-foreground sm:flex-row sm:items-center">
          <p>HearEmpathyは発表練習用のシミュレーターです。医療・心理診断を目的としません。</p>
          <p className="flex items-center gap-1.5">
            <Headphones className="size-3.5" aria-hidden="true" />
            音声フィードバックを使う場合はヘッドホン推奨
          </p>
        </footer>
      </div>

      {summaryOpen && (
        <div
          className="summary-backdrop"
          onMouseDown={(event) => {
            if (event.currentTarget === event.target) setSummaryOpen(false);
          }}
        >
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby="summary-title"
            aria-describedby="summary-description"
            tabIndex={-1}
            className="summary-dialog max-h-[calc(100vh-2rem)] w-[min(620px,calc(100%-2rem))] overflow-y-auto rounded-[28px] p-0"
          >
          <div className="summary-hero">
            <div className="summary-check">
              <Check className="size-5" strokeWidth={3} aria-hidden="true" />
            </div>
            <div className="flex flex-col gap-2">
              <h2 id="summary-title" className="text-2xl font-bold tracking-[-0.04em]">
                リハーサル、おつかれさまでした。
              </h2>
              <p id="summary-description" className="text-sm leading-6 text-muted-foreground">
                声の特徴から、今回の「伝わり方」をまとめました。
              </p>
            </div>
          </div>

          {summary && (
            <div className="px-5 pb-1 sm:px-7">
              <div className="summary-grid">
                <div><span>平均緊張度</span><strong>{summary.tension}</strong><small>/100</small></div>
                <div><span>声の熱量</span><strong>{summary.energy}</strong><small>/100</small></div>
                <div><span>平均テンポ</span><strong>{summary.pace}</strong><small>語/分</small></div>
                <div><span>練習時間</span><strong>{formatTime(summary.duration)}</strong><small>min</small></div>
              </div>

              <div className="mt-5 grid gap-3 sm:grid-cols-2">
                <div className="summary-tip summary-good">
                  <div><Volume2 aria-hidden="true" /><span>よかった点</span></div>
                  <p>
                    {summary.energy >= 65
                      ? '声の熱量が保たれ、聞き手を惹きつける時間がつくれていました。'
                      : '落ち着いた声量で、丁寧に伝える土台ができています。'}
                  </p>
                </div>
                <div className="summary-tip summary-next">
                  <div><Gauge aria-hidden="true" /><span>次のフォーカス</span></div>
                  <p>
                    {summary.pace > 170
                      ? '結論の直前に1秒の間を入れ、聞き手が追いつく余白をつくりましょう。'
                      : summary.tension > 65
                        ? '話し始める前に長く息を吐き、最初の一文をゆっくり届けましょう。'
                        : '強調したい言葉の前後に間を入れると、さらに印象が残ります。'}
                  </p>
                </div>
              </div>

              <div className="mt-4 flex items-start gap-2 rounded-2xl bg-[#f6f4ef] p-3 text-[11px] leading-5 text-muted-foreground">
                <ShieldCheck className="mt-0.5 size-4 shrink-0 text-[#6d5bd0]" aria-hidden="true" />
                このレポートは声の特徴から推定した練習用指標です。録音データは保存していません。
              </div>
            </div>
          )}

          <div className="mt-3 flex flex-col-reverse gap-2 rounded-b-[28px] border-t bg-muted/50 px-5 py-4 sm:flex-row sm:justify-end sm:px-7">
            <Button variant="outline" onClick={() => setSummaryOpen(false)} className="rounded-full">
              画面に戻る
            </Button>
            <Button onClick={resetSession} className="rounded-full bg-[#211f3b] px-5 text-white hover:bg-[#302d52]">
              <RotateCcw className="size-4" aria-hidden="true" />
              もう一度練習
            </Button>
          </div>
          </section>
        </div>
      )}
    </main>
  );
}
