import { startTransition, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkBreaks from 'remark-breaks'
import remarkGfm from 'remark-gfm'
import './App.css'

const DEFAULT_MODEL_KEY = 'e4b'
const DEFAULT_QUANTIZATION_KEY = 'bf16'
const DEFAULT_MAX_NEW_TOKENS = 512
const WORKSPACE_STORAGE_KEY = 'gemma4-lab-workspace-v1'
const DEFAULT_SYSTEM_PROMPT =
  'You are Gemma 4 running locally on a workstation. Be concise, technical, and explicit about what can be inferred from the provided media.'
const CONTINUATION_PROMPT =
  'Continue exactly where your previous answer stopped. Do not restart, summarize, apologize, or repeat any text you already produced. Output only the missing continuation.'
const TITLE_SYSTEM_PROMPT =
  'You generate short conversation titles for a local AI chat app. Return plain text only, 2 to 6 words, no quotes, no markdown, no trailing punctuation.'
const TITLE_PROMPT =
  'Write the best short title for this conversation.'
const AUTO_CONTINUE_LIMIT = 3

const PROMPT_PRESETS = [
  {
    title: 'Vision',
    icon: 'image_search',
    prompt: 'Decris cette image et cite tout texte visible.',
  },
  {
    title: 'Audio',
    icon: 'graphic_eq',
    prompt: 'Transcris cet audio en une seule ligne.',
  },
  {
    title: 'Cross-check',
    icon: 'compare_arrows',
    prompt:
      'Analyse l image puis l audio et dis si les deux racontent la meme chose.',
  },
]

const DEFAULT_LIVE_PROMPT =
  'Use the attached camera frame and microphone audio as the current live turn. Reply conversationally, like a concise video call copilot.'

const NVFP4_MAX_NEW_TOKENS = 64

function mergeAudioSamples(chunks, totalLength) {
  const merged = new Float32Array(totalLength)
  let offset = 0

  chunks.forEach((chunk) => {
    merged.set(chunk, offset)
    offset += chunk.length
  })

  return merged
}

function writeWavString(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index))
  }
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)

  writeWavString(view, 0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeWavString(view, 8, 'WAVE')
  writeWavString(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeWavString(view, 36, 'data')
  view.setUint32(40, samples.length * 2, true)

  let offset = 44
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]))
    view.setInt16(
      offset,
      sample < 0 ? sample * 0x8000 : sample * 0x7fff,
      true,
    )
    offset += 2
  }

  return new Blob([buffer], { type: 'audio/wav' })
}

function stopStreamTracks(stream) {
  if (!stream) {
    return
  }

  stream.getTracks().forEach((track) => track.stop())
}

function captureVideoFrame(videoElement) {
  return new Promise((resolve, reject) => {
    if (!videoElement || videoElement.readyState < 2) {
      reject(new Error('Camera preview is not ready yet.'))
      return
    }

    const canvas = document.createElement('canvas')
    canvas.width = videoElement.videoWidth
    canvas.height = videoElement.videoHeight

    const context = canvas.getContext('2d')
    if (!context) {
      reject(new Error('Unable to capture the current camera frame.'))
      return
    }

    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height)
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error('Unable to encode the current camera frame.'))
          return
        }

        resolve(
          new File([blob], `live-frame-${Date.now()}.jpg`, {
            type: 'image/jpeg',
          }),
        )
      },
      'image/jpeg',
      0.92,
    )
  })
}

function stopAudioPlayback(audioRef) {
  const activeAudio = audioRef.current
  if (!activeAudio) {
    return
  }

  activeAudio.onplay = null
  activeAudio.onended = null
  activeAudio.onerror = null
  activeAudio.pause()
  activeAudio.currentTime = 0
  audioRef.current = null
}

async function playGeneratedAudio(audioPayload, audioRef, onStart, onEnd, onError) {
  if (!audioPayload?.url) {
    return false
  }

  stopAudioPlayback(audioRef)

  const playback = new Audio(audioPayload.url)
  playback.preload = 'auto'
  audioRef.current = playback

  playback.onplay = () => onStart?.()
  playback.onended = () => {
    if (audioRef.current === playback) {
      audioRef.current = null
    }
    onEnd?.()
  }
  playback.onerror = () => {
    if (audioRef.current === playback) {
      audioRef.current = null
    }
    onError?.()
  }

  try {
    await playback.play()
    return true
  } catch {
    if (audioRef.current === playback) {
      audioRef.current = null
    }
    onError?.()
    return false
  }
}

function formatAudioClock(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return '0:00'
  }

  const wholeSeconds = Math.floor(seconds)
  const minutes = Math.floor(wholeSeconds / 60)
  const remainingSeconds = wholeSeconds % 60
  return `${minutes}:${String(remainingSeconds).padStart(2, '0')}`
}

function VoicePlayer({ audioPayload }) {
  const audioElementRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isReady, setIsReady] = useState(false)

  useEffect(() => {
    const audio = audioElementRef.current
    if (!audio) {
      return undefined
    }

    const handleLoadedMetadata = () => {
      setDuration(audio.duration || 0)
      setIsReady(true)
    }
    const handleTimeUpdate = () => setCurrentTime(audio.currentTime || 0)
    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)
    const handleEnded = () => {
      setIsPlaying(false)
      setCurrentTime(0)
    }

    audio.addEventListener('loadedmetadata', handleLoadedMetadata)
    audio.addEventListener('timeupdate', handleTimeUpdate)
    audio.addEventListener('play', handlePlay)
    audio.addEventListener('pause', handlePause)
    audio.addEventListener('ended', handleEnded)

    return () => {
      audio.pause()
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata)
      audio.removeEventListener('timeupdate', handleTimeUpdate)
      audio.removeEventListener('play', handlePlay)
      audio.removeEventListener('pause', handlePause)
      audio.removeEventListener('ended', handleEnded)
    }
  }, [audioPayload?.url])

  async function handleTogglePlayback() {
    const audio = audioElementRef.current
    if (!audio) {
      return
    }

    if (audio.paused) {
      try {
        await audio.play()
      } catch {
        setIsPlaying(false)
      }
      return
    }

    audio.pause()
  }

  function handleSeek(event) {
    const audio = audioElementRef.current
    if (!audio) {
      return
    }

    const nextTime = Number(event.target.value || 0)
    audio.currentTime = nextTime
    setCurrentTime(nextTime)
  }

  return (
    <div className="voice-player">
      <audio ref={audioElementRef} preload="metadata" src={audioPayload.url} />

      <button className="voice-player-toggle" type="button" onClick={handleTogglePlayback}>
        <span className="material-symbols-outlined">
          {isPlaying ? 'pause' : 'play_arrow'}
        </span>
      </button>

      <div className="voice-player-body">
        <div className="voice-player-head">
          <div className="voice-player-title">
            <strong>Local voice</strong>
            <span>{audioPayload.voice || 'Piper'}</span>
          </div>

          <div className="voice-player-time">
            <span>{formatAudioClock(currentTime)}</span>
            <span>{formatAudioClock(duration)}</span>
          </div>
        </div>

        <input
          className="voice-player-range"
          type="range"
          min="0"
          max={duration || 0}
          step="0.01"
          value={Math.min(currentTime, duration || 0)}
          onChange={handleSeek}
          disabled={!isReady || !duration}
          style={{
            '--voice-progress': `${
              duration > 0 ? Math.min((currentTime / duration) * 100, 100) : 0
            }%`,
          }}
        />

        <div className="voice-player-meta">
          <span>{audioPayload.mime_type === 'audio/wav' ? 'WAV' : 'Audio'}</span>
          <span>{audioPayload.sample_rate ? `${audioPayload.sample_rate} Hz` : 'Local clip'}</span>
          <span>
            {typeof audioPayload.elapsed_ms === 'number'
              ? `${audioPayload.elapsed_ms.toFixed(0)} ms synth`
              : 'Server-side TTS'}
          </span>
        </div>
      </div>
    </div>
  )
}

function MarkdownBlock({ content, variant = 'assistant' }) {
  if (!content) {
    return null
  }

  return (
    <ReactMarkdown
      className={`markdown-body markdown-body-${variant}`}
      remarkPlugins={[remarkGfm, remarkBreaks]}
      components={{
        a: ({ ...props }) => <a {...props} rel="noreferrer" target="_blank" />,
        pre: ({ ...props }) => <pre className="markdown-pre" {...props} />,
        code: ({ className, children, ...props }) => (
          <code className={className} {...props}>
            {children}
          </code>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}

async function parseApiPayload(response) {
  const raw = await response.text()

  if (!raw) {
    return {}
  }

  try {
    return JSON.parse(raw)
  } catch {
    return {
      detail: raw.trim() || `HTTP ${response.status}`,
    }
  }
}

function formatElapsed(milliseconds) {
  if (!milliseconds) {
    return 'n/a'
  }

  return milliseconds >= 1000
    ? `${(milliseconds / 1000).toFixed(1)} s`
    : `${Math.round(milliseconds)} ms`
}

function formatPercent(value) {
  return typeof value === 'number' ? `${value.toFixed(0)}%` : 'n/a'
}

function formatMemory(value) {
  return typeof value === 'number' ? `${value.toFixed(2)} GiB` : 'n/a'
}

function formatNumber(value, suffix = '') {
  return typeof value === 'number' ? `${value}${suffix}` : 'n/a'
}

function getModelMemoryEstimate(model, quantizationKey) {
  if (!model?.memory_requirements_gib || !quantizationKey) {
    return null
  }

  const value = model.memory_requirements_gib[quantizationKey]
  return typeof value === 'number' ? value : null
}

function getQuantizationRuntimeLabel(quantization) {
  if (!quantization) {
    return 'Unknown runtime'
  }

  if (quantization.runtime_supported) {
    if (quantization.runtime_family === 'llama.cpp') {
      return 'llama.cpp local'
    }

    if (quantization.runtime_family === 'vllm-wsl') {
      return 'WSL vLLM'
    }

    return 'Available here'
  }

  if (quantization.status === 'linux-vllm-only') {
    return 'Linux vLLM only'
  }

  return 'Planning only'
}

function formatMessageTime(value) {
  if (!value) {
    return '--:--'
  }

  return new Intl.DateTimeFormat('fr-FR', {
    hour: '2-digit',
    minute: '2-digit',
  }).format(value)
}

function formatDateTime(value) {
  if (!value) {
    return 'n/a'
  }

  return new Intl.DateTimeFormat('fr-FR', {
    dateStyle: 'short',
    timeStyle: 'medium',
  }).format(value * 1000)
}

function getRequestStatusTone(status) {
  switch (status) {
    case 'running':
      return 'is-active'
    case 'completed':
      return 'is-ready'
    case 'failed':
      return 'is-failed'
    default:
      return ''
  }
}

function summarizeRouteLabel(route) {
  if (!route) {
    return 'Unknown route'
  }

  return route.replace('/api/v1/', '').replace('/api/', '')
}

function makeAttachmentSummary(imageFile, audioFile) {
  const labels = []

  if (imageFile) {
    labels.push(`image: ${imageFile.name}`)
  }

  if (audioFile) {
    labels.push(`audio: ${audioFile.name}`)
  }

  return labels.join(' | ')
}

function makeAttachmentLabels(modelLabel, imageFile, audioFile) {
  const labels = [modelLabel]

  if (imageFile) {
    labels.push(`Image: ${imageFile.name}`)
  }

  if (audioFile) {
    labels.push(`Audio: ${audioFile.name}`)
  }

  return labels.filter(Boolean)
}

function makeFallbackPrompt(prompt, imageFile, audioFile) {
  const trimmed = prompt.trim()
  if (trimmed) {
    return trimmed
  }

  if (imageFile && audioFile) {
    return 'Decris l image et transcris l audio.'
  }

  if (imageFile) {
    return 'Decris cette image.'
  }

  if (audioFile) {
    return 'Transcris cet audio.'
  }

  return ''
}

function makeThreadTitle(prompt, imageFile, audioFile) {
  const trimmed = prompt.trim().replace(/\s+/g, ' ')

  if (trimmed) {
    return trimmed.length > 40 ? `${trimmed.slice(0, 40).trimEnd()}...` : trimmed
  }

  if (imageFile && audioFile) {
    return 'Image and audio test'
  }

  if (imageFile) {
    return 'Image test'
  }

  if (audioFile) {
    return 'Audio test'
  }

  return 'New Chat'
}

function normalizeGeneratedTitle(rawTitle, fallbackTitle) {
  const singleLine = String(rawTitle || '')
    .replace(/\[(.*?)\]\((.*?)\)/g, '$1')
    .replace(/[*_`>#]+/g, ' ')
    .replace(/^title\s*:\s*/i, '')
    .split(/\r?\n/)[0]
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/[.,;:!?-]+$/g, '')

  if (!singleLine) {
    return fallbackTitle
  }

  const words = singleLine.split(/\s+/).slice(0, 8).join(' ')
  if (!words) {
    return fallbackTitle
  }

  return words.length > 64 ? `${words.slice(0, 64).trimEnd()}...` : words
}

function createThread(title) {
  const now = Date.now()
  return {
    id: crypto.randomUUID(),
    title,
    createdAt: now,
    updatedAt: now,
    messages: [],
  }
}

function normalizeStoredMessage(message) {
  if (!message || typeof message !== 'object') {
    return null
  }

  return {
    ...message,
    streamingState:
      message.streamingState === 'waiting' || message.streamingState === 'streaming'
        ? 'done'
        : message.streamingState,
  }
}

function restoreWorkspace() {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    const raw = window.localStorage.getItem(WORKSPACE_STORAGE_KEY)
    if (!raw) {
      return null
    }

    const parsed = JSON.parse(raw)
    const restoredThreads = Array.isArray(parsed?.threads)
      ? parsed.threads
          .map((thread) => {
            if (!thread || typeof thread !== 'object' || !thread.id) {
              return null
            }

            const restoredMessages = Array.isArray(thread.messages)
              ? thread.messages.map(normalizeStoredMessage).filter(Boolean)
              : []

            return {
              ...thread,
              title: typeof thread.title === 'string' && thread.title.trim() ? thread.title : 'New Chat',
              updatedAt: Number(thread.updatedAt) || Date.now(),
              createdAt: Number(thread.createdAt) || Date.now(),
              messages: restoredMessages,
            }
          })
          .filter(Boolean)
      : []

    if (restoredThreads.length === 0) {
      return null
    }

    const activeThreadId = restoredThreads.some((thread) => thread.id === parsed?.activeThreadId)
      ? parsed.activeThreadId
      : restoredThreads[0].id

    return {
      threads: restoredThreads,
      activeThreadId,
    }
  } catch {
    return null
  }
}

function createInitialWorkspace() {
  const restoredWorkspace = restoreWorkspace()
  if (restoredWorkspace) {
    return restoredWorkspace
  }

  const firstThread = createThread('New Chat')
  return {
    threads: [firstThread],
    activeThreadId: firstThread.id,
  }
}

function persistWorkspace(workspace) {
  if (typeof window === 'undefined') {
    return
  }

  try {
    window.localStorage.setItem(WORKSPACE_STORAGE_KEY, JSON.stringify(workspace))
  } catch {
    // Best-effort persistence only.
  }
}

function updateThreadInList(currentThreads, threadId, updater) {
  const nextThreads = []
  let updatedThread = null

  currentThreads.forEach((thread) => {
    if (thread.id === threadId) {
      updatedThread = updater(thread)
      return
    }

    nextThreads.push(thread)
  })

  return updatedThread ? [updatedThread, ...nextThreads] : currentThreads
}

function updateMessageInThread(currentThreads, threadId, messageId, updater) {
  return updateThreadInList(currentThreads, threadId, (thread) => ({
    ...thread,
    updatedAt: Date.now(),
    messages: thread.messages.map((message) =>
      message.id === messageId ? updater(message) : message,
    ),
  }))
}

async function readNdjsonStream(response, onEvent) {
  if (!response.body) {
    throw new Error('Streaming response body unavailable in this browser.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) {
      break
    }

    buffer += decoder.decode(value, { stream: true })
    let newlineIndex = buffer.indexOf('\n')

    while (newlineIndex >= 0) {
      const line = buffer.slice(0, newlineIndex).trim()
      buffer = buffer.slice(newlineIndex + 1)

      if (line) {
        onEvent(JSON.parse(line))
      }

      newlineIndex = buffer.indexOf('\n')
    }
  }

  const tail = buffer.trim()
  if (tail) {
    onEvent(JSON.parse(tail))
  }
}

function getGreeting() {
  const hour = new Date().getHours()

  if (hour < 12) {
    return 'Good morning'
  }

  if (hour < 18) {
    return 'Good afternoon'
  }

  return 'Good evening'
}

function getThreadSubtitle(thread) {
  const latestMessage = thread.messages[thread.messages.length - 1]

  if (!latestMessage) {
    return 'No turns yet'
  }

  if (latestMessage.role === 'assistant' && latestMessage.streamingState === 'waiting') {
    return 'Preparing the first tokens...'
  }

  if (latestMessage.role === 'assistant' && latestMessage.streamingState === 'streaming') {
    return 'Streaming reply...'
  }

  if (latestMessage.role === 'assistant' && latestMessage.meta) {
    return `${latestMessage.meta.active_model?.label || 'Gemma 4'} / ${
      latestMessage.meta.active_quantization?.label || 'BF16'
    } / ${formatElapsed(latestMessage.meta.elapsed_ms)}`
  }

  if (latestMessage.attachmentLabels?.length) {
    return latestMessage.attachmentLabels.slice(0, 2).join(' | ')
  }

  const compact = latestMessage.content.replace(/\s+/g, ' ')
  return compact.length > 44 ? `${compact.slice(0, 44).trimEnd()}...` : compact
}

function getAssistantChips(message) {
  if (!message.meta) {
    return []
  }

  const chips = []
  const activeModel = message.meta.active_model

  if (activeModel?.label) {
    chips.push(activeModel.label)
  }

  if (activeModel?.architecture) {
    chips.push(activeModel.architecture)
  }

  if (message.meta.elapsed_ms) {
    chips.push(formatElapsed(message.meta.elapsed_ms))
  }

  if (typeof message.meta.generated_tokens === 'number') {
    chips.push(`${message.meta.generated_tokens} new tokens`)
  }

  if (message.meta.hit_max_tokens) {
    chips.push('Hit token limit')
  }

  if (message.meta.tts_audio?.voice) {
    chips.push(`Voice: ${message.meta.tts_audio.voice}`)
  }

  if (message.meta.active_quantization?.label) {
    chips.push(message.meta.active_quantization.label)
  }

  return chips
}

function serializeHistoryMessages(messageList) {
  return messageList.slice(-8).map((message) => ({
    role: message.role,
    content: message.content,
  }))
}

function shouldContinuePayload(payload) {
  return payload?.finish_reason === 'length' || payload?.hit_max_tokens === true
}

function mergeThought(existingThought, nextThought) {
  if (!existingThought) {
    return nextThought || undefined
  }

  if (!nextThought || existingThought.includes(nextThought)) {
    return existingThought
  }

  return `${existingThought}\n\n${nextThought}`
}

function getDefaultLoadState() {
  return {
    status: 'idle',
    is_loading: false,
    progress: 0,
    message: 'Select a model and click Load model.',
    error: '',
    target_model_key: DEFAULT_MODEL_KEY,
    target_model: null,
    target_quantization_key: DEFAULT_QUANTIZATION_KEY,
    target_quantization: null,
    started_at: null,
    finished_at: null,
    updated_at: null,
  }
}

function normalizeLoadState(loadState) {
  const fallback = getDefaultLoadState()
  const progressValue = Number(loadState?.progress)

  return {
    ...fallback,
    ...(loadState || {}),
    status: loadState?.status || fallback.status,
    is_loading: Boolean(loadState?.is_loading),
    progress: Number.isFinite(progressValue)
      ? Math.max(0, Math.min(100, progressValue))
      : fallback.progress,
    message: loadState?.message || fallback.message,
    error: loadState?.error || '',
  }
}

function getLoadStatusLabel(loadState, health) {
  if (loadState.is_loading) {
    return 'Loading model'
  }

  if (loadState.status === 'failed') {
    return 'Load failed'
  }

  if (health?.loaded) {
    return 'Warm GPU'
  }

  return 'Cold start'
}

function App() {
  const initialWorkspaceRef = useRef(createInitialWorkspace())
  const imageInputRef = useRef(null)
  const audioInputRef = useRef(null)
  const logRef = useRef(null)
  const modelPickerRef = useRef(null)
  const quantizationPickerRef = useRef(null)
  const liveStageRef = useRef(null)
  const liveVideoRef = useRef(null)
  const liveStreamRef = useRef(null)
  const liveAudioContextRef = useRef(null)
  const liveAudioSourceRef = useRef(null)
  const liveAudioProcessorRef = useRef(null)
  const liveAudioSinkRef = useRef(null)
  const liveAudioChunksRef = useRef([])
  const liveAudioLengthRef = useRef(0)
  const playbackAudioRef = useRef(null)

  const [threads, setThreads] = useState(initialWorkspaceRef.current.threads)
  const [activeThreadId, setActiveThreadId] = useState(
    initialWorkspaceRef.current.activeThreadId,
  )
  const [models, setModels] = useState([])
  const [quantizations, setQuantizations] = useState([])
  const [docsNote, setDocsNote] = useState('')
  const [prompt, setPrompt] = useState('')
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT)
  const [selectedModelKey, setSelectedModelKey] = useState(DEFAULT_MODEL_KEY)
  const [selectedQuantizationKey, setSelectedQuantizationKey] = useState(
    DEFAULT_QUANTIZATION_KEY,
  )
  const [imageFile, setImageFile] = useState(null)
  const [audioFile, setAudioFile] = useState(null)
  const [imagePreview, setImagePreview] = useState('')
  const [audioPreview, setAudioPreview] = useState('')
  const [health, setHealth] = useState(null)
  const [modelLoadState, setModelLoadState] = useState(getDefaultLoadState())
  const [error, setError] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [isLoadingModel, setIsLoadingModel] = useState(false)
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false)
  const [openModelInfoKey, setOpenModelInfoKey] = useState(null)
  const [isQuantizationMenuOpen, setIsQuantizationMenuOpen] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [viewMode, setViewMode] = useState('text')
  const [monitoringSnapshot, setMonitoringSnapshot] = useState(null)
  const [monitoringError, setMonitoringError] = useState('')
  const [isRefreshingMonitoring, setIsRefreshingMonitoring] = useState(false)
  const [livePrompt, setLivePrompt] = useState(DEFAULT_LIVE_PROMPT)
  const [liveStatus, setLiveStatus] = useState('Live mode idle.')
  const [liveError, setLiveError] = useState('')
  const [isLiveSessionActive, setIsLiveSessionActive] = useState(false)
  const [isLiveRecording, setIsLiveRecording] = useState(false)
  const [isLiveSubmitting, setIsLiveSubmitting] = useState(false)
  const [isLiveFullscreen, setIsLiveFullscreen] = useState(false)
  const [autoSpeak, setAutoSpeak] = useState(true)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [settings, setSettings] = useState({
    maxNewTokens: DEFAULT_MAX_NEW_TOKENS,
    temperature: 1,
    topP: 0.95,
    topK: 64,
    thinking: false,
    continuationMode: 'manual',
  })

  const activeThread =
    threads.find((thread) => thread.id === activeThreadId) ?? threads[0] ?? null
  const messages = activeThread?.messages ?? []
  const latestAssistantTurn =
    [...messages].reverse().find((message) => message.role === 'assistant') ?? null
  const latestAssistantMessage = [...messages]
    .reverse()
    .find((message) => message.role === 'assistant' && message.meta)
  const selectedModel =
    models.find((model) => model.key === selectedModelKey) ?? null
  const selectedQuantization =
    quantizations.find((quantization) => quantization.key === selectedQuantizationKey) ??
    null
  const activeModel = health?.active_model ?? null
  const activeQuantization = health?.active_quantization ?? null
  const normalizedLoadState = normalizeLoadState(modelLoadState)
  const isModelLoading = isLoadingModel || normalizedLoadState.is_loading
  const isLlamaCppRuntime = selectedQuantization?.runtime_family === 'llama.cpp'
  const isWslVllmRuntime = selectedQuantization?.runtime_family === 'vllm-wsl'
  const supportsAudio = isLlamaCppRuntime || isWslVllmRuntime
    ? false
    : selectedModel?.supports_audio ?? true
  const supportsImage = selectedModel?.supports_image ?? true
  const modalityBadges = selectedModel
    ? isWslVllmRuntime
        ? selectedModel.supported_modalities.filter((modality) => modality !== 'video')
        : selectedModel.supported_modalities
    : []
  const selectedMemoryEstimate = getModelMemoryEstimate(
    selectedModel,
    selectedQuantizationKey,
  )
  const activeMemoryEstimate = getModelMemoryEstimate(
    activeModel,
    activeQuantization?.key,
  )
  const selectedModelLoaded =
    Boolean(health?.loaded) &&
    activeModel?.key === selectedModelKey &&
    activeQuantization?.key === selectedQuantizationKey
  const modelSwitchPending =
    Boolean(selectedModelKey) &&
    Boolean(selectedQuantizationKey) &&
    !selectedModelLoaded
  const greeting = getGreeting()
  const activeThreadSubtitle = activeThread
    ? getThreadSubtitle(activeThread)
    : 'No turns yet'
  const quantizationRuntimeSupported =
    selectedQuantization?.runtime_supported ?? false
  const supportsLiveVision =
    quantizationRuntimeSupported && supportsImage
  const supportsNativeLiveAudio =
    quantizationRuntimeSupported &&
    supportsAudio &&
    ['e2b', 'e4b'].includes(selectedModelKey)
  const liveModeCapabilityLabel = !quantizationRuntimeSupported
    ? 'Quantization not runnable here'
    : supportsNativeLiveAudio
      ? 'Vision + audio input'
      : supportsLiveVision
        ? 'Vision input only'
        : 'Live input unavailable'
  const liveModeCapabilityNote = !quantizationRuntimeSupported
    ? 'Load a runnable quantization before starting live turns.'
    : supportsNativeLiveAudio
      ? `${selectedModel?.label || 'This model'} accepts camera frames and microphone turns in live mode.`
      : supportsLiveVision
        ? `${selectedModel?.label || 'This model'} accepts camera frames in live mode, but not audio input.`
        : `${selectedModel?.label || 'This model'} is not configured for live multimodal input in this app build.`
  const audioAttachmentHint = !quantizationRuntimeSupported
    ? 'This quantization is planning only in the current backend'
    : supportsAudio
      ? 'Attach audio'
      : isWslVllmRuntime
        ? 'Audio is unavailable in the current WSL vLLM bridge'
        : 'Audio is only available on E2B and E4B'
  const latestTurns = messages.slice(-4)
  const loadTargetModel = normalizedLoadState.target_model ?? selectedModel
  const loadTargetQuantization =
    normalizedLoadState.target_quantization ?? selectedQuantization
  const loadPanelState =
    normalizedLoadState.status === 'failed'
      ? 'failed'
      : isModelLoading
        ? 'loading'
        : selectedModelLoaded
          ? 'ready'
          : 'pending'
  const shouldShowLoadPanel =
    isModelLoading ||
    normalizedLoadState.status === 'failed' ||
    modelSwitchPending ||
    !health?.loaded ||
    !quantizationRuntimeSupported
  const loadProgressValue =
    loadPanelState === 'ready'
      ? 100
      : loadPanelState === 'pending'
        ? 0
        : normalizedLoadState.progress
  const loadPanelHeadline = isModelLoading
    ? `Loading ${loadTargetModel?.label || selectedModel?.label || 'Gemma 4'}`
    : normalizedLoadState.status === 'failed'
      ? normalizedLoadState.error
        ? 'Load blocked'
        : 'Load stopped'
      : selectedModelLoaded
        ? `${activeModel?.label || 'Gemma 4'} ready`
        : 'Explicit model loading'
  const loadPanelMessage = (() => {
    if (normalizedLoadState.error) {
      return normalizedLoadState.error
    }

    if (isModelLoading) {
      return normalizedLoadState.message
    }

    if (!quantizationRuntimeSupported) {
      return (
        selectedQuantization?.doc_summary ||
        'This quantization is visible for planning only in the current backend.'
      )
    }

    if (!health?.loaded) {
      return `Nothing is loaded yet. Click Load model to warm ${selectedModel?.label || 'the selected variant'} in ${selectedQuantization?.label || 'BF16'}.`
    }

    if (modelSwitchPending && activeModel && activeQuantization) {
      return `VRAM currently holds ${activeModel.label} in ${activeQuantization.label}. Click Load model to switch to ${selectedModel?.label || 'the selected variant'} / ${selectedQuantization?.label || 'BF16'}.`
    }

    return normalizedLoadState.message
  })()
  const loadStatusLabel = getLoadStatusLabel(normalizedLoadState, health)
  const loadStatusMeta = loadTargetModel?.label
    ? `${loadTargetModel.label} | ${loadTargetQuantization?.label || 'BF16'}`
    : `${selectedModel?.label || 'Gemma 4'} | ${selectedQuantization?.label || 'BF16'}`
  const liveTranscriptStatus =
    latestAssistantTurn?.streamingState === 'waiting'
      ? 'Preparing'
      : latestAssistantTurn?.streamingState === 'streaming'
        ? 'Streaming'
        : isSpeaking
          ? 'Playing'
          : 'Ready'
  const liveTranscriptText =
    latestAssistantTurn?.content ||
    (latestAssistantTurn?.streamingState === 'waiting'
      ? 'Gemma is preparing the first tokens...'
      : latestAssistantTurn?.streamingState === 'streaming'
        ? 'Gemma is streaming the reply...'
        : 'The latest assistant transcript will appear here.')
  const canSendTurns =
    Boolean(selectedModelLoaded) &&
    !isModelLoading &&
    quantizationRuntimeSupported
  const monitoringHealth = monitoringSnapshot?.health ?? health
  const monitoringCurrentModel =
    monitoringSnapshot?.current_model?.model ??
    monitoringSnapshot?.health?.active_model ??
    null
  const monitoringGpu = monitoringSnapshot?.gpu ?? null
  const monitoringMemory = monitoringSnapshot?.memory ?? null
  const monitoringQueue = monitoringSnapshot?.queue ?? null
  const monitoringRecentRequests = monitoringSnapshot?.recent_requests ?? []
  const monitoringSystemRamUsed =
    typeof monitoringMemory?.total_physical_gib === 'number' &&
    typeof monitoringMemory?.available_physical_gib === 'number'
      ? monitoringMemory.total_physical_gib - monitoringMemory.available_physical_gib
      : null

  useEffect(() => {
    persistWorkspace({
      threads,
      activeThreadId,
    })
  }, [threads, activeThreadId])

  useEffect(() => {
    if (!imageFile) {
      setImagePreview('')
      return
    }

    const objectUrl = URL.createObjectURL(imageFile)
    setImagePreview(objectUrl)

    return () => URL.revokeObjectURL(objectUrl)
  }, [imageFile])

  useEffect(() => {
    if (!audioFile) {
      setAudioPreview('')
      return
    }

    const objectUrl = URL.createObjectURL(audioFile)
    setAudioPreview(objectUrl)

    return () => URL.revokeObjectURL(objectUrl)
  }, [audioFile])

  useEffect(() => () => stopAudioPlayback(playbackAudioRef), [])

  useEffect(() => {
    if (!supportsAudio && audioFile) {
      setAudioFile(null)
      setError(
        `${
          selectedModel?.label || 'This model'
        } does not accept audio input in the official modality tables.`,
      )
    }
  }, [audioFile, selectedModel, supportsAudio])

  useEffect(() => {
    if (!supportsImage && imageFile) {
      setImageFile(null)
      setError(
        `${
          selectedModel?.label || 'This model'
        } does not accept image input in the official modality tables.`,
      )
    }
  }, [imageFile, selectedModel, selectedQuantization, supportsImage])

  useEffect(() => {
    if (!selectedModel?.memory_requirements_gib) {
      return
    }

    const compatibleQuantizations = Object.keys(selectedModel.memory_requirements_gib)
    if (compatibleQuantizations.length === 0) {
      return
    }

    if (compatibleQuantizations.includes(selectedQuantizationKey)) {
      return
    }

    const fallbackQuantization = compatibleQuantizations.includes(DEFAULT_QUANTIZATION_KEY)
      ? DEFAULT_QUANTIZATION_KEY
      : compatibleQuantizations[0]
    setSelectedQuantizationKey(fallbackQuantization)
  }, [selectedModel, selectedQuantizationKey])

  useEffect(() => {
    if (!isWslVllmRuntime || settings.maxNewTokens <= NVFP4_MAX_NEW_TOKENS) {
      return
    }

    setSettings((current) => ({
      ...current,
      maxNewTokens: NVFP4_MAX_NEW_TOKENS,
    }))
  }, [isWslVllmRuntime, settings.maxNewTokens])

  useEffect(() => {
    if (!isModelMenuOpen) {
      setOpenModelInfoKey(null)
    }
  }, [isModelMenuOpen])

  useEffect(() => {
    let isActive = true

    async function bootstrap() {
      try {
        const [healthResponse, modelsResponse] = await Promise.all([
          fetch('/api/health'),
          fetch('/api/models'),
        ])

        if (!healthResponse.ok || !modelsResponse.ok) {
          throw new Error('Local API unavailable.')
        }

        const [healthData, modelsData] = await Promise.all([
          parseApiPayload(healthResponse),
          parseApiPayload(modelsResponse),
        ])

        if (!healthResponse.ok || !modelsResponse.ok) {
          throw new Error(
            healthData.detail || modelsData.detail || 'Local API unavailable.',
          )
        }

        if (!isActive) {
          return
        }

        startTransition(() => {
          setHealth(healthData)
          setModelLoadState(normalizeLoadState(healthData.load_state))
          setModels(modelsData.models)
          setDocsNote(modelsData.docs_note)
          setQuantizations(modelsData.quantizations || [])
          setSelectedModelKey(
            healthData.active_model_key ||
              modelsData.active_model_key ||
              modelsData.default_model_key ||
              DEFAULT_MODEL_KEY,
          )
          setSelectedQuantizationKey(
            healthData.active_quantization_key ||
              modelsData.active_quantization_key ||
              modelsData.default_quantization_key ||
              DEFAULT_QUANTIZATION_KEY,
          )
        })
      } catch {
        if (!isActive) {
          return
        }

        startTransition(() => {
          setHealth(null)
          setModelLoadState(getDefaultLoadState())
          setModels([])
          setQuantizations([])
        })
      }
    }

    async function pollHealth() {
      try {
        const response = await fetch('/api/health')
        if (!response.ok) {
          throw new Error('Local API unavailable.')
        }

        const data = await parseApiPayload(response)
        if (isActive) {
          startTransition(() => {
            setHealth(data)
            setModelLoadState(normalizeLoadState(data.load_state))
          })
        }
      } catch {
        if (isActive) {
          startTransition(() => {
            setHealth(null)
            setModelLoadState(getDefaultLoadState())
          })
        }
      }
    }

    bootstrap()
    const timer = window.setInterval(pollHealth, 5000)

    return () => {
      isActive = false
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    if (!logRef.current) {
      return
    }

    const scrollBehavior =
      latestAssistantTurn?.streamingState === 'waiting' ||
      latestAssistantTurn?.streamingState === 'streaming'
        ? 'auto'
        : 'smooth'

    const animationFrame = window.requestAnimationFrame(() => {
      logRef.current?.scrollTo({
        top: logRef.current.scrollHeight,
        behavior: scrollBehavior,
      })
    })

    return () => window.cancelAnimationFrame(animationFrame)
  }, [
    messages.length,
    isSending,
    activeThreadId,
    latestAssistantTurn?.id,
    latestAssistantTurn?.streamingState,
    latestAssistantTurn?.content,
  ])

  useEffect(() => {
    function handlePointerDown(event) {
      if (
        isModelMenuOpen &&
        modelPickerRef.current &&
        !modelPickerRef.current.contains(event.target)
      ) {
        setIsModelMenuOpen(false)
      }

      if (
        isQuantizationMenuOpen &&
        quantizationPickerRef.current &&
        !quantizationPickerRef.current.contains(event.target)
      ) {
        setIsQuantizationMenuOpen(false)
      }
    }

    function handleEscape(event) {
      if (event.key === 'Escape') {
        setIsModelMenuOpen(false)
        setIsQuantizationMenuOpen(false)
        setIsSettingsOpen(false)
      }
    }

    document.addEventListener('mousedown', handlePointerDown)
    window.addEventListener('keydown', handleEscape)

    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      window.removeEventListener('keydown', handleEscape)
    }
  }, [isModelMenuOpen, isQuantizationMenuOpen])

  useEffect(() => {
    function handleFullscreenChange() {
      setIsLiveFullscreen(Boolean(document.fullscreenElement))
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
    }
  }, [])

  useEffect(() => {
    return () => {
      stopAudioPlayback(playbackAudioRef)
      void teardownLiveRecorder()
      stopStreamTracks(liveStreamRef.current)
    }
  }, [])

  async function refreshModelsAndHealth() {
    const [healthResponse, modelsResponse] = await Promise.all([
      fetch('/api/health'),
      fetch('/api/models'),
    ])

    if (!healthResponse.ok || !modelsResponse.ok) {
      throw new Error('Unable to refresh local lab state.')
    }

    const [healthData, modelsData] = await Promise.all([
      parseApiPayload(healthResponse),
      parseApiPayload(modelsResponse),
    ])

    startTransition(() => {
      setHealth(healthData)
      setModelLoadState(normalizeLoadState(healthData.load_state))
      setModels(modelsData.models)
      setDocsNote(modelsData.docs_note)
      setQuantizations(modelsData.quantizations || [])
    })
  }

  async function refreshMonitoring({ silent = false } = {}) {
    if (!silent) {
      setIsRefreshingMonitoring(true)
    }

    try {
      const response = await fetch('/api/v1/monitoring')
      if (!response.ok) {
        const data = await parseApiPayload(response)
        throw new Error(data.detail || 'Unable to refresh monitoring.')
      }

      const data = await parseApiPayload(response)
      startTransition(() => {
        setMonitoringSnapshot(data)
        setMonitoringError('')
      })
    } catch (refreshError) {
      startTransition(() => {
        setMonitoringError(refreshError.message)
      })
    } finally {
      if (!silent) {
        setIsRefreshingMonitoring(false)
      }
    }
  }

  useEffect(() => {
    if (!isModelLoading) {
      return undefined
    }

    let isActive = true

    async function pollLoadState() {
      try {
        const [loadResponse, healthResponse] = await Promise.all([
          fetch('/api/models/load-status'),
          fetch('/api/health'),
        ])

        if (!loadResponse.ok || !healthResponse.ok) {
          throw new Error('Unable to refresh model load status.')
        }

        const [loadData, healthData] = await Promise.all([
          parseApiPayload(loadResponse),
          parseApiPayload(healthResponse),
        ])

        if (!isActive) {
          return
        }

        const nextLoadState = normalizeLoadState(loadData.load_state)

        startTransition(() => {
          setModelLoadState(nextLoadState)
          setHealth(healthData)
        })

        if (!nextLoadState.is_loading) {
          setIsLoadingModel(false)

          if (nextLoadState.status === 'failed' && nextLoadState.error) {
            setError(nextLoadState.error)
          }
        }
      } catch (pollError) {
        if (!isActive) {
          return
        }

        setIsLoadingModel(false)
        setError(pollError.message)
      }
    }

    void pollLoadState()
    const timer = window.setInterval(() => {
      void pollLoadState()
    }, 900)

    return () => {
      isActive = false
      window.clearInterval(timer)
    }
  }, [isModelLoading])

  useEffect(() => {
    if (viewMode !== 'monitoring') {
      return undefined
    }

    let isActive = true

    async function pollMonitoring() {
      try {
        const response = await fetch('/api/v1/monitoring')
        if (!response.ok) {
          throw new Error('Unable to refresh monitoring.')
        }
        const data = await parseApiPayload(response)
        if (!isActive) {
          return
        }
        startTransition(() => {
          setMonitoringSnapshot(data)
          setMonitoringError('')
        })
      } catch (pollError) {
        if (!isActive) {
          return
        }
        startTransition(() => {
          setMonitoringError(pollError.message)
        })
      }
    }

    void pollMonitoring()
    const timer = window.setInterval(() => {
      void pollMonitoring()
    }, 2500)

    return () => {
      isActive = false
      window.clearInterval(timer)
    }
  }, [viewMode])

  async function runTurn({
    promptText,
    imageAsset = null,
    audioAsset = null,
    visibleText = promptText,
    titleHint = null,
    mode = 'text',
    clearComposer = false,
  }) {
    if (!activeThread) {
      return null
    }

    if (isModelLoading || normalizedLoadState.is_loading) {
      const detail =
        normalizedLoadState.error ||
        normalizedLoadState.message ||
        'A model is still loading. Wait for it to finish before sending.'
      setError(detail)
      if (mode === 'live') {
        setLiveError(detail)
      }
      return null
    }

    if (!quantizationRuntimeSupported) {
      const detail =
        selectedQuantization?.doc_summary ||
        'This quantization is visible for planning only in the current backend.'
      setError(detail)
      if (mode === 'live') {
        setLiveError(detail)
      }
      return null
    }

    if (!health?.loaded) {
      const detail = `No model is loaded yet. Click Load model for ${selectedModel?.label || 'the selected variant'} in ${selectedQuantization?.label || 'BF16'} first.`
      setError(detail)
      if (mode === 'live') {
        setLiveError(detail)
      }
      return null
    }

    if (!selectedModelLoaded) {
      const detail = `VRAM currently holds ${activeModel?.label || 'another model'} in ${activeQuantization?.label || 'BF16'}. Click Load model to switch to ${selectedModel?.label || 'the selected variant'} / ${selectedQuantization?.label || 'BF16'} before sending.`
      setError(detail)
      if (mode === 'live') {
        setLiveError(detail)
      }
      return null
    }

    const setBusy = mode === 'live' ? setIsLiveSubmitting : setIsSending
    setBusy(true)
    setError('')
    const shouldGenerateThreadTitle = mode === 'text' && messages.length === 0
    const fallbackThreadTitle = titleHint || makeThreadTitle(visibleText, imageAsset, audioAsset)

    if (mode === 'live') {
      setLiveError('')
      setLiveStatus('Model is preparing the reply...')
    }

    const threadId = activeThread.id
    const pendingUserMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: visibleText,
      createdAt: Date.now(),
      attachmentLabels: makeAttachmentLabels(
        selectedModel?.label || selectedModelKey,
        imageAsset,
        audioAsset,
      ),
      attachments: makeAttachmentSummary(imageAsset, audioAsset),
    }
    const pendingAssistantId = crypto.randomUUID()
    const pendingAssistantMessage = {
      id: pendingAssistantId,
      role: 'assistant',
      content: '',
      createdAt: Date.now(),
      streamingState: 'waiting',
    }

    startTransition(() => {
      setThreads((current) =>
        updateThreadInList(current, threadId, (thread) => ({
          ...thread,
          title:
            thread.messages.length === 0
              ? fallbackThreadTitle
              : thread.title,
          updatedAt: Date.now(),
          messages: [...thread.messages, pendingUserMessage, pendingAssistantMessage],
        })),
      )
    })

    const historyPayload = serializeHistoryMessages(messages)

    const formData = new FormData()
    formData.set('model_key', selectedModelKey)
    formData.set('quantization_key', selectedQuantizationKey)
    formData.set('prompt', promptText)
    formData.set('system_prompt', systemPrompt)
    formData.set('history_json', JSON.stringify(historyPayload))
    formData.set('max_new_tokens', String(settings.maxNewTokens))
    formData.set('temperature', String(settings.temperature))
    formData.set('top_p', String(settings.topP))
    formData.set('top_k', String(settings.topK))
    formData.set('thinking', String(settings.thinking))
    formData.set('tts_enabled', String(autoSpeak))

    if (imageAsset) {
      formData.set('image', imageAsset)
    }

    if (audioAsset) {
      formData.set('audio', audioAsset)
    }

    try {
      const response = await fetch('/api/generate-stream', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const data = await parseApiPayload(response)
        throw new Error(data.detail || 'Generation failed.')
      }

      let streamError = ''
      let streamedText = ''
      let finalPayload = null

      await readNdjsonStream(response, (event) => {
        if (event.event === 'start') {
          if (event.request_id) {
            startTransition(() => {
              setThreads((current) =>
                updateMessageInThread(current, threadId, pendingAssistantId, (message) => ({
                  ...message,
                  meta: {
                    ...(message.meta || {}),
                    request_id: event.request_id,
                    status_url: event.status_url,
                  },
                })),
              )
            })
          }
          return
        }

        if (event.event === 'token') {
          streamedText += event.text || ''
          startTransition(() => {
            setThreads((current) =>
              updateMessageInThread(current, threadId, pendingAssistantId, (message) => ({
                ...message,
                content: streamedText,
                streamingState: 'streaming',
              })),
            )
          })

          if (mode === 'live') {
            setLiveStatus('Receiving live reply...')
          }
          return
        }

        if (event.event === 'done') {
          finalPayload = event.payload
          return
        }

        if (event.event === 'error') {
          streamError = event.detail || 'Generation failed.'
        }
      })

      if (streamError) {
        throw new Error(streamError)
      }

      if (!finalPayload) {
        throw new Error('Streaming ended before the final payload arrived.')
      }

      if (mode === 'live' && finalPayload.tts_audio_error) {
        setLiveError(finalPayload.tts_audio_error)
      }

      let assistantMessage = {
        id: pendingAssistantId,
        role: 'assistant',
        content: finalPayload.reply || streamedText,
        thought: finalPayload.thought,
        meta: finalPayload,
        createdAt: Date.now(),
        streamingState: 'done',
      }

      startTransition(() => {
        setThreads((current) =>
          updateMessageInThread(current, threadId, pendingAssistantId, () => assistantMessage),
        )

        if (clearComposer) {
          setPrompt('')
          setImageFile(null)
          setAudioFile(null)
        }
      })

      if (mode === 'text' && settings.continuationMode === 'auto' && shouldContinuePayload(finalPayload)) {
        const autoContinuedMessage = await continueAssistantMessage({
          threadId,
          assistantMessageId: pendingAssistantId,
          baseMessage: assistantMessage,
          historyMessages: [...messages, pendingUserMessage, assistantMessage],
          depth: 1,
          auto: true,
          manageBusyState: false,
        })

        if (autoContinuedMessage) {
          assistantMessage = autoContinuedMessage
        }
      }

      if (shouldGenerateThreadTitle) {
        void generateThreadTitle({
          threadId,
          fallbackTitle: fallbackThreadTitle,
          userMessage: pendingUserMessage,
          assistantMessage,
        })
      }

      if (mode === 'live') {
        if (autoSpeak && finalPayload.tts_audio?.url) {
          const didSpeak = await playGeneratedAudio(
            finalPayload.tts_audio,
            playbackAudioRef,
            () => {
              setIsSpeaking(true)
              setLiveStatus('Playing local voice reply...')
            },
            () => {
              setIsSpeaking(false)
              setLiveStatus('Live session ready.')
            },
            () => {
              setIsSpeaking(false)
              setLiveStatus('Live reply ready.')
              setLiveError('Local voice playback failed.')
            },
          )

          if (!didSpeak) {
            setLiveStatus('Live reply ready.')
          }
        } else {
          setLiveStatus('Live reply ready.')
        }
      }

      await refreshModelsAndHealth()
      return assistantMessage
    } catch (submitError) {
      setError(submitError.message)

      if (mode === 'live') {
        setLiveError(submitError.message)
        setLiveStatus('Live session ready.')
      }

      startTransition(() => {
        setThreads((current) =>
          updateMessageInThread(current, threadId, pendingAssistantId, (message) => ({
            ...message,
            content: submitError.message,
            streamingState: 'error',
          })),
        )
      })

      return null
    } finally {
      setBusy(false)
    }
  }

  async function continueAssistantMessage({
    threadId,
    assistantMessageId,
    baseMessage = null,
    historyMessages = null,
    depth = 1,
    auto = false,
    manageBusyState = true,
  }) {
    const targetThread =
      threads.find((thread) => thread.id === threadId) ??
      (activeThread?.id === threadId ? activeThread : null)
    const sourceMessages = historyMessages ?? targetThread?.messages ?? []
    const existingMessage =
      baseMessage ??
      sourceMessages.find(
        (message) => message.id === assistantMessageId && message.role === 'assistant',
      ) ??
      null

    if (!existingMessage) {
      return null
    }

    const targetModelKey = existingMessage.meta?.active_model_key || selectedModelKey
    const targetQuantizationKey =
      existingMessage.meta?.active_quantization_key || selectedQuantizationKey

    if (isModelLoading || normalizedLoadState.is_loading) {
      setError('A model is still loading. Wait for it to finish before continuing.')
      return null
    }

    if (!health?.loaded) {
      setError('Nothing is loaded right now. Load the same model again before continuing.')
      return null
    }

    if (
      health.active_model_key !== targetModelKey ||
      health.active_quantization_key !== targetQuantizationKey
    ) {
      setError(
        `Continue needs ${existingMessage.meta?.active_model?.label || 'the original model'} in ${
          existingMessage.meta?.active_quantization?.label || 'its original quantization'
        }. Load it again first.`,
      )
      return null
    }

    if (manageBusyState) {
      setIsSending(true)
      setError('')
    }

    startTransition(() => {
      setThreads((current) =>
        updateMessageInThread(current, threadId, assistantMessageId, (message) => ({
          ...message,
          streamingState: 'waiting',
        })),
      )
    })

    const formData = new FormData()
    formData.set('model_key', targetModelKey)
    formData.set('quantization_key', targetQuantizationKey)
    formData.set('prompt', CONTINUATION_PROMPT)
    formData.set('system_prompt', systemPrompt)
    formData.set('history_json', JSON.stringify(serializeHistoryMessages(sourceMessages)))
    formData.set('max_new_tokens', String(settings.maxNewTokens))
    formData.set('temperature', String(settings.temperature))
    formData.set('top_p', String(settings.topP))
    formData.set('top_k', String(settings.topK))
    formData.set('thinking', String(settings.thinking))
    formData.set('tts_enabled', 'false')

    let streamedText = ''
    let finalPayload = null
    let streamError = ''
    const baseContent = existingMessage.content || ''

    try {
      const response = await fetch('/api/generate-stream', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const data = await parseApiPayload(response)
        throw new Error(data.detail || 'Continuation failed.')
      }

      await readNdjsonStream(response, (event) => {
        if (event.event === 'start') {
          if (event.request_id) {
            startTransition(() => {
              setThreads((current) =>
                updateMessageInThread(current, threadId, assistantMessageId, (message) => ({
                  ...message,
                  meta: {
                    ...(message.meta || {}),
                    request_id: event.request_id,
                    status_url: event.status_url,
                  },
                })),
              )
            })
          }
          return
        }

        if (event.event === 'token') {
          streamedText += event.text || ''
          startTransition(() => {
            setThreads((current) =>
              updateMessageInThread(current, threadId, assistantMessageId, (message) => ({
                ...message,
                content: `${baseContent}${streamedText}`,
                streamingState: 'streaming',
              })),
            )
          })
          return
        }

        if (event.event === 'done') {
          finalPayload = event.payload
          return
        }

        if (event.event === 'error') {
          streamError = event.detail || 'Continuation failed.'
        }
      })

      if (streamError) {
        throw new Error(streamError)
      }

      if (!finalPayload) {
        throw new Error('Continuation ended before the final payload arrived.')
      }

      const nextAssistantMessage = {
        ...existingMessage,
        content: `${baseContent}${finalPayload.reply || streamedText}`,
        thought: mergeThought(existingMessage.thought, finalPayload.thought),
        meta: {
          ...(existingMessage.meta || {}),
          ...finalPayload,
          tts_audio: null,
          continuation_count: Number(existingMessage.meta?.continuation_count || 0) + 1,
        },
        streamingState: 'done',
      }

      startTransition(() => {
        setThreads((current) =>
          updateMessageInThread(current, threadId, assistantMessageId, () => nextAssistantMessage),
        )
      })

      if (auto && shouldContinuePayload(finalPayload) && depth < AUTO_CONTINUE_LIMIT) {
        return continueAssistantMessage({
          threadId,
          assistantMessageId,
          baseMessage: nextAssistantMessage,
          historyMessages: sourceMessages.map((message) =>
            message.id === assistantMessageId ? nextAssistantMessage : message,
          ),
          depth: depth + 1,
          auto: true,
          manageBusyState: false,
        })
      }

      if (auto && shouldContinuePayload(finalPayload) && depth >= AUTO_CONTINUE_LIMIT) {
        setError(
          `Auto-continue stopped after ${AUTO_CONTINUE_LIMIT} passes. Click Continue to keep going if needed.`,
        )
      }

      await refreshModelsAndHealth()
      return nextAssistantMessage
    } catch (continuationError) {
      setError(continuationError.message)
      startTransition(() => {
        setThreads((current) =>
          updateMessageInThread(current, threadId, assistantMessageId, (message) => ({
            ...message,
            streamingState: 'done',
          })),
        )
      })
      return null
    } finally {
      if (manageBusyState) {
        setIsSending(false)
      }
    }
  }

  async function generateThreadTitle({
    threadId,
    fallbackTitle,
    userMessage,
    assistantMessage,
  }) {
    const modelKey = assistantMessage.meta?.active_model_key || selectedModelKey
    const quantizationKey =
      assistantMessage.meta?.active_quantization_key || selectedQuantizationKey

    const formData = new FormData()
    formData.set('model_key', modelKey)
    formData.set('quantization_key', quantizationKey)
    formData.set('prompt', TITLE_PROMPT)
    formData.set('system_prompt', TITLE_SYSTEM_PROMPT)
    formData.set(
      'history_json',
      JSON.stringify(
        serializeHistoryMessages([
          {
            role: userMessage.role,
            content: userMessage.content,
          },
          {
            role: assistantMessage.role,
            content: assistantMessage.content,
          },
        ]),
      ),
    )
    formData.set('max_new_tokens', '16')
    formData.set('temperature', '0.2')
    formData.set('top_p', '0.9')
    formData.set('top_k', '32')
    formData.set('thinking', 'false')
    formData.set('tts_enabled', 'false')

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        return
      }

      const data = await parseApiPayload(response)
      const nextTitle = normalizeGeneratedTitle(data.reply, fallbackTitle)
      if (!nextTitle) {
        return
      }

      startTransition(() => {
        setThreads((current) =>
          updateThreadInList(current, threadId, (thread) => ({
            ...thread,
            title: nextTitle,
          })),
        )
      })
    } catch {
      // Keep the fallback title if the background title pass fails.
    }
  }

  async function handleContinueTurn(messageId) {
    if (!activeThread) {
      return
    }

    await continueAssistantMessage({
      threadId: activeThread.id,
      assistantMessageId: messageId,
      auto: false,
      manageBusyState: true,
    })
  }

  async function teardownLiveRecorder() {
    try {
      liveAudioProcessorRef.current?.disconnect()
      liveAudioSourceRef.current?.disconnect()
      liveAudioSinkRef.current?.disconnect()
      liveAudioProcessorRef.current = null
      liveAudioSourceRef.current = null
      liveAudioSinkRef.current = null

      if (liveAudioContextRef.current) {
        await liveAudioContextRef.current.close()
      }
    } catch {
      // Best-effort cleanup for browser audio graph.
    } finally {
      liveAudioContextRef.current = null
    }
  }

  async function startLiveSession() {
    setLiveError('')

    if (!navigator.mediaDevices?.getUserMedia) {
      setLiveError('Camera and microphone access requires MediaDevices.getUserMedia().')
      return false
    }

    if (liveStreamRef.current) {
      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = liveStreamRef.current
        await liveVideoRef.current.play().catch(() => {})
      }

      setIsLiveSessionActive(true)
      setLiveStatus(
        supportsNativeLiveAudio
          ? 'Camera and microphone are live.'
          : supportsLiveVision
            ? 'Camera is live. This model accepts vision turns only.'
            : 'Camera preview is live. Load a model with vision support to send live turns.',
      )
      return true
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: supportsNativeLiveAudio,
      })

      liveStreamRef.current = stream

      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = stream
        liveVideoRef.current.muted = true
        liveVideoRef.current.playsInline = true
        await liveVideoRef.current.play().catch(() => {})
      }

      setIsLiveSessionActive(true)
      setLiveStatus(
        supportsNativeLiveAudio
          ? 'Camera and microphone are live.'
          : supportsLiveVision
            ? 'Camera is live. This model accepts vision turns only.'
            : 'Camera preview is live. Load a model with vision support to send live turns.',
      )
      return true
    } catch (sessionError) {
      setLiveError(
        sessionError.message || 'Unable to access the local camera or microphone.',
      )
      setIsLiveSessionActive(false)
      return false
    }
  }

  async function stopLiveSession() {
    if (isLiveRecording) {
      await stopLiveRecording({ submit: false })
    } else {
      await teardownLiveRecorder()
    }

    stopStreamTracks(liveStreamRef.current)
    liveStreamRef.current = null

    if (liveVideoRef.current) {
      liveVideoRef.current.pause()
      liveVideoRef.current.srcObject = null
    }

    if (document.fullscreenElement) {
      await document.exitFullscreen().catch(() => {})
    }

    stopAudioPlayback(playbackAudioRef)
    liveAudioChunksRef.current = []
    liveAudioLengthRef.current = 0
    setIsSpeaking(false)
    setIsLiveSessionActive(false)
    setIsLiveRecording(false)
    setIsLiveSubmitting(false)
    setLiveStatus('Live mode idle.')
    setLiveError('')
  }

  async function handleSendLiveFrame() {
    setLiveError('')

    if (!supportsLiveVision) {
      setLiveError(
        `${selectedModel?.label || 'This model'} does not accept live image turns in the current runtime.`,
      )
      return
    }

    const ready = await startLiveSession()
    if (!ready) {
      return
    }

    setLiveStatus('Capturing a live camera frame...')

    try {
      const imageCapture = await captureVideoFrame(liveVideoRef.current)
      const assistantMessage = await runTurn({
        promptText: livePrompt.trim() || 'Describe the current camera frame.',
        imageAsset: imageCapture,
        audioAsset: null,
        visibleText: 'Live camera frame turn.',
        titleHint: 'Live vision call',
        mode: 'live',
      })

      if (!assistantMessage) {
        setLiveStatus('Live reply ready.')
      }
    } catch (captureError) {
      setLiveError(captureError.message)
      setLiveStatus('Live session ready.')
    }
  }

  async function startLiveRecording() {
    setLiveError('')

    if (!supportsNativeLiveAudio) {
      setLiveError(
        `${selectedModel?.label || 'This model'} does not support native audio input. Switch to Gemma 4 E2B or E4B for voice turns.`,
      )
      return
    }

    if (
      liveStreamRef.current &&
      liveStreamRef.current.getAudioTracks().length === 0
    ) {
      stopStreamTracks(liveStreamRef.current)
      liveStreamRef.current = null
      setIsLiveSessionActive(false)
    }

    const ready = await startLiveSession()
    if (!ready) {
      return
    }

    const AudioContextClass = window.AudioContext || window.webkitAudioContext
    if (!AudioContextClass) {
      setLiveError('This browser does not expose the Web Audio API for live capture.')
      return
    }

    stopAudioPlayback(playbackAudioRef)
    setIsSpeaking(false)

    try {
      await teardownLiveRecorder()
      liveAudioChunksRef.current = []
      liveAudioLengthRef.current = 0

      const audioContext = new AudioContextClass()
      const sourceNode = audioContext.createMediaStreamSource(liveStreamRef.current)
      const processorNode = audioContext.createScriptProcessor(4096, 1, 1)
      const sinkNode = audioContext.createGain()
      sinkNode.gain.value = 0

      processorNode.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0)
        liveAudioChunksRef.current.push(new Float32Array(input))
        liveAudioLengthRef.current += input.length
      }

      sourceNode.connect(processorNode)
      processorNode.connect(sinkNode)
      sinkNode.connect(audioContext.destination)

      if (audioContext.state === 'suspended') {
        await audioContext.resume()
      }

      liveAudioContextRef.current = audioContext
      liveAudioSourceRef.current = sourceNode
      liveAudioProcessorRef.current = processorNode
      liveAudioSinkRef.current = sinkNode

      setIsLiveRecording(true)
      setLiveStatus('Recording live microphone turn...')
    } catch (recordError) {
      await teardownLiveRecorder()
      setLiveError(recordError.message || 'Unable to start live audio capture.')
    }
  }

  async function stopLiveRecording({ submit = true } = {}) {
    const sampleRate = liveAudioContextRef.current?.sampleRate || 16000

    setIsLiveRecording(false)
    await teardownLiveRecorder()

    if (!submit) {
      liveAudioChunksRef.current = []
      liveAudioLengthRef.current = 0
      setLiveStatus(isLiveSessionActive ? 'Live session ready.' : 'Live mode idle.')
      return
    }

    if (liveAudioLengthRef.current === 0) {
      setLiveError('No microphone audio was captured for this turn.')
      setLiveStatus('Live session ready.')
      return
    }

    setLiveStatus('Preparing live audio turn...')

    let imageCapture = null
    try {
      imageCapture = await captureVideoFrame(liveVideoRef.current)
    } catch (captureError) {
      setLiveError(captureError.message)
    }

    const mergedSamples = mergeAudioSamples(
      liveAudioChunksRef.current,
      liveAudioLengthRef.current,
    )
    const wavBlob = encodeWav(mergedSamples, sampleRate)
    const audioCapture = new File([wavBlob], `live-turn-${Date.now()}.wav`, {
      type: 'audio/wav',
    })

    liveAudioChunksRef.current = []
    liveAudioLengthRef.current = 0
    setLiveStatus('Sending live turn to Gemma...')

    const assistantMessage = await runTurn({
      promptText: livePrompt.trim() || DEFAULT_LIVE_PROMPT,
      imageAsset: imageCapture,
      audioAsset: audioCapture,
      visibleText: 'Live camera and microphone turn.',
      titleHint: 'Live vision call',
      mode: 'live',
    })

    if (!assistantMessage) {
      setLiveStatus('Live reply ready.')
    }
  }

  async function toggleLiveFullscreen() {
    if (!liveStageRef.current) {
      return
    }

    if (document.fullscreenElement) {
      await document.exitFullscreen().catch(() => {})
      return
    }

    if (!liveStageRef.current.requestFullscreen) {
      setLiveError('Fullscreen mode is unavailable in this browser context.')
      return
    }

    await liveStageRef.current.requestFullscreen().catch(() => {
      setLiveError('Fullscreen mode is unavailable in this browser context.')
    })
  }

  async function handleEnterLiveMode() {
    setViewMode('live')
    setIsModelMenuOpen(false)
    setIsQuantizationMenuOpen(false)
    setIsSettingsOpen(false)
    setLiveStatus('Starting live mode...')
    await startLiveSession()
  }

  async function handleLeaveLiveMode() {
    await stopLiveSession()
    setViewMode('text')
  }

  function handleStopSpeaking() {
    stopAudioPlayback(playbackAudioRef)
    setIsSpeaking(false)
    setLiveStatus('Live session ready.')
  }

  function handleCreateThread() {
    const nextThread =
      threads.length === 0
        ? createThread('New Chat')
        : createThread(`New Chat ${threads.length + 1}`)

    setError('')

    startTransition(() => {
      setThreads((current) => [nextThread, ...current])
      setActiveThreadId(nextThread.id)
      setPrompt('')
      setImageFile(null)
      setAudioFile(null)
    })
  }

  async function handleLoadModel() {
    if (!selectedModelKey || !selectedQuantizationKey) {
      return
    }

    setError('')
    setIsLoadingModel(true)
    setIsModelMenuOpen(false)
    setIsQuantizationMenuOpen(false)

    try {
      const response = await fetch('/api/models/load', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_key: selectedModelKey,
          quantization_key: selectedQuantizationKey,
        }),
      })
      const data = await parseApiPayload(response)

      if (!response.ok) {
        throw new Error(data.detail || 'Unable to load model.')
      }

      const nextLoadState = normalizeLoadState(data.load_state)

      startTransition(() => {
        setHealth(data.health)
        setModelLoadState(nextLoadState)
      })

      if (!nextLoadState.is_loading) {
        setIsLoadingModel(false)
      }

      if (nextLoadState.status === 'failed' && nextLoadState.error) {
        setError(nextLoadState.error)
      }
    } catch (loadError) {
      setError(loadError.message)
      setIsLoadingModel(false)
    }
  }

  async function submitComposerTurn() {
    const normalizedPrompt = makeFallbackPrompt(prompt, imageFile, audioFile)
    if (!normalizedPrompt) {
      setError('Add a prompt, image, or audio before sending.')
      return
    }

    return runTurn({
      promptText: normalizedPrompt,
      imageAsset: imageFile,
      audioAsset: audioFile,
      visibleText: normalizedPrompt,
      clearComposer: true,
    })
  }

  async function handleSubmit(event) {
    event.preventDefault()
    await submitComposerTurn()
  }

  function handleComposerKeyDown(event) {
    if (
      event.key !== 'Enter' ||
      event.shiftKey ||
      event.nativeEvent?.isComposing ||
      isSending
    ) {
      return
    }

    event.preventDefault()
    void submitComposerTurn()
  }

  return (
    <div className="app-shell">
      <div className="ambient-orb ambient-orb-primary" />
      <div className="ambient-orb ambient-orb-tertiary" />

      <aside className="history-rail">
        <div className="history-head">
          <div>
            <p className="rail-kicker">History</p>
            <h2>Recent threads</h2>
          </div>

          <button className="new-chat-button" type="button" onClick={handleCreateThread}>
            <span className="material-symbols-outlined">add</span>
            <span>New Chat</span>
          </button>
        </div>

        <nav className="thread-nav" aria-label="Recent conversations">
          {threads.map((thread) => (
            <button
              key={thread.id}
              className={`thread-entry ${
                thread.id === activeThreadId ? 'is-active' : ''
              }`}
              type="button"
              onClick={() => setActiveThreadId(thread.id)}
            >
              <span className="material-symbols-outlined">chat_bubble_outline</span>
              <span className="thread-copy">
                <strong>{thread.title}</strong>
                <span>{getThreadSubtitle(thread)}</span>
              </span>
            </button>
          ))}
        </nav>

        <div className="history-footer">
          <button
            className="utility-link"
            type="button"
            onClick={() => setViewMode('monitoring')}
          >
            <span className="material-symbols-outlined">monitoring</span>
            <span>Monitoring</span>
          </button>

          <button
            className="utility-link"
            type="button"
            onClick={() => setIsSettingsOpen(true)}
          >
            <span className="material-symbols-outlined">tune</span>
            <span>Settings</span>
          </button>

          <div className="profile-card">
            <div className="profile-badge">AA</div>
            <div>
              <strong>Local rig</strong>
              <span>{health?.gpu || 'GPU offline'}</span>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-canvas">
        <header className="topbar">
          <div className="topbar-brand">
            <div>
              <p className="eyebrow">Gemma 4</p>
              <h1>Intelligent interface</h1>
            </div>

            <div
              className={`status-pill ${
                isModelLoading
                  ? 'is-loading'
                  : health?.loaded
                    ? 'is-ready'
                    : 'is-cold'
              }`}
            >
              <span className="status-dot" />
              <span>{loadStatusLabel}</span>
            </div>
          </div>

          <div className="topbar-actions">
            <div className="mode-switch" role="tablist" aria-label="Interface mode">
              <button
                className={`mode-switch-button ${
                  viewMode === 'text' ? 'is-active' : ''
                }`}
                type="button"
                onClick={async () => {
                  if (viewMode === 'live') {
                    await handleLeaveLiveMode()
                    return
                  }

                  setViewMode('text')
                }}
              >
                <span className="material-symbols-outlined">chat</span>
                <span>Text studio</span>
              </button>

              <button
                className={`mode-switch-button ${
                  viewMode === 'live' ? 'is-active' : ''
                }`}
                type="button"
                onClick={handleEnterLiveMode}
              >
                <span className="material-symbols-outlined">videocam</span>
                <span>Video call</span>
              </button>

              <button
                className={`mode-switch-button ${
                  viewMode === 'monitoring' ? 'is-active' : ''
                }`}
                type="button"
                onClick={() => setViewMode('monitoring')}
              >
                <span className="material-symbols-outlined">monitoring</span>
                <span>Monitoring</span>
              </button>
            </div>

            <div className="topbar-load-group">
              <div className="model-picker" ref={modelPickerRef}>
                <button
                  className="model-trigger"
                  type="button"
                  onClick={() => {
                    setIsQuantizationMenuOpen(false)
                    setIsModelMenuOpen((current) => !current)
                  }}
                >
                  <span className="model-trigger-copy">
                    <span className="model-trigger-label">Model</span>
                    <strong>{selectedModel?.label || 'Gemma 4'}</strong>
                  </span>
                  <span className="material-symbols-outlined">expand_more</span>
                </button>

                {isModelMenuOpen ? (
                  <div className="model-menu">
                    <div className="model-menu-head">
                      <div>
                        <p className="section-label">Variant selector</p>
                        <h2>Gemma 4 lineup</h2>
                      </div>
                    </div>

                    <div className="model-menu-list">
                      {models.map((model) => {
                        const isSelected = model.key === selectedModelKey
                        const isLoaded = activeModel?.key === model.key && health?.loaded
                        const isInfoOpen = openModelInfoKey === model.key

                        return (
                          <div
                            key={model.key}
                            className={`model-menu-item ${
                              isSelected ? 'is-selected' : ''
                            } ${isLoaded ? 'is-loaded' : ''}`}
                          >
                            <div className="model-menu-item-head">
                              <button
                                className="model-select-button"
                                type="button"
                                onClick={() => {
                                  setSelectedModelKey(model.key)
                                  setIsModelMenuOpen(false)
                                }}
                              >
                                <div className="model-menu-row">
                                  <strong>{model.label}</strong>
                                  <span>{isLoaded ? 'Loaded' : model.tier}</span>
                                </div>
                                <span className="model-menu-meta-line">
                                  {model.parameter_summary}
                                </span>
                              </button>

                              <button
                                className={`model-info-button ${
                                  isInfoOpen ? 'is-open' : ''
                                }`}
                                type="button"
                                aria-label={`More info about ${model.label}`}
                                aria-expanded={isInfoOpen}
                                onClick={(event) => {
                                  event.stopPropagation()
                                  setOpenModelInfoKey((current) =>
                                    current === model.key ? null : model.key,
                                  )
                                }}
                              >
                                ?
                              </button>
                            </div>

                            {isInfoOpen ? (
                              <div className="model-info-popover">
                                <p>{model.doc_summary}</p>
                                <div className="message-chip-row">
                                  <span className="message-chip">
                                    {model.architecture}
                                  </span>
                                  <span className="message-chip">
                                    {model.context_length}
                                  </span>
                                  {model.supported_modalities.map((modality) => (
                                    <span
                                      key={`${model.key}-${modality}`}
                                      className="message-chip is-modality"
                                    >
                                      {modality}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            ) : null}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="model-picker" ref={quantizationPickerRef}>
                <button
                  className="model-trigger"
                  type="button"
                  onClick={() => {
                    setIsModelMenuOpen(false)
                    setIsQuantizationMenuOpen((current) => !current)
                  }}
                >
                  <span className="model-trigger-copy">
                    <span className="model-trigger-label">Quantization</span>
                    <strong>{selectedQuantization?.label || 'BF16'}</strong>
                  </span>
                  <span className="material-symbols-outlined">expand_more</span>
                </button>

                {isQuantizationMenuOpen ? (
                  <div className="model-menu quantization-menu">
                    <div className="model-menu-head">
                      <div>
                        <p className="section-label">Precision selector</p>
                        <h2>Gemma 4 quantization</h2>
                      </div>
                    </div>

                    <div className="model-menu-list">
                      {quantizations.map((quantization) => {
                        const isSelected =
                          quantization.key === selectedQuantizationKey
                        const isLoaded =
                          health?.loaded &&
                          activeModel?.key === selectedModelKey &&
                          activeQuantization?.key === quantization.key
                        const memoryEstimate = getModelMemoryEstimate(
                          selectedModel,
                          quantization.key,
                        )

                        return (
                          <button
                            key={quantization.key}
                            className={`model-menu-item ${
                              isSelected ? 'is-selected' : ''
                            } ${isLoaded ? 'is-loaded' : ''}`}
                            type="button"
                            onClick={() => {
                              setSelectedQuantizationKey(quantization.key)
                              setIsQuantizationMenuOpen(false)
                            }}
                          >
                            <div className="model-menu-row">
                              <strong>{quantization.label}</strong>
                              <span>
                                {isLoaded
                                  ? 'Loaded'
                                  : getQuantizationRuntimeLabel(quantization)}
                              </span>
                            </div>
                            <p>{quantization.doc_summary}</p>
                            <div className="message-chip-row">
                              <span className="message-chip">
                                {quantization.precision_bits}-bit
                              </span>
                              {memoryEstimate ? (
                                <span className="message-chip">
                                  Google est. {formatMemory(memoryEstimate)}
                                </span>
                              ) : null}
                              <span
                                className={`message-chip ${
                                  quantization.runtime_supported ? 'is-modality' : ''
                                }`}
                              >
                                {getQuantizationRuntimeLabel(quantization)}
                              </span>
                            </div>
                          </button>
                        )
                      })}
                    </div>
                  </div>
                ) : null}
              </div>

              <button
                className="topbar-load-button"
                type="button"
                onClick={handleLoadModel}
                disabled={
                  isModelLoading ||
                  isSending ||
                  isLiveSubmitting ||
                  isLiveRecording ||
                  !quantizationRuntimeSupported ||
                  !selectedModelKey ||
                  !selectedQuantizationKey
                }
              >
                <span className="material-symbols-outlined">
                  {isModelLoading ? 'hourglass_top' : 'memory'}
                </span>
                <span>
                  {isModelLoading
                    ? 'Loading...'
                    : quantizationRuntimeSupported
                      ? 'Load model'
                      : 'Planning only'}
                </span>
              </button>
            </div>
          </div>
        </header>

        {shouldShowLoadPanel ? (
          <section className={`load-progress-panel is-${loadPanelState}`}>
            <div className="load-progress-head">
              <div>
                <p className="section-label">Model loader</p>
                <h2>{loadPanelHeadline}</h2>
              </div>

              <div className="load-progress-stats">
                <strong>{Math.round(loadProgressValue)}%</strong>
                <span>{loadStatusMeta}</span>
              </div>
            </div>

            <div className="load-progress-track" aria-hidden="true">
              <div
                className="load-progress-fill"
                style={{ width: `${Math.max(loadProgressValue, loadPanelState === 'failed' ? 6 : 0)}%` }}
              />
            </div>

            <div className="load-progress-meta">
              <span>{loadPanelMessage}</span>
              <span>
                Google est. {formatMemory(selectedMemoryEstimate)} |{' '}
                {quantizationRuntimeSupported ? 'Runnable here' : 'Planning only'}
              </span>
            </div>
          </section>
        ) : null}

        {viewMode === 'text' ? (
          <>
        <section
          className={`conversation-stage ${shouldShowLoadPanel ? 'has-load-progress' : ''}`}
        >
          {isSending ? <div className="stage-shimmer" /> : null}

          <div className="conversation-scroll" ref={logRef}>
            <div className="mobile-rail">
              <button className="new-chat-button" type="button" onClick={handleCreateThread}>
                <span className="material-symbols-outlined">add</span>
                <span>New Chat</span>
              </button>

              <div className="mobile-thread-strip">
                {threads.map((thread) => (
                  <button
                    key={`mobile-${thread.id}`}
                    className={`mobile-thread-chip ${
                      thread.id === activeThreadId ? 'is-active' : ''
                    }`}
                    type="button"
                    onClick={() => setActiveThreadId(thread.id)}
                  >
                    {thread.title}
                  </button>
                ))}
              </div>
            </div>

            {messages.length === 0 ? (
              <section className="editorial-hero">
                <div className="hero-copy">
                  <p className="eyebrow">The Ethereal Intelligence</p>
                  <h2>
                    {greeting}, <span className="gradient-text">Gemma 4.</span>
                  </h2>
                  <p>
                    Prompt text, image, and audio from a single local canvas.
                    {isLlamaCppRuntime
                      ? ` ${selectedQuantization?.label || 'This quantization'} is currently routed through llama.cpp for fast local text chat in this app build.`
                      : isWslVllmRuntime
                      ? ` ${selectedModel?.label || 'This variant'} is currently routed through the experimental WSL vLLM bridge for Blackwell. Text and image are enabled here; video stays outside the current UI.`
                      : supportsAudio
                      ? ` ${selectedModel?.label || 'This variant'} is currently unlocked for text, image, and audio.`
                      : ` ${selectedModel?.label || 'This variant'} is currently scoped to text and image according to the official modality tables.`}
                  </p>

                  <div className="prompt-pill-row">
                    {PROMPT_PRESETS.map((preset) => (
                      <button
                        key={preset.title}
                        className="prompt-pill"
                        type="button"
                        onClick={() => setPrompt(preset.prompt)}
                      >
                        <span className="material-symbols-outlined">{preset.icon}</span>
                        <span>{preset.title}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </section>
            ) : null}

            <section className="conversation-thread">
              {messages.length > 0 ? (
                <div className="message-flow">
                  {messages.map((message) => (
                    <article
                      key={message.id}
                      className={`thread-message thread-message-${message.role}`}
                    >
                      {message.role === 'assistant' ? (
                        <div className="message-brand">
                          <div className="message-brand-icon">
                            <span className="material-symbols-outlined">auto_awesome</span>
                          </div>
                          <span>Gemma 4 analysis</span>
                        </div>
                      ) : null}

                      <div
                        className={`message-shell ${
                          message.role === 'user' ? 'user-bubble' : 'assistant-panel'
                        } ${
                          message.streamingState === 'waiting' ||
                          message.streamingState === 'streaming'
                            ? 'is-streaming'
                            : ''
                        }`}
                      >
                        <div className="message-header">
                          <span className="message-role">
                            {message.role === 'user' ? 'You' : 'Gemma 4'}
                          </span>
                          <span className="message-time">
                            {message.streamingState === 'waiting'
                              ? 'Preparing'
                              : message.streamingState === 'streaming'
                                ? 'Streaming'
                                : formatMessageTime(message.createdAt)}
                          </span>
                        </div>

                        {message.attachmentLabels?.length ? (
                          <div className="message-chip-row">
                            {message.attachmentLabels.map((label) => (
                              <span key={`${message.id}-${label}`} className="message-chip">
                                {label}
                              </span>
                            ))}
                          </div>
                        ) : null}

                        <div className="message-copy">
                          {message.role === 'assistant' &&
                          (message.streamingState === 'waiting' ||
                            message.streamingState === 'streaming') ? (
                            <div className="stream-status">
                              <span className="stream-spinner" aria-hidden="true" />
                              <span>
                                {message.streamingState === 'waiting'
                                  ? 'Model is preparing the first tokens...'
                                  : 'Reply is streaming live.'}
                              </span>
                            </div>
                          ) : null}
                          {message.content ? (
                            <MarkdownBlock
                              content={message.content}
                              variant={message.role === 'assistant' ? 'assistant' : 'user'}
                            />
                          ) : null}
                        </div>

                        {message.thought ? (
                          <details className="thought-card">
                            <summary>Thinking trace</summary>
                            <MarkdownBlock content={message.thought} variant="thought" />
                          </details>
                        ) : null}

                        {message.meta?.tts_audio?.url ? (
                          <div className="assistant-audio-card">
                            <div className="assistant-audio-meta">
                              <span className="material-symbols-outlined">graphic_eq</span>
                              <span>
                                Local voice | {message.meta.tts_audio.voice || 'Piper'}
                              </span>
                            </div>
                            <VoicePlayer audioPayload={message.meta.tts_audio} />
                          </div>
                        ) : null}

                        {message.meta ? (
                          <div className="message-chip-row">
                            {getAssistantChips(message).map((chip) => (
                              <span key={`${message.id}-${chip}`} className="message-chip">
                                {chip}
                              </span>
                            ))}
                          </div>
                        ) : null}

                        {message.role === 'assistant' &&
                        settings.continuationMode !== 'off' &&
                        shouldContinuePayload(message.meta) ? (
                          <div className="message-action-row">
                            <button
                              className="message-action-button"
                              type="button"
                              onClick={() => void handleContinueTurn(message.id)}
                              disabled={isSending || isModelLoading}
                            >
                              <span className="material-symbols-outlined">resume</span>
                              <span>Continue</span>
                            </button>
                          </div>
                        ) : null}
                      </div>
                    </article>
                  ))}
                </div>
              ) : null}

            </section>
          </div>
        </section>

        <footer className="command-footer">
          <form className="command-inner" onSubmit={handleSubmit}>
            {imagePreview || audioPreview ? (
              <div className="composer-previews">
                {imagePreview ? (
                  <div className="media-preview">
                    <div className="preview-meta">
                      <div>
                        <strong>{imageFile?.name || 'Image'}</strong>
                        <span>Image input queued</span>
                      </div>
                      <button
                        className="preview-remove"
                        type="button"
                        onClick={() => setImageFile(null)}
                      >
                        Remove
                      </button>
                    </div>
                    <img src={imagePreview} alt="Selected image preview" />
                  </div>
                ) : null}

                {audioPreview ? (
                  <div className="media-preview">
                    <div className="preview-meta">
                      <div>
                        <strong>{audioFile?.name || 'Audio'}</strong>
                        <span>Audio input queued</span>
                      </div>
                      <button
                        className="preview-remove"
                        type="button"
                        onClick={() => setAudioFile(null)}
                      >
                        Remove
                      </button>
                    </div>
                    <audio controls src={audioPreview} />
                  </div>
                ) : null}
              </div>
            ) : null}

            <div className="glass-dock">
              <div className="dock-leading">
                <button
                  className="dock-button"
                  type="button"
                  onClick={() => imageInputRef.current?.click()}
                  disabled={!supportsImage || isSending || isModelLoading}
                >
                  <span className="material-symbols-outlined">image</span>
                </button>

                <button
                  className="dock-button"
                  type="button"
                  onClick={() => audioInputRef.current?.click()}
                  disabled={!supportsAudio || isSending || isModelLoading}
                  title={audioAttachmentHint}
                >
                  <span className="material-symbols-outlined">mic</span>
                </button>
              </div>

              <label className="composer-field">
                <textarea
                  className="composer-textarea"
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  onKeyDown={handleComposerKeyDown}
                  placeholder="Message Gemma 4..."
                  rows={1}
                />
              </label>

              <div className="dock-actions">
                <button
                  className={`dock-toggle ${settings.thinking ? 'is-active' : ''}`}
                  type="button"
                  onClick={() =>
                    setSettings((current) => ({
                      ...current,
                      thinking: !current.thinking,
                    }))
                  }
                >
                  Thinking
                </button>

                <button
                  className={`dock-toggle ${autoSpeak ? 'is-active' : ''}`}
                  type="button"
                  onClick={() => setAutoSpeak((current) => !current)}
                >
                  {autoSpeak ? 'Local voice on' : 'Local voice off'}
                </button>

                <button
                  className="send-button"
                  type="submit"
                  disabled={isSending || !canSendTurns}
                >
                  {isSending ? (
                    <span className="button-spinner" aria-hidden="true" />
                  ) : (
                    <span className="material-symbols-outlined">arrow_upward</span>
                  )}
                </button>
              </div>
            </div>

            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
            />
            <input
              ref={audioInputRef}
              type="file"
              accept="audio/*,.wav,.mp3,.m4a,.flac"
              hidden
              onChange={(event) => setAudioFile(event.target.files?.[0] ?? null)}
            />

            <div className="dock-meta">
              <span>
                {selectedModel?.label || 'Gemma 4'} |{' '}
                {selectedQuantization?.label || 'BF16'} |{' '}
                {modalityBadges.length > 0 ? modalityBadges.join(' + ') : 'text'} |{' '}
                {activeThreadSubtitle}
              </span>
              {error ? (
                <strong>{error}</strong>
              ) : (
                <span>Gemma 4 can make mistakes. Check important info.</span>
              )}
            </div>
          </form>
        </footer>
          </>
        ) : viewMode === 'monitoring' ? (
          <section
            className={`monitoring-stage ${shouldShowLoadPanel ? 'has-load-progress' : ''}`}
          >
            <div className="monitoring-scroll">
              <section className="editorial-hero monitoring-hero">
                <div className="hero-copy">
                  <p className="eyebrow">Operations</p>
                  <h2>
                    Local rig, <span className="gradient-text">in plain sight.</span>
                  </h2>
                  <p>
                    Watch the loaded model, GPU pressure, RAM usage, queue depth,
                    and recent inference requests from one place. Every inference
                    call now gets a persistent request ID backed by SQLite.
                  </p>
                </div>

                <div className="hero-panel">
                  <div className="hero-panel-head">
                    <div>
                      <p className="section-label">Live status</p>
                      <h3>{monitoringHealth?.loaded ? 'Model warm' : 'Waiting for load'}</h3>
                    </div>

                    <button
                      className="ghost-action"
                      type="button"
                      onClick={() => void refreshMonitoring()}
                      disabled={isRefreshingMonitoring}
                    >
                      <span className="material-symbols-outlined">sync</span>
                      <span>{isRefreshingMonitoring ? 'Refreshing' : 'Refresh'}</span>
                    </button>
                  </div>

                  <div className="hero-stat-grid">
                    <div className="hero-stat">
                      <span>Current model</span>
                      <strong>{monitoringCurrentModel?.label || activeModel?.label || 'None loaded'}</strong>
                    </div>
                    <div className="hero-stat">
                      <span>Runtime</span>
                      <strong>{monitoringHealth?.runtime_family || 'idle'}</strong>
                    </div>
                    <div className="hero-stat">
                      <span>Queue depth</span>
                      <strong>{monitoringQueue?.queued_count ?? 0}</strong>
                    </div>
                    <div className="hero-stat">
                      <span>SQLite</span>
                      <strong>{monitoringSnapshot?.database?.path ? 'Tracking on' : 'Unavailable'}</strong>
                    </div>
                  </div>
                </div>
              </section>

              <section className="monitor-grid">
                <article className="monitor-panel">
                  <div className="monitor-panel-head">
                    <div>
                      <p className="section-label">Health checks</p>
                      <h3>Machine state</h3>
                    </div>
                    <span className={`message-chip ${monitoringHealth?.loaded ? 'is-modality' : ''}`}>
                      {monitoringHealth?.status || 'offline'}
                    </span>
                  </div>

                  <div className="metric-grid">
                    <div className="metric-card">
                      <span>GPU load</span>
                      <strong>{formatPercent(monitoringGpu?.utilization_gpu_percent)}</strong>
                    </div>
                    <div className="metric-card">
                      <span>GPU memory</span>
                      <strong>
                        {formatMemory(monitoringGpu?.memory_used_gib)} /{' '}
                        {formatMemory(monitoringGpu?.memory_total_gib)}
                      </strong>
                    </div>
                    <div className="metric-card">
                      <span>System RAM used</span>
                      <strong>
                        {formatMemory(monitoringSystemRamUsed)} /{' '}
                        {formatMemory(monitoringMemory?.total_physical_gib)}
                      </strong>
                    </div>
                    <div className="metric-card">
                      <span>Free commit</span>
                      <strong>{formatMemory(monitoringMemory?.available_commit_gib)}</strong>
                    </div>
                    <div className="metric-card">
                      <span>GPU temp</span>
                      <strong>{formatNumber(monitoringGpu?.temperature_c, ' C')}</strong>
                    </div>
                    <div className="metric-card">
                      <span>GPU power</span>
                      <strong>{formatNumber(monitoringGpu?.power_draw_watts, ' W')}</strong>
                    </div>
                  </div>

                  {monitoringError ? (
                    <p className="live-error">{monitoringError}</p>
                  ) : null}
                </article>

                <article className="monitor-panel">
                  <div className="monitor-panel-head">
                    <div>
                      <p className="section-label">Inference queue</p>
                      <h3>Request pipeline</h3>
                    </div>
                    <span className="message-chip">
                      {monitoringQueue?.active_request ? 'Active request' : 'Idle'}
                    </span>
                  </div>

                  {monitoringQueue?.active_request ? (
                    <div className="monitor-request-card">
                      <div className="monitor-request-head">
                        <strong>{monitoringQueue.active_request.request_id}</strong>
                        <span className={`monitor-status-pill ${getRequestStatusTone(monitoringQueue.active_request.status)}`}>
                          {monitoringQueue.active_request.status}
                        </span>
                      </div>
                      <div className="monitor-request-meta">
                        <span>{summarizeRouteLabel(monitoringQueue.active_request.route)}</span>
                        <span>
                          {monitoringQueue.active_request.model_key} /{' '}
                          {monitoringQueue.active_request.quantization_key}
                        </span>
                        <span>{monitoringQueue.active_request.progress_message || 'Running'}</span>
                      </div>
                      {monitoringQueue.active_request.request_payload?.prompt_preview ? (
                        <p className="muted-copy">
                          {monitoringQueue.active_request.request_payload.prompt_preview}
                        </p>
                      ) : null}
                    </div>
                  ) : (
                    <p className="muted-copy">
                      No active inference request right now.
                    </p>
                  )}

                  {monitoringQueue?.queued_requests?.length ? (
                    <div className="monitor-queue-list">
                      {monitoringQueue.queued_requests.map((queuedRequest) => (
                        <div key={queuedRequest.request_id} className="monitor-queue-item">
                          <strong>{queuedRequest.request_id}</strong>
                          <span>
                            #{queuedRequest.queue_position || '?'} ·{' '}
                            {queuedRequest.model_key || 'model'} /{' '}
                            {queuedRequest.quantization_key || 'quant'}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="muted-copy">The queue is empty.</p>
                  )}
                </article>
              </section>

              <section className="monitor-panel monitor-panel-wide">
                <div className="monitor-panel-head">
                  <div>
                    <p className="section-label">Recent requests</p>
                    <h3>Inference history</h3>
                  </div>
                  <span className="message-chip">
                    {monitoringRecentRequests.length} tracked
                  </span>
                </div>

                {monitoringRecentRequests.length > 0 ? (
                  <div className="monitor-request-list">
                    {monitoringRecentRequests.map((requestRow) => (
                      <article key={requestRow.request_id} className="monitor-history-item">
                        <div className="monitor-request-head">
                          <strong>{requestRow.request_id}</strong>
                          <span className={`monitor-status-pill ${getRequestStatusTone(requestRow.status)}`}>
                            {requestRow.status}
                          </span>
                        </div>
                        <div className="monitor-request-meta">
                          <span>{summarizeRouteLabel(requestRow.route)}</span>
                          <span>
                            {requestRow.model_key || 'model'} /{' '}
                            {requestRow.quantization_key || 'quant'}
                          </span>
                          <span>{formatElapsed(requestRow.elapsed_ms)}</span>
                          <span>{formatDateTime(requestRow.created_at)}</span>
                        </div>
                        {requestRow.progress_message ? (
                          <p className="muted-copy">{requestRow.progress_message}</p>
                        ) : null}
                        {requestRow.response_preview ? (
                          <p className="monitor-response-preview">
                            {requestRow.response_preview}
                          </p>
                        ) : null}
                        {requestRow.error_text ? (
                          <p className="live-error">{requestRow.error_text}</p>
                        ) : null}
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="muted-copy">
                    No inference request has been stored in SQLite yet.
                  </p>
                )}
              </section>
            </div>
          </section>
        ) : (
          <section
            className={`live-stage ${shouldShowLoadPanel ? 'has-load-progress' : ''}`}
            ref={liveStageRef}
          >
            {isLiveSubmitting || isLiveRecording ? (
              <div className="stage-shimmer live-stage-shimmer" />
            ) : null}

            <div className="live-stage-inner">
              <div className="live-video-panel">
                <video
                  ref={liveVideoRef}
                  className="live-video"
                  autoPlay
                  muted
                  playsInline
                />

                {!isLiveSessionActive ? (
                  <div className="live-video-overlay">
                    <p className="eyebrow">Live vision call</p>
                    <h2>Camera, microphone, and Gemma on one stage.</h2>
                    <p>
                      Start a local live session, preview the camera feed, and
                      send the current frame or a microphone turn when the loaded
                      model supports it.
                    </p>
                    <div className="live-overlay-actions">
                      <button
                        className="new-chat-button"
                        type="button"
                        onClick={startLiveSession}
                      >
                        <span className="material-symbols-outlined">videocam</span>
                        <span>Start camera</span>
                      </button>
                      <button
                        className="ghost-action"
                        type="button"
                        onClick={handleLeaveLiveMode}
                      >
                        Back to text
                      </button>
                    </div>
                  </div>
                ) : null}

                <div className="live-video-hud">
                  <div className="live-pill">
                    <span className="material-symbols-outlined">radio_button_checked</span>
                    <span>{isLiveRecording ? 'Recording' : 'Live'}</span>
                  </div>
                  <div className="live-pill">
                    <span className="material-symbols-outlined">memory</span>
                    <span>
                      {selectedModel?.label || 'Gemma 4'} /{' '}
                      {selectedQuantization?.label || 'BF16'}
                    </span>
                  </div>
                  <div className="live-pill">
                    <span className="material-symbols-outlined">image</span>
                    <span>{supportsLiveVision ? 'Vision enabled' : 'Vision disabled'}</span>
                  </div>
                  <div className="live-pill">
                    <span className="material-symbols-outlined">graphic_eq</span>
                    <span>{supportsNativeLiveAudio ? 'Voice in enabled' : 'Voice in disabled'}</span>
                  </div>
                </div>
              </div>

              <aside className="live-control-panel">
                <div className="live-panel-head">
                  <div>
                    <p className="section-label">Call mode</p>
                    <h2>Realtime camera loop</h2>
                  </div>

                  <button
                    className="topbar-icon"
                    type="button"
                    onClick={toggleLiveFullscreen}
                  >
                    <span className="material-symbols-outlined">
                      {isLiveFullscreen ? 'fullscreen_exit' : 'fullscreen'}
                    </span>
                  </button>
                </div>

                <div className="live-chip-row">
                  <span className="message-chip">
                    {liveModeCapabilityLabel}
                  </span>
                  <span className="message-chip">
                    {autoSpeak ? 'Local TTS on' : 'Local TTS off'}
                  </span>
                  <span className="message-chip">
                    {isSpeaking ? 'Speaking reply' : liveStatus}
                  </span>
                </div>

                <div
                  className={`live-caption-card ${
                    latestAssistantTurn?.streamingState === 'streaming' ? 'is-streaming' : ''
                  } ${
                    latestAssistantTurn?.streamingState === 'waiting' ? 'is-waiting' : ''
                  }`}
                >
                  <div className="live-caption-head">
                    <span>Live transcript</span>
                    <span>{liveTranscriptStatus}</span>
                  </div>
                  <p>{liveTranscriptText}</p>
                </div>

                <label className="field live-prompt-field">
                  <span>Live instruction</span>
                  <textarea
                    value={livePrompt}
                    onChange={(event) => setLivePrompt(event.target.value)}
                    rows={4}
                  />
                </label>

                <div className="live-control-grid">
                  <button
                    className={`live-control-button ${
                      isLiveSessionActive ? 'is-active' : ''
                    }`}
                    type="button"
                    onClick={isLiveSessionActive ? stopLiveSession : startLiveSession}
                  >
                    <span className="material-symbols-outlined">
                      {isLiveSessionActive ? 'videocam_off' : 'videocam'}
                    </span>
                    <span>{isLiveSessionActive ? 'Stop camera' : 'Start camera'}</span>
                  </button>

                  <button
                    className={`live-control-button ${
                      isLiveRecording ? 'is-recording' : ''
                    }`}
                    type="button"
                    onClick={
                      isLiveRecording
                        ? () => stopLiveRecording({ submit: true })
                        : startLiveRecording
                    }
                    disabled={
                      isLiveSubmitting ||
                      isModelLoading ||
                      !supportsNativeLiveAudio
                      || !selectedModelLoaded
                    }
                  >
                    <span className="material-symbols-outlined">
                      {isLiveRecording ? 'stop_circle' : 'mic'}
                    </span>
                    <span>{isLiveRecording ? 'Send voice turn' : 'Record voice turn'}</span>
                  </button>

                  <button
                    className="live-control-button"
                    type="button"
                    onClick={handleSendLiveFrame}
                    disabled={
                      isLiveSubmitting ||
                      isModelLoading ||
                      !selectedModelLoaded ||
                      !supportsLiveVision
                    }
                  >
                    <span className="material-symbols-outlined">photo_camera</span>
                    <span>Send camera frame</span>
                  </button>

                  <button
                    className={`live-control-button ${autoSpeak ? 'is-active' : ''}`}
                    type="button"
                    onClick={() => setAutoSpeak((current) => !current)}
                  >
                    <span className="material-symbols-outlined">
                      {autoSpeak ? 'volume_up' : 'volume_off'}
                    </span>
                    <span>{autoSpeak ? 'Local voice on' : 'Local voice off'}</span>
                  </button>
                </div>

                {isSpeaking ? (
                  <button
                    className="live-secondary-button"
                    type="button"
                    onClick={handleStopSpeaking}
                  >
                    Stop voice output
                  </button>
                ) : null}

                <div className="live-note-stack">
                  <p className="hero-note">
                    {liveModeCapabilityNote} Spoken replies still come from the
                    local Piper backend, because the current Gemma checkpoints
                    return text, not native audio.
                  </p>
                  {liveError ? <p className="live-error">{liveError}</p> : null}
                </div>

                <div className="live-transcript-panel">
                  <div className="live-panel-head">
                    <div>
                      <p className="section-label">Recent turns</p>
                      <h3>{activeThread?.title || 'Live vision call'}</h3>
                    </div>
                    <span className="message-chip">
                      {isLiveSubmitting ? 'Gemma running' : 'Ready'}
                    </span>
                  </div>

                  {latestTurns.length > 0 ? (
                    <div className="live-turn-list">
                      {latestTurns.map((message) => (
                        <article
                          key={`live-${message.id}`}
                          className={`live-turn live-turn-${message.role}`}
                        >
                          <div className="live-turn-head">
                            <strong>{message.role === 'user' ? 'You' : 'Gemma 4'}</strong>
                            {message.role === 'assistant' &&
                            (message.streamingState === 'waiting' ||
                              message.streamingState === 'streaming') ? (
                              <span className="live-turn-status">
                                <span className="stream-spinner" aria-hidden="true" />
                                <span>
                                  {message.streamingState === 'waiting'
                                    ? 'Preparing'
                                    : 'Streaming'}
                                </span>
                              </span>
                            ) : null}
                          </div>
                          <p>
                            {message.content ||
                              (message.role === 'assistant'
                                ? 'Gemma is preparing the reply...'
                                : '')}
                          </p>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <p className="muted-copy">
                      Start a live turn to see the rolling camera conversation here.
                    </p>
                  )}
                </div>
              </aside>
            </div>
          </section>
        )}
      </main>

      {isSettingsOpen ? (
        <>
          <button
            className="settings-scrim"
            type="button"
            aria-label="Close settings"
            onClick={() => setIsSettingsOpen(false)}
          />

          <aside className="settings-panel">
            <div className="settings-header">
              <div>
                <p className="section-label">Command center</p>
                <h2>Run settings</h2>
              </div>

              <button
                className="topbar-icon"
                type="button"
                onClick={() => setIsSettingsOpen(false)}
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>

            <section className="settings-section">
              <div className="settings-section-head">
                <h3>Sampling</h3>
                <p>These controls are forwarded directly to the local backend.</p>
              </div>

              <div className="field-grid">
                <label className="field">
                  <span>Max new tokens</span>
                  <input
                    type="number"
                    min="32"
                    max={isWslVllmRuntime ? NVFP4_MAX_NEW_TOKENS : 1024}
                    value={settings.maxNewTokens}
                    onChange={(event) =>
                      setSettings((current) => {
                        const fallbackValue = isWslVllmRuntime
                          ? NVFP4_MAX_NEW_TOKENS
                          : DEFAULT_MAX_NEW_TOKENS
                        const parsedValue = Number(
                          event.target.value || fallbackValue,
                        )
                        const clampedValue = isWslVllmRuntime
                          ? Math.min(NVFP4_MAX_NEW_TOKENS, parsedValue)
                          : parsedValue
                        return {
                          ...current,
                          maxNewTokens: clampedValue,
                        }
                      })
                    }
                  />
                </label>

                {isWslVllmRuntime ? (
                  <p className="muted-copy">
                    The local NVFP4 bridge runs in a compact 256-token context
                    profile on this 5090. Output is capped to 64 new tokens and
                    older turns are compressed automatically.
                  </p>
                ) : null}

                <label className="field">
                  <span>Continuation</span>
                  <select
                    value={settings.continuationMode}
                    onChange={(event) =>
                      setSettings((current) => ({
                        ...current,
                        continuationMode: event.target.value,
                      }))
                    }
                  >
                    <option value="off">Off</option>
                    <option value="manual">Continue button</option>
                    <option value="auto">Auto-continue</option>
                  </select>
                </label>

                <label className="field">
                  <span>Temperature</span>
                  <input
                    type="number"
                    min="0"
                    max="2"
                    step="0.05"
                    value={settings.temperature}
                    onChange={(event) =>
                      setSettings((current) => ({
                        ...current,
                        temperature: Number(event.target.value || 1),
                      }))
                    }
                  />
                </label>

                <label className="field">
                  <span>Top p</span>
                  <input
                    type="number"
                    min="0.1"
                    max="1"
                    step="0.01"
                    value={settings.topP}
                    onChange={(event) =>
                      setSettings((current) => ({
                        ...current,
                        topP: Number(event.target.value || 0.95),
                      }))
                    }
                  />
                </label>

                <label className="field">
                  <span>Top k</span>
                  <input
                    type="number"
                    min="1"
                    max="128"
                    value={settings.topK}
                    onChange={(event) =>
                      setSettings((current) => ({
                        ...current,
                        topK: Number(event.target.value || 64),
                      }))
                    }
                  />
                </label>
              </div>

              <label className="toggle-field">
                <span>Thinking mode</span>
                <input
                  type="checkbox"
                  checked={settings.thinking}
                  onChange={(event) =>
                    setSettings((current) => ({
                      ...current,
                      thinking: event.target.checked,
                    }))
                  }
                />
              </label>
            </section>

            <section className="settings-section">
              <div className="settings-section-head">
                <h3>System prompt</h3>
                <p>Applied to every new request on the active thread.</p>
              </div>

              <label className="field">
                <span>Instruction layer</span>
                <textarea
                  value={systemPrompt}
                  onChange={(event) => setSystemPrompt(event.target.value)}
                  rows={7}
                />
              </label>
            </section>

            <section className="settings-section">
              <div className="settings-section-head">
                <h3>Machine state</h3>
                <p>Live telemetry from the local Gemma backend.</p>
              </div>

              <div className="metric-grid">
                <div className="metric-card">
                  <span>Status</span>
                  <strong>{health?.loaded ? 'Model loaded' : 'Cold start'}</strong>
                </div>
                <div className="metric-card">
                  <span>VRAM now</span>
                  <strong>{formatMemory(health?.vram_allocated_gib)}</strong>
                </div>
                <div className="metric-card">
                  <span>Active model</span>
                  <strong>{activeModel?.label || 'Gemma 4'}</strong>
                </div>
                <div className="metric-card">
                  <span>Active quantization</span>
                  <strong>{activeQuantization?.label || 'BF16'}</strong>
                </div>
                <div className="metric-card">
                  <span>GPU total</span>
                  <strong>{formatMemory(health?.gpu_total_memory_gib)}</strong>
                </div>
                <div className="metric-card">
                  <span>Google estimate</span>
                  <strong>{formatMemory(activeMemoryEstimate)}</strong>
                </div>
                <div className="metric-card">
                  <span>Cache</span>
                  <strong>{health?.cache_dir || '.hf-cache'}</strong>
                </div>
                <div className="metric-card">
                  <span>Quant note</span>
                  <strong>{getQuantizationRuntimeLabel(selectedQuantization)}</strong>
                </div>
              </div>
            </section>

            <section className="settings-section">
              <div className="settings-section-head">
                <h3>Last answer</h3>
                <p>Telemetry for the latest assistant message in this thread.</p>
              </div>

              {latestAssistantMessage ? (
                <div className="metric-grid">
                  <div className="metric-card">
                    <span>Model</span>
                    <strong>
                      {latestAssistantMessage.meta.active_model?.label ||
                        latestAssistantMessage.meta.active_model_key}
                    </strong>
                  </div>
                  <div className="metric-card">
                    <span>Quantization</span>
                    <strong>
                      {latestAssistantMessage.meta.active_quantization?.label ||
                        latestAssistantMessage.meta.active_quantization_key ||
                        'BF16'}
                    </strong>
                  </div>
                  <div className="metric-card">
                    <span>Latency</span>
                    <strong>{formatElapsed(latestAssistantMessage.meta.elapsed_ms)}</strong>
                  </div>
                  <div className="metric-card">
                    <span>New tokens</span>
                    <strong>{latestAssistantMessage.meta.generated_tokens}</strong>
                  </div>
                  <div className="metric-card">
                    <span>Reserved VRAM</span>
                    <strong>
                      {formatMemory(latestAssistantMessage.meta.vram_reserved_gib)}
                    </strong>
                  </div>
                </div>
              ) : (
                <p className="muted-copy">
                  No answer yet on this thread. Run a prompt to inspect the local
                  telemetry.
                </p>
              )}
            </section>

            <section className="settings-section">
              <div className="settings-section-head">
                <h3>Source note</h3>
                <p>Modalities follow the official Gemma 4 tables.</p>
              </div>
              <p className="muted-copy">{docsNote}</p>
            </section>
          </aside>
        </>
      ) : null}
    </div>
  )
}

export default App
