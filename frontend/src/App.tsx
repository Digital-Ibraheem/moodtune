import { useCallback, useEffect } from 'react'
import { postAudio, warmup } from './api'
import { useAmplitude } from './audio/useAmplitude'
import { useRecorder } from './audio/useRecorder'
import { QuiverCanvas } from './gl/QuiverCanvas'
import { useStore } from './state'
import { EmotionLabel } from './ui/EmotionLabel'
import { PermissionGate } from './ui/PermissionGate'
import { RecordButton } from './ui/RecordButton'
import { StatusOverlay } from './ui/StatusOverlay'

export default function App() {
  const phase = useStore((s) => s.phase)

  const onBlob = useCallback(async (blob: Blob, peak: number) => {
    console.log('[predict] onBlob', { size: blob.size, peak })
    if (peak < 0.01) {
      console.warn('[predict] below silence threshold', peak)
      useStore.setState({ phase: 'silence' })
      window.setTimeout(() => useStore.setState({ phase: 'idle' }), 1400)
      return
    }
    useStore.setState({ phase: 'predicting' })
    try {
      const res = await postAudio(blob)
      console.log('[predict] response', res)
      if ('error' in res) {
        useStore.setState({ phase: 'silence' })
        window.setTimeout(() => useStore.setState({ phase: 'idle' }), 1400)
        return
      }
      useStore.setState({
        phase: 'result',
        probs: res.probs,
        top: res.top,
        peak: res.peak,
      })
    } catch (e) {
      console.error('[predict] failed', e)
      useStore.setState({ phase: 'error', error: String(e) })
      window.setTimeout(() => useStore.setState({ phase: 'idle', error: null }), 2400)
    }
  }, [])

  const { grantPermission, start, getAnalyser, bumpPeak } = useRecorder(onBlob)
  useAmplitude(getAnalyser, bumpPeak)

  // Warm the backend as soon as the page opens — first request no longer cold.
  useEffect(() => {
    void warmup()
  }, [])

  // Tap-anywhere-to-reset after a result.
  useEffect(() => {
    if (phase !== 'result' && phase !== 'silence') return
    const handler = () => {
      useStore.getState().reset()
    }
    // slight delay so the tap that stopped recording isn't double-counted
    const id = window.setTimeout(() => window.addEventListener('click', handler, { once: true }), 400)
    return () => {
      window.clearTimeout(id)
      window.removeEventListener('click', handler)
    }
  }, [phase])

  // Keyboard: space toggles, esc cancels (future-safe no-op for now).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'Space' && (phase === 'idle' || phase === 'needs-mic')) {
        e.preventDefault()
        phase === 'needs-mic' ? void grantPermission() : void start()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [phase, grantPermission, start])

  return (
    <div className="stage">
      <QuiverCanvas />
      <main className="stage__main">
        {phase === 'needs-mic' || phase === 'denied' ? (
          <PermissionGate onGrant={grantPermission} />
        ) : (
          <>
            <RecordButton onStart={start} />
            <EmotionLabel />
            <StatusOverlay />
          </>
        )}
      </main>
      <footer className="stage__footer">
        <a href="https://github.com/Digital-Ibraheem/moodtune" target="_blank" rel="noreferrer">
          github
        </a>
        <span>•</span>
        <span>Wav2Vec2 · 4-class · RAVDESS-trained</span>
      </footer>
    </div>
  )
}
