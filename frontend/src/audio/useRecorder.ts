import { useCallback, useEffect, useRef } from 'react'
import { useStore } from '../state'
import { RECORD_SECONDS } from '../theme'

type Handles = {
  stream: MediaStream
  ctx: AudioContext
  analyser: AnalyserNode
  recorder: MediaRecorder
}

// One resource owner. Acquires mic on grantPermission(); from then on, the
// AnalyserNode is always live so the shader stays reactive even at idle.
export function useRecorder(onBlob: (blob: Blob, peak: number) => void) {
  const handlesRef = useRef<Handles | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)
  const peakRef = useRef(0)

  const grantPermission = useCallback(async () => {
    if (handlesRef.current) return handlesRef.current
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
      // iOS / Safari: AudioContext must be created or resumed inside a user gesture.
      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      const ctx = new Ctx()
      if (ctx.state === 'suspended') await ctx.resume()
      const source = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 1024
      analyser.smoothingTimeConstant = 0.6
      source.connect(analyser)

      // Pick a mime the browser actually supports. Let backend handle decoding.
      const candidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4;codecs=mp4a.40.2',
        'audio/mp4',
        'audio/ogg;codecs=opus',
      ]
      const mime = candidates.find((m) => MediaRecorder.isTypeSupported(m)) || ''
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)

      handlesRef.current = { stream, ctx, analyser, recorder }
      useStore.setState({ phase: 'idle' })
      return handlesRef.current
    } catch (err) {
      console.error('[mic] permission denied', err)
      useStore.setState({ phase: 'denied', error: String(err) })
      return null
    }
  }, [])

  const start = useCallback(async () => {
    const h = handlesRef.current ?? (await grantPermission())
    if (!h) return
    const { recorder } = h
    chunksRef.current = []
    peakRef.current = 0

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
      console.log('[rec] stop', { size: blob.size, type: blob.type, peak: peakRef.current })
      onBlob(blob, peakRef.current)
    }
    recorder.onerror = (e) => console.error('[rec] error', e)

    recorder.start()
    console.log('[rec] start', { mime: recorder.mimeType })
    useStore.setState({ phase: 'recording', peak: 0 })

    timerRef.current = window.setTimeout(() => {
      if (recorder.state === 'recording') recorder.stop()
    }, RECORD_SECONDS * 1000)
  }, [grantPermission, onBlob])

  const stop = useCallback(() => {
    const h = handlesRef.current
    if (!h) return
    if (h.recorder.state === 'recording') h.recorder.stop()
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const getAnalyser = useCallback(() => handlesRef.current?.analyser ?? null, [])
  const bumpPeak = useCallback((p: number) => {
    if (p > peakRef.current) peakRef.current = p
  }, [])

  useEffect(
    () => () => {
      const h = handlesRef.current
      if (!h) return
      try {
        h.recorder.state === 'recording' && h.recorder.stop()
      } catch {}
      h.stream.getTracks().forEach((t) => t.stop())
      h.ctx.close().catch(() => {})
    },
    [],
  )

  return { grantPermission, start, stop, getAnalyser, bumpPeak }
}
