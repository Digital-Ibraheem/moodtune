import { create } from 'zustand'
import type { Emotion } from './theme'

export type Phase =
  | 'needs-mic'
  | 'denied'
  | 'idle'
  | 'recording'
  | 'encoding'
  | 'predicting'
  | 'result'
  | 'silence'
  | 'error'

export type Probs = Record<Emotion, number>

type Store = {
  phase: Phase
  amp: number            // smoothed RMS, 0..1
  peak: number           // peak amplitude observed during last recording
  bands: [number, number, number, number] // 4 FFT bands, 0..1
  probs: Probs | null
  top: Emotion | null
  error: string | null
  reducedMotion: boolean
  set: (patch: Partial<Store>) => void
  reset: () => void
}

export const useStore = create<Store>((set) => ({
  phase: 'needs-mic',
  amp: 0,
  peak: 0,
  bands: [0, 0, 0, 0],
  probs: null,
  top: null,
  error: null,
  reducedMotion: window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false,
  set: (patch) => set(patch),
  reset: () =>
    set({
      phase: 'idle',
      probs: null,
      top: null,
      error: null,
      peak: 0,
    }),
}))
