import { useEffect, useRef } from 'react'
import { useStore } from '../state'

// Reads the AnalyserNode on every rAF tick, computes smoothed RMS and four
// FFT bands, and pushes them into the shared store. The shader subscribes to
// those values via refs (see QuiverCanvas) so we never re-render React for
// amplitude changes.
export function useAmplitude(getAnalyser: () => AnalyserNode | null, onPeak: (p: number) => void) {
  const rafRef = useRef<number | null>(null)
  const ampEmaRef = useRef(0)
  const bandsEmaRef = useRef<[number, number, number, number]>([0, 0, 0, 0])

  useEffect(() => {
    const alpha = 0.28
    const time = new Uint8Array(1024)
    const freq = new Uint8Array(1024)

    const tick = () => {
      const analyser = getAnalyser()
      if (analyser) {
        if (analyser.fftSize !== time.length) {
          // no-op — we keep fftSize fixed at construction; safeguard
        }
        analyser.getByteTimeDomainData(time)
        analyser.getByteFrequencyData(freq)

        // RMS over time-domain samples mapped 0..255, centered at 128.
        let sumSq = 0
        for (let i = 0; i < time.length; i++) {
          const v = (time[i] - 128) / 128
          sumSq += v * v
        }
        const rms = Math.sqrt(sumSq / time.length)
        ampEmaRef.current = ampEmaRef.current * (1 - alpha) + rms * alpha

        // 4 FFT bands: low, mid-low, mid, high. log-spaced bin boundaries.
        const N = freq.length
        const bins = [
          [0, Math.floor(N * 0.04)],
          [Math.floor(N * 0.04), Math.floor(N * 0.12)],
          [Math.floor(N * 0.12), Math.floor(N * 0.32)],
          [Math.floor(N * 0.32), Math.floor(N * 0.8)],
        ] as const
        const nextBands: [number, number, number, number] = [0, 0, 0, 0]
        for (let b = 0; b < 4; b++) {
          let s = 0
          const [lo, hi] = bins[b]
          for (let i = lo; i < hi; i++) s += freq[i]
          nextBands[b] = s / ((hi - lo) * 255)
        }
        for (let b = 0; b < 4; b++) {
          bandsEmaRef.current[b] = bandsEmaRef.current[b] * (1 - alpha) + nextBands[b] * alpha
        }

        useStore.setState({
          amp: ampEmaRef.current,
          bands: [...bandsEmaRef.current] as [number, number, number, number],
        })
        if (rms > useStore.getState().peak) onPeak(rms)
      }
      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [getAnalyser, onPeak])
}
