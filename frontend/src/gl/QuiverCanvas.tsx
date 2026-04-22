import { useEffect, useRef } from 'react'
import { useStore } from '../state'
import { BASE_RGB, EMOTION_RGB } from '../theme'
import { FRAG, VERT } from './quiver.frag'
import { useFullscreenShader } from './useFullscreenShader'

// Bridges the zustand store into the shader uniforms without re-rendering React.
// The shader pulls fresh values every frame via getUniforms(). We also ease
// tintMix and focus on our own rAF so transitions don't snap.
export function QuiverCanvas() {
  const ref = useRef<HTMLCanvasElement>(null)

  // Target values derived from phase, animated via exponential easing.
  const eased = useRef({ tintMix: 0, focus: 0, tintRGB: EMOTION_RGB.neutral })

  useEffect(() => {
    let raf = 0
    const tick = () => {
      const s = useStore.getState()
      const targetTintMix = s.phase === 'result' ? 1 : 0
      const targetFocus = s.phase === 'recording' ? 1 : s.phase === 'predicting' ? 0.7 : 0.3
      const targetTint = s.top ? EMOTION_RGB[s.top] : EMOTION_RGB.neutral
      const a = 0.08
      eased.current.tintMix += (targetTintMix - eased.current.tintMix) * a
      eased.current.focus += (targetFocus - eased.current.focus) * a
      eased.current.tintRGB = [
        eased.current.tintRGB[0] + (targetTint[0] - eased.current.tintRGB[0]) * a,
        eased.current.tintRGB[1] + (targetTint[1] - eased.current.tintRGB[1]) * a,
        eased.current.tintRGB[2] + (targetTint[2] - eased.current.tintRGB[2]) * a,
      ]
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  useFullscreenShader(ref, {
    vert: VERT,
    frag: FRAG,
    getUniforms: () => {
      const s = useStore.getState()
      const motion = s.reducedMotion ? 0.1 : 1
      return {
        uAmp: s.amp,
        uBands: s.bands,
        uBase: BASE_RGB,
        uTint: eased.current.tintRGB,
        uTintMix: eased.current.tintMix,
        uMotion: motion,
        uFocus: eased.current.focus,
      }
    },
  })

  return <canvas ref={ref} className="quiver-canvas" />
}
