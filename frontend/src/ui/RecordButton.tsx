import { useEffect, useRef } from 'react'
import clsx from 'clsx'
import { useStore } from '../state'
import { RECORD_SECONDS } from '../theme'

type Props = { onStart: () => void }

// SVG ring around a circular button. Ring drains during a 4s recording.
// Inside ring: a vertical bar that tracks live amplitude.
export function RecordButton({ onStart }: Props) {
  const phase = useStore((s) => s.phase)
  const disabled = phase === 'encoding' || phase === 'predicting'
  const recording = phase === 'recording'
  const ringRef = useRef<SVGCircleElement>(null)
  const meterRef = useRef<HTMLSpanElement>(null)

  const C = 2 * Math.PI * 54 // circumference for r=54 in viewBox 120

  // Animate ring stroke-dashoffset with a synchronized rAF loop so React doesn't rerender.
  useEffect(() => {
    if (!recording) {
      if (ringRef.current) ringRef.current.style.strokeDashoffset = '0'
      return
    }
    const start = performance.now()
    let raf = 0
    const tick = () => {
      const t = Math.min(1, (performance.now() - start) / (RECORD_SECONDS * 1000))
      if (ringRef.current) ringRef.current.style.strokeDashoffset = `${C * t}`
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [recording, C])

  // Amplitude meter driven imperatively.
  useEffect(() => {
    let raf = 0
    const tick = () => {
      const s = useStore.getState()
      const h = Math.min(1, s.amp * 6) // amplify for visibility
      if (meterRef.current) meterRef.current.style.transform = `scaleY(${0.08 + h * 0.92})`
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  const label =
    phase === 'recording' ? 'Listening…'
    : phase === 'encoding' ? 'Encoding'
    : phase === 'predicting' ? 'Thinking'
    : phase === 'silence' ? 'Didn’t catch that'
    : 'Tap to speak'

  return (
    <div className="record">
      <button
        className={clsx('record__btn', {
          'record__btn--recording': recording,
          'record__btn--busy': disabled,
        })}
        onClick={onStart}
        disabled={disabled || recording}
        aria-label="Record 4 seconds of audio"
      >
        <svg viewBox="0 0 120 120" className="record__ring" aria-hidden>
          <circle cx="60" cy="60" r="54" className="record__ring-bg" />
          <circle
            ref={ringRef}
            cx="60"
            cy="60"
            r="54"
            className="record__ring-fg"
            strokeDasharray={C}
            strokeDashoffset={0}
          />
        </svg>
        <span className="record__core" aria-hidden>
          <span ref={meterRef} className="record__meter" />
        </span>
      </button>
      <div className="record__label">{label}</div>
    </div>
  )
}
