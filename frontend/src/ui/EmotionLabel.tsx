import { useStore } from '../state'
import { COPY, EMOTIONS, EMOTION_RGB } from '../theme'

export function EmotionLabel() {
  const phase = useStore((s) => s.phase)
  const probs = useStore((s) => s.probs)
  const top = useStore((s) => s.top)
  if (phase !== 'result' || !probs || !top) return null

  const rgb = EMOTION_RGB[top]
  const accent = `rgb(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(rgb[2] * 255)})`

  return (
    <div className="emotion">
      <div className="emotion__eyebrow">you sound</div>
      <div
        className="emotion__word"
        style={{ color: accent }}
      >
        {COPY[top]}
      </div>
      <div className="emotion__bars">
        {EMOTIONS.map((e) => {
          const p = probs[e] ?? 0
          return (
            <div className="emotion__row" key={e}>
              <span className="emotion__rowLabel">{COPY[e]}</span>
              <span className="emotion__bar">
                <span
                  className="emotion__barFill"
                  style={{
                    width: `${Math.round(p * 100)}%`,
                    background: e === top ? accent : 'rgba(255,255,255,0.35)',
                  }}
                />
              </span>
              <span className="emotion__rowPct">{Math.round(p * 100)}%</span>
            </div>
          )
        })}
      </div>
      <div className="emotion__hint">tap anywhere to go again</div>
    </div>
  )
}
