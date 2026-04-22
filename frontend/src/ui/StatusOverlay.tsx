import { useStore } from '../state'

// Shown during encoding / predicting / silence / error so the user sees that
// something is happening after the recording ring drains. Fades in over the
// record button without hiding the quiver background.
export function StatusOverlay() {
  const phase = useStore((s) => s.phase)
  const error = useStore((s) => s.error)

  let text: string | null = null
  let sub: string | null = null
  if (phase === 'predicting') {
    text = 'Thinking'
    sub = 'Wav2Vec2 is reading your voice'
  } else if (phase === 'encoding') {
    text = 'Encoding'
    sub = null
  } else if (phase === 'silence') {
    text = "Didn't catch that"
    sub = 'try again with a little more volume'
  } else if (phase === 'error') {
    text = 'Something broke'
    sub = error?.slice(0, 140) ?? null
  }
  if (!text) return null

  return (
    <div className="status">
      <div className="status__dots">
        <span />
        <span />
        <span />
      </div>
      <div className="status__text">{text}</div>
      {sub && <div className="status__sub">{sub}</div>}
    </div>
  )
}
