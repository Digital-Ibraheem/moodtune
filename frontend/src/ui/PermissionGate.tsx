import { useStore } from '../state'

type Props = { onGrant: () => void }

export function PermissionGate({ onGrant }: Props) {
  const phase = useStore((s) => s.phase)
  if (phase === 'denied') {
    return (
      <div className="gate">
        <h1 className="gate__title">Microphone blocked</h1>
        <p className="gate__sub">
          MoodTune needs your microphone to hear you. Enable it in your browser's site settings and reload.
        </p>
      </div>
    )
  }
  return (
    <div className="gate">
      <div className="gate__eyebrow">MoodTune</div>
      <h1 className="gate__title">
        Speak.<br />Hear the room listen back.
      </h1>
      <p className="gate__sub">
        A Wav2Vec2 model trained on acted emotional speech, brought into the open.
        Tap to enable the microphone.
      </p>
      <button className="gate__cta" onClick={onGrant}>
        Enable microphone
      </button>
    </div>
  )
}
