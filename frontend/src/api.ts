import { blobToWav } from './audio/wavEncode'
import type { Probs } from './state'

export type PredictOk = { probs: Probs; top: keyof Probs; peak: number; duration: number }
export type PredictSilence = { error: 'silence'; peak: number }
export type PredictResponse = PredictOk | PredictSilence

export async function postAudio(blob: Blob): Promise<PredictResponse> {
  // Normalize to 16kHz mono 16-bit WAV so the backend only needs soundfile.
  const wav = await blobToWav(blob, 16000)
  const form = new FormData()
  form.append('file', wav, 'rec.wav')

  const res = await fetch('/predict', { method: 'POST', body: form })
  if (!res.ok) throw new Error(`predict ${res.status}: ${await res.text()}`)
  return (await res.json()) as PredictResponse
}

export async function warmup(): Promise<void> {
  try {
    await fetch('/health')
  } catch {
    /* ignore — health is best-effort */
  }
}
