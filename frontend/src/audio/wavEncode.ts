// Decode whatever container the browser recorded (webm/opus, mp4/aac, ogg) into
// PCM Float32, then write a 16-bit mono WAV blob. Backend reads with soundfile —
// no ffmpeg dependency on the server.

export async function blobToWav(blob: Blob, targetSr = 16000): Promise<Blob> {
  const buf = await blob.arrayBuffer()
  // decodeAudioData needs its own AudioContext; we can discard it after.
  const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  const ctx = new Ctx()
  const decoded = await ctx.decodeAudioData(buf.slice(0))
  ctx.close().catch(() => {})

  // Mix down to mono.
  const srcSr = decoded.sampleRate
  const len = decoded.length
  const chans = decoded.numberOfChannels
  const mono = new Float32Array(len)
  for (let c = 0; c < chans; c++) {
    const ch = decoded.getChannelData(c)
    for (let i = 0; i < len; i++) mono[i] += ch[i] / chans
  }

  // Resample if needed (cheap linear — model already tolerates 16kHz input).
  const resampled = srcSr === targetSr ? mono : linearResample(mono, srcSr, targetSr)
  return encodeWav16(resampled, targetSr)
}

function linearResample(data: Float32Array, fromSr: number, toSr: number): Float32Array {
  const ratio = fromSr / toSr
  const outLen = Math.round(data.length / ratio)
  const out = new Float32Array(outLen)
  for (let i = 0; i < outLen; i++) {
    const src = i * ratio
    const i0 = Math.floor(src)
    const i1 = Math.min(i0 + 1, data.length - 1)
    const t = src - i0
    out[i] = data[i0] * (1 - t) + data[i1] * t
  }
  return out
}

function encodeWav16(data: Float32Array, sr: number): Blob {
  const bytesPerSample = 2
  const blockAlign = bytesPerSample // mono
  const byteRate = sr * blockAlign
  const dataBytes = data.length * bytesPerSample
  const buffer = new ArrayBuffer(44 + dataBytes)
  const v = new DataView(buffer)

  // RIFF header
  writeStr(v, 0, 'RIFF')
  v.setUint32(4, 36 + dataBytes, true)
  writeStr(v, 8, 'WAVE')
  // fmt chunk
  writeStr(v, 12, 'fmt ')
  v.setUint32(16, 16, true)        // PCM chunk size
  v.setUint16(20, 1, true)         // PCM format
  v.setUint16(22, 1, true)         // channels = 1
  v.setUint32(24, sr, true)
  v.setUint32(28, byteRate, true)
  v.setUint16(32, blockAlign, true)
  v.setUint16(34, 16, true)        // bits per sample
  // data chunk
  writeStr(v, 36, 'data')
  v.setUint32(40, dataBytes, true)

  // PCM samples
  let o = 44
  for (let i = 0; i < data.length; i++) {
    const s = Math.max(-1, Math.min(1, data[i]))
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    o += 2
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

function writeStr(v: DataView, offset: number, s: string) {
  for (let i = 0; i < s.length; i++) v.setUint8(offset + i, s.charCodeAt(i))
}
