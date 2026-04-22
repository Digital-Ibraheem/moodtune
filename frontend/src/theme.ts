// Emotion → background tint. HSL triplets for smooth interpolation in GLSL.
// Read as: [hue 0-1, sat 0-1, lightness 0-1].
// Chosen to read immediately even at low intensity — muted, filmic palette.

export type Emotion = 'neutral' | 'happy' | 'sad' | 'angry'

export const EMOTIONS: Emotion[] = ['neutral', 'happy', 'sad', 'angry']

// RGB values passed to the shader as vec3 in 0-1 space. Dark + saturated enough
// to be perceptible under the noise field but never overwhelming.
export const EMOTION_RGB: Record<Emotion, [number, number, number]> = {
  neutral: [0.42, 0.46, 0.55], //  cool slate
  happy:   [0.96, 0.66, 0.22], //  warm amber
  sad:     [0.22, 0.28, 0.62], //  deep indigo
  angry:   [0.78, 0.18, 0.18], //  ember red
}

export const BASE_RGB: [number, number, number] = [0.028, 0.031, 0.043] // #07080b-ish

export const COPY: Record<Emotion, string> = {
  neutral: 'Neutral',
  happy: 'Happy',
  sad: 'Sad',
  angry: 'Angry',
}

export const RECORD_SECONDS = 4
