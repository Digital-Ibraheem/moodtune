import { useEffect, useRef } from 'react'

export type Uniforms = Record<string, number | [number, number] | [number, number, number] | [number, number, number, number]>

type Options = {
  vert: string
  frag: string
  getUniforms: () => Uniforms
}

// Minimal WebGL 1.0 fullscreen-quad shader host. No scene graph, no
// deps. Creates one program, uploads uniforms per frame, resizes on DPR/
// window changes. Returns nothing — owns the canvas lifecycle.
export function useFullscreenShader(canvasRef: React.RefObject<HTMLCanvasElement | null>, opts: Options) {
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const gl = canvas.getContext('webgl', { antialias: false, premultipliedAlpha: false })
    if (!gl) {
      console.error('[gl] WebGL not available')
      return
    }

    const compile = (type: number, src: string) => {
      const sh = gl.createShader(type)!
      gl.shaderSource(sh, src)
      gl.compileShader(sh)
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(sh))
        throw new Error('shader compile error')
      }
      return sh
    }

    const vs = compile(gl.VERTEX_SHADER, opts.vert)
    const fs = compile(gl.FRAGMENT_SHADER, opts.frag)
    const prog = gl.createProgram()!
    gl.attachShader(prog, vs)
    gl.attachShader(prog, fs)
    gl.linkProgram(prog)
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(prog))
      return
    }

    const posLoc = gl.getAttribLocation(prog, 'a_pos')
    const buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW)

    const uniformLocs = new Map<string, WebGLUniformLocation | null>()
    const loc = (name: string) => {
      if (!uniformLocs.has(name)) uniformLocs.set(name, gl.getUniformLocation(prog, name))
      return uniformLocs.get(name)!
    }

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const w = Math.floor(window.innerWidth * dpr)
      const h = Math.floor(window.innerHeight * dpr)
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }
      canvas.style.width = window.innerWidth + 'px'
      canvas.style.height = window.innerHeight + 'px'
      gl.viewport(0, 0, w, h)
    }
    resize()
    const onResize = () => resize()
    window.addEventListener('resize', onResize)

    const start = performance.now()
    const render = () => {
      const t = (performance.now() - start) / 1000
      gl.useProgram(prog)
      gl.enableVertexAttribArray(posLoc)
      gl.bindBuffer(gl.ARRAY_BUFFER, buf)
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0)

      gl.uniform1f(loc('uTime'), t)
      gl.uniform2f(loc('uRes'), canvas.width, canvas.height)

      const u = opts.getUniforms()
      for (const [name, val] of Object.entries(u)) {
        const l = loc(name)
        if (!l) continue
        if (typeof val === 'number') gl.uniform1f(l, val)
        else if (val.length === 2) gl.uniform2f(l, val[0], val[1])
        else if (val.length === 3) gl.uniform3f(l, val[0], val[1], val[2])
        else if (val.length === 4) gl.uniform4f(l, val[0], val[1], val[2], val[3])
      }

      gl.drawArrays(gl.TRIANGLES, 0, 3)
      rafRef.current = requestAnimationFrame(render)
    }
    rafRef.current = requestAnimationFrame(render)

    return () => {
      window.removeEventListener('resize', onResize)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      gl.deleteProgram(prog)
      gl.deleteShader(vs)
      gl.deleteShader(fs)
      gl.deleteBuffer(buf)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
}
