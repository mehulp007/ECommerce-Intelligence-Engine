/** Round, size-attenuated points shared by the Galaxy and the story scenes. */

export const POINT_VERTEX = /* glsl */ `
attribute vec3 aColor;
attribute float aHighlight;
uniform float uSize;
uniform float uPixelRatio;
uniform float uFocus;
uniform float uOpacity;
uniform float uReveal;
uniform float uDim;
varying vec3 vColor;
varying float vAlpha;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float boost = mix(1.0, mix(0.55, 1.8, aHighlight), uFocus);
  // Points grow as the camera nears, up to a cap so close-ups stay crisp.
  float size = uSize * uPixelRatio * boost / -mv.z;
  gl_PointSize = clamp(size, 1.0, uPixelRatio * mix(6.0, 11.0, aHighlight * uFocus));
  vColor = aColor;
  // Points beyond the reveal edge (along x) stay hidden, for drawing in.
  float revealed = smoothstep(uReveal, uReveal - 0.08, position.x);
  vAlpha = mix(0.82, mix(uDim, 1.0, aHighlight), uFocus) * uOpacity * revealed;
}`;

export const POINT_FRAGMENT = /* glsl */ `
varying vec3 vColor;
varying float vAlpha;
void main() {
  float d = length(gl_PointCoord - 0.5);
  if (d > 0.5 || vAlpha < 0.004) discard;
  gl_FragColor = vec4(vColor, vAlpha * smoothstep(0.5, 0.3, d));
}`;

export function pointUniforms(size = 9) {
  return {
    uSize: { value: size },
    uPixelRatio: { value: 1 },
    uFocus: { value: 0 },
    uOpacity: { value: 1 },
    uReveal: { value: 1e6 },
    uDim: { value: 0.05 },
  };
}
