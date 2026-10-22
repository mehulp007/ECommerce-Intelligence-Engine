"use client";

import { Billboard, CameraControls } from "@react-three/drei";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import type { Galaxy } from "@/lib/galaxy";

const VERTEX = /* glsl */ `
attribute vec3 aColor;
attribute float aHighlight;
uniform float uSize;
uniform float uPixelRatio;
uniform float uFocus;
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
  vAlpha = mix(0.82, mix(0.05, 1.0, aHighlight), uFocus);
}`;

const FRAGMENT = /* glsl */ `
varying vec3 vColor;
varying float vAlpha;
void main() {
  float d = length(gl_PointCoord - 0.5);
  if (d > 0.5) discard;
  gl_FragColor = vec4(vColor, vAlpha * smoothstep(0.5, 0.3, d));
}`;

export const HOME = { position: [0, 0.35, 2.7] as const, target: [0, 0, 0] as const };

export interface Focus {
  center: [number, number, number];
  /** Spread of the focused points, used to choose the camera distance. */
  radius: number;
}

interface Props {
  galaxy: Galaxy;
  colours: Float32Array;
  /** 1 for highlighted points, 0 for dimmed ones; null when nothing is focused. */
  highlight: Float32Array | null;
  focus: Focus | null;
  /** A ring drawn around one product, for example when opened from its page. */
  marker?: [number, number, number] | null;
  autoRotate: boolean;
  animate: boolean;
  onHover: (index: number | null, x: number, y: number) => void;
}

function Points({ galaxy, colours, highlight, onHover }: Omit<Props, "focus" | "autoRotate" | "animate">) {
  const lastMove = useRef(0);
  const material = useRef<THREE.ShaderMaterial>(null);
  const [uniforms] = useState(() => ({
    uSize: { value: 9 },
    uPixelRatio: { value: 1 },
    uFocus: { value: 0 },
  }));
  const allVisible = useMemo(() => new Float32Array(galaxy.count).fill(1), [galaxy]);

  useEffect(() => {
    if (material.current) material.current.uniforms.uFocus.value = highlight ? 1 : 0;
  }, [highlight]);

  // Leaving the page while hovering must not leave a pointer cursor behind.
  useEffect(
    () => () => {
      document.body.style.cursor = "";
    },
    [],
  );

  useFrame(({ gl, camera, raycaster }) => {
    if (material.current) material.current.uniforms.uPixelRatio.value = gl.getPixelRatio();
    // Keep the pick radius about one point wide, whatever the zoom.
    raycaster.params.Points = { threshold: 0.006 * camera.position.length() };
  });

  /** The nearest visible point to the pointer ray. */
  const pick = (event: ThreeEvent<PointerEvent>) => {
    let best: number | null = null;
    let bestDistance = Infinity;
    for (const hit of event.intersections) {
      if (hit.object !== event.eventObject || hit.index === undefined) continue;
      if (highlight && highlight[hit.index] < 0.5) continue;
      const distance = hit.distanceToRay ?? 0;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = hit.index;
      }
    }
    return best;
  };

  return (
    <points
      onPointerMove={(event) => {
        const now = performance.now();
        if (now - lastMove.current < 40) return;
        lastMove.current = now;
        const index = pick(event);
        document.body.style.cursor = index === null ? "" : "pointer";
        onHover(index, event.nativeEvent.clientX, event.nativeEvent.clientY);
      }}
      onPointerOut={() => {
        document.body.style.cursor = "";
        onHover(null, 0, 0);
      }}
    >
      {/* Changing an array swaps the attribute; three.js uploads it to the GPU. */}
      <bufferGeometry key={galaxy.count}>
        <bufferAttribute attach="attributes-position" args={[galaxy.positions, 3]} />
        <bufferAttribute attach="attributes-aColor" args={[colours, 3]} />
        <bufferAttribute attach="attributes-aHighlight" args={[highlight ?? allVisible, 1]} />
      </bufferGeometry>
      <shaderMaterial
        ref={material}
        vertexShader={VERTEX}
        fragmentShader={FRAGMENT}
        uniforms={uniforms}
        transparent
        depthWrite={false}
      />
    </points>
  );
}

function Camera({ focus, autoRotate, animate }: Pick<Props, "focus" | "autoRotate" | "animate">) {
  const controls = useRef<CameraControls>(null);
  // Tall, narrow screens need the camera further back to fit the cloud.
  const aspect = useThree((state) => state.size.width / Math.max(1, state.size.height));
  const fit = aspect < 1 ? Math.min(2, 1 / aspect) * 0.85 : 1;
  const idleSince = useRef(0);
  const interacting = useRef(false);

  useEffect(() => {
    const c = controls.current;
    if (!c) return;
    if (focus) {
      const [x, y, z] = focus.center;
      const distance = Math.min(2.2, Math.max(0.24, focus.radius * 3.2));
      const direction = new THREE.Vector3(x, y, z);
      if (direction.lengthSq() < 1e-6) direction.set(0, 0, 1);
      direction.normalize().multiplyScalar(distance);
      c.setLookAt(x + direction.x, y + direction.y + distance * 0.15, z + direction.z, x, y, z, animate);
    } else {
      const [x, y, z] = HOME.position;
      c.setLookAt(x * fit, y * fit, z * fit, ...HOME.target, animate);
    }
  }, [focus, animate, fit]);

  useFrame((_, delta) => {
    const c = controls.current;
    if (!c || !autoRotate || focus || interacting.current) return;
    // Resume the slow spin a few seconds after the visitor lets go.
    if (performance.now() - idleSince.current < 3000) return;
    c.rotate(delta * 0.05, 0, false);
  });

  return (
    <CameraControls
      ref={controls}
      makeDefault
      minDistance={0.15}
      maxDistance={6}
      smoothTime={0.6}
      onStart={() => {
        interacting.current = true;
      }}
      onEnd={() => {
        interacting.current = false;
        idleSince.current = performance.now();
      }}
    />
  );
}

export default function GalaxyScene(props: Props) {
  return (
    <Canvas
      dpr={[1, 1.75]}
      camera={{ position: [...HOME.position], fov: 50, near: 0.01, far: 50 }}
      gl={{ antialias: true, alpha: true }}
      onPointerMissed={() => props.onHover(null, 0, 0)}
    >
      <Points
        galaxy={props.galaxy}
        colours={props.colours}
        highlight={props.highlight}
        onHover={props.onHover}
      />
      {props.marker && (
        <Billboard position={props.marker}>
          <mesh>
            <ringGeometry args={[0.009, 0.012, 48]} />
            <meshBasicMaterial color="#3ddc97" transparent opacity={0.95} depthTest={false} />
          </mesh>
        </Billboard>
      )}
      <Camera focus={props.focus} autoRotate={props.autoRotate} animate={props.animate} />
    </Canvas>
  );
}
