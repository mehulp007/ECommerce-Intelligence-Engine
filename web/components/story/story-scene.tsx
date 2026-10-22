"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { memo, useMemo, useRef, useState, type RefObject } from "react";
import * as THREE from "three";

import { POINT_FRAGMENT, POINT_VERTEX, pointUniforms } from "@/components/three/point-shader";
import { pointColours, type Galaxy } from "@/lib/galaxy";
import {
  RISK_COLOURS,
  sceneProgress,
  sceneWeight,
  smoothstep,
  SOURCE_COLOURS,
  storyScroll,
  type StoryCustomers,
} from "@/lib/story";
import type { Overview } from "@/lib/types";

/** World-space points the camera rig can look at, written by the actors. */
const anchors = { search: null as THREE.Vector3 | null };

/** A small seeded random generator, so every visit draws the same scene. */
function random(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hex(colour: string): [number, number, number] {
  const n = parseInt(colour.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

const INDIGO = "#7c83fd";
const TEAL = "#2dd4bf";

interface ParticlesProps {
  positions: Float32Array;
  colours: Float32Array;
  highlight?: Float32Array;
  size?: number;
  points: RefObject<THREE.Points | null>;
}

/** One draw call for many round points; actors drive its uniforms each frame. */
function Particles({ positions, colours, highlight, size = 9, points }: ParticlesProps) {
  const [uniforms] = useState(() => pointUniforms(size));
  const ones = useMemo(() => new Float32Array(positions.length / 3).fill(1), [positions]);
  return (
    <points ref={points} frustumCulled={false}>
      <bufferGeometry key={positions.length}>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-aColor" args={[colours, 3]} />
        <bufferAttribute attach="attributes-aHighlight" args={[highlight ?? ones, 1]} />
      </bufferGeometry>
      <shaderMaterial
        vertexShader={POINT_VERTEX}
        fragmentShader={POINT_FRAGMENT}
        uniforms={uniforms}
        transparent
        depthWrite={false}
      />
    </points>
  );
}

/** Set a uniform on a Particles material through its ref. */
function uniform(points: RefObject<THREE.Points | null>, name: string, value: number) {
  const material = points.current?.material as THREE.ShaderMaterial | undefined;
  if (material) material.uniforms[name].value = value;
}

function GalaxyActor({ galaxy, matches, calm }: { galaxy: Galaxy; matches: number[]; calm: boolean }) {
  const group = useRef<THREE.Group>(null);
  const points = useRef<THREE.Points>(null);
  const focus = useRef(0);
  const colours = useMemo(() => {
    const out = pointColours(galaxy, "category");
    // Search matches show their real garment colour, so red dresses look red.
    const garment = pointColours(galaxy, "colour");
    for (const i of matches) out.set(garment.subarray(i * 3, i * 3 + 3), i * 3);
    return out;
  }, [galaxy, matches]);
  const highlight = useMemo(() => {
    if (!matches.length) return undefined;
    const out = new Float32Array(galaxy.count);
    for (const i of matches) out[i] = 1;
    return out;
  }, [galaxy, matches]);
  const centre = useMemo(() => {
    if (!matches.length) return null;
    const c = new THREE.Vector3();
    for (const i of matches) c.add(new THREE.Vector3().fromArray(galaxy.positions, i * 3));
    return c.divideScalar(matches.length);
  }, [galaxy, matches]);

  useFrame(({ gl }, delta) => {
    const p = storyScroll.position;
    const weight = Math.max(
      sceneWeight(p, 0),
      sceneWeight(p, 3),
      0.08 * sceneWeight(p, 5),
      sceneWeight(p, 6),
    );
    const searching = sceneWeight(p, 3) > 0.5 && sceneProgress(p, 3) > 0.42 && highlight ? 1 : 0;
    focus.current += (searching - focus.current) * Math.min(1, delta * 3);
    uniform(points, "uOpacity", weight);
    uniform(points, "uFocus", focus.current);
    uniform(points, "uPixelRatio", gl.getPixelRatio());
    if (group.current) {
      group.current.visible = weight > 0.003;
      if (!calm && sceneWeight(p, 3) < 0.5) group.current.rotation.y += delta * 0.035;
      anchors.search = centre ? centre.clone().applyMatrix4(group.current.matrixWorld) : null;
    }
  });

  return (
    <group ref={group}>
      <Particles points={points} positions={galaxy.positions} colours={colours} highlight={highlight} />
    </group>
  );
}

/** Weeks fully inside the data, as on the Overview page. */
function fullWeeks(overview: Overview) {
  const day = 86_400_000;
  const { first_date, last_date } = overview.kpis;
  return overview.weekly.filter(
    (w) => Date.parse(w.week) >= Date.parse(first_date) && (Date.parse(last_date) - Date.parse(w.week)) / day >= 6,
  );
}

/** How far along x the timeline has drawn itself at progress `t`. */
const reveal = (t: number) => -1.65 + 3.5 * smoothstep(0.02, 0.8, t) + 0.25;

function TimelineActor({ overview }: { overview: Overview }) {
  const points = useRef<THREE.Points>(null);
  const band = useRef<THREE.Mesh>(null);
  const { positions, colours, closure } = useMemo(() => {
    const weeks = fullWeeks(overview);
    const peak = Math.max(...weeks.map((w) => w.revenue_units));
    const next = random(11);
    const xs: number[] = [];
    const cs: number[] = [];
    const online = hex(INDIGO);
    const store = hex(TEAL);
    const column = 3.2 / weeks.length;
    weeks.forEach((week, i) => {
      // One column per week: its height is revenue, online at the bottom and
      // store sales stacked on top, so weeks without store sales are all indigo.
      const height = week.revenue_units / peak;
      const count = Math.round(height * 190) + 6;
      const centre = -1.6 + column * (i + 0.5);
      for (let k = 0; k < count; k++) {
        const level = next();
        xs.push(centre + (next() - 0.5) * column * 0.7, -0.6 + level * height * 1.2, (next() - 0.5) * 0.12);
        cs.push(...(level < week.online_share ? online : store));
      }
    });
    // Weeks with essentially no store sales, as found on the Overview page.
    const closed = weeks.map((w, i) => (w.online_share >= 0.999 ? i : -1)).filter((i) => i >= 0);
    const at = (i: number) => -1.6 + column * i;
    return {
      positions: new Float32Array(xs),
      colours: new Float32Array(cs),
      closure: closed.length ? { x: (at(closed[0]) + at(closed.at(-1)! + 1)) / 2, width: at(closed.at(-1)! + 1) - at(closed[0]) } : null,
    };
  }, [overview]);

  useFrame(({ gl }) => {
    const p = storyScroll.position;
    const weight = sceneWeight(p, 1);
    const t = sceneProgress(p, 1);
    uniform(points, "uOpacity", weight);
    uniform(points, "uReveal", reveal(t));
    uniform(points, "uPixelRatio", gl.getPixelRatio());
    if (points.current) points.current.visible = weight > 0.003;
    if (band.current && closure) {
      const material = band.current.material as THREE.MeshBasicMaterial;
      const shown = smoothstep(closure.x - 0.1, closure.x + 0.2, reveal(t));
      material.opacity = 0.16 * weight * shown;
      band.current.visible = material.opacity > 0.002;
    }
  });

  return (
    <group>
      <Particles points={points} positions={positions} colours={colours} size={8} />
      {closure && (
        <mesh ref={band} position={[closure.x, 0.0, -0.08]}>
          <planeGeometry args={[closure.width, 1.35]} />
          <meshBasicMaterial color="#f5b547" transparent opacity={0} depthWrite={false} />
        </mesh>
      )}
    </group>
  );
}

const POOL = new THREE.Vector3(-0.35, 0, 0);
const RANKER = new THREE.Vector3(0.35, 0, 0);

function FunnelActor({ sizes }: { sizes: number[] }) {
  const points = useRef<THREE.Points>(null);
  const ring = useRef<THREE.Mesh>(null);
  const slots = useRef<THREE.Group>(null);
  const plan = useMemo(() => {
    const next = random(5);
    const particles: { source: number; offset: THREE.Vector3; delay: number; slot: number }[] = [];
    sizes.forEach((size, source) => {
      for (let k = 0; k < Math.round(size / 2); k++) {
        const offset = new THREE.Vector3(next() - 0.5, next() - 0.5, next() - 0.5).multiplyScalar(0.42);
        particles.push({ source, offset, delay: next() * 0.14, slot: -1 });
      }
    });
    // Twelve candidates, from different sources, make it to the final list.
    for (let s = 0; s < 12; s++) particles[Math.floor((s * particles.length) / 12)].slot = s;
    const colours = new Float32Array(particles.length * 3);
    particles.forEach((p, i) => colours.set(hex(SOURCE_COLOURS[p.source % SOURCE_COLOURS.length]), i * 3));
    return { particles, colours, positions: new Float32Array(particles.length * 3) };
  }, [sizes]);

  const origin = (source: number) => new THREE.Vector3(-1.25, 0.7 - (source * 1.4) / Math.max(1, sizes.length - 1), 0);
  const slot = (s: number) => new THREE.Vector3(0.82 + (s % 4) * 0.16, 0.25 - Math.floor(s / 4) * 0.25, 0);

  useFrame(({ gl, clock }) => {
    const p = storyScroll.position;
    const weight = sceneWeight(p, 2);
    const t = sceneProgress(p, 2);
    if (points.current) {
      points.current.visible = weight > 0.003;
      const geometry = points.current.geometry;
      const position = geometry.getAttribute("position") as THREE.BufferAttribute;
      const highlight = geometry.getAttribute("aHighlight") as THREE.BufferAttribute;
      const v = new THREE.Vector3();
      plan.particles.forEach((particle, i) => {
        const gather = smoothstep(particle.delay, 0.3 + particle.delay, t);
        const rank = smoothstep(0.42, 0.62, t);
        const deliver = smoothstep(0.7, 0.92, t);
        const wobble = Math.sin(clock.elapsedTime * 1.3 + i) * 0.02;
        v.copy(origin(particle.source)).lerp(POOL.clone().add(particle.offset), gather);
        v.y += Math.sin(gather * Math.PI) * 0.15 + wobble;
        if (particle.slot >= 0) {
          v.lerp(RANKER, rank).lerp(slot(particle.slot), deliver);
        }
        position.setXYZ(i, v.x, v.y, v.z);
        // Candidates the ranker drops fade out as it scores them.
        highlight.setX(i, particle.slot >= 0 ? 1 : 1 - rank);
      });
      position.needsUpdate = true;
      highlight.needsUpdate = true;
    }
    uniform(points, "uOpacity", weight);
    uniform(points, "uFocus", 1);
    uniform(points, "uPixelRatio", gl.getPixelRatio());
    if (ring.current) {
      const material = ring.current.material as THREE.MeshBasicMaterial;
      material.opacity = weight * smoothstep(0.3, 0.45, t) * 0.9;
      ring.current.visible = material.opacity > 0.002;
      ring.current.rotation.z = clock.elapsedTime * 0.4;
    }
    if (slots.current) {
      slots.current.visible = weight > 0.003;
      slots.current.children.forEach((child) => {
        const material = (child as THREE.Mesh).material as THREE.MeshBasicMaterial;
        material.opacity = weight * (0.1 + 0.25 * smoothstep(0.75, 0.95, t));
      });
    }
  });

  return (
    <group scale={0.82} position={[-0.12, 0, 0]}>
      <Particles points={points} positions={plan.positions} colours={plan.colours} size={14} />
      <mesh ref={ring} position={RANKER}>
        <torusGeometry args={[0.18, 0.006, 12, 96]} />
        <meshBasicMaterial color={INDIGO} transparent opacity={0} depthWrite={false} />
      </mesh>
      <group ref={slots}>
        {Array.from({ length: 12 }, (_, s) => (
          <mesh key={s} position={slot(s)}>
            <planeGeometry args={[0.12, 0.18]} />
            <meshBasicMaterial color={INDIGO} transparent opacity={0} depthWrite={false} />
          </mesh>
        ))}
      </group>
    </group>
  );
}


function CustomersActor({ customers, targeted }: { customers: StoryCustomers; targeted: number[] }) {
  const points = useRef<THREE.Points>(null);
  const focus = useRef(0);
  const { positions, colours } = useMemo(() => {
    const scaled = new Float32Array(customers.positions.length);
    for (let i = 0; i < customers.count; i++) {
      scaled[i * 3] = customers.positions[i * 3] * 1.25;
      scaled[i * 3 + 1] = customers.positions[i * 3 + 1] * 0.85;
      scaled[i * 3 + 2] = customers.positions[i * 3 + 2] * 1.0;
    }
    const c = new Float32Array(customers.count * 3);
    for (let i = 0; i < customers.count; i++) c.set(hex(RISK_COLOURS[customers.risk[i]] ?? "#8a90a6"), i * 3);
    return { positions: scaled, colours: c };
  }, [customers]);
  const highlight = useMemo(() => {
    if (!targeted.length) return undefined;
    const out = new Float32Array(customers.count);
    for (const i of targeted) out[i] = 1;
    return out;
  }, [customers, targeted]);

  useFrame(({ gl }, delta) => {
    const p = storyScroll.position;
    const weight = sceneWeight(p, 4);
    const want = highlight && sceneProgress(p, 4) > 0.35 ? 1 : 0;
    focus.current += (want - focus.current) * Math.min(1, delta * 3);
    uniform(points, "uOpacity", weight);
    uniform(points, "uFocus", focus.current);
    uniform(points, "uDim", 0.16);
    uniform(points, "uPixelRatio", gl.getPixelRatio());
    if (points.current) points.current.visible = weight > 0.003;
  });

  return <Particles points={points} positions={positions} colours={colours} highlight={highlight} size={10} />;
}

const HOME = new THREE.Vector3(0, 0.3, 2.6);

/** Where the camera should be for scene `index` at progress `t`. */
function shot(index: number, t: number, aspect: number): { position: THREE.Vector3; look: THREE.Vector3 } {
  const fit = aspect < 1 ? Math.min(2.2, 1 / aspect) * 0.9 : 1;
  const look = new THREE.Vector3();
  let position: THREE.Vector3;
  switch (index) {
    case 0:
      position = new THREE.Vector3(0, 0.25, 2.1).lerp(new THREE.Vector3(0, 0.4, 3.0), smoothstep(0, 1, t));
      break;
    case 1: {
      // Fly along the timeline from 2018 to 2020.
      const x = -1.15 + 2.3 * smoothstep(0.02, 0.85, t);
      position = new THREE.Vector3(x - 0.35, 0.2, 2.3);
      look.set(x + 0.2, -0.05, 0);
      break;
    }
    case 2:
      position = new THREE.Vector3(-0.1, 0.12, 2.7 + 0.25 * (1 - t));
      look.set(-0.1, 0, 0);
      break;
    case 3: {
      const dive = smoothstep(0.45, 0.95, t);
      position = new THREE.Vector3(0, 0.4, 3.0);
      if (anchors.search && dive > 0) {
        const target = anchors.search.clone();
        const close = target.clone().add(target.clone().normalize().multiplyScalar(0.55)).add(new THREE.Vector3(0, 0.08, 0));
        position.lerp(close, dive);
        look.lerp(target, dive);
      }
      return { position, look };
    }
    case 4: {
      const angle = -0.7 + 1.4 * t;
      position = new THREE.Vector3(Math.sin(angle) * 3.6, 0.75, Math.cos(angle) * 3.6);
      break;
    }
    case 5:
      position = new THREE.Vector3(0, 0.2, 3.6);
      break;
    default:
      position = new THREE.Vector3(0, 0.55, 4.0).lerp(new THREE.Vector3(0, 0.35, 3.1), smoothstep(0, 1, t));
  }
  position.multiplyScalar(fit);
  return { position, look };
}

function Rig({ calm }: { calm: boolean }) {
  const look = useRef(new THREE.Vector3());
  useFrame(({ camera, size }, delta) => {
    const p = storyScroll.position;
    const index = Math.min(6, Math.max(0, Math.floor(p)));
    const aspect = size.width / Math.max(1, size.height);
    const target = shot(index, p - index, aspect);
    // On wide screens the text sits on the left, so frame the 3D on the right.
    if (aspect > 1.15) {
      const pan = 0.26 * target.position.distanceTo(target.look);
      target.position.x -= pan;
      target.look.x -= pan;
    }
    // Ease towards the shot, so scene changes glide instead of cutting.
    const ease = calm ? 1 : 1 - Math.exp(-delta * 2.6);
    camera.position.lerp(target.position, ease);
    look.current.lerp(target.look, ease);
    camera.lookAt(look.current);
  });
  return null;
}

export interface StorySceneProps {
  galaxy: Galaxy | undefined;
  overview: Overview | undefined;
  sources: number[] | undefined;
  customers: StoryCustomers | undefined;
  searchMatches: number[];
  targeted: number[];
  calm: boolean;
}

function StoryScene({ galaxy, overview, sources, customers, searchMatches, targeted, calm }: StorySceneProps) {
  return (
    <Canvas
      dpr={[1, 1.5]}
      camera={{ position: HOME.toArray(), fov: 50, near: 0.01, far: 50 }}
      gl={{ antialias: true, alpha: true }}
    >
      <Rig calm={calm} />
      {galaxy && <GalaxyActor galaxy={galaxy} matches={searchMatches} calm={calm} />}
      {overview && <TimelineActor overview={overview} />}
      {sources && <FunnelActor sizes={sources} />}
      {customers && <CustomersActor customers={customers} targeted={targeted} />}
    </Canvas>
  );
}

export default memo(StoryScene);
