import * as THREE from 'three';

const canvas = document.getElementById('orb-canvas');

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(0, 0, 6);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

// ── Cheap 3D value noise (sum-of-sines) — organic wobble without a noise library ──
function noise3(x, y, z) {
  return (
    Math.sin(x * 1.7 + y * 0.9 - z * 1.3) * 0.4 +
    Math.sin(y * 2.1 - x * 1.1 + z * 0.7) * 0.35 +
    Math.sin(z * 1.4 + x * 0.6 + y * 1.9) * 0.25
  );
}

// ── Soft dot sprite, reused for the particle sphere + center flare ──
function makeDotTexture() {
  const size = 64;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.4, 'rgba(255,255,255,0.6)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(c);
}
const dotTexture = makeDotTexture();

const orbGroup = new THREE.Group();
scene.add(orbGroup);

// ── Dense particle sphere — wavy/organic rather than perfectly round ──
const PARTICLE_COUNT = 3600;
const BASE_RADIUS = 1.85;
const baseDirs = new Float32Array(PARTICLE_COUNT * 3);
const livePositions = new Float32Array(PARTICLE_COUNT * 3);

const goldenAngle = Math.PI * (3 - Math.sqrt(5));
for (let i = 0; i < PARTICLE_COUNT; i++) {
  const yfrac = 1 - (i / (PARTICLE_COUNT - 1)) * 2;
  const radiusAtY = Math.sqrt(Math.max(0, 1 - yfrac * yfrac));
  const theta = goldenAngle * i;
  const x = Math.cos(theta) * radiusAtY;
  const z = Math.sin(theta) * radiusAtY;
  baseDirs[i * 3] = x;
  baseDirs[i * 3 + 1] = yfrac;
  baseDirs[i * 3 + 2] = z;
  livePositions[i * 3] = x * BASE_RADIUS;
  livePositions[i * 3 + 1] = yfrac * BASE_RADIUS;
  livePositions[i * 3 + 2] = z * BASE_RADIUS;
}

const sphereGeo = new THREE.BufferGeometry();
sphereGeo.setAttribute('position', new THREE.BufferAttribute(livePositions, 3));

const sphereMat = new THREE.PointsMaterial({
  size: 0.032,
  map: dotTexture,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  color: 0x3fa8ff,
});
const particleSphere = new THREE.Points(sphereGeo, sphereMat);
orbGroup.add(particleSphere);

// ── Radiating cone beam — small contained burst right at the center ──
const BEAM_COUNT = 260;
const BEAM_AXIS = new THREE.Vector3(1, 0.12, -0.05).normalize();
const BEAM_HALF_ANGLE = 0.5;

const upHint = Math.abs(BEAM_AXIS.y) < 0.99 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
const beamTangent = new THREE.Vector3().crossVectors(upHint, BEAM_AXIS).normalize();
const beamBitangent = new THREE.Vector3().crossVectors(BEAM_AXIS, beamTangent).normalize();

const beamPositions = new Float32Array(BEAM_COUNT * 2 * 3);
const beamColors = new Float32Array(BEAM_COUNT * 2 * 3);
const beamDirs = [];
const beamLens = [];

for (let i = 0; i < BEAM_COUNT; i++) {
  const cosT = 1 - Math.random() * (1 - Math.cos(BEAM_HALF_ANGLE));
  const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
  const phi = Math.random() * Math.PI * 2;
  const dir = new THREE.Vector3()
    .addScaledVector(beamTangent, Math.cos(phi) * sinT)
    .addScaledVector(beamBitangent, Math.sin(phi) * sinT)
    .addScaledVector(BEAM_AXIS, cosT)
    .normalize();
  beamDirs.push(dir);
  beamLens.push(BASE_RADIUS * (0.14 + Math.random() * 0.18));
}

const beamGeo = new THREE.BufferGeometry();
beamGeo.setAttribute('position', new THREE.BufferAttribute(beamPositions, 3));
beamGeo.setAttribute('color', new THREE.BufferAttribute(beamColors, 3));

const beamMat = new THREE.LineBasicMaterial({
  vertexColors: true,
  transparent: true,
  opacity: 0.9,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});
const beam = new THREE.LineSegments(beamGeo, beamMat);
orbGroup.add(beam);

// Bright flare where the beam originates
const flareMat = new THREE.SpriteMaterial({
  map: dotTexture,
  color: 0xffffff,
  transparent: true,
  opacity: 0.95,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});
const flare = new THREE.Sprite(flareMat);
flare.scale.set(0.22, 0.22, 1);
orbGroup.add(flare);

// ── State-driven motion — idle drifts gently, speaking simulates audio reactivity ──
const STATE_PRESETS = {
  idle:            { speed: 0.05,  amp: 0.05, freq: 1.1, flow: 0.18, pulse: 0.12 },
  listening:       { speed: 0.10,  amp: 0.08, freq: 1.5, flow: 0.4,  pulse: 0.35 },
  speaking:        { speed: 0.18,  amp: 0.20, freq: 2.6, flow: 1.3,  pulse: 1.0 },
  muted:           { speed: 0.015, amp: 0.02, freq: 0.8, flow: 0.05, pulse: 0.04 },
  reconnecting:    { speed: 0.12,  amp: 0.06, freq: 1.0, flow: 0.5,  pulse: 0.45 },
  mic_unavailable: { speed: 0.015, amp: 0.02, freq: 0.8, flow: 0.05, pulse: 0.04 },
};

let target = STATE_PRESETS.idle;
const cur = { ...STATE_PRESETS.idle };

function setOrbState(stateName) {
  target = STATE_PRESETS[stateName] || STATE_PRESETS.idle;
}
window.setOrbState = setOrbState;

// ── Continuously cycling color palette — blue / purple-white / copper, all sweeping together ──
const HUE_BLUE = 0.58;
const HUE_PURPLE = 0.76;
const HUE_COPPER = 0.07;
const sphereColor = new THREE.Color();
const beamStartColor = new THREE.Color();
const beamEndColor = new THREE.Color();

const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  cur.speed += (target.speed - cur.speed) * 0.03;
  cur.amp += (target.amp - cur.amp) * 0.04;
  cur.freq += (target.freq - cur.freq) * 0.04;
  cur.flow += (target.flow - cur.flow) * 0.04;
  cur.pulse += (target.pulse - cur.pulse) * 0.04;

  const hueShift = (t * 0.015) % 1;
  sphereColor.setHSL((HUE_BLUE + hueShift) % 1, 0.85, 0.55);
  beamStartColor.setHSL((HUE_PURPLE + hueShift) % 1, 0.35, 0.9);
  beamEndColor.setHSL((HUE_COPPER + hueShift) % 1, 0.85, 0.55);

  sphereMat.color.copy(sphereColor);
  flare.material.color.copy(beamStartColor);

  // Organic wavy distortion — flickers harder when "speaking" to fake audio reactivity
  const flowT = t * cur.flow;
  const flicker = cur.pulse * 0.06;
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const i3 = i * 3;
    const bx = baseDirs[i3], by = baseDirs[i3 + 1], bz = baseDirs[i3 + 2];
    const n = noise3(bx * cur.freq + flowT, by * cur.freq - flowT * 0.7, bz * cur.freq + flowT * 0.5);
    const jitter = flicker ? Math.sin(t * 9 + i * 0.7) * flicker : 0;
    const r = BASE_RADIUS + n * cur.amp + jitter;
    livePositions[i3] = bx * r;
    livePositions[i3 + 1] = by * r;
    livePositions[i3 + 2] = bz * r;
  }
  sphereGeo.attributes.position.needsUpdate = true;

  // Beam lengths flicker with the same reactivity; colors sweep the same palette
  const beamFlickerAmp = 0.08 + cur.pulse * 0.5;
  for (let i = 0; i < BEAM_COUNT; i++) {
    const dir = beamDirs[i];
    const len = beamLens[i] * (1 + Math.sin(t * 14 + i * 1.3) * beamFlickerAmp);
    const i6 = i * 6;
    beamPositions[i6] = 0; beamPositions[i6 + 1] = 0; beamPositions[i6 + 2] = 0;
    beamPositions[i6 + 3] = dir.x * len;
    beamPositions[i6 + 4] = dir.y * len;
    beamPositions[i6 + 5] = dir.z * len;

    beamColors[i6] = beamStartColor.r; beamColors[i6 + 1] = beamStartColor.g; beamColors[i6 + 2] = beamStartColor.b;
    beamColors[i6 + 3] = beamEndColor.r; beamColors[i6 + 4] = beamEndColor.g; beamColors[i6 + 5] = beamEndColor.b;
  }
  beamGeo.attributes.position.needsUpdate = true;
  beamGeo.attributes.color.needsUpdate = true;

  flare.scale.setScalar(0.22 * (1 + Math.sin(t * (3 + cur.pulse * 4)) * (0.15 + cur.pulse * 0.3)));

  orbGroup.rotation.y = t * cur.speed;
  orbGroup.rotation.x = Math.sin(t * cur.speed * 0.4) * 0.15;

  const hex = '#' + sphereColor.getHexString();
  document.documentElement.style.setProperty('--state-glow', hex);
  document.documentElement.style.setProperty('--state-pulse', cur.pulse.toFixed(3));

  renderer.render(scene, camera);
}
animate();

function resize() {
  const size = canvas.parentElement.clientWidth;
  renderer.setSize(size, size, false);
  camera.aspect = 1;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();
