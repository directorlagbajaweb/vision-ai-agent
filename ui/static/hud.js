/**
 * ui/static/hud.js
 * VISION's HUD avatar: a procedural android bust — head, temple modules and
 * articulated neck — built entirely from Three.js primitives plus
 * canvas-generated plating maps (no external model or texture files).
 *
 * The facial *detail* (panel seam lines, the forehead triangle, the nose-bridge
 * seam, the mouth line, seam ambient occlusion) is painted into texture maps
 * rather than modelled, because procedural face geometry from primitives falls
 * apart fast, while a crisp albedo/bump/roughness triple over a well-shaped
 * skull reads exactly like segmented ceramic plating.
 *
 * Public contract, unchanged from the orb this replaces:
 *   window.setHudState(name)  — one of the 7 statuses vision_live.py sends
 *   --state-glow / --state-pulse CSS vars, written every frame (style.css uses both)
 */

import * as THREE from 'three';

const canvas = document.getElementById('hud-canvas');

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
camera.position.set(0, -0.40, 5.35);
camera.lookAt(0, -0.40, 0);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.98;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

// ── Studio environment ────────────────────────────────────────────────────
// Clearcoat and iridescence need something to reflect or the ceramic reads
// dead flat. RoomEnvironment is imported dynamically so an unreachable CDN
// degrades to "no reflections" instead of taking the whole HUD down with it
// (a static import failure would abort this module).
(async () => {
  try {
    const { RoomEnvironment } = await import('three/addons/environments/RoomEnvironment.js');
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
  } catch (e) {
    console.warn('[hud] studio environment unavailable, falling back to lights only:', e);
  }
})();

// ── Proportions ───────────────────────────────────────────────────────────
const HEAD_W = 0.88;
const HEAD_H = 1.04;
const HEAD_D = 0.95;

// Feature heights sit low on the sphere on purpose: with the eyes any higher
// the cranium dominates and the whole head reads as an egg.
const FACE_U = 0.25;                       // sphere UV of the front centreline
const EYES = [
  { u: FACE_U - 0.052, v: 0.510 },
  { u: FACE_U + 0.052, v: 0.510 },
];
// Sphere UV is strongly anisotropic here — one unit of u spans ~5.8 world units
// of circumference while one unit of v spans ~3.3 — so these two numbers being
// close does NOT mean a round eye. In world terms this is a ~0.19 x 0.07 almond.
const EYE_RX = 0.017;                      // painted almond half-extents, in UV
const EYE_RY = 0.011;
const SOCKET_DEPTH = 0.035;
const MOUTH_V = 0.735;
const EAR_U = 0.205;
const EAR_V = 0.518;

const COPPER = 0xb0682f;
const CHARCOAL = 0x22252b;
const DARK_MATTE = 0x1a1c21;

// ── Head shaping ──────────────────────────────────────────────────────────
/**
 * Mutates a point on the unit sphere into head proportions. Kept as a
 * standalone function so uvToHead() can place the eyes, temple modules and
 * mouth bar against the *same* surface the mesh actually has — no guessing
 * at coordinates that then drift out of the sockets.
 */
function shapeHead(p) {
  const ux = p.x, uy = p.y, uz = p.z;

  p.x *= HEAD_W;
  p.y *= HEAD_H;
  p.z *= HEAD_D;

  // Jaw taper — narrows toward the chin without coming to a point.
  const below = Math.max(0, -uy);
  const taper = 1 - 0.36 * Math.pow(below, 1.45);
  p.x *= taper;
  p.z *= taper * (1 - 0.10 * Math.pow(below, 2.4));

  const frontness = Math.max(0, uz);

  // Flatten the face plate so it reads as panels rather than a ball.
  p.z -= frontness * 0.130 * (1 - uy * uy);

  // Brow ridge above the eye line. These uy values track the painted feature
  // heights — move a seam in the texture and its ridge has to follow.
  p.z += Math.exp(-Math.pow((uy - 0.18) / 0.10, 2)) * frontness * 0.045;

  // Chin projection.
  p.z += Math.exp(-Math.pow((uy + 0.73) / 0.20, 2)) * frontness * 0.070;

  // Cheekbones — kept shallow; any stronger and they catch the fill light as
  // two pale blotches that read as blusher.
  const cheek =
    Math.exp(-Math.pow((uy + 0.28) / 0.18, 2)) *
    Math.exp(-Math.pow((Math.abs(ux) - 0.62) / 0.30, 2)) *
    frontness;
  p.x += Math.sign(ux) * cheek * 0.022;

  // Flatten the crown so the skull doesn't dome up into an egg.
  const crown = Math.max(0, uy - 0.45) / 0.55;
  p.y -= crown * crown * 0.275;

  // Occiput — a little extension at the back.
  p.z -= Math.max(0, -uz) * 0.05 * (1 - uy * uy);
}

/** Recessed eye sockets, as an almond-shaped inward push in UV space. */
function socketDepth(u, v) {
  let d = 0;
  for (const e of EYES) {
    const du = (u - e.u) / (EYE_RX + 0.006);
    const dv = (v - e.v) / (EYE_RY + 0.004);
    const r = Math.sqrt(du * du + dv * dv);
    if (r < 1) {
      const s = 1 - r;
      d = Math.max(d, s * s * (3 - 2 * s) * SOCKET_DEPTH);
    }
  }
  return d;
}

/** Unit-sphere direction for a SphereGeometry UV (matches three's own mapping). */
function uvToDir(u, v) {
  const phi = u * Math.PI * 2;
  const theta = v * Math.PI;
  return new THREE.Vector3(
    -Math.cos(phi) * Math.sin(theta),
    Math.cos(theta),
    Math.sin(phi) * Math.sin(theta)
  );
}

/** Point on the finished head surface for a given UV. */
function uvToHead(u, v, { socket = true } = {}) {
  const dir = uvToDir(u, v);
  const p = dir.clone();
  shapeHead(p);
  if (socket) p.addScaledVector(dir, -socketDepth(u, v));
  return p;
}

// ── Plating maps ──────────────────────────────────────────────────────────
// One pass paints three canvases at once — albedo, bump and roughness — so a
// seam is always a seam in all three (a dark line, a groove, and a rougher
// strip) instead of three descriptions that can drift apart.
function makePlatingMaps() {
  const W = 2048;
  const H = 1024;

  const mk = () => {
    const c = document.createElement('canvas');
    c.width = W;
    c.height = H;
    return c;
  };
  const albC = mk(), bmpC = mk(), rghC = mk();
  const alb = albC.getContext('2d');
  const bmp = bmpC.getContext('2d');
  const rgh = rghC.getContext('2d');

  // Base fills: pearl white / no displacement / glossy ceramic.
  alb.fillStyle = '#f2f1ef';
  alb.fillRect(0, 0, W, H);
  bmp.fillStyle = '#ffffff';
  bmp.fillRect(0, 0, W, H);
  rgh.fillStyle = '#242424';
  rgh.fillRect(0, 0, W, H);

  // A whisper of tonal variation so the plating isn't a uniform slab.
  const grad = alb.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, 'rgba(255,255,255,0.5)');
  grad.addColorStop(0.45, 'rgba(255,255,255,0)');
  grad.addColorStop(1, 'rgba(196,198,208,0.35)');
  alb.fillStyle = grad;
  alb.fillRect(0, 0, W, H);

  const X = (u) => u * W;
  const Y = (v) => v * H;

  const TARGETS = [
    { ctx: alb, ao: 'rgba(150,152,166,0.30)', line: 'rgba(112,114,128,0.62)', core: 'rgba(58,60,72,0.40)' },
    { ctx: bmp, ao: 'rgba(210,210,210,0.55)', line: '#6a6a6a',                core: '#2c2c2c' },
    { ctx: rgh, ao: 'rgba(150,150,150,0.35)', line: '#9c9c9c',                core: '#c4c4c4' },
  ];

  /** Stroke one polyline into all three maps: soft AO bleed, crisp line, dark core. */
  function seam(points, { width = 1, ao = 9 } = {}) {
    for (const t of TARGETS) {
      const { ctx } = t;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (const pass of [
        { color: t.ao, w: ao },
        { color: t.line, w: 2.2 * width },
        { color: t.core, w: 1.0 * width },
      ]) {
        ctx.strokeStyle = pass.color;
        ctx.lineWidth = pass.w;
        ctx.beginPath();
        points.forEach(([u, v], i) => (i ? ctx.lineTo(X(u), Y(v)) : ctx.moveTo(X(u), Y(v))));
        ctx.stroke();
      }
    }
  }

  /** Mirrored pair of seams about the facial centreline. */
  function seamPair(points, opts) {
    seam(points, opts);
    seam(points.map(([u, v]) => [2 * FACE_U - u, v]), opts);
  }

  // Latitudinal seams, wrapping the whole cranium so rotation never reveals
  // a blank hemisphere.
  const ring = (v) => seam([[0, v], [0.25, v], [0.5, v], [0.75, v], [1, v]], { width: 1, ao: 11 });
  ring(0.215);   // crown / forehead division
  ring(0.442);   // brow line
  ring(0.765);   // jawline

  // Longitudinal seams over the crown and around the back.
  seam([[FACE_U, 0.012], [FACE_U, 0.213]]);
  seamPair([[FACE_U + 0.062, 0.02], [FACE_U + 0.070, 0.213]]);
  seamPair([[FACE_U + 0.135, 0.05], [FACE_U + 0.150, 0.213]]);
  for (const u of [0.58, 0.72, 0.86, 0.98]) seam([[u, 0.055], [u, 0.860]], { ao: 8 });
  seam([[0.02, 0.055], [0.02, 0.860]], { ao: 8 });

  // Forehead split into three panels.
  seamPair([[FACE_U + 0.055, 0.217], [FACE_U + 0.055, 0.439]]);

  // Nose bridge — fine seam detail, no nostril flare. Deliberately faint; at
  // full seam weight the centreline reads as a scar down the face.
  seam([[FACE_U, 0.485], [FACE_U, 0.650]], { width: 0.55, ao: 5 });
  seamPair([[FACE_U + 0.004, 0.641], [FACE_U + 0.024, 0.653]], { width: 0.55, ao: 4 });

  // Cheek panels.
  // Kept out at the edge of the face plate — drawn closer in they read as
  // nasolabial folds rather than panel edges.
  seamPair([[FACE_U + 0.112, 0.497], [FACE_U + 0.106, 0.612], [FACE_U + 0.100, 0.726]]);
  seamPair([[FACE_U + 0.124, 0.568], [FACE_U + 0.146, 0.627], [FACE_U + 0.156, 0.688]]);

  // Jaw and chin panels. The lower chin seam is kept shallow — pulled to a
  // point it reads as an arrowhead rather than a panel edge.
  seam([[FACE_U, 0.769], [FACE_U, 0.851]]);
  seamPair([[FACE_U + 0.048, 0.769], [FACE_U + 0.082, 0.840]]);
  seam([[FACE_U - 0.095, 0.833], [FACE_U, 0.851], [FACE_U + 0.095, 0.833]], { ao: 8 });

  // Sockets. The recess is real geometry, but a bare white cavity just catches
  // the key light and blows out into a bright blob — so the socket interior is
  // painted genuinely dark and matte. That is what makes the eyes read as dark
  // almonds set into the plating instead of two hotspots.
  for (const e of EYES) {
    const cx = X(e.u), cy = Y(e.v);
    const rx = EYE_RX * W, ry = EYE_RY * H;

    // Soft outer occlusion bleeding onto the surrounding panels.
    const bleed = alb.createRadialGradient(cx, cy, ry * 0.5, cx, cy, rx * 1.5);
    bleed.addColorStop(0, 'rgba(30,32,42,0.85)');
    bleed.addColorStop(0.55, 'rgba(96,100,118,0.26)');
    bleed.addColorStop(1, 'rgba(255,255,255,0)');
    alb.fillStyle = bleed;
    alb.beginPath();
    alb.ellipse(cx, cy, rx * 1.5, ry * 1.7, 0, 0, Math.PI * 2);
    alb.fill();

    // The dark almond itself, in all three maps.
    for (const t of [
      { ctx: alb, fill: '#0d0f15' },
      { ctx: bmp, fill: '#787878' },
      { ctx: rgh, fill: '#8a8a8a' },   // matte: kills the cavity hotspot
    ]) {
      t.ctx.fillStyle = t.fill;
      t.ctx.beginPath();
      t.ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      t.ctx.fill();
    }

    // Crisp lid seam where the almond meets the plating.
    seam(
      [
        [e.u - EYE_RX - 0.006, e.v + 0.003],
        [e.u - EYE_RX * 0.5, e.v - EYE_RY - 0.010],
        [e.u + EYE_RX * 0.5, e.v - EYE_RY - 0.010],
        [e.u + EYE_RX + 0.006, e.v + 0.003],
      ],
      { width: 0.85, ao: 5 }
    );
  }

  // Forehead marking: small dark matte triangle, pointing down.
  const triW = 0.019 * W;
  const triTop = Y(0.298);
  const triBot = Y(0.353);
  const triX = X(FACE_U);
  for (const t of [
    { ctx: alb, fill: '#191b20' },
    { ctx: bmp, fill: '#8e8e8e' },
    { ctx: rgh, fill: '#dcdcdc' },   // matte, unlike the glossy plating
  ]) {
    t.ctx.fillStyle = t.fill;
    t.ctx.beginPath();
    t.ctx.moveTo(triX - triW, triTop);
    t.ctx.lineTo(triX + triW, triTop);
    t.ctx.lineTo(triX, triBot);
    t.ctx.closePath();
    t.ctx.fill();
  }

  // Closed, neutral mouth line — no lip texture, just a seam that's darker
  // and deeper than the panel seams.
  for (const t of [
    { ctx: alb, color: '#2b2d35', w: 5.5 },
    { ctx: bmp, color: '#3a3a3a', w: 6.0 },
    { ctx: rgh, color: '#8e8e8e', w: 5.5 },
  ]) {
    t.ctx.lineCap = 'round';
    t.ctx.strokeStyle = t.color;
    t.ctx.lineWidth = t.w;
    t.ctx.beginPath();
    t.ctx.moveTo(X(FACE_U - 0.038), Y(MOUTH_V));
    t.ctx.quadraticCurveTo(X(FACE_U), Y(MOUTH_V + 0.004), X(FACE_U + 0.038), Y(MOUTH_V));
    t.ctx.stroke();
  }

  const albedo = new THREE.CanvasTexture(albC);
  albedo.colorSpace = THREE.SRGBColorSpace;
  albedo.anisotropy = renderer.capabilities.getMaxAnisotropy();

  const bumpMap = new THREE.CanvasTexture(bmpC);
  const roughnessMap = new THREE.CanvasTexture(rghC);
  for (const t of [bumpMap, roughnessMap]) t.anisotropy = albedo.anisotropy;

  return { albedo, bumpMap, roughnessMap };
}

const maps = makePlatingMaps();

// ── Head mesh ─────────────────────────────────────────────────────────────
const bust = new THREE.Group();
bust.position.y = -0.05;
scene.add(bust);

const headGroup = new THREE.Group();
bust.add(headGroup);

const headGeo = new THREE.SphereGeometry(1, 160, 112);
{
  const pos = headGeo.attributes.position;
  const uv = headGeo.attributes.uv;
  const p = new THREE.Vector3();

  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i);
    shapeHead(p);
    pos.setXYZ(i, p.x, p.y, p.z);
  }
  headGeo.computeVertexNormals();

  // Second pass: sink the sockets along the *shaped* normals, so the recess
  // follows the face plate rather than the original sphere.
  const nrm = headGeo.attributes.normal;
  const n = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    const d = socketDepth(uv.getX(i), uv.getY(i));
    if (d <= 0) continue;
    p.fromBufferAttribute(pos, i);
    n.fromBufferAttribute(nrm, i);
    p.addScaledVector(n, -d);
    pos.setXYZ(i, p.x, p.y, p.z);
  }
  headGeo.computeVertexNormals();
}

const headMat = new THREE.MeshPhysicalMaterial({
  map: maps.albedo,
  bumpMap: maps.bumpMap,
  bumpScale: 0.55,
  roughnessMap: maps.roughnessMap,
  roughness: 1.0,
  metalness: 0.02,
  clearcoat: 1.0,
  clearcoatRoughness: 0.15,
  // Pearlescent: a thin-film sheen over the ceramic rather than tinted paint.
  iridescence: 0.28,
  iridescenceIOR: 1.35,
  iridescenceThicknessRange: [120, 420],
  sheen: 0.20,
  sheenColor: new THREE.Color(0xeff2ff),
  sheenRoughness: 0.55,
  envMapIntensity: 1.15,
});

const head = new THREE.Mesh(headGeo, headMat);
head.castShadow = true;
headGroup.add(head);

// ── Eyes ──────────────────────────────────────────────────────────────────
const eyeMat = new THREE.MeshPhysicalMaterial({
  color: 0x0a0c12,
  roughness: 0.28,
  metalness: 0.12,
  clearcoat: 0.85,
  clearcoatRoughness: 0.16,
  envMapIntensity: 0.55,
});

const irisMats = [];

for (const e of EYES) {
  const dir = uvToDir(e.u, e.v);
  const surface = uvToHead(e.u, e.v);

  // `surface` is the socket floor, so the lens is pushed back out until it sits
  // a hair proud of the rim — set into the recess, but not swallowed by it.
  const eye = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 32), eyeMat);
  eye.position.copy(surface).addScaledVector(dir, 0.022);
  eye.lookAt(eye.position.clone().addScaledVector(dir, 5));
  eye.scale.set(0.086, 0.032, 0.024);      // almond, not a ball
  headGroup.add(eye);

  // Subtle synthetic iris structure — a dim concentric ring, not a human iris.
  const irisMat = new THREE.MeshBasicMaterial({
    color: 0xb87a45,
    transparent: true,
    opacity: 0.0,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  irisMats.push(irisMat);

  const iris = new THREE.Mesh(new THREE.TorusGeometry(0.019, 0.0032, 8, 40), irisMat);
  iris.position.copy(surface).addScaledVector(dir, 0.044);
  iris.lookAt(iris.position.clone().addScaledVector(dir, 5));
  headGroup.add(iris);
}

// ── Temple modules ────────────────────────────────────────────────────────
const housingMat = new THREE.MeshPhysicalMaterial({
  color: CHARCOAL,
  roughness: 0.58,
  metalness: 0.35,
  clearcoat: 0.4,
  clearcoatRoughness: 0.4,
  envMapIntensity: 0.8,
});
const copperMat = new THREE.MeshPhysicalMaterial({
  color: COPPER,
  roughness: 0.38,          // brushed, not polished — a mirror finish puts a
  metalness: 0.96,          // hot band straight across every neck ring
  envMapIntensity: 1.0,
});

const indicatorMats = [];
const indicatorGlows = [];

/** Soft radial sprite, reused for the indicator bloom. */
function makeGlowTexture() {
  const size = 128;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.28, 'rgba(255,255,255,0.42)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(c);
}
const glowTexture = makeGlowTexture();

// Cylinder axis is +y by default; rotate the geometry so each module's axis
// points out of the temple once the group is aimed along the surface normal.
function disc(radius, height, mat) {
  const g = new THREE.CylinderGeometry(radius, radius, height, 56);
  g.rotateX(Math.PI / 2);
  return new THREE.Mesh(g, mat);
}

for (const side of [-1, 1]) {
  const u = FACE_U + side * EAR_U;
  const dir = uvToDir(u, EAR_V);
  const surface = uvToHead(u, EAR_V, { socket: false });

  const ear = new THREE.Group();
  // Sits right against the plating — pushed further out it reads as a floating
  // disc rather than a module mounted at the temple.
  ear.position.copy(surface).addScaledVector(dir, -0.022);
  ear.lookAt(ear.position.clone().addScaledVector(dir, 5));
  headGroup.add(ear);

  // Mount stub, deliberately fat and sunk deep: the housing's back face is flat
  // and the temple is curved, so without this the module's rim lifts off the
  // plating and the whole ear looks stuck on rather than mounted.
  const stub = disc(0.168, 0.36, housingMat);
  stub.position.z = -0.145;
  ear.add(stub);

  const housing = disc(0.178, 0.105, housingMat);
  housing.position.z = 0.028;
  housing.castShadow = true;
  ear.add(housing);

  const bezel = new THREE.Mesh(new THREE.TorusGeometry(0.167, 0.019, 12, 64), housingMat);
  bezel.position.z = 0.072;
  ear.add(bezel);

  const accent = disc(0.110, 0.026, copperMat);
  accent.position.z = 0.084;
  ear.add(accent);

  const indicatorMat = new THREE.MeshBasicMaterial({ color: 0xc87b3c, toneMapped: false });
  indicatorMats.push(indicatorMat);
  const indicator = disc(0.034, 0.018, indicatorMat);
  indicator.position.z = 0.098;
  ear.add(indicator);

  const glowMat = new THREE.SpriteMaterial({
    map: glowTexture,
    color: 0xc87b3c,
    transparent: true,
    opacity: 0.7,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  indicatorGlows.push(glowMat);
  const glow = new THREE.Sprite(glowMat);
  glow.scale.set(0.32, 0.32, 1);
  glow.position.z = 0.12;
  ear.add(glow);
}

// ── Articulated neck ──────────────────────────────────────────────────────
// Alternating brushed-copper rings and dark matte segments, widening toward
// the base. The top bands sit up inside the jaw so the neck reads as
// articulated rather than as a post the head is balanced on.
const neckMatteMat = new THREE.MeshPhysicalMaterial({
  color: DARK_MATTE,
  roughness: 0.82,
  metalness: 0.22,
  envMapIntensity: 0.6,
});

const neck = new THREE.Group();
bust.add(neck);

const BANDS = 6;
let bandY = -0.86;
for (let i = 0; i < BANDS; i++) {
  const t = i / (BANDS - 1);
  const r = 0.335 + t * 0.095;   // gentle widening — a neck, not a lampshade

  const segment = new THREE.Mesh(new THREE.CylinderGeometry(r, r + 0.010, 0.098, 48), neckMatteMat);
  segment.position.y = bandY;
  segment.receiveShadow = true;
  neck.add(segment);
  bandY -= 0.071;

  const ringR = r + 0.017;
  const ring = new THREE.Mesh(new THREE.CylinderGeometry(ringR, ringR, 0.040, 48), copperMat);
  ring.position.y = bandY;
  ring.receiveShadow = true;
  ring.castShadow = true;
  neck.add(ring);
  bandY -= 0.072;
}

const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.465, 0.560, 0.125, 56), neckMatteMat);
collar.position.y = bandY - 0.04;
collar.receiveShadow = true;
neck.add(collar);

// Cabling suggestion at the base.
const cableMat = new THREE.MeshPhysicalMaterial({ color: 0x14161a, roughness: 0.7, metalness: 0.3 });
for (const a of [-0.9, -0.3, 0.35, 1.0]) {
  const x = Math.cos(a) * 0.34;
  const z = Math.sin(a) * 0.24 + 0.14;
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(x * 0.75, bandY + 0.18, z * 0.75),
    new THREE.Vector3(x * 1.05, bandY + 0.02, z * 1.15),
    new THREE.Vector3(x * 1.15, bandY - 0.14, z * 1.25),
  ]);
  neck.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 24, 0.017, 10, false), cableMat));
}

// ── Voice bar + scan line ─────────────────────────────────────────────────
// The head is a single mesh, so there's no jaw to articulate. A thin additive
// bar over the painted mouth line carries speech instead — readable at HUD
// scale and consistent with the rest of the interface.
const mouthSurface = uvToHead(FACE_U, MOUTH_V, { socket: false });
const mouthDir = uvToDir(FACE_U, MOUTH_V);

const voiceMat = new THREE.MeshBasicMaterial({
  color: 0xff9a4d,
  transparent: true,
  opacity: 0,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
  toneMapped: false,
});
const voiceBar = new THREE.Mesh(new THREE.PlaneGeometry(0.105, 0.012), voiceMat);
voiceBar.position.copy(mouthSurface).addScaledVector(mouthDir, 0.010);
voiceBar.lookAt(voiceBar.position.clone().addScaledVector(mouthDir, 5));
headGroup.add(voiceBar);

const scanMat = new THREE.MeshBasicMaterial({
  color: 0x64d2ff,
  transparent: true,
  opacity: 0,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
  toneMapped: false,
});
const scanLine = new THREE.Mesh(new THREE.PlaneGeometry(1.75, 0.016), scanMat);
scanLine.position.z = 1.02;
headGroup.add(scanLine);

// ── Lighting: soft studio rig ─────────────────────────────────────────────
scene.add(new THREE.HemisphereLight(0x93aadd, 0x191a20, 0.72));

const keyLight = new THREE.DirectionalLight(0xffffff, 1.35);
keyLight.position.set(2.4, 3.0, 4.2);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(1024, 1024);
keyLight.shadow.camera.near = 1;
keyLight.shadow.camera.far = 14;
keyLight.shadow.camera.left = -3;
keyLight.shadow.camera.right = 3;
keyLight.shadow.camera.top = 3;
keyLight.shadow.camera.bottom = -3;
keyLight.shadow.bias = -0.0012;
keyLight.shadow.radius = 3;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xaac6ff, 0.48);
fillLight.position.set(-3.2, 0.6, 2.6);
scene.add(fillLight);

const bounceLight = new THREE.DirectionalLight(0xffb37a, 0.24);
bounceLight.position.set(0, -2.4, 2.2);
scene.add(bounceLight);

// Rim light along the cranium — state-coloured, so the whole bust shifts mood
// with what VISION is doing.
const rimLight = new THREE.DirectionalLight(0xc87b3c, 1.5);
rimLight.position.set(-0.6, 2.6, -2.8);
scene.add(rimLight);

// Warm accent from below, kept low and close so it grazes the jaw underside and
// the neck bands. Any higher and it puts a copper blush on the cheeks and a hot
// dot on the nose.
const accentLight = new THREE.PointLight(COPPER, 1.4, 2.6, 2.2);
accentLight.position.set(0, -1.62, 0.55);
scene.add(accentLight);

// ── State machine ─────────────────────────────────────────────────────────
// Names match the 7 statuses vision_live.py pushes through window.updateStatus.
const STATES = {
  idle:            { color: 0xc87b3c, ind: 0.55, rate: 1.1, eye: 0.16, motion: 1.00, scan: 0.00, voice: 0.0, pulse: 0.10, tilt:  0.000, rim: 0.95 },
  listening:       { color: 0xffb347, ind: 1.00, rate: 2.2, eye: 0.52, motion: 0.60, scan: 0.00, voice: 0.0, pulse: 0.35, tilt: -0.055, rim: 1.30 },
  processing:      { color: 0x54c8ff, ind: 0.95, rate: 4.5, eye: 0.42, motion: 1.40, scan: 1.00, voice: 0.0, pulse: 0.60, tilt:  0.020, rim: 1.15 },
  speaking:        { color: 0xff8a3d, ind: 1.00, rate: 3.0, eye: 0.62, motion: 1.10, scan: 0.00, voice: 1.0, pulse: 1.00, tilt: -0.020, rim: 1.45 },
  muted:           { color: 0x7a5058, ind: 0.13, rate: 0.5, eye: 0.03, motion: 0.30, scan: 0.00, voice: 0.0, pulse: 0.03, tilt:  0.100, rim: 0.45 },
  reconnecting:    { color: 0xffc24a, ind: 0.80, rate: 6.0, eye: 0.20, motion: 0.70, scan: 0.35, voice: 0.0, pulse: 0.45, tilt:  0.040, rim: 0.85 },
  mic_unavailable: { color: 0xff5a5a, ind: 0.28, rate: 1.0, eye: 0.05, motion: 0.25, scan: 0.00, voice: 0.0, pulse: 0.06, tilt:  0.120, rim: 0.50 },
};

const NUMERIC = ['ind', 'rate', 'eye', 'motion', 'scan', 'voice', 'pulse', 'tilt', 'rim'];

let target = STATES.idle;
const cur = { ...STATES.idle };
const targetColor = new THREE.Color(STATES.idle.color);
const curColor = new THREE.Color(STATES.idle.color);

function setHudState(stateName) {
  const next = STATES[stateName] || STATES.idle;
  target = next;
  targetColor.set(next.color);
}
window.setHudState = setHudState;

// ── Frame loop ────────────────────────────────────────────────────────────
const clock = new THREE.Clock();
const root = document.documentElement;

function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  for (const k of NUMERIC) cur[k] += (target[k] - cur[k]) * 0.05;
  curColor.lerp(targetColor, 0.05);

  // Indicator: pulses at the state's own rate.
  const beat = 0.5 + 0.5 * Math.sin(t * cur.rate * Math.PI);
  const level = cur.ind * (0.55 + 0.45 * beat);

  for (const m of indicatorMats) m.color.copy(curColor).multiplyScalar(0.35 + level);
  for (const m of indicatorGlows) {
    m.color.copy(curColor);
    m.opacity = 0.20 + level * 0.75;
  }
  for (const m of irisMats) {
    m.color.copy(curColor);
    m.opacity = cur.eye * (0.65 + 0.35 * beat);
  }

  rimLight.color.copy(curColor);
  rimLight.intensity = cur.rim;
  accentLight.color.copy(curColor);
  accentLight.intensity = 0.8 + level * 1.1;

  // Fake speech envelope — layered sines read as speech rhythm without any
  // real audio amplitude plumbed through from the backend.
  const env = Math.abs(Math.sin(t * 5.7) * 0.55 + Math.sin(t * 11.3) * 0.30 + Math.sin(t * 19.1) * 0.15);
  voiceMat.opacity = cur.voice * (0.25 + env * 0.75);
  voiceBar.scale.set(0.55 + env * 0.85, 0.7 + env * 1.9, 1);

  // Processing scan sweep down the face plate.
  if (cur.scan > 0.01) {
    const sweep = (t * 0.55) % 1;
    scanLine.position.y = 1.15 - sweep * 2.35;
    scanMat.color.copy(curColor);
    scanMat.opacity = cur.scan * 0.5 * Math.sin(sweep * Math.PI);
    scanLine.visible = true;
  } else {
    scanLine.visible = false;
  }

  // Idle life: a slow look-around plus a breathing rise and fall.
  const m = cur.motion;
  headGroup.rotation.y = Math.sin(t * 0.17) * 0.13 * m + Math.sin(t * 0.41) * 0.035 * m;
  headGroup.rotation.x = cur.tilt + Math.sin(t * 0.23) * 0.030 * m;
  headGroup.rotation.z = Math.sin(t * 0.13) * 0.016 * m;
  bust.position.y = -0.05 + Math.sin(t * 0.6) * 0.010 * m;
  neck.rotation.y = headGroup.rotation.y * 0.35;

  // style.css drives the surrounding HUD — glow, label, mute button — off both.
  root.style.setProperty('--state-glow', '#' + curColor.getHexString());
  root.style.setProperty('--state-pulse', cur.pulse.toFixed(3));

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
