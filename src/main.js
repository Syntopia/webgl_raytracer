import { createLogger } from "./logger.js";
import { loadGltfFromText } from "./gltf.js";
import { buildUnifiedBVH, flattenBVH } from "./bvh.js";
import {
  packBvhNodes, packTriangles, packTriNormals, packTriColors, packPrimIndices,
  packSpheres, packSphereColors, packCylinders, packCylinderColors
} from "./packing.js";
import { loadHDR, buildEnvSamplingData } from "./hdr.js";
import {
  parsePDB,
  parseSDF,
  parseAutoDetect,
  moleculeToGeometry,
  splitMolDataByHetatm,
  fetchPDB,
  getBuiltinMolecule,
  BUILTIN_MOLECULES
} from "./molecular.js";
import { computeSES, sesToTriangles } from "./surface.js";
import {
  initWebGL,
  createDataTexture,
  createEnvTexture,
  createCdfTexture,
  createAccumTargets,
  resizeAccumTargets,
  createTextureUnit,
  setTraceUniforms,
  setDisplayUniforms,
  drawFullscreen,
  MAX_BRUTE_FORCE_TRIS
} from "./webgl.js";

const canvas = document.getElementById("view");
const renderOverlay = document.getElementById("renderOverlay");
const loadingOverlay = document.getElementById("loadingOverlay");
const statusEl = document.getElementById("status");
const logger = createLogger(statusEl);

const exampleSelect = document.getElementById("exampleSelect");
const loadExampleBtn = document.getElementById("loadExample");
const envSelect = document.getElementById("envSelect");
const envIntensityInput = document.getElementById("envIntensity");
const envMaxLumInput = document.getElementById("envMaxLum");
const fileInput = document.getElementById("fileInput");
const molFileInput = document.getElementById("molFileInput");
const pdbIdInput = document.getElementById("pdbIdInput");
const loadPdbIdBtn = document.getElementById("loadPdbId");
const pdbDisplayStyle = document.getElementById("pdbDisplayStyle");
const pdbAtomScale = document.getElementById("pdbAtomScale");
const pdbBondRadius = document.getElementById("pdbBondRadius");
const showSurfaceToggle = document.getElementById("showSurface");
const probeRadiusInput = document.getElementById("probeRadius");
const surfaceResolutionInput = document.getElementById("surfaceResolution");
const smoothNormalsToggle = document.getElementById("smoothNormals");
const renderBtn = document.getElementById("renderBtn");
const scaleSelect = document.getElementById("scaleSelect");
const fastScaleSelect = document.getElementById("fastScaleSelect");
const bruteforceToggle = document.getElementById("bruteforceToggle");
const useGltfColorToggle = document.getElementById("useGltfColor");
const baseColorInput = document.getElementById("baseColor");
const materialSelect = document.getElementById("materialSelect");
const metallicInput = document.getElementById("metallic");
const roughnessInput = document.getElementById("roughness");
const matteSpecularInput = document.getElementById("matteSpecular");
const matteRoughnessInput = document.getElementById("matteRoughness");
const matteDiffuseRoughnessInput = document.getElementById("matteDiffuseRoughness");
const wrapDiffuseInput = document.getElementById("wrapDiffuse");
const maxBouncesInput = document.getElementById("maxBounces");
const exposureInput = document.getElementById("exposure");
const ambientIntensityInput = document.getElementById("ambientIntensity");
const ambientColorInput = document.getElementById("ambientColor");
const rayBiasInput = document.getElementById("rayBias");
const tMinInput = document.getElementById("tMin");
const samplesPerBounceInput = document.getElementById("samplesPerBounce");
const maxFramesInput = document.getElementById("maxFrames");
const toneMapSelect = document.getElementById("toneMapSelect");
const shadowToggle = document.getElementById("shadowToggle");
const light1Enable = document.getElementById("light1Enable");
const light1Azimuth = document.getElementById("light1Azimuth");
const light1Elevation = document.getElementById("light1Elevation");
const light1Intensity = document.getElementById("light1Intensity");
const light1Extent = document.getElementById("light1Extent");
const light1Color = document.getElementById("light1Color");
const light2Enable = document.getElementById("light2Enable");
const light2Azimuth = document.getElementById("light2Azimuth");
const light2Elevation = document.getElementById("light2Elevation");
const light2Intensity = document.getElementById("light2Intensity");
const light2Extent = document.getElementById("light2Extent");
const light2Color = document.getElementById("light2Color");
const visModeSelect = document.getElementById("visModeSelect");

const tabButtons = Array.from(document.querySelectorAll("[data-tab-button]"));
const tabPanels = Array.from(document.querySelectorAll("[data-tab-panel]"));


let sceneData = null;
let glState = null;
let isRendering = false;
let isLoading = false;
let loggedFirstFrame = false;
let glInitFailed = false;

const cameraState = {
  target: [0, 0, 0],
  distance: 4,
  rotation: [0, 0, 0, 1],
  fov: Math.PI / 3,
  width: 1,
  height: 1
};

const renderState = {
  renderScale: 1.0,
  fastScale: 0.25,
  scale: 1.0,
  frameIndex: 0,
  cameraDirty: true,
  useBvh: true,
  useGltfColor: true,
  baseColor: [0.8, 0.8, 0.8],
  materialMode: "metallic",
  metallic: 0.0,
  roughness: 0.4,
  matteSpecular: 0.03,
  matteRoughness: 0.5,
  matteDiffuseRoughness: 0.5,
  wrapDiffuse: 0.2,
  maxBounces: 2,
  maxFrames: 100,
  exposure: 1.0,
  toneMap: "aces",
  ambientIntensity: 0.0,
  ambientColor: [1.0, 1.0, 1.0],
  envUrl: null,
  envIntensity: 1.0,
  envMaxLuminance: 200.0,
  envData: null,
  rayBias: 1e-5,
  tMin: 1e-5,
  samplesPerBounce: 1,
  castShadows: true,
  lights: [
    // Camera-relative studio lighting: key, fill, rim
    { enabled: true, azimuth: -40, elevation: -30, intensity: 5.0, angle: 22, color: [1.0, 1.0, 1.0] },
    { enabled: true, azimuth: 40, elevation: 0, intensity: 0.6, angle: 50, color: [1.0, 1.0, 1.0] },
    { enabled: true, azimuth: 170, elevation: 10, intensity: 0.35, angle: 6, color: [1.0, 1.0, 1.0] }
  ],
  visMode: 0
};

const envCache = new Map();

const inputState = {
  dragging: false,
  lastX: 0,
  lastY: 0,
  arcballPrev: null,
  keys: new Set()
};

const interactionState = {
  lastActive: 0
};

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}`);
  }
  return await res.text();
}

async function loadGltfText(text, baseUrl = null) {
  logger.info("Parsing glTF");
  const { positions, indices, normals, triColors } = await loadGltfFromText(text, baseUrl, fetch);
  logger.info(`Loaded ${positions.length / 3} vertices, ${indices.length / 3} triangles`);
  if (positions.length === 0 || indices.length === 0) {
    throw new Error(
      `Loaded empty geometry (positions: ${positions.length}, indices: ${indices.length}).`
    );
  }

  // Empty sphere/cylinder arrays for glTF (will be populated by SDF/PDB loader)
  const spheres = [];
  const cylinders = [];

  logger.info("Building unified SAH BVH on CPU");
  const bvh = buildUnifiedBVH(
    { positions, indices },
    spheres,
    cylinders,
    { maxLeafSize: 4, maxDepth: 32 }
  );
  logger.info(`BVH nodes: ${bvh.nodes.length}, primitives: ${bvh.primitives.length}`);

  const flat = flattenBVH(bvh.nodes, bvh.primitives, bvh.triCount, bvh.sphereCount, bvh.cylinderCount);

  sceneData = {
    positions,
    indices,
    normals,
    triColors,
    nodes: bvh.nodes,
    tris: bvh.tris,
    primitives: bvh.primitives,
    primIndexBuffer: flat.primIndexBuffer,
    triCount: bvh.triCount,
    sphereCount: bvh.sphereCount,
    cylinderCount: bvh.cylinderCount,
    spheres,
    cylinders,
    sceneScale: 1.0
  };
  renderState.frameIndex = 0;
  renderState.cameraDirty = true;

  const bounds = computeBounds(positions);
  if (bounds) {
    logger.info(
      `Bounds min (${bounds.minX.toFixed(2)}, ${bounds.minY.toFixed(2)}, ${bounds.minZ.toFixed(2)}) max (${bounds.maxX.toFixed(2)}, ${bounds.maxY.toFixed(2)}, ${bounds.maxZ.toFixed(2)})`
    );
    const dx = bounds.maxX - bounds.minX;
    const dy = bounds.maxY - bounds.minY;
    const dz = bounds.maxZ - bounds.minZ;
    sceneData.sceneScale = Math.max(1e-3, Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5);
    const suggestedBias = Math.max(1e-5, sceneData.sceneScale * 1e-5);
    rayBiasInput.value = suggestedBias.toFixed(6);
    tMinInput.value = suggestedBias.toFixed(6);
    renderState.rayBias = suggestedBias;
    renderState.tMin = suggestedBias;
    applyCameraToBounds(bounds);
  }
}

/**
 * Create a test scene with spheres and cylinders for debugging.
 * This can be called manually or hooked up to a UI element.
 */
function loadTestPrimitives() {
  logger.info("Creating test scene with spheres and cylinders");

  // Empty triangle data
  const positions = new Float32Array(0);
  const indices = new Uint32Array(0);
  const normals = new Float32Array(0);
  const triColors = new Float32Array(0);

  // Test spheres - a small molecule-like arrangement
  const spheres = [
    { center: [0, 0, 0], radius: 0.5, color: [1.0, 0.2, 0.2] },    // Central red sphere
    { center: [1.2, 0, 0], radius: 0.35, color: [0.2, 0.2, 1.0] }, // Right blue sphere
    { center: [-1.2, 0, 0], radius: 0.35, color: [0.2, 1.0, 0.2] }, // Left green sphere
    { center: [0, 1.2, 0], radius: 0.35, color: [1.0, 1.0, 0.2] }, // Top yellow sphere
    { center: [0, -1.2, 0], radius: 0.35, color: [1.0, 0.5, 0.0] }, // Bottom orange sphere
  ];

  // Test cylinders - bonds connecting spheres
  const cylinders = [
    { p1: [0.5, 0, 0], p2: [0.85, 0, 0], radius: 0.1, color: [0.8, 0.8, 0.8] }, // Center to right
    { p1: [-0.5, 0, 0], p2: [-0.85, 0, 0], radius: 0.1, color: [0.8, 0.8, 0.8] }, // Center to left
    { p1: [0, 0.5, 0], p2: [0, 0.85, 0], radius: 0.1, color: [0.8, 0.8, 0.8] }, // Center to top
    { p1: [0, -0.5, 0], p2: [0, -0.85, 0], radius: 0.1, color: [0.8, 0.8, 0.8] }, // Center to bottom
  ];

  logger.info(`Created ${spheres.length} spheres, ${cylinders.length} cylinders`);

  const bvh = buildUnifiedBVH(
    { positions, indices },
    spheres,
    cylinders,
    { maxLeafSize: 4, maxDepth: 32 }
  );
  logger.info(`BVH nodes: ${bvh.nodes.length}, primitives: ${bvh.primitives.length}`);

  const flat = flattenBVH(bvh.nodes, bvh.primitives, bvh.triCount, bvh.sphereCount, bvh.cylinderCount);

  sceneData = {
    positions,
    indices,
    normals,
    triColors,
    nodes: bvh.nodes,
    tris: bvh.tris,
    primitives: bvh.primitives,
    primIndexBuffer: flat.primIndexBuffer,
    triCount: bvh.triCount,
    sphereCount: bvh.sphereCount,
    cylinderCount: bvh.cylinderCount,
    spheres,
    cylinders,
    sceneScale: 1.0
  };
  renderState.frameIndex = 0;
  renderState.cameraDirty = true;

  // Compute bounds from spheres and cylinders
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (const s of spheres) {
    minX = Math.min(minX, s.center[0] - s.radius);
    minY = Math.min(minY, s.center[1] - s.radius);
    minZ = Math.min(minZ, s.center[2] - s.radius);
    maxX = Math.max(maxX, s.center[0] + s.radius);
    maxY = Math.max(maxY, s.center[1] + s.radius);
    maxZ = Math.max(maxZ, s.center[2] + s.radius);
  }

  for (const c of cylinders) {
    minX = Math.min(minX, c.p1[0] - c.radius, c.p2[0] - c.radius);
    minY = Math.min(minY, c.p1[1] - c.radius, c.p2[1] - c.radius);
    minZ = Math.min(minZ, c.p1[2] - c.radius, c.p2[2] - c.radius);
    maxX = Math.max(maxX, c.p1[0] + c.radius, c.p2[0] + c.radius);
    maxY = Math.max(maxY, c.p1[1] + c.radius, c.p2[1] + c.radius);
    maxZ = Math.max(maxZ, c.p1[2] + c.radius, c.p2[2] + c.radius);
  }

  const bounds = { minX, minY, minZ, maxX, maxY, maxZ };
  logger.info(
    `Bounds min (${bounds.minX.toFixed(2)}, ${bounds.minY.toFixed(2)}, ${bounds.minZ.toFixed(2)}) max (${bounds.maxX.toFixed(2)}, ${bounds.maxY.toFixed(2)}, ${bounds.maxZ.toFixed(2)})`
  );

  const dx = bounds.maxX - bounds.minX;
  const dy = bounds.maxY - bounds.minY;
  const dz = bounds.maxZ - bounds.minZ;
  sceneData.sceneScale = Math.max(1e-3, Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5);
  const suggestedBias = Math.max(1e-5, sceneData.sceneScale * 1e-5);
  rayBiasInput.value = suggestedBias.toFixed(6);
  tMinInput.value = suggestedBias.toFixed(6);
  renderState.rayBias = suggestedBias;
  renderState.tMin = suggestedBias;
  applyCameraToBounds(bounds);

  // Reset textures so they get recreated
  if (glState) {
    glState.textures = null;
  }

  logger.info("Test primitives loaded.");
}

// Expose for debugging in console
window.loadTestPrimitives = loadTestPrimitives;

/**
 * Generate a large test scene with many random spheres.
 * Uses a seeded pseudo-random number generator for reproducibility.
 */
function loadRandomSpheres(count) {
  logger.info(`Creating test scene with ${count} random spheres`);

  // Simple seeded PRNG (mulberry32)
  let seed = 12345;
  function random() {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // Empty triangle data
  const positions = new Float32Array(0);
  const indices = new Uint32Array(0);
  const normals = new Float32Array(0);
  const triColors = new Float32Array(0);

  // Generate random spheres in a cube
  const spheres = [];
  const cubeSize = Math.pow(count, 1/3) * 2; // Scale cube size with sphere count
  const minRadius = 0.15;
  const maxRadius = 0.4;

  for (let i = 0; i < count; i++) {
    const x = (random() - 0.5) * cubeSize;
    const y = (random() - 0.5) * cubeSize;
    const z = (random() - 0.5) * cubeSize;
    const radius = minRadius + random() * (maxRadius - minRadius);

    // Random vibrant colors
    const hue = random();
    const saturation = 0.6 + random() * 0.4;
    const lightness = 0.4 + random() * 0.3;
    const color = hslToRgb(hue, saturation, lightness);

    spheres.push({ center: [x, y, z], radius, color });
  }

  // No cylinders for this test
  const cylinders = [];

  logger.info(`Generated ${spheres.length} spheres, building BVH...`);
  const startTime = performance.now();

  const bvh = buildUnifiedBVH(
    { positions, indices },
    spheres,
    cylinders,
    { maxLeafSize: 4, maxDepth: 32 }
  );

  const bvhTime = performance.now() - startTime;
  logger.info(`BVH built in ${bvhTime.toFixed(1)}ms: ${bvh.nodes.length} nodes, ${bvh.primitives.length} primitives`);

  const flat = flattenBVH(bvh.nodes, bvh.primitives, bvh.triCount, bvh.sphereCount, bvh.cylinderCount);

  sceneData = {
    positions,
    indices,
    normals,
    triColors,
    nodes: bvh.nodes,
    tris: bvh.tris,
    primitives: bvh.primitives,
    primIndexBuffer: flat.primIndexBuffer,
    triCount: bvh.triCount,
    sphereCount: bvh.sphereCount,
    cylinderCount: bvh.cylinderCount,
    spheres,
    cylinders,
    sceneScale: 1.0
  };
  renderState.frameIndex = 0;
  renderState.cameraDirty = true;

  // Compute bounds from spheres
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (const s of spheres) {
    minX = Math.min(minX, s.center[0] - s.radius);
    minY = Math.min(minY, s.center[1] - s.radius);
    minZ = Math.min(minZ, s.center[2] - s.radius);
    maxX = Math.max(maxX, s.center[0] + s.radius);
    maxY = Math.max(maxY, s.center[1] + s.radius);
    maxZ = Math.max(maxZ, s.center[2] + s.radius);
  }

  const bounds = { minX, minY, minZ, maxX, maxY, maxZ };
  logger.info(
    `Bounds min (${bounds.minX.toFixed(2)}, ${bounds.minY.toFixed(2)}, ${bounds.minZ.toFixed(2)}) max (${bounds.maxX.toFixed(2)}, ${bounds.maxY.toFixed(2)}, ${bounds.maxZ.toFixed(2)})`
  );

  const dx = bounds.maxX - bounds.minX;
  const dy = bounds.maxY - bounds.minY;
  const dz = bounds.maxZ - bounds.minZ;
  sceneData.sceneScale = Math.max(1e-3, Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5);
  const suggestedBias = Math.max(1e-5, sceneData.sceneScale * 1e-5);
  rayBiasInput.value = suggestedBias.toFixed(6);
  tMinInput.value = suggestedBias.toFixed(6);
  renderState.rayBias = suggestedBias;
  renderState.tMin = suggestedBias;
  applyCameraToBounds(bounds);

  // Reset textures so they get recreated
  if (glState) {
    glState.textures = null;
  }

  logger.info(`${count} spheres loaded successfully.`);
}

// HSL to RGB conversion helper
function hslToRgb(h, s, l) {
  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  return [r, g, b];
}

// Expose for debugging in console
window.loadRandomSpheres = loadRandomSpheres;

/**
 * Load a molecular file (PDB or SDF) and render it.
 * @param {string} text - File content
 * @param {string} filename - Original filename for format detection
 */
/**
 * Get current molecular display options from UI.
 */
function getMolecularDisplayOptions() {
  const style = pdbDisplayStyle?.value || "ball-and-stick";
  const atomScale = parseFloat(pdbAtomScale?.value) || 1.0;
  const bondRadius = parseFloat(pdbBondRadius?.value) || 0.12;

  // Adjust based on display style
  if (style === "vdw") {
    // Van der Waals: full-size atoms, no bonds shown
    return { radiusScale: 1.0, bondRadius: 0, showBonds: false };
  } else if (style === "stick") {
    // Stick: tiny atoms, normal bonds
    return { radiusScale: 0.15, bondRadius: bondRadius, showBonds: true };
  } else {
    // Ball and stick: scaled atoms with bonds
    return { radiusScale: atomScale, bondRadius: bondRadius, showBonds: true };
  }
}

function loadMolecularFile(text, filename) {
  logger.info(`Parsing molecular file: ${filename}`);

  const molData = parseAutoDetect(text, filename);
  logger.info(`Parsed ${molData.atoms.length} atoms, ${molData.bonds.length} bonds`);

  const options = getMolecularDisplayOptions();
  const { spheres, cylinders } = moleculeToGeometry(molData, options);

  loadMolecularGeometry(spheres, cylinders, molData, options);
}

/**
 * Load molecular geometry from spheres and cylinders.
 */
function loadMolecularGeometry(spheres, cylinders, molData = null, options = null) {
  const showSurface = showSurfaceToggle?.checked || false;
  const displayOptions = options ?? getMolecularDisplayOptions();
  const split = molData ? splitMolDataByHetatm(molData) : null;
  const heteroGeometry = split ? moleculeToGeometry(split.hetero, displayOptions) : { spheres: [], cylinders: [] };

  // If showing surface, hide atoms/bonds unless it's VdW mode
  let displaySpheres = spheres;
  let displayCylinders = cylinders;

  let positions = new Float32Array(0);
  let indices = new Uint32Array(0);
  let normals = new Float32Array(0);
  let triColors = new Float32Array(0);

  if (showSurface && molData && molData.atoms && molData.atoms.length > 0) {
    const surfaceAtoms = split ? split.standard.atoms : molData.atoms;
    if (surfaceAtoms.length === 0) {
      logger.warn("No non-HETATM atoms available for surface; rendering atoms only.");
    } else {
      const probeRadius = parseFloat(probeRadiusInput?.value) || 1.4;
    const resolution = parseFloat(surfaceResolutionInput?.value) || 0.25;
      const smoothNormals = smoothNormalsToggle?.checked || false;

      logger.info(
        `Computing SES surface (probe=${probeRadius}Å, resolution=${resolution}Å, smoothNormals=${smoothNormals})...`
      );
      const surfaceStart = performance.now();

      // Use VdW radii from molecular data
      const ELEMENT_RADII = {
        H: 1.20, C: 1.70, N: 1.55, O: 1.52, S: 1.80, P: 1.80,
        F: 1.47, Cl: 1.75, Br: 1.85, I: 1.98, DEFAULT: 1.70
      };

      const atoms = surfaceAtoms.map(a => ({
        center: a.position,
        radius: ELEMENT_RADII[a.element] || ELEMENT_RADII.DEFAULT
      }));

      const sesMesh = computeSES(atoms, { probeRadius, resolution, smoothNormals });
      const surfaceTime = performance.now() - surfaceStart;

      if (sesMesh.vertices.length > 0) {
        logger.info(`SES computed in ${surfaceTime.toFixed(0)}ms: ${sesMesh.indices.length / 3} triangles`);

        const surfaceData = sesToTriangles(sesMesh, [0.7, 0.75, 0.9]);
        positions = surfaceData.positions;
        indices = surfaceData.indices;
        normals = surfaceData.normals;
        triColors = surfaceData.triColors;

        // Keep HETATM atoms/bonds visible.
        displaySpheres = heteroGeometry.spheres;
        displayCylinders = heteroGeometry.cylinders;
      } else {
        logger.warn("SES computation produced no surface");
      }
    }
  }

  logger.info(`Loading ${displaySpheres.length} atoms, ${displayCylinders.length} bonds, ${indices.length / 3} triangles`);

  const startTime = performance.now();

  const bvh = buildUnifiedBVH(
    { positions, indices },
    displaySpheres,
    displayCylinders,
    { maxLeafSize: 4, maxDepth: 32 }
  );

  const bvhTime = performance.now() - startTime;
  logger.info(`BVH built in ${bvhTime.toFixed(1)}ms: ${bvh.nodes.length} nodes`);

  const flat = flattenBVH(bvh.nodes, bvh.primitives, bvh.triCount, bvh.sphereCount, bvh.cylinderCount);

  sceneData = {
    positions,
    indices,
    normals,
    triColors,
    nodes: bvh.nodes,
    tris: bvh.tris,
    primitives: bvh.primitives,
    primIndexBuffer: flat.primIndexBuffer,
    triCount: bvh.triCount,
    sphereCount: bvh.sphereCount,
    cylinderCount: bvh.cylinderCount,
    spheres: displaySpheres,
    cylinders: displayCylinders,
    sceneScale: 1.0
  };
  renderState.frameIndex = 0;
  renderState.cameraDirty = true;

  // Compute bounds from all geometry
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (const s of displaySpheres) {
    minX = Math.min(minX, s.center[0] - s.radius);
    minY = Math.min(minY, s.center[1] - s.radius);
    minZ = Math.min(minZ, s.center[2] - s.radius);
    maxX = Math.max(maxX, s.center[0] + s.radius);
    maxY = Math.max(maxY, s.center[1] + s.radius);
    maxZ = Math.max(maxZ, s.center[2] + s.radius);
  }

  for (const c of displayCylinders) {
    minX = Math.min(minX, c.p1[0] - c.radius, c.p2[0] - c.radius);
    minY = Math.min(minY, c.p1[1] - c.radius, c.p2[1] - c.radius);
    minZ = Math.min(minZ, c.p1[2] - c.radius, c.p2[2] - c.radius);
    maxX = Math.max(maxX, c.p1[0] + c.radius, c.p2[0] + c.radius);
    maxY = Math.max(maxY, c.p1[1] + c.radius, c.p2[1] + c.radius);
    maxZ = Math.max(maxZ, c.p1[2] + c.radius, c.p2[2] + c.radius);
  }

  // Include surface triangles in bounds
  for (let i = 0; i < positions.length; i += 3) {
    minX = Math.min(minX, positions[i]);
    minY = Math.min(minY, positions[i + 1]);
    minZ = Math.min(minZ, positions[i + 2]);
    maxX = Math.max(maxX, positions[i]);
    maxY = Math.max(maxY, positions[i + 1]);
    maxZ = Math.max(maxZ, positions[i + 2]);
  }

  const bounds = { minX, minY, minZ, maxX, maxY, maxZ };
  logger.info(
    `Bounds: (${bounds.minX.toFixed(1)}, ${bounds.minY.toFixed(1)}, ${bounds.minZ.toFixed(1)}) to (${bounds.maxX.toFixed(1)}, ${bounds.maxY.toFixed(1)}, ${bounds.maxZ.toFixed(1)})`
  );

  const dx = bounds.maxX - bounds.minX;
  const dy = bounds.maxY - bounds.minY;
  const dz = bounds.maxZ - bounds.minZ;
  sceneData.sceneScale = Math.max(1e-3, Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5);
  const suggestedBias = Math.max(1e-5, sceneData.sceneScale * 1e-5);
  rayBiasInput.value = suggestedBias.toFixed(6);
  tMinInput.value = suggestedBias.toFixed(6);
  renderState.rayBias = suggestedBias;
  renderState.tMin = suggestedBias;
  applyCameraToBounds(bounds);

  if (glState) {
    glState.textures = null;
  }

  logger.info("Molecular structure loaded.");
}

/**
 * Fetch and load a PDB file by ID from RCSB.
 */
async function loadPDBById(pdbId) {
  logger.info(`Fetching PDB: ${pdbId}`);
  const molData = await fetchPDB(pdbId);
  logger.info(`Parsed ${molData.atoms.length} atoms, ${molData.bonds.length} bonds`);

  const options = getMolecularDisplayOptions();
  const { spheres, cylinders } = moleculeToGeometry(molData, options);

  loadMolecularGeometry(spheres, cylinders, molData, options);
}

/**
 * Load a built-in small molecule by name.
 */
function loadBuiltinMolecule(name) {
  logger.info(`Loading built-in molecule: ${name}`);
  const molData = getBuiltinMolecule(name);
  logger.info(`Parsed ${molData.atoms.length} atoms, ${molData.bonds.length} bonds`);

  const options = getMolecularDisplayOptions();
  const { spheres, cylinders } = moleculeToGeometry(molData, options);

  loadMolecularGeometry(spheres, cylinders, molData, options);
}

// Expose for debugging in console
window.loadMolecularFile = loadMolecularFile;
window.loadPDBById = loadPDBById;
window.loadBuiltinMolecule = loadBuiltinMolecule;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function formatPolyCount(count) {
  if (!Number.isFinite(count)) return "0";
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1).replace(/\.0$/, "")}K`;
  return String(count);
}

function setLoadingOverlay(visible, message = "Loading...") {
  if (!loadingOverlay) return;
  if (visible) {
    loadingOverlay.textContent = message;
    loadingOverlay.style.display = "block";
  } else {
    loadingOverlay.style.display = "none";
  }
}

function markInteractionActive(time = performance.now()) {
  interactionState.lastActive = time;
}

function isCameraInteracting(time = performance.now()) {
  if (inputState.dragging || inputState.keys.size > 0) {
    return true;
  }
  return time - interactionState.lastActive < 120;
}

function hexToRgb(hex) {
  const value = hex.startsWith("#") ? hex.slice(1) : hex;
  if (value.length !== 6) return [1, 1, 1];
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return [r / 255, g / 255, b / 255];
}

// Deprecated: kept for potential world-space lights.
function lightDirFromAngles(azimuthDeg, elevationDeg) {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  const x = Math.cos(el) * Math.sin(az);
  const y = Math.sin(el);
  const z = Math.cos(el) * Math.cos(az);
  return [x, y, z];
}

function resetAccumulation(reason) {
  renderState.frameIndex = 0;
  renderState.cameraDirty = true;
  loggedFirstFrame = false;
  if (renderOverlay) {
    renderOverlay.style.display = "none";
  }
  if (reason) {
    logger.info(reason);
  }
}

function setActiveTab(name) {
  tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tabButton === name);
  });
  tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.tabPanel === name);
  });
}

function updateMaterialState() {
  renderState.useGltfColor = useGltfColorToggle.checked;
  renderState.baseColor = hexToRgb(baseColorInput.value);
  renderState.materialMode = materialSelect?.value || "metallic";
  renderState.metallic = clamp(Number(metallicInput.value), 0, 1);
  renderState.roughness = clamp(Number(roughnessInput.value), 0.02, 1);
  renderState.matteSpecular = clamp(Number(matteSpecularInput?.value ?? 0.03), 0.0, 0.08);
  renderState.matteRoughness = clamp(Number(matteRoughnessInput?.value ?? 0.5), 0.1, 1.0);
  renderState.matteDiffuseRoughness = clamp(Number(matteDiffuseRoughnessInput?.value ?? 0.5), 0.0, 1.0);
  renderState.wrapDiffuse = clamp(Number(wrapDiffuseInput?.value ?? 0.2), 0.0, 0.5);
  renderState.maxBounces = clamp(Number(maxBouncesInput.value), 0, 6);
  renderState.exposure = clamp(Number(exposureInput.value), 0, 5);
  renderState.ambientIntensity = clamp(Number(ambientIntensityInput.value), 0, 2);
  renderState.ambientColor = hexToRgb(ambientColorInput.value);
  renderState.envIntensity = clamp(Number(envIntensityInput.value), 0, 5);
  renderState.rayBias = clamp(Number(rayBiasInput.value), 0, 1);
  renderState.tMin = clamp(Number(tMinInput.value), 0, 1);
  renderState.samplesPerBounce = clamp(Number(samplesPerBounceInput.value), 1, 8);
  renderState.castShadows = shadowToggle.checked;
  renderState.toneMap = toneMapSelect?.value || "reinhard";
  resetAccumulation("Material settings updated.");
}

function updateMaterialVisibility() {
  const mode = materialSelect?.value || "metallic";
  const metallicGroup = document.querySelector(".material-metallic");
  const matteGroup = document.querySelector(".material-matte");
  if (metallicGroup) metallicGroup.style.display = mode === "metallic" ? "block" : "none";
  if (matteGroup) matteGroup.style.display = mode === "matte" ? "block" : "none";
}

function updateRenderLimits() {
  const raw = Number(maxFramesInput?.value);
  const maxFrames = clamp(Number.isFinite(raw) ? Math.floor(raw) : 0, 0, 2000);
  renderState.maxFrames = maxFrames;
}

async function loadEnvironment(url) {
  if (!url) {
    renderState.envUrl = null;
    renderState.envData = null;
    if (glState) {
      if (glState.envTex && glState.envTex !== glState.blackEnvTex) {
        glState.gl.deleteTexture(glState.envTex);
      }
      glState.envTex = glState.blackEnvTex;
      glState.envSize = [1, 1];
      glState.envUrl = null;
    }
    return;
  }

  if (envCache.has(url)) {
    renderState.envData = envCache.get(url);
  } else {
    logger.info(`Loading environment: ${url}`);
    const env = await loadHDR(url, logger);
    // Build importance sampling data
    env.samplingData = buildEnvSamplingData(env.data, env.width, env.height);
    envCache.set(url, env);
    renderState.envData = env;
  }
  renderState.envUrl = url;

  if (glState && renderState.envData) {
    if (glState.envTex && glState.envTex !== glState.blackEnvTex) {
      glState.gl.deleteTexture(glState.envTex);
    }
    if (glState.envMarginalCdfTex) {
      glState.gl.deleteTexture(glState.envMarginalCdfTex);
    }
    if (glState.envConditionalCdfTex) {
      glState.gl.deleteTexture(glState.envConditionalCdfTex);
    }
    glState.envTex = createEnvTexture(
      glState.gl,
      renderState.envData.width,
      renderState.envData.height,
      renderState.envData.data
    );
    // Create CDF textures for importance sampling
    const samplingData = renderState.envData.samplingData;
    glState.envMarginalCdfTex = createCdfTexture(
      glState.gl,
      samplingData.marginalCdf,
      samplingData.height + 1,
      1
    );
    glState.envConditionalCdfTex = createCdfTexture(
      glState.gl,
      samplingData.conditionalCdf,
      samplingData.width + 1,
      samplingData.height
    );
    glState.envSize = [renderState.envData.width, renderState.envData.height];
    glState.envUrl = url;
  }
}

async function updateEnvironmentState() {
  setLoadingOverlay(true, "Loading environment...");
  renderState.envIntensity = clamp(Number(envIntensityInput.value), 0, 5);
  renderState.envMaxLuminance = clamp(Number(envMaxLumInput?.value ?? 50), 0, 500);
  const url = envSelect.value || null;
  if (url !== renderState.envUrl) {
    try {
      await loadEnvironment(url);
      resetAccumulation("Environment updated.");
    } catch (err) {
      logger.error(err.message || String(err));
    }
  } else {
    resetAccumulation("Environment intensity updated.");
  }
  setLoadingOverlay(false);
}

function updateLightState() {
  renderState.lights[0] = {
    enabled: light1Enable.checked,
    azimuth: Number(light1Azimuth.value),
    elevation: Number(light1Elevation.value),
    intensity: clamp(Number(light1Intensity.value), 0, 20),
    angle: clamp(Number(light1Extent.value), 0, 60),
    color: hexToRgb(light1Color.value)
  };
  renderState.lights[1] = {
    enabled: light2Enable.checked,
    azimuth: Number(light2Azimuth.value),
    elevation: Number(light2Elevation.value),
    intensity: clamp(Number(light2Intensity.value), 0, 20),
    angle: clamp(Number(light2Extent.value), 0, 60),
    color: hexToRgb(light2Color.value)
  };
  resetAccumulation("Lighting updated.");
}

function cameraRelativeLightDir(azimuthDeg, elevationDeg, forward, right, up) {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  const cosEl = Math.cos(el);
  const sinEl = Math.sin(el);
  const sinAz = Math.sin(az);
  const cosAz = Math.cos(az);
  const lx = right[0] * cosEl * sinAz + up[0] * sinEl + forward[0] * cosEl * cosAz;
  const ly = right[1] * cosEl * sinAz + up[1] * sinEl + forward[1] * cosEl * cosAz;
  const lz = right[2] * cosEl * sinAz + up[2] * sinEl + forward[2] * cosEl * cosAz;
  const len = Math.hypot(lx, ly, lz) || 1;
  return [lx / len, ly / len, lz / len];
}

function computeBounds(positions) {
  if (!positions || positions.length < 3) return null;
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }
  return { minX, minY, minZ, maxX, maxY, maxZ };
}

function applyCameraToBounds(bounds) {
  const cx = (bounds.minX + bounds.maxX) * 0.5;
  const cy = (bounds.minY + bounds.maxY) * 0.5;
  const cz = (bounds.minZ + bounds.maxZ) * 0.5;
  const dx = bounds.maxX - bounds.minX;
  const dy = bounds.maxY - bounds.minY;
  const dz = bounds.maxZ - bounds.minZ;
  const radius = Math.max(1e-3, Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5);
  const distance = radius / Math.tan(cameraState.fov / 2) * 1.4;
  cameraState.target = [cx, cy, cz];
  cameraState.distance = distance;
  cameraState.rotation = [0, 0, 0, 1];
  renderState.cameraDirty = true;
  renderState.frameIndex = 0;
  logger.info(
    `Camera fit to bounds center (${cx.toFixed(2)}, ${cy.toFixed(2)}, ${cz.toFixed(2)}) radius ${radius.toFixed(2)}`
  );
}

function normalizeQuat(q) {
  const len = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

function quatFromAxisAngle(axis, angle) {
  const half = angle * 0.5;
  const s = Math.sin(half);
  return normalizeQuat([axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(half)]);
}

function quatMultiply(a, b) {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const bx = b[0], by = b[1], bz = b[2], bw = b[3];
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz
  ];
}

function quatRotateVec(q, v) {
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const vx = v[0], vy = v[1], vz = v[2];
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  return [
    vx + qw * tx + (qy * tz - qz * ty),
    vy + qw * ty + (qz * tx - qx * tz),
    vz + qw * tz + (qx * ty - qy * tx)
  ];
}

function computeCameraVectors() {
  const { rotation, distance, target, fov, width, height } = cameraState;
  const forward = quatRotateVec(rotation, [0, 0, 1]);
  const origin = [
    target[0] - forward[0] * distance,
    target[1] - forward[1] * distance,
    target[2] - forward[2] * distance
  ];

  const up = quatRotateVec(rotation, [0, 1, 0]);
  const right = [
    forward[1] * up[2] - forward[2] * up[1],
    forward[2] * up[0] - forward[0] * up[2],
    forward[0] * up[1] - forward[1] * up[0]
  ];
  const rightLen = Math.hypot(right[0], right[1], right[2]) || 1;
  right[0] /= rightLen;
  right[1] /= rightLen;
  right[2] /= rightLen;

  const upOrtho = [
    right[1] * forward[2] - right[2] * forward[1],
    right[2] * forward[0] - right[0] * forward[2],
    right[0] * forward[1] - right[1] * forward[0]
  ];

  const aspect = width / height;
  const scale = Math.tan(fov / 2);
  const rightScaled = [right[0] * scale * aspect, right[1] * scale * aspect, right[2] * scale * aspect];
  const upScaled = [upOrtho[0] * scale, upOrtho[1] * scale, upOrtho[2] * scale];

  return {
    origin,
    forward: [forward[0], forward[1], forward[2]],
    right: rightScaled,
    up: upScaled,
    width,
    height
  };
}

function updateCameraFromInput(dt) {
  if (inputState.keys.size === 0) return false;
  const moveSpeed = cameraState.distance * 0.6 * dt;
  const orbit = computeCameraVectors();
  const forward = [orbit.forward[0], orbit.forward[1], orbit.forward[2]];
  const right = [
    orbit.right[0] / Math.hypot(orbit.right[0], orbit.right[1], orbit.right[2]),
    orbit.right[1] / Math.hypot(orbit.right[0], orbit.right[1], orbit.right[2]),
    orbit.right[2] / Math.hypot(orbit.right[0], orbit.right[1], orbit.right[2])
  ];

  let moved = false;

  if (inputState.keys.has("w")) {
    cameraState.target[0] += forward[0] * moveSpeed;
    cameraState.target[1] += forward[1] * moveSpeed;
    cameraState.target[2] += forward[2] * moveSpeed;
    moved = true;
  }
  if (inputState.keys.has("s")) {
    cameraState.target[0] -= forward[0] * moveSpeed;
    cameraState.target[1] -= forward[1] * moveSpeed;
    cameraState.target[2] -= forward[2] * moveSpeed;
    moved = true;
  }
  if (inputState.keys.has("a")) {
    cameraState.target[0] -= right[0] * moveSpeed;
    cameraState.target[1] -= right[1] * moveSpeed;
    cameraState.target[2] -= right[2] * moveSpeed;
    moved = true;
  }
  if (inputState.keys.has("d")) {
    cameraState.target[0] += right[0] * moveSpeed;
    cameraState.target[1] += right[1] * moveSpeed;
    cameraState.target[2] += right[2] * moveSpeed;
    moved = true;
  }
  if (inputState.keys.has("q")) {
    cameraState.target[1] += moveSpeed;
    moved = true;
  }
  if (inputState.keys.has("e")) {
    cameraState.target[1] -= moveSpeed;
    moved = true;
  }

  return moved;
}

function arcballVector(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * 2 - 1;
  const y = 1 - ((clientY - rect.top) / rect.height) * 2;
  const len2 = x * x + y * y;
  if (len2 <= 1) {
    return [x, y, Math.sqrt(1 - len2)];
  }
  const len = Math.sqrt(len2) || 1;
  return [x / len, y / len, 0];
}

function ensureWebGL() {
  if (!glState) {
    if (glInitFailed) {
      throw new Error("WebGL initialization previously failed.");
    }
    try {
      const { gl, traceProgram, displayProgram, vao } = initWebGL(canvas, logger);
      const blackEnvTex = createEnvTexture(gl, 1, 1, new Float32Array([0, 0, 0, 1]));
      // Create dummy CDF textures (1x1 with value 1.0) for when no env is loaded
      const dummyCdfTex = createCdfTexture(gl, new Float32Array([0, 1]), 2, 1);
      glState = {
        gl,
        traceProgram,
        displayProgram,
        vao,
        textures: null,
        accum: null,
        frameParity: 0,
        envTex: blackEnvTex,
        blackEnvTex,
        envMarginalCdfTex: dummyCdfTex,
        envConditionalCdfTex: dummyCdfTex,
        dummyCdfTex,
        envSize: [1, 1],
        envUrl: null
      };
    } catch (err) {
      glInitFailed = true;
      throw err;
    }
  }
  return glState;
}

function uploadSceneTextures(gl, maxTextureSize) {
  const bvh = packBvhNodes(sceneData.nodes, maxTextureSize);
  const tris = packTriangles(sceneData.tris, sceneData.positions, maxTextureSize);
  const triNormals = packTriNormals(sceneData.tris, sceneData.normals, maxTextureSize);
  const triColors = packTriColors(sceneData.triColors, maxTextureSize);
  const primIndices = packPrimIndices(sceneData.primIndexBuffer, maxTextureSize);
  const spheresPacked = packSpheres(sceneData.spheres, maxTextureSize);
  const sphereColors = packSphereColors(sceneData.spheres, maxTextureSize);
  const cylindersPacked = packCylinders(sceneData.cylinders, maxTextureSize);
  const cylinderColors = packCylinderColors(sceneData.cylinders, maxTextureSize);

  const bvhTex = createDataTexture(gl, bvh.width, bvh.height, bvh.data);
  const triTex = createDataTexture(gl, tris.width, tris.height, tris.data);
  const triNormalTex = createDataTexture(gl, triNormals.width, triNormals.height, triNormals.data);
  const triColorTex = createDataTexture(gl, triColors.width, triColors.height, triColors.data);
  const primIndexTex = createDataTexture(gl, primIndices.width, primIndices.height, primIndices.data);
  const sphereTex = createDataTexture(gl, spheresPacked.width, spheresPacked.height, spheresPacked.data);
  const sphereColorTex = createDataTexture(gl, sphereColors.width, sphereColors.height, sphereColors.data);
  const cylinderTex = createDataTexture(gl, cylindersPacked.width, cylindersPacked.height, cylindersPacked.data);
  const cylinderColorTex = createDataTexture(gl, cylinderColors.width, cylinderColors.height, cylinderColors.data);

  return {
    bvh,
    tris,
    triNormals,
    triColors,
    primIndices,
    spheresPacked,
    sphereColors,
    cylindersPacked,
    cylinderColors,
    bvhTex,
    triTex,
    triNormalTex,
    triColorTex,
    primIndexTex,
    sphereTex,
    sphereColorTex,
    cylinderTex,
    cylinderColorTex,
  };
}

function renderFrame() {
  if (!sceneData) {
    logger.warn("No scene loaded yet.");
    return;
  }
  const { gl, traceProgram, displayProgram, vao } = ensureWebGL();

  const displayWidth = Math.max(1, Math.floor(canvas.clientWidth));
  const displayHeight = Math.max(1, Math.floor(canvas.clientHeight));
  const scale = renderState.scale;
  const renderWidth = Math.max(1, Math.floor(displayWidth * scale));
  const renderHeight = Math.max(1, Math.floor(displayHeight * scale));

  if (renderWidth <= 1 || renderHeight <= 1) {
    logger.warn(`Canvas size is too small: ${renderWidth}x${renderHeight}`);
    return;
  }

  if (!loggedFirstFrame) {
    logger.info(`Rendering ${renderWidth}x${renderHeight} (scale ${scale.toFixed(2)}x)`);
    loggedFirstFrame = true;
  }

  canvas.width = displayWidth;
  canvas.height = displayHeight;
  cameraState.width = renderWidth;
  cameraState.height = renderHeight;

  if (!glState.textures) {
    const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    logger.info(`Uploading textures (MAX_TEXTURE_SIZE ${maxTextureSize})`);
    glState.textures = uploadSceneTextures(gl, maxTextureSize);
  }

  if (renderState.envData && glState.envUrl !== renderState.envUrl) {
    if (glState.envTex && glState.envTex !== glState.blackEnvTex) {
      gl.deleteTexture(glState.envTex);
    }
    glState.envTex = createEnvTexture(
      gl,
      renderState.envData.width,
      renderState.envData.height,
      renderState.envData.data
    );
    glState.envSize = [renderState.envData.width, renderState.envData.height];
    glState.envUrl = renderState.envUrl;
  }

  if (!renderState.useBvh && sceneData.triCount > MAX_BRUTE_FORCE_TRIS) {
    throw new Error(
      `Brute force mode supports up to ${MAX_BRUTE_FORCE_TRIS} triangles; scene has ${sceneData.triCount}.`
    );
  }

  if (!glState.accum) {
    logger.info("Allocating accumulation targets");
    glState.accum = createAccumTargets(gl, renderWidth, renderHeight);
    renderState.frameIndex = 0;
  } else {
    glState.accum = resizeAccumTargets(gl, glState.accum, renderWidth, renderHeight);
  }

  const camera = computeCameraVectors();
  const forwardLen = Math.hypot(camera.forward[0], camera.forward[1], camera.forward[2]) || 1;
  const rightLen = Math.hypot(camera.right[0], camera.right[1], camera.right[2]) || 1;
  const upLen = Math.hypot(camera.up[0], camera.up[1], camera.up[2]) || 1;
  const camForward = [camera.forward[0] / forwardLen, camera.forward[1] / forwardLen, camera.forward[2] / forwardLen];
  const camRight = [camera.right[0] / rightLen, camera.right[1] / rightLen, camera.right[2] / rightLen];
  const camUp = [camera.up[0] / upLen, camera.up[1] / upLen, camera.up[2] / upLen];
  const lightDirs = renderState.lights.map((light) =>
    cameraRelativeLightDir(light.azimuth, light.elevation, camForward, camRight, camUp)
  );

  gl.disable(gl.DEPTH_TEST);
  gl.bindVertexArray(vao);

  const accumIndex = glState.frameParity % 2;
  const prevIndex = (glState.frameParity + 1) % 2;

  if (renderState.cameraDirty) {
    renderState.frameIndex = 0;
  }

  gl.viewport(0, 0, renderWidth, renderHeight);
  gl.bindFramebuffer(gl.FRAMEBUFFER, glState.accum.framebuffers[accumIndex]);

  createTextureUnit(gl, glState.textures.bvhTex, 0);
  createTextureUnit(gl, glState.textures.triTex, 1);
  createTextureUnit(gl, glState.textures.triNormalTex, 2);
  createTextureUnit(gl, glState.textures.triColorTex, 3);
  createTextureUnit(gl, glState.textures.primIndexTex, 4);
  createTextureUnit(gl, glState.accum.textures[prevIndex], 5);
  createTextureUnit(gl, glState.envTex || glState.blackEnvTex, 6);
  createTextureUnit(gl, glState.envMarginalCdfTex || glState.dummyCdfTex, 7);
  createTextureUnit(gl, glState.envConditionalCdfTex || glState.dummyCdfTex, 8);
  createTextureUnit(gl, glState.textures.sphereTex, 9);
  createTextureUnit(gl, glState.textures.sphereColorTex, 10);
  createTextureUnit(gl, glState.textures.cylinderTex, 11);
  createTextureUnit(gl, glState.textures.cylinderColorTex, 12);

setTraceUniforms(gl, traceProgram, {
    bvhUnit: 0,
    triUnit: 1,
    triNormalUnit: 2,
    triColorUnit: 3,
    primIndexUnit: 4,
    accumUnit: 5,
    envUnit: 6,
    sphereUnit: 9,
    sphereColorUnit: 10,
    cylinderUnit: 11,
    cylinderColorUnit: 12,
    camOrigin: camera.origin,
    camRight: camera.right,
    camUp: camera.up,
    camForward: camera.forward,
    resolution: [renderWidth, renderHeight],
    bvhTexSize: [glState.textures.bvh.width, glState.textures.bvh.height],
    triTexSize: [glState.textures.tris.width, glState.textures.tris.height],
    triNormalTexSize: [glState.textures.triNormals.width, glState.textures.triNormals.height],
    triColorTexSize: [glState.textures.triColors.width, glState.textures.triColors.height],
    primIndexTexSize: [glState.textures.primIndices.width, glState.textures.primIndices.height],
    sphereTexSize: [glState.textures.spheresPacked.width, glState.textures.spheresPacked.height],
    cylinderTexSize: [glState.textures.cylindersPacked.width, glState.textures.cylindersPacked.height],
    envTexSize: glState.envSize || [1, 1],
    frameIndex: renderState.frameIndex,
    triCount: sceneData.triCount,
    sphereCount: sceneData.sphereCount,
    cylinderCount: sceneData.cylinderCount,
    useBvh: renderState.useBvh ? 1 : 0,
    useGltfColor: renderState.useGltfColor ? 1 : 0,
    baseColor: renderState.baseColor,
    metallic: renderState.metallic,
    roughness: renderState.roughness,
    maxBounces: renderState.maxBounces,
    exposure: renderState.exposure,
    ambientIntensity: renderState.ambientIntensity,
    ambientColor: renderState.ambientColor,
    envIntensity: renderState.envIntensity,
    envMaxLuminance: renderState.envMaxLuminance,
    useEnv: renderState.envUrl ? 1 : 0,
    materialMode: renderState.materialMode,
    matteSpecular: renderState.matteSpecular,
    matteRoughness: renderState.matteRoughness,
    matteDiffuseRoughness: renderState.matteDiffuseRoughness,
    wrapDiffuse: renderState.wrapDiffuse,
    envMarginalCdfUnit: 7,
    envConditionalCdfUnit: 8,
    envSize: glState.envSize || [1, 1],
    samplesPerBounce: renderState.samplesPerBounce,
    castShadows: renderState.castShadows ? 1 : 0,
    rayBias: renderState.rayBias,
    tMin: renderState.tMin,
    lights: renderState.lights,
    lightDirs,
    visMode: renderState.visMode
  });

  gl.useProgram(traceProgram);
  drawFullscreen(gl);

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, displayWidth, displayHeight);
  createTextureUnit(gl, glState.accum.textures[accumIndex], 0);

  setDisplayUniforms(gl, displayProgram, {
    displayUnit: 0,
    displayResolution: [displayWidth, displayHeight],
    toneMap: renderState.toneMap
  });

  gl.useProgram(displayProgram);
  drawFullscreen(gl);

  glState.frameParity = prevIndex;
  renderState.frameIndex += 1;
  renderState.cameraDirty = false;

  if (renderOverlay && sceneData) {
    const maxFrames = renderState.maxFrames > 0 ? renderState.maxFrames : "∞";
    const polyCount = sceneData.triCount || 0;
    const primCount = (sceneData.sphereCount || 0) + (sceneData.cylinderCount || 0);
    renderOverlay.textContent = `${renderState.frameIndex}/${maxFrames} ${formatPolyCount(polyCount)} plys, ${formatPolyCount(primCount)} prims`;
    renderOverlay.style.display = "block";
  }
}

async function startRenderLoop() {
  if (isRendering) {
    return;
  }
  renderBtn.disabled = true;
  renderBtn.textContent = "Pause";
  isRendering = true;
  logger.info("Interactive render started.");
  let lastTime = performance.now();

  const loop = (time) => {
    if (!isRendering) {
      renderBtn.disabled = false;
      return;
    }
    const movingNow = isCameraInteracting(time);
    const targetScale = movingNow ? renderState.fastScale : renderState.renderScale;
    if (renderState.scale !== targetScale) {
      renderState.scale = targetScale;
      resetAccumulation();
      glState = null;
    }
    const dt = Math.max(0.001, (time - lastTime) / 1000);
    lastTime = time;
    const moved = updateCameraFromInput(dt);
    if (moved) {
      renderState.cameraDirty = true;
      markInteractionActive(time);
    }
    if (renderState.maxFrames > 0 && renderState.frameIndex >= renderState.maxFrames && !renderState.cameraDirty) {
      requestAnimationFrame(loop);
      return;
    }
    try {
      renderFrame();
    } catch (err) {
      logger.error(err.message || String(err));
      isRendering = false;
      renderBtn.disabled = false;
      renderBtn.textContent = "Render";
      return;
    }
    requestAnimationFrame(loop);
  };

  requestAnimationFrame(loop);
}

function stopRenderLoop() {
  if (!isRendering) {
    return;
  }
  isRendering = false;
  renderBtn.textContent = "Render";
  if (renderOverlay) {
    renderOverlay.style.display = "none";
  }
  logger.info("Paused.");
}

async function loadExampleScene(url) {
  if (isLoading) return;
  isLoading = true;
  renderBtn.disabled = true;
  setLoadingOverlay(true, "Loading scene...");
  let success = false;
  try {
    if (url === "__test_primitives__") {
      loadTestPrimitives();
      success = true;
    } else if (url === "__test_1000_spheres__") {
      loadRandomSpheres(1000);
      success = true;
    } else if (url === "__test_10000_spheres__") {
      loadRandomSpheres(10000);
      success = true;
    } else if (url.startsWith("__sdf_")) {
      // Built-in SDF molecule
      const molName = url.replace("__sdf_", "").replace("__", "");
      loadBuiltinMolecule(molName);
      success = true;
    } else {
      logger.info(`Loading example: ${url}`);
      const text = await fetchText(url);
      const baseUrl = new URL(url, window.location.href).toString();
      await loadGltfText(text, baseUrl);
      logger.info("Example loaded.");
      success = true;
    }
  } catch (err) {
    logger.error(err.message || String(err));
  } finally {
    renderBtn.disabled = false;
    isLoading = false;
    glState = null;
    setLoadingOverlay(false);
  }
  return success;
}

loadExampleBtn.addEventListener("click", async () => {
  const value = exampleSelect.value;
  renderBtn.disabled = true;
  setLoadingOverlay(true, "Loading example...");
  let loaded = false;
  try {
    if (value.startsWith("mol:")) {
      // Small molecule (SDF)
      const molName = value.slice(4);
      loadBuiltinMolecule(molName);
      glState = null;
      loaded = true;
    } else if (value.startsWith("pdb:")) {
      // Protein (PDB)
      const pdbId = value.slice(4);
      await loadPDBById(pdbId);
      glState = null;
      loaded = true;
    } else {
      // glTF or test scene
      loaded = await loadExampleScene(value);
    }
  } catch (err) {
    logger.error(err.message || String(err));
  } finally {
    renderBtn.disabled = false;
    setLoadingOverlay(false);
  }
  if (loaded && sceneData) {
    await startRenderLoop();
  }
});

// Load glTF file from file input
fileInput.addEventListener("change", async () => {
  renderBtn.disabled = true;
  setLoadingOverlay(true, "Loading glTF...");
  let loaded = false;
  try {
    const file = fileInput.files?.[0];
    if (!file) return;
    logger.info(`Loading file: ${file.name}`);
    const text = await file.text();
    await loadGltfText(text, null);
    logger.info("File loaded.");
    glState = null;
    loaded = true;
  } catch (err) {
    logger.error(err.message || String(err));
  } finally {
    renderBtn.disabled = false;
    setLoadingOverlay(false);
  }
  if (loaded && sceneData) {
    await startRenderLoop();
  }
});

// Load molecular file from file input
molFileInput.addEventListener("change", async () => {
  renderBtn.disabled = true;
  setLoadingOverlay(true, "Loading molecule...");
  let loaded = false;
  try {
    const file = molFileInput.files?.[0];
    if (!file) return;
    const text = await file.text();
    loadMolecularFile(text, file.name);
    glState = null;
    loaded = true;
  } catch (err) {
    logger.error(err.message || String(err));
  } finally {
    renderBtn.disabled = false;
    setLoadingOverlay(false);
  }
  if (loaded && sceneData) {
    await startRenderLoop();
  }
});

loadPdbIdBtn.addEventListener("click", async () => {
  renderBtn.disabled = true;
  setLoadingOverlay(true, "Fetching PDB...");
  let loaded = false;
  try {
    const pdbId = pdbIdInput.value.trim();
    if (!pdbId || pdbId.length !== 4) {
      throw new Error("Please enter a valid 4-letter PDB ID.");
    }
    await loadPDBById(pdbId);
    glState = null;
    loaded = true;
  } catch (err) {
    logger.error(err.message || String(err));
  } finally {
    renderBtn.disabled = false;
    setLoadingOverlay(false);
  }
  if (loaded && sceneData) {
    await startRenderLoop();
  }
});

renderBtn.addEventListener("click", async () => {
  if (isRendering) {
    stopRenderLoop();
    return;
  }
  await startRenderLoop();
});

canvas.addEventListener("mousedown", (event) => {
  inputState.dragging = true;
  inputState.lastX = event.clientX;
  inputState.lastY = event.clientY;
  inputState.arcballPrev = arcballVector(event.clientX, event.clientY);
  markInteractionActive();
});

canvas.addEventListener("mouseup", () => {
  inputState.dragging = false;
  inputState.arcballPrev = null;
});

canvas.addEventListener("mouseleave", () => {
  inputState.dragging = false;
  inputState.arcballPrev = null;
});

canvas.addEventListener("mousemove", (event) => {
  if (!inputState.dragging) return;
  const dx = event.clientX - inputState.lastX;
  const dy = event.clientY - inputState.lastY;
  inputState.lastX = event.clientX;
  inputState.lastY = event.clientY;
  markInteractionActive();

  if (event.shiftKey) {
    const panScale = cameraState.distance * 0.002;
    const orbit = computeCameraVectors();
    const rightLen = Math.hypot(orbit.right[0], orbit.right[1], orbit.right[2]) || 1;
    const upLen = Math.hypot(orbit.up[0], orbit.up[1], orbit.up[2]) || 1;
    const right = [orbit.right[0] / rightLen, orbit.right[1] / rightLen, orbit.right[2] / rightLen];
    const up = [orbit.up[0] / upLen, orbit.up[1] / upLen, orbit.up[2] / upLen];

    cameraState.target[0] -= right[0] * dx * panScale;
    cameraState.target[1] -= right[1] * dx * panScale;
    cameraState.target[2] -= right[2] * dx * panScale;
    cameraState.target[0] += up[0] * dy * panScale;
    cameraState.target[1] += up[1] * dy * panScale;
    cameraState.target[2] += up[2] * dy * panScale;
    inputState.arcballPrev = arcballVector(event.clientX, event.clientY);
    renderState.cameraDirty = true;
    return;
  }

  if (event.ctrlKey) {
    const zoom = Math.exp(dy * 0.005);
    const sceneScale = sceneData?.sceneScale || 1.0;
    const minDist = Math.max(0.1, sceneScale * 0.1);
    const maxDist = Math.max(100, sceneScale * 20);
    cameraState.distance = clamp(cameraState.distance * zoom, minDist, maxDist);
    inputState.arcballPrev = arcballVector(event.clientX, event.clientY);
    renderState.cameraDirty = true;
    return;
  }

  const prev = inputState.arcballPrev ?? arcballVector(event.clientX, event.clientY);
  const curr = arcballVector(event.clientX, event.clientY);
  const dot = clamp(prev[0] * curr[0] + prev[1] * curr[1] + prev[2] * curr[2], -1, 1);
  const angle = Math.acos(dot);
  const axis = [
    prev[1] * curr[2] - prev[2] * curr[1],
    prev[2] * curr[0] - prev[0] * curr[2],
    prev[0] * curr[1] - prev[1] * curr[0]
  ];
  const axisLen = Math.hypot(axis[0], axis[1], axis[2]);
  if (axisLen > 1e-6 && angle > 1e-6) {
    axis[0] /= axisLen;
    axis[1] /= axisLen;
    axis[2] /= axisLen;
    const delta = quatFromAxisAngle(axis, angle);
    cameraState.rotation = normalizeQuat(quatMultiply(cameraState.rotation, delta));
    renderState.cameraDirty = true;
  }
  inputState.arcballPrev = curr;
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const zoom = Math.exp(event.deltaY * 0.0015);
  // Dynamic zoom limits based on scene scale
  const sceneScale = sceneData?.sceneScale || 1.0;
  const minDist = Math.max(0.1, sceneScale * 0.1);
  const maxDist = Math.max(100, sceneScale * 20);
  cameraState.distance = clamp(cameraState.distance * zoom, minDist, maxDist);
  renderState.cameraDirty = true;
  markInteractionActive();
}, { passive: false });

window.addEventListener("keydown", (event) => {
  inputState.keys.add(event.key.toLowerCase());
});

window.addEventListener("keyup", (event) => {
  inputState.keys.delete(event.key.toLowerCase());
});

scaleSelect.addEventListener("change", () => {
  const value = Number(scaleSelect.value);
  if (!Number.isFinite(value) || value <= 0) {
    return;
  }
  renderState.renderScale = value;
  if (!isCameraInteracting()) {
    renderState.scale = value;
    resetAccumulation(`Render scale set to ${value.toFixed(2)}x`);
    glState = null;
  }
});

fastScaleSelect.addEventListener("change", () => {
  const value = Number(fastScaleSelect.value);
  if (!Number.isFinite(value) || value <= 0) {
    return;
  }
  renderState.fastScale = value;
  if (isCameraInteracting()) {
    renderState.scale = value;
    resetAccumulation();
    glState = null;
  }
});

envSelect.addEventListener("change", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});
envIntensityInput.addEventListener("input", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});
envMaxLumInput?.addEventListener("input", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setActiveTab(button.dataset.tabButton);
  });
});

bruteforceToggle.addEventListener("change", () => {
  const mode = bruteforceToggle.value;
  renderState.useBvh = mode !== "bruteforce";
  resetAccumulation(`Traversal mode: ${renderState.useBvh ? "BVH" : "Brute force"}`);
});

useGltfColorToggle.addEventListener("change", updateMaterialState);
materialSelect?.addEventListener("change", () => {
  updateMaterialVisibility();
  updateMaterialState();
});
baseColorInput.addEventListener("input", updateMaterialState);
metallicInput.addEventListener("input", updateMaterialState);
roughnessInput.addEventListener("input", updateMaterialState);
matteSpecularInput?.addEventListener("input", updateMaterialState);
matteRoughnessInput?.addEventListener("input", updateMaterialState);
matteDiffuseRoughnessInput?.addEventListener("input", updateMaterialState);
wrapDiffuseInput?.addEventListener("input", updateMaterialState);
maxBouncesInput.addEventListener("input", updateMaterialState);
exposureInput.addEventListener("input", updateMaterialState);
toneMapSelect?.addEventListener("change", updateMaterialState);
ambientIntensityInput.addEventListener("input", updateMaterialState);
ambientColorInput.addEventListener("input", updateMaterialState);
rayBiasInput.addEventListener("input", updateMaterialState);
tMinInput.addEventListener("input", updateMaterialState);
samplesPerBounceInput.addEventListener("input", updateMaterialState);
maxFramesInput?.addEventListener("input", updateRenderLimits);
shadowToggle.addEventListener("change", updateMaterialState);

light1Enable.addEventListener("change", updateLightState);
light1Azimuth.addEventListener("input", updateLightState);
light1Elevation.addEventListener("input", updateLightState);
light1Intensity.addEventListener("input", updateLightState);
light1Extent.addEventListener("input", updateLightState);
light1Color.addEventListener("input", updateLightState);
light2Enable.addEventListener("change", updateLightState);
light2Azimuth.addEventListener("input", updateLightState);
light2Elevation.addEventListener("input", updateLightState);
light2Intensity.addEventListener("input", updateLightState);
light2Extent.addEventListener("input", updateLightState);
light2Color.addEventListener("input", updateLightState);

visModeSelect?.addEventListener("change", () => {
  renderState.visMode = parseInt(visModeSelect.value, 10) || 0;
  resetAccumulation("Visualization mode changed.");
});

// Load HDR environment map manifest and populate dropdown
async function loadEnvManifest() {
  try {
    const res = await fetch("assets/env/manifest.json");
    if (!res.ok) return;
    const manifest = await res.json();

    // Add options for each HDR map
    for (const entry of manifest) {
      const option = document.createElement("option");
      option.value = `assets/env/${entry.file}`;
      option.textContent = entry.name;
      envSelect.appendChild(option);
    }

    // Select first HDR by default if available
    if (manifest.length > 0) {
      envSelect.value = `assets/env/${manifest[0].file}`;
    }
  } catch (err) {
    console.warn("Could not load HDR manifest:", err);
  }
}

const params = new URLSearchParams(window.location.search);
const autorun = params.get("autorun");
const exampleParam = params.get("example");
if (exampleParam) {
  const option = Array.from(exampleSelect.options).find((opt) => opt.value === exampleParam);
  if (option) {
    exampleSelect.value = exampleParam;
  }
}

// Initialize: load manifest then start
loadEnvManifest().then(() => {
  if (autorun === "1") {
    logger.info("Autorun enabled via query string.");
    loadExampleScene(exampleSelect.value).then(() => startRenderLoop());
  } else {
    logger.info("Ready. Load an example or choose a .gltf file.");
  }

  updateMaterialState();
  updateMaterialVisibility();
  updateRenderLimits();
  updateLightState();
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});

setActiveTab("scene");
