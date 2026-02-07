import { createLogger } from "./logger.js";
import { applyOrbitDragToRotation, resolveRotationLock } from "./camera_orbit.js";
import { primTypeLabel, traceSceneRay } from "./ray_pick.js";
import { computePrimitiveWorldBounds, projectAabbToCanvasRect } from "./overlay_bbox.js";
import { buildUnifiedBVH, flattenBVH } from "./bvh.js";
import {
  packBvhNodes, packTriangles, packTriNormals, packTriColors, packTriFlags, packPrimIndices,
  packSpheres, packSphereColors, packCylinders, packCylinderColors
} from "./packing.js";
import { loadHDR, buildEnvSamplingData } from "./hdr.js";
import {
  ANALYTIC_SKY_ID,
  analyticSkyCacheKey,
  generateAnalyticSkyEnvironment,
  normalizeAnalyticSkySettings
} from "./analytic_sky.js";
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
import { buildBackboneCartoon, buildSheetHbondCylinders } from "./cartoon.js";
import { computeSES, sesToTriangles } from "./surface.js";
import { computeSESWasm, initSurfaceWasm, surfaceWasmReady } from "./surface_wasm.js";
import { computeSESWebGL, webglSurfaceAvailable } from "./surface_webgl.js";
import { buildNitrogenDensityVolume } from "./volume.js";
import {
  initWebGL,
  createDataTexture,
  createEnvTexture,
  createCdfTexture,
  createVolumeTexture,
  createAccumTargets,
  resizeAccumTargets,
  createTextureUnit,
  createTextureUnit3D,
  setTraceUniforms,
  setDisplayUniforms,
  drawFullscreen,
  MAX_BRUTE_FORCE_TRIS
} from "./webgl.js";

const canvas = document.getElementById("view");
const canvasContainer = canvas?.closest(".canvas-container");
const renderOverlay = document.getElementById("renderOverlay");
const loadingOverlay = document.getElementById("loadingOverlay");
const hoverBoxOverlay = document.getElementById("hoverBoxOverlay");
const statusEl = document.getElementById("status");
const logger = createLogger(statusEl);

const exampleSelect = document.getElementById("exampleSelect");
const loadExampleBtn = document.getElementById("loadExample");
const envSelect = document.getElementById("envSelect");
const envIntensityInput = document.getElementById("envIntensity");
const envMaxLumInput = document.getElementById("envMaxLum");
const analyticSkyResolutionSelect = document.getElementById("analyticSkyResolution");
const analyticSkyTurbidityInput = document.getElementById("analyticSkyTurbidity");
const analyticSkySunAzimuthInput = document.getElementById("analyticSkySunAzimuth");
const analyticSkySunElevationInput = document.getElementById("analyticSkySunElevation");
const analyticSkyIntensityInput = document.getElementById("analyticSkyIntensity");
const analyticSkySunIntensityInput = document.getElementById("analyticSkySunIntensity");
const analyticSkySunRadiusInput = document.getElementById("analyticSkySunRadius");
const analyticSkyGroundAlbedoInput = document.getElementById("analyticSkyGroundAlbedo");
const analyticSkyHorizonSoftnessInput = document.getElementById("analyticSkyHorizonSoftness");
const molFileInput = document.getElementById("molFileInput");
const pdbIdInput = document.getElementById("pdbIdInput");
const loadPdbIdBtn = document.getElementById("loadPdbId");
const pdbDisplayStyle = document.getElementById("pdbDisplayStyle");
const pdbAtomScale = document.getElementById("pdbAtomScale");
const pdbBondRadius = document.getElementById("pdbBondRadius");
const surfaceModeSelect = document.getElementById("surfaceMode");
const probeRadiusInput = document.getElementById("probeRadius");
const surfaceResolutionInput = document.getElementById("surfaceResolution");
const smoothNormalsToggle = document.getElementById("smoothNormals");
const volumeImportToggle = document.getElementById("volumeImportToggle");
const volumeGridSpacing = document.getElementById("volumeGridSpacing");
const volumeGaussianScale = document.getElementById("volumeGaussianScale");
const clipEnableToggle = document.getElementById("clipEnable");
const clipDistanceInput = document.getElementById("clipDistance");
const clipLockToggle = document.getElementById("clipLock");
const scaleSelect = document.getElementById("scaleSelect");
const fastScaleSelect = document.getElementById("fastScaleSelect");
const useImportedColorToggle = document.getElementById("useImportedColor");
const baseColorInput = document.getElementById("baseColor");
const materialSelect = document.getElementById("materialSelect");
const metallicInput = document.getElementById("metallic");
const roughnessInput = document.getElementById("roughness");
const rimBoostInput = document.getElementById("rimBoost");
const matteSpecularInput = document.getElementById("matteSpecular");
const matteRoughnessInput = document.getElementById("matteRoughness");
const matteDiffuseRoughnessInput = document.getElementById("matteDiffuseRoughness");
const wrapDiffuseInput = document.getElementById("wrapDiffuse");
const surfaceShowAtomsToggle = document.getElementById("surfaceShowAtoms");
const surfaceIorInput = document.getElementById("surfaceIor");
const surfaceTransmissionInput = document.getElementById("surfaceTransmission");
const showSheetHbondsToggle = document.getElementById("showSheetHbonds");
const surfaceOpacityInput = document.getElementById("surfaceOpacity");
const maxBouncesInput = document.getElementById("maxBounces");
const exposureInput = document.getElementById("exposure");
const dofEnableToggle = document.getElementById("dofEnable");
const dofApertureInput = document.getElementById("dofAperture");
const dofFocusDistanceInput = document.getElementById("dofFocusDistance");
const ambientIntensityInput = document.getElementById("ambientIntensity");
const ambientColorInput = document.getElementById("ambientColor");
const samplesPerBounceInput = document.getElementById("samplesPerBounce");
const maxFramesInput = document.getElementById("maxFrames");
const toneMapSelect = document.getElementById("toneMapSelect");
const shadowToggle = document.getElementById("shadowToggle");
const volumeEnableToggle = document.getElementById("volumeEnable");
const volumeColorInput = document.getElementById("volumeColor");
const volumeDensityInput = document.getElementById("volumeDensity");
const volumeOpacityInput = document.getElementById("volumeOpacity");
const volumeStepInput = document.getElementById("volumeStep");
const volumeMaxStepsInput = document.getElementById("volumeMaxSteps");
const volumeThresholdInput = document.getElementById("volumeThreshold");
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
let lastMolContext = null;

function hasSurfaceFlags(triFlags) {
  if (!triFlags || triFlags.length === 0) return false;
  for (let i = 0; i < triFlags.length; i += 1) {
    if (triFlags[i] > 0.5) return true;
  }
  return false;
}

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
  useImportedColor: true,
  baseColor: [0.8, 0.8, 0.8],
  materialMode: "metallic",
  metallic: 0.0,
  roughness: 0.4,
  rimBoost: 0.2,
  matteSpecular: 0.03,
  matteRoughness: 0.5,
  matteDiffuseRoughness: 0.5,
  wrapDiffuse: 0.2,
  surfaceShowAtoms: true,
  surfaceIor: 1.33,
  surfaceTransmission: 0.35,
  surfaceOpacity: 0.0,
  maxBounces: 4,
  maxFrames: 100,
  exposure: 1.0,
  dofEnabled: false,
  dofAperture: 0.03,
  dofFocusDistance: 4.0,
  toneMap: "aces",
  ambientIntensity: 0.0,
  ambientColor: [1.0, 1.0, 1.0],
  envUrl: null,
  envCacheKey: null,
  envIntensity: 0.1,
  envMaxLuminance: 200.0,
  envData: null,
  rayBias: 1e-5,
  tMin: 1e-5,
  samplesPerBounce: 1,
  castShadows: true,
  volumeEnabled: false,
  volumeColor: [0.435, 0.643, 1.0],
  volumeDensity: 1.0,
  volumeOpacity: 1.0,
  volumeStep: 0.5,
  volumeMaxSteps: 256,
  volumeThreshold: 0.0,
  lights: [
    // Camera-relative studio lighting: key, fill, rim
    { enabled: true, azimuth: -40, elevation: -30, intensity: 5.0, angle: 22, color: [1.0, 1.0, 1.0] },
    { enabled: true, azimuth: 40, elevation: 0, intensity: 0.6, angle: 50, color: [1.0, 1.0, 1.0] },
    { enabled: true, azimuth: 170, elevation: 10, intensity: 0.35, angle: 6, color: [1.0, 1.0, 1.0] }
  ],
  clipEnabled: false,
  clipDistance: 0.0,
  clipLocked: false,
  clipLockedNormal: null,
  clipLockedOffset: null,
  clipLockedSide: null,
  visMode: 0
};

const envCache = new Map();

const inputState = {
  dragging: false,
  lastX: 0,
  lastY: 0,
  dragMode: "rotate",
  rotateAxisLock: null,
  keys: new Set()
};

const interactionState = {
  lastActive: 0
};

const pointerState = {
  overCanvas: false,
  x: 0,
  y: 0
};

let hoverOverlayErrorMessage = null;

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
  const triFlags = new Float32Array(0);

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
    triFlags,
    hasSurfaceFlags: hasSurfaceFlags(triFlags),
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
  updateClipRange();
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
    triFlags,
    hasSurfaceFlags: hasSurfaceFlags(triFlags),
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
  updateClipRange();
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
    return { displayStyle: style, radiusScale: 1.0, bondRadius: 0, showBonds: false };
  } else if (style === "cartoon") {
    return { displayStyle: style, radiusScale: 0.0, bondRadius: 0.0, showBonds: false };
  } else if (style === "stick") {
    // Stick: tiny atoms, normal bonds
    return { displayStyle: style, radiusScale: 0.15, bondRadius: bondRadius, showBonds: true };
  } else {
    // Ball and stick: scaled atoms with bonds
    return { displayStyle: style, radiusScale: atomScale, bondRadius: bondRadius, showBonds: true };
  }
}

function requireNumberInput(input, label) {
  if (!input) {
    throw new Error(`${label} input is missing.`);
  }
  const value = Number(input.value);
  if (!Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number.`);
  }
  return value;
}

function getVolumeImportOptions({ requireEnabled = false } = {}) {
  if (!volumeImportToggle) {
    throw new Error("Volume import toggle is missing.");
  }
  const enabled = volumeImportToggle.checked;
  if (!enabled && !requireEnabled) {
    return { enabled: false };
  }

  const spacing = requireNumberInput(volumeGridSpacing, "Volume grid spacing");
  const gaussianScale = requireNumberInput(volumeGaussianScale, "Gaussian scale");

  if (spacing <= 0) {
    throw new Error("Volume grid spacing must be > 0.");
  }
  if (gaussianScale <= 0) {
    throw new Error("Gaussian scale must be > 0.");
  }

  return { enabled: true, spacing, gaussianScale };
}

function isLikelyPdbSource(filename, molData) {
  if (filename && filename.toLowerCase().endsWith(".pdb")) {
    return true;
  }
  return Boolean(molData && molData.secondary);
}

function buildNitrogenVolume(molData, volumeOpts) {
  logger.info(
    `Building nitrogen density volume (spacing=${volumeOpts.spacing}Å, gaussian=3xVdW, scale=${volumeOpts.gaussianScale})...`
  );
  const start = performance.now();
  const volume = buildNitrogenDensityVolume(molData, {
    spacing: volumeOpts.spacing,
    gaussianScale: volumeOpts.gaussianScale,
    cutoffSigma: 3.0
  });
  const elapsed = performance.now() - start;
  const [nx, ny, nz] = volume.dims;
  logger.info(
    `Volume built in ${elapsed.toFixed(0)}ms: ${nx}x${ny}x${nz}, N atoms=${volume.nitrogenCount}, max=${volume.maxValue.toFixed(3)}`
  );
  return volume;
}

async function loadMolecularFile(text, filename) {
  logger.info(`Parsing molecular file: ${filename}`);

  const molData = parseAutoDetect(text, filename);
  logger.info(`Parsed ${molData.atoms.length} atoms, ${molData.bonds.length} bonds`);

  const options = getMolecularDisplayOptions();
  const { spheres, cylinders } = moleculeToGeometry(molData, options);

  const volumeOpts = getVolumeImportOptions();
  let volumeData = null;
  if (volumeOpts.enabled) {
    if (!isLikelyPdbSource(filename, molData)) {
      throw new Error("Volume import is only supported for PDB files.");
    }
    volumeData = buildNitrogenVolume(molData, volumeOpts);
  }

  const surfaceAtomMode = (
    (renderState.materialMode === "surface-glass" || renderState.materialMode === "translucent-plastic")
    && renderState.surfaceShowAtoms
  ) ? "all" : "hetero";
  lastMolContext = { spheres, cylinders, molData, options, volumeData, surfaceAtomMode };
  await loadMolecularGeometry(spheres, cylinders, molData, options, volumeData, surfaceAtomMode);
}

function mergeTriangleMeshes(a, b) {
  if (!a || a.positions.length === 0) return b;
  if (!b || b.positions.length === 0) return a;

  const aTriCount = a.indices.length / 3;
  const bTriCount = b.indices.length / 3;

  const positions = new Float32Array(a.positions.length + b.positions.length);
  positions.set(a.positions, 0);
  positions.set(b.positions, a.positions.length);

  const normals = new Float32Array(a.normals.length + b.normals.length);
  normals.set(a.normals, 0);
  normals.set(b.normals, a.normals.length);

  const indices = new Uint32Array(a.indices.length + b.indices.length);
  indices.set(a.indices, 0);
  const offset = a.positions.length / 3;
  for (let i = 0; i < b.indices.length; i += 1) {
    indices[a.indices.length + i] = b.indices[i] + offset;
  }

  const triColors = new Float32Array(a.triColors.length + b.triColors.length);
  triColors.set(a.triColors, 0);
  triColors.set(b.triColors, a.triColors.length);

  const aFlags = a.triFlags && a.triFlags.length === aTriCount ? a.triFlags : new Float32Array(aTriCount);
  const bFlags = b.triFlags && b.triFlags.length === bTriCount ? b.triFlags : new Float32Array(bTriCount);
  const triFlags = new Float32Array(aFlags.length + bFlags.length);
  triFlags.set(aFlags, 0);
  triFlags.set(bFlags, aFlags.length);

  return { positions, indices, normals, triColors, triFlags };
}

/**
 * Load molecular geometry from spheres and cylinders.
 */
async function loadMolecularGeometry(
  spheres,
  cylinders,
  molData = null,
  options = null,
  volumeData = null,
  surfaceAtomMode = "hetero"
) {
  const surfaceMode = surfaceModeSelect?.value || "none";
  const showSurface = surfaceMode !== "none";
  const displayOptions = options ?? getMolecularDisplayOptions();
  const displayStyle = displayOptions.displayStyle || "ball-and-stick";
  const heteroDisplayOptions = displayStyle === "cartoon"
    ? { radiusScale: 0.4, bondRadius: 0.12, showBonds: true }
    : displayOptions;
  const split = molData ? splitMolDataByHetatm(molData) : null;
  const heteroGeometry = split ? moleculeToGeometry(split.hetero, heteroDisplayOptions) : { spheres: [], cylinders: [] };

  // If showing surface, hide atoms/bonds unless it's VdW mode
  let displaySpheres = spheres;
  let displayCylinders = cylinders;

  let positions = new Float32Array(0);
  let indices = new Uint32Array(0);
  let normals = new Float32Array(0);
  let triColors = new Float32Array(0);
  let triFlags = new Float32Array(0);
  let debugHbonds = [];

  if (displayStyle === "cartoon" && molData) {
    logger.info("Computing backbone cartoon (DSSP)...");
    if (molData.secondary) {
      const helixCount = molData.secondary.helices?.length || 0;
      const sheetCount = molData.secondary.sheets?.length || 0;
      if (helixCount + sheetCount > 0) {
        logger.info(`PDB secondary structure: ${helixCount} helices, ${sheetCount} sheets`);
      }
    }
    const cartoonStart = performance.now();
    const cartoon = buildBackboneCartoon(molData, {
      debugSheetOrientation: true,
      debugLog: (msg) => logger.info(msg)
    });
    const cartoonTime = performance.now() - cartoonStart;
    logger.info(`Cartoon built in ${cartoonTime.toFixed(0)}ms: ${cartoon.indices.length / 3} triangles`);

    positions = cartoon.positions;
    indices = cartoon.indices;
    normals = cartoon.normals;
    triColors = cartoon.triColors;
    triFlags = new Float32Array(cartoon.indices.length / 3);

    displaySpheres = heteroGeometry.spheres;
    displayCylinders = heteroGeometry.cylinders;

    if (showSheetHbondsToggle?.checked) {
      debugHbonds = buildSheetHbondCylinders(molData);
      if (debugHbonds.length > 0) {
        logger.info(`Debug: ${debugHbonds.length} sheet H-bonds`);
      }
    }
  }

  if (showSurface && molData && molData.atoms && molData.atoms.length > 0) {
    const surfaceAtoms = split ? split.standard.atoms : molData.atoms;
    if (surfaceAtoms.length === 0) {
      logger.warn("No non-HETATM atoms available for surface; rendering atoms only.");
    } else {
      const probeRadius = parseFloat(probeRadiusInput?.value) || 1.4;
      const resolution = parseFloat(surfaceResolutionInput?.value) || 0.25;
      const smoothNormals = smoothNormalsToggle?.checked || false;

      const backendLabels = { js: "JS", wasm: "WASM", webgl: "WebGL" };
      const backendLabel = backendLabels[surfaceMode] || surfaceMode;
      logger.info(
        `Computing SES surface (${backendLabel}, probe=${probeRadius}Å, resolution=${resolution}Å, smoothNormals=${smoothNormals})...`
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

      let sesMesh = null;
      try {
        if (surfaceMode === "wasm") {
          if (!surfaceWasmReady()) {
            logger.info("Initializing SES WASM module...");
          }
          sesMesh = await computeSESWasm(atoms, { probeRadius, resolution, smoothNormals });
        } else if (surfaceMode === "webgl") {
          if (!webglSurfaceAvailable()) {
            throw new Error("WebGL surface computation not available (requires WebGL2 + EXT_color_buffer_float)");
          }
          sesMesh = computeSESWebGL(atoms, { probeRadius, resolution, smoothNormals });
        } else {
          sesMesh = computeSES(atoms, { probeRadius, resolution, smoothNormals });
        }
      } catch (err) {
        logger.error(err.message || String(err));
        throw err;
      }
      const surfaceTime = performance.now() - surfaceStart;
      logger.info(
        `SES ${backendLabel} completed in ${surfaceTime.toFixed(0)}ms: ${sesMesh.indices.length / 3} triangles`
      );

      if (sesMesh.vertices.length > 0) {

        const surfaceData = sesToTriangles(sesMesh, [0.7, 0.75, 0.9]);
        const surfaceFlags = new Float32Array(surfaceData.indices.length / 3);
        surfaceFlags.fill(1);
        const surfaceMesh = {
          positions: surfaceData.positions,
          indices: surfaceData.indices,
          normals: surfaceData.normals,
          triColors: surfaceData.triColors,
          triFlags: surfaceFlags
        };

        if (displayStyle === "cartoon") {
          const merged = mergeTriangleMeshes({ positions, indices, normals, triColors, triFlags }, surfaceMesh);
          positions = merged.positions;
          indices = merged.indices;
          normals = merged.normals;
          triColors = merged.triColors;
          triFlags = merged.triFlags;
        } else {
          positions = surfaceData.positions;
          indices = surfaceData.indices;
          normals = surfaceData.normals;
          triColors = surfaceData.triColors;
          triFlags = surfaceFlags;
        }

        if (surfaceAtomMode === "all") {
          displaySpheres = spheres;
          displayCylinders = cylinders;
        } else {
          // Keep HETATM atoms/bonds visible.
          displaySpheres = heteroGeometry.spheres;
          displayCylinders = heteroGeometry.cylinders;
        }
      } else {
        logger.warn("SES computation produced no surface");
      }
    }
  }

  if (debugHbonds.length > 0) {
    displayCylinders = displayCylinders.concat(debugHbonds);
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
    triFlags,
    hasSurfaceFlags: hasSurfaceFlags(triFlags),
    nodes: bvh.nodes,
    tris: bvh.tris,
    primitives: bvh.primitives,
    primIndexBuffer: flat.primIndexBuffer,
    triCount: bvh.triCount,
    sphereCount: bvh.sphereCount,
    cylinderCount: bvh.cylinderCount,
    spheres: displaySpheres,
    cylinders: displayCylinders,
    surfaceAtomMode,
    sceneScale: 1.0,
    volume: volumeData
  };
  renderState.frameIndex = 0;
  renderState.cameraDirty = true;

  // Compute bounds from all geometry
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let hasBounds = false;

  for (const s of displaySpheres) {
    minX = Math.min(minX, s.center[0] - s.radius);
    minY = Math.min(minY, s.center[1] - s.radius);
    minZ = Math.min(minZ, s.center[2] - s.radius);
    maxX = Math.max(maxX, s.center[0] + s.radius);
    maxY = Math.max(maxY, s.center[1] + s.radius);
    maxZ = Math.max(maxZ, s.center[2] + s.radius);
    hasBounds = true;
  }

  for (const c of displayCylinders) {
    minX = Math.min(minX, c.p1[0] - c.radius, c.p2[0] - c.radius);
    minY = Math.min(minY, c.p1[1] - c.radius, c.p2[1] - c.radius);
    minZ = Math.min(minZ, c.p1[2] - c.radius, c.p2[2] - c.radius);
    maxX = Math.max(maxX, c.p1[0] + c.radius, c.p2[0] + c.radius);
    maxY = Math.max(maxY, c.p1[1] + c.radius, c.p2[1] + c.radius);
    maxZ = Math.max(maxZ, c.p1[2] + c.radius, c.p2[2] + c.radius);
    hasBounds = true;
  }

  // Include surface triangles in bounds
  for (let i = 0; i < positions.length; i += 3) {
    minX = Math.min(minX, positions[i]);
    minY = Math.min(minY, positions[i + 1]);
    minZ = Math.min(minZ, positions[i + 2]);
    maxX = Math.max(maxX, positions[i]);
    maxY = Math.max(maxY, positions[i + 1]);
    maxZ = Math.max(maxZ, positions[i + 2]);
    hasBounds = true;
  }

  if (volumeData && volumeData.bounds) {
    if (!hasBounds) {
      minX = volumeData.bounds.minX;
      minY = volumeData.bounds.minY;
      minZ = volumeData.bounds.minZ;
      maxX = volumeData.bounds.maxX;
      maxY = volumeData.bounds.maxY;
      maxZ = volumeData.bounds.maxZ;
      hasBounds = true;
    } else {
      minX = Math.min(minX, volumeData.bounds.minX);
      minY = Math.min(minY, volumeData.bounds.minY);
      minZ = Math.min(minZ, volumeData.bounds.minZ);
      maxX = Math.max(maxX, volumeData.bounds.maxX);
      maxY = Math.max(maxY, volumeData.bounds.maxY);
      maxZ = Math.max(maxZ, volumeData.bounds.maxZ);
    }
  }

  if (!hasBounds) {
    throw new Error("Could not determine scene bounds (no geometry or volume data).");
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

  const volumeOpts = getVolumeImportOptions();
  const volumeData = volumeOpts.enabled ? buildNitrogenVolume(molData, volumeOpts) : null;

  const surfaceAtomMode = (
    (renderState.materialMode === "surface-glass" || renderState.materialMode === "translucent-plastic")
    && renderState.surfaceShowAtoms
  ) ? "all" : "hetero";
  lastMolContext = { spheres, cylinders, molData, options, volumeData, surfaceAtomMode };
  await loadMolecularGeometry(spheres, cylinders, molData, options, volumeData, surfaceAtomMode);
}

/**
 * Load a built-in small molecule by name.
 */
async function loadBuiltinMolecule(name) {
  logger.info(`Loading built-in molecule: ${name}`);
  const molData = getBuiltinMolecule(name);
  logger.info(`Parsed ${molData.atoms.length} atoms, ${molData.bonds.length} bonds`);

  const options = getMolecularDisplayOptions();
  const { spheres, cylinders } = moleculeToGeometry(molData, options);

  const surfaceAtomMode = (
    (renderState.materialMode === "surface-glass" || renderState.materialMode === "translucent-plastic")
    && renderState.surfaceShowAtoms
  ) ? "all" : "hetero";
  lastMolContext = { spheres, cylinders, molData, options, volumeData: null, surfaceAtomMode };
  await loadMolecularGeometry(spheres, cylinders, molData, options, null, surfaceAtomMode);
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
  renderState.useImportedColor = useImportedColorToggle.checked;
  renderState.baseColor = hexToRgb(baseColorInput.value);
  renderState.materialMode = materialSelect?.value || "metallic";
  renderState.metallic = clamp(Number(metallicInput.value), 0, 1);
  renderState.roughness = clamp(Number(roughnessInput.value), 0.02, 1);
  renderState.rimBoost = clamp(Number(rimBoostInput?.value ?? 0.2), 0.0, 1.0);
  renderState.matteSpecular = clamp(Number(matteSpecularInput?.value ?? 0.03), 0.0, 0.08);
  renderState.matteRoughness = clamp(Number(matteRoughnessInput?.value ?? 0.5), 0.1, 1.0);
  renderState.matteDiffuseRoughness = clamp(Number(matteDiffuseRoughnessInput?.value ?? 0.5), 0.0, 1.0);
  renderState.wrapDiffuse = clamp(Number(wrapDiffuseInput?.value ?? 0.2), 0.0, 0.5);
  renderState.surfaceShowAtoms = surfaceShowAtomsToggle?.checked ?? true;
  renderState.surfaceIor = clamp(Number(surfaceIorInput?.value ?? 1.33), 1.0, 2.5);
  renderState.surfaceTransmission = clamp(Number(surfaceTransmissionInput?.value ?? 0.35), 0.0, 1.0);
  renderState.surfaceOpacity = clamp(Number(surfaceOpacityInput?.value ?? 0.0), 0.0, 1.0);
  renderState.maxBounces = clamp(Number(maxBouncesInput.value), 0, 6);
  renderState.exposure = clamp(Number(exposureInput.value), 0, 5);
  renderState.dofEnabled = dofEnableToggle?.checked ?? false;
  const dofAperture = requireNumberInput(dofApertureInput, "Depth-of-field aperture");
  const dofFocusDistance = requireNumberInput(dofFocusDistanceInput, "Depth-of-field focus distance");
  if (dofAperture < 0.0 || dofAperture > 1.0) {
    throw new Error("Depth-of-field aperture must be between 0 and 1.0.");
  }
  if (dofFocusDistance <= 0.0 || dofFocusDistance > 1000.0) {
    throw new Error("Depth-of-field focus distance must be > 0 and <= 1000.");
  }
  renderState.dofAperture = dofAperture;
  renderState.dofFocusDistance = dofFocusDistance;
  renderState.ambientIntensity = clamp(Number(ambientIntensityInput.value), 0, 2);
  renderState.ambientColor = hexToRgb(ambientColorInput.value);
  renderState.envIntensity = clamp(Number(envIntensityInput.value), 0, 1.0);
  renderState.useBvh = true;
  renderState.rayBias = clamp(renderState.rayBias, 1e-7, 1);
  renderState.tMin = clamp(renderState.tMin, 1e-7, 1);
  renderState.samplesPerBounce = clamp(Number(samplesPerBounceInput.value), 1, 8);
  renderState.castShadows = shadowToggle.checked;
  renderState.toneMap = toneMapSelect?.value || "reinhard";
  resetAccumulation("Material settings updated.");
}

function updateVolumeState() {
  if (!volumeEnableToggle) {
    throw new Error("Volume controls are missing.");
  }
  renderState.volumeEnabled = volumeEnableToggle.checked;
  if (!volumeColorInput) {
    throw new Error("Volume color input is missing.");
  }
  renderState.volumeColor = hexToRgb(volumeColorInput.value);

  const density = requireNumberInput(volumeDensityInput, "Volume density");
  const opacity = requireNumberInput(volumeOpacityInput, "Volume opacity");
  const step = requireNumberInput(volumeStepInput, "Volume step");
  const maxSteps = requireNumberInput(volumeMaxStepsInput, "Volume max steps");
  const threshold = requireNumberInput(volumeThresholdInput, "Volume threshold");

  if (density < 0) {
    throw new Error("Volume density must be >= 0.");
  }
  if (opacity < 0) {
    throw new Error("Volume opacity must be >= 0.");
  }
  if (step <= 0) {
    throw new Error("Volume step must be > 0.");
  }
  if (maxSteps <= 0) {
    throw new Error("Volume max steps must be > 0.");
  }
  if (threshold < 0 || threshold > 1) {
    throw new Error("Volume threshold must be between 0 and 1.");
  }

  renderState.volumeDensity = density;
  renderState.volumeOpacity = opacity;
  renderState.volumeStep = step;
  renderState.volumeMaxSteps = Math.floor(maxSteps);
  renderState.volumeThreshold = threshold;

  if (renderState.volumeEnabled && sceneData && !sceneData.volume) {
    logger.warn("Volume enabled but no volume data is available. Reimport a PDB with volume enabled.");
  }

  resetAccumulation("Volume settings updated.");
}

function updateMaterialVisibility() {
  const mode = materialSelect?.value || "metallic";
  const metallicGroup = document.querySelector(".material-metallic");
  const matteGroup = document.querySelector(".material-matte");
  const surfaceGroup = document.querySelector(".material-surface");
  if (metallicGroup) metallicGroup.style.display = mode === "metallic" ? "block" : "none";
  if (matteGroup) matteGroup.style.display = mode === "matte" ? "block" : "none";
  if (surfaceGroup) {
    surfaceGroup.style.display = (mode === "surface-glass" || mode === "translucent-plastic") ? "block" : "none";
  }
}

function updateDofVisibility() {
  const controls = document.querySelector(".dof-controls");
  if (!controls) return;
  controls.style.display = dofEnableToggle?.checked ? "block" : "none";
}

function setSliderValue(input, value) {
  if (!input) return;
  input.value = String(value);
  const valueInput = document.querySelector(`.value-input[data-for="${input.id}"]`);
  if (valueInput) {
    const step = parseFloat(input.step) || 1;
    const decimals = step < 1 ? Math.max(0, -Math.floor(Math.log10(step))) : 0;
    valueInput.value = Number(value).toFixed(decimals);
  }
}

function applyMaterialPreset(mode) {
  if (mode !== "translucent-plastic") return;
  // Dielectric translucent plastic defaults.
  setSliderValue(metallicInput, 0.0);
  setSliderValue(roughnessInput, 0.22);
  setSliderValue(rimBoostInput, 0.0);
  setSliderValue(surfaceIorInput, 1.46);
  setSliderValue(surfaceTransmissionInput, 0.55);
  setSliderValue(surfaceOpacityInput, 0.15);
  logger.info("Applied preset: Translucent Plastic");
}

async function refreshSurfaceAtomMode() {
  if (!lastMolContext || !lastMolContext.molData) return;
  const surfaceMode = surfaceModeSelect?.value || "none";
  if (surfaceMode === "none") return;
  const usesSurfaceTranslucency = (
    renderState.materialMode === "surface-glass" || renderState.materialMode === "translucent-plastic"
  );
  const desiredMode = usesSurfaceTranslucency && renderState.surfaceShowAtoms ? "all" : "hetero";
  if (lastMolContext.surfaceAtomMode === desiredMode) return;
  lastMolContext.surfaceAtomMode = desiredMode;
  logger.info(`Reloading molecular geometry for surface (${desiredMode} atoms).`);
  await loadMolecularGeometry(
    lastMolContext.spheres,
    lastMolContext.cylinders,
    lastMolContext.molData,
    lastMolContext.options,
    lastMolContext.volumeData,
    desiredMode
  );
}

function updateRenderLimits() {
  const raw = Number(maxFramesInput?.value);
  const maxFrames = clamp(Number.isFinite(raw) ? Math.floor(raw) : 0, 0, 2000);
  renderState.maxFrames = maxFrames;
}

function updateClipRange() {
  if (!clipDistanceInput || !sceneData) return;
  const max = Math.max(1, sceneData.sceneScale * 4);
  clipDistanceInput.max = max.toFixed(2);
  const current = Number(clipDistanceInput.value) || 0;
  if (current > max) {
    clipDistanceInput.value = max.toFixed(2);
    clipDistanceInput.dispatchEvent(new Event("input", { bubbles: true }));
  }
}

function updateClipState({ preserveLock = false } = {}) {
  renderState.clipEnabled = clipEnableToggle?.checked || false;
  renderState.clipDistance = clamp(Number(clipDistanceInput?.value ?? 0), 0, 1e6);
  const lock = clipLockToggle?.checked || false;
  if (lock) {
    const cam = computeCameraVectors();
    if (!renderState.clipLocked || !renderState.clipLockedNormal || !preserveLock) {
      const len = Math.hypot(cam.forward[0], cam.forward[1], cam.forward[2]) || 1;
      renderState.clipLockedNormal = [cam.forward[0] / len, cam.forward[1] / len, cam.forward[2] / len];
    }
    const n = renderState.clipLockedNormal;
    if (n) {
      const planePoint = [
        cam.origin[0] + n[0] * renderState.clipDistance,
        cam.origin[1] + n[1] * renderState.clipDistance,
        cam.origin[2] + n[2] * renderState.clipDistance
      ];
      renderState.clipLockedOffset = n[0] * planePoint[0] + n[1] * planePoint[1] + n[2] * planePoint[2];
      const camSide = n[0] * cam.origin[0] + n[1] * cam.origin[1] + n[2] * cam.origin[2] - renderState.clipLockedOffset;
      renderState.clipLockedSide = camSide >= 0 ? 1 : -1;
    }
  } else {
    renderState.clipLockedNormal = null;
    renderState.clipLockedOffset = null;
    renderState.clipLockedSide = null;
  }
  renderState.clipLocked = lock;
  resetAccumulation("Slice plane updated.");
}

function parseAnalyticResolution(value) {
  if (!value || typeof value !== "string") {
    throw new Error("Analytic sky resolution is missing.");
  }
  const parts = value.toLowerCase().split("x").map((v) => Number(v));
  if (parts.length !== 2 || !Number.isInteger(parts[0]) || !Number.isInteger(parts[1])) {
    throw new Error(`Invalid analytic sky resolution: ${value}`);
  }
  return { width: parts[0], height: parts[1] };
}

function getAnalyticSkySettingsFromUi() {
  const { width, height } = parseAnalyticResolution(analyticSkyResolutionSelect?.value || "");
  return normalizeAnalyticSkySettings({
    width,
    height,
    turbidity: requireNumberInput(analyticSkyTurbidityInput, "Analytic sky turbidity"),
    sunAzimuthDeg: requireNumberInput(analyticSkySunAzimuthInput, "Analytic sky sun azimuth"),
    sunElevationDeg: requireNumberInput(analyticSkySunElevationInput, "Analytic sky sun elevation"),
    skyIntensity: requireNumberInput(analyticSkyIntensityInput, "Analytic sky intensity"),
    sunIntensity: requireNumberInput(analyticSkySunIntensityInput, "Analytic sky sun intensity"),
    sunAngularRadiusDeg: requireNumberInput(analyticSkySunRadiusInput, "Analytic sky sun radius"),
    groundAlbedo: requireNumberInput(analyticSkyGroundAlbedoInput, "Analytic sky ground albedo"),
    horizonSoftness: requireNumberInput(analyticSkyHorizonSoftnessInput, "Analytic sky horizon softness")
  });
}

function updateEnvironmentVisibility() {
  const analyticControls = document.querySelector(".analytic-sky-controls");
  if (!analyticControls) return;
  const selected = envSelect?.value || "";
  analyticControls.style.display = selected === ANALYTIC_SKY_ID ? "block" : "none";
}

function uploadEnvironmentToGl(env) {
  if (!glState || !env) return;
  const gl = glState.gl;
  if (glState.envTex && glState.envTex !== glState.blackEnvTex) {
    gl.deleteTexture(glState.envTex);
  }
  if (glState.envMarginalCdfTex && glState.envMarginalCdfTex !== glState.dummyCdfTex) {
    gl.deleteTexture(glState.envMarginalCdfTex);
  }
  if (
    glState.envConditionalCdfTex
    && glState.envConditionalCdfTex !== glState.dummyCdfTex
    && glState.envConditionalCdfTex !== glState.envMarginalCdfTex
  ) {
    gl.deleteTexture(glState.envConditionalCdfTex);
  }

  glState.envTex = createEnvTexture(gl, env.width, env.height, env.data);
  glState.envMarginalCdfTex = createCdfTexture(
    gl,
    env.samplingData.marginalCdf,
    env.samplingData.height + 1,
    1
  );
  glState.envConditionalCdfTex = createCdfTexture(
    gl,
    env.samplingData.conditionalCdf,
    env.samplingData.width + 1,
    env.samplingData.height
  );
  glState.envSize = [env.width, env.height];
  glState.envUrl = renderState.envUrl;
  glState.envCacheKey = env.version || renderState.envCacheKey;
}

async function loadEnvironment(url, analyticSettings = null) {
  if (!url) {
    renderState.envUrl = null;
    renderState.envCacheKey = null;
    renderState.envData = null;
    if (glState) {
      const gl = glState.gl;
      if (glState.envTex && glState.envTex !== glState.blackEnvTex) {
        gl.deleteTexture(glState.envTex);
      }
      if (glState.envMarginalCdfTex && glState.envMarginalCdfTex !== glState.dummyCdfTex) {
        gl.deleteTexture(glState.envMarginalCdfTex);
      }
      if (glState.envConditionalCdfTex && glState.envConditionalCdfTex !== glState.dummyCdfTex) {
        gl.deleteTexture(glState.envConditionalCdfTex);
      }
      glState.envTex = glState.blackEnvTex;
      glState.envMarginalCdfTex = glState.dummyCdfTex;
      glState.envConditionalCdfTex = glState.dummyCdfTex;
      glState.envSize = [1, 1];
      glState.envUrl = null;
      glState.envCacheKey = null;
    }
    return;
  }

  let env = null;
  if (url === ANALYTIC_SKY_ID) {
    if (!analyticSettings) {
      throw new Error("Analytic sky settings are required.");
    }
    const settings = normalizeAnalyticSkySettings(analyticSettings);
    const key = `${ANALYTIC_SKY_ID}:${analyticSkyCacheKey(settings)}`;
    if (envCache.has(key)) {
      env = envCache.get(key);
    } else {
      logger.info("Generating analytic sky (Preetham/Perez) with WebGPU...");
      env = await generateAnalyticSkyEnvironment(settings, logger);
      env.samplingData = buildEnvSamplingData(env.data, env.width, env.height);
      env.version = key;
      envCache.set(key, env);
    }
  } else if (envCache.has(url)) {
    env = envCache.get(url);
  } else {
    logger.info(`Loading environment: ${url}`);
    env = await loadHDR(url, logger);
    env.samplingData = buildEnvSamplingData(env.data, env.width, env.height);
    env.version = url;
    envCache.set(url, env);
  }

  renderState.envData = env;
  renderState.envUrl = url;
  renderState.envCacheKey = env.version || url;

  if (glState && renderState.envData) {
    uploadEnvironmentToGl(renderState.envData);
  }
}

async function updateEnvironmentState() {
  setLoadingOverlay(true, "Loading environment...");
  renderState.envIntensity = clamp(Number(envIntensityInput.value), 0, 1.0);
  renderState.envMaxLuminance = clamp(Number(envMaxLumInput?.value ?? 50), 0, 500);
  const url = envSelect.value || null;
  let envChanged = false;

  try {
    if (url === ANALYTIC_SKY_ID) {
      const analyticSettings = getAnalyticSkySettingsFromUi();
      const analyticKey = `${ANALYTIC_SKY_ID}:${analyticSkyCacheKey(analyticSettings)}`;
      if (url !== renderState.envUrl || analyticKey !== renderState.envCacheKey) {
        await loadEnvironment(url, analyticSettings);
        envChanged = true;
      }
    } else if (url !== renderState.envUrl) {
      await loadEnvironment(url);
      envChanged = true;
    }
    resetAccumulation(envChanged ? "Environment updated." : "Environment intensity updated.");
  } catch (err) {
    logger.error(err.message || String(err));
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

function applyOrbitDrag(dx, dy) {
  cameraState.rotation = applyOrbitDragToRotation(cameraState.rotation, dx, dy);
}

function isTextEntryTarget(target) {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

function updatePointerFromMouseEvent(event) {
  if (!canvas) {
    throw new Error("Render canvas is missing.");
  }
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    throw new Error("Canvas has invalid size for pointer tracking.");
  }
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  pointerState.x = clamp(x, 0, rect.width);
  pointerState.y = clamp(y, 0, rect.height);
  pointerState.overCanvas = true;
}

function normalizeVec3(v) {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len < 1e-10) {
    throw new Error("Cannot normalize zero-length vector.");
  }
  return [v[0] / len, v[1] / len, v[2] / len];
}

function buildCameraRayFromCanvasPixel(camera, canvasX, canvasY) {
  if (!canvas) {
    throw new Error("Render canvas is missing.");
  }
  const width = Math.max(1, Math.floor(canvas.clientWidth));
  const height = Math.max(1, Math.floor(canvas.clientHeight));
  if (width <= 0 || height <= 0) {
    throw new Error("Canvas size is invalid for ray picking.");
  }

  const ndcX = (canvasX / width) * 2.0 - 1.0;
  const ndcY = 1.0 - (canvasY / height) * 2.0;
  const dir = [
    camera.forward[0] + camera.right[0] * ndcX + camera.up[0] * ndcY,
    camera.forward[1] + camera.right[1] * ndcX + camera.up[1] * ndcY,
    camera.forward[2] + camera.right[2] * ndcX + camera.up[2] * ndcY
  ];
  return normalizeVec3(dir);
}

function tracePointerHit(camera) {
  if (!sceneData) {
    return null;
  }
  const rayDir = buildCameraRayFromCanvasPixel(camera, pointerState.x, pointerState.y);
  const clip = getActiveClipPlane(camera);
  return traceSceneRay(sceneData, camera.origin, rayDir, {
    tMin: Math.max(1e-6, renderState.tMin),
    clip: clip.enabled ? { normal: clip.normal, offset: clip.offset, side: clip.side, enabled: true } : null
  });
}

function getActiveClipPlane(camera) {
  const enabled = Boolean(renderState.clipEnabled);
  const camForward = normalizeVec3(camera.forward);
  let normal = camForward;
  let offset = 0.0;
  let side = 1.0;

  if (renderState.clipLocked && renderState.clipLockedNormal) {
    normal = normalizeVec3(renderState.clipLockedNormal);
    if (renderState.clipLockedOffset != null) {
      offset = renderState.clipLockedOffset;
    }
    if (renderState.clipLockedSide != null) {
      side = renderState.clipLockedSide;
    }
  }

  if (enabled && !(renderState.clipLocked && renderState.clipLockedOffset != null)) {
    const planePoint = [
      camera.origin[0] + normal[0] * renderState.clipDistance,
      camera.origin[1] + normal[1] * renderState.clipDistance,
      camera.origin[2] + normal[2] * renderState.clipDistance
    ];
    offset = normal[0] * planePoint[0] + normal[1] * planePoint[1] + normal[2] * planePoint[2];
  }

  if (enabled && !(renderState.clipLocked && renderState.clipLockedSide != null)) {
    const camSide = normal[0] * camera.origin[0] + normal[1] * camera.origin[1] + normal[2] * camera.origin[2] - offset;
    side = camSide >= 0 ? 1 : -1;
  }

  return { enabled, normal, offset, side };
}

function hideHoverBoxOverlay() {
  if (!hoverBoxOverlay) return;
  hoverBoxOverlay.style.display = "none";
}

function drawHoverBoxOverlay(box) {
  if (!hoverBoxOverlay || !canvas || !canvasContainer) return;
  const canvasRect = canvas.getBoundingClientRect();
  const containerRect = canvasContainer.getBoundingClientRect();
  const offsetLeft = canvasRect.left - containerRect.left;
  const offsetTop = canvasRect.top - containerRect.top;
  hoverBoxOverlay.style.left = `${offsetLeft + box.minX}px`;
  hoverBoxOverlay.style.top = `${offsetTop + box.minY}px`;
  hoverBoxOverlay.style.width = `${box.width}px`;
  hoverBoxOverlay.style.height = `${box.height}px`;
  hoverBoxOverlay.style.display = "block";
}

function updateHoverBoxOverlay(camera = null) {
  if (!hoverBoxOverlay || !canvas) return;
  if (!sceneData || !pointerState.overCanvas) {
    hideHoverBoxOverlay();
    return;
  }

  const activeCamera = camera || computeCameraVectors();
  const hit = tracePointerHit(activeCamera);
  if (!hit) {
    hideHoverBoxOverlay();
    return;
  }

  const canvasWidth = Math.max(1, Math.floor(canvas.clientWidth));
  const canvasHeight = Math.max(1, Math.floor(canvas.clientHeight));
  const bounds = computePrimitiveWorldBounds(sceneData, hit.primType, hit.primIndex);
  const box = projectAabbToCanvasRect(bounds, activeCamera, canvasWidth, canvasHeight);
  if (!box) {
    hideHoverBoxOverlay();
    return;
  }
  drawHoverBoxOverlay(box);
}

function safeUpdateHoverBoxOverlay(camera = null) {
  try {
    updateHoverBoxOverlay(camera);
    hoverOverlayErrorMessage = null;
  } catch (err) {
    hideHoverBoxOverlay();
    const msg = err?.message || String(err);
    if (hoverOverlayErrorMessage !== msg) {
      hoverOverlayErrorMessage = msg;
      logger.warn(`[hover] ${msg}`);
    }
  }
}

function autofocusFromMouseRay() {
  if (!sceneData) {
    logger.warn("Focus not updated: no scene is loaded.");
    return;
  }
  if (!pointerState.overCanvas) {
    logger.info("Focus not updated: mouse is not over the render canvas.");
    return;
  }
  if (!dofFocusDistanceInput) {
    throw new Error("Depth-of-field focus distance input is missing.");
  }

  const camera = computeCameraVectors();
  const hit = tracePointerHit(camera);

  if (!hit) {
    logger.info("Focus not updated: no object found under mouse.");
    return;
  }

  const minFocus = Number(dofFocusDistanceInput.min);
  const maxFocus = Number(dofFocusDistanceInput.max);
  if (!Number.isFinite(minFocus) || !Number.isFinite(maxFocus)) {
    throw new Error("Depth-of-field focus slider range is invalid.");
  }
  const clampedFocus = clamp(hit.t, minFocus, maxFocus);
  setSliderValue(dofFocusDistanceInput, clampedFocus);
  dofFocusDistanceInput.dispatchEvent(new Event("input", { bubbles: true }));

  const message = `Focal distance updated to ${clampedFocus.toFixed(1)}`;
  logger.info(
    `[focus] ${message} (hit ${primTypeLabel(hit.primType)} ${hit.primIndex}, t=${hit.t.toFixed(3)})`
  );
  if (Math.abs(clampedFocus - hit.t) > 1e-6) {
    logger.warn(
      `[focus] Requested hit distance ${hit.t.toFixed(3)} exceeded focus slider range [${minFocus}, ${maxFocus}].`
    );
  }
}

function ensureWebGL() {
  if (!glState) {
    if (glInitFailed) {
      throw new Error("WebGL initialization previously failed.");
    }
    try {
      const { gl, traceProgram, displayProgram, vao } = initWebGL(canvas, logger);
      const blackEnvTex = createEnvTexture(gl, 1, 1, new Float32Array([0, 0, 0, 1]));
      const dummyVolumeTex = createVolumeTexture(gl, 1, 1, 1, new Float32Array([0]));
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
        volumeTex: dummyVolumeTex,
        dummyVolumeTex,
        volumeVersion: null,
        envMarginalCdfTex: dummyCdfTex,
        envConditionalCdfTex: dummyCdfTex,
        dummyCdfTex,
        envSize: [1, 1],
        envUrl: null,
        envCacheKey: null
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
  const triFlags = packTriFlags(sceneData.triFlags || new Float32Array(0), maxTextureSize);
  const primIndices = packPrimIndices(sceneData.primIndexBuffer, maxTextureSize);
  const spheresPacked = packSpheres(sceneData.spheres, maxTextureSize);
  const sphereColors = packSphereColors(sceneData.spheres, maxTextureSize);
  const cylindersPacked = packCylinders(sceneData.cylinders, maxTextureSize);
  const cylinderColors = packCylinderColors(sceneData.cylinders, maxTextureSize);

  const bvhTex = createDataTexture(gl, bvh.width, bvh.height, bvh.data);
  const triTex = createDataTexture(gl, tris.width, tris.height, tris.data);
  const triNormalTex = createDataTexture(gl, triNormals.width, triNormals.height, triNormals.data);
  const triColorTex = createDataTexture(gl, triColors.width, triColors.height, triColors.data);
  const triFlagTex = createDataTexture(gl, triFlags.width, triFlags.height, triFlags.data);
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
    triFlags,
    primIndices,
    spheresPacked,
    sphereColors,
    cylindersPacked,
    cylinderColors,
    bvhTex,
    triTex,
    triNormalTex,
    triColorTex,
    triFlagTex,
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

  if (renderState.envData && glState.envCacheKey !== renderState.envCacheKey) {
    uploadEnvironmentToGl(renderState.envData);
  }

  let volumeEnabled = renderState.volumeEnabled ? 1 : 0;
  let volumeMin = [0, 0, 0];
  let volumeMax = [0, 0, 0];
  let volumeInvSize = [0, 0, 0];
  let volumeMaxValue = 1.0;

  if (volumeEnabled) {
    if (!sceneData.volume) {
      throw new Error("Volume rendering enabled but no volume data is available. Reimport a PDB with volume enabled.");
    }
    const volume = sceneData.volume;
    const [nx, ny, nz] = volume.dims;
    if (!glState.volumeTex || glState.volumeVersion !== volume.version) {
      if (glState.volumeTex && glState.volumeTex !== glState.dummyVolumeTex) {
        gl.deleteTexture(glState.volumeTex);
      }
      logger.info(`Uploading volume texture (${nx}x${ny}x${nz})`);
      glState.volumeTex = createVolumeTexture(gl, nx, ny, nz, volume.data);
      glState.volumeVersion = volume.version;
    }
    volumeMin = [volume.bounds.minX, volume.bounds.minY, volume.bounds.minZ];
    volumeMax = [volume.bounds.maxX, volume.bounds.maxY, volume.bounds.maxZ];
    const sizeX = volumeMax[0] - volumeMin[0];
    const sizeY = volumeMax[1] - volumeMin[1];
    const sizeZ = volumeMax[2] - volumeMin[2];
    volumeInvSize = [
      sizeX > 0 ? 1 / sizeX : 0,
      sizeY > 0 ? 1 / sizeY : 0,
      sizeZ > 0 ? 1 / sizeZ : 0
    ];
    volumeMaxValue = volume.maxValue;
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
  const clip = getActiveClipPlane(camera);
  const clipEnabled = clip.enabled ? 1 : 0;
  const clipNormal = clip.normal;
  const clipOffset = clip.offset;
  const clipSide = clip.side;

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
  createTextureUnit3D(gl, glState.volumeTex || glState.dummyVolumeTex, 13);
  createTextureUnit(gl, glState.textures.triFlagTex, 14);

setTraceUniforms(gl, traceProgram, {
    bvhUnit: 0,
    triUnit: 1,
    triNormalUnit: 2,
    triColorUnit: 3,
    triFlagUnit: 14,
    primIndexUnit: 4,
    accumUnit: 5,
    envUnit: 6,
    sphereUnit: 9,
    sphereColorUnit: 10,
    cylinderUnit: 11,
    cylinderColorUnit: 12,
    volumeUnit: 13,
    camOrigin: camera.origin,
    camRight: camera.right,
    camUp: camera.up,
    camForward: camera.forward,
    resolution: [renderWidth, renderHeight],
    bvhTexSize: [glState.textures.bvh.width, glState.textures.bvh.height],
    triTexSize: [glState.textures.tris.width, glState.textures.tris.height],
    triNormalTexSize: [glState.textures.triNormals.width, glState.textures.triNormals.height],
    triColorTexSize: [glState.textures.triColors.width, glState.textures.triColors.height],
    triFlagTexSize: [glState.textures.triFlags.width, glState.textures.triFlags.height],
    primIndexTexSize: [glState.textures.primIndices.width, glState.textures.primIndices.height],
    sphereTexSize: [glState.textures.spheresPacked.width, glState.textures.spheresPacked.height],
    cylinderTexSize: [glState.textures.cylindersPacked.width, glState.textures.cylindersPacked.height],
    envTexSize: glState.envSize || [1, 1],
    frameIndex: renderState.frameIndex,
    triCount: sceneData.triCount,
    sphereCount: sceneData.sphereCount,
    cylinderCount: sceneData.cylinderCount,
    volumeEnabled,
    volumeMin,
    volumeMax,
    volumeInvSize,
    volumeMaxValue,
    volumeColor: renderState.volumeColor,
    volumeDensity: renderState.volumeDensity,
    volumeOpacity: renderState.volumeOpacity,
    volumeStep: renderState.volumeStep,
    volumeMaxSteps: renderState.volumeMaxSteps,
    volumeThreshold: renderState.volumeThreshold,
    useBvh: renderState.useBvh ? 1 : 0,
    useImportedColor: renderState.useImportedColor ? 1 : 0,
    baseColor: renderState.baseColor,
    metallic: renderState.metallic,
    roughness: renderState.roughness,
    rimBoost: renderState.rimBoost,
    maxBounces: renderState.maxBounces,
    exposure: renderState.exposure,
    dofEnabled: renderState.dofEnabled ? 1 : 0,
    dofAperture: renderState.dofAperture,
    dofFocusDistance: renderState.dofFocusDistance,
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
    surfaceIor: renderState.surfaceIor,
    surfaceTransmission: renderState.surfaceTransmission,
    surfaceOpacity: renderState.surfaceOpacity,
    surfaceFlagMode: sceneData.hasSurfaceFlags ? 1 : 0,
    envMarginalCdfUnit: 7,
    envConditionalCdfUnit: 8,
    envSize: glState.envSize || [1, 1],
    samplesPerBounce: renderState.samplesPerBounce,
    castShadows: renderState.castShadows ? 1 : 0,
    rayBias: renderState.rayBias,
    tMin: renderState.tMin,
    lights: renderState.lights,
    lightDirs,
    clipEnabled,
    clipNormal,
    clipOffset,
    clipSide,
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
  safeUpdateHoverBoxOverlay(camera);
}

async function startRenderLoop() {
  if (isRendering) {
    return;
  }
  isRendering = true;
  logger.info("Interactive render started.");
  let lastTime = performance.now();

  const loop = (time) => {
    if (!isRendering) {
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
  if (renderOverlay) {
    renderOverlay.style.display = "none";
  }
  logger.info("Paused.");
}

async function loadExampleScene(url) {
  if (isLoading) return;
  isLoading = true;
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
    } else if (url.startsWith("mol:")) {
      const molName = url.slice(4);
      await loadBuiltinMolecule(molName);
      success = true;
    } else if (url.startsWith("pdb:")) {
      const pdbId = url.slice(4);
      await loadPDBById(pdbId);
      success = true;
    } else {
      throw new Error(`Unsupported example selection: ${url}`);
    }
  } catch (err) {
    logger.error(err.message || String(err));
  } finally {
    isLoading = false;
    glState = null;
    setLoadingOverlay(false);
  }
  return success;
}

loadExampleBtn.addEventListener("click", async () => {
  const value = exampleSelect.value;
  setLoadingOverlay(true, "Loading example...");
  let loaded = false;
  try {
    loaded = await loadExampleScene(value);
  } catch (err) {
    logger.error(err.message || String(err));
  } finally {
    setLoadingOverlay(false);
  }
  if (loaded && sceneData) {
    await startRenderLoop();
  }
});

// Load molecular file from file input
molFileInput.addEventListener("change", async () => {
  setLoadingOverlay(true, "Loading molecule...");
  let loaded = false;
  try {
    const file = molFileInput.files?.[0];
    if (!file) return;
    const text = await file.text();
    await loadMolecularFile(text, file.name);
    glState = null;
    loaded = true;
  } catch (err) {
    logger.error(err.message || String(err));
  } finally {
    setLoadingOverlay(false);
  }
  if (loaded && sceneData) {
    await startRenderLoop();
  }
});

loadPdbIdBtn.addEventListener("click", async () => {
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
    setLoadingOverlay(false);
  }
  if (loaded && sceneData) {
    await startRenderLoop();
  }
});

canvas.addEventListener("mousedown", (event) => {
  updatePointerFromMouseEvent(event);
  safeUpdateHoverBoxOverlay();
  inputState.dragging = true;
  inputState.lastX = event.clientX;
  inputState.lastY = event.clientY;
  inputState.rotateAxisLock = null;
  if (event.button === 2) {
    inputState.dragMode = "pan";
  } else if (event.shiftKey) {
    inputState.dragMode = "pan";
  } else if (event.ctrlKey) {
    inputState.dragMode = "zoom";
  } else {
    inputState.dragMode = "rotate";
  }
  markInteractionActive();
});

canvas.addEventListener("mouseup", () => {
  inputState.dragging = false;
  inputState.rotateAxisLock = null;
});

canvas.addEventListener("contextmenu", (event) => {
  event.preventDefault();
});
canvasContainer?.addEventListener("contextmenu", (event) => {
  event.preventDefault();
});

canvas.addEventListener("mouseleave", () => {
  pointerState.overCanvas = false;
  hideHoverBoxOverlay();
  inputState.dragging = false;
  inputState.rotateAxisLock = null;
});

canvas.addEventListener("mousemove", (event) => {
  updatePointerFromMouseEvent(event);
  if (!inputState.dragging) {
    safeUpdateHoverBoxOverlay();
    return;
  }
  const dx = event.clientX - inputState.lastX;
  const dy = event.clientY - inputState.lastY;
  inputState.lastX = event.clientX;
  inputState.lastY = event.clientY;
  markInteractionActive();

  const rightDown = (event.buttons & 2) !== 0;
  const mode = rightDown ? "pan" : (event.shiftKey ? "pan" : (event.ctrlKey ? "zoom" : inputState.dragMode));

  if (mode === "pan") {
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
    renderState.cameraDirty = true;
    safeUpdateHoverBoxOverlay();
    return;
  }

  if (mode === "zoom") {
    const zoom = Math.exp(dy * 0.005);
    const sceneScale = sceneData?.sceneScale || 1.0;
    const minDist = Math.max(0.1, sceneScale * 0.1);
    const maxDist = Math.max(100, sceneScale * 20);
    cameraState.distance = clamp(cameraState.distance * zoom, minDist, maxDist);
    renderState.cameraDirty = true;
    safeUpdateHoverBoxOverlay();
    return;
  }
  inputState.rotateAxisLock = resolveRotationLock(inputState.rotateAxisLock, dx, dy);
  if (!inputState.rotateAxisLock) {
    return;
  }
  const lockDx = inputState.rotateAxisLock === "yaw" ? dx : 0;
  const lockDy = inputState.rotateAxisLock === "pitch" ? dy : 0;
  applyOrbitDrag(lockDx, lockDy);
  renderState.cameraDirty = true;
  safeUpdateHoverBoxOverlay();
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
  safeUpdateHoverBoxOverlay();
}, { passive: false });

window.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  const isFocusShortcut = event.code === "KeyF" || key === "f";
  if (isFocusShortcut && !event.repeat && !isTextEntryTarget(event.target)) {
    try {
      autofocusFromMouseRay();
    } catch (err) {
      const msg = err?.message || String(err);
      logger.error(msg);
    }
    event.preventDefault();
    return;
  }
  inputState.keys.add(key);
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
  updateEnvironmentVisibility();
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});
envIntensityInput.addEventListener("input", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});
envMaxLumInput?.addEventListener("input", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});
analyticSkyResolutionSelect?.addEventListener("change", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});
analyticSkyTurbidityInput?.addEventListener("input", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});
analyticSkySunAzimuthInput?.addEventListener("input", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});
analyticSkySunElevationInput?.addEventListener("input", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});
analyticSkyIntensityInput?.addEventListener("input", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});
analyticSkySunIntensityInput?.addEventListener("input", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});
analyticSkySunRadiusInput?.addEventListener("input", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});
analyticSkyGroundAlbedoInput?.addEventListener("input", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});
analyticSkyHorizonSoftnessInput?.addEventListener("input", () => {
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setActiveTab(button.dataset.tabButton);
  });
});

useImportedColorToggle.addEventListener("change", updateMaterialState);
clipEnableToggle?.addEventListener("change", () => updateClipState({ preserveLock: true }));
clipDistanceInput?.addEventListener("input", () => updateClipState({ preserveLock: true }));
clipLockToggle?.addEventListener("change", () => updateClipState({ preserveLock: false }));
surfaceModeSelect?.addEventListener("change", async () => {
  const mode = surfaceModeSelect.value;
  if (mode === "wasm") {
    try {
      logger.info("Loading SES WASM module...");
      await initSurfaceWasm();
      logger.info("SES WASM module ready.");
    } catch (err) {
      logger.error(err.message || String(err));
    }
  } else if (mode === "webgl") {
    if (!webglSurfaceAvailable()) {
      logger.warn("WebGL surface not available - requires WebGL2 + EXT_color_buffer_float");
    } else {
      logger.info("WebGL surface computation available.");
    }
  }
  refreshSurfaceAtomMode().catch((err) => logger.error(err.message || String(err)));
});
materialSelect?.addEventListener("change", () => {
  applyMaterialPreset(materialSelect.value);
  updateMaterialVisibility();
  updateMaterialState();
  refreshSurfaceAtomMode().catch((err) => logger.error(err.message || String(err)));
});
baseColorInput.addEventListener("input", updateMaterialState);
metallicInput.addEventListener("input", updateMaterialState);
roughnessInput.addEventListener("input", updateMaterialState);
rimBoostInput?.addEventListener("input", updateMaterialState);
matteSpecularInput?.addEventListener("input", updateMaterialState);
matteRoughnessInput?.addEventListener("input", updateMaterialState);
matteDiffuseRoughnessInput?.addEventListener("input", updateMaterialState);
wrapDiffuseInput?.addEventListener("input", updateMaterialState);
surfaceShowAtomsToggle?.addEventListener("change", () => {
  updateMaterialState();
  refreshSurfaceAtomMode().catch((err) => logger.error(err.message || String(err)));
});
surfaceIorInput?.addEventListener("input", updateMaterialState);
surfaceTransmissionInput?.addEventListener("input", updateMaterialState);
showSheetHbondsToggle?.addEventListener("change", () => {
  if (!lastMolContext || !lastMolContext.molData) {
    logger.error("No molecular data loaded for sheet H-bond debug rendering.");
    return;
  }
  loadMolecularGeometry(
    lastMolContext.spheres,
    lastMolContext.cylinders,
    lastMolContext.molData,
    lastMolContext.options,
    lastMolContext.volumeData,
    lastMolContext.surfaceAtomMode
  ).catch((err) => logger.error(err.message || String(err)));
});
surfaceOpacityInput?.addEventListener("input", updateMaterialState);
maxBouncesInput.addEventListener("input", updateMaterialState);
exposureInput.addEventListener("input", updateMaterialState);
dofEnableToggle?.addEventListener("change", () => {
  updateDofVisibility();
  updateMaterialState();
});
dofApertureInput?.addEventListener("input", updateMaterialState);
dofFocusDistanceInput?.addEventListener("input", updateMaterialState);
toneMapSelect?.addEventListener("change", updateMaterialState);
ambientIntensityInput.addEventListener("input", updateMaterialState);
ambientColorInput.addEventListener("input", updateMaterialState);
samplesPerBounceInput.addEventListener("input", updateMaterialState);
maxFramesInput?.addEventListener("input", updateRenderLimits);
shadowToggle.addEventListener("change", updateMaterialState);

volumeEnableToggle?.addEventListener("change", () => {
  try {
    updateVolumeState();
  } catch (err) {
    logger.error(err.message || String(err));
  }
});
volumeColorInput?.addEventListener("input", () => {
  try {
    updateVolumeState();
  } catch (err) {
    logger.error(err.message || String(err));
  }
});
volumeDensityInput?.addEventListener("input", () => {
  try {
    updateVolumeState();
  } catch (err) {
    logger.error(err.message || String(err));
  }
});
volumeOpacityInput?.addEventListener("input", () => {
  try {
    updateVolumeState();
  } catch (err) {
    logger.error(err.message || String(err));
  }
});
volumeStepInput?.addEventListener("input", () => {
  try {
    updateVolumeState();
  } catch (err) {
    logger.error(err.message || String(err));
  }
});
volumeMaxStepsInput?.addEventListener("input", () => {
  try {
    updateVolumeState();
  } catch (err) {
    logger.error(err.message || String(err));
  }
});
volumeThresholdInput?.addEventListener("input", () => {
  try {
    updateVolumeState();
  } catch (err) {
    logger.error(err.message || String(err));
  }
});

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
    logger.info("Ready. Load an example or choose a molecular file.");
  }

  updateMaterialState();
  updateMaterialVisibility();
  updateDofVisibility();
  updateEnvironmentVisibility();
  updateClipState({ preserveLock: true });
  updateRenderLimits();
  updateLightState();
  updateEnvironmentState().catch((err) => logger.error(err.message || String(err)));
});

setActiveTab("scene");
