import { createLogger } from "./logger.js";
import { loadGltfFromText } from "./gltf.js";
import { buildSAHBVH, flattenBVH } from "./bvh.js";
import { packBvhNodes, packTriangles, packTriIndices, packMaterials } from "./packing.js";
import {
  initWebGL,
  createDataTexture,
  createAccumTargets,
  resizeAccumTargets,
  createTextureUnit,
  setTraceUniforms,
  setDisplayUniforms,
  drawFullscreen,
  MAX_BRUTE_FORCE_TRIS
} from "./webgl.js";

const canvas = document.getElementById("view");
const statusEl = document.getElementById("status");
const logger = createLogger(statusEl);

const exampleSelect = document.getElementById("exampleSelect");
const loadExampleBtn = document.getElementById("loadExample");
const fileInput = document.getElementById("fileInput");
const loadFileBtn = document.getElementById("loadFile");
const renderBtn = document.getElementById("renderBtn");
const scaleSelect = document.getElementById("scaleSelect");
const bruteforceToggle = document.getElementById("bruteforceToggle");

let sceneData = null;
let glState = null;
let isRendering = false;
let isLoading = false;
let loggedFirstFrame = false;
let glInitFailed = false;

const cameraState = {
  target: [0, 0, 0],
  distance: 4,
  yaw: 0,
  pitch: 0,
  fov: Math.PI / 3,
  width: 1,
  height: 1
};

const renderState = {
  scale: 0.75,
  frameIndex: 0,
  cameraDirty: true,
  useBvh: true
};

const inputState = {
  dragging: false,
  lastX: 0,
  lastY: 0,
  keys: new Set()
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
  const { positions, indices } = await loadGltfFromText(text, baseUrl, fetch);
  logger.info(`Loaded ${positions.length / 3} vertices, ${indices.length / 3} triangles`);
  if (positions.length === 0 || indices.length === 0) {
    throw new Error(
      `Loaded empty geometry (positions: ${positions.length}, indices: ${indices.length}).`
    );
  }

  logger.info("Building SAH BVH on CPU");
  const bvh = buildSAHBVH(positions, indices, { maxLeafSize: 4, maxDepth: 32 });
  logger.info(`BVH nodes: ${bvh.nodes.length}`);

  const flat = flattenBVH(bvh.nodes, bvh.tris);

  sceneData = {
    positions,
    indices,
    nodes: bvh.nodes,
    tris: bvh.tris,
    triIndexBuffer: flat.triIndexBuffer,
    triCount: bvh.tris.length,
    triIndexCount: flat.triIndexBuffer.length
  };
  renderState.frameIndex = 0;
  renderState.cameraDirty = true;

  const bounds = computeBounds(positions);
  if (bounds) {
    logger.info(
      `Bounds min (${bounds.minX.toFixed(2)}, ${bounds.minY.toFixed(2)}, ${bounds.minZ.toFixed(2)}) max (${bounds.maxX.toFixed(2)}, ${bounds.maxY.toFixed(2)}, ${bounds.maxZ.toFixed(2)})`
    );
    applyCameraToBounds(bounds);
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
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
  cameraState.yaw = 0;
  cameraState.pitch = 0;
  renderState.cameraDirty = true;
  renderState.frameIndex = 0;
  logger.info(
    `Camera fit to bounds center (${cx.toFixed(2)}, ${cy.toFixed(2)}, ${cz.toFixed(2)}) radius ${radius.toFixed(2)}`
  );
}

function computeCameraVectors() {
  const { yaw, pitch, distance, target, fov, width, height } = cameraState;
  const cosPitch = Math.cos(pitch);
  const sinPitch = Math.sin(pitch);
  const cosYaw = Math.cos(yaw);
  const sinYaw = Math.sin(yaw);

  const forward = [cosPitch * sinYaw, sinPitch, cosPitch * cosYaw];
  const origin = [
    target[0] - forward[0] * distance,
    target[1] - forward[1] * distance,
    target[2] - forward[2] * distance
  ];

  const worldUp = [0, 1, 0];
  const right = [
    worldUp[1] * forward[2] - worldUp[2] * forward[1],
    worldUp[2] * forward[0] - worldUp[0] * forward[2],
    worldUp[0] * forward[1] - worldUp[1] * forward[0]
  ];
  const rightLen = Math.hypot(right[0], right[1], right[2]) || 1;
  right[0] /= rightLen;
  right[1] /= rightLen;
  right[2] /= rightLen;

  const up = [
    forward[1] * right[2] - forward[2] * right[1],
    forward[2] * right[0] - forward[0] * right[2],
    forward[0] * right[1] - forward[1] * right[0]
  ];

  const aspect = width / height;
  const scale = Math.tan(fov / 2);
  const rightScaled = [right[0] * scale * aspect, right[1] * scale * aspect, right[2] * scale * aspect];
  const upScaled = [up[0] * scale, up[1] * scale, up[2] * scale];

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

function ensureWebGL() {
  if (!glState) {
    if (glInitFailed) {
      throw new Error("WebGL initialization previously failed.");
    }
    try {
      const { gl, traceProgram, displayProgram, vao } = initWebGL(canvas, logger);
      glState = {
        gl,
        traceProgram,
        displayProgram,
        vao,
        textures: null,
        accum: null,
        frameParity: 0
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
  const triIndices = packTriIndices(sceneData.triIndexBuffer, maxTextureSize);
  const materials = packMaterials(maxTextureSize);

  const bvhTex = createDataTexture(gl, bvh.width, bvh.height, bvh.data);
  const triTex = createDataTexture(gl, tris.width, tris.height, tris.data);
  const triIndexTex = createDataTexture(gl, triIndices.width, triIndices.height, triIndices.data);
  const matTex = createDataTexture(gl, materials.width, materials.height, materials.data);

  return {
    bvh,
    tris,
    triIndices,
    materials,
    bvhTex,
    triTex,
    triIndexTex,
    matTex
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
  createTextureUnit(gl, glState.textures.triIndexTex, 2);
  createTextureUnit(gl, glState.accum.textures[prevIndex], 3);

  setTraceUniforms(gl, traceProgram, {
    bvhUnit: 0,
    triUnit: 1,
    triIndexUnit: 2,
    accumUnit: 3,
    camOrigin: camera.origin,
    camRight: camera.right,
    camUp: camera.up,
    camForward: camera.forward,
    resolution: [renderWidth, renderHeight],
    bvhTexSize: [glState.textures.bvh.width, glState.textures.bvh.height],
    triTexSize: [glState.textures.tris.width, glState.textures.tris.height],
    triIndexTexSize: [glState.textures.triIndices.width, glState.textures.triIndices.height],
    frameIndex: renderState.frameIndex,
    triCount: sceneData.triCount,
    useBvh: renderState.useBvh ? 1 : 0
  });

  gl.useProgram(traceProgram);
  drawFullscreen(gl);

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, displayWidth, displayHeight);
  createTextureUnit(gl, glState.accum.textures[accumIndex], 0);

  setDisplayUniforms(gl, displayProgram, {
    displayUnit: 0,
    displayResolution: [displayWidth, displayHeight]
  });

  gl.useProgram(displayProgram);
  drawFullscreen(gl);

  glState.frameParity = prevIndex;
  renderState.frameIndex += 1;
  renderState.cameraDirty = false;
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
    const dt = Math.max(0.001, (time - lastTime) / 1000);
    lastTime = time;
    const moved = updateCameraFromInput(dt);
    if (moved) {
      renderState.cameraDirty = true;
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
  logger.info("Paused.");
}

async function loadExampleScene(url) {
  if (isLoading) return;
  isLoading = true;
  renderBtn.disabled = true;
  try {
    logger.info(`Loading example: ${url}`);
    const text = await fetchText(url);
    const baseUrl = new URL(url, window.location.href).toString();
    await loadGltfText(text, baseUrl);
    logger.info("Example loaded.");
  } catch (err) {
    logger.error(err.message || String(err));
  } finally {
    renderBtn.disabled = false;
    isLoading = false;
    glState = null;
  }
}

loadExampleBtn.addEventListener("click", async () => {
  const url = exampleSelect.value;
  await loadExampleScene(url);
});

loadFileBtn.addEventListener("click", async () => {
  renderBtn.disabled = true;
  try {
    const file = fileInput.files?.[0];
    if (!file) {
      throw new Error("Please pick a .gltf file.");
    }
    logger.info(`Loading file: ${file.name}`);
    const text = await file.text();
    await loadGltfText(text, null);
    logger.info("File loaded.");
    glState = null;
  } catch (err) {
    logger.error(err.message || String(err));
  } finally {
    renderBtn.disabled = false;
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
});

canvas.addEventListener("mouseup", () => {
  inputState.dragging = false;
});

canvas.addEventListener("mouseleave", () => {
  inputState.dragging = false;
});

canvas.addEventListener("mousemove", (event) => {
  if (!inputState.dragging) return;
  const dx = event.clientX - inputState.lastX;
  const dy = event.clientY - inputState.lastY;
  inputState.lastX = event.clientX;
  inputState.lastY = event.clientY;

  const rotateSpeed = 0.005;
  cameraState.yaw -= dx * rotateSpeed;
  cameraState.pitch -= dy * rotateSpeed;
  cameraState.pitch = clamp(cameraState.pitch, -1.45, 1.45);
  renderState.cameraDirty = true;
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const zoom = Math.exp(event.deltaY * 0.0015);
  cameraState.distance = clamp(cameraState.distance * zoom, 0.6, 25);
  renderState.cameraDirty = true;
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
  renderState.scale = value;
  renderState.frameIndex = 0;
  renderState.cameraDirty = true;
  loggedFirstFrame = false;
  glState = null;
  logger.info(`Render scale set to ${value.toFixed(2)}x`);
});

bruteforceToggle.addEventListener("change", () => {
  const mode = bruteforceToggle.value;
  renderState.useBvh = mode !== "bruteforce";
  renderState.frameIndex = 0;
  renderState.cameraDirty = true;
  loggedFirstFrame = false;
  logger.info(`Traversal mode: ${renderState.useBvh ? "BVH" : "Brute force"}`);
});

const params = new URLSearchParams(window.location.search);
const autorun = params.get("autorun");
const exampleParam = params.get("example");
if (exampleParam) {
  const option = Array.from(exampleSelect.options).find((opt) => opt.value === exampleParam);
  if (option) {
    exampleSelect.value = exampleParam;
  }
}

if (autorun === "1") {
  logger.info("Autorun enabled via query string.");
  setTimeout(() => {
    loadExampleScene(exampleSelect.value).then(() => startRenderLoop());
  }, 0);
} else {
  logger.info("Ready. Load an example or choose a .gltf file.");
}
