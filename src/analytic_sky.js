export const ANALYTIC_SKY_ID = "analytic://preetham-perez";

export const DEFAULT_ANALYTIC_SKY_SETTINGS = {
  width: 1024,
  height: 512,
  turbidity: 2.5,
  sunAzimuthDeg: 30.0,
  sunElevationDeg: 35.0,
  skyIntensity: 1.0,
  sunIntensity: 20.0,
  sunAngularRadiusDeg: 0.27,
  groundAlbedo: 0.2,
  horizonSoftness: 0.12
};

function degToRad(deg) {
  return deg * Math.PI / 180.0;
}

function formatFloat(value) {
  return Number(value).toFixed(4);
}

export function normalizeAnalyticSkySettings(raw = {}) {
  const settings = { ...DEFAULT_ANALYTIC_SKY_SETTINGS, ...(raw || {}) };

  const width = Number(settings.width);
  const height = Number(settings.height);
  if (!Number.isInteger(width) || width <= 0) {
    throw new Error("Analytic sky width must be a positive integer.");
  }
  if (!Number.isInteger(height) || height <= 0) {
    throw new Error("Analytic sky height must be a positive integer.");
  }

  const turbidity = Number(settings.turbidity);
  const sunAzimuthDeg = Number(settings.sunAzimuthDeg);
  const sunElevationDeg = Number(settings.sunElevationDeg);
  const skyIntensity = Number(settings.skyIntensity);
  const sunIntensity = Number(settings.sunIntensity);
  const sunAngularRadiusDeg = Number(settings.sunAngularRadiusDeg);
  const groundAlbedo = Number(settings.groundAlbedo);
  const horizonSoftness = Number(settings.horizonSoftness);

  const numeric = [
    ["turbidity", turbidity],
    ["sunAzimuthDeg", sunAzimuthDeg],
    ["sunElevationDeg", sunElevationDeg],
    ["skyIntensity", skyIntensity],
    ["sunIntensity", sunIntensity],
    ["sunAngularRadiusDeg", sunAngularRadiusDeg],
    ["groundAlbedo", groundAlbedo],
    ["horizonSoftness", horizonSoftness]
  ];
  for (const [name, value] of numeric) {
    if (!Number.isFinite(value)) {
      throw new Error(`Analytic sky ${name} must be finite.`);
    }
  }

  if (turbidity < 1.0 || turbidity > 20.0) {
    throw new Error("Analytic sky turbidity must be between 1 and 20.");
  }
  if (sunElevationDeg < -10.0 || sunElevationDeg > 90.0) {
    throw new Error("Analytic sky sun elevation must be between -10 and 90 degrees.");
  }
  if (skyIntensity < 0.0 || skyIntensity > 100.0) {
    throw new Error("Analytic sky intensity must be between 0 and 100.");
  }
  if (sunIntensity < 0.0 || sunIntensity > 10000.0) {
    throw new Error("Analytic sky sun intensity must be between 0 and 10000.");
  }
  if (sunAngularRadiusDeg <= 0.0 || sunAngularRadiusDeg > 5.0) {
    throw new Error("Analytic sky sun angular radius must be > 0 and <= 5 degrees.");
  }
  if (groundAlbedo < 0.0 || groundAlbedo > 1.0) {
    throw new Error("Analytic sky ground albedo must be between 0 and 1.");
  }
  if (horizonSoftness <= 0.0 || horizonSoftness > 1.0) {
    throw new Error("Analytic sky horizon softness must be > 0 and <= 1.");
  }

  return {
    width,
    height,
    turbidity,
    sunAzimuthDeg,
    sunElevationDeg,
    skyIntensity,
    sunIntensity,
    sunAngularRadiusDeg,
    groundAlbedo,
    horizonSoftness
  };
}

export function analyticSkyCacheKey(rawSettings = {}) {
  const s = normalizeAnalyticSkySettings(rawSettings);
  return [
    s.width,
    s.height,
    formatFloat(s.turbidity),
    formatFloat(s.sunAzimuthDeg),
    formatFloat(s.sunElevationDeg),
    formatFloat(s.skyIntensity),
    formatFloat(s.sunIntensity),
    formatFloat(s.sunAngularRadiusDeg),
    formatFloat(s.groundAlbedo),
    formatFloat(s.horizonSoftness)
  ].join("|");
}

export function computeSunDirection(sunAzimuthDeg, sunElevationDeg) {
  const az = degToRad(sunAzimuthDeg);
  const el = degToRad(sunElevationDeg);
  const cosEl = Math.cos(el);
  const dir = [
    Math.cos(az) * cosEl,
    Math.sin(el),
    Math.sin(az) * cosEl
  ];
  const len = Math.hypot(dir[0], dir[1], dir[2]) || 1.0;
  return [dir[0] / len, dir[1] / len, dir[2] / len];
}

const SKY_COMPUTE_WGSL = `
const PI: f32 = 3.14159265358979323846;

struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
  sun_dir: vec3<f32>,
  turbidity: f32,
  sky_intensity: f32,
  sun_intensity: f32,
  sun_angular_radius: f32,
  ground_albedo: f32,
  horizon_softness: f32,
  _pad2: f32,
};

struct Perez {
  A: f32,
  B: f32,
  C: f32,
  D: f32,
  E: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_pixels: array<vec4<f32>>;

fn perez_coeff_y(T: f32) -> Perez {
  return Perez(
    0.1787 * T - 1.4630,
    -0.3554 * T + 0.4275,
    -0.0227 * T + 5.3251,
    0.1206 * T - 2.5771,
    -0.0670 * T + 0.3703
  );
}

fn perez_coeff_x(T: f32) -> Perez {
  return Perez(
    -0.0193 * T - 0.2592,
    -0.0665 * T + 0.0008,
    -0.0004 * T + 0.2125,
    -0.0641 * T - 0.8989,
    -0.0033 * T + 0.0452
  );
}

fn perez_coeff_yy(T: f32) -> Perez {
  return Perez(
    -0.0167 * T - 0.2608,
    -0.0950 * T + 0.0092,
    -0.0079 * T + 0.2102,
    -0.0441 * T - 1.6537,
    -0.0109 * T + 0.0529
  );
}

fn perez_eval(c: Perez, theta: f32, gamma: f32) -> f32 {
  let cos_theta = max(cos(theta), 0.01);
  let part1 = 1.0 + c.A * exp(c.B / cos_theta);
  let cos_gamma = cos(gamma);
  let part2 = 1.0 + c.C * exp(c.D * gamma) + c.E * cos_gamma * cos_gamma;
  return part1 * part2;
}

fn zenith_luminance(T: f32, theta_s: f32) -> f32 {
  let chi = (4.0 / 9.0 - T / 120.0) * (PI - 2.0 * theta_s);
  let yz = (4.0453 * T - 4.9710) * tan(chi) - 0.2155 * T + 2.4192;
  return max(yz, 0.001);
}

fn zenith_x(T: f32, theta_s: f32) -> f32 {
  let t2 = T * T;
  let th = theta_s;
  let th2 = th * th;
  let th3 = th2 * th;
  let term1 = (0.00165 * th3 - 0.00374 * th2 + 0.00208 * th + 0.0) * t2;
  let term2 = (-0.02902 * th3 + 0.06377 * th2 - 0.03202 * th + 0.00394) * T;
  let term3 = 0.11693 * th3 - 0.21196 * th2 + 0.06052 * th + 0.25885;
  return clamp(term1 + term2 + term3, 0.001, 0.999);
}

fn zenith_y(T: f32, theta_s: f32) -> f32 {
  let t2 = T * T;
  let th = theta_s;
  let th2 = th * th;
  let th3 = th2 * th;
  let term1 = (0.00275 * th3 - 0.00610 * th2 + 0.00317 * th + 0.0) * t2;
  let term2 = (-0.04214 * th3 + 0.08970 * th2 - 0.04153 * th + 0.00516) * T;
  let term3 = 0.15346 * th3 - 0.26756 * th2 + 0.06669 * th + 0.26688;
  return clamp(term1 + term2 + term3, 0.001, 0.999);
}

fn xyY_to_rgb(x: f32, y: f32, Y: f32) -> vec3<f32> {
  if (y < 1e-4) {
    return vec3<f32>(0.0);
  }
  let X = (x / y) * Y;
  let Z = ((1.0 - x - y) / y) * Y;
  let r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
  let g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
  let b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z;
  return max(vec3<f32>(r, g, b), vec3<f32>(0.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let width_f = f32(params.width);
  let height_f = f32(params.height);
  let uv = vec2<f32>((f32(gid.x) + 0.5) / width_f, (f32(gid.y) + 0.5) / height_f);
  let phi = uv.x * 2.0 * PI;
  let theta = uv.y * PI;
  let sin_theta = sin(theta);
  let dir = vec3<f32>(cos(phi) * sin_theta, cos(theta), sin(phi) * sin_theta);
  let sun_dir = normalize(params.sun_dir);
  let theta_s = acos(clamp(sun_dir.y, -1.0, 1.0));
  let gamma = acos(clamp(dot(dir, sun_dir), -1.0, 1.0));

  let T = max(params.turbidity, 1.0);
  let y_coeff = perez_coeff_y(T);
  let x_coeff = perez_coeff_x(T);
  let yy_coeff = perez_coeff_yy(T);

  let y_zenith = zenith_luminance(T, theta_s);
  let x_zenith = zenith_x(T, theta_s);
  let yy_zenith = zenith_y(T, theta_s);

  let y_norm = perez_eval(y_coeff, 0.0, theta_s);
  let x_norm = perez_eval(x_coeff, 0.0, theta_s);
  let yy_norm = perez_eval(yy_coeff, 0.0, theta_s);

  let Y = max(0.0, y_zenith * perez_eval(y_coeff, theta, gamma) / max(y_norm, 1e-4));
  var x = x_zenith * perez_eval(x_coeff, theta, gamma) / max(x_norm, 1e-4);
  var yy = yy_zenith * perez_eval(yy_coeff, theta, gamma) / max(yy_norm, 1e-4);
  x = clamp(x, 0.001, 0.999);
  yy = clamp(yy, 0.001, 0.999);
  if (x + yy > 0.999) {
    let scale = 0.999 / (x + yy);
    x = x * scale;
    yy = yy * scale;
  }

  var rgb = xyY_to_rgb(x, yy, Y) * params.sky_intensity;

  if (dir.y < 0.0) {
    let t = clamp(abs(dir.y) / max(params.horizon_softness, 1e-4), 0.0, 1.0);
    let ground = vec3<f32>(params.ground_albedo * params.sky_intensity);
    rgb = mix(rgb * 0.05, ground, t);
  }

  let sun_sigma = max(params.sun_angular_radius, 1e-4);
  let sun_glow = exp(-0.5 * pow(gamma / sun_sigma, 2.0));
  rgb = rgb + vec3<f32>(params.sun_intensity * sun_glow);

  let idx = gid.y * params.width + gid.x;
  out_pixels[idx] = vec4<f32>(max(rgb, vec3<f32>(0.0)), 1.0);
}
`;

let cachedDevicePromise = null;
let cachedPipeline = null;

async function getWebGPUDevice() {
  const nav = globalThis.navigator;
  if (!nav || !nav.gpu) {
    throw new Error("WebGPU is required for analytic sky generation.");
  }
  if (!cachedDevicePromise) {
    cachedDevicePromise = (async () => {
      const adapter = await nav.gpu.requestAdapter();
      if (!adapter) {
        throw new Error("Failed to acquire WebGPU adapter for analytic sky generation.");
      }
      return adapter.requestDevice();
    })();
  }
  return cachedDevicePromise;
}

async function getComputePipeline(device) {
  if (cachedPipeline) return cachedPipeline;
  const module = device.createShaderModule({ code: SKY_COMPUTE_WGSL });
  cachedPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module,
      entryPoint: "main"
    }
  });
  return cachedPipeline;
}

function buildParamsBuffer(settings) {
  const sunDir = computeSunDirection(settings.sunAzimuthDeg, settings.sunElevationDeg);
  const params = new ArrayBuffer(64);
  const view = new DataView(params);

  view.setUint32(0, settings.width, true);
  view.setUint32(4, settings.height, true);
  view.setUint32(8, 0, true);
  view.setUint32(12, 0, true);
  view.setFloat32(16, sunDir[0], true);
  view.setFloat32(20, sunDir[1], true);
  view.setFloat32(24, sunDir[2], true);
  view.setFloat32(28, settings.turbidity, true);
  view.setFloat32(32, settings.skyIntensity, true);
  view.setFloat32(36, settings.sunIntensity, true);
  view.setFloat32(40, degToRad(settings.sunAngularRadiusDeg), true);
  view.setFloat32(44, settings.groundAlbedo, true);
  view.setFloat32(48, settings.horizonSoftness, true);
  view.setFloat32(52, 0.0, true);

  return new Uint8Array(params);
}

export async function generateAnalyticSkyEnvironment(rawSettings = {}, logger = null) {
  const settings = normalizeAnalyticSkySettings(rawSettings);
  const cacheKey = analyticSkyCacheKey(settings);
  const device = await getWebGPUDevice();
  const pipeline = await getComputePipeline(device);

  const gpuBufferUsage = globalThis.GPUBufferUsage;
  if (!gpuBufferUsage) {
    throw new Error("GPUBufferUsage is unavailable; WebGPU context is incomplete.");
  }

  const pixelCount = settings.width * settings.height;
  const pixelBytes = pixelCount * 4 * 4;
  const paramsBytes = buildParamsBuffer(settings);

  const paramsBuffer = device.createBuffer({
    size: 64,
    usage: gpuBufferUsage.UNIFORM | gpuBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsBytes);

  const outputBuffer = device.createBuffer({
    size: pixelBytes,
    usage: gpuBufferUsage.STORAGE | gpuBufferUsage.COPY_SRC
  });
  const readbackBuffer = device.createBuffer({
    size: pixelBytes,
    usage: gpuBufferUsage.COPY_DST | gpuBufferUsage.MAP_READ
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } }
    ]
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(
    Math.ceil(settings.width / 8),
    Math.ceil(settings.height / 8),
    1
  );
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, pixelBytes);
  device.queue.submit([encoder.finish()]);

  await readbackBuffer.mapAsync(globalThis.GPUMapMode.READ);
  const mapped = readbackBuffer.getMappedRange();
  const data = new Float32Array(mapped.byteLength / 4);
  data.set(new Float32Array(mapped));
  readbackBuffer.unmap();

  paramsBuffer.destroy();
  outputBuffer.destroy();
  readbackBuffer.destroy();

  if (logger) {
    logger.info(
      `Analytic sky generated via WebGPU (${settings.width}x${settings.height}, turbidity ${settings.turbidity.toFixed(2)})`
    );
  }

  return {
    source: ANALYTIC_SKY_ID,
    version: `${ANALYTIC_SKY_ID}:${cacheKey}`,
    settings,
    width: settings.width,
    height: settings.height,
    data
  };
}

export const __test__ANALYTIC_SKY_ID = ANALYTIC_SKY_ID;
