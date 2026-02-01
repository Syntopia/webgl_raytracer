export const MAX_BRUTE_FORCE_TRIS = 65536;

const TRACE_VS = `#version 300 es
precision highp float;
const vec2 positions[3] = vec2[3](
  vec2(-1.0, -3.0),
  vec2(3.0, 1.0),
  vec2(-1.0, 1.0)
);
const vec2 uvs[3] = vec2[3](
  vec2(0.0, 2.0),
  vec2(2.0, 0.0),
  vec2(0.0, 0.0)
);
out vec2 vUv;
void main() {
  gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
  vUv = uvs[gl_VertexID];
}
`;

const TRACE_FS = `#version 300 es
precision highp float;
precision highp int;

in vec2 vUv;
layout(location = 0) out vec4 outColor;

uniform sampler2D uBvhTex;
uniform sampler2D uTriTex;
uniform sampler2D uTriIndexTex;
uniform sampler2D uAccumTex;
uniform vec3 uCamOrigin;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamForward;
uniform vec2 uResolution;
uniform vec2 uBvhTexSize;
uniform vec2 uTriTexSize;
uniform vec2 uTriIndexTexSize;
uniform int uFrameIndex;
uniform int uTriCount;
uniform int uUseBvh;

ivec2 texCoordFromIndex(int index, ivec2 size) {
  int x = index % size.x;
  int y = index / size.x;
  return ivec2(x, y);
}

vec4 fetchTexel(sampler2D tex, int index, ivec2 size) {
  ivec2 coord = texCoordFromIndex(index, size);
  return texelFetch(tex, coord, 0);
}

bool intersectAABB(vec3 bmin, vec3 bmax, vec3 origin, vec3 dir, float tMax) {
  float tmin = 0.0;
  float tmax = tMax;

  if (abs(dir.x) < 1e-8) {
    if (origin.x < bmin.x || origin.x > bmax.x) return false;
  } else {
    float inv = 1.0 / dir.x;
    float t1 = (bmin.x - origin.x) * inv;
    float t2 = (bmax.x - origin.x) * inv;
    float tNear = min(t1, t2);
    float tFar = max(t1, t2);
    tmin = max(tmin, tNear);
    tmax = min(tmax, tFar);
    if (tmax < tmin) return false;
  }

  if (abs(dir.y) < 1e-8) {
    if (origin.y < bmin.y || origin.y > bmax.y) return false;
  } else {
    float inv = 1.0 / dir.y;
    float t1 = (bmin.y - origin.y) * inv;
    float t2 = (bmax.y - origin.y) * inv;
    float tNear = min(t1, t2);
    float tFar = max(t1, t2);
    tmin = max(tmin, tNear);
    tmax = min(tmax, tFar);
    if (tmax < tmin) return false;
  }

  if (abs(dir.z) < 1e-8) {
    if (origin.z < bmin.z || origin.z > bmax.z) return false;
  } else {
    float inv = 1.0 / dir.z;
    float t1 = (bmin.z - origin.z) * inv;
    float t2 = (bmax.z - origin.z) * inv;
    float tNear = min(t1, t2);
    float tFar = max(t1, t2);
    tmin = max(tmin, tNear);
    tmax = min(tmax, tFar);
    if (tmax < tmin) return false;
  }

  return tmax >= max(tmin, 0.0);
}

vec4 intersectTri(vec3 origin, vec3 dir, vec3 v0, vec3 v1, vec3 v2) {
  vec3 e1 = v1 - v0;
  vec3 e2 = v2 - v0;
  vec3 p = cross(dir, e2);
  float det = dot(e1, p);
  if (abs(det) < 1e-6) {
    return vec4(-1.0);
  }
  float invDet = 1.0 / det;
  vec3 tvec = origin - v0;
  float u = dot(tvec, p) * invDet;
  vec3 q = cross(tvec, e1);
  float v = dot(dir, q) * invDet;
  if (u < 0.0 || v < 0.0 || u + v > 1.0) {
    return vec4(-1.0);
  }
  float t = dot(e2, q) * invDet;
  if (t <= 0.0) {
    return vec4(-1.0);
  }
  vec3 n = normalize(cross(e1, e2));
  if (det < 0.0) {
    n = -n;
  }
  return vec4(t, n);
}

void main() {
  vec2 pixel = gl_FragCoord.xy;
  vec2 uv = (pixel + vec2(0.5)) / uResolution * 2.0 - 1.0;
  vec3 dir = normalize(uCamForward + uv.x * uCamRight + uv.y * uCamUp);
  vec3 origin = uCamOrigin;
  float closest = 1e20;
  vec3 hitNormal = vec3(0.0);

  if (uUseBvh == 0) {
    for (int i = 0; i < ${MAX_BRUTE_FORCE_TRIS}; i += 1) {
      if (i >= uTriCount) {
        break;
      }
      int triBase = i * 3;
      vec3 v0 = fetchTexel(uTriTex, triBase + 0, ivec2(uTriTexSize)).xyz;
      vec3 v1 = fetchTexel(uTriTex, triBase + 1, ivec2(uTriTexSize)).xyz;
      vec3 v2 = fetchTexel(uTriTex, triBase + 2, ivec2(uTriTexSize)).xyz;
      vec4 hit = intersectTri(origin, dir, v0, v1, v2);
      if (hit.x > 0.0 && hit.x < closest) {
        closest = hit.x;
        hitNormal = hit.yzw;
      }
    }
  } else {
    int stack[128];
    int stackPtr = 0;
    stack[stackPtr] = 0;
    stackPtr += 1;

    for (int step = 0; step < 1024; step += 1) {
      if (stackPtr == 0) {
        break;
      }
      stackPtr -= 1;
      int nodeIndex = stack[stackPtr];
      int baseIndex = nodeIndex * 3;

      vec4 t0 = fetchTexel(uBvhTex, baseIndex + 0, ivec2(uBvhTexSize));
      vec4 t1 = fetchTexel(uBvhTex, baseIndex + 1, ivec2(uBvhTexSize));
      vec4 t2 = fetchTexel(uBvhTex, baseIndex + 2, ivec2(uBvhTexSize));

      vec3 bmin = t0.xyz;
      float leftFirst = t0.w;
      vec3 bmax = t1.xyz;
      float primCount = t1.w;
      float rightChild = t2.x;

      if (!intersectAABB(bmin, bmax, origin, dir, closest)) {
        continue;
      }

      if (primCount > 0.5) {
        int first = int(leftFirst + 0.5);
        int count = int(primCount + 0.5);
        for (int i = 0; i < 64; i += 1) {
          if (i >= count) {
            break;
          }
          int triListIndex = first + i;
          int triIndex = int(fetchTexel(uTriIndexTex, triListIndex, ivec2(uTriIndexTexSize)).x + 0.5);
          int triBase = triIndex * 3;
          vec3 v0 = fetchTexel(uTriTex, triBase + 0, ivec2(uTriTexSize)).xyz;
          vec3 v1 = fetchTexel(uTriTex, triBase + 1, ivec2(uTriTexSize)).xyz;
          vec3 v2 = fetchTexel(uTriTex, triBase + 2, ivec2(uTriTexSize)).xyz;
          vec4 hit = intersectTri(origin, dir, v0, v1, v2);
          if (hit.x > 0.0 && hit.x < closest) {
            closest = hit.x;
            hitNormal = hit.yzw;
          }
        }
      } else {
        int left = int(leftFirst + 0.5);
        int right = int(rightChild + 0.5);
        if (stackPtr < 127) {
          stack[stackPtr] = right;
          stackPtr += 1;
        }
        if (stackPtr < 127) {
          stack[stackPtr] = left;
          stackPtr += 1;
        }
      }
    }
  }

  vec3 color = vec3(0.95);
  if (closest < 1e19) {
    color = 0.5 * (hitNormal + vec3(1.0));
  }

  vec4 prev = texelFetch(uAccumTex, ivec2(gl_FragCoord.xy), 0);
  if (uFrameIndex == 0) {
    outColor = vec4(color, 1.0);
  } else {
    float fi = float(uFrameIndex);
    vec3 accum = (prev.rgb * fi + color) / (fi + 1.0);
    outColor = vec4(accum, 1.0);
  }
}
`;

const DISPLAY_VS = TRACE_VS;

const DISPLAY_FS = `#version 300 es
precision highp float;

in vec2 vUv;
layout(location = 0) out vec4 outColor;

uniform sampler2D uDisplayTex;
uniform vec2 uDisplayResolution;

void main() {
  vec2 uv = gl_FragCoord.xy / uDisplayResolution;
  vec3 color = texture(uDisplayTex, uv).rgb;
  vec3 mapped = color / (vec3(1.0) + color);
  outColor = vec4(mapped, 1.0);
}
`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(info || "Shader compilation failed");
  }
  return shader;
}

function createProgram(gl, vsSource, fsSource) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(info || "Program link failed");
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return program;
}

function createFloatTexture(gl, width, height, data = null) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

function createFramebuffer(gl, tex) {
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error(`Framebuffer incomplete: ${status}`);
  }
  return fb;
}

export function initWebGL(canvas, logger) {
  const gl = canvas.getContext("webgl2");
  if (!gl) {
    throw new Error("WebGL2 is not available in this browser.");
  }
  const ext = gl.getExtension("EXT_color_buffer_float");
  if (!ext) {
    throw new Error("EXT_color_buffer_float is required for float accumulation.");
  }
  logger.info("WebGL2 context created");

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  const traceProgram = createProgram(gl, TRACE_VS, TRACE_FS);
  const displayProgram = createProgram(gl, DISPLAY_VS, DISPLAY_FS);

  return { gl, traceProgram, displayProgram, vao };
}

export function createDataTexture(gl, width, height, data) {
  return createFloatTexture(gl, width, height, data);
}

export function createAccumTargets(gl, width, height) {
  const texA = createFloatTexture(gl, width, height, null);
  const texB = createFloatTexture(gl, width, height, null);
  const fbA = createFramebuffer(gl, texA);
  const fbB = createFramebuffer(gl, texB);
  return {
    textures: [texA, texB],
    framebuffers: [fbA, fbB],
    width,
    height
  };
}

export function resizeAccumTargets(gl, targets, width, height) {
  if (targets.width === width && targets.height === height) {
    return targets;
  }
  targets.textures.forEach((tex) => gl.deleteTexture(tex));
  targets.framebuffers.forEach((fb) => gl.deleteFramebuffer(fb));
  return createAccumTargets(gl, width, height);
}

export function createTextureUnit(gl, texture, unit) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
}

export function setTraceUniforms(gl, program, uniforms) {
  gl.useProgram(program);
  gl.uniform1i(gl.getUniformLocation(program, "uBvhTex"), uniforms.bvhUnit);
  gl.uniform1i(gl.getUniformLocation(program, "uTriTex"), uniforms.triUnit);
  gl.uniform1i(gl.getUniformLocation(program, "uTriIndexTex"), uniforms.triIndexUnit);
  gl.uniform1i(gl.getUniformLocation(program, "uAccumTex"), uniforms.accumUnit);
  gl.uniform3fv(gl.getUniformLocation(program, "uCamOrigin"), uniforms.camOrigin);
  gl.uniform3fv(gl.getUniformLocation(program, "uCamRight"), uniforms.camRight);
  gl.uniform3fv(gl.getUniformLocation(program, "uCamUp"), uniforms.camUp);
  gl.uniform3fv(gl.getUniformLocation(program, "uCamForward"), uniforms.camForward);
  gl.uniform2fv(gl.getUniformLocation(program, "uResolution"), uniforms.resolution);
  gl.uniform2fv(gl.getUniformLocation(program, "uBvhTexSize"), uniforms.bvhTexSize);
  gl.uniform2fv(gl.getUniformLocation(program, "uTriTexSize"), uniforms.triTexSize);
  gl.uniform2fv(gl.getUniformLocation(program, "uTriIndexTexSize"), uniforms.triIndexTexSize);
  gl.uniform1i(gl.getUniformLocation(program, "uFrameIndex"), uniforms.frameIndex);
  gl.uniform1i(gl.getUniformLocation(program, "uTriCount"), uniforms.triCount);
  gl.uniform1i(gl.getUniformLocation(program, "uUseBvh"), uniforms.useBvh);
}

export function setDisplayUniforms(gl, program, uniforms) {
  gl.useProgram(program);
  gl.uniform1i(gl.getUniformLocation(program, "uDisplayTex"), uniforms.displayUnit);
  gl.uniform2fv(gl.getUniformLocation(program, "uDisplayResolution"), uniforms.displayResolution);
}

export function drawFullscreen(gl) {
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}
