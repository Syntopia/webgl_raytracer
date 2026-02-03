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
uniform sampler2D uTriNormalTex;
uniform sampler2D uTriColorTex;
uniform sampler2D uPrimIndexTex;
uniform sampler2D uSphereTex;
uniform sampler2D uSphereColorTex;
uniform sampler2D uCylinderTex;
uniform sampler2D uCylinderColorTex;
uniform sampler2D uAccumTex;
uniform sampler2D uEnvTex;
uniform vec3 uCamOrigin;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamForward;
uniform vec2 uResolution;
uniform vec2 uBvhTexSize;
uniform vec2 uTriTexSize;
uniform vec2 uTriNormalTexSize;
uniform vec2 uTriColorTexSize;
uniform vec2 uPrimIndexTexSize;
uniform vec2 uSphereTexSize;
uniform vec2 uCylinderTexSize;
uniform int uFrameIndex;
uniform int uTriCount;
uniform int uSphereCount;
uniform int uCylinderCount;

// Primitive type constants
const int PRIM_TRIANGLE = 0;
const int PRIM_SPHERE = 1;
const int PRIM_CYLINDER = 2;
uniform int uUseBvh;
uniform int uUseGltfColor;
uniform vec3 uBaseColor;
uniform float uMetallic;
uniform float uRoughness;
uniform int uMaterialMode;
uniform float uMatteSpecular;
uniform float uMatteRoughness;
uniform float uMatteDiffuseRoughness;
uniform float uWrapDiffuse;
uniform float uRimBoost;
uniform int uMaxBounces;
uniform float uExposure;
uniform float uAmbientIntensity;
uniform vec3 uAmbientColor;
uniform int uSamplesPerBounce;
uniform int uCastShadows;
uniform float uRayBias;
uniform float uTMin;
uniform float uEnvIntensity;
uniform int uUseEnv;
uniform float uEnvMaxLuminance;
uniform sampler2D uEnvMarginalCdf;
uniform sampler2D uEnvConditionalCdf;
uniform vec2 uEnvSize;
uniform int uLightEnabled[3];
uniform vec3 uLightDir[3];
uniform vec3 uLightColor[3];
uniform float uLightIntensity[3];
uniform float uLightAngle[3];

// Visualization modes: 0=normal, 1=normals, 2=BVH traversal cost, 3=BVH depth
uniform int uVisMode;

const float PI = 3.14159265359;

float powerHeuristic(float pdfA, float pdfB);
float brdfPdf(vec3 N, vec3 V, vec3 L, float roughness, float specProb);

ivec2 texCoordFromIndex(int index, ivec2 size) {
  int x = index % size.x;
  int y = index / size.x;
  return ivec2(x, y);
}

vec4 fetchTexel(sampler2D tex, int index, ivec2 size) {
  ivec2 coord = texCoordFromIndex(index, size);
  return texelFetch(tex, coord, 0);
}

vec3 fetchTriNormal(int triIndex, vec3 bary) {
  int base = triIndex * 3;
  vec3 n0 = fetchTexel(uTriNormalTex, base + 0, ivec2(uTriNormalTexSize)).xyz;
  vec3 n1 = fetchTexel(uTriNormalTex, base + 1, ivec2(uTriNormalTexSize)).xyz;
  vec3 n2 = fetchTexel(uTriNormalTex, base + 2, ivec2(uTriNormalTexSize)).xyz;
  vec3 n = n0 * bary.x + n1 * bary.y + n2 * bary.z;
  return normalize(n);
}

vec3 fetchTriColor(int triIndex) {
  return fetchTexel(uTriColorTex, triIndex, ivec2(uTriColorTexSize)).rgb;
}

void fetchTriVerts(int triIndex, out vec3 v0, out vec3 v1, out vec3 v2) {
  int base = triIndex * 3;
  v0 = fetchTexel(uTriTex, base + 0, ivec2(uTriTexSize)).xyz;
  v1 = fetchTexel(uTriTex, base + 1, ivec2(uTriTexSize)).xyz;
  v2 = fetchTexel(uTriTex, base + 2, ivec2(uTriTexSize)).xyz;
}

float maxComponent(vec3 v) {
  return max(v.x, max(v.y, v.z));
}

float luminance(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float wrapNdotL(float ndotl, float wrap) {
  return clamp((ndotl + wrap) / (1.0 + wrap), 0.0, 1.0);
}

vec3 orenNayarDiffuse(vec3 N, vec3 V, vec3 L, vec3 baseColor, float sigma) {
  float NdotL = max(dot(N, L), 0.0);
  float NdotV = max(dot(N, V), 0.0);
  if (NdotL <= 0.0 || NdotV <= 0.0) {
    return vec3(0.0);
  }
  float sigma2 = sigma * sigma;
  float A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33));
  float B = 0.45 * (sigma2 / (sigma2 + 0.09));

  float sinThetaL = sqrt(max(0.0, 1.0 - NdotL * NdotL));
  float sinThetaV = sqrt(max(0.0, 1.0 - NdotV * NdotV));
  float tanThetaL = sinThetaL / max(NdotL, 1e-4);
  float tanThetaV = sinThetaV / max(NdotV, 1e-4);
  float sinAlpha = max(sinThetaL, sinThetaV);
  float tanBeta = min(tanThetaL, tanThetaV);

  vec3 Lp = normalize(L - N * NdotL);
  vec3 Vp = normalize(V - N * NdotV);
  float cosPhi = max(0.0, dot(Lp, Vp));

  float oren = A + B * cosPhi * sinAlpha * tanBeta;
  return baseColor * oren / PI;
}

vec3 evalDiffuseBrdf(vec3 N, vec3 V, vec3 L, vec3 baseColor, float diffRough, float wrap) {
  vec3 brdf = diffRough > 1e-4 ? orenNayarDiffuse(N, V, L, baseColor, diffRough) : (baseColor / PI);
  if (wrap > 0.0) {
    float ndotl = max(dot(N, L), 0.0);
    float ndotlWrap = wrapNdotL(ndotl, wrap);
    float scale = ndotl > 1e-4 ? (ndotlWrap / ndotl) : 0.0;
    brdf *= scale;
  }
  return brdf;
}

vec3 sampleEnv(vec3 dir) {
  if (uUseEnv == 0) {
    return vec3(0.0);
  }
  vec3 d = normalize(dir);
  float u = atan(d.z, d.x) / (2.0 * PI) + 0.5;
  float v = acos(clamp(d.y, -1.0, 1.0)) / PI;
  vec3 color = texture(uEnvTex, vec2(u, v)).rgb * uEnvIntensity;

  // Soft clamp to reduce fireflies from extremely bright light sources
  // Compresses luminance above threshold while preserving color ratios
  if (uEnvMaxLuminance > 0.0) {
    float lum = luminance(color);
    if (lum > uEnvMaxLuminance) {
      // Soft knee: asymptotically approaches 2x max luminance
      float excess = lum - uEnvMaxLuminance;
      float compressed = uEnvMaxLuminance + excess / (1.0 + excess / uEnvMaxLuminance);
      color *= compressed / lum;
    }
  }
  return color;
}

// Convert direction to environment map UV coordinates
vec2 dirToEnvUv(vec3 dir) {
  vec3 d = normalize(dir);
  float u = atan(d.z, d.x) / (2.0 * PI) + 0.5;
  float v = acos(clamp(d.y, -1.0, 1.0)) / PI;
  return vec2(u, v);
}

// Binary search to find index where CDF >= xi
// Returns the index of the interval containing xi (0 to size-2)
float binarySearchCdf(sampler2D cdfTex, float y, int size, float xi) {
  int lo = 0;
  int hi = size - 1;

  // Binary search for the bucket containing xi
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    float cdfVal = texelFetch(cdfTex, ivec2(mid, int(y)), 0).r;
    if (cdfVal <= xi) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  // Clamp to valid interval index
  int idx = max(lo - 1, 0);

  // Linear interpolation within the bucket
  float cdfLo = texelFetch(cdfTex, ivec2(idx, int(y)), 0).r;
  float cdfHi = texelFetch(cdfTex, ivec2(idx + 1, int(y)), 0).r;
  float t = (cdfHi > cdfLo) ? clamp((xi - cdfLo) / (cdfHi - cdfLo), 0.0, 1.0) : 0.0;

  return float(idx) + t;
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
  if (t <= uTMin) {
    return vec4(-1.0);
  }
  return vec4(t, u, v, 1.0);
}

// Fetch sphere data: returns vec4(center.xyz, radius)
vec4 fetchSphere(int sphereIndex) {
  return fetchTexel(uSphereTex, sphereIndex, ivec2(uSphereTexSize));
}

vec3 fetchSphereColor(int sphereIndex) {
  return fetchTexel(uSphereColorTex, sphereIndex, ivec2(uSphereTexSize)).rgb;
}

// Fetch cylinder data: p1 + radius in texel 0, p2 in texel 1
void fetchCylinder(int cylIndex, out vec3 p1, out vec3 p2, out float radius) {
  int base = cylIndex * 2;
  vec4 t0 = fetchTexel(uCylinderTex, base + 0, ivec2(uCylinderTexSize));
  vec4 t1 = fetchTexel(uCylinderTex, base + 1, ivec2(uCylinderTexSize));
  p1 = t0.xyz;
  radius = t0.w;
  p2 = t1.xyz;
}

vec3 fetchCylinderColor(int cylIndex) {
  return fetchTexel(uCylinderColorTex, cylIndex, ivec2(uCylinderTexSize)).rgb;
}

// Sphere intersection: returns vec2(t, 0) or vec2(-1, 0) if no hit
// Normal can be computed as (hitPos - center) / radius
vec2 intersectSphere(vec3 origin, vec3 dir, vec3 center, float radius) {
  vec3 oc = origin - center;
  float b = dot(oc, dir);
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - c;
  if (discriminant < 0.0) {
    return vec2(-1.0, 0.0);
  }
  float sqrtD = sqrt(discriminant);
  float t = -b - sqrtD;
  if (t <= uTMin) {
    t = -b + sqrtD;
    if (t <= uTMin) {
      return vec2(-1.0, 0.0);
    }
  }
  return vec2(t, 0.0);
}

// Cylinder intersection (finite cylinder with hemispherical caps)
// Returns vec2(t, hitType) where hitType: 0=side, 1=cap1, 2=cap2
vec2 intersectCylinder(vec3 origin, vec3 dir, vec3 p1, vec3 p2, float radius) {
  vec3 axis = p2 - p1;
  float height = length(axis);
  if (height < 1e-6) {
    // Degenerate cylinder, treat as sphere
    return intersectSphere(origin, dir, p1, radius);
  }
  axis /= height;

  // Transform to cylinder local space (p1 at origin, axis along Y)
  vec3 oc = origin - p1;

  // Project ray onto plane perpendicular to axis
  float dirDotAxis = dot(dir, axis);
  float ocDotAxis = dot(oc, axis);

  vec3 dirPerp = dir - axis * dirDotAxis;
  vec3 ocPerp = oc - axis * ocDotAxis;

  float a = dot(dirPerp, dirPerp);
  float b = 2.0 * dot(dirPerp, ocPerp);
  float c = dot(ocPerp, ocPerp) - radius * radius;

  float bestT = -1.0;
  float hitType = 0.0;

  // Check infinite cylinder
  if (a > 1e-8) {
    float discriminant = b * b - 4.0 * a * c;
    if (discriminant >= 0.0) {
      float sqrtD = sqrt(discriminant);
      float t1 = (-b - sqrtD) / (2.0 * a);
      float t2 = (-b + sqrtD) / (2.0 * a);

      // Check t1
      if (t1 > uTMin) {
        float h = ocDotAxis + t1 * dirDotAxis;
        if (h >= 0.0 && h <= height) {
          bestT = t1;
          hitType = 0.0;
        }
      }

      // Check t2 if t1 didn't work
      if (bestT < 0.0 && t2 > uTMin) {
        float h = ocDotAxis + t2 * dirDotAxis;
        if (h >= 0.0 && h <= height) {
          bestT = t2;
          hitType = 0.0;
        }
      }
    }
  }

  // Check hemispherical caps
  // Cap 1 at p1
  vec2 capHit1 = intersectSphere(origin, dir, p1, radius);
  if (capHit1.x > uTMin && (bestT < 0.0 || capHit1.x < bestT)) {
    vec3 hitPos = origin + dir * capHit1.x;
    float h = dot(hitPos - p1, axis);
    if (h <= 0.0) {
      bestT = capHit1.x;
      hitType = 1.0;
    }
  }

  // Cap 2 at p2
  vec2 capHit2 = intersectSphere(origin, dir, p2, radius);
  if (capHit2.x > uTMin && (bestT < 0.0 || capHit2.x < bestT)) {
    vec3 hitPos = origin + dir * capHit2.x;
    float h = dot(hitPos - p2, axis);
    if (h >= 0.0) {
      bestT = capHit2.x;
      hitType = 2.0;
    }
  }

  return vec2(bestT, hitType);
}

// Compute cylinder normal at hit point
vec3 cylinderNormal(vec3 hitPos, vec3 p1, vec3 p2, float radius, float hitType) {
  vec3 axis = normalize(p2 - p1);
  if (hitType == 1.0) {
    // Cap 1 (hemisphere at p1)
    return normalize(hitPos - p1);
  } else if (hitType == 2.0) {
    // Cap 2 (hemisphere at p2)
    return normalize(hitPos - p2);
  } else {
    // Side surface
    float h = dot(hitPos - p1, axis);
    vec3 pointOnAxis = p1 + axis * h;
    return normalize(hitPos - pointOnAxis);
  }
}

// Decode primitive index: upper 2 bits = type, lower 30 bits = index
void decodePrimIndex(float encoded, out int primType, out int primIndex) {
  // Use floatBitsToInt to preserve exact bit pattern from the texture
  int bits = floatBitsToInt(encoded);
  primType = (bits >> 30) & 0x3;
  primIndex = bits & 0x3FFFFFFF;
}

// Hit info structure encoded in vec4: (primType, primIndex, extra1, extra2)
// For triangles: extra = bary.y, bary.z (bary.x = 1 - y - z)
// For spheres: extra = 0, 0
// For cylinders: extra = hitType, 0

bool traceClosest(vec3 origin, vec3 dir, out float outT, out int outPrimType, out int outPrimIndex, out vec3 outExtra, out int outTraversalCost) {
  float closest = 1e20;
  int hitPrimType = -1;
  int hitPrimIndex = -1;
  vec3 hitExtra = vec3(0.0);
  int traversalCost = 0;

  if (uUseBvh == 0) {
    // Brute force: test all triangles, spheres, cylinders
    for (int i = 0; i < ${MAX_BRUTE_FORCE_TRIS}; i += 1) {
      if (i >= uTriCount) break;
      traversalCost += 1;
      int triBase = i * 3;
      vec3 v0 = fetchTexel(uTriTex, triBase + 0, ivec2(uTriTexSize)).xyz;
      vec3 v1 = fetchTexel(uTriTex, triBase + 1, ivec2(uTriTexSize)).xyz;
      vec3 v2 = fetchTexel(uTriTex, triBase + 2, ivec2(uTriTexSize)).xyz;
      vec4 hit = intersectTri(origin, dir, v0, v1, v2);
      if (hit.x > 0.0 && hit.x < closest) {
        closest = hit.x;
        hitPrimType = PRIM_TRIANGLE;
        hitPrimIndex = i;
        hitExtra = vec3(hit.y, hit.z, 0.0);
      }
    }
    for (int i = 0; i < 1024; i += 1) {
      if (i >= uSphereCount) break;
      traversalCost += 1;
      vec4 s = fetchSphere(i);
      vec2 hit = intersectSphere(origin, dir, s.xyz, s.w);
      if (hit.x > 0.0 && hit.x < closest) {
        closest = hit.x;
        hitPrimType = PRIM_SPHERE;
        hitPrimIndex = i;
        hitExtra = vec3(0.0);
      }
    }
    for (int i = 0; i < 1024; i += 1) {
      if (i >= uCylinderCount) break;
      traversalCost += 1;
      vec3 p1, p2; float radius;
      fetchCylinder(i, p1, p2, radius);
      vec2 hit = intersectCylinder(origin, dir, p1, p2, radius);
      if (hit.x > 0.0 && hit.x < closest) {
        closest = hit.x;
        hitPrimType = PRIM_CYLINDER;
        hitPrimIndex = i;
        hitExtra = vec3(hit.y, 0.0, 0.0);
      }
    }
  } else {
    int stack[128];
    int stackPtr = 0;
    stack[stackPtr] = 0;
    stackPtr += 1;

    for (int step = 0; step < 2048; step += 1) {
      if (stackPtr == 0) break;
      stackPtr -= 1;
      int nodeIndex = stack[stackPtr];
      int baseIndex = nodeIndex * 3;
      traversalCost += 1; // Count node visits

      vec4 t0 = fetchTexel(uBvhTex, baseIndex + 0, ivec2(uBvhTexSize));
      vec4 t1 = fetchTexel(uBvhTex, baseIndex + 1, ivec2(uBvhTexSize));
      vec4 t2 = fetchTexel(uBvhTex, baseIndex + 2, ivec2(uBvhTexSize));

      vec3 bmin = t0.xyz;
      float leftFirst = t0.w;
      vec3 bmax = t1.xyz;
      float primCount = t1.w;
      float rightChild = t2.x;

      if (!intersectAABB(bmin, bmax, origin, dir, closest)) continue;

      if (primCount > 0.5) {
        int first = int(leftFirst + 0.5);
        int count = int(primCount + 0.5);
        for (int i = 0; i < 64; i += 1) {
          if (i >= count) break;
          traversalCost += 1; // Count primitive tests
          int primListIndex = first + i;
          float encoded = fetchTexel(uPrimIndexTex, primListIndex, ivec2(uPrimIndexTexSize)).x;
          int primType, primIndex;
          decodePrimIndex(encoded, primType, primIndex);

          if (primType == PRIM_TRIANGLE) {
            int triBase = primIndex * 3;
            vec3 v0 = fetchTexel(uTriTex, triBase + 0, ivec2(uTriTexSize)).xyz;
            vec3 v1 = fetchTexel(uTriTex, triBase + 1, ivec2(uTriTexSize)).xyz;
            vec3 v2 = fetchTexel(uTriTex, triBase + 2, ivec2(uTriTexSize)).xyz;
            vec4 hit = intersectTri(origin, dir, v0, v1, v2);
            if (hit.x > 0.0 && hit.x < closest) {
              closest = hit.x;
              hitPrimType = PRIM_TRIANGLE;
              hitPrimIndex = primIndex;
              hitExtra = vec3(hit.y, hit.z, 0.0);
            }
          } else if (primType == PRIM_SPHERE) {
            vec4 s = fetchSphere(primIndex);
            vec2 hit = intersectSphere(origin, dir, s.xyz, s.w);
            if (hit.x > 0.0 && hit.x < closest) {
              closest = hit.x;
              hitPrimType = PRIM_SPHERE;
              hitPrimIndex = primIndex;
              hitExtra = vec3(0.0);
            }
          } else if (primType == PRIM_CYLINDER) {
            vec3 p1, p2; float radius;
            fetchCylinder(primIndex, p1, p2, radius);
            vec2 hit = intersectCylinder(origin, dir, p1, p2, radius);
            if (hit.x > 0.0 && hit.x < closest) {
              closest = hit.x;
              hitPrimType = PRIM_CYLINDER;
              hitPrimIndex = primIndex;
              hitExtra = vec3(hit.y, 0.0, 0.0);
            }
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

  outT = closest;
  outPrimType = hitPrimType;
  outPrimIndex = hitPrimIndex;
  outExtra = hitExtra;
  outTraversalCost = traversalCost;
  return hitPrimType >= 0;
}

bool traceAny(vec3 origin, vec3 dir, float tMax) {
  if (uUseBvh == 0) {
    // Brute force: test all primitives
    for (int i = 0; i < ${MAX_BRUTE_FORCE_TRIS}; i += 1) {
      if (i >= uTriCount) break;
      int triBase = i * 3;
      vec3 v0 = fetchTexel(uTriTex, triBase + 0, ivec2(uTriTexSize)).xyz;
      vec3 v1 = fetchTexel(uTriTex, triBase + 1, ivec2(uTriTexSize)).xyz;
      vec3 v2 = fetchTexel(uTriTex, triBase + 2, ivec2(uTriTexSize)).xyz;
      vec4 hit = intersectTri(origin, dir, v0, v1, v2);
      if (hit.x > 0.0 && hit.x < tMax) return true;
    }
    for (int i = 0; i < 1024; i += 1) {
      if (i >= uSphereCount) break;
      vec4 s = fetchSphere(i);
      vec2 hit = intersectSphere(origin, dir, s.xyz, s.w);
      if (hit.x > 0.0 && hit.x < tMax) return true;
    }
    for (int i = 0; i < 1024; i += 1) {
      if (i >= uCylinderCount) break;
      vec3 p1, p2; float radius;
      fetchCylinder(i, p1, p2, radius);
      vec2 hit = intersectCylinder(origin, dir, p1, p2, radius);
      if (hit.x > 0.0 && hit.x < tMax) return true;
    }
    return false;
  }

  int stack[128];
  int stackPtr = 0;
  stack[stackPtr] = 0;
  stackPtr += 1;

  for (int step = 0; step < 2048; step += 1) {
    if (stackPtr == 0) break;
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

    if (!intersectAABB(bmin, bmax, origin, dir, tMax)) {
      continue;
    }

    if (primCount > 0.5) {
      int first = int(leftFirst + 0.5);
      int count = int(primCount + 0.5);
      for (int i = 0; i < 64; i += 1) {
        if (i >= count) break;
        int primListIndex = first + i;
        float encoded = fetchTexel(uPrimIndexTex, primListIndex, ivec2(uPrimIndexTexSize)).x;
        int primType, primIndex;
        decodePrimIndex(encoded, primType, primIndex);

        float hitT = -1.0;
        if (primType == PRIM_TRIANGLE) {
          int triBase = primIndex * 3;
          vec3 v0 = fetchTexel(uTriTex, triBase + 0, ivec2(uTriTexSize)).xyz;
          vec3 v1 = fetchTexel(uTriTex, triBase + 1, ivec2(uTriTexSize)).xyz;
          vec3 v2 = fetchTexel(uTriTex, triBase + 2, ivec2(uTriTexSize)).xyz;
          hitT = intersectTri(origin, dir, v0, v1, v2).x;
        } else if (primType == PRIM_SPHERE) {
          vec4 s = fetchSphere(primIndex);
          hitT = intersectSphere(origin, dir, s.xyz, s.w).x;
        } else if (primType == PRIM_CYLINDER) {
          vec3 p1, p2; float radius;
          fetchCylinder(primIndex, p1, p2, radius);
          hitT = intersectCylinder(origin, dir, p1, p2, radius).x;
        }
        if (hitT > 0.0 && hitT < tMax) return true;
      }
    } else {
      int left = int(leftFirst + 0.5);
      int right = int(rightChild + 0.5);
      if (stackPtr < 127) { stack[stackPtr] = right; stackPtr += 1; }
      if (stackPtr < 127) { stack[stackPtr] = left; stackPtr += 1; }
    }
  }
  return false;
}

bool traceAnyMin(vec3 origin, vec3 dir, float tMax, float tMin) {
  if (uUseBvh == 0) {
    for (int i = 0; i < ${MAX_BRUTE_FORCE_TRIS}; i += 1) {
      if (i >= uTriCount) break;
      int triBase = i * 3;
      vec3 v0 = fetchTexel(uTriTex, triBase + 0, ivec2(uTriTexSize)).xyz;
      vec3 v1 = fetchTexel(uTriTex, triBase + 1, ivec2(uTriTexSize)).xyz;
      vec3 v2 = fetchTexel(uTriTex, triBase + 2, ivec2(uTriTexSize)).xyz;
      vec4 hit = intersectTri(origin, dir, v0, v1, v2);
      if (hit.x > tMin && hit.x < tMax) return true;
    }
    for (int i = 0; i < 1024; i += 1) {
      if (i >= uSphereCount) break;
      vec4 s = fetchSphere(i);
      vec2 hit = intersectSphere(origin, dir, s.xyz, s.w);
      if (hit.x > tMin && hit.x < tMax) return true;
    }
    for (int i = 0; i < 1024; i += 1) {
      if (i >= uCylinderCount) break;
      vec3 p1, p2; float radius;
      fetchCylinder(i, p1, p2, radius);
      vec2 hit = intersectCylinder(origin, dir, p1, p2, radius);
      if (hit.x > tMin && hit.x < tMax) return true;
    }
    return false;
  }

  int stack[128];
  int stackPtr = 0;
  stack[stackPtr] = 0;
  stackPtr += 1;

  for (int step = 0; step < 2048; step += 1) {
    if (stackPtr == 0) break;
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

    if (!intersectAABB(bmin, bmax, origin, dir, tMax)) continue;

    if (primCount > 0.5) {
      int first = int(leftFirst + 0.5);
      int count = int(primCount + 0.5);
      for (int i = 0; i < 64; i += 1) {
        if (i >= count) break;
        int primListIndex = first + i;
        float encoded = fetchTexel(uPrimIndexTex, primListIndex, ivec2(uPrimIndexTexSize)).x;
        int primType, primIndex;
        decodePrimIndex(encoded, primType, primIndex);

        float hitT = -1.0;
        if (primType == PRIM_TRIANGLE) {
          int triBase = primIndex * 3;
          vec3 v0 = fetchTexel(uTriTex, triBase + 0, ivec2(uTriTexSize)).xyz;
          vec3 v1 = fetchTexel(uTriTex, triBase + 1, ivec2(uTriTexSize)).xyz;
          vec3 v2 = fetchTexel(uTriTex, triBase + 2, ivec2(uTriTexSize)).xyz;
          hitT = intersectTri(origin, dir, v0, v1, v2).x;
        } else if (primType == PRIM_SPHERE) {
          vec4 s = fetchSphere(primIndex);
          hitT = intersectSphere(origin, dir, s.xyz, s.w).x;
        } else if (primType == PRIM_CYLINDER) {
          vec3 p1, p2; float radius;
          fetchCylinder(primIndex, p1, p2, radius);
          hitT = intersectCylinder(origin, dir, p1, p2, radius).x;
        }
        if (hitT > tMin && hitT < tMax) {
          return true;
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
  return false;
}

// PCG-style hash for better seed initialization
uint pcgHash(uint v) {
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

uint initSeed() {
  uint sx = uint(gl_FragCoord.x);
  uint sy = uint(gl_FragCoord.y);
  uint seed = sx + sy * 65536u + uint(uFrameIndex) * 15485863u;
  return pcgHash(seed);
}

float rand(inout uint state) {
  state = state * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return float((word >> 22u) ^ word) / 4294967295.0;
}

// Sample direction from environment map using importance sampling
// Returns sampled direction and PDF
vec3 sampleEnvDirection(inout uint seed, out float pdf) {
  if (uUseEnv == 0 || uEnvSize.x < 1.0) {
    // Fallback to uniform sphere sampling
    pdf = 1.0 / (4.0 * PI);
    float r1 = rand(seed);
    float r2 = rand(seed);
    float phi = 2.0 * PI * r1;
    float cosTheta = 1.0 - 2.0 * r2;
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    return vec3(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);
  }

  float r1 = rand(seed);
  float r2 = rand(seed);

  int height = int(uEnvSize.y);
  int width = int(uEnvSize.x);

  // Sample row (v) from marginal CDF
  float vFloat = binarySearchCdf(uEnvMarginalCdf, 0.0, height + 1, r1);
  int vIdx = clamp(int(vFloat), 0, height - 1);
  float v = (vFloat + 0.5) / float(height);

  // Sample column (u) from conditional CDF for this row
  float uFloat = binarySearchCdf(uEnvConditionalCdf, float(vIdx), width + 1, r2);
  float u = (uFloat + 0.5) / float(width);

  // Convert UV to direction
  float theta = v * PI;
  float phi = u * 2.0 * PI - PI;
  float sinTheta = sin(theta);
  vec3 dir = vec3(sinTheta * cos(phi), cos(theta), sinTheta * sin(phi));

  // Compute PDF from the CDF differences
  float marginalPdf = texelFetch(uEnvMarginalCdf, ivec2(vIdx + 1, 0), 0).r -
                      texelFetch(uEnvMarginalCdf, ivec2(vIdx, 0), 0).r;
  float conditionalPdf = texelFetch(uEnvConditionalCdf, ivec2(int(uFloat) + 1, vIdx), 0).r -
                         texelFetch(uEnvConditionalCdf, ivec2(int(uFloat), vIdx), 0).r;

  // PDF in UV space
  float pdfUv = marginalPdf * float(height) * conditionalPdf * float(width);

  // Convert to solid angle PDF
  sinTheta = max(sinTheta, 1e-4);
  pdf = pdfUv / (2.0 * PI * PI * sinTheta);
  pdf = max(pdf, 1e-6);

  return dir;
}

// Compute PDF for sampling a given direction from the environment map
float envPdf(vec3 dir) {
  if (uUseEnv == 0 || uEnvSize.x < 1.0) {
    return 1.0 / (4.0 * PI);
  }

  vec2 uv = dirToEnvUv(dir);
  int width = int(uEnvSize.x);
  int height = int(uEnvSize.y);
  int uIdx = clamp(int(uv.x * float(width)), 0, width - 1);
  int vIdx = clamp(int(uv.y * float(height)), 0, height - 1);

  float marginalPdf = texelFetch(uEnvMarginalCdf, ivec2(vIdx + 1, 0), 0).r -
                      texelFetch(uEnvMarginalCdf, ivec2(vIdx, 0), 0).r;
  float conditionalPdf = texelFetch(uEnvConditionalCdf, ivec2(uIdx + 1, vIdx), 0).r -
                         texelFetch(uEnvConditionalCdf, ivec2(uIdx, vIdx), 0).r;

  float pdfUv = marginalPdf * float(height) * conditionalPdf * float(width);

  float theta = uv.y * PI;
  float sinTheta = max(sin(theta), 1e-4);

  return max(pdfUv / (2.0 * PI * PI * sinTheta), 1e-6);
}

vec3 cosineSampleHemisphere(vec3 n, inout uint state) {
  float r1 = rand(state);
  float r2 = rand(state);
  float phi = 2.0 * PI * r1;
  float cosTheta = sqrt(1.0 - r2);
  float sinTheta = sqrt(r2);
  vec3 local = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
  vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangent = normalize(cross(up, n));
  vec3 bitangent = cross(n, tangent);
  return normalize(tangent * local.x + bitangent * local.y + n * local.z);
}

vec3 sampleConeDirection(vec3 axis, float angle, inout uint state, out float pdf) {
  float cosMax = cos(angle);
  float r1 = rand(state);
  float r2 = rand(state);
  float cosTheta = mix(cosMax, 1.0, r1);
  float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  float phi = 2.0 * PI * r2;
  vec3 local = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
  vec3 up = abs(axis.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangent = normalize(cross(up, axis));
  vec3 bitangent = cross(axis, tangent);
  float solidAngle = max(2.0 * PI * (1.0 - cosMax), 1e-6);
  pdf = 1.0 / solidAngle;
  return normalize(tangent * local.x + bitangent * local.y + axis * local.z);
}

vec3 reflectSample(vec3 dir, vec3 n, float roughness, inout uint state) {
  vec3 r = reflect(dir, n);
  if (roughness <= 0.02) {
    return normalize(r);
  }
  float r1 = rand(state);
  float r2 = rand(state);
  float phi = 2.0 * PI * r1;
  float cosTheta = pow(1.0 - r2, 1.0 / (roughness * 4.0 + 1.0));
  float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  vec3 local = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
  vec3 up = abs(r.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangent = normalize(cross(up, r));
  vec3 bitangent = cross(r, tangent);
  return normalize(tangent * local.x + bitangent * local.y + r * local.z);
}

vec3 sampleGGXHalfVector(vec3 n, float roughness, inout uint state) {
  float a = roughness * roughness;
  float a2 = a * a;
  float r1 = rand(state);
  float r2 = rand(state);
  float phi = 2.0 * PI * r1;
  float cosTheta = sqrt((1.0 - r2) / (1.0 + (a2 - 1.0) * r2));
  float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  vec3 local = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
  vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangent = normalize(cross(up, n));
  vec3 bitangent = cross(n, tangent);
  return normalize(tangent * local.x + bitangent * local.y + n * local.z);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float distributionGGX(float NdotH, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
  return a2 / (PI * denom * denom + 1e-6);
}

float geometrySchlickGGX(float NdotV, float roughness) {
  float r = roughness + 1.0;
  float k = (r * r) / 8.0;
  return NdotV / (NdotV * (1.0 - k) + k + 0.0001);
}

float geometrySmith(float NdotV, float NdotL, float roughness) {
  float ggx1 = geometrySchlickGGX(NdotV, roughness);
  float ggx2 = geometrySchlickGGX(NdotL, roughness);
  return ggx1 * ggx2;
}

vec3 shadeDirect(vec3 hitPos, vec3 shadingNormal, vec3 geomNormal, vec3 baseColor, vec3 V, inout uint seed) {
  vec3 direct = vec3(0.0);
  float bias = max(uRayBias, 1e-4);
  float metallic = (uMaterialMode == 1) ? 0.0 : uMetallic;
  float rough = (uMaterialMode == 1) ? uMatteRoughness : uRoughness;
  float diffRough = (uMaterialMode == 1) ? uMatteDiffuseRoughness : 0.0;
  float wrap = (uMaterialMode == 1) ? uWrapDiffuse : 0.0;
  vec3 F0 = (uMaterialMode == 1) ? vec3(uMatteSpecular) : mix(vec3(0.04), baseColor, metallic);
  for (int i = 0; i < 3; i += 1) {
    if (uLightEnabled[i] == 0) {
      continue;
    }
    float angle = clamp(radians(uLightAngle[i]), 0.001, PI);
    float lightPdf;
    vec3 lightDir = sampleConeDirection(normalize(-uLightDir[i]), angle, seed, lightPdf);
    float NdotL = max(dot(shadingNormal, lightDir), 0.0);
    if (NdotL <= 0.0) {
      continue;
    }
    if (uCastShadows == 1) {
      float tmin = max(bias, uTMin);
      bool occluded = traceAnyMin(hitPos + geomNormal * bias, lightDir, 1e20, tmin);
      if (occluded) {
        continue;
      }
    }

    vec3 H = normalize(V + lightDir);
    float NdotV = max(dot(shadingNormal, V), 0.001);
    float NdotH = max(dot(shadingNormal, H), 0.001);
    float VdotH = max(dot(V, H), 0.001);
    float D = distributionGGX(NdotH, rough);
    float G = geometrySmith(NdotV, NdotL, rough);
    vec3 F = fresnelSchlick(VdotH, F0);
    vec3 specBrdf = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
    vec3 diffBrdf = evalDiffuseBrdf(shadingNormal, V, lightDir, baseColor, diffRough, wrap) * (1.0 - metallic);
    vec3 brdf = specBrdf + diffBrdf * (vec3(1.0) - F);

    float specWeight = maxComponent(F0);
    float diffWeight = (1.0 - metallic) * maxComponent(baseColor);
    float sumW = specWeight + diffWeight;
    float specProb = sumW > 0.0 ? specWeight / sumW : 0.5;
    float brdfPdfVal = brdfPdf(shadingNormal, V, lightDir, rough, specProb);
    float misWeight = powerHeuristic(lightPdf, brdfPdfVal);

    vec3 radiance = uLightColor[i] * uLightIntensity[i];
    vec3 contrib = brdf * radiance * NdotL * misWeight / max(lightPdf, 1e-6);
    direct += contrib;
  }
  return direct;
}

// Power heuristic for MIS (beta = 2)
float powerHeuristic(float pdfA, float pdfB) {
  float a2 = pdfA * pdfA;
  float b2 = pdfB * pdfB;
  return a2 / max(a2 + b2, 1e-8);
}

// Compute BRDF PDF for a given direction
float brdfPdf(vec3 N, vec3 V, vec3 L, float roughness, float specProb) {
  float NdotL = max(dot(N, L), 0.0);
  if (NdotL <= 0.0) return 0.0;

  // Diffuse PDF (cosine-weighted)
  float diffPdf = NdotL / PI;

  // Specular PDF (GGX)
  vec3 H = normalize(V + L);
  float NdotH = max(dot(N, H), 0.001);
  float VdotH = max(dot(V, H), 0.001);
  float D = distributionGGX(NdotH, roughness);
  float specPdf = D * NdotH / (4.0 * VdotH);

  // Combined PDF
  return specProb * specPdf + (1.0 - specProb) * diffPdf;
}

vec3 tracePath(vec3 origin, vec3 dir, inout uint seed) {
  vec3 radiance = vec3(0.0);
  vec3 throughput = vec3(1.0);
  float bias = max(uRayBias, 1e-4);
  float lastBrdfPdf = 0.0; // Track BRDF PDF for MIS when hitting environment

  for (int bounce = 0; bounce < 8; bounce += 1) {
    if (bounce >= uMaxBounces) {
      break;
    }
    float t;
    int primType;
    int primIndex;
    vec3 extra;
    int dummyCost;
    bool hit = traceClosest(origin, dir, t, primType, primIndex, extra, dummyCost);
    if (!hit) {
      vec3 envContrib = uAmbientColor * uAmbientIntensity + sampleEnv(dir);
      // Apply MIS weight for environment hit via BRDF sampling
      if (bounce > 0 && uUseEnv == 1 && lastBrdfPdf > 0.0) {
        float envPdfVal = envPdf(dir);
        float misWeight = powerHeuristic(lastBrdfPdf, envPdfVal);
        radiance += throughput * envContrib * misWeight;
      } else {
        // First bounce or no env: full contribution
        radiance += throughput * envContrib;
      }
      break;
    }

    vec3 hitPos = origin + dir * t;
    vec3 geomNormal;
    vec3 shadingNormal;
    vec3 baseColor;

    if (primType == PRIM_TRIANGLE) {
      vec3 bary = vec3(1.0 - extra.x - extra.y, extra.x, extra.y);
      vec3 v0, v1, v2;
      fetchTriVerts(primIndex, v0, v1, v2);
      geomNormal = normalize(cross(v1 - v0, v2 - v0));
      if (dot(geomNormal, dir) > 0.0) {
        geomNormal = -geomNormal;
      }
      shadingNormal = fetchTriNormal(primIndex, bary);
      if (dot(shadingNormal, geomNormal) < 0.0) {
        shadingNormal = -shadingNormal;
      }
      baseColor = mix(uBaseColor, fetchTriColor(primIndex), float(uUseGltfColor));
    } else if (primType == PRIM_SPHERE) {
      vec4 s = fetchSphere(primIndex);
      geomNormal = normalize(hitPos - s.xyz);
      if (dot(geomNormal, dir) > 0.0) {
        geomNormal = -geomNormal;
      }
      shadingNormal = geomNormal;
      baseColor = mix(uBaseColor, fetchSphereColor(primIndex), float(uUseGltfColor));
    } else { // PRIM_CYLINDER
      vec3 p1, p2; float radius;
      fetchCylinder(primIndex, p1, p2, radius);
      float hitType = extra.x;
      geomNormal = cylinderNormal(hitPos, p1, p2, radius, hitType);
      if (dot(geomNormal, dir) > 0.0) {
        geomNormal = -geomNormal;
      }
      shadingNormal = geomNormal;
      baseColor = mix(uBaseColor, fetchCylinderColor(primIndex), float(uUseGltfColor));
    }

    vec3 V = normalize(-dir);

    // Direct lighting from analytical lights
    vec3 direct = shadeDirect(hitPos, shadingNormal, geomNormal, baseColor, V, seed);
    radiance += throughput * direct;

    float metallic = (uMaterialMode == 1) ? 0.0 : uMetallic;
    float rough = (uMaterialMode == 1) ? uMatteRoughness : uRoughness;
    float diffRough = (uMaterialMode == 1) ? uMatteDiffuseRoughness : 0.0;
    float wrap = (uMaterialMode == 1) ? uWrapDiffuse : 0.0;
    vec3 F0 = (uMaterialMode == 1) ? vec3(uMatteSpecular) : mix(vec3(0.04), baseColor, metallic);

    if (bounce == 0 && uMaterialMode == 0 && uRimBoost > 0.0) {
      float NdotV = max(dot(shadingNormal, V), 0.0);
      float rim = pow(1.0 - NdotV, 3.0);
      vec3 rimColor = baseColor;
      radiance += throughput * rimColor * (uRimBoost * rim);
    }

    // Next Event Estimation: Sample environment directly with MIS
    if (uUseEnv == 1) {
      float envSamplePdf;
      vec3 envDir = sampleEnvDirection(seed, envSamplePdf);
      float envNdotL = dot(shadingNormal, envDir);

      if (envNdotL > 0.0) {
        // Check visibility
        bool occluded = traceAny(hitPos + geomNormal * bias, envDir, 1e20);
        if (!occluded) {
          // Evaluate BRDF for this direction
          vec3 H = normalize(V + envDir);
          float NdotV = max(dot(shadingNormal, V), 0.001);
          float NdotH = max(dot(shadingNormal, H), 0.001);
          float VdotH = max(dot(V, H), 0.001);

          // Specular BRDF
          float D = distributionGGX(NdotH, rough);
          float G = geometrySmith(NdotV, envNdotL, rough);
          vec3 F = fresnelSchlick(VdotH, F0);
          vec3 specBrdf = (D * G * F) / max(4.0 * NdotV * envNdotL, 0.001);

          // Diffuse BRDF
          vec3 diffBrdf = evalDiffuseBrdf(shadingNormal, V, envDir, baseColor, diffRough, wrap) * (1.0 - metallic);

          // Combined BRDF
          vec3 brdf = specBrdf + diffBrdf * (vec3(1.0) - F);

          // Get environment radiance
          vec3 envRadiance = sampleEnv(envDir);

          // Compute BRDF PDF for MIS
          float specWeight = maxComponent(F0);
          float diffWeight = (1.0 - metallic) * maxComponent(baseColor);
          float sumW = specWeight + diffWeight;
          float specProb = sumW > 0.0 ? specWeight / sumW : 0.5;
          float brdfPdfVal = brdfPdf(shadingNormal, V, envDir, rough, specProb);

          // MIS weight (environment sampling)
          float misWeight = powerHeuristic(envSamplePdf, brdfPdfVal);

          // Add contribution
          vec3 contrib = throughput * brdf * envRadiance * envNdotL * misWeight / max(envSamplePdf, 1e-6);

          // Clamp to prevent fireflies
          float maxContrib = maxComponent(contrib);
          if (maxContrib > 20.0) {
            contrib *= 20.0 / maxContrib;
          }
          radiance += contrib;
        }
      }
    }

    float specWeight = maxComponent(F0);
    float diffWeight = (1.0 - metallic) * maxComponent(baseColor);
    float sum = specWeight + diffWeight;
    float specProb = sum > 0.0 ? specWeight / sum : 1.0;
    specProb = clamp(specProb, 0.0, 1.0);

    float r = rand(seed);
    vec3 newDir;
    float NdotL;
    if (r < specProb) {
      vec3 H = sampleGGXHalfVector(shadingNormal, rough, seed);
      newDir = normalize(reflect(-V, H));
      NdotL = max(dot(shadingNormal, newDir), 0.0);
      if (NdotL <= 0.0) {
        break;
      }
      float NdotV = max(dot(shadingNormal, V), 0.001);
      float NdotH = max(dot(shadingNormal, H), 0.001);
      float VdotH = max(dot(V, H), 0.001);
      float G = geometrySmith(NdotV, NdotL, rough);
      vec3 F = fresnelSchlick(VdotH, F0);
      // Simplified importance sampling weight for GGX: G * F * VdotH / (NdotV * NdotH)
      vec3 weight = G * F * VdotH / (NdotV * NdotH * max(specProb, 0.01));
      throughput *= weight;

      // Store BRDF PDF for MIS when hitting environment
      float D = distributionGGX(NdotH, rough);
      lastBrdfPdf = specProb * D * NdotH / (4.0 * VdotH);
    } else {
      newDir = cosineSampleHemisphere(shadingNormal, seed);
      NdotL = max(dot(shadingNormal, newDir), 0.0);
      vec3 diffBrdf = evalDiffuseBrdf(shadingNormal, V, newDir, baseColor, diffRough, wrap) * (1.0 - metallic);
      throughput *= diffBrdf * PI / max(1.0 - specProb, 0.01);

      // Store BRDF PDF for MIS
      lastBrdfPdf = (1.0 - specProb) * NdotL / PI;
    }

    // Clamp throughput to prevent fireflies from extreme weights
    float maxThroughput = maxComponent(throughput);
    if (maxThroughput > 10.0) {
      throughput *= 10.0 / maxThroughput;
    }

    origin = hitPos + geomNormal * bias;
    dir = newDir;

    // Apply Russian roulette starting from bounce 1 to reduce fireflies
    if (bounce >= 1) {
      float p = clamp(maxComponent(throughput), 0.05, 0.95);
      if (rand(seed) > p) {
        break;
      }
      throughput /= p;
    }
  }

  // Final clamp to catch any remaining extreme values
  radiance = min(radiance, vec3(100.0));
  return radiance;
}

// Heat map color from blue (cold) to red (hot)
vec3 heatMap(float t) {
  t = clamp(t, 0.0, 1.0);
  // Blue -> Cyan -> Green -> Yellow -> Red
  vec3 c;
  if (t < 0.25) {
    c = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t * 4.0);
  } else if (t < 0.5) {
    c = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.25) * 4.0);
  } else if (t < 0.75) {
    c = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.5) * 4.0);
  } else {
    c = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) * 4.0);
  }
  return c;
}

// Visualize normals, BVH traversal cost, or depth
vec3 traceVisualization(vec3 origin, vec3 dir) {
  float t;
  int primType;
  int primIndex;
  vec3 extra;
  int traversalCost;
  bool hit = traceClosest(origin, dir, t, primType, primIndex, extra, traversalCost);

  if (uVisMode == 1) {
    // Normal visualization
    if (!hit) return vec3(0.0);
    vec3 hitPos = origin + dir * t;
    vec3 normal;
    if (primType == PRIM_TRIANGLE) {
      vec3 bary = vec3(1.0 - extra.x - extra.y, extra.x, extra.y);
      normal = fetchTriNormal(primIndex, bary);
    } else if (primType == PRIM_SPHERE) {
      vec4 s = fetchSphere(primIndex);
      normal = normalize(hitPos - s.xyz);
    } else if (primType == PRIM_CYLINDER) {
      vec3 p1, p2; float radius;
      fetchCylinder(primIndex, p1, p2, radius);
      normal = cylinderNormal(hitPos, p1, p2, radius, extra.x);
    }
    // Map normal from [-1,1] to [0,1] for display
    return normal * 0.5 + 0.5;
  } else if (uVisMode == 2) {
    // BVH traversal cost (node visits + primitive tests)
    // Scale: 0-200 iterations mapped to full color range
    float costNorm = float(traversalCost) / 200.0;
    return heatMap(costNorm);
  } else if (uVisMode == 3) {
    // Depth visualization
    if (!hit) return vec3(0.0);
    // Map depth to grayscale, assuming scene scale ~10 units
    float depthNorm = 1.0 - clamp(t / 10.0, 0.0, 1.0);
    return vec3(depthNorm);
  }
  return vec3(0.0);
}

void main() {
  vec2 uv = (gl_FragCoord.xy + vec2(0.5)) / uResolution * 2.0 - 1.0;
  vec3 dir = normalize(uCamForward + uv.x * uCamRight + uv.y * uCamUp);

  // Visualization modes (no accumulation needed)
  if (uVisMode > 0) {
    vec3 color = traceVisualization(uCamOrigin, dir);
    outColor = vec4(color, 1.0);
    return;
  }

  // Normal path tracing with accumulation
  int spp = clamp(uSamplesPerBounce, 1, 8);
  vec3 sum = vec3(0.0);
  for (int s = 0; s < 8; s += 1) {
    if (s >= spp) {
      break;
    }
    uint seed = initSeed() + uint(s) * 747796405u;
    vec2 jitter = vec2(rand(seed), rand(seed)) - vec2(0.5);
    vec2 pixel = gl_FragCoord.xy + jitter;
    vec2 uvJittered = (pixel + vec2(0.5)) / uResolution * 2.0 - 1.0;
    vec3 dirJittered = normalize(uCamForward + uvJittered.x * uCamRight + uvJittered.y * uCamUp);
    sum += tracePath(uCamOrigin, dirJittered, seed);
  }
  vec3 color = sum / float(spp);
  color *= uExposure;

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
uniform int uToneMapMode;

vec3 toneMapReinhard(vec3 c) {
  return c / (vec3(1.0) + c);
}

vec3 toneMapACES(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  vec2 uv = gl_FragCoord.xy / uDisplayResolution;
  vec3 color = texture(uDisplayTex, uv).rgb;
  vec3 mapped = color;
  if (uToneMapMode == 1) {
    mapped = toneMapACES(color);
  } else if (uToneMapMode == 2) {
    mapped = toneMapReinhard(color);
  }
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

export function createEnvTexture(gl, width, height, data = null) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

export function createCdfTexture(gl, data, width, height) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  // Use NEAREST filtering for CDF lookup (we do manual interpolation in shader)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, data);
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
  const floatLinear = gl.getExtension("OES_texture_float_linear");
  if (!floatLinear) {
    throw new Error("OES_texture_float_linear is required for environment lighting.");
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
  gl.uniform1i(gl.getUniformLocation(program, "uTriNormalTex"), uniforms.triNormalUnit);
  gl.uniform1i(gl.getUniformLocation(program, "uTriColorTex"), uniforms.triColorUnit);
  gl.uniform1i(gl.getUniformLocation(program, "uPrimIndexTex"), uniforms.primIndexUnit);
  gl.uniform1i(gl.getUniformLocation(program, "uSphereTex"), uniforms.sphereUnit);
  gl.uniform1i(gl.getUniformLocation(program, "uSphereColorTex"), uniforms.sphereColorUnit);
  gl.uniform1i(gl.getUniformLocation(program, "uCylinderTex"), uniforms.cylinderUnit);
  gl.uniform1i(gl.getUniformLocation(program, "uCylinderColorTex"), uniforms.cylinderColorUnit);
  gl.uniform1i(gl.getUniformLocation(program, "uAccumTex"), uniforms.accumUnit);
  gl.uniform1i(gl.getUniformLocation(program, "uEnvTex"), uniforms.envUnit);
  gl.uniform3fv(gl.getUniformLocation(program, "uCamOrigin"), uniforms.camOrigin);
  gl.uniform3fv(gl.getUniformLocation(program, "uCamRight"), uniforms.camRight);
  gl.uniform3fv(gl.getUniformLocation(program, "uCamUp"), uniforms.camUp);
  gl.uniform3fv(gl.getUniformLocation(program, "uCamForward"), uniforms.camForward);
  gl.uniform2fv(gl.getUniformLocation(program, "uResolution"), uniforms.resolution);
  gl.uniform2fv(gl.getUniformLocation(program, "uBvhTexSize"), uniforms.bvhTexSize);
  gl.uniform2fv(gl.getUniformLocation(program, "uTriTexSize"), uniforms.triTexSize);
  gl.uniform2fv(gl.getUniformLocation(program, "uTriNormalTexSize"), uniforms.triNormalTexSize);
  gl.uniform2fv(gl.getUniformLocation(program, "uTriColorTexSize"), uniforms.triColorTexSize);
  gl.uniform2fv(gl.getUniformLocation(program, "uPrimIndexTexSize"), uniforms.primIndexTexSize);
  gl.uniform2fv(gl.getUniformLocation(program, "uSphereTexSize"), uniforms.sphereTexSize);
  gl.uniform2fv(gl.getUniformLocation(program, "uCylinderTexSize"), uniforms.cylinderTexSize);
  gl.uniform1i(gl.getUniformLocation(program, "uFrameIndex"), uniforms.frameIndex);
  gl.uniform1i(gl.getUniformLocation(program, "uTriCount"), uniforms.triCount);
  gl.uniform1i(gl.getUniformLocation(program, "uSphereCount"), uniforms.sphereCount);
  gl.uniform1i(gl.getUniformLocation(program, "uCylinderCount"), uniforms.cylinderCount);
  gl.uniform1i(gl.getUniformLocation(program, "uUseBvh"), uniforms.useBvh);
  gl.uniform1i(gl.getUniformLocation(program, "uUseGltfColor"), uniforms.useGltfColor);
  gl.uniform3fv(gl.getUniformLocation(program, "uBaseColor"), uniforms.baseColor);
  gl.uniform1f(gl.getUniformLocation(program, "uMetallic"), uniforms.metallic);
  gl.uniform1f(gl.getUniformLocation(program, "uRoughness"), uniforms.roughness);
  const materialMode = uniforms.materialMode === "matte" || uniforms.materialMode === 1 ? 1 : 0;
  gl.uniform1i(gl.getUniformLocation(program, "uMaterialMode"), materialMode);
  gl.uniform1f(gl.getUniformLocation(program, "uMatteSpecular"), uniforms.matteSpecular ?? 0.03);
  gl.uniform1f(gl.getUniformLocation(program, "uMatteRoughness"), uniforms.matteRoughness ?? 0.5);
  gl.uniform1f(gl.getUniformLocation(program, "uMatteDiffuseRoughness"), uniforms.matteDiffuseRoughness ?? 0.5);
  gl.uniform1f(gl.getUniformLocation(program, "uWrapDiffuse"), uniforms.wrapDiffuse ?? 0.2);
  gl.uniform1f(gl.getUniformLocation(program, "uRimBoost"), uniforms.rimBoost ?? 0.0);
  gl.uniform1i(gl.getUniformLocation(program, "uMaxBounces"), uniforms.maxBounces);
  gl.uniform1f(gl.getUniformLocation(program, "uExposure"), uniforms.exposure);
  gl.uniform1f(gl.getUniformLocation(program, "uAmbientIntensity"), uniforms.ambientIntensity);
  gl.uniform3fv(gl.getUniformLocation(program, "uAmbientColor"), uniforms.ambientColor);
  gl.uniform1i(gl.getUniformLocation(program, "uSamplesPerBounce"), uniforms.samplesPerBounce);
  gl.uniform1i(gl.getUniformLocation(program, "uCastShadows"), uniforms.castShadows);
  gl.uniform1f(gl.getUniformLocation(program, "uRayBias"), uniforms.rayBias);
  gl.uniform1f(gl.getUniformLocation(program, "uTMin"), uniforms.tMin);
  gl.uniform1f(gl.getUniformLocation(program, "uEnvIntensity"), uniforms.envIntensity);
  gl.uniform1f(gl.getUniformLocation(program, "uEnvMaxLuminance"), uniforms.envMaxLuminance ?? 50.0);
  gl.uniform1i(gl.getUniformLocation(program, "uUseEnv"), uniforms.useEnv);
  gl.uniform1i(gl.getUniformLocation(program, "uEnvMarginalCdf"), uniforms.envMarginalCdfUnit ?? 0);
  gl.uniform1i(gl.getUniformLocation(program, "uEnvConditionalCdf"), uniforms.envConditionalCdfUnit ?? 0);
  gl.uniform2fv(gl.getUniformLocation(program, "uEnvSize"), uniforms.envSize ?? [0, 0]);

  const lightEnabled = new Int32Array(3);
  const lightDir = new Float32Array(9);
  const lightColor = new Float32Array(9);
  const lightIntensity = new Float32Array(3);
  const lightAngle = new Float32Array(3);
  for (let i = 0; i < 3; i += 1) {
    const light = uniforms.lights[i];
    const dir = uniforms.lightDirs[i];
    lightEnabled[i] = light.enabled ? 1 : 0;
    lightDir[i * 3] = dir[0];
    lightDir[i * 3 + 1] = dir[1];
    lightDir[i * 3 + 2] = dir[2];
    lightColor[i * 3] = light.color[0];
    lightColor[i * 3 + 1] = light.color[1];
    lightColor[i * 3 + 2] = light.color[2];
    lightIntensity[i] = light.intensity;
    lightAngle[i] = light.angle ?? 0.0;
  }
  gl.uniform1iv(gl.getUniformLocation(program, "uLightEnabled"), lightEnabled);
  gl.uniform3fv(gl.getUniformLocation(program, "uLightDir"), lightDir);
  gl.uniform3fv(gl.getUniformLocation(program, "uLightColor"), lightColor);
  gl.uniform1fv(gl.getUniformLocation(program, "uLightIntensity"), lightIntensity);
  gl.uniform1fv(gl.getUniformLocation(program, "uLightAngle"), lightAngle);

  // Visualization mode
  gl.uniform1i(gl.getUniformLocation(program, "uVisMode"), uniforms.visMode || 0);
}

export function setDisplayUniforms(gl, program, uniforms) {
  gl.useProgram(program);
  gl.uniform1i(gl.getUniformLocation(program, "uDisplayTex"), uniforms.displayUnit);
  gl.uniform2fv(gl.getUniformLocation(program, "uDisplayResolution"), uniforms.displayResolution);
  const mode = uniforms.toneMap === "aces" ? 1 : (uniforms.toneMap === "linear" ? 0 : 2);
  gl.uniform1i(gl.getUniformLocation(program, "uToneMapMode"), mode);
}

export function drawFullscreen(gl) {
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}
