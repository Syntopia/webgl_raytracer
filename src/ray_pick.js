import { PRIM_TRIANGLE, PRIM_SPHERE, PRIM_CYLINDER } from "./bvh.js";

function vec3Dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vec3Sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vec3Cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}

function vec3Length(a) {
  return Math.sqrt(vec3Dot(a, a));
}

function passesClip(origin, dir, t, clip) {
  if (!clip || !clip.enabled) return true;
  const hitPos = [origin[0] + dir[0] * t, origin[1] + dir[1] * t, origin[2] + dir[2] * t];
  const side = vec3Dot(clip.normal, hitPos) - clip.offset;
  return side * clip.side <= 0.0;
}

function rayIntersectsAabb(origin, dir, bounds, tMax) {
  let tmin = 0.0;
  let tmax = tMax;

  for (const axis of ["x", "y", "z"]) {
    const i = axis === "x" ? 0 : axis === "y" ? 1 : 2;
    const bmin = axis === "x" ? bounds.minX : axis === "y" ? bounds.minY : bounds.minZ;
    const bmax = axis === "x" ? bounds.maxX : axis === "y" ? bounds.maxY : bounds.maxZ;
    if (Math.abs(dir[i]) < 1e-8) {
      if (origin[i] < bmin || origin[i] > bmax) return false;
      continue;
    }
    const inv = 1.0 / dir[i];
    const t1 = (bmin - origin[i]) * inv;
    const t2 = (bmax - origin[i]) * inv;
    const tNear = Math.min(t1, t2);
    const tFar = Math.max(t1, t2);
    tmin = Math.max(tmin, tNear);
    tmax = Math.min(tmax, tFar);
    if (tmax < tmin) return false;
  }
  return true;
}

function intersectTriangle(origin, dir, v0, v1, v2, tMin, clip = null) {
  const e1 = vec3Sub(v1, v0);
  const e2 = vec3Sub(v2, v0);
  const p = vec3Cross(dir, e2);
  const det = vec3Dot(e1, p);
  if (Math.abs(det) < 1e-6) return -1;
  const invDet = 1.0 / det;
  const tvec = vec3Sub(origin, v0);
  const u = vec3Dot(tvec, p) * invDet;
  const q = vec3Cross(tvec, e1);
  const v = vec3Dot(dir, q) * invDet;
  if (u < 0.0 || v < 0.0 || u + v > 1.0) return -1;
  const t = vec3Dot(e2, q) * invDet;
  if (t <= tMin) return -1;
  if (!passesClip(origin, dir, t, clip)) return -1;
  return t;
}

function intersectSphere(origin, dir, center, radius, tMin) {
  const oc = vec3Sub(origin, center);
  const b = vec3Dot(oc, dir);
  const c = vec3Dot(oc, oc) - radius * radius;
  const disc = b * b - c;
  if (disc < 0.0) return -1;
  const sqrtD = Math.sqrt(disc);
  let t = -b - sqrtD;
  if (t <= tMin) {
    t = -b + sqrtD;
    if (t <= tMin) return -1;
  }
  return t;
}

function intersectCylinder(origin, dir, p1, p2, radius, tMin) {
  const axisRaw = vec3Sub(p2, p1);
  const height = vec3Length(axisRaw);
  if (height < 1e-6) {
    return intersectSphere(origin, dir, p1, radius, tMin);
  }
  const axis = [axisRaw[0] / height, axisRaw[1] / height, axisRaw[2] / height];
  const oc = vec3Sub(origin, p1);
  const dirDotAxis = vec3Dot(dir, axis);
  const ocDotAxis = vec3Dot(oc, axis);
  const dirPerp = [
    dir[0] - axis[0] * dirDotAxis,
    dir[1] - axis[1] * dirDotAxis,
    dir[2] - axis[2] * dirDotAxis
  ];
  const ocPerp = [
    oc[0] - axis[0] * ocDotAxis,
    oc[1] - axis[1] * ocDotAxis,
    oc[2] - axis[2] * ocDotAxis
  ];
  const a = vec3Dot(dirPerp, dirPerp);
  const b = 2.0 * vec3Dot(dirPerp, ocPerp);
  const c = vec3Dot(ocPerp, ocPerp) - radius * radius;

  let bestT = -1;

  if (a > 1e-8) {
    const disc = b * b - 4.0 * a * c;
    if (disc >= 0.0) {
      const sqrtD = Math.sqrt(disc);
      const t1 = (-b - sqrtD) / (2.0 * a);
      const t2 = (-b + sqrtD) / (2.0 * a);
      if (t1 > tMin) {
        const h = ocDotAxis + t1 * dirDotAxis;
        if (h >= 0.0 && h <= height) bestT = t1;
      }
      if (bestT < 0.0 && t2 > tMin) {
        const h = ocDotAxis + t2 * dirDotAxis;
        if (h >= 0.0 && h <= height) bestT = t2;
      }
    }
  }

  const cap1 = intersectSphere(origin, dir, p1, radius, tMin);
  if (cap1 > tMin && (bestT < 0.0 || cap1 < bestT)) {
    const hitPos = [origin[0] + dir[0] * cap1, origin[1] + dir[1] * cap1, origin[2] + dir[2] * cap1];
    const h = vec3Dot(vec3Sub(hitPos, p1), axis);
    if (h <= 0.0) bestT = cap1;
  }

  const cap2 = intersectSphere(origin, dir, p2, radius, tMin);
  if (cap2 > tMin && (bestT < 0.0 || cap2 < bestT)) {
    const hitPos = [origin[0] + dir[0] * cap2, origin[1] + dir[1] * cap2, origin[2] + dir[2] * cap2];
    const h = vec3Dot(vec3Sub(hitPos, p2), axis);
    if (h >= 0.0) bestT = cap2;
  }

  return bestT;
}

function intersectPrimitive(scene, prim, origin, dir, tMin, clip) {
  if (prim.type === PRIM_TRIANGLE) {
    const tri = scene.tris[prim.index];
    if (!tri) throw new Error(`Missing triangle ${prim.index} for ray picking.`);
    const i0 = tri[0] * 3;
    const i1 = tri[1] * 3;
    const i2 = tri[2] * 3;
    const v0 = [scene.positions[i0], scene.positions[i0 + 1], scene.positions[i0 + 2]];
    const v1 = [scene.positions[i1], scene.positions[i1 + 1], scene.positions[i1 + 2]];
    const v2 = [scene.positions[i2], scene.positions[i2 + 1], scene.positions[i2 + 2]];
    return intersectTriangle(origin, dir, v0, v1, v2, tMin, clip);
  }
  if (prim.type === PRIM_SPHERE) {
    const sphere = scene.spheres[prim.index];
    if (!sphere) throw new Error(`Missing sphere ${prim.index} for ray picking.`);
    const t = intersectSphere(origin, dir, sphere.center, sphere.radius, tMin);
    if (t <= tMin) return -1;
    return passesClip(origin, dir, t, clip) ? t : -1;
  }
  if (prim.type === PRIM_CYLINDER) {
    const cyl = scene.cylinders[prim.index];
    if (!cyl) throw new Error(`Missing cylinder ${prim.index} for ray picking.`);
    const t = intersectCylinder(origin, dir, cyl.p1, cyl.p2, cyl.radius, tMin);
    if (t <= tMin) return -1;
    return passesClip(origin, dir, t, clip) ? t : -1;
  }
  throw new Error(`Unknown primitive type ${prim.type} during ray picking.`);
}

export function primTypeLabel(type) {
  if (type === PRIM_TRIANGLE) return "triangle";
  if (type === PRIM_SPHERE) return "sphere";
  if (type === PRIM_CYLINDER) return "cylinder";
  return "primitive";
}

export function traceSceneRay(scene, origin, dir, options = {}) {
  if (!scene || !Array.isArray(scene.nodes) || !Array.isArray(scene.primitives)) {
    throw new Error("Scene BVH data is required for ray picking.");
  }
  if (scene.nodes.length === 0) {
    throw new Error("Scene has no BVH nodes for ray picking.");
  }
  const tMin = options.tMin ?? 1e-6;
  const clip = options.clip ?? null;
  let closest = Infinity;
  let best = null;

  const stack = [0];
  while (stack.length > 0) {
    const nodeIndex = stack.pop();
    const node = scene.nodes[nodeIndex];
    if (!node) continue;
    if (!rayIntersectsAabb(origin, dir, node.bounds, closest)) continue;

    if (node.primCount > 0) {
      if (!Array.isArray(node.primIndices)) {
        throw new Error("Leaf node primitive indices are missing for ray picking.");
      }
      for (const primRef of node.primIndices) {
        const prim = scene.primitives[primRef];
        if (!prim) continue;
        const t = intersectPrimitive(scene, prim, origin, dir, tMin, clip);
        if (t > tMin && t < closest) {
          closest = t;
          best = { t, primType: prim.type, primIndex: prim.index };
        }
      }
    } else {
      if (Number.isInteger(node.rightChild) && node.rightChild >= 0) {
        stack.push(node.rightChild);
      }
      if (Number.isInteger(node.leftFirst) && node.leftFirst >= 0) {
        stack.push(node.leftFirst);
      }
    }
  }

  return best;
}
