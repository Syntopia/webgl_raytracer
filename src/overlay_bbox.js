import { PRIM_TRIANGLE, PRIM_SPHERE, PRIM_CYLINDER } from "./bvh.js";

function vec3Dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function buildTriangleBounds(scene, primIndex) {
  const tri = scene.tris?.[primIndex];
  if (!tri) {
    throw new Error(`Missing triangle ${primIndex} for hover bounds.`);
  }
  const i0 = tri[0] * 3;
  const i1 = tri[1] * 3;
  const i2 = tri[2] * 3;
  const positions = scene.positions;
  const x0 = positions[i0], y0 = positions[i0 + 1], z0 = positions[i0 + 2];
  const x1 = positions[i1], y1 = positions[i1 + 1], z1 = positions[i1 + 2];
  const x2 = positions[i2], y2 = positions[i2 + 1], z2 = positions[i2 + 2];
  return {
    minX: Math.min(x0, x1, x2),
    minY: Math.min(y0, y1, y2),
    minZ: Math.min(z0, z1, z2),
    maxX: Math.max(x0, x1, x2),
    maxY: Math.max(y0, y1, y2),
    maxZ: Math.max(z0, z1, z2)
  };
}

function buildSphereBounds(scene, primIndex) {
  const sphere = scene.spheres?.[primIndex];
  if (!sphere) {
    throw new Error(`Missing sphere ${primIndex} for hover bounds.`);
  }
  return {
    minX: sphere.center[0] - sphere.radius,
    minY: sphere.center[1] - sphere.radius,
    minZ: sphere.center[2] - sphere.radius,
    maxX: sphere.center[0] + sphere.radius,
    maxY: sphere.center[1] + sphere.radius,
    maxZ: sphere.center[2] + sphere.radius
  };
}

function buildCylinderBounds(scene, primIndex) {
  const cyl = scene.cylinders?.[primIndex];
  if (!cyl) {
    throw new Error(`Missing cylinder ${primIndex} for hover bounds.`);
  }

  const dx = cyl.p2[0] - cyl.p1[0];
  const dy = cyl.p2[1] - cyl.p1[1];
  const dz = cyl.p2[2] - cyl.p1[2];
  const height = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const axis = height > 1e-8 ? [dx / height, dy / height, dz / height] : [0, 1, 0];

  const extentX = cyl.radius * Math.sqrt(Math.max(0, 1 - axis[0] * axis[0]));
  const extentY = cyl.radius * Math.sqrt(Math.max(0, 1 - axis[1] * axis[1]));
  const extentZ = cyl.radius * Math.sqrt(Math.max(0, 1 - axis[2] * axis[2]));

  return {
    minX: Math.min(cyl.p1[0], cyl.p2[0]) - extentX,
    minY: Math.min(cyl.p1[1], cyl.p2[1]) - extentY,
    minZ: Math.min(cyl.p1[2], cyl.p2[2]) - extentZ,
    maxX: Math.max(cyl.p1[0], cyl.p2[0]) + extentX,
    maxY: Math.max(cyl.p1[1], cyl.p2[1]) + extentY,
    maxZ: Math.max(cyl.p1[2], cyl.p2[2]) + extentZ
  };
}

export function computePrimitiveWorldBounds(scene, primType, primIndex) {
  if (!scene) {
    throw new Error("Scene is required for primitive hover bounds.");
  }
  if (primType === PRIM_TRIANGLE) {
    return buildTriangleBounds(scene, primIndex);
  }
  if (primType === PRIM_SPHERE) {
    return buildSphereBounds(scene, primIndex);
  }
  if (primType === PRIM_CYLINDER) {
    return buildCylinderBounds(scene, primIndex);
  }
  throw new Error(`Unknown primitive type ${primType} for hover bounds.`);
}

function projectPointToCanvas(point, camera, width, height) {
  const d = [
    point[0] - camera.origin[0],
    point[1] - camera.origin[1],
    point[2] - camera.origin[2]
  ];
  const z = vec3Dot(d, camera.forward);
  if (z <= 1e-6) {
    return null;
  }

  const rightLenSq = vec3Dot(camera.right, camera.right);
  const upLenSq = vec3Dot(camera.up, camera.up);
  if (rightLenSq <= 1e-10 || upLenSq <= 1e-10) {
    throw new Error("Camera projection basis is degenerate.");
  }

  const ndcX = vec3Dot(d, camera.right) / (z * rightLenSq);
  const ndcY = vec3Dot(d, camera.up) / (z * upLenSq);
  const x = (ndcX * 0.5 + 0.5) * width;
  const y = (1.0 - (ndcY * 0.5 + 0.5)) * height;
  return { x, y };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function projectAabbToCanvasRect(bounds, camera, width, height) {
  if (!bounds || !camera) {
    throw new Error("Bounds and camera are required for AABB projection.");
  }
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error("Projection width and height must be > 0.");
  }

  const corners = [
    [bounds.minX, bounds.minY, bounds.minZ],
    [bounds.minX, bounds.minY, bounds.maxZ],
    [bounds.minX, bounds.maxY, bounds.minZ],
    [bounds.minX, bounds.maxY, bounds.maxZ],
    [bounds.maxX, bounds.minY, bounds.minZ],
    [bounds.maxX, bounds.minY, bounds.maxZ],
    [bounds.maxX, bounds.maxY, bounds.minZ],
    [bounds.maxX, bounds.maxY, bounds.maxZ]
  ];

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let visible = 0;

  for (const corner of corners) {
    const p = projectPointToCanvas(corner, camera, width, height);
    if (!p) continue;
    visible += 1;
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }

  if (visible === 0) {
    return null;
  }
  if (maxX < 0 || maxY < 0 || minX > width || minY > height) {
    return null;
  }

  const clampedMinX = clamp(minX, 0, width);
  const clampedMinY = clamp(minY, 0, height);
  const clampedMaxX = clamp(maxX, 0, width);
  const clampedMaxY = clamp(maxY, 0, height);
  const boxWidth = clampedMaxX - clampedMinX;
  const boxHeight = clampedMaxY - clampedMinY;
  if (boxWidth < 1 || boxHeight < 1) {
    return null;
  }

  return {
    minX: clampedMinX,
    minY: clampedMinY,
    maxX: clampedMaxX,
    maxY: clampedMaxY,
    width: boxWidth,
    height: boxHeight
  };
}
