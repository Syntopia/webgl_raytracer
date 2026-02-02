const BIN_COUNT = 8;

// Primitive type constants
export const PRIM_TRIANGLE = 0;
export const PRIM_SPHERE = 1;
export const PRIM_CYLINDER = 2;

// Generic primitive wrapper that provides bounds, centroid, and surface area
function createPrimitiveInfo(type, index, data) {
  let bounds, centroid, surfaceArea;

  if (type === PRIM_TRIANGLE) {
    const { positions, tri } = data;
    const i0 = tri[0] * 3;
    const i1 = tri[1] * 3;
    const i2 = tri[2] * 3;

    const x0 = positions[i0], y0 = positions[i0 + 1], z0 = positions[i0 + 2];
    const x1 = positions[i1], y1 = positions[i1 + 1], z1 = positions[i1 + 2];
    const x2 = positions[i2], y2 = positions[i2 + 1], z2 = positions[i2 + 2];

    bounds = {
      minX: Math.min(x0, x1, x2),
      minY: Math.min(y0, y1, y2),
      minZ: Math.min(z0, z1, z2),
      maxX: Math.max(x0, x1, x2),
      maxY: Math.max(y0, y1, y2),
      maxZ: Math.max(z0, z1, z2)
    };

    centroid = {
      x: (x0 + x1 + x2) / 3,
      y: (y0 + y1 + y2) / 3,
      z: (z0 + z1 + z2) / 3
    };

    // Triangle surface area: 0.5 * |cross(e1, e2)|
    const e1x = x1 - x0, e1y = y1 - y0, e1z = z1 - z0;
    const e2x = x2 - x0, e2y = y2 - y0, e2z = z2 - z0;
    const cx = e1y * e2z - e1z * e2y;
    const cy = e1z * e2x - e1x * e2z;
    const cz = e1x * e2y - e1y * e2x;
    surfaceArea = 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
  } else if (type === PRIM_SPHERE) {
    const { center, radius } = data;
    bounds = {
      minX: center[0] - radius,
      minY: center[1] - radius,
      minZ: center[2] - radius,
      maxX: center[0] + radius,
      maxY: center[1] + radius,
      maxZ: center[2] + radius
    };
    centroid = { x: center[0], y: center[1], z: center[2] };
    surfaceArea = 4 * Math.PI * radius * radius;
  } else if (type === PRIM_CYLINDER) {
    const { p1, p2, radius } = data;
    // Cylinder bounds: union of two endpoint spheres expanded by radius
    const dx = p2[0] - p1[0], dy = p2[1] - p1[1], dz = p2[2] - p1[2];
    const height = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // Compute tight AABB for cylinder
    // For an arbitrarily oriented cylinder, we need to consider the full extent
    const axis = height > 0 ? [dx / height, dy / height, dz / height] : [0, 1, 0];

    // Extent in each direction perpendicular to axis
    const extentX = radius * Math.sqrt(1 - axis[0] * axis[0]);
    const extentY = radius * Math.sqrt(1 - axis[1] * axis[1]);
    const extentZ = radius * Math.sqrt(1 - axis[2] * axis[2]);

    bounds = {
      minX: Math.min(p1[0], p2[0]) - extentX - 0.001,
      minY: Math.min(p1[1], p2[1]) - extentY - 0.001,
      minZ: Math.min(p1[2], p2[2]) - extentZ - 0.001,
      maxX: Math.max(p1[0], p2[0]) + extentX + 0.001,
      maxY: Math.max(p1[1], p2[1]) + extentY + 0.001,
      maxZ: Math.max(p1[2], p2[2]) + extentZ + 0.001
    };

    centroid = {
      x: (p1[0] + p2[0]) / 2,
      y: (p1[1] + p2[1]) / 2,
      z: (p1[2] + p2[2]) / 2
    };

    // Cylinder surface area: 2*pi*r*h (side) + 2*pi*r^2 (caps)
    surfaceArea = 2 * Math.PI * radius * (height + radius);
  }

  return { type, index, bounds, centroid, surfaceArea };
}

function boundsUnion(a, b) {
  return {
    minX: Math.min(a.minX, b.minX),
    minY: Math.min(a.minY, b.minY),
    minZ: Math.min(a.minZ, b.minZ),
    maxX: Math.max(a.maxX, b.maxX),
    maxY: Math.max(a.maxY, b.maxY),
    maxZ: Math.max(a.maxZ, b.maxZ)
  };
}

function boundsFromPrimitives(primitives, primIndices) {
  let bounds = {
    minX: Infinity, minY: Infinity, minZ: Infinity,
    maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity
  };
  for (const idx of primIndices) {
    bounds = boundsUnion(bounds, primitives[idx].bounds);
  }
  return bounds;
}

function centroidBoundsFromPrimitives(primitives, primIndices) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const idx of primIndices) {
    const c = primitives[idx].centroid;
    minX = Math.min(minX, c.x);
    minY = Math.min(minY, c.y);
    minZ = Math.min(minZ, c.z);
    maxX = Math.max(maxX, c.x);
    maxY = Math.max(maxY, c.y);
    maxZ = Math.max(maxZ, c.z);
  }
  return { minX, minY, minZ, maxX, maxY, maxZ };
}

function boundsSurfaceArea(bounds) {
  const dx = bounds.maxX - bounds.minX;
  const dy = bounds.maxY - bounds.minY;
  const dz = bounds.maxZ - bounds.minZ;
  return 2 * (dx * dy + dy * dz + dz * dx);
}

function buildBins(primitives, primIndices, axis, cmin, cmax) {
  const bins = Array.from({ length: BIN_COUNT }, () => ({
    count: 0,
    bounds: {
      minX: Infinity, minY: Infinity, minZ: Infinity,
      maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity
    }
  }));

  const scale = cmax - cmin > 0 ? BIN_COUNT / (cmax - cmin) : 0;

  for (const idx of primIndices) {
    const prim = primitives[idx];
    const c = axis === 0 ? prim.centroid.x : axis === 1 ? prim.centroid.y : prim.centroid.z;

    let bin = Math.floor((c - cmin) * scale);
    if (bin < 0) bin = 0;
    if (bin >= BIN_COUNT) bin = BIN_COUNT - 1;

    bins[bin].count += 1;
    bins[bin].bounds = boundsUnion(bins[bin].bounds, prim.bounds);
  }

  return bins;
}

function findBestSplit(primitives, primIndices) {
  const cBounds = centroidBoundsFromPrimitives(primitives, primIndices);
  const cmin = [cBounds.minX, cBounds.minY, cBounds.minZ];
  const cmax = [cBounds.maxX, cBounds.maxY, cBounds.maxZ];

  let bestAxis = -1;
  let bestIndex = -1;
  let bestCost = Infinity;

  for (let axis = 0; axis < 3; axis += 1) {
    const bins = buildBins(primitives, primIndices, axis, cmin[axis], cmax[axis]);

    const leftBounds = new Array(BIN_COUNT);
    const rightBounds = new Array(BIN_COUNT);
    const leftCount = new Array(BIN_COUNT).fill(0);
    const rightCount = new Array(BIN_COUNT).fill(0);

    let accumBounds = {
      minX: Infinity, minY: Infinity, minZ: Infinity,
      maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity
    };
    let accumCount = 0;
    for (let i = 0; i < BIN_COUNT; i += 1) {
      if (bins[i].count > 0) {
        accumBounds = boundsUnion(accumBounds, bins[i].bounds);
      }
      accumCount += bins[i].count;
      leftBounds[i] = { ...accumBounds };
      leftCount[i] = accumCount;
    }

    accumBounds = {
      minX: Infinity, minY: Infinity, minZ: Infinity,
      maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity
    };
    accumCount = 0;
    for (let i = BIN_COUNT - 1; i >= 0; i -= 1) {
      if (bins[i].count > 0) {
        accumBounds = boundsUnion(accumBounds, bins[i].bounds);
      }
      accumCount += bins[i].count;
      rightBounds[i] = { ...accumBounds };
      rightCount[i] = accumCount;
    }

    for (let i = 0; i < BIN_COUNT - 1; i += 1) {
      const lCount = leftCount[i];
      const rCount = rightCount[i + 1];
      if (lCount === 0 || rCount === 0) continue;
      const cost = boundsSurfaceArea(leftBounds[i]) * lCount + boundsSurfaceArea(rightBounds[i + 1]) * rCount;
      if (cost < bestCost) {
        bestCost = cost;
        bestAxis = axis;
        bestIndex = i;
      }
    }
  }

  return { bestAxis, bestIndex, cBounds };
}

function splitPrimitives(primitives, primIndices, axis, splitPos) {
  const left = [];
  const right = [];

  for (const idx of primIndices) {
    const c = axis === 0 ? primitives[idx].centroid.x :
              axis === 1 ? primitives[idx].centroid.y :
              primitives[idx].centroid.z;
    if (c < splitPos) {
      left.push(idx);
    } else {
      right.push(idx);
    }
  }

  if (left.length === 0 || right.length === 0) {
    const half = Math.floor(primIndices.length / 2);
    left.splice(0, left.length, ...primIndices.slice(0, half));
    right.splice(0, right.length, ...primIndices.slice(half));
  }

  return { left, right };
}

function buildNode(primitives, primIndices, maxLeafSize, depth, maxDepth, nodes) {
  const bounds = boundsFromPrimitives(primitives, primIndices);

  if (primIndices.length <= maxLeafSize || depth >= maxDepth) {
    const nodeIndex = nodes.length;
    nodes.push({
      bounds,
      leftFirst: -1,
      primCount: primIndices.length,
      rightChild: -1,
      primIndices: [...primIndices]
    });
    return nodeIndex;
  }

  const { bestAxis, bestIndex, cBounds } = findBestSplit(primitives, primIndices);
  if (bestAxis === -1) {
    const nodeIndex = nodes.length;
    nodes.push({
      bounds,
      leftFirst: -1,
      primCount: primIndices.length,
      rightChild: -1,
      primIndices: [...primIndices]
    });
    return nodeIndex;
  }

  const cmin = [cBounds.minX, cBounds.minY, cBounds.minZ][bestAxis];
  const cmax = [cBounds.maxX, cBounds.maxY, cBounds.maxZ][bestAxis];
  const splitPos = cmin + ((bestIndex + 1) / BIN_COUNT) * (cmax - cmin);

  const { left, right } = splitPrimitives(primitives, primIndices, bestAxis, splitPos);

  const nodeIndex = nodes.length;
  nodes.push({
    bounds,
    leftFirst: -1,
    primCount: 0,
    rightChild: -1,
    primIndices: []
  });

  const leftChild = buildNode(primitives, left, maxLeafSize, depth + 1, maxDepth, nodes);
  const rightChild = buildNode(primitives, right, maxLeafSize, depth + 1, maxDepth, nodes);

  nodes[nodeIndex].leftFirst = leftChild;
  nodes[nodeIndex].rightChild = rightChild;
  return nodeIndex;
}

/**
 * Build SAH BVH for triangles only (legacy interface)
 */
export function buildSAHBVH(positions, indices, options = {}) {
  return buildUnifiedBVH({ positions, indices }, [], [], options);
}

/**
 * Build unified BVH for all primitive types
 * @param {Object} triangleData - { positions: Float32Array, indices: Uint32Array }
 * @param {Array} spheres - Array of { center: [x,y,z], radius: number, color: [r,g,b] }
 * @param {Array} cylinders - Array of { p1: [x,y,z], p2: [x,y,z], radius: number, color: [r,g,b] }
 * @param {Object} options - { maxLeafSize, maxDepth }
 */
export function buildUnifiedBVH(triangleData, spheres = [], cylinders = [], options = {}) {
  const maxLeafSize = options.maxLeafSize ?? 4;
  const maxDepth = options.maxDepth ?? 32;

  const { positions, indices } = triangleData;
  const primitives = [];

  // Add triangles
  const triCount = indices ? indices.length / 3 : 0;
  const tris = [];
  for (let i = 0; i < triCount; i += 1) {
    tris.push([indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]]);
    primitives.push(createPrimitiveInfo(PRIM_TRIANGLE, i, { positions, tri: tris[i] }));
  }

  // Add spheres (indices continue from triangles)
  for (let i = 0; i < spheres.length; i += 1) {
    primitives.push(createPrimitiveInfo(PRIM_SPHERE, i, spheres[i]));
  }

  // Add cylinders
  for (let i = 0; i < cylinders.length; i += 1) {
    primitives.push(createPrimitiveInfo(PRIM_CYLINDER, i, cylinders[i]));
  }

  if (primitives.length === 0) {
    return {
      nodes: [{
        bounds: { minX: 0, minY: 0, minZ: 0, maxX: 0, maxY: 0, maxZ: 0 },
        leftFirst: -1,
        primCount: 0,
        rightChild: -1,
        primIndices: []
      }],
      tris: [],
      primitives,
      triCount,
      sphereCount: spheres.length,
      cylinderCount: cylinders.length
    };
  }

  const primIndices = Array.from({ length: primitives.length }, (_, i) => i);
  const nodes = [];
  buildNode(primitives, primIndices, maxLeafSize, 0, maxDepth, nodes);

  return {
    nodes,
    tris,
    primitives,
    triCount,
    sphereCount: spheres.length,
    cylinderCount: cylinders.length
  };
}

export function flattenBVH(nodes, primitives, triCount, sphereCount, cylinderCount) {
  const nodeCount = nodes.length;
  const primIndexList = [];

  // Each node uses 12 floats (3 RGBA32F texels):
  // Texel 0: minX, minY, minZ, leftFirst
  // Texel 1: maxX, maxY, maxZ, primCount
  // Texel 2: rightChild, 0, 0, 0
  const nodeBuffer = new ArrayBuffer(nodeCount * 48);
  const nodeFloats = new Float32Array(nodeBuffer);

  for (let i = 0; i < nodeCount; i += 1) {
    const node = nodes[i];
    const base = i * 12;

    nodeFloats[base + 0] = node.bounds.minX;
    nodeFloats[base + 1] = node.bounds.minY;
    nodeFloats[base + 2] = node.bounds.minZ;
    nodeFloats[base + 4] = node.bounds.maxX;
    nodeFloats[base + 5] = node.bounds.maxY;
    nodeFloats[base + 6] = node.bounds.maxZ;

    if (node.primCount > 0) {
      // Leaf node: store primitive indices
      nodeFloats[base + 3] = primIndexList.length;
      nodeFloats[base + 7] = node.primCount;
      nodeFloats[base + 8] = 0;

      // Store unified primitive indices
      // Each entry encodes: type (upper 2 bits) + index (lower 30 bits)
      for (const primIdx of node.primIndices) {
        const prim = primitives[primIdx];
        // Encode type and local index
        const encoded = (prim.type << 30) | (prim.index & 0x3FFFFFFF);
        primIndexList.push(encoded);
      }
    } else {
      nodeFloats[base + 3] = node.leftFirst;
      nodeFloats[base + 7] = 0;
      nodeFloats[base + 8] = node.rightChild;
    }

    nodeFloats[base + 9] = 0;
    nodeFloats[base + 10] = 0;
    nodeFloats[base + 11] = 0;
  }

  const primIndexBuffer = new Uint32Array(primIndexList.length);
  primIndexBuffer.set(primIndexList);

  return {
    nodeBuffer,
    nodeCount,
    primIndexBuffer,
    triCount,
    sphereCount,
    cylinderCount
  };
}

export function packGeometry(positions, indices) {
  const vertexCount = positions.length / 3;
  const positionVec4 = new Float32Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; i += 1) {
    positionVec4[i * 4] = positions[i * 3];
    positionVec4[i * 4 + 1] = positions[i * 3 + 1];
    positionVec4[i * 4 + 2] = positions[i * 3 + 2];
    positionVec4[i * 4 + 3] = 1;
  }

  const triCount = indices.length / 3;
  const triIndexVec4 = new Uint32Array(triCount * 4);
  for (let i = 0; i < triCount; i += 1) {
    triIndexVec4[i * 4] = indices[i * 3];
    triIndexVec4[i * 4 + 1] = indices[i * 3 + 1];
    triIndexVec4[i * 4 + 2] = indices[i * 3 + 2];
    triIndexVec4[i * 4 + 3] = 0;
  }

  return { positionVec4, triIndexVec4 };
}
