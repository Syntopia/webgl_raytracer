const BIN_COUNT = 8;

function triBounds(positions, tri) {
  const i0 = tri[0] * 3;
  const i1 = tri[1] * 3;
  const i2 = tri[2] * 3;

  const x0 = positions[i0];
  const y0 = positions[i0 + 1];
  const z0 = positions[i0 + 2];
  const x1 = positions[i1];
  const y1 = positions[i1 + 1];
  const z1 = positions[i1 + 2];
  const x2 = positions[i2];
  const y2 = positions[i2 + 1];
  const z2 = positions[i2 + 2];

  const minX = Math.min(x0, x1, x2);
  const minY = Math.min(y0, y1, y2);
  const minZ = Math.min(z0, z1, z2);
  const maxX = Math.max(x0, x1, x2);
  const maxY = Math.max(y0, y1, y2);
  const maxZ = Math.max(z0, z1, z2);

  return { minX, minY, minZ, maxX, maxY, maxZ };
}

function boundsFromTris(tris, triIndices, positions) {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  for (const triIndex of triIndices) {
    const tri = tris[triIndex];
    const b = triBounds(positions, tri);
    minX = Math.min(minX, b.minX);
    minY = Math.min(minY, b.minY);
    minZ = Math.min(minZ, b.minZ);
    maxX = Math.max(maxX, b.maxX);
    maxY = Math.max(maxY, b.maxY);
    maxZ = Math.max(maxZ, b.maxZ);
  }

  return { minX, minY, minZ, maxX, maxY, maxZ };
}

function centroidBounds(tris, triIndices, positions) {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  for (const triIndex of triIndices) {
    const tri = tris[triIndex];
    const i0 = tri[0] * 3;
    const i1 = tri[1] * 3;
    const i2 = tri[2] * 3;
    const cx = (positions[i0] + positions[i1] + positions[i2]) / 3;
    const cy = (positions[i0 + 1] + positions[i1 + 1] + positions[i2 + 1]) / 3;
    const cz = (positions[i0 + 2] + positions[i1 + 2] + positions[i2 + 2]) / 3;

    minX = Math.min(minX, cx);
    minY = Math.min(minY, cy);
    minZ = Math.min(minZ, cz);
    maxX = Math.max(maxX, cx);
    maxY = Math.max(maxY, cy);
    maxZ = Math.max(maxZ, cz);
  }

  return { minX, minY, minZ, maxX, maxY, maxZ };
}

function surfaceArea(bounds) {
  const dx = bounds.maxX - bounds.minX;
  const dy = bounds.maxY - bounds.minY;
  const dz = bounds.maxZ - bounds.minZ;
  return 2 * (dx * dy + dy * dz + dz * dx);
}

function buildBins(tris, triIndices, positions, axis, cmin, cmax) {
  const bins = Array.from({ length: BIN_COUNT }, () => ({
    count: 0,
    bounds: {
      minX: Infinity,
      minY: Infinity,
      minZ: Infinity,
      maxX: -Infinity,
      maxY: -Infinity,
      maxZ: -Infinity
    }
  }));

  const scale = cmax - cmin > 0 ? BIN_COUNT / (cmax - cmin) : 0;

  for (const triIndex of triIndices) {
    const tri = tris[triIndex];
    const i0 = tri[0] * 3;
    const i1 = tri[1] * 3;
    const i2 = tri[2] * 3;
    const cx = (positions[i0] + positions[i1] + positions[i2]) / 3;
    const cy = (positions[i0 + 1] + positions[i1 + 1] + positions[i2 + 1]) / 3;
    const cz = (positions[i0 + 2] + positions[i1 + 2] + positions[i2 + 2]) / 3;
    const c = axis === 0 ? cx : axis === 1 ? cy : cz;

    let bin = Math.floor((c - cmin) * scale);
    if (bin < 0) bin = 0;
    if (bin >= BIN_COUNT) bin = BIN_COUNT - 1;

    const b = triBounds(positions, tri);
    const binData = bins[bin];
    binData.count += 1;
    binData.bounds.minX = Math.min(binData.bounds.minX, b.minX);
    binData.bounds.minY = Math.min(binData.bounds.minY, b.minY);
    binData.bounds.minZ = Math.min(binData.bounds.minZ, b.minZ);
    binData.bounds.maxX = Math.max(binData.bounds.maxX, b.maxX);
    binData.bounds.maxY = Math.max(binData.bounds.maxY, b.maxY);
    binData.bounds.maxZ = Math.max(binData.bounds.maxZ, b.maxZ);
  }

  return bins;
}

function findBestSplit(tris, triIndices, positions) {
  const cBounds = centroidBounds(tris, triIndices, positions);
  const cmin = [cBounds.minX, cBounds.minY, cBounds.minZ];
  const cmax = [cBounds.maxX, cBounds.maxY, cBounds.maxZ];

  let bestAxis = -1;
  let bestIndex = -1;
  let bestCost = Infinity;

  for (let axis = 0; axis < 3; axis += 1) {
    const bins = buildBins(tris, triIndices, positions, axis, cmin[axis], cmax[axis]);

    const leftBounds = new Array(BIN_COUNT);
    const rightBounds = new Array(BIN_COUNT);
    const leftCount = new Array(BIN_COUNT).fill(0);
    const rightCount = new Array(BIN_COUNT).fill(0);

    let accumBounds = {
      minX: Infinity,
      minY: Infinity,
      minZ: Infinity,
      maxX: -Infinity,
      maxY: -Infinity,
      maxZ: -Infinity
    };
    let accumCount = 0;
    for (let i = 0; i < BIN_COUNT; i += 1) {
      if (bins[i].count > 0) {
        accumBounds = {
          minX: Math.min(accumBounds.minX, bins[i].bounds.minX),
          minY: Math.min(accumBounds.minY, bins[i].bounds.minY),
          minZ: Math.min(accumBounds.minZ, bins[i].bounds.minZ),
          maxX: Math.max(accumBounds.maxX, bins[i].bounds.maxX),
          maxY: Math.max(accumBounds.maxY, bins[i].bounds.maxY),
          maxZ: Math.max(accumBounds.maxZ, bins[i].bounds.maxZ)
        };
      }
      accumCount += bins[i].count;
      leftBounds[i] = { ...accumBounds };
      leftCount[i] = accumCount;
    }

    accumBounds = {
      minX: Infinity,
      minY: Infinity,
      minZ: Infinity,
      maxX: -Infinity,
      maxY: -Infinity,
      maxZ: -Infinity
    };
    accumCount = 0;
    for (let i = BIN_COUNT - 1; i >= 0; i -= 1) {
      if (bins[i].count > 0) {
        accumBounds = {
          minX: Math.min(accumBounds.minX, bins[i].bounds.minX),
          minY: Math.min(accumBounds.minY, bins[i].bounds.minY),
          minZ: Math.min(accumBounds.minZ, bins[i].bounds.minZ),
          maxX: Math.max(accumBounds.maxX, bins[i].bounds.maxX),
          maxY: Math.max(accumBounds.maxY, bins[i].bounds.maxY),
          maxZ: Math.max(accumBounds.maxZ, bins[i].bounds.maxZ)
        };
      }
      accumCount += bins[i].count;
      rightBounds[i] = { ...accumBounds };
      rightCount[i] = accumCount;
    }

    for (let i = 0; i < BIN_COUNT - 1; i += 1) {
      const lCount = leftCount[i];
      const rCount = rightCount[i + 1];
      if (lCount === 0 || rCount === 0) continue;
      const cost = surfaceArea(leftBounds[i]) * lCount + surfaceArea(rightBounds[i + 1]) * rCount;
      if (cost < bestCost) {
        bestCost = cost;
        bestAxis = axis;
        bestIndex = i;
      }
    }
  }

  return { bestAxis, bestIndex, cBounds };
}

function splitTriangles(tris, triIndices, positions, axis, splitPos) {
  const left = [];
  const right = [];

  for (const triIndex of triIndices) {
    const tri = tris[triIndex];
    const i0 = tri[0] * 3;
    const i1 = tri[1] * 3;
    const i2 = tri[2] * 3;
    const cx = (positions[i0] + positions[i1] + positions[i2]) / 3;
    const cy = (positions[i0 + 1] + positions[i1 + 1] + positions[i2 + 1]) / 3;
    const cz = (positions[i0 + 2] + positions[i1 + 2] + positions[i2 + 2]) / 3;
    const c = axis === 0 ? cx : axis === 1 ? cy : cz;
    if (c < splitPos) {
      left.push(triIndex);
    } else {
      right.push(triIndex);
    }
  }

  if (left.length === 0 || right.length === 0) {
    const half = Math.floor(triIndices.length / 2);
    left.splice(0, left.length, ...triIndices.slice(0, half));
    right.splice(0, right.length, ...triIndices.slice(half));
  }

  return { left, right };
}

function buildNode(tris, triIndices, positions, maxLeafSize, depth, maxDepth, nodes) {
  const bounds = boundsFromTris(tris, triIndices, positions);

  if (triIndices.length <= maxLeafSize || depth >= maxDepth) {
    const nodeIndex = nodes.length;
    nodes.push({
      bounds,
      leftFirst: -1,
      primCount: triIndices.length,
      rightChild: -1,
      triIndices: [...triIndices]
    });
    return nodeIndex;
  }

  const { bestAxis, bestIndex, cBounds } = findBestSplit(tris, triIndices, positions);
  if (bestAxis === -1) {
    const nodeIndex = nodes.length;
    nodes.push({
      bounds,
      leftFirst: -1,
      primCount: triIndices.length,
      rightChild: -1,
      triIndices: [...triIndices]
    });
    return nodeIndex;
  }

  const cmin = [cBounds.minX, cBounds.minY, cBounds.minZ][bestAxis];
  const cmax = [cBounds.maxX, cBounds.maxY, cBounds.maxZ][bestAxis];
  const splitPos = cmin + ((bestIndex + 1) / BIN_COUNT) * (cmax - cmin);

  const { left, right } = splitTriangles(tris, triIndices, positions, bestAxis, splitPos);

  const nodeIndex = nodes.length;
  nodes.push({
    bounds,
    leftFirst: -1,
    primCount: 0,
    rightChild: -1,
    triIndices: []
  });

  const leftChild = buildNode(tris, left, positions, maxLeafSize, depth + 1, maxDepth, nodes);
  const rightChild = buildNode(tris, right, positions, maxLeafSize, depth + 1, maxDepth, nodes);

  nodes[nodeIndex].leftFirst = leftChild;
  nodes[nodeIndex].rightChild = rightChild;
  return nodeIndex;
}

export function buildSAHBVH(positions, indices, options = {}) {
  const maxLeafSize = options.maxLeafSize ?? 4;
  const maxDepth = options.maxDepth ?? 32;

  if (indices.length % 3 !== 0) {
    throw new Error("Indices length must be divisible by 3.");
  }

  const triCount = indices.length / 3;
  const tris = new Array(triCount);
  for (let i = 0; i < triCount; i += 1) {
    tris[i] = [indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]];
  }

  const triIndices = Array.from({ length: triCount }, (_, i) => i);
  const nodes = [];
  buildNode(tris, triIndices, positions, maxLeafSize, 0, maxDepth, nodes);

  return {
    nodes,
    tris
  };
}

export function flattenBVH(nodes, tris) {
  const nodeCount = nodes.length;
  const triIndexList = [];

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
      // Leaf node: leftFirst = first triangle index, primCount = count
      nodeFloats[base + 3] = triIndexList.length;
      nodeFloats[base + 7] = node.primCount;
      nodeFloats[base + 8] = 0;
      triIndexList.push(...node.triIndices);
    } else {
      // Internal node: leftFirst = left child, rightChild in texel 2
      nodeFloats[base + 3] = node.leftFirst;
      nodeFloats[base + 7] = 0;
      nodeFloats[base + 8] = node.rightChild;
    }

    nodeFloats[base + 9] = 0;
    nodeFloats[base + 10] = 0;
    nodeFloats[base + 11] = 0;
  }

  const triIndexBuffer = new Uint32Array(triIndexList.length);
  triIndexBuffer.set(triIndexList);

  return {
    nodeBuffer,
    nodeCount,
    triIndexBuffer
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
