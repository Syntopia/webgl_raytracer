function computeTextureLayout(texelCount, maxSize, preferredWidth = 1024) {
  if (texelCount === 0) {
    return { width: 1, height: 1 };
  }
  const width = Math.min(maxSize, Math.max(1, Math.min(preferredWidth, texelCount)));
  const height = Math.ceil(texelCount / width);
  if (height > maxSize) {
    throw new Error(`Texture size ${width}x${height} exceeds MAX_TEXTURE_SIZE ${maxSize}.`);
  }
  return { width, height };
}

function writeTexel(data, texelIndex, rgba) {
  const base = texelIndex * 4;
  data[base + 0] = rgba[0];
  data[base + 1] = rgba[1];
  data[base + 2] = rgba[2];
  data[base + 3] = rgba[3];
}

export function packBvhNodes(nodes, maxTextureSize) {
  const texelsPerNode = 3;
  const texelCount = nodes.length * texelsPerNode;
  const layout = computeTextureLayout(texelCount, maxTextureSize);
  const data = new Float32Array(layout.width * layout.height * 4);

  // First pass: compute leftFirst for leaf nodes (index into primIndexList)
  let primIndexOffset = 0;
  const leafFirstIndices = new Array(nodes.length);
  for (let i = 0; i < nodes.length; i += 1) {
    const node = nodes[i];
    if (node.primCount > 0) {
      leafFirstIndices[i] = primIndexOffset;
      primIndexOffset += node.primIndices.length;
    }
  }

  for (let i = 0; i < nodes.length; i += 1) {
    const node = nodes[i];
    const base = i * texelsPerNode;
    const primCount = node.primCount;
    const rightChild = node.rightChild;

    // For leaf nodes, leftFirst is the starting index in the primitive index list
    // For internal nodes, leftFirst is the left child node index
    const leftFirst = primCount > 0 ? leafFirstIndices[i] : node.leftFirst;

    writeTexel(data, base + 0, [node.bounds.minX, node.bounds.minY, node.bounds.minZ, leftFirst]);
    writeTexel(data, base + 1, [node.bounds.maxX, node.bounds.maxY, node.bounds.maxZ, primCount]);
    writeTexel(data, base + 2, [rightChild, 0, 0, 0]);
  }

  return {
    data,
    width: layout.width,
    height: layout.height,
    texelsPerNode
  };
}

export function packTriangles(tris, positions, maxTextureSize) {
  const texelsPerTri = 3;
  const triCount = tris.length;
  const texelCount = Math.max(1, triCount * texelsPerTri);
  const layout = computeTextureLayout(texelCount, maxTextureSize);
  const data = new Float32Array(layout.width * layout.height * 4);

  for (let i = 0; i < triCount; i += 1) {
    const tri = tris[i];
    const i0 = tri[0] * 3;
    const i1 = tri[1] * 3;
    const i2 = tri[2] * 3;

    const v0 = [positions[i0], positions[i0 + 1], positions[i0 + 2], 0];
    const v1 = [positions[i1], positions[i1 + 1], positions[i1 + 2], 0];
    const v2 = [positions[i2], positions[i2 + 1], positions[i2 + 2], 0];

    const base = i * texelsPerTri;
    writeTexel(data, base + 0, v0);
    writeTexel(data, base + 1, v1);
    writeTexel(data, base + 2, v2);
  }

  return {
    data,
    width: layout.width,
    height: layout.height,
    texelsPerTri
  };
}

export function packTriNormals(tris, normals, maxTextureSize) {
  const texelsPerTri = 3;
  const triCount = tris.length;
  const texelCount = Math.max(1, triCount * texelsPerTri);
  const layout = computeTextureLayout(texelCount, maxTextureSize);
  const data = new Float32Array(layout.width * layout.height * 4);

  for (let i = 0; i < triCount; i += 1) {
    const tri = tris[i];
    const i0 = tri[0] * 3;
    const i1 = tri[1] * 3;
    const i2 = tri[2] * 3;

    const n0 = [normals[i0], normals[i0 + 1], normals[i0 + 2], 0];
    const n1 = [normals[i1], normals[i1 + 1], normals[i1 + 2], 0];
    const n2 = [normals[i2], normals[i2 + 1], normals[i2 + 2], 0];

    const base = i * texelsPerTri;
    writeTexel(data, base + 0, n0);
    writeTexel(data, base + 1, n1);
    writeTexel(data, base + 2, n2);
  }

  return {
    data,
    width: layout.width,
    height: layout.height,
    texelsPerTri
  };
}

export function packTriColors(triColors, maxTextureSize) {
  const triCount = Math.max(1, triColors.length / 3);
  const texelCount = triCount;
  const layout = computeTextureLayout(texelCount, maxTextureSize);
  const data = new Float32Array(layout.width * layout.height * 4);
  for (let i = 0; i < triColors.length / 3; i += 1) {
    const base = i * 3;
    writeTexel(data, i, [triColors[base], triColors[base + 1], triColors[base + 2], 1]);
  }
  return {
    data,
    width: layout.width,
    height: layout.height,
    texelsPerTri: 1
  };
}

export function packPrimIndices(primIndexBuffer, maxTextureSize) {
  const texelCount = Math.max(1, primIndexBuffer.length);
  const layout = computeTextureLayout(texelCount, maxTextureSize);
  const data = new Float32Array(layout.width * layout.height * 4);
  // Create a view to reinterpret float32 as uint32 bits
  const dataView = new DataView(data.buffer);
  for (let i = 0; i < primIndexBuffer.length; i += 1) {
    // Store the uint32 bits directly in the float32's bit pattern
    // This preserves the exact bit pattern for floatBitsToInt in the shader
    const byteOffset = i * 16; // 4 floats per texel, 4 bytes per float
    dataView.setUint32(byteOffset, primIndexBuffer[i], true); // little-endian
    // Other components remain 0
  }
  return {
    data,
    width: layout.width,
    height: layout.height,
    texelsPerIndex: 1
  };
}

// Legacy alias
export function packTriIndices(triIndexBuffer, maxTextureSize) {
  return packPrimIndices(triIndexBuffer, maxTextureSize);
}

/**
 * Pack spheres into a texture
 * Each sphere uses 1 texel: [centerX, centerY, centerZ, radius]
 */
export function packSpheres(spheres, maxTextureSize) {
  const texelCount = Math.max(1, spheres.length);
  const layout = computeTextureLayout(texelCount, maxTextureSize);
  const data = new Float32Array(layout.width * layout.height * 4);

  for (let i = 0; i < spheres.length; i += 1) {
    const s = spheres[i];
    writeTexel(data, i, [s.center[0], s.center[1], s.center[2], s.radius]);
  }

  return {
    data,
    width: layout.width,
    height: layout.height,
    count: spheres.length
  };
}

/**
 * Pack sphere colors into a texture
 * Each sphere uses 1 texel: [r, g, b, 1]
 */
export function packSphereColors(spheres, maxTextureSize) {
  const texelCount = Math.max(1, spheres.length);
  const layout = computeTextureLayout(texelCount, maxTextureSize);
  const data = new Float32Array(layout.width * layout.height * 4);

  for (let i = 0; i < spheres.length; i += 1) {
    const s = spheres[i];
    const c = s.color || [0.8, 0.8, 0.8];
    writeTexel(data, i, [c[0], c[1], c[2], 1]);
  }

  return {
    data,
    width: layout.width,
    height: layout.height,
    count: spheres.length
  };
}

/**
 * Pack cylinders into a texture
 * Each cylinder uses 2 texels:
 *   Texel 0: [p1.x, p1.y, p1.z, radius]
 *   Texel 1: [p2.x, p2.y, p2.z, 0]
 */
export function packCylinders(cylinders, maxTextureSize) {
  const texelsPerCyl = 2;
  const texelCount = Math.max(1, cylinders.length * texelsPerCyl);
  const layout = computeTextureLayout(texelCount, maxTextureSize);
  const data = new Float32Array(layout.width * layout.height * 4);

  for (let i = 0; i < cylinders.length; i += 1) {
    const c = cylinders[i];
    const base = i * texelsPerCyl;
    writeTexel(data, base + 0, [c.p1[0], c.p1[1], c.p1[2], c.radius]);
    writeTexel(data, base + 1, [c.p2[0], c.p2[1], c.p2[2], 0]);
  }

  return {
    data,
    width: layout.width,
    height: layout.height,
    count: cylinders.length,
    texelsPerCyl
  };
}

/**
 * Pack cylinder colors into a texture
 * Each cylinder uses 1 texel: [r, g, b, 1]
 */
export function packCylinderColors(cylinders, maxTextureSize) {
  const texelCount = Math.max(1, cylinders.length);
  const layout = computeTextureLayout(texelCount, maxTextureSize);
  const data = new Float32Array(layout.width * layout.height * 4);

  for (let i = 0; i < cylinders.length; i += 1) {
    const c = cylinders[i];
    const col = c.color || [0.8, 0.8, 0.8];
    writeTexel(data, i, [col[0], col[1], col[2], 1]);
  }

  return {
    data,
    width: layout.width,
    height: layout.height,
    count: cylinders.length
  };
}

export function computeLayoutForTests(texelCount, maxSize, preferredWidth) {
  return computeTextureLayout(texelCount, maxSize, preferredWidth);
}
