function decodeDataUri(uri) {
  if (!uri.startsWith("data:")) {
    throw new Error("Only data: URIs are supported for buffers.");
  }
  const match = uri.match(/^data:.*?;base64,(.*)$/);
  if (!match) {
    throw new Error("Only base64-encoded data URIs are supported.");
  }
  const binary = atob(match[1]);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

function mapObjectToArray(obj, label) {
  if (Array.isArray(obj)) {
    return { array: obj, map: null };
  }
  if (!obj || typeof obj !== "object") {
    throw new Error(`${label} must be an array or object.`);
  }
  const keys = Object.keys(obj);
  const array = keys.map((key) => obj[key]);
  const map = new Map();
  keys.forEach((key, index) => {
    map.set(key, index);
  });
  return { array, map };
}

function resolveIndex(value, map, label) {
  if (value === undefined || value === null) return value;
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    if (!map || !map.has(value)) {
      throw new Error(`Unknown ${label} reference: ${value}`);
    }
    return map.get(value);
  }
  throw new Error(`Invalid ${label} reference type.`);
}

function normalizeGltf(raw) {
  const { array: buffers, map: bufferMap } = mapObjectToArray(raw.buffers, "buffers");
  const { array: bufferViews, map: bufferViewMap } = mapObjectToArray(raw.bufferViews, "bufferViews");
  const { array: accessors, map: accessorMap } = mapObjectToArray(raw.accessors, "accessors");
  const { array: meshes, map: meshMap } = mapObjectToArray(raw.meshes, "meshes");
  const { array: nodes, map: nodeMap } = mapObjectToArray(raw.nodes, "nodes");
  const { array: scenes, map: sceneMap } = mapObjectToArray(raw.scenes, "scenes");

  accessors.forEach((accessor) => {
    accessor.bufferView = resolveIndex(accessor.bufferView, bufferViewMap, "bufferView");
  });

  bufferViews.forEach((view) => {
    view.buffer = resolveIndex(view.buffer, bufferMap, "buffer");
  });

  meshes.forEach((mesh) => {
    mesh.primitives.forEach((primitive) => {
      Object.keys(primitive.attributes).forEach((key) => {
        primitive.attributes[key] = resolveIndex(
          primitive.attributes[key],
          accessorMap,
          `accessor (${key})`
        );
      });
      if (primitive.indices !== undefined) {
        primitive.indices = resolveIndex(primitive.indices, accessorMap, "indices accessor");
      }
    });
  });

  nodes.forEach((node) => {
    if (node.mesh !== undefined) {
      node.mesh = resolveIndex(node.mesh, meshMap, "mesh");
    }
    if (node.meshes) {
      node.meshes = node.meshes.map((mesh) => resolveIndex(mesh, meshMap, "mesh"));
    }
    if (node.children) {
      node.children = node.children.map((child) => resolveIndex(child, nodeMap, "node"));
    }
  });

  scenes.forEach((scene) => {
    if (scene.nodes) {
      scene.nodes = scene.nodes.map((node) => resolveIndex(node, nodeMap, "node"));
    }
  });

  const sceneIndex = resolveIndex(raw.scene ?? 0, sceneMap, "scene") ?? 0;

  return {
    buffers,
    bufferViews,
    accessors,
    meshes,
    nodes,
    scenes,
    scene: sceneIndex
  };
}

async function loadBuffers(gltf, baseUrl, fetchFn) {
  const buffers = [];
  for (const buffer of gltf.buffers) {
    if (!buffer.uri) {
      throw new Error("Buffer missing uri.");
    }
    if (buffer.uri.startsWith("data:")) {
      buffers.push(decodeDataUri(buffer.uri));
    } else {
      if (!baseUrl) {
        throw new Error("External buffer URI requires a base URL.");
      }
      const url = new URL(buffer.uri, baseUrl).toString();
      const res = await fetchFn(url);
      if (!res.ok) {
        throw new Error(`Failed to fetch buffer: ${url}`);
      }
      buffers.push(await res.arrayBuffer());
    }
  }
  return buffers;
}

function getAccessorData(gltf, accessorIndex, buffers) {
  const accessor = gltf.accessors[accessorIndex];
  const view = gltf.bufferViews[accessor.bufferView];
  const buffer = buffers[view.buffer];
  const byteOffset = (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const count = accessor.count;
  const componentType = accessor.componentType;
  const type = accessor.type;

  if (type === "VEC3" && componentType !== 5126) {
    throw new Error("POSITION accessors must be Float32 (componentType 5126).");
  }

  let components = 1;
  if (type === "SCALAR") components = 1;
  if (type === "VEC2") components = 2;
  if (type === "VEC3") components = 3;
  if (type === "VEC4") components = 4;

  const componentSize = componentType === 5126 || componentType === 5125 ? 4 : 2;
  const stride = view.byteStride && view.byteStride > 0 ? view.byteStride : components * componentSize;

  const slice = buffer.slice(byteOffset, byteOffset + view.byteLength);
  return { accessor, slice, components, byteStride: stride, count, componentType };
}

function readAccessorToArray(gltf, accessorIndex, buffers) {
  const { slice, components, byteStride, count, componentType } = getAccessorData(
    gltf,
    accessorIndex,
    buffers
  );

  if (componentType === 5126) {
    const out = new Float32Array(count * components);
    const view = new DataView(slice);
    for (let i = 0; i < count; i += 1) {
      for (let c = 0; c < components; c += 1) {
        out[i * components + c] = view.getFloat32(i * byteStride + c * 4, true);
      }
    }
    return out;
  }

  if (componentType === 5123) {
    const out = new Uint16Array(count * components);
    const view = new DataView(slice);
    for (let i = 0; i < count; i += 1) {
      for (let c = 0; c < components; c += 1) {
        out[i * components + c] = view.getUint16(i * byteStride + c * 2, true);
      }
    }
    return out;
  }

  if (componentType === 5125) {
    const out = new Uint32Array(count * components);
    const view = new DataView(slice);
    for (let i = 0; i < count; i += 1) {
      for (let c = 0; c < components; c += 1) {
        out[i * components + c] = view.getUint32(i * byteStride + c * 4, true);
      }
    }
    return out;
  }

  throw new Error(`Unsupported accessor componentType: ${componentType}`);
}

function mat4Identity() {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

function mat4Multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      out[r * 4 + c] =
        a[r * 4 + 0] * b[0 * 4 + c] +
        a[r * 4 + 1] * b[1 * 4 + c] +
        a[r * 4 + 2] * b[2 * 4 + c] +
        a[r * 4 + 3] * b[3 * 4 + c];
    }
  }
  return out;
}

function mat4FromTRS(translation, rotation, scale) {
  const [x, y, z, w] = rotation;
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;

  const sx = scale[0];
  const sy = scale[1];
  const sz = scale[2];

  return [
    (1 - 2 * (yy + zz)) * sx,
    (2 * (xy + wz)) * sx,
    (2 * (xz - wy)) * sx,
    0,
    (2 * (xy - wz)) * sy,
    (1 - 2 * (xx + zz)) * sy,
    (2 * (yz + wx)) * sy,
    0,
    (2 * (xz + wy)) * sz,
    (2 * (yz - wx)) * sz,
    (1 - 2 * (xx + yy)) * sz,
    0,
    translation[0],
    translation[1],
    translation[2],
    1
  ];
}

function transformPoint(m, p) {
  const x = p[0];
  const y = p[1];
  const z = p[2];
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14]
  ];
}

function resolveNodeMatrix(node) {
  if (node.matrix) {
    return node.matrix;
  }
  const translation = node.translation || [0, 0, 0];
  const rotation = node.rotation || [0, 0, 0, 1];
  const scale = node.scale || [1, 1, 1];
  return mat4FromTRS(translation, rotation, scale);
}

function traverseNodes(gltf, nodeIndex, parentMatrix, meshes) {
  const node = gltf.nodes[nodeIndex];
  const local = resolveNodeMatrix(node);
  const world = mat4Multiply(parentMatrix, local);

  if (node.mesh !== undefined) {
    meshes.push({ mesh: gltf.meshes[node.mesh], matrix: world });
  }
  if (node.meshes) {
    for (const meshIndex of node.meshes) {
      meshes.push({ mesh: gltf.meshes[meshIndex], matrix: world });
    }
  }

  if (node.children) {
    for (const child of node.children) {
      traverseNodes(gltf, child, world, meshes);
    }
  }
}

export async function loadGltfFromText(text, baseUrl = null, fetchFn = fetch) {
  const raw = JSON.parse(text);
  if (!raw.buffers || !raw.bufferViews || !raw.accessors) {
    throw new Error("glTF is missing buffers/bufferViews/accessors.");
  }
  const gltf = normalizeGltf(raw);
  if (!gltf.meshes || gltf.meshes.length === 0) {
    throw new Error("glTF has no meshes.");
  }

  const buffers = await loadBuffers(gltf, baseUrl, fetchFn);
  const meshes = [];
  const scene = gltf.scenes[gltf.scene ?? 0];
  if (!scene || !scene.nodes) {
    throw new Error("glTF has no default scene nodes.");
  }

  for (const nodeIndex of scene.nodes) {
    traverseNodes(gltf, nodeIndex, mat4Identity(), meshes);
  }

  const positions = [];
  const indices = [];
  let vertexOffset = 0;

  for (const entry of meshes) {
    for (const primitive of entry.mesh.primitives) {
      if (primitive.attributes.POSITION === undefined) {
        throw new Error("Primitive missing POSITION attribute.");
      }

      const posArray = readAccessorToArray(gltf, primitive.attributes.POSITION, buffers);
      const transformed = new Float32Array(posArray.length);
      for (let i = 0; i < posArray.length; i += 3) {
        const p = transformPoint(entry.matrix, [posArray[i], posArray[i + 1], posArray[i + 2]]);
        transformed[i] = p[0];
        transformed[i + 1] = p[1];
        transformed[i + 2] = p[2];
      }
      positions.push(transformed);

      if (primitive.indices !== undefined) {
        const idxArray = readAccessorToArray(gltf, primitive.indices, buffers);
        for (let i = 0; i < idxArray.length; i += 1) {
          indices.push(Number(idxArray[i]) + vertexOffset);
        }
      } else {
        const triCount = transformed.length / 3;
        for (let i = 0; i < triCount; i += 1) {
          indices.push(vertexOffset + i);
        }
      }

      vertexOffset += transformed.length / 3;
    }
  }

  const mergedPositions = new Float32Array(vertexOffset * 3);
  let pOffset = 0;
  for (const chunk of positions) {
    mergedPositions.set(chunk, pOffset);
    pOffset += chunk.length;
  }

  return {
    positions: mergedPositions,
    indices: new Uint32Array(indices)
  };
}
