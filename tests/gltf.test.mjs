import test from "node:test";
import assert from "node:assert/strict";
import { loadGltfFromText } from "../src/gltf.js";

function makeTriangleGltf() {
  const positions = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0
  ]);
  const indices = new Uint16Array([0, 1, 2]);
  const buffer = new Uint8Array(positions.byteLength + indices.byteLength + 2);
  buffer.set(new Uint8Array(positions.buffer), 0);
  buffer.set(new Uint8Array(indices.buffer), positions.byteLength);
  const base64 = Buffer.from(buffer).toString("base64");

  return JSON.stringify({
    asset: { version: "2.0" },
    buffers: [{ byteLength: buffer.byteLength, uri: `data:application/octet-stream;base64,${base64}` }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
      { buffer: 0, byteOffset: positions.byteLength, byteLength: indices.byteLength }
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
      { bufferView: 1, componentType: 5123, count: 3, type: "SCALAR" }
    ],
    meshes: [
      { primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }
    ],
    nodes: [{ mesh: 0 }],
    scenes: [{ nodes: [0] }],
    scene: 0
  });
}

test("loadGltfFromText loads triangle mesh", async () => {
  const text = makeTriangleGltf();
  const { positions, indices } = await loadGltfFromText(text);
  assert.equal(positions.length, 9);
  assert.equal(indices.length, 3);
  assert.deepEqual(Array.from(indices), [0, 1, 2]);
});
