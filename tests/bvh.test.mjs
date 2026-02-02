import test from "node:test";
import assert from "node:assert/strict";
import { buildSAHBVH, flattenBVH, packGeometry } from "../src/bvh.js";

test("buildSAHBVH creates nodes and flattens", () => {
  const positions = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    1, 1, 0
  ]);
  const indices = new Uint32Array([0, 1, 2, 1, 3, 2]);
  const bvh = buildSAHBVH(positions, indices, { maxLeafSize: 2 });
  assert.ok(bvh.nodes.length >= 1);
  assert.ok(bvh.primitives.length === 2); // 2 triangles

  const flat = flattenBVH(bvh.nodes, bvh.primitives, bvh.triCount, bvh.sphereCount, bvh.cylinderCount);
  assert.ok(flat.nodeBuffer.byteLength > 0);
  assert.ok(flat.primIndexBuffer.length > 0);

  const packed = packGeometry(positions, indices);
  assert.equal(packed.positionVec4.length, (positions.length / 3) * 4);
  assert.equal(packed.triIndexVec4.length, (indices.length / 3) * 4);
});
