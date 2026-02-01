import test from "node:test";
import assert from "node:assert/strict";
import { packBvhNodes, packTriangles, packTriIndices, computeLayoutForTests } from "../src/packing.js";

const simpleNodes = [
  {
    bounds: { minX: 0, minY: 0, minZ: 0, maxX: 1, maxY: 1, maxZ: 1 },
    leftFirst: 0,
    primCount: 1,
    rightChild: 0
  }
];

const tris = [[0, 1, 2]];
const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);


test("compute layout respects max size", () => {
  const layout = computeLayoutForTests(10, 8, 4);
  assert.equal(layout.width, 4);
  assert.equal(layout.height, 3);
});

test("packBvhNodes encodes node texels", () => {
  const packed = packBvhNodes(simpleNodes, 64);
  assert.equal(packed.width * packed.height * 4, packed.data.length);
  assert.equal(packed.texelsPerNode, 3);
  assert.equal(packed.data[0], 0);
  assert.equal(packed.data[3], 0);
  assert.equal(packed.data[7], 1);
});

test("packTriangles expands triangles", () => {
  const packed = packTriangles(tris, positions, 64);
  assert.equal(packed.texelsPerTri, 3);
  assert.equal(packed.data[0], 0);
  assert.equal(packed.data[4], 1);
  assert.equal(packed.data[8], 0);
});

test("packTriIndices packs indices into texels", () => {
  const packed = packTriIndices(new Uint32Array([2, 5, 7]), 64);
  assert.equal(packed.data[0], 2);
  assert.equal(packed.data[4], 5);
  assert.equal(packed.data[8], 7);
});
