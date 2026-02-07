import test from "node:test";
import assert from "node:assert/strict";
import { PRIM_SPHERE, PRIM_CYLINDER } from "../src/bvh.js";
import { computePrimitiveWorldBounds, projectAabbToCanvasRect } from "../src/overlay_bbox.js";

test("computePrimitiveWorldBounds handles spheres", () => {
  const scene = {
    spheres: [{ center: [2, 3, 4], radius: 1.5 }],
    cylinders: [],
    tris: [],
    positions: new Float32Array(0)
  };
  const bounds = computePrimitiveWorldBounds(scene, PRIM_SPHERE, 0);
  assert.deepEqual(bounds, {
    minX: 0.5,
    minY: 1.5,
    minZ: 2.5,
    maxX: 3.5,
    maxY: 4.5,
    maxZ: 5.5
  });
});

test("computePrimitiveWorldBounds handles cylinders", () => {
  const scene = {
    spheres: [],
    cylinders: [{ p1: [0, 0, 0], p2: [0, 2, 0], radius: 0.5 }],
    tris: [],
    positions: new Float32Array(0)
  };
  const bounds = computePrimitiveWorldBounds(scene, PRIM_CYLINDER, 0);
  assert(Math.abs(bounds.minX + 0.5) < 1e-6);
  assert(Math.abs(bounds.maxX - 0.5) < 1e-6);
  assert(Math.abs(bounds.minY - 0.0) < 1e-6);
  assert(Math.abs(bounds.maxY - 2.0) < 1e-6);
});

test("projectAabbToCanvasRect projects a centered box", () => {
  const camera = {
    origin: [0, 0, 0],
    forward: [0, 0, 1],
    right: [1, 0, 0],
    up: [0, 1, 0]
  };
  const bounds = {
    minX: -1,
    minY: -1,
    minZ: 4,
    maxX: 1,
    maxY: 1,
    maxZ: 4
  };
  const rect = projectAabbToCanvasRect(bounds, camera, 200, 100);
  assert(rect);
  assert(Math.abs(rect.minX - 75) < 1e-6);
  assert(Math.abs(rect.maxX - 125) < 1e-6);
  assert(Math.abs(rect.minY - 37.5) < 1e-6);
  assert(Math.abs(rect.maxY - 62.5) < 1e-6);
});

test("projectAabbToCanvasRect returns null for bounds fully behind camera", () => {
  const camera = {
    origin: [0, 0, 0],
    forward: [0, 0, 1],
    right: [1, 0, 0],
    up: [0, 1, 0]
  };
  const bounds = {
    minX: -1,
    minY: -1,
    minZ: -3,
    maxX: 1,
    maxY: 1,
    maxZ: -1
  };
  const rect = projectAabbToCanvasRect(bounds, camera, 200, 100);
  assert.equal(rect, null);
});
