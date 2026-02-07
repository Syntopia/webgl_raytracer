import test from "node:test";
import assert from "node:assert/strict";
import { PRIM_SPHERE } from "../src/bvh.js";
import { traceSceneRay } from "../src/ray_pick.js";

function makeSingleSphereScene() {
  return {
    positions: new Float32Array(0),
    tris: [],
    spheres: [{ center: [0, 0, 0], radius: 1.0 }],
    cylinders: [],
    primitives: [{ type: PRIM_SPHERE, index: 0 }],
    nodes: [
      {
        bounds: { minX: -1, minY: -1, minZ: -1, maxX: 1, maxY: 1, maxZ: 1 },
        leftFirst: -1,
        rightChild: -1,
        primCount: 1,
        primIndices: [0]
      }
    ]
  };
}

test("traceSceneRay hits sphere and returns nearest t", () => {
  const scene = makeSingleSphereScene();
  const hit = traceSceneRay(scene, [0, 0, -5], [0, 0, 1], { tMin: 1e-6 });
  assert(hit);
  assert.equal(hit.primIndex, 0);
  assert(Math.abs(hit.t - 4.0) < 1e-6);
});

test("traceSceneRay respects clipping plane rejection", () => {
  const scene = makeSingleSphereScene();
  const hit = traceSceneRay(scene, [0, 0, -5], [0, 0, 1], {
    tMin: 1e-6,
    clip: {
      enabled: true,
      normal: [0, 0, 1],
      offset: -2.0,
      side: 1.0
    }
  });
  assert.equal(hit, null);
});

test("traceSceneRay returns null on miss", () => {
  const scene = makeSingleSphereScene();
  const hit = traceSceneRay(scene, [0, 0, -5], [0, 1, 0], { tMin: 1e-6 });
  assert.equal(hit, null);
});
