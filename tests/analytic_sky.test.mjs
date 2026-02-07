import test from "node:test";
import assert from "node:assert/strict";
import {
  ANALYTIC_SKY_ID,
  analyticSkyCacheKey,
  computeSunDirection,
  normalizeAnalyticSkySettings
} from "../src/analytic_sky.js";

test("analytic sky identifier is stable", () => {
  assert.equal(ANALYTIC_SKY_ID, "analytic://preetham-perez");
});

test("analytic sky cache key changes when settings change", () => {
  const keyA = analyticSkyCacheKey({
    width: 1024,
    height: 512,
    turbidity: 2.5,
    sunAzimuthDeg: 30,
    sunElevationDeg: 35
  });
  const keyB = analyticSkyCacheKey({
    width: 1024,
    height: 512,
    turbidity: 3.0,
    sunAzimuthDeg: 30,
    sunElevationDeg: 35
  });
  assert.notEqual(keyA, keyB);
});

test("analytic sky settings reject invalid ranges", () => {
  assert.throws(
    () => normalizeAnalyticSkySettings({ width: 0, height: 512 }),
    /width/i
  );
  assert.throws(
    () => normalizeAnalyticSkySettings({ turbidity: 50 }),
    /turbidity/i
  );
  assert.throws(
    () => normalizeAnalyticSkySettings({ sunAngularRadiusDeg: -1 }),
    /angular radius/i
  );
});

test("sun direction is normalized", () => {
  const dir = computeSunDirection(45, 25);
  const len = Math.hypot(dir[0], dir[1], dir[2]);
  assert(Math.abs(len - 1) < 1e-6);
});
