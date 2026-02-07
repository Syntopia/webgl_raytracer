import test from "node:test";
import assert from "node:assert/strict";
import { __test__mapMaterialMode } from "../src/webgl.js";

test("material mode mapping includes translucent plastic", () => {
  assert.equal(__test__mapMaterialMode("metallic"), 0);
  assert.equal(__test__mapMaterialMode("matte"), 1);
  assert.equal(__test__mapMaterialMode("surface-glass"), 2);
  assert.equal(__test__mapMaterialMode("translucent-plastic"), 3);
  assert.equal(__test__mapMaterialMode(3), 3);
});
