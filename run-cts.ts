/**
 * CTS Runner — installs polyfill, then runs CTS command-line.
 *
 * Usage (from wgpu-polyfill/):
 *   bun run run-cts.ts 'webgpu:api,validation,error_scope:*'
 *   bun run run-cts.ts --verbose 'webgpu:api,operation,buffer,map:*'
 */

// Install polyfill (import TS source directly to keep lib/ path correct)
const { installPolyfill } = await import("./src/index.ts");
installPolyfill(); // Sets navigator.gpu automatically

// CTS checks for src/ in CWD; temporarily chdir to CTS root
const realCwd = process.cwd();
process.chdir("D:/dev/cts");

try {
  await import("../cts/out-node/common/runtime/cmdline.js");
} finally {
  process.chdir(realCwd);
}
