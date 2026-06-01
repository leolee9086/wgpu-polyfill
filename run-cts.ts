/**
 * CTS Runner — installs polyfill, registers globals, runs CTS.
 */

// Install polyfill (this sets navigator.gpu + WebGPU error globals)
const { installPolyfill } = await import("./src/index.ts");
installPolyfill();

// Register Impl classes as standard WebGPU names for CTS cleanup tracking
import {
  GPUDeviceImpl, GPUAdapterImpl, GPUBufferImpl, GPUTextureImpl,
  GPUSamplerImpl, GPUCommandEncoderImpl, GPUCommandBufferImpl,
  GPUComputePassEncoderImpl, GPURenderPassEncoderImpl,
  GPURenderPipelineImpl, GPUQuerySetImpl,
  GPURenderBundleImpl, GPURenderBundleEncoderImpl,
  GPUTextureViewImpl,
  GPUBufferUsage, GPUTextureUsage, GPUMapMode,
} from "./src/index.ts";

const globals: Record<string, any> = {
  GPUDevice: GPUDeviceImpl,
  GPUAdapter: GPUAdapterImpl,
  GPUBuffer: GPUBufferImpl,
  GPUTexture: GPUTextureImpl,
  GPUSampler: GPUSamplerImpl,
  GPUCommandEncoder: GPUCommandEncoderImpl,
  GPUCommandBuffer: GPUCommandBufferImpl,
  GPUComputePassEncoder: GPUComputePassEncoderImpl,
  GPURenderPassEncoder: GPURenderPassEncoderImpl,
  GPURenderPipeline: GPURenderPipelineImpl,
  GPUQuerySet: GPUQuerySetImpl,
  GPURenderBundle: GPURenderBundleImpl,
  GPURenderBundleEncoder: GPURenderBundleEncoderImpl,
  GPUTextureView: GPUTextureViewImpl,
  GPUBufferUsage, GPUTextureUsage, GPUMapMode,
};
for (const [name, val] of Object.entries(globals)) {
  (globalThis as any)[name] = val;
}

// CTS checks for src/ in CWD; chdir to CTS root
const realCwd = process.cwd();
process.chdir("D:/dev/cts");

try {
  await import("../cts/out-node/common/runtime/cmdline.js");
} finally {
  process.chdir(realCwd);
}
