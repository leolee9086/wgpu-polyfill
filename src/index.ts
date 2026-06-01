/**
 * WebGPU Polyfill for Bun
 *
 * Provides WebGPU implementation using wgpu-native for headless testing.
 */

import { GPUImpl } from "./gpu";
import { clearAllBuffers as clearAllBuffersFn } from "./cleanup";

// WebGPU error types (globals, per spec)
export class GPUValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GPUValidationError";
  }
}

export class GPUOutOfMemoryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GPUOutOfMemoryError";
  }
}

export class GPUInternalError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GPUInternalError";
  }
}

// Register as globals for CTS compatibility
(globalThis as any).GPUValidationError = GPUValidationError;
(globalThis as any).GPUOutOfMemoryError = GPUOutOfMemoryError;
(globalThis as any).GPUInternalError = GPUInternalError;

// Re-export types for convenience
export type { Pointer } from "./ffi/types";
export { GPUImpl } from "./gpu";
export { GPUAdapterImpl } from "./objects/adapter";
export { GPUDeviceImpl } from "./objects/device";
export { GPUQueueImpl } from "./objects/queue";
export { GPUBufferImpl } from "./objects/buffer";
export { GPUCommandEncoderImpl, GPUCommandBufferImpl } from "./objects/command-encoder";
export { GPUComputePassEncoderImpl } from "./objects/compute-pass";
export { GPURenderPassEncoderImpl } from "./objects/render-pass";
export { GPUTextureImpl, GPUTextureViewImpl } from "./objects/texture";
export { GPURenderPipelineImpl } from "./objects/pipeline";
export { GPUSamplerImpl } from "./objects/sampler";
export { GPUQuerySetImpl } from "./objects/query-set";
export { GPURenderBundleImpl, GPURenderBundleEncoderImpl } from "./objects/render-bundle";
export { clearAllBuffers } from "./cleanup";

// Global GPU instance
let globalGPU: GPUImpl | null = null;

/**
 * Install the WebGPU polyfill
 *
 * Sets up navigator.gpu with our wgpu-native implementation.
 */
export function installPolyfill(): GPU {
  if (globalGPU) {
    return globalGPU as unknown as GPU;
  }

  globalGPU = new GPUImpl();

  // Install on navigator if it exists
  if (typeof navigator !== "undefined") {
    Object.defineProperty(navigator, "gpu", {
      value: globalGPU,
      writable: false,
      configurable: true,
    });
  }

  // Also set on globalThis for environments without navigator
  Object.defineProperty(globalThis, "gpu", {
    value: globalGPU,
    writable: false,
    configurable: true,
  });

  return globalGPU as unknown as GPU;
}

/**
 * Get the GPU instance without installing to navigator
 *
 * Useful for explicit usage without global side effects.
 */
export function getGPU(): GPU {
  if (!globalGPU) {
    globalGPU = new GPUImpl();
  }
  return globalGPU as unknown as GPU;
}

/**
 * Uninstall the polyfill and clean up resources
 */
export function uninstallPolyfill(): void {
  // Clear all temporary buffers
  clearAllBuffersFn();

  if (globalGPU) {
    globalGPU.destroy();
    globalGPU = null;
  }

  // Remove from navigator
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    delete (navigator as unknown as Record<string, unknown>).gpu;
  }

  // Remove from globalThis
  if ("gpu" in globalThis) {
    delete (globalThis as Record<string, unknown>).gpu;
  }
}

// Export constants matching WebGPU spec
export const GPUBufferUsage = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
} as const;

export const GPUTextureUsage = {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08,
  RENDER_ATTACHMENT: 0x10,
} as const;

export const GPUShaderStage = {
  VERTEX: 0x1,
  FRAGMENT: 0x2,
  COMPUTE: 0x4,
} as const;

export const GPUMapMode = {
  READ: 0x0001,
  WRITE: 0x0002,
} as const;

export const GPUColorWrite = {
  RED: 0x1,
  GREEN: 0x2,
  BLUE: 0x4,
  ALPHA: 0x8,
  ALL: 0xf,
} as const;
