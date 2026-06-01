/**
 * GPU implementation - navigator.gpu entry point
 */

import { ptr } from "bun:ffi";
import { GPUAdapterImpl } from "./objects/adapter";
import { getLib, type Pointer } from "./ffi";
import { StructEncoder } from "./structs/encoder";
import { getCallbackRegistry, createHandle } from "./async/callback-registry";
import { pollUntilComplete } from "./async/polling";
import { WGPUInstanceDescriptor } from "./structs/definitions/common";
import { WGPURequestAdapterOptions } from "./structs/definitions/adapter";

export class GPUImpl implements GPU {
  readonly __brand = "GPU";
  private _instance: Pointer | null;
  private _wgslLanguageFeatures: WGSLLanguageFeatures;

  constructor() {
    const encoder = new StructEncoder();

    // Create instance with default capabilities
    const descPtr = encoder.encode(WGPUInstanceDescriptor, {
      nextInChain: 0,
      features: {
        nextInChain: 0,
        timedWaitAnyEnable: 0,
        timedWaitAnyMaxCount: 0,
      },
    }).ptr;

    this._instance = getLib().wgpuCreateInstance(descPtr);
    encoder.freeAll();

    if (!this._instance) {
      throw new Error("Failed to create WebGPU instance");
    }

    // WGSL language features (empty set for now)
    this._wgslLanguageFeatures = new Set() as WGSLLanguageFeatures;
  }

  get wgslLanguageFeatures(): WGSLLanguageFeatures {
    return this._wgslLanguageFeatures;
  }

  async requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null> {
    if (!this._instance) {
      return null;
    }

    const encoder = new StructEncoder();
    const registry = getCallbackRegistry();

    try {
      // Map power preference
      let powerPref = 0; // Undefined
      if (options?.powerPreference === "low-power") powerPref = 1;
      else if (options?.powerPreference === "high-performance") powerPref = 2;

      // Map feature level (core by default)
      let featureLevel = 2; // Core
      if ((options as Record<string, unknown>)?.featureLevel === "compatibility") featureLevel = 1;

      const optionsPtr = encoder.encode(WGPURequestAdapterOptions, {
        nextInChain: 0,
        featureLevel,
        powerPreference: powerPref,
        forceFallbackAdapter: options?.forceFallbackAdapter ? 1 : 0,
        backendType: 0, // Undefined - let wgpu choose
        compatibleSurface: 0, // No surface for headless
      }).ptr;

      const handle = createHandle<Pointer>();
      const callbackInfoPtr = registry.createAdapterCallback(encoder, handle);

      getLib().wgpuInstanceRequestAdapter(this._instance, optionsPtr, callbackInfoPtr);

      const adapterHandle = await pollUntilComplete(this._instance, handle);

      return new GPUAdapterImpl(adapterHandle, this._instance) as unknown as GPUAdapter;
    } catch (error) {
      // Return null on adapter request failure (per spec)
      console.error("Failed to request adapter:", error);
      return null;
    } finally {
      encoder.freeAll();
    }
  }

  getPreferredCanvasFormat(): GPUTextureFormat {
    // For headless, return a common format
    return "bgra8unorm";
  }

  /**
   * Release the WebGPU instance
   */
  destroy(): void {
    if (this._instance) {
      getLib().wgpuInstanceRelease(this._instance);
      this._instance = null;
    }
  }
}
