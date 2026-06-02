/**
 * GPUShaderModule implementation
 */

import { GPUObjectBase } from "./base";
import { getLib, type Pointer } from "../ffi";
import { StructEncoder } from "../structs/encoder";
import type { EntryPointInfo } from "../wgsl/parser";
import { parseEntryPoints } from "../wgsl/parser";

export class GPUShaderModuleImpl extends GPUObjectBase implements GPUShaderModule {
  readonly __brand = "GPUShaderModule";
  private _instance: Pointer;
  /** The device that created this shader module (for cross-device validation). */
  readonly devicePtr: Pointer;
  /** Entry points parsed from WGSL source, used for entry point validation. */
  readonly entryPoints: EntryPointInfo[];
  /** Immediate data size in bytes (0 if not using var<immediate>). */
  readonly immediateDataSize: number;
  /** Whether the shader references @builtin(frag_depth). */
  readonly hasFragDepth: boolean;
  /** Whether the shader outputs @builtin(sample_mask). */
  readonly hasSampleMask: boolean;
  /** The original WGSL source code (used for vertex/fragment input validation). */
  readonly code: string;
  /** True if this is an error/invalid shader module. */
  readonly isError: boolean;

  constructor(handle: Pointer, instance: Pointer, devicePtr: Pointer, label?: string, entryPoints?: EntryPointInfo[], immediateDataSize = 0, hasFragDepth = false, code = "", hasSampleMask = false, isError = false) {
    super(handle, label);
    this._instance = instance;
    this.devicePtr = devicePtr;
    this.entryPoints = entryPoints ?? [];
    this.immediateDataSize = immediateDataSize;
    this.hasFragDepth = hasFragDepth;
    this.hasSampleMask = hasSampleMask;
    this.code = code;
    this.isError = isError;
  }

  protected releaseImpl(): void {
    getLib().wgpuShaderModuleRelease(this._handle);
  }

  protected setLabelImpl(label: string): void {
    const encoder = new StructEncoder();
    const { data, length } = encoder.encodeString(label);
    const stringView = encoder.alloc(16);
    const view = new DataView(stringView.buffer.buffer);
    view.setBigUint64(0, BigInt(data), true);
    view.setBigUint64(8, BigInt(length), true);
    getLib().wgpuShaderModuleSetLabel(this._handle, stringView.ptr);
    encoder.freeAll();
  }

  async getCompilationInfo(): Promise<GPUCompilationInfo> {
    // Note: wgpuShaderModuleGetCompilationInfo is not implemented in wgpu-native
    // Return empty compilation info (which is what native WebGPU does for valid shaders)
    // If the shader had errors, createShaderModule would have thrown already
    return {
      __brand: "GPUCompilationInfo",
      messages: [],
    } as GPUCompilationInfo;
  }
}
