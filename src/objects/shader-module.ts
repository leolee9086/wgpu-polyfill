/**
 * GPUShaderModule implementation
 */

import { GPUObjectBase } from "./base";
import { getLib, type Pointer } from "../ffi";
import { StructEncoder } from "../structs/encoder";

/**
 * Entry point info parsed from WGSL source.
 */
export interface EntryPointInfo {
  name: string;
  stage: "vertex" | "fragment" | "compute";
}

/**
 * Parse entry points from WGSL source code.
 * Matches patterns like:
 *   @vertex fn main(...) -> ...
 *   @fragment fn frag_main(...)
 *   @compute fn compute_main(...)
 *
 * Handles:
 * - Whitespace and newlines between attribute and fn
 * - Multiple attributes per function
 * - Attributes with arguments like @binding(0) @group(0)
 */
export function parseEntryPoints(wgsl: string): EntryPointInfo[] {
  const entries: EntryPointInfo[] = [];
  // Match @vertex/@fragment/@compute followed by whitespace and fn <name>(
  const regex = /@(vertex|fragment|compute)[\s\S]*?\bfn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(wgsl)) !== null) {
    const stage = match[1] as "vertex" | "fragment" | "compute";
    const name = match[2];
    // Avoid duplicates
    if (!entries.some(e => e.name === name && e.stage === stage)) {
      entries.push({ name, stage });
    }
  }
  return entries;
}

export class GPUShaderModuleImpl extends GPUObjectBase implements GPUShaderModule {
  readonly __brand = "GPUShaderModule";
  private _instance: Pointer;
  /** Entry points parsed from WGSL source, used for entry point validation. */
  readonly entryPoints: EntryPointInfo[];
  /** Immediate data size in bytes (0 if not using var<immediate>). */
  readonly immediateDataSize: number;

  constructor(handle: Pointer, instance: Pointer, label?: string, entryPoints?: EntryPointInfo[], immediateDataSize = 0) {
    super(handle, label);
    this._instance = instance;
    this.entryPoints = entryPoints ?? [];
    this.immediateDataSize = immediateDataSize;
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
