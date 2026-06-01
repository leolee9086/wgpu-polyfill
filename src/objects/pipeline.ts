/**
 * GPUPipelineLayout and GPUComputePipeline implementations
 */

import { GPUObjectBase } from "./base";
import { getLib, type Pointer } from "../ffi";
import { StructEncoder } from "../structs/encoder";
import { GPUBindGroupLayoutImpl } from "./bind-group";

export class GPUPipelineLayoutImpl extends GPUObjectBase implements GPUPipelineLayout {
  readonly __brand = "GPUPipelineLayout";
  /** The immediate data size configured for this layout (0 if none). */
  readonly immediateSize: number;

  constructor(handle: Pointer, label?: string, immediateSize = 0) {
    super(handle, label);
    this.immediateSize = immediateSize;
  }

  protected releaseImpl(): void {
    getLib().wgpuPipelineLayoutRelease(this._handle);
  }

  protected setLabelImpl(label: string): void {
    const encoder = new StructEncoder();
    const { data, length } = encoder.encodeString(label);
    const stringView = encoder.alloc(16);
    const view = new DataView(stringView.buffer.buffer);
    view.setBigUint64(0, BigInt(data), true);
    view.setBigUint64(8, BigInt(length), true);
    getLib().wgpuPipelineLayoutSetLabel(this._handle, stringView.ptr);
    encoder.freeAll();
  }
}

export class GPUComputePipelineImpl extends GPUObjectBase implements GPUComputePipeline {
  readonly __brand = "GPUComputePipeline";

  constructor(handle: Pointer, label?: string) {
    super(handle, label);
  }

  protected releaseImpl(): void {
    getLib().wgpuComputePipelineRelease(this._handle);
  }

  protected setLabelImpl(label: string): void {
    const encoder = new StructEncoder();
    const { data, length } = encoder.encodeString(label);
    const stringView = encoder.alloc(16);
    const view = new DataView(stringView.buffer.buffer);
    view.setBigUint64(0, BigInt(data), true);
    view.setBigUint64(8, BigInt(length), true);
    getLib().wgpuComputePipelineSetLabel(this._handle, stringView.ptr);
    encoder.freeAll();
  }

  getBindGroupLayout(index: number): GPUBindGroupLayout {
    const layoutHandle = getLib().wgpuComputePipelineGetBindGroupLayout(this._handle, index);
    if (!layoutHandle) {
      throw new Error(`No bind group layout at index ${index}`);
    }
    return new GPUBindGroupLayoutImpl(layoutHandle) as unknown as GPUBindGroupLayout;
  }
}

export class GPURenderPipelineImpl extends GPUObjectBase implements GPURenderPipeline {
  readonly __brand = "GPURenderPipeline";

  constructor(handle: Pointer, label?: string) {
    super(handle, label);
  }

  protected releaseImpl(): void {
    getLib().wgpuRenderPipelineRelease(this._handle);
  }

  protected setLabelImpl(label: string): void {
    const encoder = new StructEncoder();
    const { data, length } = encoder.encodeString(label);
    const stringView = encoder.alloc(16);
    const view = new DataView(stringView.buffer.buffer);
    view.setBigUint64(0, BigInt(data), true);
    view.setBigUint64(8, BigInt(length), true);
    getLib().wgpuRenderPipelineSetLabel(this._handle, stringView.ptr);
    encoder.freeAll();
  }

  getBindGroupLayout(index: number): GPUBindGroupLayout {
    const layoutHandle = getLib().wgpuRenderPipelineGetBindGroupLayout(this._handle, index);
    if (!layoutHandle) {
      throw new Error(`No bind group layout at index ${index}`);
    }
    return new GPUBindGroupLayoutImpl(layoutHandle) as unknown as GPUBindGroupLayout;
  }
}
