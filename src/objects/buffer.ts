/**
 * GPUBuffer implementation
 */

import { toArrayBuffer } from "bun:ffi";
import { GPUObjectBase } from "./base";
import { getLib, type Pointer } from "../ffi";
import { StructEncoder } from "../structs/encoder";
import { getCallbackRegistry, createHandle } from "../async/callback-registry";
import { pollUntilComplete } from "../async/polling";
import { WGPUMapMode } from "../ffi/types";
import { GPUMapMode } from "../index";

export class GPUBufferImpl extends GPUObjectBase implements GPUBuffer {
  readonly __brand = "GPUBuffer";
  private _instance: Pointer;
  private _device: Pointer;
  private _size: GPUSize64Out;
  private _usage: GPUFlagsConstant;
  private _mapState: GPUBufferMapState = "unmapped";
  private _mappedRanges: Map<number, ArrayBuffer> = new Map();
  private _mappedAtCreation: boolean;

  constructor(
    handle: Pointer,
    instance: Pointer,
    device: Pointer,
    size: GPUSize64,
    usage: GPUBufferUsageFlags,
    mappedAtCreation: boolean = false,
    label?: string
  ) {
    super(handle, label);
    this._instance = instance;
    this._device = device;
    this._size = size;
    this._usage = usage;
    this._mappedAtCreation = mappedAtCreation;

    if (mappedAtCreation) {
      this._mapState = "mapped";
    }
  }

  protected releaseImpl(): void {
    getLib().wgpuBufferRelease(this._handle);
  }

  protected setLabelImpl(label: string): void {
    const encoder = new StructEncoder();
    const { data, length } = encoder.encodeString(label);
    const stringView = encoder.alloc(16);
    const view = new DataView(stringView.buffer.buffer);
    view.setBigUint64(0, BigInt(data), true);
    view.setBigUint64(8, BigInt(length), true);
    getLib().wgpuBufferSetLabel(this._handle, stringView.ptr);
    encoder.freeAll();
  }

  get size(): GPUSize64Out {
    return this._size;
  }

  get usage(): GPUFlagsConstant {
    return this._usage;
  }

  get mapState(): GPUBufferMapState {
    return this._mapState;
  }

  async mapAsync(mode: GPUMapModeFlags, offset?: GPUSize64, size?: GPUSize64): Promise<undefined> {
    if (this._mapState !== "unmapped") {
      throw new Error("Buffer is already mapped or mapping is pending");
    }

    // Validate map mode against buffer usage flags
    const MAP_READ = 0x0001;
    const MAP_WRITE = 0x0002;
    if ((mode & GPUMapMode.READ) && !(this._usage & MAP_READ)) {
      throw new GPUValidationError("Buffer does not have MAP_READ usage flag");
    }
    if ((mode & GPUMapMode.WRITE) && !(this._usage & MAP_WRITE)) {
      throw new GPUValidationError("Buffer does not have MAP_WRITE usage flag");
    }

    this._mapState = "pending";

    const mapOffset = offset ?? 0;
    const mapSize = size ?? this._size - mapOffset;

    const encoder = new StructEncoder();
    const registry = getCallbackRegistry();

    try {
      const handle = createHandle<void>();
      const callbackInfoPtr = registry.createBufferMapCallback(encoder, handle);

      getLib().wgpuBufferMapAsync(this._handle, mode, 0, this._size, callbackInfoPtr, 0);

      await pollUntilComplete(this._instance, handle);

      this._mapState = "mapped";
    } catch (error) {
      this._mapState = "unmapped";
      throw error;
    } finally {
      encoder.freeAll();
    }

    return;
  }

  getMappedRange(offset?: GPUSize64, size?: GPUSize64): ArrayBuffer {
    if (this._mapState !== "mapped") {
      throw new Error("Buffer is not mapped");
    }

    const rangeOffset = Number(offset ?? 0);
    const rangeSize = Number(size ?? this._size - rangeOffset);

    // Get the mapped range pointer from native
    const ptr = getLib().wgpuBufferGetMappedRange(
      this._handle,
      rangeOffset,
      rangeSize
    );

    if (!ptr) {
      throw new Error("Failed to get mapped range");
    }

    // Create a copy of the native memory as an ArrayBuffer
    // Note: This creates a copy, not a view, because the native memory
    // will become invalid after unmap
    const nativeBuffer = toArrayBuffer(ptr, 0, rangeSize);
    const copy = new ArrayBuffer(rangeSize);
    new Uint8Array(copy).set(new Uint8Array(nativeBuffer));

    this._mappedRanges.set(rangeOffset, copy);

    return copy;
  }

  unmap(): undefined {
    if (this._mapState !== "mapped") {
      return;
    }

    getLib().wgpuBufferUnmap(this._handle);
    this._mapState = "unmapped";
    this._mappedRanges.clear();
    return;
  }

  destroy(): undefined {
    if (this._released) return undefined;
    if (this._mapState === "mapped") {
      this.unmap();
    }
    getLib().wgpuBufferDestroy(this._handle);
    super.destroy();
    return undefined;
  }
}

// Re-export GPUMapMode constants for convenience
export const GPUMapModeConstants = {
  READ: 0x0001,
  WRITE: 0x0002,
} as const;
