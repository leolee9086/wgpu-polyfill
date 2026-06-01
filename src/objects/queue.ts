/**
 * GPUQueue implementation
 */

import { GPUObjectBase } from "./base";
import { getLib, memory, type Pointer } from "../ffi";
import { StructEncoder } from "../structs/encoder";
import { getCallbackRegistry, createHandle } from "../async/callback-registry";
import { pollUntilComplete } from "../async/polling";
import { ptr } from "bun:ffi";

// Buffer storage to prevent GC during FFI calls
const queueBuffers: Uint8Array[] = [];

export function clearQueueBuffers(): void {
  queueBuffers.length = 0;
}

export class GPUQueueImpl extends GPUObjectBase implements GPUQueue {
  readonly __brand = "GPUQueue";
  private _instance: Pointer;

  constructor(handle: Pointer, instance: Pointer, label: string = "") {
    super(handle, label);
    this._instance = instance;
  }

  protected releaseImpl(): void {
    getLib().wgpuQueueRelease(this._handle);
  }

  protected setLabelImpl(label: string): void {
    const encoder = new StructEncoder();
    const { data, length } = encoder.encodeString(label);
    const stringView = encoder.alloc(16);
    const view = new DataView(stringView.buffer.buffer);
    view.setBigUint64(0, BigInt(data), true);
    view.setBigUint64(8, BigInt(length), true);
    getLib().wgpuQueueSetLabel(this._handle, stringView.ptr);
    encoder.freeAll();
  }

  submit(commandBuffers: Iterable<GPUCommandBuffer>): undefined {
    const buffers = Array.from(commandBuffers);
    if (buffers.length === 0) {
      getLib().wgpuQueueSubmit(this._handle, 0, 0 as Pointer);
      return;
    }

    const encoder = new StructEncoder();
    const handles = buffers.map((b) => (b as unknown as { handle: Pointer }).handle);
    const ptrsPtr = encoder.ptrArray(handles);

    getLib().wgpuQueueSubmit(this._handle, handles.length, ptrsPtr);
    encoder.freeAll();
  }

  async onSubmittedWorkDone(): Promise<undefined> {
    const encoder = new StructEncoder();
    const registry = getCallbackRegistry();
    const handle = createHandle<void>();
    const callbackInfoPtr = registry.createQueueWorkDoneCallback(encoder, handle);
    getLib().wgpuQueueOnSubmittedWorkDone(this._handle, callbackInfoPtr);

    await pollUntilComplete(this._instance, handle);
    encoder.freeAll();
    return;
  }

  writeBuffer(
    buffer: GPUBuffer,
    bufferOffset: GPUSize64,
    data: AllowSharedBufferSource,
    dataOffset?: GPUSize64,
    size?: GPUSize64
  ): undefined {
    const bufferHandle = (buffer as unknown as { handle: Pointer }).handle;

    let sourceBuffer: ArrayBuffer | SharedArrayBuffer;
    let byteOffset = 0;
    let byteLength: number;

    if (data instanceof ArrayBuffer || data instanceof SharedArrayBuffer) {
      sourceBuffer = data;
      byteOffset = Number(dataOffset ?? 0);
      byteLength = size !== undefined ? Number(size) : data.byteLength - byteOffset;
    } else {
      // TypedArray or DataView
      sourceBuffer = data.buffer;
      byteOffset = data.byteOffset + Number(dataOffset ?? 0);
      byteLength = size !== undefined ? Number(size) : data.byteLength - Number(dataOffset ?? 0);
    }

    // Copy to regular ArrayBuffer if SharedArrayBuffer
    const arrayBuffer = sourceBuffer instanceof SharedArrayBuffer
      ? new Uint8Array(sourceBuffer).slice(byteOffset, byteOffset + byteLength).buffer
      : sourceBuffer;

    const finalOffset = sourceBuffer instanceof SharedArrayBuffer ? 0 : byteOffset;
    const dataPtr = memory.ptr(new Uint8Array(arrayBuffer, finalOffset, byteLength));
    getLib().wgpuQueueWriteBuffer(
      this._handle,
      bufferHandle,
      BigInt(bufferOffset),
      dataPtr,
      byteLength
    );
  }

  writeTexture(
    destination: GPUTexelCopyTextureInfo,
    data: AllowSharedBufferSource,
    dataLayout: GPUTexelCopyBufferLayout,
    size: GPUExtent3DStrict
  ): undefined {
    // WGPUTexelCopyTextureInfo (28 bytes, aligned to 32):
    // offset 0:  texture (ptr, 8)
    // offset 8:  mipLevel (u32, 4)
    // offset 12: origin.x (u32, 4)
    // offset 16: origin.y (u32, 4)
    // offset 20: origin.z (u32, 4)
    // offset 24: aspect (u32, 4)
    const destBuffer = new Uint8Array(32);
    queueBuffers.push(destBuffer);
    const destView = new DataView(destBuffer.buffer);

    const textureHandle = (destination.texture as unknown as { handle: Pointer }).handle;
    const origin = destination.origin ?? { x: 0, y: 0, z: 0 };
    const originX = Array.isArray(origin) ? origin[0] ?? 0 : origin.x ?? 0;
    const originY = Array.isArray(origin) ? origin[1] ?? 0 : origin.y ?? 0;
    const originZ = Array.isArray(origin) ? origin[2] ?? 0 : origin.z ?? 0;

    const aspectMap: Record<string, number> = {
      "all": 1,
      "stencil-only": 2,
      "depth-only": 3,
    };

    destView.setBigUint64(0, BigInt(textureHandle as unknown as number), true);
    destView.setUint32(8, destination.mipLevel ?? 0, true);
    destView.setUint32(12, originX, true);
    destView.setUint32(16, originY, true);
    destView.setUint32(20, originZ, true);
    destView.setUint32(24, aspectMap[destination.aspect ?? "all"] ?? 1, true);

    // WGPUTexelCopyBufferLayout (16 bytes):
    // offset 0: offset (u64, 8)
    // offset 8: bytesPerRow (u32, 4)
    // offset 12: rowsPerImage (u32, 4)
    const layoutBuffer = new Uint8Array(16);
    queueBuffers.push(layoutBuffer);
    const layoutView = new DataView(layoutBuffer.buffer);

    layoutView.setBigUint64(0, BigInt(dataLayout.offset ?? 0), true);
    layoutView.setUint32(8, dataLayout.bytesPerRow ?? 0, true);
    layoutView.setUint32(12, dataLayout.rowsPerImage ?? 0, true);

    // WGPUExtent3D (12 bytes):
    const sizeBuffer = new Uint8Array(12);
    queueBuffers.push(sizeBuffer);
    const sizeView = new DataView(sizeBuffer.buffer);

    const width = Array.isArray(size) ? size[0] : size.width;
    const height = Array.isArray(size) ? size[1] ?? 1 : size.height ?? 1;
    const depthOrArrayLayers = Array.isArray(size) ? size[2] ?? 1 : size.depthOrArrayLayers ?? 1;

    sizeView.setUint32(0, width, true);
    sizeView.setUint32(4, height, true);
    sizeView.setUint32(8, depthOrArrayLayers, true);

    // Prepare data buffer
    let sourceBuffer: ArrayBuffer | SharedArrayBuffer;
    let byteOffset = 0;
    let byteLength: number;

    if (data instanceof ArrayBuffer || data instanceof SharedArrayBuffer) {
      sourceBuffer = data;
      byteLength = data.byteLength;
    } else {
      // TypedArray or DataView
      sourceBuffer = data.buffer;
      byteOffset = data.byteOffset;
      byteLength = data.byteLength;
    }

    // Copy to regular ArrayBuffer if SharedArrayBuffer
    const arrayBuffer = sourceBuffer instanceof SharedArrayBuffer
      ? new Uint8Array(sourceBuffer).slice(byteOffset, byteOffset + byteLength).buffer
      : sourceBuffer;

    const finalOffset = sourceBuffer instanceof SharedArrayBuffer ? 0 : byteOffset;
    const dataArray = new Uint8Array(arrayBuffer, finalOffset, byteLength);
    queueBuffers.push(dataArray);

    getLib().wgpuQueueWriteTexture(
      this._handle,
      ptr(destBuffer),
      ptr(dataArray),
      byteLength,
      ptr(layoutBuffer),
      ptr(sizeBuffer)
    );

    return undefined;
  }

  copyExternalImageToTexture(
    source: GPUCopyExternalImageSourceInfo,
    destination: GPUCopyExternalImageDestInfo,
    copySize: GPUExtent3DStrict
  ): undefined {
    // Not applicable for headless
    throw new Error("copyExternalImageToTexture not supported in headless mode");
  }
}
