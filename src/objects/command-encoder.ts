/**
 * GPUCommandEncoder and GPUCommandBuffer implementation
 */

import { ptr } from "bun:ffi";
import { GPUObjectBase } from "./base";
import { getLib, type Pointer } from "../ffi";
import { StructEncoder } from "../structs/encoder";
import { WGPUCommandBufferDescriptor, WGPUComputePassDescriptor } from "../structs/definitions/buffer";
import { WGPU_STRLEN } from "../ffi/types";
import { getTextureFormat } from "./texture";

// Global buffer storage to prevent GC issues
const encoderBuffers: Uint8Array[] = [];

// Load ops
const WGPULoadOp = {
  Undefined: 0,
  Clear: 1,
  Load: 2,
} as const;

// Store ops
const WGPUStoreOp = {
  Undefined: 0,
  Store: 1,
  Discard: 2,
} as const;

export class GPUCommandBufferImpl extends GPUObjectBase implements GPUCommandBuffer {
  readonly __brand = "GPUCommandBuffer";

  constructor(handle: Pointer, label?: string) {
    super(handle, label);
  }

  protected releaseImpl(): void {
    getLib().wgpuCommandBufferRelease(this._handle);
  }

  protected setLabelImpl(label: string): void {
    const encoder = new StructEncoder();
    const { data, length } = encoder.encodeString(label);
    const stringView = encoder.alloc(16);
    const view = new DataView(stringView.buffer.buffer);
    view.setBigUint64(0, BigInt(data), true);
    view.setBigUint64(8, BigInt(length), true);
    getLib().wgpuCommandBufferSetLabel(this._handle, stringView.ptr);
    encoder.freeAll();
  }
}

export class GPUCommandEncoderImpl extends GPUObjectBase implements GPUCommandEncoder {
  readonly __brand = "GPUCommandEncoder";
  private _instance: Pointer;

  constructor(handle: Pointer, instance: Pointer, label?: string) {
    super(handle, label);
    this._instance = instance;
  }

  protected releaseImpl(): void {
    getLib().wgpuCommandEncoderRelease(this._handle);
  }

  protected setLabelImpl(label: string): void {
    const encoder = new StructEncoder();
    const { data, length } = encoder.encodeString(label);
    const stringView = encoder.alloc(16);
    const view = new DataView(stringView.buffer.buffer);
    view.setBigUint64(0, BigInt(data), true);
    view.setBigUint64(8, BigInt(length), true);
    getLib().wgpuCommandEncoderSetLabel(this._handle, stringView.ptr);
    encoder.freeAll();
  }

  beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder {
    const encoder = new StructEncoder();
    let descPtr: Pointer | null = null;

    if (descriptor) {
      const labelStr = descriptor.label ? encoder.encodeString(descriptor.label) : { data: 0, length: 0 };
      descPtr = encoder.encode(WGPUComputePassDescriptor, {
        nextInChain: 0,
        label: { data: labelStr.data, length: labelStr.length },
        timestampWrites: 0,
      }).ptr;
    }

    const passHandle = getLib().wgpuCommandEncoderBeginComputePass(this._handle, descPtr);
    encoder.freeAll();

    const { GPUComputePassEncoderImpl } = require("./compute-pass");
    return new GPUComputePassEncoderImpl(passHandle, descriptor?.label) as unknown as GPUComputePassEncoder;
  }

  beginRenderPass(descriptor: GPURenderPassDescriptor): GPURenderPassEncoder {
    // WGPURenderPassColorAttachment (72 bytes):
    // offset 0:  nextInChain (ptr, 8)
    // offset 8:  view (ptr, 8)
    // offset 16: depthSlice (u32, 4)
    // offset 20: padding (4)
    // offset 24: resolveTarget (ptr, 8)
    // offset 32: loadOp (u32, 4)
    // offset 36: storeOp (u32, 4)
    // offset 40: clearValue.r (f64, 8)
    // offset 48: clearValue.g (f64, 8)
    // offset 56: clearValue.b (f64, 8)
    // offset 64: clearValue.a (f64, 8)
    // Total: 72 bytes

    const colorAttachments = Array.from(descriptor.colorAttachments);
    const colorAttachmentSize = 72;
    const colorAttachmentsBuffer = new Uint8Array(colorAttachments.length * colorAttachmentSize);
    encoderBuffers.push(colorAttachmentsBuffer);

    for (let i = 0; i < colorAttachments.length; i++) {
      const attachment = colorAttachments[i];
      if (!attachment) continue;

      const offset = i * colorAttachmentSize;
      const view = new DataView(colorAttachmentsBuffer.buffer, offset, colorAttachmentSize);

      view.setBigUint64(0, BigInt(0), true); // nextInChain

      const textureViewHandle = attachment.view
        ? (attachment.view as unknown as { handle: Pointer }).handle
        : 0;
      view.setBigUint64(8, BigInt(textureViewHandle as unknown as number), true);

      view.setUint32(16, 0xFFFFFFFF, true); // depthSlice = WGPU_DEPTH_SLICE_UNDEFINED
      view.setUint32(20, 0, true); // padding

      const resolveTargetHandle = attachment.resolveTarget
        ? (attachment.resolveTarget as unknown as { handle: Pointer }).handle
        : 0;
      view.setBigUint64(24, BigInt(resolveTargetHandle as unknown as number), true);

      // Map load/store ops
      const loadOpMap: Record<string, number> = {
        "clear": WGPULoadOp.Clear,
        "load": WGPULoadOp.Load,
      };
      const storeOpMap: Record<string, number> = {
        "store": WGPUStoreOp.Store,
        "discard": WGPUStoreOp.Discard,
      };

      view.setUint32(32, loadOpMap[attachment.loadOp] ?? WGPULoadOp.Clear, true);
      view.setUint32(36, storeOpMap[attachment.storeOp] ?? WGPUStoreOp.Store, true);

      // Clear color
      const clearValue = attachment.clearValue;
      if (clearValue) {
        if (Array.isArray(clearValue)) {
          view.setFloat64(40, clearValue[0] ?? 0, true);
          view.setFloat64(48, clearValue[1] ?? 0, true);
          view.setFloat64(56, clearValue[2] ?? 0, true);
          view.setFloat64(64, clearValue[3] ?? 1, true);
        } else {
          view.setFloat64(40, clearValue.r ?? 0, true);
          view.setFloat64(48, clearValue.g ?? 0, true);
          view.setFloat64(56, clearValue.b ?? 0, true);
          view.setFloat64(64, clearValue.a ?? 1, true);
        }
      } else {
        view.setFloat64(40, 0, true);
        view.setFloat64(48, 0, true);
        view.setFloat64(56, 0, true);
        view.setFloat64(64, 1, true);
      }
    }

    const colorAttachmentsPtr = ptr(colorAttachmentsBuffer) as unknown as number;

    // Encode depth/stencil attachment if present
    // WGPURenderPassDepthStencilAttachment (48 bytes in v29):
    // offset 0:  nextInChain (ptr, 8) — NEW in v29!
    // offset 8:  view (ptr, 8)
    // offset 16: depthLoadOp (u32, 4)
    // offset 20: depthStoreOp (u32, 4)
    // offset 24: depthClearValue (f32, 4)
    // offset 28: depthReadOnly (u32, 4)
    // offset 32: stencilLoadOp (u32, 4)
    // offset 36: stencilStoreOp (u32, 4)
    // offset 40: stencilClearValue (u32, 4)
    // offset 44: stencilReadOnly (u32, 4)
    // Total: 48 bytes

    let depthStencilAttachmentPtr = 0;
    if (descriptor.depthStencilAttachment) {
      const dsAttachment = descriptor.depthStencilAttachment;
      const dsBuffer = new Uint8Array(48);
      encoderBuffers.push(dsBuffer);
      const dsView = new DataView(dsBuffer.buffer);

      dsView.setBigUint64(0, BigInt(0), true); // nextInChain = null

      const dsTextureViewHandle = dsAttachment.view
        ? (dsAttachment.view as unknown as { handle: Pointer }).handle
        : 0;
      dsView.setBigUint64(8, BigInt(dsTextureViewHandle as unknown as number), true);

      // Map load/store ops for depth
      const loadOpMap: Record<string, number> = {
        "clear": WGPULoadOp.Clear,
        "load": WGPULoadOp.Load,
      };
      const storeOpMap: Record<string, number> = {
        "store": WGPUStoreOp.Store,
        "discard": WGPUStoreOp.Discard,
      };

      // depthLoadOp - use Undefined (0) if not specified
      dsView.setUint32(16, dsAttachment.depthLoadOp ? loadOpMap[dsAttachment.depthLoadOp] ?? 0 : 0, true);
      // depthStoreOp - use Undefined (0) if not specified
      dsView.setUint32(20, dsAttachment.depthStoreOp ? storeOpMap[dsAttachment.depthStoreOp] ?? 0 : 0, true);
      // depthClearValue
      dsView.setFloat32(24, dsAttachment.depthClearValue ?? 1.0, true);
      // depthReadOnly
      dsView.setUint32(28, dsAttachment.depthReadOnly ? 1 : 0, true);

      // stencilLoadOp - use Undefined (0) if not specified
      dsView.setUint32(32, dsAttachment.stencilLoadOp ? loadOpMap[dsAttachment.stencilLoadOp] ?? 0 : 0, true);
      // stencilStoreOp - use Undefined (0) if not specified
      dsView.setUint32(36, dsAttachment.stencilStoreOp ? storeOpMap[dsAttachment.stencilStoreOp] ?? 0 : 0, true);
      // stencilClearValue
      dsView.setUint32(40, dsAttachment.stencilClearValue ?? 0, true);
      // stencilReadOnly
      dsView.setUint32(44, dsAttachment.stencilReadOnly ? 1 : 0, true);

      depthStencilAttachmentPtr = ptr(dsBuffer) as unknown as number;
    }

    // WGPURenderPassDescriptor (64 bytes):
    // offset 0:  nextInChain (ptr, 8)
    // offset 8:  label.data (ptr, 8)
    // offset 16: label.length (size_t, 8)
    // offset 24: colorAttachmentCount (size_t, 8)
    // offset 32: colorAttachments (ptr, 8)
    // offset 40: depthStencilAttachment (ptr, 8)
    // offset 48: occlusionQuerySet (ptr, 8)
    // offset 56: timestampWrites (ptr, 8)

    const desc = new Uint8Array(64);
    encoderBuffers.push(desc);
    const descView = new DataView(desc.buffer);

    descView.setBigUint64(0, BigInt(0), true);                    // nextInChain
    descView.setBigUint64(8, BigInt(0), true);                    // label.data
    descView.setBigUint64(16, WGPU_STRLEN, true);                 // label.length
    descView.setBigUint64(24, BigInt(colorAttachments.length), true); // colorAttachmentCount
    descView.setBigUint64(32, BigInt(colorAttachmentsPtr), true); // colorAttachments
    descView.setBigUint64(40, BigInt(depthStencilAttachmentPtr), true); // depthStencilAttachment
    descView.setBigUint64(48, BigInt(0), true);                   // occlusionQuerySet
    descView.setBigUint64(56, BigInt(0), true);                   // timestampWrites

    const descPtr = ptr(desc);
    const passHandle = getLib().wgpuCommandEncoderBeginRenderPass(this._handle, descPtr);

    if (!passHandle) {
      throw new Error("Failed to begin render pass");
    }

    const { GPURenderPassEncoderImpl } = require("./render-pass");
    return new GPURenderPassEncoderImpl(passHandle, descriptor.label) as unknown as GPURenderPassEncoder;
  }

  // Overloaded method to match WebGPU spec
  copyBufferToBuffer(
    source: GPUBuffer,
    sourceOffsetOrDestination: GPUSize64 | GPUBuffer,
    destinationOrSize?: GPUBuffer | GPUSize64,
    destinationOffset?: GPUSize64,
    size?: GPUSize64
  ): undefined {
    const srcHandle = (source as unknown as { handle: Pointer }).handle;

    // Detect which overload is being used
    if (typeof sourceOffsetOrDestination === "number" || typeof sourceOffsetOrDestination === "bigint") {
      // Full form: copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size?)
      const sourceOffset = sourceOffsetOrDestination;
      const destination = destinationOrSize as GPUBuffer;
      const dstOffset = destinationOffset ?? 0;
      const copySize = size ?? 0; // 0 means copy full buffer
      const dstHandle = (destination as unknown as { handle: Pointer }).handle;

      getLib().wgpuCommandEncoderCopyBufferToBuffer(
        this._handle,
        srcHandle,
        BigInt(sourceOffset),
        dstHandle,
        BigInt(dstOffset),
        BigInt(copySize)
      );
    } else {
      // Short form: copyBufferToBuffer(source, destination, size?)
      const destination = sourceOffsetOrDestination as GPUBuffer;
      const copySize = (destinationOrSize as GPUSize64) ?? 0;
      const dstHandle = (destination as unknown as { handle: Pointer }).handle;

      getLib().wgpuCommandEncoderCopyBufferToBuffer(
        this._handle,
        srcHandle,
        BigInt(0),
        dstHandle,
        BigInt(0),
        BigInt(copySize)
      );
    }
    return undefined;
  }

  copyBufferToTexture(
    source: GPUTexelCopyBufferInfo,
    destination: GPUTexelCopyTextureInfo,
    copySize: GPUExtent3DStrict
  ): undefined {
    // WGPUTexelCopyBufferInfo (24 bytes):
    // offset 0:  layout.offset (u64, 8)
    // offset 8:  layout.bytesPerRow (u32, 4)
    // offset 12: layout.rowsPerImage (u32, 4)
    // offset 16: buffer (ptr, 8)
    // Total: 24 bytes

    const sourceBuffer = new Uint8Array(24);
    encoderBuffers.push(sourceBuffer);
    const sourceView = new DataView(sourceBuffer.buffer);

    const bufferHandle = (source.buffer as unknown as { handle: Pointer }).handle;
    sourceView.setBigUint64(0, BigInt(source.offset ?? 0), true);
    sourceView.setUint32(8, source.bytesPerRow ?? 0, true);
    sourceView.setUint32(12, source.rowsPerImage ?? 0, true);
    sourceView.setBigUint64(16, BigInt(bufferHandle as unknown as number), true);

    // WGPUTexelCopyTextureInfo (24 bytes):
    // offset 0:  texture (ptr, 8)
    // offset 8:  mipLevel (u32, 4)
    // offset 12: origin.x (u32, 4)
    // offset 16: origin.y (u32, 4)
    // offset 20: origin.z (u32, 4)
    // offset 24: aspect (u32, 4)
    // Total: 28 bytes (aligned to 32)

    const destBuffer = new Uint8Array(32);
    encoderBuffers.push(destBuffer);
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

    // WGPUExtent3D (12 bytes):
    const sizeBuffer = new Uint8Array(12);
    encoderBuffers.push(sizeBuffer);
    const sizeView = new DataView(sizeBuffer.buffer);

    const width = Array.isArray(copySize) ? copySize[0] : copySize.width;
    const height = Array.isArray(copySize) ? copySize[1] ?? 1 : copySize.height ?? 1;
    const depthOrArrayLayers = Array.isArray(copySize) ? copySize[2] ?? 1 : copySize.depthOrArrayLayers ?? 1;

    sizeView.setUint32(0, width, true);
    sizeView.setUint32(4, height, true);
    sizeView.setUint32(8, depthOrArrayLayers, true);

    getLib().wgpuCommandEncoderCopyBufferToTexture(
      this._handle,
      ptr(sourceBuffer),
      ptr(destBuffer),
      ptr(sizeBuffer)
    );
    return undefined;
  }

  copyTextureToBuffer(
    source: GPUTexelCopyTextureInfo,
    destination: GPUTexelCopyBufferInfo,
    copySize: GPUExtent3DStrict
  ): undefined {
    // WGPUTexelCopyTextureInfo (28 bytes, aligned to 32):
    const sourceBuffer = new Uint8Array(32);
    encoderBuffers.push(sourceBuffer);
    const sourceView = new DataView(sourceBuffer.buffer);

    const textureHandle = (source.texture as unknown as { handle: Pointer }).handle;
    const origin = source.origin ?? { x: 0, y: 0, z: 0 };
    const originX = Array.isArray(origin) ? origin[0] ?? 0 : origin.x ?? 0;
    const originY = Array.isArray(origin) ? origin[1] ?? 0 : origin.y ?? 0;
    const originZ = Array.isArray(origin) ? origin[2] ?? 0 : origin.z ?? 0;

    const aspectMap: Record<string, number> = {
      "all": 1,
      "stencil-only": 2,
      "depth-only": 3,
    };

    sourceView.setBigUint64(0, BigInt(textureHandle as unknown as number), true);
    sourceView.setUint32(8, source.mipLevel ?? 0, true);
    sourceView.setUint32(12, originX, true);
    sourceView.setUint32(16, originY, true);
    sourceView.setUint32(20, originZ, true);
    sourceView.setUint32(24, aspectMap[source.aspect ?? "all"] ?? 1, true);

    // WGPUTexelCopyBufferInfo (24 bytes):
    const destBuffer = new Uint8Array(24);
    encoderBuffers.push(destBuffer);
    const destView = new DataView(destBuffer.buffer);

    const bufferHandle = (destination.buffer as unknown as { handle: Pointer }).handle;
    destView.setBigUint64(0, BigInt(destination.offset ?? 0), true);
    destView.setUint32(8, destination.bytesPerRow ?? 0, true);
    destView.setUint32(12, destination.rowsPerImage ?? 0, true);
    destView.setBigUint64(16, BigInt(bufferHandle as unknown as number), true);

    // WGPUExtent3D (12 bytes):
    const sizeBuffer = new Uint8Array(12);
    encoderBuffers.push(sizeBuffer);
    const sizeView = new DataView(sizeBuffer.buffer);

    const width = Array.isArray(copySize) ? copySize[0] : copySize.width;
    const height = Array.isArray(copySize) ? copySize[1] ?? 1 : copySize.height ?? 1;
    const depthOrArrayLayers = Array.isArray(copySize) ? copySize[2] ?? 1 : copySize.depthOrArrayLayers ?? 1;

    sizeView.setUint32(0, width, true);
    sizeView.setUint32(4, height, true);
    sizeView.setUint32(8, depthOrArrayLayers, true);

    getLib().wgpuCommandEncoderCopyTextureToBuffer(
      this._handle,
      ptr(sourceBuffer),
      ptr(destBuffer),
      ptr(sizeBuffer)
    );
    return undefined;
  }

  copyTextureToTexture(
    source: GPUTexelCopyTextureInfo,
    destination: GPUTexelCopyTextureInfo,
    copySize: GPUExtent3DStrict
  ): undefined {
    // WGPUTexelCopyTextureInfo (28 bytes, aligned to 32):
    const sourceBuffer = new Uint8Array(32);
    encoderBuffers.push(sourceBuffer);
    const sourceView = new DataView(sourceBuffer.buffer);

    const srcTextureHandle = (source.texture as unknown as { handle: Pointer }).handle;
    const srcOrigin = source.origin ?? { x: 0, y: 0, z: 0 };
    const srcOriginX = Array.isArray(srcOrigin) ? srcOrigin[0] ?? 0 : srcOrigin.x ?? 0;
    const srcOriginY = Array.isArray(srcOrigin) ? srcOrigin[1] ?? 0 : srcOrigin.y ?? 0;
    const srcOriginZ = Array.isArray(srcOrigin) ? srcOrigin[2] ?? 0 : srcOrigin.z ?? 0;

    const aspectMap: Record<string, number> = {
      "all": 1,
      "stencil-only": 2,
      "depth-only": 3,
    };

    sourceView.setBigUint64(0, BigInt(srcTextureHandle as unknown as number), true);
    sourceView.setUint32(8, source.mipLevel ?? 0, true);
    sourceView.setUint32(12, srcOriginX, true);
    sourceView.setUint32(16, srcOriginY, true);
    sourceView.setUint32(20, srcOriginZ, true);
    sourceView.setUint32(24, aspectMap[source.aspect ?? "all"] ?? 1, true);

    const destBuffer = new Uint8Array(32);
    encoderBuffers.push(destBuffer);
    const destView = new DataView(destBuffer.buffer);

    const dstTextureHandle = (destination.texture as unknown as { handle: Pointer }).handle;
    const dstOrigin = destination.origin ?? { x: 0, y: 0, z: 0 };
    const dstOriginX = Array.isArray(dstOrigin) ? dstOrigin[0] ?? 0 : dstOrigin.x ?? 0;
    const dstOriginY = Array.isArray(dstOrigin) ? dstOrigin[1] ?? 0 : dstOrigin.y ?? 0;
    const dstOriginZ = Array.isArray(dstOrigin) ? dstOrigin[2] ?? 0 : dstOrigin.z ?? 0;

    destView.setBigUint64(0, BigInt(dstTextureHandle as unknown as number), true);
    destView.setUint32(8, destination.mipLevel ?? 0, true);
    destView.setUint32(12, dstOriginX, true);
    destView.setUint32(16, dstOriginY, true);
    destView.setUint32(20, dstOriginZ, true);
    destView.setUint32(24, aspectMap[destination.aspect ?? "all"] ?? 1, true);

    // WGPUExtent3D (12 bytes):
    const sizeBuffer = new Uint8Array(12);
    encoderBuffers.push(sizeBuffer);
    const sizeView = new DataView(sizeBuffer.buffer);

    const width = Array.isArray(copySize) ? copySize[0] : copySize.width;
    const height = Array.isArray(copySize) ? copySize[1] ?? 1 : copySize.height ?? 1;
    const depthOrArrayLayers = Array.isArray(copySize) ? copySize[2] ?? 1 : copySize.depthOrArrayLayers ?? 1;

    sizeView.setUint32(0, width, true);
    sizeView.setUint32(4, height, true);
    sizeView.setUint32(8, depthOrArrayLayers, true);

    getLib().wgpuCommandEncoderCopyTextureToTexture(
      this._handle,
      ptr(sourceBuffer),
      ptr(destBuffer),
      ptr(sizeBuffer)
    );
    return undefined;
  }

  clearBuffer(buffer: GPUBuffer, offset?: GPUSize64, size?: GPUSize64): undefined {
    const bufferHandle = (buffer as unknown as { handle: Pointer }).handle;
    const clearOffset = BigInt(offset ?? 0);
    const clearSize = BigInt(size ?? 0xFFFFFFFFFFFFFFFFn); // WGPU_WHOLE_SIZE

    getLib().wgpuCommandEncoderClearBuffer(this._handle, bufferHandle, clearOffset, clearSize);
  }

  resolveQuerySet(
    querySet: GPUQuerySet,
    firstQuery: GPUSize32,
    queryCount: GPUSize32,
    destination: GPUBuffer,
    destinationOffset: GPUSize64
  ): undefined {
    const querySetHandle = (querySet as unknown as { handle: Pointer }).handle;
    const dstHandle = (destination as unknown as { handle: Pointer }).handle;

    getLib().wgpuCommandEncoderResolveQuerySet(
      this._handle,
      querySetHandle,
      firstQuery,
      queryCount,
      dstHandle,
      BigInt(destinationOffset)
    );
  }

  finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer {
    const encoder = new StructEncoder();
    let descPtr: Pointer | null = null;

    if (descriptor) {
      const labelStr = descriptor.label ? encoder.encodeString(descriptor.label) : { data: 0, length: 0 };
      descPtr = encoder.encode(WGPUCommandBufferDescriptor, {
        nextInChain: 0,
        label: { data: labelStr.data, length: labelStr.length },
      }).ptr;
    }

    const bufferHandle = getLib().wgpuCommandEncoderFinish(this._handle, descPtr);
    encoder.freeAll();

    if (!bufferHandle) {
      throw new Error("Failed to finish command encoder");
    }

    return new GPUCommandBufferImpl(bufferHandle, descriptor?.label) as unknown as GPUCommandBuffer;
  }

  pushDebugGroup(groupLabel: string): undefined {
    const encoder = new StructEncoder();
    const { data, length } = encoder.encodeString(groupLabel);
    const stringView = encoder.alloc(16);
    const view = new DataView(stringView.buffer.buffer);
    view.setBigUint64(0, BigInt(data), true);
    view.setBigUint64(8, BigInt(length), true);
    getLib().wgpuCommandEncoderPushDebugGroup(this._handle, stringView.ptr);
    encoder.freeAll();
  }

  popDebugGroup(): undefined {
    getLib().wgpuCommandEncoderPopDebugGroup(this._handle);
  }

  insertDebugMarker(markerLabel: string): undefined {
    const encoder = new StructEncoder();
    const { data, length } = encoder.encodeString(markerLabel);
    const stringView = encoder.alloc(16);
    const view = new DataView(stringView.buffer.buffer);
    view.setBigUint64(0, BigInt(data), true);
    view.setBigUint64(8, BigInt(length), true);
    getLib().wgpuCommandEncoderInsertDebugMarker(this._handle, stringView.ptr);
    encoder.freeAll();
  }
}

/**
 * Clear encoder buffers to free memory
 */
export function clearEncoderBuffers(): void {
  encoderBuffers.length = 0;
}
