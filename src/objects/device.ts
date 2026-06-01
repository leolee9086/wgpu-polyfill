/**
 * GPUDevice implementation
 */

import { GPUObjectBase } from "./base";
import { GPUQueueImpl } from "./queue";
import { getLib, type Pointer } from "../ffi";
import { StructEncoder } from "../structs/encoder";
import { WGPUBufferDescriptor, WGPUCommandEncoderDescriptor } from "../structs/definitions/buffer";
import {
  WGPUShaderModuleDescriptor,
  WGPUShaderSourceWGSL,
  WGPUComputePipelineDescriptor,
  WGPUBindGroupLayoutDescriptor,
  WGPUBindGroupDescriptor,
  WGPUPipelineLayoutDescriptor,
  WGPUBindGroupEntry,
} from "../structs/definitions/shader";
import { WGPUSType, WGPU_STRLEN } from "../ffi/types";
import { ptr } from "bun:ffi";
import { GPUQuerySetImpl, createQuerySetDescriptor } from "./query-set";
import { GPURenderBundleEncoderImpl, createRenderBundleEncoderDescriptor } from "./render-bundle";
import { getCallbackRegistry, createHandle } from "../async/callback-registry";
import { pollUntilComplete } from "../async/polling";
import {
  validateBufferSize,
  validateBufferUsage,
  validateTextureUsage,
  validateTextureDimensions,
  validateHandle,
} from "../validation";

// Global buffer storage to prevent GC issues
const shaderBuffers: Uint8Array[] = [];
const pipelineBuffers: Uint8Array[] = [];

// WGPUFeatureName to GPUFeatureName mapping
const FEATURE_NAME_MAP: Record<number, GPUFeatureName> = {
  0x01: "depth-clip-control",
  0x02: "depth32float-stencil8",
  0x03: "timestamp-query",
  0x04: "texture-compression-bc",
  0x05: "texture-compression-bc-sliced-3d",
  0x06: "texture-compression-etc2",
  0x07: "texture-compression-astc",
  0x08: "indirect-first-instance",
  0x09: "shader-f16",
  0x0A: "rg11b10ufloat-renderable",
  0x0B: "bgra8unorm-storage",
  0x0C: "float32-filterable",
  0x0D: "float32-blendable",
  0x0E: "clip-distances",
  0x0F: "dual-source-blending",
};

type EventListenerEntry = {
  listener: EventListenerOrEventListenerObject;
  options?: AddEventListenerOptions;
};

export class GPUDeviceImpl extends GPUObjectBase implements GPUDevice {
  readonly __brand = "GPUDevice";
  private _queue: GPUQueueImpl;
  private _instance: Pointer;
  private _lost: Promise<GPUDeviceLostInfo>;
  private _lostResolve!: (info: GPUDeviceLostInfo) => void;
  private _features: GPUSupportedFeatures;
  private _limits: GPUSupportedLimits;
  private _adapterInfo: GPUAdapterInfo;
  private _eventListeners: Map<string, EventListenerEntry[]> = new Map();

  constructor(handle: Pointer, instance: Pointer, label: string = "", adapterInfo?: GPUAdapterInfo) {
    super(handle, label);
    this._instance = instance;

    // Get the queue
    const queueHandle = getLib().wgpuDeviceGetQueue(handle);
    if (!queueHandle) {
      throw new Error("Failed to get device queue");
    }
    this._queue = new GPUQueueImpl(queueHandle, instance);

    // Set up lost promise
    this._lost = new Promise((resolve) => {
      this._lostResolve = resolve;
    });

    // Query actual features and limits from device
    this._features = this.queryFeatures();
    this._limits = this.queryLimits();
    this._adapterInfo = adapterInfo ?? {
      __brand: "GPUAdapterInfo",
      vendor: "",
      architecture: "",
      device: "",
      description: "",
      isFallbackAdapter: false,
    } as GPUAdapterInfo;
  }

  private queryLimits(): GPUSupportedLimits {
    // WGPULimits struct layout (152 bytes)
    const limitsBuffer = new Uint8Array(152);
    const limitsView = new DataView(limitsBuffer.buffer);
    limitsView.setBigUint64(0, BigInt(0), true); // nextInChain = null

    const status = getLib().wgpuDeviceGetLimits(this._handle, ptr(limitsBuffer));

    if (status !== 1) {
      return this.createDefaultLimits();
    }

    return {
      __brand: "GPUSupportedLimits",
      maxTextureDimension1D: limitsView.getUint32(8, true),
      maxTextureDimension2D: limitsView.getUint32(12, true),
      maxTextureDimension3D: limitsView.getUint32(16, true),
      maxTextureArrayLayers: limitsView.getUint32(20, true),
      maxBindGroups: limitsView.getUint32(24, true),
      maxBindGroupsPlusVertexBuffers: limitsView.getUint32(28, true),
      maxBindingsPerBindGroup: limitsView.getUint32(32, true),
      maxDynamicUniformBuffersPerPipelineLayout: limitsView.getUint32(36, true),
      maxDynamicStorageBuffersPerPipelineLayout: limitsView.getUint32(40, true),
      maxSampledTexturesPerShaderStage: limitsView.getUint32(44, true),
      maxSamplersPerShaderStage: limitsView.getUint32(48, true),
      maxStorageBuffersPerShaderStage: limitsView.getUint32(52, true),
      maxStorageTexturesPerShaderStage: limitsView.getUint32(56, true),
      maxUniformBuffersPerShaderStage: limitsView.getUint32(60, true),
      maxUniformBufferBindingSize: Number(limitsView.getBigUint64(64, true)),
      maxStorageBufferBindingSize: Number(limitsView.getBigUint64(72, true)),
      minUniformBufferOffsetAlignment: limitsView.getUint32(80, true),
      minStorageBufferOffsetAlignment: limitsView.getUint32(84, true),
      maxVertexBuffers: limitsView.getUint32(88, true),
      maxBufferSize: Number(limitsView.getBigUint64(96, true)),
      maxVertexAttributes: limitsView.getUint32(104, true),
      maxVertexBufferArrayStride: limitsView.getUint32(108, true),
      maxInterStageShaderVariables: limitsView.getUint32(112, true),
      maxColorAttachments: limitsView.getUint32(116, true),
      maxColorAttachmentBytesPerSample: limitsView.getUint32(120, true),
      maxComputeWorkgroupStorageSize: limitsView.getUint32(124, true),
      maxComputeInvocationsPerWorkgroup: limitsView.getUint32(128, true),
      maxComputeWorkgroupSizeX: limitsView.getUint32(132, true),
      maxComputeWorkgroupSizeY: limitsView.getUint32(136, true),
      maxComputeWorkgroupSizeZ: limitsView.getUint32(140, true),
      maxComputeWorkgroupsPerDimension: limitsView.getUint32(144, true),
    } as GPUSupportedLimits;
  }

  private queryFeatures(): GPUSupportedFeatures {
    const features = new Set<GPUFeatureName>();

    // Check each feature individually using wgpuDeviceHasFeature
    for (const [featureId, featureName] of Object.entries(FEATURE_NAME_MAP)) {
      const hasFeature = getLib().wgpuDeviceHasFeature(this._handle, Number(featureId));
      if (hasFeature === 1) {
        features.add(featureName as GPUFeatureName);
      }
    }

    return features as GPUSupportedFeatures;
  }

  protected releaseImpl(): void {
    this._queue.release();
    getLib().wgpuDeviceRelease(this._handle);
  }

  protected setLabelImpl(label: string): void {
    const encoder = new StructEncoder();
    const { data, length } = encoder.encodeString(label);
    const stringView = encoder.alloc(16);
    const view = new DataView(stringView.buffer.buffer);
    view.setBigUint64(0, BigInt(data), true);
    view.setBigUint64(8, BigInt(length), true);
    getLib().wgpuDeviceSetLabel(this._handle, stringView.ptr);
    encoder.freeAll();
  }

  get queue(): GPUQueue {
    return this._queue as unknown as GPUQueue;
  }

  get lost(): Promise<GPUDeviceLostInfo> {
    return this._lost;
  }

  get features(): GPUSupportedFeatures {
    return this._features;
  }

  get limits(): GPUSupportedLimits {
    return this._limits;
  }

  get adapterInfo(): GPUAdapterInfo {
    return this._adapterInfo;
  }

  destroy(): undefined {
    getLib().wgpuDeviceDestroy(this._handle);
    this._lostResolve({
      __brand: "GPUDeviceLostInfo",
      reason: "destroyed",
      message: "Device was destroyed",
    } as GPUDeviceLostInfo);
    super.destroy();
    return undefined;
  }

  importExternalTexture(descriptor: GPUExternalTextureDescriptor): GPUExternalTexture {
    throw new Error("importExternalTexture not supported in headless mode");
  }

  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
    // Validate before calling wgpu-native to provide better errors
    validateBufferUsage(descriptor.usage, descriptor.label);
    if (descriptor.size > 268435456) {
      throw new GPUValidationError(
        `Buffer size ${descriptor.size} exceeds maxBufferSize (268435456)${descriptor.label ? ` for buffer "${descriptor.label}"` : ""}`
      );
    }

    const encoder = new StructEncoder();
    const labelStr = descriptor.label ? encoder.encodeString(descriptor.label) : { data: 0, length: 0 };

    const descPtr = encoder.encode(WGPUBufferDescriptor, {
      nextInChain: 0,
      label: { data: labelStr.data, length: labelStr.length },
      usage: descriptor.usage,
      size: descriptor.size,
      mappedAtCreation: descriptor.mappedAtCreation ? 1 : 0,
    }).ptr;

    const bufferHandle = getLib().wgpuDeviceCreateBuffer(this._handle, descPtr);
    encoder.freeAll();

    // Import dynamically to avoid circular dependency
    const { GPUBufferImpl } = require("./buffer");
    return new GPUBufferImpl(
      bufferHandle,
      this._instance,
      this._handle,
      descriptor.size,
      descriptor.usage,
      descriptor.mappedAtCreation ?? false,
      descriptor.label
    );
  }

  createTexture(descriptor: GPUTextureDescriptor): GPUTexture {
    // Parse size
    const size = Array.isArray(descriptor.size)
      ? descriptor.size
      : [descriptor.size.width, descriptor.size.height ?? 1, descriptor.size.depthOrArrayLayers ?? 1];

    // Map dimension string to enum
    const { getTextureFormat, WGPUTextureDimension } = require("./texture");
    let dimension = WGPUTextureDimension._2D;
    const dimStr = descriptor.dimension ?? "2d";
    if (descriptor.dimension) {
      const dimMap: Record<string, number> = {
        "1d": WGPUTextureDimension._1D,
        "2d": WGPUTextureDimension._2D,
        "3d": WGPUTextureDimension._3D,
      };
      dimension = dimMap[descriptor.dimension] ?? WGPUTextureDimension._2D;
    }

    // Validate before calling wgpu-native
    validateTextureUsage(descriptor.usage, descriptor.label);
    validateTextureDimensions(size[0], size[1] ?? 1, size[2] ?? 1, dimStr, descriptor.label);

    // WGPUTextureDescriptor (80 bytes):
    // offset 0:  nextInChain (ptr, 8)
    // offset 8:  label.data (ptr, 8)
    // offset 16: label.length (size_t, 8)
    // offset 24: usage (u64, 8) - WGPUFlags is u64!
    // offset 32: dimension (u32, 4)
    // offset 36: size.width (u32, 4)
    // offset 40: size.height (u32, 4)
    // offset 44: size.depthOrArrayLayers (u32, 4)
    // offset 48: format (u32, 4)
    // offset 52: mipLevelCount (u32, 4)
    // offset 56: sampleCount (u32, 4)
    // offset 60: padding (4)
    // offset 64: viewFormatCount (size_t, 8)
    // offset 72: viewFormats (ptr, 8)
    // Total: 80 bytes

    const desc = new Uint8Array(80);
    pipelineBuffers.push(desc);
    const view = new DataView(desc.buffer);

    view.setBigUint64(0, BigInt(0), true);                      // nextInChain
    view.setBigUint64(8, BigInt(0), true);                      // label.data
    view.setBigUint64(16, WGPU_STRLEN, true);                   // label.length
    view.setBigUint64(24, BigInt(descriptor.usage), true);      // usage (u64!)
    view.setUint32(32, dimension, true);                        // dimension
    view.setUint32(36, size[0], true);                          // size.width
    view.setUint32(40, size[1] ?? 1, true);                     // size.height
    view.setUint32(44, size[2] ?? 1, true);                     // size.depthOrArrayLayers
    view.setUint32(48, getTextureFormat(descriptor.format), true); // format
    view.setUint32(52, descriptor.mipLevelCount ?? 1, true);    // mipLevelCount
    view.setUint32(56, descriptor.sampleCount ?? 1, true);      // sampleCount
    view.setUint32(60, 0, true);                                // padding
    view.setBigUint64(64, BigInt(0), true);                     // viewFormatCount
    view.setBigUint64(72, BigInt(0), true);                     // viewFormats

    const descPtr = ptr(desc);
    const textureHandle = getLib().wgpuDeviceCreateTexture(this._handle, descPtr);

    if (!textureHandle) {
      throw new Error("Failed to create texture");
    }

    const { GPUTextureImpl } = require("./texture");
    return new GPUTextureImpl(textureHandle, descriptor, descriptor.label) as unknown as GPUTexture;
  }

  createSampler(descriptor?: GPUSamplerDescriptor): GPUSampler {
    const { createSamplerDescriptor, GPUSamplerImpl } = require("./sampler");
    const desc = createSamplerDescriptor(descriptor);
    const descPtr = ptr(desc);

    const samplerHandle = getLib().wgpuDeviceCreateSampler(this._handle, descPtr);

    if (!samplerHandle) {
      throw new Error("Failed to create sampler");
    }

    return new GPUSamplerImpl(samplerHandle, descriptor?.label) as unknown as GPUSampler;
  }

  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup {
    // Use manual buffer management
    const entries = Array.from(descriptor.entries);
    const layoutHandle = (descriptor.layout as unknown as { handle: Pointer }).handle;

    // WGPUBindGroupEntry (56 bytes each):
    // offset 0:  nextInChain (ptr, 8)
    // offset 8:  binding (u32, 4)
    // offset 12: padding (4)
    // offset 16: buffer (ptr, 8)
    // offset 24: offset (u64, 8)
    // offset 32: size (u64, 8)
    // offset 40: sampler (ptr, 8)
    // offset 48: textureView (ptr, 8)
    const entrySize = 56;
    const entriesBuffer = new Uint8Array(entries.length * entrySize);
    pipelineBuffers.push(entriesBuffer);

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      const view = new DataView(entriesBuffer.buffer, i * entrySize, entrySize);

      view.setBigUint64(0, BigInt(0), true); // nextInChain
      view.setUint32(8, entry.binding, true); // binding
      view.setUint32(12, 0, true); // padding

      // Handle resource binding
      if (entry.resource && "buffer" in entry.resource) {
        const bufferBinding = entry.resource as GPUBufferBinding;
        const bufferHandle = (bufferBinding.buffer as unknown as { handle: Pointer }).handle;
        const bufferSize = (bufferBinding.buffer as unknown as { size: number }).size;
        view.setBigUint64(16, BigInt(bufferHandle as unknown as number), true); // buffer
        view.setBigUint64(24, BigInt(bufferBinding.offset ?? 0), true); // offset
        // Use actual buffer size if size not specified, wgpu needs actual size not 0
        view.setBigUint64(32, BigInt(bufferBinding.size ?? bufferSize), true); // size
        view.setBigUint64(40, BigInt(0), true); // sampler
        view.setBigUint64(48, BigInt(0), true); // textureView
      } else {
        // Sampler or TextureView binding
        view.setBigUint64(16, BigInt(0), true); // buffer
        view.setBigUint64(24, BigInt(0), true); // offset
        view.setBigUint64(32, BigInt(0), true); // size
        view.setBigUint64(40, BigInt(0), true); // sampler
        view.setBigUint64(48, BigInt(0), true); // textureView

        if (entry.resource && "__brand" in entry.resource) {
          const resource = entry.resource as unknown as { handle: Pointer; __brand: string };
          if (resource.__brand === "GPUSampler") {
            view.setBigUint64(40, BigInt(resource.handle as unknown as number), true);
          } else if (resource.__brand === "GPUTextureView") {
            view.setBigUint64(48, BigInt(resource.handle as unknown as number), true);
          }
        }
      }
    }

    const entriesPtr = ptr(entriesBuffer) as unknown as number;

    // WGPUBindGroupDescriptor (48 bytes):
    // offset 0:  nextInChain (ptr, 8)
    // offset 8:  label.data (ptr, 8)
    // offset 16: label.length (size_t, 8)
    // offset 24: layout (ptr, 8)
    // offset 32: entryCount (size_t, 8)
    // offset 40: entries (ptr, 8)
    const desc = new Uint8Array(48);
    pipelineBuffers.push(desc);
    const descView = new DataView(desc.buffer);

    descView.setBigUint64(0, BigInt(0), true);                      // nextInChain = null
    descView.setBigUint64(8, BigInt(0), true);                      // label.data = null
    descView.setBigUint64(16, WGPU_STRLEN, true);                   // label.length = WGPU_STRLEN
    descView.setBigUint64(24, BigInt(layoutHandle as unknown as number), true); // layout
    descView.setBigUint64(32, BigInt(entries.length), true);        // entryCount
    descView.setBigUint64(40, BigInt(entriesPtr), true);            // entries

    const descPtr = ptr(desc);
    const bindGroupHandle = getLib().wgpuDeviceCreateBindGroup(this._handle, descPtr);

    if (!bindGroupHandle) {
      throw new Error("Failed to create bind group");
    }

    const { GPUBindGroupImpl } = require("./bind-group");
    return new GPUBindGroupImpl(bindGroupHandle, descriptor.label) as unknown as GPUBindGroup;
  }

  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout {
    const entries = descriptor.entries ? Array.from(descriptor.entries) : [];

    // WGPUBindGroupLayoutEntry (104 bytes):
    // offset 0:  nextInChain (ptr, 8)
    // offset 8:  binding (u32, 4)
    // offset 12: padding (4)
    // offset 16: visibility (u64, 8) - WGPUShaderStageFlags
    // offset 24: buffer.nextInChain (ptr, 8)
    // offset 32: buffer.type (u32, 4)
    // offset 36: buffer.hasDynamicOffset (u32, 4)
    // offset 40: buffer.minBindingSize (u64, 8)
    // offset 48: sampler.nextInChain (ptr, 8)
    // offset 56: sampler.type (u32, 4)
    // offset 60: padding (4)
    // offset 64: texture.nextInChain (ptr, 8)
    // offset 72: texture.sampleType (u32, 4)
    // offset 76: texture.viewDimension (u32, 4)
    // offset 80: texture.multisampled (u32, 4)
    // offset 84: padding (4)
    // offset 88: storageTexture.nextInChain (ptr, 8)
    // offset 96: storageTexture.access (u32, 4)
    // offset 100: storageTexture.format (u32, 4)
    // offset 104: storageTexture.viewDimension (u32, 4)
    // Total: ~108 bytes, aligned to 112

    const entrySize = 112;
    const entriesBuffer = new Uint8Array(entries.length * entrySize);
    pipelineBuffers.push(entriesBuffer);

    // Buffer type map
    const bufferTypeMap: Record<string, number> = {
      "undefined": 0,
      "uniform": 1,
      "storage": 2,
      "read-only-storage": 3,
    };

    // Sampler type map
    const samplerTypeMap: Record<string, number> = {
      "undefined": 0,
      "filtering": 1,
      "non-filtering": 2,
      "comparison": 3,
    };

    // Texture sample type map
    const sampleTypeMap: Record<string, number> = {
      "undefined": 0,
      "float": 1,
      "unfilterable-float": 2,
      "depth": 3,
      "sint": 4,
      "uint": 5,
    };

    // View dimension map
    const viewDimensionMap: Record<string, number> = {
      "undefined": 0,
      "1d": 1,
      "2d": 2,
      "2d-array": 3,
      "cube": 4,
      "cube-array": 5,
      "3d": 6,
    };

    // Storage texture access map
    const storageAccessMap: Record<string, number> = {
      "undefined": 0,
      "write-only": 1,
      "read-only": 2,
      "read-write": 3,
    };

    const { getTextureFormat } = require("./texture");

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      const view = new DataView(entriesBuffer.buffer, i * entrySize, entrySize);

      view.setBigUint64(0, BigInt(0), true); // nextInChain
      view.setUint32(8, entry.binding, true);
      view.setUint32(12, 0, true); // padding
      view.setBigUint64(16, BigInt(entry.visibility), true);

      // Buffer binding layout
      view.setBigUint64(24, BigInt(0), true); // buffer.nextInChain
      view.setUint32(32, entry.buffer ? bufferTypeMap[entry.buffer.type ?? "undefined"] ?? 0 : 0, true);
      view.setUint32(36, entry.buffer?.hasDynamicOffset ? 1 : 0, true);
      view.setBigUint64(40, BigInt(entry.buffer?.minBindingSize ?? 0), true);

      // Sampler binding layout
      view.setBigUint64(48, BigInt(0), true); // sampler.nextInChain
      view.setUint32(56, entry.sampler ? samplerTypeMap[entry.sampler.type ?? "undefined"] ?? 0 : 0, true);
      view.setUint32(60, 0, true); // padding

      // Texture binding layout
      view.setBigUint64(64, BigInt(0), true); // texture.nextInChain
      view.setUint32(72, entry.texture ? sampleTypeMap[entry.texture.sampleType ?? "undefined"] ?? 0 : 0, true);
      view.setUint32(76, entry.texture ? viewDimensionMap[entry.texture.viewDimension ?? "undefined"] ?? 0 : 0, true);
      view.setUint32(80, entry.texture?.multisampled ? 1 : 0, true);
      view.setUint32(84, 0, true); // padding

      // Storage texture binding layout
      view.setBigUint64(88, BigInt(0), true); // storageTexture.nextInChain
      view.setUint32(96, entry.storageTexture ? storageAccessMap[entry.storageTexture.access ?? "undefined"] ?? 0 : 0, true);
      view.setUint32(100, entry.storageTexture ? getTextureFormat(entry.storageTexture.format) : 0, true);
      view.setUint32(104, entry.storageTexture ? viewDimensionMap[entry.storageTexture.viewDimension ?? "undefined"] ?? 0 : 0, true);
      view.setUint32(108, 0, true); // padding
    }

    const entriesPtr = entries.length > 0 ? ptr(entriesBuffer) as unknown as number : 0;

    // WGPUBindGroupLayoutDescriptor (40 bytes):
    const desc = new Uint8Array(40);
    pipelineBuffers.push(desc);
    const descView = new DataView(desc.buffer);

    descView.setBigUint64(0, BigInt(0), true); // nextInChain
    descView.setBigUint64(8, BigInt(0), true); // label.data
    descView.setBigUint64(16, WGPU_STRLEN, true); // label.length
    descView.setBigUint64(24, BigInt(entries.length), true); // entryCount
    descView.setBigUint64(32, BigInt(entriesPtr), true); // entries

    const descPtr = ptr(desc);
    const layoutHandle = getLib().wgpuDeviceCreateBindGroupLayout(this._handle, descPtr);

    if (!layoutHandle) {
      throw new Error("Failed to create bind group layout");
    }

    const { GPUBindGroupLayoutImpl } = require("./bind-group");
    return new GPUBindGroupLayoutImpl(layoutHandle, descriptor.label) as unknown as GPUBindGroupLayout;
  }

  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout {
    const encoder = new StructEncoder();
    const labelStr = descriptor.label ? encoder.encodeString(descriptor.label) : { data: 0, length: 0 };

    const layouts = Array.from(descriptor.bindGroupLayouts);
    const layoutHandles = layouts.map((l) => (l as unknown as { handle: Pointer }).handle);
    const layoutsPtrValue = layoutHandles.length > 0 ? encoder.ptrArray(layoutHandles) : (0 as unknown as Pointer);

    const descPtr = encoder.encode(WGPUPipelineLayoutDescriptor, {
      nextInChain: 0,
      label: { data: labelStr.data, length: labelStr.length },
      bindGroupLayoutCount: layouts.length,
      bindGroupLayouts: layoutsPtrValue,
    }).ptr;

    const pipelineLayoutHandle = getLib().wgpuDeviceCreatePipelineLayout(this._handle, descPtr);
    encoder.freeAll();

    const { GPUPipelineLayoutImpl } = require("./pipeline");
    return new GPUPipelineLayoutImpl(pipelineLayoutHandle, descriptor.label) as unknown as GPUPipelineLayout;
  }

  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule {
    // Use same approach as working debug-shader5.ts - manual buffer management
    const code = descriptor.code;

    // Validate shader code
    if (!code || code.trim().length === 0) {
      throw new GPUValidationError("Shader code cannot be empty");
    }

    // Allocate null-terminated code string
    const codeBytes = new TextEncoder().encode(code + "\0");
    shaderBuffers.push(codeBytes);
    const codePtr = ptr(codeBytes) as unknown as number;

    // WGPUShaderSourceWGSL (32 bytes)
    const wgsl = new Uint8Array(32);
    shaderBuffers.push(wgsl);
    const wgslView = new DataView(wgsl.buffer);

    wgslView.setBigUint64(0, BigInt(0), true);          // chain.next = null
    wgslView.setUint32(8, 2, true);                     // chain.sType = ShaderSourceWGSL (2)
    wgslView.setBigUint64(16, BigInt(codePtr), true);   // code.data
    wgslView.setBigUint64(24, WGPU_STRLEN, true);       // code.length = WGPU_STRLEN

    const wgslPtr = ptr(wgsl) as unknown as number;

    // WGPUShaderModuleDescriptor (24 bytes)
    const desc = new Uint8Array(24);
    shaderBuffers.push(desc);
    const descView = new DataView(desc.buffer);

    descView.setBigUint64(0, BigInt(wgslPtr), true);    // nextInChain -> WGSL source
    descView.setBigUint64(8, BigInt(0), true);          // label.data = null
    descView.setBigUint64(16, WGPU_STRLEN, true);       // label.length = WGPU_STRLEN

    const descPtr = ptr(desc);

    const moduleHandle = getLib().wgpuDeviceCreateShaderModule(this._handle, descPtr);

    if (!moduleHandle) {
      throw new Error("Failed to create shader module");
    }

    const { GPUShaderModuleImpl } = require("./shader-module");
    return new GPUShaderModuleImpl(moduleHandle, this._instance, descriptor.label) as unknown as GPUShaderModule;
  }

  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline {
    // Use manual buffer management like createShaderModule
    const moduleHandle = (descriptor.compute.module as unknown as { handle: Pointer }).handle;
    const layoutHandle = descriptor.layout === "auto"
      ? 0
      : ((descriptor.layout as unknown as { handle: Pointer })?.handle as unknown as number) ?? 0;

    // Allocate entry point string (null-terminated)
    let entryPointPtr = 0;
    if (descriptor.compute.entryPoint) {
      const entryPointBytes = new TextEncoder().encode(descriptor.compute.entryPoint + "\0");
      pipelineBuffers.push(entryPointBytes);
      entryPointPtr = ptr(entryPointBytes) as unknown as number;
    }

    // WGPUComputePipelineDescriptor (80 bytes):
    // offset 0:  nextInChain (ptr, 8)
    // offset 8:  label.data (ptr, 8)
    // offset 16: label.length (size_t, 8)
    // offset 24: layout (ptr, 8)
    // offset 32: compute.nextInChain (ptr, 8)
    // offset 40: compute.module (ptr, 8)
    // offset 48: compute.entryPoint.data (ptr, 8)
    // offset 56: compute.entryPoint.length (size_t, 8)
    // offset 64: compute.constantCount (size_t, 8)
    // offset 72: compute.constants (ptr, 8)
    const desc = new Uint8Array(80);
    pipelineBuffers.push(desc);
    const view = new DataView(desc.buffer);

    view.setBigUint64(0, BigInt(0), true);                      // nextInChain = null
    view.setBigUint64(8, BigInt(0), true);                      // label.data = null
    view.setBigUint64(16, WGPU_STRLEN, true);                   // label.length = WGPU_STRLEN
    view.setBigUint64(24, BigInt(layoutHandle), true);          // layout
    view.setBigUint64(32, BigInt(0), true);                     // compute.nextInChain = null
    view.setBigUint64(40, BigInt(moduleHandle as unknown as number), true); // compute.module
    view.setBigUint64(48, BigInt(entryPointPtr), true);         // compute.entryPoint.data
    view.setBigUint64(56, WGPU_STRLEN, true);                   // compute.entryPoint.length = WGPU_STRLEN
    view.setBigUint64(64, BigInt(0), true);                     // compute.constantCount = 0
    view.setBigUint64(72, BigInt(0), true);                     // compute.constants = null

    const descPtr = ptr(desc);
    const pipelineHandle = getLib().wgpuDeviceCreateComputePipeline(this._handle, descPtr);

    if (!pipelineHandle) {
      throw new Error("Failed to create compute pipeline");
    }

    const { GPUComputePipelineImpl } = require("./pipeline");
    return new GPUComputePipelineImpl(pipelineHandle, descriptor.label) as unknown as GPUComputePipeline;
  }

  createComputePipelineAsync(descriptor: GPUComputePipelineDescriptor): Promise<GPUComputePipeline> {
    // wgpu-native async pipeline creation is not fully implemented,
    // so we use the sync version wrapped in a microtask to provide
    // the async API contract.
    return new Promise((resolve, reject) => {
      queueMicrotask(() => {
        try {
          resolve(this.createComputePipeline(descriptor));
        } catch (error) {
          reject(error);
        }
      });
    });
  }

  createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline {
    const { getTextureFormat } = require("./texture");
    const { getCompareFunction } = require("./sampler");

    // Get handles
    const vertexModuleHandle = (descriptor.vertex.module as unknown as { handle: Pointer }).handle;
    const fragmentModuleHandle = descriptor.fragment
      ? (descriptor.fragment.module as unknown as { handle: Pointer }).handle
      : 0;
    const layoutHandle = descriptor.layout === "auto"
      ? 0
      : ((descriptor.layout as unknown as { handle: Pointer })?.handle as unknown as number) ?? 0;

    // Allocate entry point strings
    let vertexEntryPointPtr = 0;
    if (descriptor.vertex.entryPoint) {
      const bytes = new TextEncoder().encode(descriptor.vertex.entryPoint + "\0");
      pipelineBuffers.push(bytes);
      vertexEntryPointPtr = ptr(bytes) as unknown as number;
    }

    let fragmentEntryPointPtr = 0;
    if (descriptor.fragment?.entryPoint) {
      const bytes = new TextEncoder().encode(descriptor.fragment.entryPoint + "\0");
      pipelineBuffers.push(bytes);
      fragmentEntryPointPtr = ptr(bytes) as unknown as number;
    }

    // Vertex format enum mapping
    const vertexFormatMap: Record<string, number> = {
      "uint8": 0x01, "uint8x2": 0x02, "uint8x4": 0x03,
      "sint8": 0x04, "sint8x2": 0x05, "sint8x4": 0x06,
      "unorm8": 0x07, "unorm8x2": 0x08, "unorm8x4": 0x09,
      "snorm8": 0x0A, "snorm8x2": 0x0B, "snorm8x4": 0x0C,
      "uint16": 0x0D, "uint16x2": 0x0E, "uint16x4": 0x0F,
      "sint16": 0x10, "sint16x2": 0x11, "sint16x4": 0x12,
      "unorm16": 0x13, "unorm16x2": 0x14, "unorm16x4": 0x15,
      "snorm16": 0x16, "snorm16x2": 0x17, "snorm16x4": 0x18,
      "float16": 0x19, "float16x2": 0x1A, "float16x4": 0x1B,
      "float32": 0x1C, "float32x2": 0x1D, "float32x3": 0x1E, "float32x4": 0x1F,
      "uint32": 0x20, "uint32x2": 0x21, "uint32x3": 0x22, "uint32x4": 0x23,
      "sint32": 0x24, "sint32x2": 0x25, "sint32x3": 0x26, "sint32x4": 0x27,
      "unorm10-10-10-2": 0x28, "unorm8x4-bgra": 0x29,
    };

    // Vertex step mode enum
    const stepModeMap: Record<string, number> = {
      "vertex": 0x02,
      "instance": 0x03,
    };

    // Encode vertex buffer layouts
    // WGPUVertexAttribute (24 bytes with alignment):
    // offset 0:  format (u32, 4)
    // offset 4:  padding (4 for 8-byte alignment)
    // offset 8:  offset (u64, 8)
    // offset 16: shaderLocation (u32, 4)
    // offset 20: padding (4)
    // Total: 24 bytes

    // WGPUVertexBufferLayout (32 bytes):
    // offset 0:  stepMode (u32, 4)
    // offset 4:  padding (4)
    // offset 8:  arrayStride (u64, 8)
    // offset 16: attributeCount (size_t, 8)
    // offset 24: attributes (ptr, 8)
    // Total: 32 bytes

    let vertexBuffersPtr = 0;
    let vertexBufferCount = 0;
    if (descriptor.vertex.buffers && descriptor.vertex.buffers.length > 0) {
      const buffers = Array.from(descriptor.vertex.buffers);
      vertexBufferCount = buffers.length;

      // First, encode all attributes for all buffers
      const allAttributePtrs: number[] = [];
      for (const buffer of buffers) {
        if (!buffer || !buffer.attributes || buffer.attributes.length === 0) {
          allAttributePtrs.push(0);
          continue;
        }
        const attributes = Array.from(buffer.attributes);
        const attrsBuffer = new Uint8Array(attributes.length * 24);
        pipelineBuffers.push(attrsBuffer);

        for (let j = 0; j < attributes.length; j++) {
          const attr = attributes[j];
          const attrView = new DataView(attrsBuffer.buffer, j * 24, 24);
          attrView.setUint32(0, vertexFormatMap[attr.format] ?? 0x1C, true); // format
          attrView.setUint32(4, 0, true); // padding
          attrView.setBigUint64(8, BigInt(attr.offset), true); // offset
          attrView.setUint32(16, attr.shaderLocation, true); // shaderLocation
          attrView.setUint32(20, 0, true); // padding
        }
        allAttributePtrs.push(ptr(attrsBuffer) as unknown as number);
      }

      // Now encode the buffer layouts
      const buffersBuffer = new Uint8Array(buffers.length * 32);
      pipelineBuffers.push(buffersBuffer);

      for (let i = 0; i < buffers.length; i++) {
        const buffer = buffers[i];
        const bufView = new DataView(buffersBuffer.buffer, i * 32, 32);
        if (!buffer) {
          // Hole in array - use VertexBufferNotUsed
          bufView.setUint32(0, 0, true); // stepMode = VertexBufferNotUsed
          bufView.setUint32(4, 0, true); // padding
          bufView.setBigUint64(8, BigInt(0), true); // arrayStride
          bufView.setBigUint64(16, BigInt(0), true); // attributeCount
          bufView.setBigUint64(24, BigInt(0), true); // attributes
        } else {
          bufView.setUint32(0, stepModeMap[buffer.stepMode ?? "vertex"] ?? 0x02, true);
          bufView.setUint32(4, 0, true); // padding
          bufView.setBigUint64(8, BigInt(buffer.arrayStride), true);
          bufView.setBigUint64(16, BigInt(buffer.attributes?.length ?? 0), true);
          bufView.setBigUint64(24, BigInt(allAttributePtrs[i]), true);
        }
      }
      vertexBuffersPtr = ptr(buffersBuffer) as unknown as number;
    }

    // Blend factor enum mapping
    const blendFactorMap: Record<string, number> = {
      "zero": 0x01, "one": 0x02,
      "src": 0x03, "one-minus-src": 0x04,
      "src-alpha": 0x05, "one-minus-src-alpha": 0x06,
      "dst": 0x07, "one-minus-dst": 0x08,
      "dst-alpha": 0x09, "one-minus-dst-alpha": 0x0A,
      "src-alpha-saturated": 0x0B,
      "constant": 0x0C, "one-minus-constant": 0x0D,
      "src1": 0x0E, "one-minus-src1": 0x0F,
      "src1-alpha": 0x10, "one-minus-src1-alpha": 0x11,
    };

    // Blend operation enum mapping
    const blendOpMap: Record<string, number> = {
      "add": 0x01, "subtract": 0x02, "reverse-subtract": 0x03,
      "min": 0x04, "max": 0x05,
    };

    // WGPUBlendComponent (12 bytes):
    // offset 0: operation (u32, 4)
    // offset 4: srcFactor (u32, 4)
    // offset 8: dstFactor (u32, 4)

    // WGPUBlendState (24 bytes):
    // offset 0:  color (WGPUBlendComponent, 12)
    // offset 12: alpha (WGPUBlendComponent, 12)

    // Encode blend states for targets
    const blendStatePtrs: number[] = [];

    // WGPUColorTargetState (32 bytes):
    // offset 0:  nextInChain (ptr, 8)
    // offset 8:  format (u32, 4)
    // offset 12: padding (4)
    // offset 16: blend (ptr, 8)
    // offset 24: writeMask (u32, 4)
    // offset 28: padding (4)

    let colorTargetsPtr = 0;
    let colorTargetCount = 0;
    if (descriptor.fragment?.targets) {
      const targets = Array.from(descriptor.fragment.targets);
      colorTargetCount = targets.length;
      const colorTargetsBuffer = new Uint8Array(targets.length * 32);
      pipelineBuffers.push(colorTargetsBuffer);

      for (let i = 0; i < targets.length; i++) {
        const target = targets[i];
        if (!target) continue;
        const view = new DataView(colorTargetsBuffer.buffer, i * 32, 32);
        view.setBigUint64(0, BigInt(0), true); // nextInChain
        view.setUint32(8, getTextureFormat(target.format), true); // format
        view.setUint32(12, 0, true); // padding

        // Encode blend state if present
        let blendPtr = 0;
        if (target.blend) {
          const blendBuffer = new Uint8Array(24);
          pipelineBuffers.push(blendBuffer);
          const blendView = new DataView(blendBuffer.buffer);

          // Color blend component
          blendView.setUint32(0, blendOpMap[target.blend.color?.operation ?? "add"] ?? 0x01, true);
          blendView.setUint32(4, blendFactorMap[target.blend.color?.srcFactor ?? "one"] ?? 0x02, true);
          blendView.setUint32(8, blendFactorMap[target.blend.color?.dstFactor ?? "zero"] ?? 0x01, true);

          // Alpha blend component
          blendView.setUint32(12, blendOpMap[target.blend.alpha?.operation ?? "add"] ?? 0x01, true);
          blendView.setUint32(16, blendFactorMap[target.blend.alpha?.srcFactor ?? "one"] ?? 0x02, true);
          blendView.setUint32(20, blendFactorMap[target.blend.alpha?.dstFactor ?? "zero"] ?? 0x01, true);

          blendPtr = ptr(blendBuffer) as unknown as number;
        }

        view.setBigUint64(16, BigInt(blendPtr), true); // blend
        view.setUint32(24, target.writeMask ?? 0xF, true); // writeMask (default ALL)
        view.setUint32(28, 0, true); // padding
      }
      colorTargetsPtr = ptr(colorTargetsBuffer) as unknown as number;
    }

    // WGPUFragmentState (64 bytes):
    // offset 0:  nextInChain (ptr, 8)
    // offset 8:  module (ptr, 8)
    // offset 16: entryPoint.data (ptr, 8)
    // offset 24: entryPoint.length (size_t, 8)
    // offset 32: constantCount (size_t, 8)
    // offset 40: constants (ptr, 8)
    // offset 48: targetCount (size_t, 8)
    // offset 56: targets (ptr, 8)

    let fragmentStatePtr = 0;
    if (descriptor.fragment) {
      const fragmentState = new Uint8Array(64);
      pipelineBuffers.push(fragmentState);
      const view = new DataView(fragmentState.buffer);
      view.setBigUint64(0, BigInt(0), true); // nextInChain
      view.setBigUint64(8, BigInt(fragmentModuleHandle as unknown as number), true);
      view.setBigUint64(16, BigInt(fragmentEntryPointPtr), true);
      view.setBigUint64(24, WGPU_STRLEN, true);
      view.setBigUint64(32, BigInt(0), true); // constantCount
      view.setBigUint64(40, BigInt(0), true); // constants
      view.setBigUint64(48, BigInt(colorTargetCount), true);
      view.setBigUint64(56, BigInt(colorTargetsPtr), true);
      fragmentStatePtr = ptr(fragmentState) as unknown as number;
    }

    // Map primitive topology
    const topologyMap: Record<string, number> = {
      "point-list": 1,
      "line-list": 2,
      "line-strip": 3,
      "triangle-list": 4,
      "triangle-strip": 5,
    };
    const topology = topologyMap[descriptor.primitive?.topology ?? "triangle-list"] ?? 4;

    // Map strip index format
    const stripIndexFormatMap: Record<string, number> = {
      "uint16": 1,
      "uint32": 2,
    };
    const stripIndexFormat = descriptor.primitive?.stripIndexFormat
      ? stripIndexFormatMap[descriptor.primitive.stripIndexFormat] ?? 0
      : 0;

    // Map cull mode
    const cullModeMap: Record<string, number> = {
      "none": 1,
      "front": 2,
      "back": 3,
    };
    const cullMode = cullModeMap[descriptor.primitive?.cullMode ?? "none"] ?? 1;

    // Map front face
    const frontFaceMap: Record<string, number> = {
      "ccw": 1,
      "cw": 2,
    };
    const frontFace = frontFaceMap[descriptor.primitive?.frontFace ?? "ccw"] ?? 1;

    // Stencil operation enum mapping
    const stencilOpMap: Record<string, number> = {
      "keep": 0x01, "zero": 0x02, "replace": 0x03,
      "invert": 0x04, "increment-clamp": 0x05, "decrement-clamp": 0x06,
      "increment-wrap": 0x07, "decrement-wrap": 0x08,
    };

    // Encode depth/stencil state if present
    // WGPUStencilFaceState (16 bytes):
    // offset 0: compare (u32, 4)
    // offset 4: failOp (u32, 4)
    // offset 8: depthFailOp (u32, 4)
    // offset 12: passOp (u32, 4)

    // WGPUDepthStencilState (72 bytes):
    // offset 0:  nextInChain (ptr, 8)
    // offset 8:  format (u32, 4)
    // offset 12: depthWriteEnabled (u32, 4) - WGPUOptionalBool
    // offset 16: depthCompare (u32, 4)
    // offset 20: stencilFront.compare (u32, 4)
    // offset 24: stencilFront.failOp (u32, 4)
    // offset 28: stencilFront.depthFailOp (u32, 4)
    // offset 32: stencilFront.passOp (u32, 4)
    // offset 36: stencilBack.compare (u32, 4)
    // offset 40: stencilBack.failOp (u32, 4)
    // offset 44: stencilBack.depthFailOp (u32, 4)
    // offset 48: stencilBack.passOp (u32, 4)
    // offset 52: stencilReadMask (u32, 4)
    // offset 56: stencilWriteMask (u32, 4)
    // offset 60: depthBias (i32, 4)
    // offset 64: depthBiasSlopeScale (f32, 4)
    // offset 68: depthBiasClamp (f32, 4)
    // Total: 72 bytes

    let depthStencilStatePtr = 0;
    if (descriptor.depthStencil) {
      const ds = descriptor.depthStencil;
      const depthStencilBuffer = new Uint8Array(72);
      pipelineBuffers.push(depthStencilBuffer);
      const dsView = new DataView(depthStencilBuffer.buffer);

      dsView.setBigUint64(0, BigInt(0), true); // nextInChain
      dsView.setUint32(8, getTextureFormat(ds.format), true); // format

      // WGPUOptionalBool: 0=False, 1=True, 2=Undefined
      const depthWriteEnabled = ds.depthWriteEnabled === undefined ? 2 : (ds.depthWriteEnabled ? 1 : 0);
      dsView.setUint32(12, depthWriteEnabled, true);

      dsView.setUint32(16, getCompareFunction(ds.depthCompare), true);

      // stencilFront
      const sf = ds.stencilFront ?? {};
      dsView.setUint32(20, getCompareFunction(sf.compare ?? "always"), true);
      dsView.setUint32(24, stencilOpMap[sf.failOp ?? "keep"] ?? 0x01, true);
      dsView.setUint32(28, stencilOpMap[sf.depthFailOp ?? "keep"] ?? 0x01, true);
      dsView.setUint32(32, stencilOpMap[sf.passOp ?? "keep"] ?? 0x01, true);

      // stencilBack
      const sb = ds.stencilBack ?? {};
      dsView.setUint32(36, getCompareFunction(sb.compare ?? "always"), true);
      dsView.setUint32(40, stencilOpMap[sb.failOp ?? "keep"] ?? 0x01, true);
      dsView.setUint32(44, stencilOpMap[sb.depthFailOp ?? "keep"] ?? 0x01, true);
      dsView.setUint32(48, stencilOpMap[sb.passOp ?? "keep"] ?? 0x01, true);

      dsView.setUint32(52, ds.stencilReadMask ?? 0xFFFFFFFF, true);
      dsView.setUint32(56, ds.stencilWriteMask ?? 0xFFFFFFFF, true);
      dsView.setInt32(60, ds.depthBias ?? 0, true);
      dsView.setFloat32(64, ds.depthBiasSlopeScale ?? 0, true);
      dsView.setFloat32(68, ds.depthBiasClamp ?? 0, true);

      depthStencilStatePtr = ptr(depthStencilBuffer) as unknown as number;
    }

    // WGPURenderPipelineDescriptor layout (complex, many inline structs)
    // Create a large enough buffer
    const desc = new Uint8Array(256);
    pipelineBuffers.push(desc);
    const view = new DataView(desc.buffer);

    let offset = 0;

    // nextInChain (8)
    view.setBigUint64(offset, BigInt(0), true); offset += 8;
    // label.data (8)
    view.setBigUint64(offset, BigInt(0), true); offset += 8;
    // label.length (8)
    view.setBigUint64(offset, WGPU_STRLEN, true); offset += 8;
    // layout (8)
    view.setBigUint64(offset, BigInt(layoutHandle), true); offset += 8;

    // WGPUVertexState (inline, 64 bytes):
    // nextInChain (8)
    view.setBigUint64(offset, BigInt(0), true); offset += 8;
    // module (8)
    view.setBigUint64(offset, BigInt(vertexModuleHandle as unknown as number), true); offset += 8;
    // entryPoint.data (8)
    view.setBigUint64(offset, BigInt(vertexEntryPointPtr), true); offset += 8;
    // entryPoint.length (8)
    view.setBigUint64(offset, WGPU_STRLEN, true); offset += 8;
    // constantCount (8)
    view.setBigUint64(offset, BigInt(0), true); offset += 8;
    // constants (8)
    view.setBigUint64(offset, BigInt(0), true); offset += 8;
    // bufferCount (8)
    view.setBigUint64(offset, BigInt(vertexBufferCount), true); offset += 8;
    // buffers (8)
    view.setBigUint64(offset, BigInt(vertexBuffersPtr), true); offset += 8;

    // WGPUPrimitiveState (inline, 24 bytes):
    // nextInChain (8)
    view.setBigUint64(offset, BigInt(0), true); offset += 8;
    // topology (4)
    view.setUint32(offset, topology, true); offset += 4;
    // stripIndexFormat (4)
    view.setUint32(offset, stripIndexFormat, true); offset += 4;
    // frontFace (4)
    view.setUint32(offset, frontFace, true); offset += 4;
    // cullMode (4)
    view.setUint32(offset, cullMode, true); offset += 4;
    // unclippedDepth (4)
    view.setUint32(offset, descriptor.primitive?.unclippedDepth ? 1 : 0, true); offset += 4;
    // padding (4)
    view.setUint32(offset, 0, true); offset += 4;

    // depthStencil ptr (8)
    view.setBigUint64(offset, BigInt(depthStencilStatePtr), true); offset += 8;

    // WGPUMultisampleState (inline, 24 bytes):
    // nextInChain (8)
    view.setBigUint64(offset, BigInt(0), true); offset += 8;
    // count (4)
    view.setUint32(offset, descriptor.multisample?.count ?? 1, true); offset += 4;
    // mask (4)
    view.setUint32(offset, descriptor.multisample?.mask ?? 0xFFFFFFFF, true); offset += 4;
    // alphaToCoverageEnabled (4)
    view.setUint32(offset, descriptor.multisample?.alphaToCoverageEnabled ? 1 : 0, true); offset += 4;
    // padding (4)
    view.setUint32(offset, 0, true); offset += 4;

    // fragment ptr (8)
    view.setBigUint64(offset, BigInt(fragmentStatePtr), true); offset += 8;

    const descPtr = ptr(desc);
    const pipelineHandle = getLib().wgpuDeviceCreateRenderPipeline(this._handle, descPtr);

    if (!pipelineHandle) {
      throw new Error("Failed to create render pipeline");
    }

    const { GPURenderPipelineImpl } = require("./pipeline");
    return new GPURenderPipelineImpl(pipelineHandle, descriptor.label) as unknown as GPURenderPipeline;
  }

  createRenderPipelineAsync(descriptor: GPURenderPipelineDescriptor): Promise<GPURenderPipeline> {
    // For render pipeline async, we use the sync version wrapped in a microtask
    // since the full async implementation would require duplicating all the
    // complex descriptor encoding. This still provides the async API contract.
    return new Promise((resolve, reject) => {
      queueMicrotask(() => {
        try {
          resolve(this.createRenderPipeline(descriptor));
        } catch (error) {
          reject(error);
        }
      });
    });
  }

  createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder {
    const encoder = new StructEncoder();
    let descPtr: Pointer = 0 as Pointer;

    if (descriptor) {
      const labelStr = descriptor.label ? encoder.encodeString(descriptor.label) : { data: 0, length: 0 };
      descPtr = encoder.encode(WGPUCommandEncoderDescriptor, {
        nextInChain: 0,
        label: { data: labelStr.data, length: labelStr.length },
      }).ptr;
    }

    const encoderHandle = getLib().wgpuDeviceCreateCommandEncoder(this._handle, descPtr);
    encoder.freeAll();

    const { GPUCommandEncoderImpl } = require("./command-encoder");
    return new GPUCommandEncoderImpl(encoderHandle, this._instance, descriptor?.label);
  }

  createRenderBundleEncoder(descriptor: GPURenderBundleEncoderDescriptor): GPURenderBundleEncoder {
    const descBytes = createRenderBundleEncoderDescriptor(descriptor);
    const handle = getLib().wgpuDeviceCreateRenderBundleEncoder(this._handle, ptr(descBytes));
    if (!handle) {
      throw new Error("Failed to create render bundle encoder");
    }
    return new GPURenderBundleEncoderImpl(handle as Pointer, descriptor.label);
  }

  createQuerySet(descriptor: GPUQuerySetDescriptor): GPUQuerySet {
    const descBytes = createQuerySetDescriptor(descriptor);
    const handle = getLib().wgpuDeviceCreateQuerySet(this._handle, ptr(descBytes));
    if (!handle) {
      throw new Error("Failed to create query set");
    }
    return new GPUQuerySetImpl(handle as Pointer, descriptor.type, descriptor.count, descriptor.label);
  }

  pushErrorScope(filter: GPUErrorFilter): undefined {
    const filterMap: Record<GPUErrorFilter, number> = {
      validation: 1,
      "out-of-memory": 2,
      internal: 3,
    };
    getLib().wgpuDevicePushErrorScope(this._handle, filterMap[filter]);
    return;
  }

  async popErrorScope(): Promise<GPUError | null> {
    const encoder = new StructEncoder();
    const registry = getCallbackRegistry();
    const handle = createHandle<GPUError | null>();
    const callbackInfoPtr = registry.createPopErrorScopeCallback(encoder, handle);

    getLib().wgpuDevicePopErrorScope(this._handle, callbackInfoPtr);

    const result = await pollUntilComplete(this._instance, handle);
    encoder.freeAll();
    return result;
  }

  // Event handlers
  onuncapturederror: ((this: GPUDevice, ev: GPUUncapturedErrorEvent) => unknown) | null = null;

  addEventListener(
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: boolean | AddEventListenerOptions
  ): void {
    const opts = typeof options === "boolean" ? { capture: options } : options;
    let listeners = this._eventListeners.get(type);
    if (!listeners) {
      listeners = [];
      this._eventListeners.set(type, listeners);
    }

    // Check if listener already exists
    const exists = listeners.some((entry) => entry.listener === listener);
    if (!exists) {
      listeners.push({ listener, options: opts });
    }
  }

  removeEventListener(
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: boolean | EventListenerOptions
  ): void {
    const listeners = this._eventListeners.get(type);
    if (!listeners) return;

    const index = listeners.findIndex((entry) => entry.listener === listener);
    if (index !== -1) {
      listeners.splice(index, 1);
    }
  }

  dispatchEvent(event: Event): boolean {
    const type = event.type;

    // Call the onuncapturederror handler if this is an uncapturederror event
    if (type === "uncapturederror" && this.onuncapturederror) {
      try {
        this.onuncapturederror.call(this as unknown as GPUDevice, event as GPUUncapturedErrorEvent);
      } catch {
        // Ignore errors in event handlers
      }
    }

    // Call registered event listeners
    const listeners = this._eventListeners.get(type);
    if (listeners) {
      for (const entry of listeners) {
        try {
          if (typeof entry.listener === "function") {
            entry.listener.call(this, event);
          } else if (entry.listener.handleEvent) {
            entry.listener.handleEvent(event);
          }
        } catch {
          // Ignore errors in event handlers
        }

        // Handle once option
        if (entry.options?.once) {
          this.removeEventListener(type, entry.listener);
        }
      }
    }

    return !event.defaultPrevented;
  }

  /**
   * Dispatch an uncaptured error event
   */
  dispatchUncapturedError(error: GPUError): void {
    const event = {
      type: "uncapturederror",
      error,
      defaultPrevented: false,
      preventDefault: function () {
        this.defaultPrevented = true;
      },
    } as unknown as GPUUncapturedErrorEvent;

    this.dispatchEvent(event as unknown as Event);
  }

  private createDefaultLimits(): GPUSupportedLimits {
    return {
      __brand: "GPUSupportedLimits",
      maxTextureDimension1D: 8192,
      maxTextureDimension2D: 8192,
      maxTextureDimension3D: 2048,
      maxTextureArrayLayers: 256,
      maxBindGroups: 4,
      maxBindGroupsPlusVertexBuffers: 24,
      maxBindingsPerBindGroup: 1000,
      maxDynamicUniformBuffersPerPipelineLayout: 8,
      maxDynamicStorageBuffersPerPipelineLayout: 4,
      maxSampledTexturesPerShaderStage: 16,
      maxSamplersPerShaderStage: 16,
      maxStorageBuffersPerShaderStage: 8,
      maxStorageTexturesPerShaderStage: 4,
      maxUniformBuffersPerShaderStage: 12,
      maxUniformBufferBindingSize: 65536,
      maxStorageBufferBindingSize: 134217728,
      minUniformBufferOffsetAlignment: 256,
      minStorageBufferOffsetAlignment: 256,
      maxVertexBuffers: 8,
      maxBufferSize: 268435456,
      maxVertexAttributes: 16,
      maxVertexBufferArrayStride: 2048,
      maxInterStageShaderVariables: 16,
      maxColorAttachments: 8,
      maxColorAttachmentBytesPerSample: 32,
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupSizeZ: 64,
      maxComputeWorkgroupsPerDimension: 65535,
    } as GPUSupportedLimits;
  }
}

/**
 * Clear shader buffers to free memory
 */
export function clearShaderBuffers(): void {
  shaderBuffers.length = 0;
}

/**
 * Clear pipeline buffers to free memory
 */
export function clearPipelineBuffers(): void {
  pipelineBuffers.length = 0;
}
