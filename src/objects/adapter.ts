/**
 * GPUAdapter implementation
 */

import { GPUObjectBase } from "./base";
import { GPUDeviceImpl } from "./device";
import { getLib, type Pointer } from "../ffi";
import { StructEncoder } from "../structs/encoder";
import { getCallbackRegistry, createHandle } from "../async/callback-registry";
import { pollUntilComplete } from "../async/polling";
import { WGPUDeviceDescriptor, WGPULimits } from "../structs/definitions/device";
import { ptr, JSCallback } from "bun:ffi";

// WGPUFeatureName to GPUFeatureName mapping (v29 enum values)
const FEATURE_NAME_MAP: Record<number, GPUFeatureName> = {
  0x01: "core-features-and-limits",
  0x02: "depth-clip-control",
  0x03: "depth32float-stencil8",
  0x04: "texture-compression-bc",
  0x05: "texture-compression-bc-sliced-3d",
  0x06: "texture-compression-etc2",
  0x07: "texture-compression-astc",
  0x08: "texture-compression-astc-sliced-3d",
  0x09: "timestamp-query",
  0x0A: "indirect-first-instance",
  0x0B: "shader-f16",
  0x0C: "rg11b10ufloat-renderable",
  0x0D: "bgra8unorm-storage",
  0x0E: "float32-filterable",
  0x0F: "float32-blendable",
  0x10: "clip-distances",
  0x11: "dual-source-blending",
  0x12: "subgroups",
  0x13: "texture-formats-tier-1",
  0x14: "texture-formats-tier-2",
  0x15: "primitive-index",
  0x16: "texture-component-swizzle",
};

// Reverse map: GPUFeatureName → native feature ID
const FEATURE_NAME_TO_ID: Record<string, number> = {};
for (const [id, name] of Object.entries(FEATURE_NAME_MAP)) {
  FEATURE_NAME_TO_ID[name] = Number(id);
}

// Native extension features (WGPUNativeFeature) mapped to GPUFeatureName
const NATIVE_FEATURE_MAP: Record<number, GPUFeatureName> = {
  0x00030001: "immediate-data",
};
// Reverse map for native features
const NATIVE_FEATURE_NAME_TO_ID: Record<string, number> = {};
for (const [id, name] of Object.entries(NATIVE_FEATURE_MAP)) {
  NATIVE_FEATURE_NAME_TO_ID[name] = Number(id);
}

// Backend type mapping
const BACKEND_TYPE_MAP: Record<number, string> = {
  0: "undefined",
  1: "null",
  2: "webgpu",
  3: "d3d11",
  4: "d3d12",
  5: "metal",
  6: "vulkan",
  7: "opengl",
  8: "opengles",
};

// Adapter type mapping
const ADAPTER_TYPE_MAP: Record<number, string> = {
  0: "discrete GPU",
  1: "integrated GPU",
  2: "CPU",
  3: "unknown",
};

export class GPUAdapterImpl extends GPUObjectBase implements GPUAdapter {
  readonly __brand = "GPUAdapter";
  private _instance: Pointer;
  private _features: GPUSupportedFeatures;
  private _limits: GPUSupportedLimits;
  private _info: GPUAdapterInfo;
  private _isFallbackAdapter: boolean;

  constructor(handle: Pointer, instance: Pointer) {
    super(handle);
    this._instance = instance;

    // Query actual features and limits from adapter
    this._features = this.queryFeatures();
    this._limits = this.queryLimits();
    this._info = this.queryAdapterInfo();
    this._isFallbackAdapter = false;
  }

  private queryLimits(): GPUSupportedLimits {
    // WGPULimits struct layout (v29: 160 bytes with maxImmediateSize)
    const limitsBuffer = new Uint8Array(160);
    const limitsView = new DataView(limitsBuffer.buffer);
    limitsView.setBigUint64(0, BigInt(0), true); // nextInChain = null

    const status = getLib().wgpuAdapterGetLimits(this._handle, ptr(limitsBuffer));

    if (status !== 1) {
      // Return default limits on failure
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
      maxImmediateSize: limitsView.getUint32(148, true),
    } as GPUSupportedLimits;
  }

  private queryFeatures(): GPUSupportedFeatures {
    // WGPUSupportedFeatures struct (16 bytes):
    // offset 0: featureCount (usize, 8)
    // offset 8: features (ptr, 8)
    const featuresBuffer = new Uint8Array(16);
    const featuresView = new DataView(featuresBuffer.buffer);
    featuresView.setBigUint64(0, BigInt(0), true); // featureCount = 0
    featuresView.setBigUint64(8, BigInt(0), true); // features = null

    getLib().wgpuAdapterGetFeatures(this._handle, ptr(featuresBuffer));

    const featureCount = Number(featuresView.getBigUint64(0, true));
    const featuresPtr = featuresView.getBigUint64(8, true);

    const features = new Set<GPUFeatureName>();

    if (featureCount > 0 && featuresPtr !== 0n) {
      // Read feature array (each feature is a u32)
      const featuresArray = new Uint32Array(
        new Uint8Array(featureCount * 4).buffer
      );

      // We need to read from the pointer - create a view into native memory
      // This is tricky with Bun FFI, so let's use wgpuAdapterHasFeature instead
      for (const [featureId, featureName] of Object.entries(FEATURE_NAME_MAP)) {
        const hasFeature = getLib().wgpuAdapterHasFeature(this._handle, Number(featureId));
        if (hasFeature === 1) {
          features.add(featureName as GPUFeatureName);
        }
      }
      // Check native extension features
      for (const [featureId, featureName] of Object.entries(NATIVE_FEATURE_MAP)) {
        const hasFeature = getLib().wgpuAdapterHasFeature(this._handle, Number(featureId));
        if (hasFeature === 1) {
          features.add(featureName as GPUFeatureName);
        }
      }
    } else {
      // Fallback: check each feature individually
      for (const [featureId, featureName] of Object.entries(FEATURE_NAME_MAP)) {
        const hasFeature = getLib().wgpuAdapterHasFeature(this._handle, Number(featureId));
        if (hasFeature === 1) {
          features.add(featureName as GPUFeatureName);
        }
      }
    }

    return features as GPUSupportedFeatures;
  }

  private queryAdapterInfo(): GPUAdapterInfo {
    // WGPUAdapterInfo struct (88 bytes):
    // offset 0: nextInChain (ptr, 8)
    // offset 8: vendor.data (ptr, 8)
    // offset 16: vendor.length (usize, 8)
    // offset 24: architecture.data (ptr, 8)
    // offset 32: architecture.length (usize, 8)
    // offset 40: device.data (ptr, 8)
    // offset 48: device.length (usize, 8)
    // offset 56: description.data (ptr, 8)
    // offset 64: description.length (usize, 8)
    // offset 72: backendType (u32, 4)
    // offset 76: adapterType (u32, 4)
    // offset 80: vendorID (u32, 4)
    // offset 84: deviceID (u32, 4)
    const infoBuffer = new Uint8Array(88);
    const infoView = new DataView(infoBuffer.buffer);
    infoView.setBigUint64(0, BigInt(0), true); // nextInChain = null

    const status = getLib().wgpuAdapterGetInfo(this._handle, ptr(infoBuffer));

    if (status !== 1) {
      return {
        __brand: "GPUAdapterInfo",
        vendor: "",
        architecture: "",
        device: "",
        description: "",
        isFallbackAdapter: false,
      } as GPUAdapterInfo;
    }

    // Read string pointers and lengths
    const vendorPtr = infoView.getBigUint64(8, true);
    const vendorLen = Number(infoView.getBigUint64(16, true));
    const archPtr = infoView.getBigUint64(24, true);
    const archLen = Number(infoView.getBigUint64(32, true));
    const devicePtr = infoView.getBigUint64(40, true);
    const deviceLen = Number(infoView.getBigUint64(48, true));
    const descPtr = infoView.getBigUint64(56, true);
    const descLen = Number(infoView.getBigUint64(64, true));
    const backendType = infoView.getUint32(72, true);
    const adapterType = infoView.getUint32(76, true);

    // Helper to read string from pointer
    const readString = (ptrVal: bigint, len: number): string => {
      if (ptrVal === 0n || len === 0) return "";
      // For reading native strings, we need to copy from the pointer
      // wgpu-native returns null-terminated strings, so we can read up to len bytes
      try {
        const buffer = Buffer.from(
          new Uint8Array(Bun.mmap(Number(ptrVal), len))
        );
        const nullIndex = buffer.indexOf(0);
        return buffer.slice(0, nullIndex === -1 ? len : nullIndex).toString("utf8");
      } catch {
        return "";
      }
    };

    const backend = BACKEND_TYPE_MAP[backendType] || "unknown";
    const type = ADAPTER_TYPE_MAP[adapterType] || "unknown";

    return {
      __brand: "GPUAdapterInfo",
      vendor: readString(vendorPtr, vendorLen),
      architecture: readString(archPtr, archLen),
      device: readString(devicePtr, deviceLen),
      description: `${readString(descPtr, descLen)} (${backend}, ${type})`,
      isFallbackAdapter: false,
    } as GPUAdapterInfo;
  }

  protected releaseImpl(): void {
    getLib().wgpuAdapterRelease(this._handle);
  }

  protected setLabelImpl(_label: string): void {
    // Adapters don't have labels in WebGPU spec
  }

  get features(): GPUSupportedFeatures {
    return this._features;
  }

  get limits(): GPUSupportedLimits {
    return this._limits;
  }

  get info(): GPUAdapterInfo {
    return this._info;
  }

  get isFallbackAdapter(): boolean {
    return this._isFallbackAdapter;
  }

  async requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice> {
    const encoder = new StructEncoder();
    const registry = getCallbackRegistry();

    try {
      // Build device descriptor
      const labelStr = descriptor?.label ? encoder.encodeString(descriptor.label) : { data: 0, length: 0 };
      const queueLabelStr = descriptor?.defaultQueue?.label
        ? encoder.encodeString(descriptor.defaultQueue.label)
        : { data: 0, length: 0 };

      // Register native uncaptured error callback so wgpu doesn't panic
      const errorCallback = new JSCallback(
        (_device: Pointer, type: number, _msgData: Pointer, _msgLen: number) => {
          // Forward to device error handler; just logging is enough to prevent panic
        },
        { args: ["ptr", "u32", "ptr", "usize"], returns: "void" }
      );

      // Build required features array from descriptor
      let requiredFeaturesPtr: Pointer = 0 as Pointer;
      let requiredFeatureCount = 0;
      if (descriptor?.requiredFeatures) {
        const requested = Array.from(descriptor.requiredFeatures);
        const featureIds = new Uint32Array(requested.length);
        let idx = 0;
        for (const name of requested) {
          // Try standard feature map first, then native
          const id = FEATURE_NAME_TO_ID[name] ?? NATIVE_FEATURE_NAME_TO_ID[name];
          if (id !== undefined) {
            featureIds[idx++] = id;
          }
        }
        if (idx > 0) {
          const featuresBytes = new Uint8Array(idx * 4);
          new Uint32Array(featuresBytes.buffer).set(featureIds.slice(0, idx));
          (encoder as any).allocations.push(featuresBytes);
          requiredFeaturesPtr = ptr(featuresBytes) as Pointer;
          requiredFeatureCount = idx;
        }
      }

      // Build required limits from descriptor (if provided)
      let requiredLimitsPtr: Pointer = 0 as Pointer;
      if (descriptor?.requiredLimits) {
        const limitsValues: Record<string, unknown> = { nextInChain: 0 };
        // Copy all provided limits, defaulting undefined values to 0
        for (const field of WGPULimits.fields) {
          if (field.name === "nextInChain") continue;
          const key = field.name as keyof GPUSupportedLimits;
          limitsValues[field.name] = (descriptor.requiredLimits as any)?.[key] ?? 0;
        }
        requiredLimitsPtr = encoder.encode(WGPULimits, limitsValues).ptr;
      }

      const descPtr = encoder.encode(WGPUDeviceDescriptor, {
        nextInChain: 0,
        label: { data: labelStr.data, length: labelStr.length },
        requiredFeatureCount,
        requiredFeatures: requiredFeaturesPtr,
        requiredLimits: requiredLimitsPtr,
        defaultQueue: {
          nextInChain: 0,
          label: { data: queueLabelStr.data, length: queueLabelStr.length },
        },
        deviceLostCallbackInfo: {
          nextInChain: 0,
          mode: 2,
          callback: 0,
          userdata1: 0,
          userdata2: 0,
        },
        uncapturedErrorCallbackInfo: {
          nextInChain: 0,
          callback: errorCallback.ptr,
          userdata1: 0,
          userdata2: 0,
        },
      }).ptr;

      const handle = createHandle<Pointer>();
      const callbackInfoPtr = registry.createDeviceCallback(encoder, handle);

      getLib().wgpuAdapterRequestDevice(this._handle, descPtr, callbackInfoPtr);

      const deviceHandle = await pollUntilComplete(this._instance, handle);

      // Keep the callback alive (store on device)
      const device = new GPUDeviceImpl(deviceHandle, this._instance, descriptor?.label, this._info, errorCallback);
      return device as unknown as GPUDevice;
    } finally {
      encoder.freeAll();
    }
  }

  async requestAdapterInfo(): Promise<GPUAdapterInfo> {
    return this._info;
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
      maxImmediateSize: 256,
    } as GPUSupportedLimits;
  }
}
