/**
 * Device and adapter related struct definitions
 */

import { defineStruct } from "../layout";
import { WGPUStringView } from "./common";
import { WGPUDeviceLostCallbackInfo, WGPUUncapturedErrorCallbackInfo } from "./callbacks";

// WGPURequestAdapterOptions
export const WGPURequestAdapterOptions = defineStruct("WGPURequestAdapterOptions", [
  { name: "nextInChain", type: "ptr" },
  { name: "featureLevel", type: "u32" }, // WGPUFeatureLevel
  { name: "powerPreference", type: "u32" }, // WGPUPowerPreference
  { name: "forceFallbackAdapter", type: "bool" },
  { name: "backendType", type: "u32" }, // WGPUBackendType
  { name: "compatibleSurface", type: "ptr" }, // nullable
]);

// WGPUQueueDescriptor
export const WGPUQueueDescriptor = defineStruct("WGPUQueueDescriptor", [
  { name: "nextInChain", type: "ptr" },
  { name: "label", type: { struct: WGPUStringView } },
]);

// WGPUDeviceDescriptor
export const WGPUDeviceDescriptor = defineStruct("WGPUDeviceDescriptor", [
  { name: "nextInChain", type: "ptr" },
  { name: "label", type: { struct: WGPUStringView } },
  { name: "requiredFeatureCount", type: "usize" },
  { name: "requiredFeatures", type: "ptr" }, // *WGPUFeatureName
  { name: "requiredLimits", type: "ptr" }, // *WGPULimits (nullable)
  { name: "defaultQueue", type: { struct: WGPUQueueDescriptor } },
  { name: "deviceLostCallbackInfo", type: { struct: WGPUDeviceLostCallbackInfo } },
  { name: "uncapturedErrorCallbackInfo", type: { struct: WGPUUncapturedErrorCallbackInfo } },
]);

// WGPULimits - all the device limits (v29 adds maxImmediateSize)
export const WGPULimits = defineStruct("WGPULimits", [
  { name: "nextInChain", type: "ptr" },
  { name: "maxTextureDimension1D", type: "u32" },
  { name: "maxTextureDimension2D", type: "u32" },
  { name: "maxTextureDimension3D", type: "u32" },
  { name: "maxTextureArrayLayers", type: "u32" },
  { name: "maxBindGroups", type: "u32" },
  { name: "maxBindGroupsPlusVertexBuffers", type: "u32" },
  { name: "maxBindingsPerBindGroup", type: "u32" },
  { name: "maxDynamicUniformBuffersPerPipelineLayout", type: "u32" },
  { name: "maxDynamicStorageBuffersPerPipelineLayout", type: "u32" },
  { name: "maxSampledTexturesPerShaderStage", type: "u32" },
  { name: "maxSamplersPerShaderStage", type: "u32" },
  { name: "maxStorageBuffersPerShaderStage", type: "u32" },
  { name: "maxStorageTexturesPerShaderStage", type: "u32" },
  { name: "maxUniformBuffersPerShaderStage", type: "u32" },
  { name: "maxUniformBufferBindingSize", type: "u64" },
  { name: "maxStorageBufferBindingSize", type: "u64" },
  { name: "minUniformBufferOffsetAlignment", type: "u32" },
  { name: "minStorageBufferOffsetAlignment", type: "u32" },
  { name: "maxVertexBuffers", type: "u32" },
  { name: "maxBufferSize", type: "u64" },
  { name: "maxVertexAttributes", type: "u32" },
  { name: "maxVertexBufferArrayStride", type: "u32" },
  { name: "maxInterStageShaderVariables", type: "u32" },
  { name: "maxColorAttachments", type: "u32" },
  { name: "maxColorAttachmentBytesPerSample", type: "u32" },
  { name: "maxComputeWorkgroupStorageSize", type: "u32" },
  { name: "maxComputeInvocationsPerWorkgroup", type: "u32" },
  { name: "maxComputeWorkgroupSizeX", type: "u32" },
  { name: "maxComputeWorkgroupSizeY", type: "u32" },
  { name: "maxComputeWorkgroupSizeZ", type: "u32" },
  { name: "maxComputeWorkgroupsPerDimension", type: "u32" },
  { name: "maxImmediateSize", type: "u32" },
]);

// WGPUAdapterInfo
export const WGPUAdapterInfo = defineStruct("WGPUAdapterInfo", [
  { name: "nextInChain", type: "ptr" },
  { name: "vendor", type: { struct: WGPUStringView } },
  { name: "architecture", type: { struct: WGPUStringView } },
  { name: "device", type: { struct: WGPUStringView } },
  { name: "description", type: { struct: WGPUStringView } },
  { name: "backendType", type: "u32" },
  { name: "adapterType", type: "u32" },
  { name: "vendorID", type: "u32" },
  { name: "deviceID", type: "u32" },
]);

// WGPUSupportedFeatures
export const WGPUSupportedFeatures = defineStruct("WGPUSupportedFeatures", [
  { name: "featureCount", type: "usize" },
  { name: "features", type: "ptr" }, // *WGPUFeatureName
]);
