/**
 * Shader and pipeline struct definitions
 */

import { defineStruct } from "../layout";
import { WGPUStringView } from "./common";

// WGPUShaderSourceWGSL - chained struct for WGSL source
export const WGPUShaderSourceWGSL = defineStruct("WGPUShaderSourceWGSL", [
  { name: "chain", type: "ptr" }, // nextInChain
  { name: "sType", type: "u32" }, // WGPUSType_ShaderSourceWGSL = 2
  { name: "code", type: { struct: WGPUStringView } },
]);

// WGPUShaderModuleDescriptor
export const WGPUShaderModuleDescriptor = defineStruct("WGPUShaderModuleDescriptor", [
  { name: "nextInChain", type: "ptr" },
  { name: "label", type: { struct: WGPUStringView } },
]);

// WGPUProgrammableStageDescriptor
export const WGPUProgrammableStageDescriptor = defineStruct("WGPUProgrammableStageDescriptor", [
  { name: "nextInChain", type: "ptr" },
  { name: "module", type: "ptr" },
  { name: "entryPoint", type: { struct: WGPUStringView } },
  { name: "constantCount", type: "usize" },
  { name: "constants", type: "ptr" },
]);

// WGPUComputePipelineDescriptor
export const WGPUComputePipelineDescriptor = defineStruct("WGPUComputePipelineDescriptor", [
  { name: "nextInChain", type: "ptr" },
  { name: "label", type: { struct: WGPUStringView } },
  { name: "layout", type: "ptr" }, // nullable, auto layout
  { name: "compute", type: { struct: WGPUProgrammableStageDescriptor } },
]);

// WGPUBindGroupLayoutEntry
export const WGPUBindGroupLayoutEntry = defineStruct("WGPUBindGroupLayoutEntry", [
  { name: "nextInChain", type: "ptr" },
  { name: "binding", type: "u32" },
  { name: "visibility", type: "u32" }, // WGPUShaderStage flags
  { name: "buffer", type: "ptr" }, // WGPUBufferBindingLayout (struct value, 32 bytes)
  { name: "sampler", type: "ptr" }, // WGPUSamplerBindingLayout
  { name: "texture", type: "ptr" }, // WGPUTextureBindingLayout
  { name: "storageTexture", type: "ptr" }, // WGPUStorageTextureBindingLayout
]);

// WGPUBufferBindingLayout
export const WGPUBufferBindingLayout = defineStruct("WGPUBufferBindingLayout", [
  { name: "nextInChain", type: "ptr" },
  { name: "type", type: "u32" }, // WGPUBufferBindingType
  { name: "hasDynamicOffset", type: "bool" },
  { name: "minBindingSize", type: "u64" },
]);

// WGPUBindGroupLayoutDescriptor
export const WGPUBindGroupLayoutDescriptor = defineStruct("WGPUBindGroupLayoutDescriptor", [
  { name: "nextInChain", type: "ptr" },
  { name: "label", type: { struct: WGPUStringView } },
  { name: "entryCount", type: "usize" },
  { name: "entries", type: "ptr" },
]);

// WGPUBindGroupEntry
export const WGPUBindGroupEntry = defineStruct("WGPUBindGroupEntry", [
  { name: "nextInChain", type: "ptr" },
  { name: "binding", type: "u32" },
  { name: "buffer", type: "ptr" }, // nullable
  { name: "offset", type: "u64" },
  { name: "size", type: "u64" },
  { name: "sampler", type: "ptr" }, // nullable
  { name: "textureView", type: "ptr" }, // nullable
]);

// WGPUBindGroupDescriptor
export const WGPUBindGroupDescriptor = defineStruct("WGPUBindGroupDescriptor", [
  { name: "nextInChain", type: "ptr" },
  { name: "label", type: { struct: WGPUStringView } },
  { name: "layout", type: "ptr" },
  { name: "entryCount", type: "usize" },
  { name: "entries", type: "ptr" },
]);

// WGPUPipelineLayoutDescriptor (v29 adds immediateSize at end)
export const WGPUPipelineLayoutDescriptor = defineStruct("WGPUPipelineLayoutDescriptor", [
  { name: "nextInChain", type: "ptr" },
  { name: "label", type: { struct: WGPUStringView } },
  { name: "bindGroupLayoutCount", type: "usize" },
  { name: "bindGroupLayouts", type: "ptr" },
  { name: "immediateSize", type: "u32" },
]);
