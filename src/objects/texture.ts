/**
 * GPUTexture and GPUTextureView implementation
 */

import { ptr } from "bun:ffi";
import { GPUObjectBase } from "./base";
import { getLib, type Pointer } from "../ffi";
import { WGPU_STRLEN } from "../ffi/types";

// Global buffer storage to prevent GC issues
const textureBuffers: Uint8Array[] = [];

// Texture format enum values from wgpu-native
export const WGPUTextureFormat = {
  Undefined: 0,
  R8Unorm: 1,
  R8Snorm: 2,
  R8Uint: 3,
  R8Sint: 4,
  R16Uint: 5,
  R16Sint: 6,
  R16Float: 7,
  RG8Unorm: 8,
  RG8Snorm: 9,
  RG8Uint: 10,
  RG8Sint: 11,
  R32Float: 12,
  R32Uint: 13,
  R32Sint: 14,
  RG16Uint: 15,
  RG16Sint: 16,
  RG16Float: 17,
  RGBA8Unorm: 18,
  RGBA8UnormSrgb: 19,
  RGBA8Snorm: 20,
  RGBA8Uint: 21,
  RGBA8Sint: 22,
  BGRA8Unorm: 23,
  BGRA8UnormSrgb: 24,
  RGB10A2Uint: 25,
  RGB10A2Unorm: 26,
  RG11B10Ufloat: 27,
  RGB9E5Ufloat: 28,
  RG32Float: 29,
  RG32Uint: 30,
  RG32Sint: 31,
  RGBA16Uint: 32,
  RGBA16Sint: 33,
  RGBA16Float: 34,
  RGBA32Float: 35,
  RGBA32Uint: 36,
  RGBA32Sint: 37,
  Stencil8: 38,
  Depth16Unorm: 39,
  Depth24Plus: 40,
  Depth24PlusStencil8: 41,
  Depth32Float: 42,
  Depth32FloatStencil8: 43,
  // BC compressed formats
  BC1RGBAUnorm: 44,
  BC1RGBAUnormSrgb: 45,
  BC2RGBAUnorm: 46,
  BC2RGBAUnormSrgb: 47,
  BC3RGBAUnorm: 48,
  BC3RGBAUnormSrgb: 49,
  BC4RUnorm: 50,
  BC4RSnorm: 51,
  BC5RGUnorm: 52,
  BC5RGSnorm: 53,
  BC6HRGBUfloat: 54,
  BC6HRGBFloat: 55,
  BC7RGBAUnorm: 56,
  BC7RGBAUnormSrgb: 57,
  // ETC2 compressed formats
  ETC2RGB8Unorm: 58,
  ETC2RGB8UnormSrgb: 59,
  ETC2RGB8A1Unorm: 60,
  ETC2RGB8A1UnormSrgb: 61,
  ETC2RGBA8Unorm: 62,
  ETC2RGBA8UnormSrgb: 63,
  EACR11Unorm: 64,
  EACR11Snorm: 65,
  EACRG11Unorm: 66,
  EACRG11Snorm: 67,
  // ASTC compressed formats
  ASTC4x4Unorm: 68,
  ASTC4x4UnormSrgb: 69,
  ASTC5x4Unorm: 70,
  ASTC5x4UnormSrgb: 71,
  ASTC5x5Unorm: 72,
  ASTC5x5UnormSrgb: 73,
  ASTC6x5Unorm: 74,
  ASTC6x5UnormSrgb: 75,
  ASTC6x6Unorm: 76,
  ASTC6x6UnormSrgb: 77,
  ASTC8x5Unorm: 78,
  ASTC8x5UnormSrgb: 79,
  ASTC8x6Unorm: 80,
  ASTC8x6UnormSrgb: 81,
  ASTC8x8Unorm: 82,
  ASTC8x8UnormSrgb: 83,
  ASTC10x5Unorm: 84,
  ASTC10x5UnormSrgb: 85,
  ASTC10x6Unorm: 86,
  ASTC10x6UnormSrgb: 87,
  ASTC10x8Unorm: 88,
  ASTC10x8UnormSrgb: 89,
  ASTC10x10Unorm: 90,
  ASTC10x10UnormSrgb: 91,
  ASTC12x10Unorm: 92,
  ASTC12x10UnormSrgb: 93,
  ASTC12x12Unorm: 94,
  ASTC12x12UnormSrgb: 95,
} as const;

// Texture dimension enum
export const WGPUTextureDimension = {
  Undefined: 0,
  _1D: 1,
  _2D: 2,
  _3D: 3,
} as const;

// Texture view dimension enum
export const WGPUTextureViewDimension = {
  Undefined: 0,
  _1D: 1,
  _2D: 2,
  _2DArray: 3,
  Cube: 4,
  CubeArray: 5,
  _3D: 6,
} as const;

// Texture aspect enum
export const WGPUTextureAspect = {
  Undefined: 0,
  All: 1,
  StencilOnly: 2,
  DepthOnly: 3,
} as const;

// Map JS format strings to native format values
const formatMap: Record<string, number> = {
  "r8unorm": WGPUTextureFormat.R8Unorm,
  "r8snorm": WGPUTextureFormat.R8Snorm,
  "r8uint": WGPUTextureFormat.R8Uint,
  "r8sint": WGPUTextureFormat.R8Sint,
  "r16uint": WGPUTextureFormat.R16Uint,
  "r16sint": WGPUTextureFormat.R16Sint,
  "r16float": WGPUTextureFormat.R16Float,
  "rg8unorm": WGPUTextureFormat.RG8Unorm,
  "rg8snorm": WGPUTextureFormat.RG8Snorm,
  "rg8uint": WGPUTextureFormat.RG8Uint,
  "rg8sint": WGPUTextureFormat.RG8Sint,
  "r32float": WGPUTextureFormat.R32Float,
  "r32uint": WGPUTextureFormat.R32Uint,
  "r32sint": WGPUTextureFormat.R32Sint,
  "rg16uint": WGPUTextureFormat.RG16Uint,
  "rg16sint": WGPUTextureFormat.RG16Sint,
  "rg16float": WGPUTextureFormat.RG16Float,
  "rgba8unorm": WGPUTextureFormat.RGBA8Unorm,
  "rgba8unorm-srgb": WGPUTextureFormat.RGBA8UnormSrgb,
  "rgba8snorm": WGPUTextureFormat.RGBA8Snorm,
  "rgba8uint": WGPUTextureFormat.RGBA8Uint,
  "rgba8sint": WGPUTextureFormat.RGBA8Sint,
  "bgra8unorm": WGPUTextureFormat.BGRA8Unorm,
  "bgra8unorm-srgb": WGPUTextureFormat.BGRA8UnormSrgb,
  "rgb10a2uint": WGPUTextureFormat.RGB10A2Uint,
  "rgb10a2unorm": WGPUTextureFormat.RGB10A2Unorm,
  "rg11b10ufloat": WGPUTextureFormat.RG11B10Ufloat,
  "rgb9e5ufloat": WGPUTextureFormat.RGB9E5Ufloat,
  "rg32float": WGPUTextureFormat.RG32Float,
  "rg32uint": WGPUTextureFormat.RG32Uint,
  "rg32sint": WGPUTextureFormat.RG32Sint,
  "rgba16uint": WGPUTextureFormat.RGBA16Uint,
  "rgba16sint": WGPUTextureFormat.RGBA16Sint,
  "rgba16float": WGPUTextureFormat.RGBA16Float,
  "rgba32float": WGPUTextureFormat.RGBA32Float,
  "rgba32uint": WGPUTextureFormat.RGBA32Uint,
  "rgba32sint": WGPUTextureFormat.RGBA32Sint,
  "stencil8": WGPUTextureFormat.Stencil8,
  "depth16unorm": WGPUTextureFormat.Depth16Unorm,
  "depth24plus": WGPUTextureFormat.Depth24Plus,
  "depth24plus-stencil8": WGPUTextureFormat.Depth24PlusStencil8,
  "depth32float": WGPUTextureFormat.Depth32Float,
  "depth32float-stencil8": WGPUTextureFormat.Depth32FloatStencil8,
  // BC compressed formats
  "bc1-rgba-unorm": WGPUTextureFormat.BC1RGBAUnorm,
  "bc1-rgba-unorm-srgb": WGPUTextureFormat.BC1RGBAUnormSrgb,
  "bc2-rgba-unorm": WGPUTextureFormat.BC2RGBAUnorm,
  "bc2-rgba-unorm-srgb": WGPUTextureFormat.BC2RGBAUnormSrgb,
  "bc3-rgba-unorm": WGPUTextureFormat.BC3RGBAUnorm,
  "bc3-rgba-unorm-srgb": WGPUTextureFormat.BC3RGBAUnormSrgb,
  "bc4-r-unorm": WGPUTextureFormat.BC4RUnorm,
  "bc4-r-snorm": WGPUTextureFormat.BC4RSnorm,
  "bc5-rg-unorm": WGPUTextureFormat.BC5RGUnorm,
  "bc5-rg-snorm": WGPUTextureFormat.BC5RGSnorm,
  "bc6h-rgb-ufloat": WGPUTextureFormat.BC6HRGBUfloat,
  "bc6h-rgb-float": WGPUTextureFormat.BC6HRGBFloat,
  "bc7-rgba-unorm": WGPUTextureFormat.BC7RGBAUnorm,
  "bc7-rgba-unorm-srgb": WGPUTextureFormat.BC7RGBAUnormSrgb,
  // ETC2 compressed formats
  "etc2-rgb8unorm": WGPUTextureFormat.ETC2RGB8Unorm,
  "etc2-rgb8unorm-srgb": WGPUTextureFormat.ETC2RGB8UnormSrgb,
  "etc2-rgb8a1unorm": WGPUTextureFormat.ETC2RGB8A1Unorm,
  "etc2-rgb8a1unorm-srgb": WGPUTextureFormat.ETC2RGB8A1UnormSrgb,
  "etc2-rgba8unorm": WGPUTextureFormat.ETC2RGBA8Unorm,
  "etc2-rgba8unorm-srgb": WGPUTextureFormat.ETC2RGBA8UnormSrgb,
  "eac-r11unorm": WGPUTextureFormat.EACR11Unorm,
  "eac-r11snorm": WGPUTextureFormat.EACR11Snorm,
  "eac-rg11unorm": WGPUTextureFormat.EACRG11Unorm,
  "eac-rg11snorm": WGPUTextureFormat.EACRG11Snorm,
  // ASTC compressed formats
  "astc-4x4-unorm": WGPUTextureFormat.ASTC4x4Unorm,
  "astc-4x4-unorm-srgb": WGPUTextureFormat.ASTC4x4UnormSrgb,
  "astc-5x4-unorm": WGPUTextureFormat.ASTC5x4Unorm,
  "astc-5x4-unorm-srgb": WGPUTextureFormat.ASTC5x4UnormSrgb,
  "astc-5x5-unorm": WGPUTextureFormat.ASTC5x5Unorm,
  "astc-5x5-unorm-srgb": WGPUTextureFormat.ASTC5x5UnormSrgb,
  "astc-6x5-unorm": WGPUTextureFormat.ASTC6x5Unorm,
  "astc-6x5-unorm-srgb": WGPUTextureFormat.ASTC6x5UnormSrgb,
  "astc-6x6-unorm": WGPUTextureFormat.ASTC6x6Unorm,
  "astc-6x6-unorm-srgb": WGPUTextureFormat.ASTC6x6UnormSrgb,
  "astc-8x5-unorm": WGPUTextureFormat.ASTC8x5Unorm,
  "astc-8x5-unorm-srgb": WGPUTextureFormat.ASTC8x5UnormSrgb,
  "astc-8x6-unorm": WGPUTextureFormat.ASTC8x6Unorm,
  "astc-8x6-unorm-srgb": WGPUTextureFormat.ASTC8x6UnormSrgb,
  "astc-8x8-unorm": WGPUTextureFormat.ASTC8x8Unorm,
  "astc-8x8-unorm-srgb": WGPUTextureFormat.ASTC8x8UnormSrgb,
  "astc-10x5-unorm": WGPUTextureFormat.ASTC10x5Unorm,
  "astc-10x5-unorm-srgb": WGPUTextureFormat.ASTC10x5UnormSrgb,
  "astc-10x6-unorm": WGPUTextureFormat.ASTC10x6Unorm,
  "astc-10x6-unorm-srgb": WGPUTextureFormat.ASTC10x6UnormSrgb,
  "astc-10x8-unorm": WGPUTextureFormat.ASTC10x8Unorm,
  "astc-10x8-unorm-srgb": WGPUTextureFormat.ASTC10x8UnormSrgb,
  "astc-10x10-unorm": WGPUTextureFormat.ASTC10x10Unorm,
  "astc-10x10-unorm-srgb": WGPUTextureFormat.ASTC10x10UnormSrgb,
  "astc-12x10-unorm": WGPUTextureFormat.ASTC12x10Unorm,
  "astc-12x10-unorm-srgb": WGPUTextureFormat.ASTC12x10UnormSrgb,
  "astc-12x12-unorm": WGPUTextureFormat.ASTC12x12Unorm,
  "astc-12x12-unorm-srgb": WGPUTextureFormat.ASTC12x12UnormSrgb,
};

export function getTextureFormat(format: GPUTextureFormat): number {
  return formatMap[format] ?? WGPUTextureFormat.Undefined;
}

// Reverse map for format lookup
const reverseFormatMap: Record<number, GPUTextureFormat> = {};
for (const [key, value] of Object.entries(formatMap)) {
  reverseFormatMap[value] = key as GPUTextureFormat;
}

export function getTextureFormatString(format: number): GPUTextureFormat {
  return reverseFormatMap[format] ?? "rgba8unorm";
}

export class GPUTextureImpl extends GPUObjectBase implements GPUTexture {
  readonly __brand = "GPUTexture";
  private _width: number;
  private _height: number;
  private _depthOrArrayLayers: number;
  private _mipLevelCount: number;
  private _sampleCount: number;
  private _dimension: GPUTextureDimension;
  private _format: GPUTextureFormat;
  private _usage: GPUFlagsConstant;

  constructor(
    handle: Pointer,
    descriptor: GPUTextureDescriptor,
    label?: string
  ) {
    super(handle, label);
    const size = Array.isArray(descriptor.size) ? descriptor.size : [descriptor.size.width, descriptor.size.height ?? 1, descriptor.size.depthOrArrayLayers ?? 1];
    this._width = size[0];
    this._height = size[1] ?? 1;
    this._depthOrArrayLayers = size[2] ?? 1;
    this._mipLevelCount = descriptor.mipLevelCount ?? 1;
    this._sampleCount = descriptor.sampleCount ?? 1;
    this._dimension = descriptor.dimension ?? "2d";
    this._format = descriptor.format;
    this._usage = descriptor.usage;
  }

  protected releaseImpl(): void {
    getLib().wgpuTextureRelease(this._handle);
  }

  protected setLabelImpl(label: string): void {
    const labelBytes = new TextEncoder().encode(label + "\0");
    textureBuffers.push(labelBytes);
    const labelPtr = ptr(labelBytes) as unknown as number;

    const stringView = new Uint8Array(16);
    textureBuffers.push(stringView);
    const view = new DataView(stringView.buffer);
    view.setBigUint64(0, BigInt(labelPtr), true);
    view.setBigUint64(8, WGPU_STRLEN, true);

    getLib().wgpuTextureSetLabel(this._handle, ptr(stringView));
  }

  get width(): number {
    return this._width;
  }

  get height(): number {
    return this._height;
  }

  get depthOrArrayLayers(): number {
    return this._depthOrArrayLayers;
  }

  get mipLevelCount(): number {
    return this._mipLevelCount;
  }

  get sampleCount(): number {
    return this._sampleCount;
  }

  get dimension(): GPUTextureDimension {
    return this._dimension;
  }

  get format(): GPUTextureFormat {
    return this._format;
  }

  get usage(): GPUFlagsConstant {
    return this._usage;
  }

  createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView {
    // WGPUTextureViewDescriptor (48 bytes):
    // offset 0:  nextInChain (ptr, 8)
    // offset 8:  label.data (ptr, 8)
    // offset 16: label.length (size_t, 8)
    // offset 24: format (u32, 4)
    // offset 28: dimension (u32, 4)
    // offset 32: baseMipLevel (u32, 4)
    // offset 36: mipLevelCount (u32, 4)
    // offset 40: baseArrayLayer (u32, 4)
    // offset 44: arrayLayerCount (u32, 4)
    // offset 48: aspect (u32, 4)
    // offset 52: usage (u32, 4)
    // Total: 56 bytes

    const desc = new Uint8Array(56);
    textureBuffers.push(desc);
    const view = new DataView(desc.buffer);

    view.setBigUint64(0, BigInt(0), true); // nextInChain
    view.setBigUint64(8, BigInt(0), true); // label.data
    view.setBigUint64(16, WGPU_STRLEN, true); // label.length

    const format = descriptor?.format ? getTextureFormat(descriptor.format) : getTextureFormat(this._format);
    view.setUint32(24, format, true);

    // Map dimension string to enum
    let dimension = WGPUTextureViewDimension._2D;
    if (descriptor?.dimension) {
      const dimMap: Record<string, number> = {
        "1d": WGPUTextureViewDimension._1D,
        "2d": WGPUTextureViewDimension._2D,
        "2d-array": WGPUTextureViewDimension._2DArray,
        "cube": WGPUTextureViewDimension.Cube,
        "cube-array": WGPUTextureViewDimension.CubeArray,
        "3d": WGPUTextureViewDimension._3D,
      };
      dimension = dimMap[descriptor.dimension] ?? WGPUTextureViewDimension._2D;
    }
    view.setUint32(28, dimension, true);

    view.setUint32(32, descriptor?.baseMipLevel ?? 0, true);
    view.setUint32(36, descriptor?.mipLevelCount ?? this._mipLevelCount, true);
    view.setUint32(40, descriptor?.baseArrayLayer ?? 0, true);
    view.setUint32(44, descriptor?.arrayLayerCount ?? this._depthOrArrayLayers, true);

    // Map aspect string to enum
    let aspect = WGPUTextureAspect.All;
    if (descriptor?.aspect) {
      const aspectMap: Record<string, number> = {
        "all": WGPUTextureAspect.All,
        "stencil-only": WGPUTextureAspect.StencilOnly,
        "depth-only": WGPUTextureAspect.DepthOnly,
      };
      aspect = aspectMap[descriptor.aspect] ?? WGPUTextureAspect.All;
    }
    view.setUint32(48, aspect, true);
    view.setUint32(52, 0, true); // usage: let wgpu-native infer from texture

    const descPtr = ptr(desc);
    const viewHandle = getLib().wgpuTextureCreateView(this._handle, descPtr);

    if (!viewHandle) {
      throw new Error("Failed to create texture view");
    }

    return new GPUTextureViewImpl(viewHandle, this._format, descriptor?.label) as unknown as GPUTextureView;
  }

  destroy(): undefined {
    if (this._released) return undefined;
    getLib().wgpuTextureDestroy(this._handle);
    super.destroy();
    return undefined;
  }
}

export class GPUTextureViewImpl extends GPUObjectBase implements GPUTextureView {
  readonly __brand = "GPUTextureView";
  private _format: GPUTextureFormat;

  constructor(handle: Pointer, format: GPUTextureFormat, label?: string) {
    super(handle, label);
    this._format = format;
  }

  protected releaseImpl(): void {
    getLib().wgpuTextureViewRelease(this._handle);
  }

  protected setLabelImpl(label: string): void {
    const labelBytes = new TextEncoder().encode(label + "\0");
    textureBuffers.push(labelBytes);
    const labelPtr = ptr(labelBytes) as unknown as number;

    const stringView = new Uint8Array(16);
    textureBuffers.push(stringView);
    const view = new DataView(stringView.buffer);
    view.setBigUint64(0, BigInt(labelPtr), true);
    view.setBigUint64(8, WGPU_STRLEN, true);

    getLib().wgpuTextureViewSetLabel(this._handle, ptr(stringView));
  }

  get format(): GPUTextureFormat {
    return this._format;
  }
}

/**
 * Clear texture buffers to free memory
 */
export function clearTextureBuffers(): void {
  textureBuffers.length = 0;
}
