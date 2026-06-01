# wgpu-polyfill

[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/wiedymi)
[![Twitter](https://img.shields.io/badge/-Twitter-1DA1F2?style=flat-square&logo=twitter&logoColor=white)](https://x.com/wiedymi)
[![Email](https://img.shields.io/badge/-Email-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:contact@wiedymi.com)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/zemMZtrkSb)

> [!IMPORTANT]
> This polyfill is **Bun-only**. It uses Bun's native FFI to interface with wgpu-native and is not compatible with Node.js or browsers.

A WebGPU implementation for Bun using wgpu-native, enabling headless GPU computing and rendering.

> [!NOTE]
> This is a fork of [wiedymi/wgpu-polyfill](https://github.com/wiedymi/wgpu-polyfill). Thanks to [@wiedymi](https://github.com/wiedymi) for the excellent foundation.

## Installation

```bash
bun add wgpu-polyfill
```

## Quick Start

```typescript
import { installPolyfill } from "wgpu-polyfill";

// Install the polyfill
installPolyfill();

// Use standard WebGPU API
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// Create a compute shader
const shader = device.createShaderModule({
  code: `
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
      data[gid.x] = data[gid.x] * 2.0;
    }
  `,
});

// ... rest of your WebGPU code
```

## Features

### Compute Pipeline
- Shader modules (WGSL)
- Compute pipelines (sync and async creation)
- Bind groups and bind group layouts
- Buffer operations (map, unmap, write)
- Indirect dispatch

### Render Pipeline
- Render pipelines (sync and async creation)
- Vertex buffers with multiple attributes
- Index buffers (uint16, uint32)
- Render passes with multiple color attachments (MRT)
- Depth/stencil attachments
- MSAA (multi-sampled anti-aliasing) with resolve targets
- Blend states and color write masks
- Viewport and scissor rects
- Render bundles

### Textures
- 2D, 3D, and array textures
- Mip levels
- Multiple sample counts (MSAA)
- All standard formats (rgba8unorm, bgra8unorm, depth24plus, etc.)
- Compressed formats (BC, ETC2, ASTC)
- Texture views with aspect selection
- Samplers with filtering and addressing modes

### Resource Management
- Query sets (occlusion, timestamp)
- Error scopes (pushErrorScope/popErrorScope)
- Debug markers and groups
- Proper resource cleanup

## API

### Module Exports

```typescript
// Install polyfill to navigator.gpu
function installPolyfill(): GPU;

// Get GPU instance without installing to navigator
function getGPU(): GPU;

// Uninstall and cleanup
function uninstallPolyfill(): void;

// Clean up temporary buffers
function clearAllBuffers(): void;

// Constants
const GPUBufferUsage: { MAP_READ, MAP_WRITE, COPY_SRC, COPY_DST, INDEX, VERTEX, UNIFORM, STORAGE, INDIRECT, QUERY_RESOLVE };
const GPUTextureUsage: { COPY_SRC, COPY_DST, TEXTURE_BINDING, STORAGE_BINDING, RENDER_ATTACHMENT };
const GPUShaderStage: { VERTEX, FRAGMENT, COMPUTE };
const GPUMapMode: { READ, WRITE };
const GPUColorWrite: { RED, GREEN, BLUE, ALPHA, ALL };
```

## Examples

### Compute Shader

```typescript
import { installPolyfill, GPUBufferUsage } from "wgpu-polyfill";

installPolyfill();

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// Create buffers
const data = new Float32Array([1, 2, 3, 4]);
const gpuBuffer = device.createBuffer({
  size: data.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  mappedAtCreation: true,
});
new Float32Array(gpuBuffer.getMappedRange()).set(data);
gpuBuffer.unmap();

// Create shader and pipeline
const shader = device.createShaderModule({
  code: `
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;
    @compute @workgroup_size(4)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
      data[gid.x] *= 2.0;
    }
  `,
});

const pipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module: shader, entryPoint: "main" },
});

const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: gpuBuffer } }],
});

// Execute
const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(1);
pass.end();
device.queue.submit([encoder.finish()]);
```

### Render to Texture

```typescript
import { installPolyfill, GPUTextureUsage, GPUBufferUsage } from "wgpu-polyfill";

installPolyfill();

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// Create render target
const texture = device.createTexture({
  size: [256, 256],
  format: "rgba8unorm",
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});

// Create shader
const shader = device.createShaderModule({
  code: `
    @vertex
    fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
      var pos = array<vec2f, 3>(
        vec2f(0.0, 0.5),
        vec2f(-0.5, -0.5),
        vec2f(0.5, -0.5)
      );
      return vec4f(pos[i], 0.0, 1.0);
    }

    @fragment
    fn fs() -> @location(0) vec4f {
      return vec4f(1.0, 0.0, 0.0, 1.0);
    }
  `,
});

// Create pipeline
const pipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: { module: shader, entryPoint: "vs" },
  fragment: {
    module: shader,
    entryPoint: "fs",
    targets: [{ format: "rgba8unorm" }],
  },
});

// Render
const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass({
  colorAttachments: [{
    view: texture.createView(),
    loadOp: "clear",
    storeOp: "store",
    clearValue: { r: 0, g: 0, b: 0, a: 1 },
  }],
});
pass.setPipeline(pipeline);
pass.draw(3);
pass.end();
device.queue.submit([encoder.finish()]);
```

### MSAA Rendering

```typescript
// Create MSAA texture (4x samples)
const msaaTexture = device.createTexture({
  size: [256, 256],
  format: "rgba8unorm",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
  sampleCount: 4,
});

// Create resolve target
const resolveTexture = device.createTexture({
  size: [256, 256],
  format: "rgba8unorm",
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});

// Create pipeline with multisample state
const pipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: { module: shader, entryPoint: "vs" },
  fragment: {
    module: shader,
    entryPoint: "fs",
    targets: [{ format: "rgba8unorm" }],
  },
  multisample: { count: 4 },
});

// Render with resolve
const pass = encoder.beginRenderPass({
  colorAttachments: [{
    view: msaaTexture.createView(),
    resolveTarget: resolveTexture.createView(),
    loadOp: "clear",
    storeOp: "store",
    clearValue: { r: 0, g: 0, b: 0, a: 1 },
  }],
});
```

### Error Handling

```typescript
device.pushErrorScope("validation");

// Do potentially invalid operations
const badBuffer = device.createBuffer({
  size: 0, // Invalid!
  usage: GPUBufferUsage.STORAGE,
});

const error = await device.popErrorScope();
if (error) {
  console.error(`Validation error: ${error.message}`);
}
```

## Platform Support

The polyfill bundles wgpu-native binaries for:
- macOS arm64 (Apple Silicon)
- macOS x64 (Intel)
- Linux x64
- Windows x64

## Requirements

- Bun 1.0 or later
- A GPU with Vulkan, Metal, or DirectX 12 support

## License

MIT
