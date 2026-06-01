/**
 * WGSL parsers — re-exported from wgsl-module-loader.
 *
 * All WGSL parsing logic lives in my-wgsl-loader (published as
 * wgsl-module-loader on npm). This file is just a thin re-export
 * for convenience.
 */

export type { EntryPointInfo } from "wgsl-module-loader/parser";
export { parseEntryPoints, detectImmediateSize, utils } from "wgsl-module-loader/parser";
