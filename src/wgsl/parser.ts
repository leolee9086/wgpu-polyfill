/**
 * WGSL parsers — entry point extraction and immediate size detection.
 *
 * Low-level scanning primitives come from my-wgsl-loader's utils.
 */

import { utils } from "../../../my-wgsl-loader/src/parser.js";

export interface EntryPointInfo {
  name: string;
  stage: "vertex" | "fragment" | "compute";
}

const { isInCommentOrString, extractStruct } = utils;

/**
 * Parse entry points from WGSL source code.
 * Uses a stateful O(n) scanner that correctly handles comments,
 * strings, and Unicode identifiers.
 */
export function parseEntryPoints(wgsl: string): EntryPointInfo[] {
  const entries: EntryPointInfo[] = [];
  const len = wgsl.length;

  const isIdentStart = (c: string) => /[a-zA-Z_]/.test(c) || (c.codePointAt(0) ?? 0) > 0x7f;
  const isIdentCont = (c: string) => isIdentStart(c) || /[0-9]/.test(c);

  let i = 0;
  let pendingAttr: string | null = null;

  const peek = (n = 0) => (i + n < len ? wgsl[i + n] : "");

  while (i < len) {
    const c = wgsl[i];

    // Line comment
    if (c === "/" && peek(1) === "/") {
      while (i < len && wgsl[i] !== "\n") i++;
      continue;
    }

    // Block comment
    if (c === "/" && peek(1) === "*") {
      i += 2;
      while (i < len - 1 && !(wgsl[i] === "*" && peek(1) === "/")) i++;
      i += 2;
      continue;
    }

    // String literals
    if (c === '"' || c === "'") {
      const quote = c;
      i++;
      while (i < len && wgsl[i] !== quote) {
        if (wgsl[i] === "\\") i++;
        i++;
      }
      if (i < len) i++;
      continue;
    }

    // Attribute: @identifier — only track stage attributes
    if (c === "@") {
      i++;
      let attr = "";
      while (i < len && isIdentCont(wgsl[i])) {
        attr += wgsl[i];
        i++;
      }
      if (attr === "vertex" || attr === "fragment" || attr === "compute") {
        pendingAttr = attr;
      }
      continue;
    }

    // fn keyword
    if (c === "f" && peek(1) === "n" && !isIdentCont(peek(2))) {
      i += 2;
      while (i < len && /[\s]/.test(wgsl[i])) i++;
      let name = "";
      if (i < len && isIdentStart(wgsl[i])) {
        while (i < len && isIdentCont(wgsl[i])) {
          name += wgsl[i];
          i++;
        }
      }
      if (name && pendingAttr) {
        if (!entries.some(e => e.name === name && e.stage === pendingAttr)) {
          entries.push({ name, stage: pendingAttr as EntryPointInfo["stage"] });
        }
      }
      pendingAttr = null;
      continue;
    }

    i++;
  }

  return entries;
}

/**
 * Detect the immediate data size (in bytes) used by a WGSL shader.
 * Looks for `var<immediate> data: <StructName>;` and then finds the
 * struct definition using my-wgsl-loader's extractStruct + isInCommentOrString.
 *
 * Example:
 *   struct Immediates { m0: u32, m1: u32, m2: u32, m3: u32 }
 *   var<immediate> data: Immediates;
 *   → returns 16 (4 fields × 4 bytes)
 */
export function detectImmediateSize(wgsl: string): number {
  if (!wgsl.includes("immediate")) return 0;

  const varMatch = wgsl.match(/var<immediate>\s+\w+\s*:\s*(\w+)\s*;/);
  if (!varMatch) return 0;

  const structName = varMatch[1];

  try {
    const structDef = extractStruct(wgsl, structName);
    const fieldRegex = /(\w+)\s*:\s*\w+/g;
    const fields = structDef.match(fieldRegex);
    return (fields?.length ?? 0) * 4;
  } catch {
    return 0;
  }
}
