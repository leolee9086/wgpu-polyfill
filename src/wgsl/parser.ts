/**
 * WGSL source code parser — extracts entry points, immediate sizes, etc.
 *
 * These parsers use a character-level scanner (not regex) to correctly
 * handle comments, string literals, and other WGSL syntax.
 */

export interface EntryPointInfo {
  name: string;
  stage: "vertex" | "fragment" | "compute";
}

/**
 * Parse entry points from WGSL source code.
 *
 * Handles:
 * - Line/block comments (// and /* *​/)
 * - String literals (' and " with escape sequences)
 * - Multiple attributes before fn
 * - Unicode identifiers
 */
export function parseEntryPoints(wgsl: string): EntryPointInfo[] {
  const entries: EntryPointInfo[] = [];
  const len = wgsl.length;

  const isIdentStart = (c: string) => /[a-zA-Z_]/.test(c) || (c.codePointAt(0) ?? 0) > 0x7f;
  const isIdentCont = (c: string) => isIdentStart(c) || /[0-9]/.test(c);
  const isSpace = (c: string) => c === " " || c === "\t" || c === "\n" || c === "\r";

  let i = 0;
  let pendingAttr: string | null = null;

  const peek = (n = 0) => (i + n < len ? wgsl[i + n] : "");
  const skipSpace = () => { while (i < len && isSpace(wgsl[i])) i++; };

  while (i < len) {
    const c = wgsl[i];

    // Line comment: //
    if (c === "/" && peek(1) === "/") {
      while (i < len && wgsl[i] !== "\n") i++;
      continue;
    }

    // Block comment: /* ... */
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
      skipSpace();
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
 * struct definition to count fields. Returns 0 if no immediate data.
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

  // Find struct definition by scanning
  const searchStr = "struct " + structName;
  const idx = wgsl.indexOf(searchStr);
  if (idx === -1) return 0;

  const openBrace = wgsl.indexOf("{", idx);
  if (openBrace === -1) return 0;

  const closeBrace = wgsl.indexOf("}", openBrace);
  if (closeBrace === -1) return 0;

  const structBody = wgsl.slice(openBrace + 1, closeBrace);
  const fieldRegex = /(\w+)\s*:\s*\w+/g;
  const fields = structBody.match(fieldRegex);
  return (fields?.length ?? 0) * 4;
}
