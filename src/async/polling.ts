import { getLib, type Pointer } from "../ffi";
import type { CompletionHandle } from "./callback-registry";

export function processEvents(instance: Pointer): void {
  getLib().wgpuInstanceProcessEvents(instance);
}

/**
 * Poll until a CompletionHandle signals done.
 * The JSCallback fires synchronously (WaitAnyOnly) so handle.done is set
 * before this function even starts executing. This loop is a safety net.
 */
export async function pollUntilComplete<T>(
  instance: Pointer,
  handle: CompletionHandle<T>,
  options: { maxIterations?: number; intervalMs?: number } = {},
): Promise<T> {
  const { maxIterations = 100, intervalMs = 1 } = options;

  for (let i = 0; i < maxIterations; i++) {
    if (handle.done) {
      if (handle.error) throw handle.error;
      return handle.result as T;
    }
    processEvents(instance);
    await new Promise((r) => setTimeout(r, intervalMs));
  }

  throw new Error(`Polling timed out after ${maxIterations} iterations`);
}

export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
