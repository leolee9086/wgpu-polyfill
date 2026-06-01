/**
 * Base class for all GPU objects
 */

import type { Pointer } from "../ffi/types";

export abstract class GPUObjectBase {
  protected _handle: Pointer;
  protected _released = false;
  private _label: string;

  // Brand property for TypeScript nominal typing
  readonly __brand: string = "GPUObjectBase";

  constructor(handle: Pointer, label: string = "") {
    this._handle = handle;
    this._label = label;
  }

  get label(): string {
    return this._label;
  }

  set label(value: string) {
    this._label = value;
    this.setLabelImpl(value);
  }

  /**
   * Get the native handle (throws if released)
   */
  get handle(): Pointer {
    if (this._released) {
      throw new Error(`${this.constructor.name} has been released`);
    }
    return this._handle;
  }

  /**
   * Check if the object has been released
   */
  get isReleased(): boolean {
    return this._released;
  }

  /**
   * Release the native resource
   */
  release(): void {
    if (this._released) return;
    this._released = true;
    // Skip null/zero handles (error placeholder objects)
    if (this._handle) {
      this.releaseImpl();
    }
  }

  /**
   * Destroy the object (alias for release in most cases)
   */
  destroy(): undefined {
    this.release();
    return undefined;
  }

  protected abstract releaseImpl(): void;
  protected abstract setLabelImpl(label: string): void;
}
