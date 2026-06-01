/**
 * Callback registry - converts C callbacks to JavaScript completion handles.
 *
 * Uses a polling-friendly completion handle pattern:
 * - JSCallback sets a synchronous `done` flag + `result`/`error`
 * - `pollUntilComplete` checks the flag directly (no Promise.resolve issue in Bun)
 * - Promise wrapping is done at the `pollUntilComplete` level for API convenience
 */

import { JSCallback } from "bun:ffi";
import { memory, type Pointer } from "../ffi";
import { StructEncoder } from "../structs/encoder";
import {
  WGPURequestAdapterCallbackInfo,
  WGPURequestDeviceCallbackInfo,
  WGPUBufferMapCallbackInfo,
  WGPUQueueWorkDoneCallbackInfo,
  WGPUPopErrorScopeCallbackInfo,
  WGPUCreateComputePipelineAsyncCallbackInfo,
  WGPUCreateRenderPipelineAsyncCallbackInfo,
  WGPUCompilationInfoCallbackInfo,
} from "../structs/definitions/callbacks";
import {
  WGPUCallbackMode,
  WGPURequestAdapterStatus,
  WGPURequestDeviceStatus,
  WGPUBufferMapAsyncStatus,
  WGPUErrorType,
  WGPUCreatePipelineAsyncStatus,
  WGPUCompilationInfoRequestStatus,
  WGPUCompilationMessageType,
} from "../ffi/types";

export interface CompletionHandle<T> {
  done: boolean;
  result: T | undefined;
  error: Error | undefined;
}

export function createHandle<T>(): CompletionHandle<T> {
  return { done: false, result: undefined, error: undefined };
}

/**
 * Manages callback registration
 */
export class CallbackRegistry {

  createAdapterCallback(encoder: StructEncoder, handle: CompletionHandle<Pointer>): Pointer {
    const jsCallback = new JSCallback(
      (status: number, adapter: Pointer, messageData: Pointer, messageLength: number) => {
        if (status === WGPURequestAdapterStatus.Success) {
          handle.result = adapter;
        } else {
          handle.error = new Error(
            `Failed to request adapter: ${messageLength > 0 ? memory.readString(messageData, messageLength) : "Unknown"}`
          );
        }
        handle.done = true;
        setTimeout(() => jsCallback.close(), 0);
      },
      { args: ["u32", "ptr", "ptr", "usize"], returns: "void" }
    );

    return encoder.encode(WGPURequestAdapterCallbackInfo, {
      nextInChain: 0, mode: WGPUCallbackMode.WaitAnyOnly,
      callback: jsCallback.ptr, userdata1: 0, userdata2: 0,
    }).ptr;
  }

  createDeviceCallback(encoder: StructEncoder, handle: CompletionHandle<Pointer>): Pointer {
    const jsCallback = new JSCallback(
      (status: number, device: Pointer, messageData: Pointer, messageLength: number) => {
        if (status === WGPURequestDeviceStatus.Success) {
          handle.result = device;
        } else {
          handle.error = new Error(
            `Failed to request device: ${messageLength > 0 ? memory.readString(messageData, messageLength) : "Unknown"}`
          );
        }
        handle.done = true;
        setTimeout(() => jsCallback.close(), 0);
      },
      { args: ["u32", "ptr", "ptr", "usize"], returns: "void" }
    );

    return encoder.encode(WGPURequestDeviceCallbackInfo, {
      nextInChain: 0, mode: WGPUCallbackMode.WaitAnyOnly,
      callback: jsCallback.ptr, userdata1: 0, userdata2: 0,
    }).ptr;
  }

  createBufferMapCallback(encoder: StructEncoder, handle: CompletionHandle<void>): Pointer {
    const jsCallback = new JSCallback(
      (status: number, messageData: Pointer, messageLength: number) => {
        if (status === WGPUBufferMapAsyncStatus.Success) {
          handle.result = undefined;
        } else {
          handle.error = new Error(
            `Failed to map buffer: ${messageLength > 0 ? memory.readString(messageData, messageLength) : "Unknown"}`
          );
        }
        handle.done = true;
        setTimeout(() => jsCallback.close(), 0);
      },
      { args: ["u32", "ptr", "usize"], returns: "void" }
    );

    return encoder.encode(WGPUBufferMapCallbackInfo, {
      nextInChain: 0, mode: WGPUCallbackMode.WaitAnyOnly,
      callback: jsCallback.ptr, userdata1: 0, userdata2: 0,
    }).ptr;
  }

  createQueueWorkDoneCallback(encoder: StructEncoder, handle: CompletionHandle<void>): Pointer {
    const jsCallback = new JSCallback(
      (status: number) => {
        handle.done = true;
        if (status !== 1) handle.error = new Error(`Queue work done failed (status ${status})`);
        else handle.result = undefined;
        setTimeout(() => jsCallback.close(), 0);
      },
      { args: ["u32"], returns: "void" }
    );

    return encoder.encode(WGPUQueueWorkDoneCallbackInfo, {
      nextInChain: 0, mode: WGPUCallbackMode.WaitAnyOnly,
      callback: jsCallback.ptr, userdata1: 0, userdata2: 0,
    }).ptr;
  }

  createPopErrorScopeCallback(encoder: StructEncoder, handle: CompletionHandle<GPUError | null>): Pointer {
    const jsCallback = new JSCallback(
      (status: number, errorType: number, messageData: Pointer, messageLength: number) => {
        // status: 1=Success, 3=EmptyStack
        if (status === 1) {
          const msg = messageLength > 0 ? memory.readString(messageData, messageLength) : "";
          handle.result = this.makeGPUError(errorType, msg);
        } else if (status === 3) {
          handle.error = new Error("Error scope stack is empty");
        } else {
          handle.error = new Error(`Failed to pop error scope (status ${status})`);
        }
        handle.done = true;
        setTimeout(() => jsCallback.close(), 0);
      },
      { args: ["u32", "u32", "ptr", "usize"], returns: "void" }
    );

    return encoder.encode(WGPUPopErrorScopeCallbackInfo, {
      nextInChain: 0, mode: WGPUCallbackMode.WaitAnyOnly,
      callback: jsCallback.ptr, userdata1: 0, userdata2: 0,
    }).ptr;
  }

  createComputePipelineAsyncCallback(encoder: StructEncoder, handle: CompletionHandle<Pointer>): Pointer {
    const jsCallback = new JSCallback(
      (status: number, pipeline: Pointer, messageData: Pointer, messageLength: number) => {
        if (status === WGPUCreatePipelineAsyncStatus.Success) {
          handle.result = pipeline;
        } else {
          handle.error = new Error(
            `Create compute pipeline failed: ${messageLength > 0 ? memory.readString(messageData, messageLength) : "Unknown"}`
          );
        }
        handle.done = true;
        setTimeout(() => jsCallback.close(), 0);
      },
      { args: ["u32", "ptr", "ptr", "usize"], returns: "void" }
    );

    return encoder.encode(WGPUCreateComputePipelineAsyncCallbackInfo, {
      nextInChain: 0, mode: WGPUCallbackMode.WaitAnyOnly,
      callback: jsCallback.ptr, userdata1: 0, userdata2: 0,
    }).ptr;
  }

  createRenderPipelineAsyncCallback(encoder: StructEncoder, handle: CompletionHandle<Pointer>): Pointer {
    const jsCallback = new JSCallback(
      (status: number, pipeline: Pointer, messageData: Pointer, messageLength: number) => {
        if (status === WGPUCreatePipelineAsyncStatus.Success) {
          handle.result = pipeline;
        } else {
          handle.error = new Error(
            `Create render pipeline failed: ${messageLength > 0 ? memory.readString(messageData, messageLength) : "Unknown"}`
          );
        }
        handle.done = true;
        setTimeout(() => jsCallback.close(), 0);
      },
      { args: ["u32", "ptr", "ptr", "usize"], returns: "void" }
    );

    return encoder.encode(WGPUCreateRenderPipelineAsyncCallbackInfo, {
      nextInChain: 0, mode: WGPUCallbackMode.WaitAnyOnly,
      callback: jsCallback.ptr, userdata1: 0, userdata2: 0,
    }).ptr;
  }

  createCompilationInfoCallback(encoder: StructEncoder, handle: CompletionHandle<GPUCompilationInfo>): Pointer {
    const jsCallback = new JSCallback(
      (status: number, compilationInfo: Pointer) => {
        handle.done = true;
        if (status === WGPUCompilationInfoRequestStatus.Success && compilationInfo) {
          handle.result = {
            __brand: "GPUCompilationInfo",
            messages: this.parseCompilationInfo(compilationInfo),
          } as GPUCompilationInfo;
        } else {
          handle.result = { __brand: "GPUCompilationInfo", messages: [] } as GPUCompilationInfo;
        }
        setTimeout(() => jsCallback.close(), 0);
      },
      { args: ["u32", "ptr"], returns: "void" }
    );

    return encoder.encode(WGPUCompilationInfoCallbackInfo, {
      nextInChain: 0, mode: WGPUCallbackMode.WaitAnyOnly,
      callback: jsCallback.ptr, userdata1: 0, userdata2: 0,
    }).ptr;
  }

  private makeGPUError(type: number, message: string): GPUError | null {
    if (type === WGPUErrorType.NoError) return null;
    const m = message || "Unknown GPU error";
    if (type === WGPUErrorType.Validation) return new GPUValidationError(m);
    if (type === WGPUErrorType.OutOfMemory) return new GPUOutOfMemoryError(m);
    if (type === WGPUErrorType.Internal) return new GPUInternalError(m);
    return new GPUValidationError(m);
  }

  private parseCompilationInfo(infoPtr: Pointer): GPUCompilationMessage[] {
    const messages: GPUCompilationMessage[] = [];
    const infoView = new DataView(new Uint8Array(memory.read(infoPtr, 24)).buffer);
    const messageCount = Number(infoView.getBigUint64(8, true));
    const messagesPtr = Number(infoView.getBigUint64(16, true)) as unknown as Pointer;
    if (messageCount === 0 || !messagesPtr) return messages;

    const MESSAGE_SIZE = 88;
    for (let i = 0; i < messageCount; i++) {
      const msgPtr = (Number(messagesPtr) + i * MESSAGE_SIZE) as unknown as Pointer;
      const msgView = new DataView(new Uint8Array(memory.read(msgPtr, MESSAGE_SIZE)).buffer);
      const msgText = memory.readString(
        Number(msgView.getBigUint64(8, true)) as unknown as Pointer,
        Number(msgView.getBigUint64(16, true))
      );
      const type = msgView.getUint32(24, true) === WGPUCompilationMessageType.Error ? "error"
        : msgView.getUint32(24, true) === WGPUCompilationMessageType.Warning ? "warning" : "info";
      messages.push({
        message: msgText, type,
        lineNum: Number(msgView.getBigUint64(32, true)),
        linePos: Number(msgView.getBigUint64(40, true)),
        offset: Number(msgView.getBigUint64(48, true)),
        length: Number(msgView.getBigUint64(56, true)),
      } as GPUCompilationMessage);
    }
    return messages;
  }
}

let globalRegistry: CallbackRegistry | null = null;

export function getCallbackRegistry(): CallbackRegistry {
  if (!globalRegistry) globalRegistry = new CallbackRegistry();
  return globalRegistry;
}
