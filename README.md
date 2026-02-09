# Triton Hidden Kernel Parameters: Global Scratch & Profile Scratch

## Executive Summary

Triton's compiler pipeline transparently injects **two hidden pointer arguments** into every compiled kernel function — **global scratch memory** and **profile scratch memory** — that are completely invisible in the user's Python `@triton.jit` function signature. These pointers are appended to the kernel's parameter list during the MLIR→LLVM IR lowering, and the Python-side launcher allocates the GPU memory and passes the pointers at launch time. The user never sees them.

---

## 1. Architecture Overview

Three hidden arguments are managed by the system (though shared memory is handled differently for kernels):

| Constant | Value | Purpose |
|---|---|---|
| `kSharedMemoryOffset` | `-3` | Shared memory base (device functions only) |
| `kGlobalScratchBufferOffset` | `-2` | Global scratch memory pointer |
| `kProfileScratchBufferOffset` | `-1` | Profile scratch memory pointer |

These constants are defined in `include/triton/Conversion/TritonGPUToLLVM/Utility.h` (lines ~323-335) and are used as negative offsets from `funcOp.getNumArguments()` to retrieve the hidden arguments.

For **kernel functions** (public entry points), shared memory uses a global symbol rather than a function argument, but global scratch and profile scratch are still passed as explicit function arguments appended to the end of the parameter list.

---

## 2. Compiler-Side: How Hidden Parameters Get Added

### 2.1 The `GlobalScratchAllocOp` MLIR Operation

Various compiler passes introduce `ttg.global_scratch_alloc` operations when they need device-side global memory workspace. Each alloc specifies:
- `nbytes`: size in bytes
- `alignment`: required alignment
- `backend`: either `"default"` (compiler-internal) or `"proton"` (profiling)

**Who creates these ops:**
- **FP Sanitizer** (`FpSanitizer.cpp`): For floating-point instrumentation scratch buffers
- **Concurrency Sanitizer** (ConSan): For buffer visibility tracking
- **Proton Profiler**: For profiling data collection with `backend = "proton"`

Example MLIR:
```mlir
%0 = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 100 : i32} : !tt.ptr<i8>
%1 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
```

### 2.2 The Allocation Pass: `TritonGPUGlobalScratchAllocationPass`

**File:** `lib/Conversion/TritonGPUToLLVM/GlobalScratchMemoryAllocation.cpp`

This pass runs over the entire module and computes the total global scratch memory layout:

1. **Recursive call-graph walk**: For each function, it first recurses into any called functions that don't yet have their scratch size computed.

2. **Linear bump allocator**: Walks all operations in post-order. For each `GlobalScratchAllocOp` (with `backend == "default"` only), it rounds up the current offset to the requested alignment and assigns `ttg.global_scratch_memory_offset` to the op.

3. **Propagation to module**: After processing all functions, the public kernel's scratch size and alignment are promoted to module-level attributes:
   - `ttg.global_scratch_memory_size` — total bytes needed per CTA
   - `ttg.global_scratch_memory_alignment` — maximum alignment required

```cpp
// From GlobalScratchMemoryAllocation.cpp
static void allocateGMem(Operation *parentOp, ...) {
  // ... for each GlobalScratchAllocOp with backend == "default":
  offset = roundUp(offset, align);
  op->setAttr("ttg.global_scratch_memory_offset",
              builder.getI32IntegerAttr(offset));
  offset += nbytes;
}
```

After the pass, the MLIR module looks like:
```mlir
module attributes {
  ttg.global_scratch_memory_alignment = 128 : i32,
  ttg.global_scratch_memory_size = 256 : i32
}
```

### 2.3 Profile Scratch: Separate Allocation Pass

Profile scratch uses its own pass (`allocate-proton-global-scratch-buffer`) which handles `GlobalScratchAllocOp` with `backend = "proton"`. It produces:
- `ttg.profile_scratch_memory_size`
- `ttg.profile_scratch_memory_alignment`

These are distinct from the global scratch attributes, allowing the two systems to be sized independently.

### 2.4 `amendFuncOp()`: Injecting Hidden Arguments into Function Signatures

**File:** `lib/Conversion/TritonGPUToLLVM/Utility.cpp` (lines ~1692-1745)

This is the core function that physically modifies every Triton function's signature to include the hidden arguments:

```cpp
triton::FuncOp amendFuncOp(triton::FuncOp funcOp, ...) {
  auto globalPtrTy = LLVM::LLVMPointerType::get(ctx, 1);   // address space 1 = global
  auto profilePtrTy = LLVM::LLVMPointerType::get(ctx, 1);  // address space 1 = global

  auto amendedInputTy = llvm::to_vector<4>(funcTy.getInputs());

  if (!isKernel) {
    amendedInputTy.push_back(sharedPtrTy);  // shared memory (device functions only)
  }
  amendedInputTy.push_back(globalPtrTy);    // always: global scratch
  amendedInputTy.push_back(profilePtrTy);   // always: profile scratch

  // Create new function type and update the function
  auto amendedFuncTy = FunctionType::get(ctx, amendedInputTy, funcTy.getResults());
}
```

**Key point**: `amendFuncOp` runs during the `convert-triton-gpu-to-llvm` pass. After it executes, a Triton function like:
```mlir
tt.func @my_kernel(%arg0: !tt.ptr<f32>)
```
becomes:
```mlir
llvm.func @my_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>)
//                    ^user arg        ^global scratch      ^profile scratch
```

This is confirmed by the test `global_scratch_to_llvm.mlir`:
```mlir
// CHECK-LABEL: @global_scratch_alloc_warpgroup(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>)
tt.func @global_scratch_alloc_warpgroup() {
```
A function with **zero** user arguments gets **two** hidden pointer arguments.

### 2.5 `FuncOpConversion`: The LLVM Lowering Pattern

**File:** `lib/Conversion/TritonGPUToLLVM/FuncOpToLLVM.cpp` (lines 8-26)

The critical documentation comment:

```
NOTE: [Additional Function Arguments]
Triton patches additional arguments to the function signature to support
(1) shared memory, (2) global scratch memory, and (3) profile scratch memory.
To support use of shared memory and global scratch memory inside of a
function, the caller allocates a single large block of the relevant memory
and calls the function with these extra arguments at the end.
Profile scratch memory is only used when the function is instrumented for
profiling.

For the kernel function itself, the shared memory base is a global symbol
so no additional function argument is required but global scratch memory
allocation is still passed in as the last argument. Though here the scratch
memory is shared between all programs, so a linear offset based on the
program id is required to get the local scratch base.
```

### 2.6 `getGlobalScratchPtr()`: How the Kernel Accesses Its Scratch

**File:** `lib/Conversion/TritonGPUToLLVM/Utility.cpp` (lines ~1143-1199)

This function is called whenever a `GlobalScratchAllocOp` is lowered to LLVM IR. It retrieves the hidden function argument and computes the per-CTA pointer:

**For device functions (non-kernels):** Simply returns the argument directly (the caller already computed the correct base):
```cpp
auto gmemBase = funcOp.getArgument(funcOp.getNumArguments() + kGlobalScratchBufferOffset);
return gep(ptrTy, i8_ty, gmemBase, allocOffset);
```

**For kernel functions:** Must compute a per-program offset because all CTAs share the same global buffer:
```cpp
// Compute linear CTA ID: linearId = z * (dimY * dimX) + y * dimX + x
Value linearId = gridIdx[2];
for (int k = 0; k < 2; ++k) {
    linearId = add(gridIdx[1-k], mul(linearId, gridDim[1-k]));
}
// For multi-CTA clusters:
if (numCTAs > 1) {
    linearId = mul(linearId, i32_val(numCTAs));
    linearId = add(linearId, targetInfo.getClusterCTAId(rewriter, loc));
}
// Final pointer: base + linearId * allocSize + allocOffset
Value offset = mul(linearId, i32_val(allocSize));
if (allocOffset) offset = add(offset, allocOffset);
return gep(LLVMPointerType::get(ctx, 1), i8_ty, gmemBase, offset);
```

This means the **total GPU allocation** for global scratch is:
$$\text{total} = \text{global\_scratch\_size} \times \text{gridX} \times \text{gridY} \times \text{gridZ} \times \text{numCTAs}$$

### 2.7 `getProfileScratchPtr()`: Profile Scratch Access

**File:** `lib/Conversion/TritonGPUToLLVM/Utility.cpp` (lines ~1201-1209)

Much simpler — just reads the last function argument:
```cpp
Value getProfileScratchPtr(..., FunctionOpInterface funcOp) {
  return funcOp.getArgument(funcOp.getNumArguments() + kProfileScratchBufferOffset);
}
```

Note the FIXME comment: *"This is broken when we have device functions, we need to implement proper calling convention"*.

### 2.8 `GlobalScratchAllocOpConversion`: Lowering to LLVM GEP

**File:** `lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp` (lines ~175-189)

Each `ttg.global_scratch_alloc` op is converted to an LLVM GEP instruction:
```cpp
LogicalResult matchAndRewrite(GlobalScratchAllocOp op, ...) {
    auto opOffset = op->getAttr("ttg.global_scratch_memory_offset");
    Value ptr = LLVM::getGlobalScratchPtr(loc, rewriter, *targetInfo, funcOp,
                                           b.i32_val(opOffset));
    rewriter.replaceOp(op, ptr);
}
```

### 2.9 Call Propagation: `CallOpConversion`

**File:** `lib/Conversion/TritonGPUToLLVM/ControlFlowOpToLLVM.cpp` (line ~100+)

When converting `tt.call` to LLVM, the scratch pointers are automatically forwarded:
```cpp
// Append hidden args to the promoted operands
promotedOperands.push_back(LLVM::getGlobalScratchPtr(...));
promotedOperands.push_back(LLVM::getProfileScratchPtr(...));
```

---

## 3. Python-Side: Metadata Flow and Kernel Launch

### 3.1 Compilation: Metadata Extraction

**File:** `python/triton/compiler/compiler.py`

After MLIR compilation, the module-level attributes (`ttg.global_scratch_memory_size`, `ttg.profile_scratch_memory_size`) are extracted into the kernel's metadata object. This metadata is serialized and cached alongside the compiled binary (cubin/hsaco).

Key metadata fields:
- `metadata.global_scratch_size` — bytes per CTA for global scratch
- `metadata.profile_scratch_size` — bytes per CTA for profile scratch
- `metadata.shared` — shared memory size

### 3.2 `CompiledKernel._init_handles()`: Loading the Binary

**File:** `python/triton/compiler/compiler.py` (line ~465+)

```python
def _init_handles(self):
    # Loads the compiled binary (cubin) onto the GPU
    self.module, self.function, ... = driver.active.utils.load_binary(
        self.name, self.kernel, self.metadata.shared, device)
```

### 3.3 `JITFunction.run()`: The Kernel Launch Path

**File:** `python/triton/runtime/jit.py` (lines ~708-763)

```python
def run(self, *args, grid, warmup, **kwargs):
    # ... argument binding, specialization, compilation ...

    # Launch:
    kernel.run(grid_0, grid_1, grid_2, stream,
               kernel.function, kernel.packed_metadata,
               launch_metadata,
               knobs.runtime.launch_enter_hook,
               knobs.runtime.launch_exit_hook,
               *bound_args.values())
```

The `kernel.run` here is a C-extension function generated by `make_launcher`. The C code:
1. Parses all user arguments from the Python call
2. Reads `global_scratch_size` and `profile_scratch_size` from the packed metadata
3. If `global_scratch_size > 0`: calls the user-registered allocator (`triton.set_allocator(fn)`) to allocate `global_scratch_size * grid_x * grid_y * grid_z` bytes of GPU memory
4. If `profile_scratch_size > 0`: calls the profile allocator (`triton.set_profile_allocator(fn)`) similarly
5. Appends the resulting `CUdeviceptr` values to the kernel parameter array
6. Calls `cuLaunchKernel` with all parameters (user args + global_scratch_ptr + profile_scratch_ptr)

### 3.4 The Allocator Callbacks

**File:** `python/triton/runtime/_allocation.py`

Two separate allocator registries:

```python
# For global scratch (fpsan, consan, etc.)
_allocator: ContextVar[Allocator] = ContextVar("_allocator", default=_NULL_ALLOCATOR)

def set_allocator(allocator: Allocator) -> None:
    _allocator.set(allocator)

# For profile scratch (Proton)
_profile_allocator = _AllocatorWrapper(_NULL_ALLOCATOR)

def set_profile_allocator(allocator: Optional[Allocator]) -> None:
    """Called before kernel launch for kernels that require
    additional global memory workspace."""
    _profile_allocator.set(allocator)
```

User code must register an allocator if using features that need global scratch:
```python
def my_allocator(size: int, alignment: int, stream: int):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(my_allocator)
```

### 3.5 AOT Compilation: Explicit Handling

**File:** `python/triton/tools/compile.py` (lines ~143-199)

The AOT compiler explicitly accounts for the two hidden arguments:
```python
# Check if scratch is needed (not yet supported for AOT)
if metadata.global_scratch_size > 0:
    raise RuntimeError("AOT compiling kernels with global scratch "
                       "requirements is not yet implemented")
if metadata.profile_scratch_size > 0:
    raise RuntimeError("AOT compiling kernels with profile scratch "
                       "requirements is not yet implemented")

# In the C stub template:
"arg_pointers": [...user_args..., "&global_scratch", "&profile_scratch"],
"num_args": len(user_args) + 2,  # +2 for global and profile scratch
```

---

## 4. End-to-End Flow Diagram

```
User writes:                    @triton.jit
                                def kernel(x_ptr, n):
                                    ...

                                         │
                                         ▼
1. MLIR Passes              ttg.global_scratch_alloc {nbytes=100, alignment=8}
   (fpsan/consan/proton)     ttg.global_scratch_alloc {nbytes=128, alignment=128,
                                                       backend="proton"}
                                         │
                                         ▼
2. Allocation Pass           ttg.global_scratch_memory_size = 100 (module attr)
   (GlobalScratchAllocation) ttg.global_scratch_memory_offset = 0 (per-op attr)
                             ttg.profile_scratch_memory_size = 128 (module attr)
                                         │
                                         ▼
3. FuncOp → LLVM             amendFuncOp() adds 2 ptr<1> args:
   (FuncOpConversion)        @kernel(%x: ptr, %n: i32,
                                     %global: ptr<1>, %profile: ptr<1>)
                                         │
                                         ▼
4. Op Lowering               GlobalScratchAllocOp → getGlobalScratchPtr()
   (MemoryOpToLLVM)          → GEP from %global with per-CTA offset
                                         │
                                         ▼
5. PTX Codegen               .param .u64 kernel_param_0   // x_ptr
                             .param .u32 kernel_param_1   // n
                             .param .u64 kernel_param_2   // global_scratch  ← HIDDEN
                             .param .u64 kernel_param_3   // profile_scratch ← HIDDEN
                                         │
                                         ▼
6. Python Metadata           metadata.global_scratch_size = 100
                             metadata.profile_scratch_size = 128
                                         │
                                         ▼
7. Kernel Launch             allocator(100 * grid_total, 8, stream) → CUdeviceptr
   (C extension)             profile_alloc(128 * grid_total, 128, stream) → CUdeviceptr
                             cuLaunchKernel(...,
                                params=[x, n, scratch, profile])
```

---

## 5. Extensibility: Adding a Custom Hidden Buffer

The mechanism is architecturally clean and could be extended for custom data-collection buffers. The steps would be:

1. **Add a new constant** in `Utility.h`:
   ```cpp
   constexpr int kCustomBufferOffset = -4;
   // Shift existing: kSharedMemoryOffset from -3 to -4, etc.
   ```

2. **Extend `amendFuncOp()`** to push another `ptr<1>` argument.

3. **Create an allocation pass** (similar to `TritonGPUGlobalScratchAllocationPass`) that walks operations and computes sizes, setting module-level attributes like `ttg.custom_buffer_size`.

4. **Add a `getCustomBufferPtr()` function** following the pattern of `getGlobalScratchPtr()` (with or without per-CTA offset calculation depending on use case).

5. **On the Python side**: Add a new allocator callback (like `set_allocator` / `set_profile_allocator`) and extend the C launcher to read the new metadata field and allocate/pass the buffer.

6. **Extend `CallOpConversion`** to forward the new pointer to called functions.

The profile scratch system (`backend = "proton"`) is itself already a proof-of-concept of this extensibility — it was added after global scratch using the exact same pattern but with separate attributes, a separate allocation pass, and a separate Python-side allocator.

---

## 6. Key Source Files Reference

| File | Purpose |
|---|---|
| `lib/Conversion/TritonGPUToLLVM/FuncOpToLLVM.cpp` | `FuncOpConversion` + NOTE comment |
| `lib/Conversion/TritonGPUToLLVM/Utility.cpp` | `amendFuncOp`, `getGlobalScratchPtr`, `getProfileScratchPtr` |
| `include/triton/Conversion/TritonGPUToLLVM/Utility.h` | Constants: `kProfileScratchBufferOffset`, `kGlobalScratchBufferOffset`, `kSharedMemoryOffset` |
| `lib/Conversion/TritonGPUToLLVM/GlobalScratchMemoryAllocation.cpp` | `TritonGPUGlobalScratchAllocationPass`, `allocateGMem()` |
| `lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp` | `GlobalScratchAllocOpConversion` |
| `lib/Conversion/TritonGPUToLLVM/ControlFlowOpToLLVM.cpp` | `CallOpConversion` (forwards scratch ptrs) |
| `lib/Dialect/TritonInstrument/Transforms/FpSanitizer.cpp` | Creates `GlobalScratchAllocOp` for FP sanitization |
| `python/triton/compiler/compiler.py` | `CompiledKernel`, `_init_handles()`, metadata extraction |
| `python/triton/runtime/jit.py` | `JITFunction.run()` — kernel launch path |
| `python/triton/runtime/_allocation.py` | `set_allocator()`, `set_profile_allocator()` |
| `python/triton/tools/compile.py` | AOT compilation handling (+2 for scratch args) |
| `python/triton/knobs.py` | `proton_knobs.profile_buffer_size` (64 MB default) |
| `test/TritonGPU/global_scratch_alloc.mlir` | Tests for allocation pass |
| `test/TritonGPU/global_scratch_to_llvm.mlir` | Tests for LLVM lowering (shows hidden args) |
| `test/Proton/allocate_global_scratch_buffer.mlir` | Tests for Proton profile scratch allocation |
