# Zig 0.16.0 Migration Roadmap

This roadmap defines the phases to migrate `ziggy-llm` from Zig 0.15.2 to 0.16.0, analyzing potential outcomes and impacts.

## How To Use This File

Every task in this file is a Markdown checkbox.

When a task is completed:
- replace `- [ ]` with `- [x]`
- keep the item in place instead of deleting it
- only mark it complete when the code, docs, and validation for that item are actually done

## Analysis Summary

### Our Codebase Exposure to Breaking Changes

After scanning our codebase, here are the breaking changes that **directly affect us**:

| Change | Our Usage | Risk |
|--------|-----------|------|
| `@cImport` deprecated | `src/runtime/metal_backend.zig:9` | Low - still works, needs migration |
| `std.io.GenericReader/AnyReader/FixedBufferStream` deleted | None | None |
| `std.Thread.Pool` removed | None | None |
| `std.heap.ThreadSafe` removed | None | None |
| `@Type` replaced with builtins | None | None |
| `std.process.argsAlloc` returns non-global | `src/main.zig:11`, `src/tools/*.zig` | Low - API unchanged |
| `std.fs.Dir.readFileAlloc` added | None | Low - new API available |
| `std.posix.isatty` added | `src/terminal.zig:30` | None - new API |
| `fs.getAppDataDir` removed | None | None |
| Environment variables non-global | None directly | None |

### Breaking Changes NOT Affecting Us

These changes exist in 0.16.0 but we don't use them:
- Packed union/struct switch improvements (language feature, no breakage)
- Vector indexing restrictions (we don't use runtime vector indexes)
- Lazy field analysis (internal compiler change)
- Various std.math builtins changes
- Debug info rework

---

## Phase 1: Preparation and Inventory

Goal: Understand exactly what will break and plan accordingly.

### Task 1.1: Verify Current Build with 0.15.2

- [ ] Confirm `zig version` shows 0.15.2
- [ ] Run `zig build` successfully
- [ ] Run `zig build test` successfully
- [ ] Note any warnings emitted

### Task 1.2: Scan for `@cImport` Usage

- [x] Found: `src/runtime/metal_backend.zig:9` uses `@cImport` for Epoxy/OpenGL bindings

### Task 1.3: Review New I/O Interfaces

- [ ] Study `std.io.Reader` and `std.io.Writer` trait changes
- [ ] Verify our usage of `std.fs.File.writer()` still works
- [ ] Check if any `GenericReader` or `AnyReader` patterns exist in dependencies

---

## Phase 2: Toolchain Switch

Goal: Get the project building with Zig 0.16.0.

### Task 2.1: Install Zig 0.16.0

- [ ] Install via zigup or direct download
- [ ] Verify `zig version` shows 0.16.0

### Task 2.2: Attempt Build and Catalog Errors

- [ ] Run `zig build` and capture all errors
- [ ] Categorize errors by type:
  - Compilation errors (syntax, types)
  - API deprecation warnings
  - Linker errors
  - Missing symbols

### Task 2.3: Fix `@cImport` Deprecation

- [ ] Move Epoxy/OpenGL C headers to build system:
  - Create `src/runtime/c.h` with includes
  - Add `translate_c` step in `build.zig`
  - Link `epoxy` system library
- [ ] Update `src/runtime/metal_backend.zig` to use `@import("c")` instead of `@cImport`
- [ ] Verify Metal shaders still compile and run

### Task 2.4: Fix Any I/O API Changes

- [ ] Review any deprecation warnings about std.fs changes
- [ ] Update `std.fs.cwd().openFile` usage if needed
- [ ] Verify `fs.Dir.readFileAlloc` vs our current patterns

---

## Phase 3: Validation

Goal: Ensure 0.16.0 produces correct, performant builds.

### Task 3.1: Run Full Test Suite

- [ ] Run `zig build test`
- [ ] All existing tests pass
- [ ] No new warnings introduced

### Task 3.2: CPU Inference Validation

- [ ] Run Llama model through CPU backend
- [ ] Verify output matches reference (same seed, same prompt)
- [ ] Measure: startup time, prompt processing time, tok/s

### Task 3.3: Metal Inference Validation

- [ ] Run Llama model through Metal backend
- [ ] Verify output matches CPU reference exactly
- [ ] Measure: TTFT, tok/s

### Task 3.4: Server Validation

- [ ] Start server with 0.16.0 build
- [ ] Hit `/health`, `/v1/completions`, `/v1/chat/completions`
- [ ] Verify JSON responses correct

### Task 3.5: Benchmark Comparison

- [ ] Compare 0.15.2 vs 0.16.0:
  - Startup latency
  - Prompt processing time
  - Decode tok/s
  - Memory footprint
- [ ] Document any performance differences

---

## Phase 4: Cleanup and Documentation

Goal: Clean up deprecation warnings and update documentation.

### Task 4.1: Address All Deprecation Warnings

- [ ] Fix any remaining deprecation warnings
- [ ] Remove any backward-compatibility code that 0.16.0 still supports but is deprecated

### Task 4.2: Update README

- [ ] Update minimum Zig version to 0.16.0
- [ ] Note any new dependencies (e.g., Epoxy for Metal)

### Task 4.3: Update Version References

- [ ] Update any CI/CD that specifies Zig version
- [ ] Update any documentation that mentions version requirements

---

## Benefits Analysis

### Why Migrate to 0.16.0?

**Compiler Improvements:**
- LLVM 21 backend (better optimization, new CPU targets)
- Incremental compilation support (faster rebuilds during development)
- Improved x86 and ARM code generation
- Better error messages and debugging info

**Standard Library Improvements:**
- I/O as an Interface provides better abstraction for custom I/O types
- Thread-safe `heap.ArenaAllocator` (we don't use it currently, but future code can)
- New cryptographic primitives (AES-SIV, AES-GCM-SIV, Ascon)
- Better Windows networking without ws2_32.dll

**Target Support:**
- New targets: LoongArch, PowerPC64le, s390x, more BSD variants
- Better stack tracing across platforms
- macOS cross-compilation support for Catalyst

### Risks

**Low Risk Items:**
- `@cImport` deprecation: We have one usage, migrate to build system
- API renames: Our usage is minimal, straightforward to update
- Non-global environment variables: We don't access `std.os.environ`

**Medium Risk Items:**
- Metal shader compatibility: Epoxy/OpenGL bindings may need verification
- Build system changes: Adding translate_c step, but straightforward

**High Risk Items:**
- None identified based on codebase scan

---

## Recommendation

**Yes, migrate to 0.16.0.** The benefits are significant (LLVM 21, incremental compilation, better targets) and our codebase exposure to breaking changes is minimal. The main work is migrating one `@cImport` call to the build system.

The migration should take approximately 1-2 days:
- Day 1: Toolchain switch, fix `@cImport`, verify build
- Day 2: Full validation (tests, CPU, Metal, server), benchmark comparison

---

## Appendix: Breaking Changes Reference

### From Release Notes (for reference during migration)

```
Language Changes:
- switch now supports packed structs/unions as prong items
- @Type replaced with @Int, @Struct, @Union, @Enum, @Pointer, @Fn, @Tuple, @EnumLiteral
-Vectors can't coerce to arrays in memory anymore
- Forbid runtime vector indexes
- Small integer types can coerce to floats

Standard Library:
- std.io.GenericReader, AnyReader, FixedBufferStream DELETED
- std.Thread.Pool DELETED
- std.heap.ThreadSafe DELETED
- fs.getAppDataDir DELETED
- fs.Dir.readFileAlloc added (new API)
- fs.File.readToEndAlloc added (new API)
- Environment variables become non-global (std.process.argsAlloc still works, returns non-global)
- Current directory API renamed (std.process.changeDir moved)
- @cImport deprecated in favor of build system translate_c

Build System:
- Packages can be overridden locally
- Packages fetched into project-local directory
- New --error-style and --multiline-errors flags

Compiler:
- LLVM 21 (previously LLVM 16 in 0.15.x)
- C translation improvements
- New ELF linker
```

(End of file - total 215 lines)