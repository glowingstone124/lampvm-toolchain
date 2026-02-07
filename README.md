# lampvm-toolchain

Currently aligned VM version: [f7cf3534eed81454ed3f5db0a9097ed72920533d](https://github.com/glowingstone124/lamp-vm/commit/f7cf3534eed81454ed3f5db0a9097ed72920533d)

This is the development toolchain for [lampvm](https://github.com/glowingstone124/lamp-vm).

> The built-in C compiler (`cc`) is **deprecated**.
> Please use the [**LLVM backend**](https://github.com/glowingstone124/lamp-vm) for new projects.
> The `cc` path is kept only for compatibility and may be removed in a future release.

# Compilation

This project is written in Rust, so you can compile it easily using `cargo`:

```bash
cargo build --release
```

# Basic Usage

For assemble:
```bash
./bin asm --input program.asm --output program.bin
```

For compile C code (deprecated):
```bash
./bin cc main.c
```

Detailed information for the deprecated C compiler is in [document](/docs/CCompiler.md).

The assembler requires a config.yml file to be present in the current working directory.

By default, the configuration should look like this:

```yml
reg_count: 32
reg_prefix: "r"

macros:
  MEM_SIZE: 4194304
  IO_SIZE: 256
  IVT_SIZE: 256
  CALL_STACK_SIZE: 256
  DATA_STACK_SIZE: 256
  ISR_STACK_SIZE: 256
  IVT_BASE: 0x0000
  CALL_STACK_BASE: 0x800
  DATA_STACK_BASE: 0x1000
  PROGRAM_BASE: 0x201C
  FB_WIDTH: 640
  FB_HEIGHT: 480
  FB_BPP: 4
  FB_SIZE: 1228800
  FB_BASE: 0x400000
```

These settings are consistent with the VM's default configuration. In most cases, you do not need to change them. However, if you have a customized VM, please update this file to match your modifications.

# Output Format (Single File)

The toolchain now emits a single binary file. The file layout is:

- Header: 6 little‑endian `u32` values
  1. `TEXT_BASE`
  2. `TEXT_SIZE` (bytes)
  3. `DATA_BASE`
  4. `DATA_SIZE` (bytes)
  5. `BSS_BASE`
  6. `BSS_SIZE` (bytes)
- Text section: instruction stream, `u64` little‑endian
- Data section: raw bytes

No separate `.data` or `.layout` files are produced.

# Roadmap

✅ Support all instructions in lampVM

✅ Support labels

Support a subset of C
