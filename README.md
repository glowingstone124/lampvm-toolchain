# lampvm-toolchain

Currently aligned VM version: [c4bbaec62cddf3ae512e699b7faaa1fe90e74253](https://github.com/glowingstone124/lamp-vm/commit/c4bbaec62cddf3ae512e699b7faaa1fe90e74253)

This is the development toolchain for [lampvm](https://github.com/glowingstone124/lamp-vm).

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

For compile C code:
```bash
./bin cc main.c
```

Detailed information for C is in [document](/docs/CCompiler.md)

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
  FB_BASE: 0x2D4000
```

These settings are consistent with the VM's default configuration. In most cases, you do not need to change them. However, if you have a customized VM, please update this file to match your modifications.

# Roadmap

✅ Support all instructions in lampVM

✅ Support labels

Support a subset of C
