# lampvm-toolchain

This is the assembler for [lampvm](https://github.com/glowingstone124/lamp-vm).

# Compilation

This project is written in Rust, so you can compile it easily using `cargo`:

```bash
cargo build --release
```

# Basic Usage
The assembler requires a config.yml file to be present in the current working directory.

By default, the configuration should look like this:

```yml
reg_count: 8
reg_prefix: "r"

macros:
  MEM_SIZE: 4194304
  IO_SIZE: 256
  IVT_SIZE: 256
  CALL_STACK_SIZE: 256
  DATA_STACK_SIZE: 256
  IVT_BASE: 0x0000
  CALL_STACK_BASE: 0x800
  DATA_STACK_BASE: 0x1000
  PROGRAM_BASE: 0x1818
  FB_WIDTH: 640
  FB_HEIGHT: 480
  FB_BPP: 4
  FB_SIZE: 1228800
  FB_BASE: 0x2D4000
```

These settings are consistent with the VM's default configuration. In most cases, you do not need to change them. However, if you have a customized VM, please update this file to match your modifications.