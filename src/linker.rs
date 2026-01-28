use std::collections::HashMap;
use std::fs;

use crate::assemble::assemble::AssembledProgram;
use crate::assemble::config::ArchConfig;
use crate::assemble::inst::inst;
use crate::assemble::object::{ObjectFile, RelocKind, Section};
use crate::assemble::opcode::Opcode;

fn align_up_u32(v: u32, align: u32) -> u32 {
    if align == 0 {
        return v;
    }
    (v + align - 1) & !(align - 1)
}

fn read_object(path: &str) -> ObjectFile {
    let text = fs::read_to_string(path).expect("Failed to read object file");
    serde_yaml::from_str(&text).expect("Invalid object file format")
}

fn patch_text_imm(text: &mut [u64], inst_index: usize, value: u32) {
    let inst = text[inst_index];
    text[inst_index] = (inst & 0xFFFF_FFFF_0000_0000) | (value as u64);
}

fn patch_data_bytes(data: &mut [u8], offset: usize, kind: RelocKind, value: u32) {
    match kind {
        RelocKind::Abs8 => {
            if value > u8::MAX as u32 {
                panic!("Relocation value out of range for Abs8: {}", value);
            }
            if offset >= data.len() {
                panic!("Data relocation out of range");
            }
            data[offset] = value as u8;
        }
        RelocKind::Abs16 => {
            if value > u16::MAX as u32 {
                panic!("Relocation value out of range for Abs16: {}", value);
            }
            if offset + 1 >= data.len() {
                panic!("Data relocation out of range");
            }
            data[offset..offset + 2].copy_from_slice(&(value as u16).to_le_bytes());
        }
        RelocKind::Abs32 => {
            if offset + 3 >= data.len() {
                panic!("Data relocation out of range");
            }
            data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
}

pub fn link_objects_from_memory(objects: &[ObjectFile], arch: &ArchConfig) -> AssembledProgram {
    if objects.is_empty() {
        panic!("No input objects provided");
    }

    let objects = objects.to_vec();

    let text_base = *arch.macros.get("PROGRAM_BASE").expect("No PROGRAM_BASE found");
    let mem_size = *arch.macros.get("MEM_SIZE").expect("No MEM_SIZE found");

    let entry_size = 3u32 * 8;

    let mut text_offsets: Vec<u32> = Vec::new();
    let mut text_cursor = entry_size;
    for obj in &objects {
        text_offsets.push(text_cursor);
        text_cursor += (obj.text.len() as u32) * 8;
    }

    let data_base = text_base + text_cursor;

    let mut data_offsets: Vec<u32> = Vec::new();
    let mut data_cursor = 0u32;
    for obj in &objects {
        data_cursor = align_up_u32(data_cursor, 4);
        data_offsets.push(data_cursor);
        data_cursor += obj.data.len() as u32;
    }

    let bss_base = data_base + data_cursor;
    let mut bss_offsets: Vec<u32> = Vec::new();
    let mut bss_cursor = 0u32;
    for obj in &objects {
        bss_cursor = align_up_u32(bss_cursor, 4);
        bss_offsets.push(bss_cursor);
        bss_cursor += obj.bss_size;
    }

    let mut global_symbols: HashMap<String, u32> = HashMap::new();
    let mut local_symbols: Vec<HashMap<String, u32>> = Vec::new();

    for (idx, obj) in objects.iter().enumerate() {
        let mut locals: HashMap<String, u32> = HashMap::new();
        for sym in &obj.symbols {
            if let Some(section) = sym.section {
                let base = match section {
                    Section::Text => text_base + text_offsets[idx],
                    Section::Data => data_base + data_offsets[idx],
                    Section::Bss => bss_base + bss_offsets[idx],
                };
                let addr = base + sym.offset;
                locals.insert(sym.name.clone(), addr);
                if sym.global {
                    if let Some(prev) = global_symbols.insert(sym.name.clone(), addr) {
                        panic!("Duplicate global symbol: {} (prev at 0x{:08X})", sym.name, prev);
                    }
                }
            }
        }
        local_symbols.push(locals);
    }

    let mut text: Vec<u64> = Vec::new();
    text.push(inst(Opcode::OP_MOVI as u8, 30, 0, 0, mem_size));
    text.push(inst(Opcode::OP_CALL as u8, 0, 0, 0, 0));
    text.push(inst(Opcode::OP_HALT as u8, 0, 0, 0, 0));
    for obj in &objects {
        text.extend_from_slice(&obj.text);
    }

    let mut data = vec![0u8; data_cursor as usize];
    for (idx, obj) in objects.iter().enumerate() {
        let start = data_offsets[idx] as usize;
        let end = start + obj.data.len();
        data[start..end].copy_from_slice(&obj.data);
    }

    for (idx, obj) in objects.iter().enumerate() {
        for reloc in &obj.relocations {
            let sym_addr = local_symbols[idx]
                .get(&reloc.symbol)
                .copied()
                .or_else(|| global_symbols.get(&reloc.symbol).copied())
                .unwrap_or_else(|| panic!("Undefined symbol: {}", reloc.symbol));
            let value_i64 = (sym_addr as i64) + (reloc.addend as i64);
            if value_i64 < 0 || value_i64 > u32::MAX as i64 {
                panic!(
                    "Relocation overflow for {}: {}",
                    reloc.symbol, value_i64
                );
            }
            let value = value_i64 as u32;

            match reloc.section {
                Section::Text => {
                    let abs_off = text_offsets[idx] + reloc.offset;
                    if abs_off % 8 != 0 {
                        panic!("Text relocation not aligned: {}", abs_off);
                    }
                    let inst_index = (abs_off / 8) as usize;
                    if inst_index >= text.len() {
                        panic!("Text relocation out of range: {}", abs_off);
                    }
                    patch_text_imm(&mut text, inst_index, value);
                }
                Section::Data => {
                    let abs_off = data_offsets[idx] + reloc.offset;
                    let off = abs_off as usize;
                    if off >= data.len() {
                        panic!("Data relocation out of range: {}", abs_off);
                    }
                    patch_data_bytes(&mut data, off, reloc.kind, value);
                }
                Section::Bss => {
                    panic!("Relocation in .bss is not supported");
                }
            }
        }
    }

    let main_addr = global_symbols
        .get("main")
        .copied()
        .or_else(|| {
            local_symbols
                .iter()
                .find_map(|m| m.get("main").copied())
        })
        .unwrap_or_else(|| panic!("No 'main' symbol found for entry"));
    patch_text_imm(&mut text, 1, main_addr);

    AssembledProgram {
        text,
        data,
        bss_size: bss_cursor,
        text_base,
        data_base,
        bss_base,
    }
}

pub fn link_objects(inputs: &[String], arch: &ArchConfig) -> AssembledProgram {
    if inputs.is_empty() {
        panic!("No input objects provided");
    }
    let objects: Vec<ObjectFile> = inputs.iter().map(|p| read_object(p)).collect();
    link_objects_from_memory(&objects, arch)
}

#[cfg(test)]
mod tests {
    use super::link_objects_from_memory;
    use crate::assemble::assemble::assemble_string_to_object;
    use crate::assemble::config::ArchConfig;

    fn test_arch() -> ArchConfig {
        let mut arch = ArchConfig::default();
        arch.macros.insert("PROGRAM_BASE".to_string(), 0x1000);
        arch.macros.insert("MEM_SIZE".to_string(), 0x4000);
        arch
    }

    #[test]
    fn link_two_objects_with_relocs() {
        let arch = test_arch();

        let asm1 = r#"
.text
.globl main
main:
    movi r1, foo
    call bar
    ret
.data
.globl foo
foo:
    .long 123
"#;

        let asm2 = r#"
.text
.globl bar
bar:
    ret
"#;

        let obj1 = assemble_string_to_object(asm1, &arch);
        let obj2 = assemble_string_to_object(asm2, &arch);

        let prog = link_objects_from_memory(&[obj1, obj2], &arch);

        let text_base = 0x1000u32;
        let entry_size = 24u32;
        let obj1_text_off = entry_size;
        let obj2_text_off = entry_size + 24;
        let data_base = text_base + entry_size + 24 + 8;

        let foo_addr = data_base;
        let bar_addr = text_base + obj2_text_off;
        let main_addr = text_base + obj1_text_off;

        let imm = |inst: u64| inst as u32;

        assert_eq!(imm(prog.text[1]), main_addr);
        assert_eq!(imm(prog.text[3]), foo_addr);
        assert_eq!(imm(prog.text[4]), bar_addr);
    }
}
