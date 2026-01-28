use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::path::Path;
use crate::assemble::config::ArchConfig;
use crate::assemble::inst::inst;
use crate::assemble::opcode::Opcode;
use crate::assemble::object::{ObjectFile, RelocKind, Relocation, Section as ObjSection, Symbol};
use crate::assemble::parse::{parse_imm, parse_operands, parse_operands_reloc, parse_imm_reloc, RelocRef};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Section {
    Text,
    Data,
    Bss,
}

#[derive(Clone, Copy, Debug)]
struct LabelRef {
    section: Section,
    offset: u32,
}

pub struct AssembledProgram {
    pub text: Vec<u64>,
    pub data: Vec<u8>,
    pub bss_size: u32,
    pub text_base: u32,
    pub data_base: u32,
    pub bss_base: u32,
}

fn clean_line(line: &str) -> &str {
    line.split(';').next().unwrap_or("").trim()
}
fn split_label(line: &str) -> (Option<&str>, &str) {
    if let Some((label, rest)) = line.split_once(':') {
        (Some(label.trim()), rest.trim())
    } else {
        (None, line)
    }
}
pub fn assemble_line(
    line: &str,
    arch: &ArchConfig,
    labels: &HashMap<String, u32>
) -> Option<u64> {
    if line.is_empty() {
        return None;
    }

    let mut iter = line.splitn(2, char::is_whitespace);
    let opcode_str = iter.next()?.trim();
    let operands_str = iter.next().unwrap_or("").trim();

    let opcode = Opcode::parse(opcode_str)
        .unwrap_or_else(|| panic!("Unknown opcode {}", opcode_str));

    let args: Vec<&str> = operands_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let (rd, rs1, rs2, imm) = parse_operands(opcode.format(), &args, arch, labels);

    Some(inst(opcode as u8, rd, rs1, rs2, imm))
}

fn parse_directive(line: &str) -> (&str, &str) {
    let mut iter = line.splitn(2, char::is_whitespace);
    let name = iter.next().unwrap();
    let rest = iter.next().unwrap_or("").trim();
    (name, rest)
}

fn parse_symbol_list(args: &str) -> Vec<String> {
    args.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn parse_string_literal(s: &str) -> Vec<u8> {
    let s = s.trim();
    if !s.starts_with('"') {
        panic!("Expected string literal");
    }
    let mut chars = s[1..].chars();
    let mut out = Vec::new();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if chars.as_str().trim().is_empty() {
                return out;
            }
            panic!("Unexpected trailing characters after string literal");
        }
        if ch != '\\' {
            out.push(ch as u8);
            continue;
        }
        let esc = chars.next().unwrap_or_else(|| panic!("Unterminated escape sequence"));
        let byte = match esc {
            'n' => b'\n',
            'r' => b'\r',
            't' => b'\t',
            '0' => b'\0',
            '\\' => b'\\',
            '"' => b'"',
            _ => panic!("Unsupported escape sequence: \\{}", esc),
        };
        out.push(byte);
    }
    panic!("Unterminated string literal");
}

fn parse_list_args(args: &str) -> Vec<String> {
    args.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn data_directive_size(
    directive: &str,
    args: &str,
    arch: &ArchConfig,
    labels: &HashMap<String, u32>,
) -> u32 {
    match directive {
        ".byte" => parse_list_args(args).len() as u32,
        ".word" => (parse_list_args(args).len() as u32) * 2,
        ".long" => (parse_list_args(args).len() as u32) * 4,
        ".space" | ".zero" => parse_imm(args, arch, labels),
        ".ascii" => parse_string_literal(args).len() as u32,
        ".asciz" => (parse_string_literal(args).len() as u32) + 1,
        _ => 0,
    }
}

fn emit_data_directive(
    data: &mut Vec<u8>,
    directive: &str,
    args: &str,
    arch: &ArchConfig,
    labels: &HashMap<String, u32>,
) {
    match directive {
        ".byte" => {
            for arg in parse_list_args(args) {
                let val = parse_imm(&arg, arch, labels) as u8;
                data.push(val);
            }
        }
        ".word" => {
            for arg in parse_list_args(args) {
                let val = parse_imm(&arg, arch, labels) as u16;
                data.extend_from_slice(&val.to_le_bytes());
            }
        }
        ".long" => {
            for arg in parse_list_args(args) {
                let val = parse_imm(&arg, arch, labels);
                data.extend_from_slice(&val.to_le_bytes());
            }
        }
        ".space" | ".zero" => {
            let count = parse_imm(args, arch, labels) as usize;
            data.extend(std::iter::repeat(0u8).take(count));
        }
        ".ascii" => {
            data.extend_from_slice(&parse_string_literal(args));
        }
        ".asciz" => {
            let mut bytes = parse_string_literal(args);
            bytes.push(0);
            data.extend_from_slice(&bytes);
        }
        _ => {}
    }
}

fn emit_data_directive_reloc(
    data: &mut Vec<u8>,
    relocations: &mut Vec<Relocation>,
    directive: &str,
    args: &str,
    arch: &ArchConfig,
    section_offset: u32,
) -> u32 {
    let mut written = 0u32;
    match directive {
        ".byte" => {
            for arg in parse_list_args(args) {
                let (imm, reloc) = parse_imm_reloc(&arg, arch);
                if let Some(RelocRef { symbol, addend }) = reloc {
                    relocations.push(Relocation {
                        section: ObjSection::Data,
                        offset: section_offset + written,
                        kind: RelocKind::Abs8,
                        symbol,
                        addend,
                    });
                    data.push(0);
                } else {
                    data.push(imm as u8);
                }
                written += 1;
            }
        }
        ".word" => {
            for arg in parse_list_args(args) {
                let (imm, reloc) = parse_imm_reloc(&arg, arch);
                if let Some(RelocRef { symbol, addend }) = reloc {
                    relocations.push(Relocation {
                        section: ObjSection::Data,
                        offset: section_offset + written,
                        kind: RelocKind::Abs16,
                        symbol,
                        addend,
                    });
                    data.extend_from_slice(&0u16.to_le_bytes());
                } else {
                    data.extend_from_slice(&(imm as u16).to_le_bytes());
                }
                written += 2;
            }
        }
        ".long" => {
            for arg in parse_list_args(args) {
                let (imm, reloc) = parse_imm_reloc(&arg, arch);
                if let Some(RelocRef { symbol, addend }) = reloc {
                    relocations.push(Relocation {
                        section: ObjSection::Data,
                        offset: section_offset + written,
                        kind: RelocKind::Abs32,
                        symbol,
                        addend,
                    });
                    data.extend_from_slice(&0u32.to_le_bytes());
                } else {
                    data.extend_from_slice(&(imm as u32).to_le_bytes());
                }
                written += 4;
            }
        }
        ".space" | ".zero" => {
            let count = parse_imm(args, arch, &HashMap::new()) as usize;
            data.extend(std::iter::repeat(0u8).take(count));
            written += count as u32;
        }
        ".ascii" => {
            let bytes = parse_string_literal(args);
            written += bytes.len() as u32;
            data.extend_from_slice(&bytes);
        }
        ".asciz" => {
            let mut bytes = parse_string_literal(args);
            bytes.push(0);
            written += bytes.len() as u32;
            data.extend_from_slice(&bytes);
        }
        _ => {}
    }
    written
}

pub fn assemble_file_with_sections<P: AsRef<Path>>(path: P, arch: &ArchConfig) -> AssembledProgram {
    let file = File::open(path).expect("Could not open file");
    let reader = io::BufReader::new(file);
    let lines: Vec<String> = reader.lines().filter_map(Result::ok).collect();
    assemble_lines_with_sections(&lines, arch)
}

pub fn assemble_string_with_sections(input: &str, arch: &ArchConfig) -> AssembledProgram {
    let lines: Vec<String> = input.lines().map(|s| s.to_string()).collect();
    assemble_lines_with_sections(&lines, arch)
}

pub fn assemble_file_to_object<P: AsRef<Path>>(path: P, arch: &ArchConfig) -> ObjectFile {
    let file = File::open(path).expect("Could not open file");
    let reader = io::BufReader::new(file);
    let lines: Vec<String> = reader.lines().filter_map(Result::ok).collect();
    assemble_lines_to_object(&lines, arch)
}

pub fn assemble_string_to_object(input: &str, arch: &ArchConfig) -> ObjectFile {
    let lines: Vec<String> = input.lines().map(|s| s.to_string()).collect();
    assemble_lines_to_object(&lines, arch)
}

fn assemble_lines_to_object(lines: &[String], arch: &ArchConfig) -> ObjectFile {
    let mut label_refs: HashMap<String, LabelRef> = HashMap::new();
    let mut section = Section::Text;
    let mut text_size: u32 = 0;
    let mut data_size: u32 = 0;
    let mut bss_size: u32 = 0;
    let mut globals: HashMap<String, bool> = HashMap::new();
    let mut externs: HashMap<String, bool> = HashMap::new();

    const INST_SIZE: u32 = 8;

    for line in lines {
        let clean = clean_line(line);
        if clean.is_empty() {
            continue;
        }
        let (label_opt, inst_part) = split_label(clean);
        if let Some(label) = label_opt {
            if label_refs.insert(label.to_string(), LabelRef { section, offset: match section {
                Section::Text => text_size,
                Section::Data => data_size,
                Section::Bss => bss_size,
            }}).is_some() {
                panic!("Duplicate label definition: {}", label);
            }
        }
        if inst_part.is_empty() {
            continue;
        }
        if inst_part.starts_with('.') {
            let (directive, args) = parse_directive(inst_part);
            match directive {
                ".text" => section = Section::Text,
                ".data" => section = Section::Data,
                ".bss" => section = Section::Bss,
                ".globl" => {
                    for name in parse_symbol_list(args) {
                        globals.insert(name, true);
                    }
                }
                ".extern" => {
                    for name in parse_symbol_list(args) {
                        externs.insert(name, true);
                    }
                }
                ".byte" | ".word" | ".long" | ".space" | ".zero" | ".ascii" | ".asciz" => {
                    let size = match directive {
                        ".byte" => parse_list_args(args).len() as u32,
                        ".word" => (parse_list_args(args).len() as u32) * 2,
                        ".long" => (parse_list_args(args).len() as u32) * 4,
                        ".space" | ".zero" => parse_imm(args, arch, &HashMap::new()),
                        ".ascii" => parse_string_literal(args).len() as u32,
                        ".asciz" => (parse_string_literal(args).len() as u32) + 1,
                        _ => 0,
                    };
                    match section {
                        Section::Text => panic!("Data directive {} not allowed in .text", directive),
                        Section::Data => data_size += size,
                        Section::Bss => {
                            if matches!(directive, ".ascii" | ".asciz" | ".byte" | ".word" | ".long") {
                                panic!("Initialized data not allowed in .bss");
                            }
                            bss_size += size;
                        }
                    }
                }
                _ => panic!("Unknown directive {}", directive),
            }
            continue;
        }
        if section != Section::Text {
            panic!("Instructions are only allowed in .text");
        }
        text_size += INST_SIZE;
    }

    let mut symbols: Vec<Symbol> = Vec::new();
    for (name, label_ref) in &label_refs {
        let section = match label_ref.section {
            Section::Text => ObjSection::Text,
            Section::Data => ObjSection::Data,
            Section::Bss => ObjSection::Bss,
        };
        let global = globals.get(name).copied().unwrap_or(false);
        symbols.push(Symbol {
            name: name.clone(),
            section: Some(section),
            offset: label_ref.offset,
            global,
        });
    }
    for name in externs.keys() {
        symbols.push(Symbol {
            name: name.clone(),
            section: None,
            offset: 0,
            global: true,
        });
    }

    let mut program: Vec<u64> = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    let mut relocations: Vec<Relocation> = Vec::new();
    section = Section::Text;
    let mut text_offset: u32 = 0;
    let mut data_offset: u32 = 0;

    for line in lines {
        let clean = clean_line(line);
        if clean.is_empty() {
            continue;
        }
        let (_, inst_part) = split_label(clean);
        if inst_part.is_empty() {
            continue;
        }
        if inst_part.starts_with('.') {
            let (directive, args) = parse_directive(inst_part);
            match directive {
                ".text" => section = Section::Text,
                ".data" => section = Section::Data,
                ".bss" => section = Section::Bss,
                ".globl" | ".extern" => {}
                ".byte" | ".word" | ".long" | ".space" | ".zero" | ".ascii" | ".asciz" => {
                    if section == Section::Data {
                        let written = emit_data_directive_reloc(
                            &mut data,
                            &mut relocations,
                            directive,
                            args,
                            arch,
                            data_offset,
                        );
                        data_offset += written;
                    } else if section == Section::Bss {
                        // bss represented only by size
                    } else {
                        panic!("Data directive {} not allowed in .text", directive);
                    }
                }
                _ => panic!("Unknown directive {}", directive),
            }
            continue;
        }
        if section != Section::Text {
            panic!("Instructions are only allowed in .text");
        }
        let mut iter = inst_part.splitn(2, char::is_whitespace);
        let opcode_str = iter.next().unwrap().trim();
        let operands_str = iter.next().unwrap_or("").trim();

        let opcode = Opcode::parse(opcode_str)
            .unwrap_or_else(|| panic!("Unknown opcode {}", opcode_str));

        let args: Vec<&str> = operands_str
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let (rd, rs1, rs2, imm, reloc) = parse_operands_reloc(opcode.format(), &args, arch);
        if let Some(RelocRef { symbol, addend }) = reloc {
            relocations.push(Relocation {
                section: ObjSection::Text,
                offset: text_offset,
                kind: RelocKind::Abs32,
                symbol,
                addend,
            });
        }
        program.push(inst(opcode as u8, rd, rs1, rs2, imm));
        text_offset += INST_SIZE;
    }

    ObjectFile {
        text: program,
        data,
        bss_size,
        symbols,
        relocations,
    }
}

fn assemble_lines_with_sections(lines: &[String], arch: &ArchConfig) -> AssembledProgram {
    let mut label_refs: HashMap<String, LabelRef> = HashMap::new();
    let mut section = Section::Text;
    let mut text_size: u32 = 0;
    let mut data_size: u32 = 0;
    let mut bss_size: u32 = 0;

    const INST_SIZE: u32 = 8;

    let text_base = *arch.macros.get("PROGRAM_BASE").expect("No PROGRAM_BASE found");
    let dummy_labels: HashMap<String, u32> = HashMap::new();

    for line in lines {
        let clean = clean_line(line);
        if clean.is_empty() { continue; }

        let (label_opt, inst_part) = split_label(clean);
        if let Some(label) = label_opt {
            if label_refs.insert(label.to_string(), LabelRef { section, offset: match section {
                Section::Text => text_size,
                Section::Data => data_size,
                Section::Bss => bss_size,
            }}).is_some() {
                panic!("Duplicate label definition: {}", label);
            }
        }

        if inst_part.is_empty() { continue; }

        if inst_part.starts_with('.') {
            let (directive, args) = parse_directive(inst_part);
            match directive {
                ".text" => section = Section::Text,
                ".data" => section = Section::Data,
                ".bss" => section = Section::Bss,
                ".globl" | ".extern" => {}
                ".byte" | ".word" | ".long" | ".space" | ".zero" | ".ascii" | ".asciz" => {
                    let size = data_directive_size(directive, args, arch, &dummy_labels);
                    match section {
                        Section::Text => panic!("Data directive {} not allowed in .text", directive),
                        Section::Data => data_size += size,
                        Section::Bss => {
                            if matches!(directive, ".ascii" | ".asciz" | ".byte" | ".word" | ".long") {
                                panic!("Initialized data not allowed in .bss");
                            }
                            bss_size += size;
                        }
                    }
                }
                _ => panic!("Unknown directive {}", directive),
            }
            continue;
        }

        if section != Section::Text {
            panic!("Instructions are only allowed in .text");
        }
        text_size += INST_SIZE;
    }

    let data_base = text_base + text_size;
    let bss_base = data_base + data_size;

    let mut labels: HashMap<String, u32> = HashMap::new();
    for (name, label_ref) in label_refs {
        let base = match label_ref.section {
            Section::Text => text_base,
            Section::Data => data_base,
            Section::Bss => bss_base,
        };
        labels.insert(name, base + label_ref.offset);
    }

    let mut program: Vec<u64> = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    section = Section::Text;

    for line in lines {
        let clean = clean_line(line);
        if clean.is_empty() { continue; }
        let (_, inst_part) = split_label(clean);
        if inst_part.is_empty() { continue; }

        if inst_part.starts_with('.') {
            let (directive, args) = parse_directive(inst_part);
            match directive {
                ".text" => section = Section::Text,
                ".data" => section = Section::Data,
                ".bss" => section = Section::Bss,
                ".globl" | ".extern" => {}
                ".byte" | ".word" | ".long" | ".space" | ".zero" | ".ascii" | ".asciz" => {
                    if section == Section::Data {
                        emit_data_directive(&mut data, directive, args, arch, &labels);
                    } else if section == Section::Bss {
                        // bss is represented only by size, data not emitted
                    } else {
                        panic!("Data directive {} not allowed in .text", directive);
                    }
                }
                _ => panic!("Unknown directive {}", directive),
            }
            continue;
        }

        if section != Section::Text {
            panic!("Instructions are only allowed in .text");
        }
        if let Some(inst64) = assemble_line(inst_part, arch, &labels) {
            program.push(inst64);
        }
    }

    AssembledProgram {
        text: program,
        data,
        bss_size,
        text_base,
        data_base,
        bss_base,
    }
}

pub fn assemble_file<P: AsRef<Path>>(path: P, arch: &ArchConfig) -> Vec<u64> {
    assemble_file_with_sections(path, arch).text
}

pub fn assemble_string(input: &str, arch: &ArchConfig) -> Vec<u64> {
    assemble_string_with_sections(input, arch).text
}
