use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::path::Path;
use crate::assemble::config::ArchConfig;
use crate::assemble::inst::inst;
use crate::assemble::opcode::Opcode;
use crate::assemble::parse::parse_operands;

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

pub fn assemble_file<P: AsRef<Path>>(path: P, arch: &ArchConfig) -> Vec<u64> {
    let file = File::open(path).expect("Could not open file");
    let reader = io::BufReader::new(file);

    let lines: Vec<String> = reader.lines().filter_map(Result::ok).collect();

    let mut labels = HashMap::new();
    let mut program: Vec<u64> = Vec::new();

    const INST_SIZE: u32 = 8;

    let mut pc = arch.macros.get("PROGRAM_BASE").expect("No PROGRAM_BASE found").clone();
    for line in &lines {
        let clean = clean_line(line);
        if clean.is_empty() { continue; }

        let (label_opt, inst_part) = split_label(clean);

        if let Some(label) = label_opt {
            if labels.insert(label.to_string(), pc).is_some() {
                panic!("Duplicate label definition: {}", label);
            }
        }
        if !inst_part.is_empty() {
            pc += INST_SIZE;
        }
    }

    for line in &lines {
        let clean = clean_line(line);
        if clean.is_empty() { continue; }

        let (_, inst_part) = split_label(clean);

        if inst_part.is_empty() { continue; }

        if let Some(inst64) = assemble_line(inst_part, arch, &labels) {
            program.push(inst64);
        }
    }

    program
}