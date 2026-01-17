use std::fs::File;
use std::io;
use std::io::BufRead;
use std::path::Path;
use crate::config::ArchConfig;
use crate::inst::inst;
use crate::opcode::Opcode;
use crate::parse::parse_operands;

pub fn assemble_line(line: &str, arch: &ArchConfig) -> Option<u64> {
    let line = line.split(';').next()?.trim();
    if line.is_empty() {
        return None;
    }

    let mut iter = line.splitn(2, char::is_whitespace);
    let opcode_str = iter.next()?.trim();
    let operands_str = iter.next().unwrap_or("").trim();

    let opcode =
        Opcode::parse(opcode_str).unwrap_or_else(|| panic!("Unknown opcode {}", opcode_str));

    let args: Vec<&str> = operands_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let (rd, rs1, rs2, imm) = parse_operands(opcode.format(), &args, arch);

    Some(inst(opcode as u8, rd, rs1, rs2, imm))
}

pub fn assemble_file<P: AsRef<Path>>(path: P, arch: &ArchConfig) -> Vec<u64> {
    let file = File::open(path).expect("Could not open file");
    let reader = io::BufReader::new(file);
    let mut program: Vec<u64> = Vec::new();
    for line in reader.lines() {
        if let Ok(l) = line {
            if let Some(inst64) = assemble_line(&l, arch) {
                program.push(inst64);
            }
        }
    }
    program
}