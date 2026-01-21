pub fn inst(op: u8, rd: u8, rs1: u8, rs2: u8, imm: u32) -> u64 {
    ((op as u64) << 56)
        | ((rd as u64) << 48)
        | ((rs1 as u64) << 40)
        | ((rs2 as u64) << 32)
        | (imm as u64)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstFormat {
    None,
    RdRsRs,         // rd, rs1, rs2
    RdRs,           // rd, rs1
    I,              // imm
    Rd,             // rd
    RdImm,          // rd, imm
    RdRsImm,        // rd, rs1, imm
    RdRsRsImm,      // rd, rs1, rs2, imm
}
