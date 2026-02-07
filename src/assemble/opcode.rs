use crate::assemble::inst::InstFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum Opcode {
    OP_ADD = 1,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_HALT,
    OP_JMP,
    OP_JZ,
    OP_PUSH,
    OP_POP,
    OP_CALL,
    OP_RET,
    OP_LOAD,
    OP_LOAD32,
    OP_LOADX32,
    OP_STORE,
    OP_STORE32,
    OP_STOREX32,
    OP_CMP,
    OP_CMPI,
    OP_MOV,
    OP_MOVI,
    OP_MEMSET,
    OP_MEMCPY,
    OP_IN,
    OP_OUT,
    OP_INT,
    OP_IRET,
    OP_MOD,
    OP_AND,
    OP_OR,
    OP_XOR,
    OP_NOT,
    OP_SHL,
    OP_SHR,
    OP_SAR,
    OP_JNZ,
    OP_JG,
    OP_JGE,
    OP_JL,
    OP_JLE,
    OP_JC,
    OP_JNC,
    OP_FADD,
    OP_FSUB,
    OP_FMUL,
    OP_FDIV,
    OP_FNEG,
    OP_FABS,
    OP_FSQRT,
    OP_FCMP,
    OP_ITOF,
    OP_FTOI,
    OP_FLOAD32,
    OP_FSTORE32,
    OP_INC,
    OP_ADDI,
    OP_SUBI,
    OP_ANDI,
    OP_ORI,
    OP_XORI,
    OP_SHLI,
    OP_SHRI,
    OP_CAS,
    OP_XADD,
    OP_XCHG,
    OP_LDAR,
    OP_STLR,
    OP_FENCE,
    OP_PAUSE,
    OP_STARTAP,
    OP_IPI,
    OP_CPUID,
}
impl Opcode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "ADD" => Some(Opcode::OP_ADD),
            "SUB" => Some(Opcode::OP_SUB),
            "MUL" => Some(Opcode::OP_MUL),
            "DIV" => Some(Opcode::OP_DIV),
            "HALT" => Some(Opcode::OP_HALT),
            "JMP" => Some(Opcode::OP_JMP),
            "JZ" => Some(Opcode::OP_JZ),
            "PUSH" => Some(Opcode::OP_PUSH),
            "POP" => Some(Opcode::OP_POP),
            "CALL" => Some(Opcode::OP_CALL),
            "RET" => Some(Opcode::OP_RET),
            "LOAD" => Some(Opcode::OP_LOAD),
            "LOAD32" => Some(Opcode::OP_LOAD32),
            "LOADX32" => Some(Opcode::OP_LOADX32),
            "STORE" => Some(Opcode::OP_STORE),
            "STORE32" => Some(Opcode::OP_STORE32),
            "STOREX32" => Some(Opcode::OP_STOREX32),
            "CMP" => Some(Opcode::OP_CMP),
            "CMPI" => Some(Opcode::OP_CMPI),
            "MOV" => Some(Opcode::OP_MOV),
            "MOVI" => Some(Opcode::OP_MOVI),
            "MEMSET" => Some(Opcode::OP_MEMSET),
            "MEMCPY" => Some(Opcode::OP_MEMCPY),
            "IN" => Some(Opcode::OP_IN),
            "OUT" => Some(Opcode::OP_OUT),
            "INT" => Some(Opcode::OP_INT),
            "IRET" => Some(Opcode::OP_IRET),
            "MOD" => Some(Opcode::OP_MOD),
            "AND" => Some(Opcode::OP_AND),
            "OR" => Some(Opcode::OP_OR),
            "XOR" => Some(Opcode::OP_XOR),
            "NOT" => Some(Opcode::OP_NOT),
            "SHL" => Some(Opcode::OP_SHL),
            "SHR" => Some(Opcode::OP_SHR),
            "SAR" => Some(Opcode::OP_SAR),
            "JNZ" => Some(Opcode::OP_JNZ),
            "JG" => Some(Opcode::OP_JG),
            "JGE" => Some(Opcode::OP_JGE),
            "JL" => Some(Opcode::OP_JL),
            "JLE" => Some(Opcode::OP_JLE),
            "JC" => Some(Opcode::OP_JC),
            "JNC" => Some(Opcode::OP_JNC),
            "FADD" => Some(Opcode::OP_FADD),
            "FSUB" => Some(Opcode::OP_FSUB),
            "FMUL" => Some(Opcode::OP_FMUL),
            "FDIV" => Some(Opcode::OP_FDIV),
            "FNEG" => Some(Opcode::OP_FNEG),
            "FABS" => Some(Opcode::OP_FABS),
            "FSQRT" => Some(Opcode::OP_FSQRT),
            "FCMP" => Some(Opcode::OP_FCMP),
            "ITOF" => Some(Opcode::OP_ITOF),
            "FTOI" => Some(Opcode::OP_FTOI),
            "FLOAD32" => Some(Opcode::OP_FLOAD32),
            "FSTORE32" => Some(Opcode::OP_FSTORE32),
            "INC" => Some(Opcode::OP_INC),
            "ADDI" => Some(Opcode::OP_ADDI),
            "SUBI" => Some(Opcode::OP_SUBI),
            "ANDI" => Some(Opcode::OP_ANDI),
            "ORI" => Some(Opcode::OP_ORI),
            "XORI" => Some(Opcode::OP_XORI),
            "SHLI" => Some(Opcode::OP_SHLI),
            "SHRI" => Some(Opcode::OP_SHRI),
            "CAS" => Some(Opcode::OP_CAS),
            "XADD" => Some(Opcode::OP_XADD),
            "XCHG" => Some(Opcode::OP_XCHG),
            "LDAR" => Some(Opcode::OP_LDAR),
            "STLR" => Some(Opcode::OP_STLR),
            "FENCE" => Some(Opcode::OP_FENCE),
            "PAUSE" => Some(Opcode::OP_PAUSE),
            "STARTAP" => Some(Opcode::OP_STARTAP),
            "IPI" => Some(Opcode::OP_IPI),
            "CPUID" => Some(Opcode::OP_CPUID),
            _ => None,
        }
    }

    pub fn format(self) -> InstFormat {
        match self {
            Opcode::OP_ADD
            | Opcode::OP_SUB
            | Opcode::OP_MUL
            | Opcode::OP_DIV
            | Opcode::OP_MOD
            | Opcode::OP_AND
            | Opcode::OP_OR
            | Opcode::OP_XOR
            | Opcode::OP_SHL
            | Opcode::OP_SHR
            | Opcode::OP_SAR
            | Opcode::OP_FADD
            | Opcode::OP_FSUB
            | Opcode::OP_FMUL
            | Opcode::OP_FDIV => InstFormat::RdRsRs,

            Opcode::OP_MOV
            | Opcode::OP_NOT
            | Opcode::OP_CMP
            | Opcode::OP_IN
            | Opcode::OP_OUT
            | Opcode::OP_FNEG
            | Opcode::OP_FABS
            | Opcode::OP_FSQRT
            | Opcode::OP_FCMP
            | Opcode::OP_ITOF
            | Opcode::OP_FTOI => {
                InstFormat::RdRs
            }

            Opcode::OP_LOAD
            | Opcode::OP_LOAD32
            | Opcode::OP_STORE32
            | Opcode::OP_STORE
            | Opcode::OP_MEMSET
            | Opcode::OP_MEMCPY
            | Opcode::OP_FLOAD32
            | Opcode::OP_FSTORE32
            | Opcode::OP_ADDI
            | Opcode::OP_SUBI
            | Opcode::OP_ANDI
            | Opcode::OP_ORI
            | Opcode::OP_XORI
            | Opcode::OP_SHLI
            | Opcode::OP_SHRI
            | Opcode::OP_LDAR
            | Opcode::OP_STLR
            | Opcode::OP_STARTAP => InstFormat::RdRsImm,

            Opcode::OP_JMP
            | Opcode::OP_JZ
            | Opcode::OP_JNZ
            | Opcode::OP_JG
            | Opcode::OP_JGE
            | Opcode::OP_JL
            | Opcode::OP_JLE
            | Opcode::OP_JC
            | Opcode::OP_JNC
            | Opcode::OP_CALL => InstFormat::I,

            Opcode::OP_INC
            | Opcode::OP_PUSH
            | Opcode::OP_POP
            | Opcode::OP_INT
            | Opcode::OP_CPUID => InstFormat::Rd,

            Opcode::OP_STOREX32
            | Opcode::OP_LOADX32
            | Opcode::OP_CAS
            | Opcode::OP_XADD
            | Opcode::OP_XCHG => InstFormat::RdRsRsImm,

            Opcode::OP_HALT
            | Opcode::OP_RET
            | Opcode::OP_IRET
            | Opcode::OP_FENCE
            | Opcode::OP_PAUSE => InstFormat::None,

            Opcode::OP_IPI => InstFormat::RdRs,

            Opcode::OP_CMPI | Opcode::OP_MOVI => InstFormat::RdImm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Opcode;
    use crate::assemble::inst::InstFormat;

    #[test]
    fn parses_new_smp_atomic_opcodes() {
        assert_eq!(Opcode::parse("CAS"), Some(Opcode::OP_CAS));
        assert_eq!(Opcode::parse("XADD"), Some(Opcode::OP_XADD));
        assert_eq!(Opcode::parse("XCHG"), Some(Opcode::OP_XCHG));
        assert_eq!(Opcode::parse("LDAR"), Some(Opcode::OP_LDAR));
        assert_eq!(Opcode::parse("STLR"), Some(Opcode::OP_STLR));
        assert_eq!(Opcode::parse("FENCE"), Some(Opcode::OP_FENCE));
        assert_eq!(Opcode::parse("PAUSE"), Some(Opcode::OP_PAUSE));
        assert_eq!(Opcode::parse("STARTAP"), Some(Opcode::OP_STARTAP));
        assert_eq!(Opcode::parse("IPI"), Some(Opcode::OP_IPI));
        assert_eq!(Opcode::parse("CPUID"), Some(Opcode::OP_CPUID));
    }

    #[test]
    fn new_opcode_values_match_vm_isa() {
        assert_eq!(Opcode::OP_CAS as u8, 0x3F);
        assert_eq!(Opcode::OP_XADD as u8, 0x40);
        assert_eq!(Opcode::OP_XCHG as u8, 0x41);
        assert_eq!(Opcode::OP_LDAR as u8, 0x42);
        assert_eq!(Opcode::OP_STLR as u8, 0x43);
        assert_eq!(Opcode::OP_FENCE as u8, 0x44);
        assert_eq!(Opcode::OP_PAUSE as u8, 0x45);
        assert_eq!(Opcode::OP_STARTAP as u8, 0x46);
        assert_eq!(Opcode::OP_IPI as u8, 0x47);
        assert_eq!(Opcode::OP_CPUID as u8, 0x48);
    }

    #[test]
    fn formats_for_new_opcodes_are_correct() {
        assert_eq!(Opcode::OP_CAS.format(), InstFormat::RdRsRsImm);
        assert_eq!(Opcode::OP_XADD.format(), InstFormat::RdRsRsImm);
        assert_eq!(Opcode::OP_XCHG.format(), InstFormat::RdRsRsImm);
        assert_eq!(Opcode::OP_LDAR.format(), InstFormat::RdRsImm);
        assert_eq!(Opcode::OP_STLR.format(), InstFormat::RdRsImm);
        assert_eq!(Opcode::OP_FENCE.format(), InstFormat::None);
        assert_eq!(Opcode::OP_PAUSE.format(), InstFormat::None);
        assert_eq!(Opcode::OP_STARTAP.format(), InstFormat::RdRsImm);
        assert_eq!(Opcode::OP_IPI.format(), InstFormat::RdRs);
        assert_eq!(Opcode::OP_CPUID.format(), InstFormat::Rd);
    }
}
