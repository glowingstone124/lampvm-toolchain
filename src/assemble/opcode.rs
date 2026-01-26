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
    OP_FSTORE32
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

            Opcode::OP_SHL
            | Opcode::OP_SHR
            | Opcode::OP_LOAD
            | Opcode::OP_LOAD32
            | Opcode::OP_STORE32
            | Opcode::OP_STORE
            | Opcode::OP_MEMSET
            | Opcode::OP_MEMCPY
            | Opcode::OP_FLOAD32
            | Opcode::OP_FSTORE32 => InstFormat::RdRsImm,

            Opcode::OP_JMP
            | Opcode::OP_JZ
            | Opcode::OP_JNZ
            | Opcode::OP_JG
            | Opcode::OP_JGE
            | Opcode::OP_JLE
            | Opcode::OP_JC
            | Opcode::OP_JNC
            | Opcode::OP_CALL => InstFormat::I,

            Opcode::OP_PUSH | Opcode::OP_POP | Opcode::OP_INT => InstFormat::Rd,

            Opcode::OP_STOREX32 | Opcode::OP_LOADX32 => InstFormat::RdRsRsImm,

            Opcode::OP_HALT | Opcode::OP_RET | Opcode::OP_IRET => InstFormat::None,

            Opcode::OP_CMPI | Opcode::OP_MOVI => InstFormat::RdImm,
            _ => panic!("Unknown opcode"),
        }
    }
}
