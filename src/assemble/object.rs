use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Section {
    Text,
    Data,
    Bss,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub section: Option<Section>,
    pub offset: u32,
    pub global: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelocKind {
    Abs32,
    Abs16,
    Abs8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Relocation {
    pub section: Section,
    pub offset: u32,
    pub kind: RelocKind,
    pub symbol: String,
    pub addend: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectFile {
    pub text: Vec<u64>,
    pub data: Vec<u8>,
    pub bss_size: u32,
    pub symbols: Vec<Symbol>,
    pub relocations: Vec<Relocation>,
}
