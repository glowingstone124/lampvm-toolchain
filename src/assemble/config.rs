use std::collections::HashMap;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ArchConfig {
    pub reg_count: u8,
    pub reg_prefix: String,
    #[serde(default)]
    pub macros: HashMap<String, u32>,
}
impl ArchConfig {
    pub fn new(reg_count: u8, reg_prefix: &str) -> Self {
        if reg_count == 0 {
            panic!("Register count must be > 0");
        }
        Self { reg_count, reg_prefix: reg_prefix.to_string(), macros: HashMap::new() }
    }
    #[inline]
    pub fn check_reg(&self, reg: u8) {
        if reg >= self.reg_count {
            panic!(
                "Register r{} out of range (max r{})",
                reg,
                self.reg_count - 1
            );
        }
    }
    pub(crate) fn default() -> Self {
        ArchConfig {
            reg_count: 8,
            reg_prefix: String::from("r"),
            macros: HashMap::new(),
        }
    }
}
impl Default for ArchConfig {
    fn default() -> Self {
        ArchConfig {
            reg_count: 8,
            reg_prefix: "r".to_string(),
            macros: Default::default(),
        }
    }
}
