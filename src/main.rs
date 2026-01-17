mod inst;
mod opcode;
mod config;
mod assemble;
mod parse;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use crate::assemble::assemble_file;
use crate::config::ArchConfig;
fn main() {
    let path = Path::new("config.yml");

    if !path.exists() {
        let default_config = ArchConfig::default();
        let yaml_string = serde_yaml::to_string(&default_config)
            .expect("Failed to serialize default ArchConfig");
        fs::write(path, yaml_string)
            .expect("Failed to write default config.yml");
        println!("Generated default config.yml");
    }

    let yaml_string = fs::read_to_string(path)
        .expect("Failed to read config.yml");

    let arch: ArchConfig = serde_yaml::from_str(&yaml_string)
        .expect("Invalid YAML format");

    println!("Using architecture: {:?}", arch);

    let program = assemble_file("program.asm", &arch);

    for inst in &program {
        println!("0x{:016X}", inst);
    }
    let mut bin_file = File::create("program.bin").expect("Failed to create program.bin");
    for inst in &program {
        bin_file.write_all(&inst.to_le_bytes())
            .expect("Failed to write instruction");
    }

    println!("Binary written to program.bin");
    println!("Done.");
}
