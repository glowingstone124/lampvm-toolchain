mod assemble {
    pub mod assemble;
    pub mod inst;
    pub mod config;
    pub mod opcode;
    pub mod parse;
}

mod compiler {
    pub mod compiler;
    pub mod tokenizer;
}

use clap::{Parser, Subcommand};
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::assemble::config::ArchConfig;
use crate::compiler::compiler::compile;

#[derive(Parser, Debug)]
#[command(author = "Glowingstone", version = "0.01", about = "lampVM's toolchain", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Asm {
        #[arg(long, default_value = "program.asm")]
        input: String,

        #[arg(long, default_value = "program.bin")]
        output: String,
    },
    Cc {
        input: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Asm { input, output } => {
            println!("Assembly mode");
            println!("Input file: {}", input);
            println!("Output file: {}", output);

            let arch = load_or_generate_arch_config("config.yml");
            println!("Using architecture: {:?}", arch);

            println!("Assembling {}...", input);
            let program = assemble::assemble::assemble_file(&input, &arch);
            println!("Assembly finished. {} instructions generated.", program.len());

            for (i, inst) in program.iter().enumerate() {
                println!("{}: 0x{:016X}", i, inst);
            }

            write_program_bin(&program, &output);
        }

        Commands::Cc { input } => {
            println!("C compile mode: {}", input);

            let arch = load_or_generate_arch_config("config.yml");
            println!("Using architecture: {:?}", arch);

            let content = fs::read_to_string(&input).expect("Failed to read file");
            let asm_text = compile(&content);
            println!("Compile finished. asm text length = {} chars", asm_text.len());
            println!("Result:\n{}", asm_text);

            println!("Assembling code...");
            let program = assemble::assemble::assemble_string(&asm_text, &arch);
            println!("Assembly finished. {} instructions generated.", program.len());

            let output_name = replace_suffix_or_append(&input, ".c", ".bin");
            write_program_bin(&program, &output_name);

            println!("Output: {}", output_name);
        }
    }
}

fn load_or_generate_arch_config(config_path: &str) -> ArchConfig {
    let config_path = Path::new(config_path);

    if !config_path.exists() {
        println!("config.yml not found. Generating default config.");
        let default_config = ArchConfig::default();
        let yaml_string =
            serde_yaml::to_string(&default_config).expect("Failed to serialize default ArchConfig");
        fs::write(config_path, yaml_string).expect("Failed to write config.yml");
        println!("Generated default config.yml");
    }

    let yaml_string = fs::read_to_string(config_path).expect("Failed to read config.yml");
    serde_yaml::from_str(&yaml_string).expect("Invalid YAML format")
}

fn replace_suffix_or_append(input: &str, from_suffix: &str, to_suffix: &str) -> String {
    if let Some(stem) = input.strip_suffix(from_suffix) {
        format!("{}{}", stem, to_suffix)
    } else {
        format!("{}{}", input, to_suffix)
    }
}

fn write_program_bin(binary: &[u64], output: &str) {
    let mut bin_file = File::create(output).expect("Failed to create output file");
    for inst in binary {
        bin_file
            .write_all(&inst.to_le_bytes())
            .expect("Failed to write instruction");
    }
    bin_file.flush().expect("Failed to flush output file");
    println!("Binary written to {}", output);
}
