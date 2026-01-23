
mod assemble{
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
use std::{fs, path::Path, fs::File, io::Write};
use clap::{Parser, Subcommand};
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

            let config_path = Path::new("config.yml");
            if !config_path.exists() {
                println!("config.yml not found. Generating default config.");
                let default_config = assemble::config::ArchConfig::default();
                let yaml_string = serde_yaml::to_string(&default_config)
                    .expect("Failed to serialize default ArchConfig");
                fs::write(config_path, yaml_string)
                    .expect("Failed to write default config.yml");
                println!("Generated default config.yml");
            }

            let yaml_string = fs::read_to_string(config_path)
                .expect("Failed to read config.yml");
            let arch: assemble::config::ArchConfig = serde_yaml::from_str(&yaml_string)
                .expect("Invalid YAML format");

            println!("Using architecture: {:?}", arch);

            println!("Assembling {}...", input);
            let program = assemble::assemble::assemble_file(&input, &arch);
            println!("Assembly finished. {} instructions generated.", program.len());

            for (i, inst) in program.iter().enumerate() {
                println!("{}: 0x{:016X}", i, inst);
            }

            let mut bin_file = File::create(&output)
                .expect("Failed to create output file");
            for inst in &program {
                bin_file.write_all(&inst.to_le_bytes())
                    .expect("Failed to write instruction");
            }

            println!("Binary written to {}", output);
        }

        Commands::Cc { input } => {
            println!("C compile mode: {}", input);
            let content = fs::read_to_string(input).expect("Failed to read file");
            compile(&content);
        }
    }
}
