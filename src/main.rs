mod assemble {
    pub mod assemble;
    pub mod inst;
    pub mod config;
    pub mod opcode;
    pub mod parse;
    pub mod object;
}

mod compiler {
    pub mod compiler;
    pub mod tokenizer;
    pub mod parser;
}

mod linker;

use clap::{Parser, Subcommand};
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::assemble::config::ArchConfig;
use crate::assemble::assemble::AssembledProgram;
use crate::compiler::compiler::{compile, EmitMode};
use crate::linker::link_objects;

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

        #[arg(long)]
        emit_obj: bool,
    },
    #[command(about = "DEPRECATED: compile C source with the legacy frontend (use LLVM backend instead)")]
    Cc {
        input: String,

        #[arg(short = 'c', long)]
        emit_obj: bool,

        #[arg(long)]
        output: Option<String>,
    },
    Link {
        #[arg(required = true)]
        inputs: Vec<String>,

        #[arg(long, default_value = "a.bin")]
        output: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Asm { input, output, emit_obj } => {
            println!("Assembly mode");
            println!("Input file: {}", input);
            let obj_output = if emit_obj && output.ends_with(".bin") {
                replace_suffix_or_append(&output, ".bin", ".o")
            } else {
                output.clone()
            };
            println!("Output file: {}", obj_output);

            let arch = load_or_generate_arch_config("config.yml");
            println!("Using architecture: {:?}", arch);

            if emit_obj {
                println!("Assembling relocatable object {}...", input);
                let obj = assemble::assemble::assemble_file_to_object(&input, &arch);
                let yaml_string = serde_yaml::to_string(&obj).expect("Failed to serialize object");
                fs::write(&obj_output, yaml_string).expect("Failed to write object file");
                println!("Object written to {}", obj_output);
            } else {
                println!("Assembling {}...", input);
                let program = assemble::assemble::assemble_file_with_sections(&input, &arch);
                println!("Assembly finished. {} instructions generated.", program.text.len());

                for (i, inst) in program.text.iter().enumerate() {
                    println!("{}: 0x{:016X}", i, inst);
                }

                write_program_single(&program, &output);
            }
        }

        Commands::Cc { input, emit_obj, output } => {
            eprintln!("[DEPRECATED] `cc` is deprecated and kept for compatibility.");
            eprintln!("[DEPRECATED] Please use the LLVM backend for new projects.");
            println!("C compile mode: {}", input);

            let arch = load_or_generate_arch_config("config.yml");
            println!("Using architecture: {:?}", arch);

            let content = fs::read_to_string(&input).expect("Failed to read file");
            let mode = if emit_obj { EmitMode::Object } else { EmitMode::Executable };
            let asm_text = compile(&content, &arch, mode);
            println!("Compile finished. asm text length = {} chars", asm_text.len());
            println!("Result:\n{}", asm_text);

            if emit_obj {
                println!("Assembling relocatable object...");
                let obj = assemble::assemble::assemble_string_to_object(&asm_text, &arch);
                let output_name = output.unwrap_or_else(|| replace_suffix_or_append(&input, ".c", ".o"));
                let yaml_string = serde_yaml::to_string(&obj).expect("Failed to serialize object");
                fs::write(&output_name, yaml_string).expect("Failed to write object file");
                println!("Output: {}", output_name);
            } else {
                println!("Assembling code...");
                let program = assemble::assemble::assemble_string_with_sections(&asm_text, &arch);
                println!("Assembly finished. {} instructions generated.", program.text.len());

                let output_name = output.unwrap_or_else(|| replace_suffix_or_append(&input, ".c", ".bin"));
                write_program_single(&program, &output_name);
                println!("Output: {}", output_name);
            }
        }

        Commands::Link { inputs, output } => {
            println!("Link mode");
            let arch = load_or_generate_arch_config("config.yml");
            println!("Using architecture: {:?}", arch);

            let program = link_objects(&inputs, &arch);
            println!("Link finished. {} instructions generated.", program.text.len());

            write_program_single(&program, &output);
            println!("Output: {}", output);
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

fn write_program_single(program: &AssembledProgram, output: &str) {
    let mut bin_file = File::create(output).expect("Failed to create output file");

    let text_size = (program.text.len() as u32) * 8;
    let data_size = program.data.len() as u32;

    for v in [
        program.text_base,
        text_size,
        program.data_base,
        data_size,
        program.bss_base,
        program.bss_size,
    ] {
        bin_file
            .write_all(&v.to_le_bytes())
            .expect("Failed to write header");
    }

    for inst in &program.text {
        bin_file
            .write_all(&inst.to_le_bytes())
            .expect("Failed to write instruction");
    }

    if !program.data.is_empty() {
        bin_file
            .write_all(&program.data)
            .expect("Failed to write data segment");
    }

    bin_file.flush().expect("Failed to flush output file");
    println!("Binary written to {}", output);
    println!(
        "Header: TEXT_BASE=0x{:08X} TEXT_SIZE={} DATA_BASE=0x{:08X} DATA_SIZE={} BSS_BASE=0x{:08X} BSS_SIZE={}",
        program.text_base, text_size, program.data_base, data_size, program.bss_base, program.bss_size
    );
}
