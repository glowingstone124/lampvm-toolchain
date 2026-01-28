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

                write_program_bin(&program.text, &output);
                write_data_and_layout(&program, &output);
            }
        }

        Commands::Cc { input, emit_obj, output } => {
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
                write_program_bin(&program.text, &output_name);
                write_data_and_layout(&program, &output_name);
                println!("Output: {}", output_name);
            }
        }

        Commands::Link { inputs, output } => {
            println!("Link mode");
            let arch = load_or_generate_arch_config("config.yml");
            println!("Using architecture: {:?}", arch);

            let program = link_objects(&inputs, &arch);
            println!("Link finished. {} instructions generated.", program.text.len());

            write_program_bin(&program.text, &output);
            write_data_and_layout(&program, &output);
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

fn write_data_and_layout(program: &AssembledProgram, output: &str) {
    if !program.data.is_empty() {
        let data_output = replace_suffix_or_append(output, ".bin", ".data");
        let mut data_file = File::create(&data_output).expect("Failed to create data output file");
        data_file.write_all(&program.data).expect("Failed to write data segment");
        data_file.flush().expect("Failed to flush data output file");
        println!("Data segment written to {}", data_output);
    }

    let layout_output = replace_suffix_or_append(output, ".bin", ".layout");
    let layout = format!(
        "TEXT_BASE=0x{0:08X}\nTEXT_SIZE={1}\nDATA_BASE=0x{2:08X}\nDATA_SIZE={3}\nBSS_BASE=0x{4:08X}\nBSS_SIZE={5}\n",
        program.text_base,
        (program.text.len() as u32) * 8,
        program.data_base,
        program.data.len() as u32,
        program.bss_base,
        program.bss_size,
    );
    fs::write(&layout_output, layout).expect("Failed to write layout file");
    println!("Layout written to {}", layout_output);
}
