use crate::config::ArchConfig;
use crate::inst::InstFormat;
use std::collections::HashMap;

pub fn parse_operands(format: InstFormat, args: &[&str], arch: &ArchConfig, labels: &HashMap<String,u32>) -> (u8, u8, u8, u32) {
    match format {
        InstFormat::None => {
            if !args.is_empty() {
                panic!("Instruction takes no operands");
            }
            (0, 0, 0, 0)
        }

        InstFormat::RdRsRs => {
            if args.len() != 3 {
                panic!("R format expects rd, rs1, rs2");
            }
            (
                parse_reg(args[0], arch),
                parse_reg(args[1], arch),
                parse_reg(args[2], arch),
                0,
            )
        }

        InstFormat::RdRs => {
            if args.len() != 2 {
                panic!("R1 format expects rd, rs1");
            }
            (parse_reg(args[0], arch), parse_reg(args[1], arch), 0, 0)
        }

        InstFormat::I => {
            if args.len() != 1 {
                panic!("I format expects imm");
            }
            (0, 0, 0, parse_imm(args[0], arch, labels))
        }

        InstFormat::Rd => {
            if args.len() != 1 {
                panic!("Rd format expects rd");
            }
            (parse_reg(args[0], arch), 0, 0, 0)
        }

        InstFormat::RdRsImm => {
            if args.len() != 3 {
                panic!("RdRsImm expects rd, rs1, imm");
            }
            (
                parse_reg(args[0], arch),
                parse_reg(args[1], arch),
                0,
                parse_imm(args[2], arch,labels),
            )
        }
        InstFormat::RdImm => {
            if args.len() != 2 {
                panic!("RdImm expects rd, imm");
            }
            (parse_reg(args[0], arch), 0, 0, parse_imm(args[1], arch, labels))
        }
        InstFormat::RdRsRsImm => {
            if args.len() != 4 {
                panic!("RdRsRsImm expects rd, rs1, rs2, imm");
            }
            (
                parse_reg(args[0], arch),
                parse_reg(args[1], arch),
                parse_reg(args[2], arch),
                parse_imm(args[3], arch, labels),
            )
        }
    }
}

pub fn parse_reg(s: &str, arch: &ArchConfig) -> u8 {
    let r = s
        .trim_start_matches(&arch.reg_prefix)
        .parse::<u8>()
        .expect("Invalid register");

    arch.check_reg(r);
    r
}

pub fn parse_imm(s: &str,  arch: &ArchConfig,labels: &HashMap<String, u32>) -> u32 {
    let expr = s.trim();
    if let Some (&val) = &arch.macros.get(expr) {
        return val;
    }

    if let Some(&val) = labels.get(expr) {
        return val;
    }

    if let Some(hex) = expr.strip_prefix("0x") {
        return u32::from_str_radix(hex, 16).expect("Invalid hex immediate");
    }

    if let Ok(num) = expr.parse::<u32>() {
        return num;
    }

    parse_simple_expr(expr, &arch.macros, labels)
}

fn parse_simple_expr(expr: &str, macros: &HashMap<String, u32>, labels: &HashMap<String,u32>) -> u32 {
    let expr = expr.trim();

    if expr.starts_with("(") && expr.ends_with(")") {
        return parse_simple_expr(&expr[1..expr.len() - 1], macros, labels);
    }

    for op in ['+', '-'] {
        if let Some(pos) = expr.rfind(op) {
            let left = &expr[..pos];
            let right = &expr[pos + 1..];
            let lv = parse_simple_expr(left, macros, labels);
            let rv = parse_simple_expr(right, macros, labels);
            return match op {
                '+' => lv + rv,
                '-' => lv - rv,
                _ => unreachable!(),
            };
        }
    }
    for op in ['*', '/'] {
        if let Some(pos) = expr.rfind(op) {
            let left = &expr[..pos];
            let right = &expr[pos + 1..];
            let lv = parse_simple_expr(left, macros, labels);
            let rv = parse_simple_expr(right, macros, labels);
            return match op {
                '*' => lv * rv,
                '/' => lv / rv,
                _ => unreachable!(),
            }
        }
    }

    if let Some(&val) = macros.get(expr) {
        return val;
    }

    if let Some(&val) = labels.get(expr) {
        return val;
    }

    if let Some(hex) = expr.strip_prefix("0x") {
        return u32::from_str_radix(hex, 16).expect("Invalid hex immediate");
    }

    expr.parse::<u32>().unwrap_or_else(|_| {
        panic!("Invalid immediate or undefined label: '{}'", expr)
    })
}
