use std::collections::HashMap;
use crate::assemble::config::ArchConfig;
use crate::assemble::inst::InstFormat;

fn is_reg_token(s: &str, arch: &ArchConfig) -> bool {
    let s = s.trim();
    if !s.starts_with(&arch.reg_prefix) {
        return false;
    }
    let rest = &s[arch.reg_prefix.len()..];
    !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
}

fn is_mem_token(s: &str) -> bool {
    let s = s.trim();
    s.starts_with('[') && s.ends_with(']')
}

fn parse_mem_operand(
    s: &str,
    arch: &ArchConfig,
) -> (u8, Option<u8>, Option<String>) {
    let s = s.trim();
    if !is_mem_token(s) {
        panic!("Invalid memory operand: {}", s);
    }
    let inner = s[1..s.len() - 1].trim();
    if inner.is_empty() {
        panic!("Empty memory operand");
    }

    let parts: Vec<&str> = inner
        .split('+')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();

    let mut regs: Vec<&str> = Vec::new();
    let mut imm_parts: Vec<&str> = Vec::new();
    for part in parts {
        if is_reg_token(part, arch) {
            regs.push(part);
        } else {
            imm_parts.push(part);
        }
    }

    if regs.is_empty() {
        panic!("Memory operand missing base register: {}", s);
    }
    if regs.len() > 2 {
        panic!("Memory operand has too many registers: {}", s);
    }

    let rs1 = parse_reg(regs[0], arch);
    let rs2 = if regs.len() == 2 {
        Some(parse_reg(regs[1], arch))
    } else {
        None
    };

    let imm_expr = if imm_parts.is_empty() {
        None
    } else {
        Some(imm_parts.join("+"))
    };

    (rs1, rs2, imm_expr)
}

fn mem_imm_value(
    imm_expr: Option<String>,
    arch: &ArchConfig,
    labels: &HashMap<String, u32>,
) -> u32 {
    if let Some(expr) = imm_expr {
        parse_imm(&expr, arch, labels)
    } else {
        0
    }
}

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
            if is_mem_token(args[0]) || is_mem_token(args[1]) {
                if is_mem_token(args[0]) && is_mem_token(args[1]) {
                    panic!("R1 format expects one memory operand at most");
                }
                let (rd_str, mem_str) = if is_mem_token(args[0]) {
                    (args[1], args[0])
                } else {
                    (args[0], args[1])
                };
                let rd = parse_reg(rd_str, arch);
                let (rs1, rs2, imm_expr) = parse_mem_operand(mem_str, arch);
                if rs2.is_some() {
                    panic!("R1 format does not allow indexed memory operand");
                }
                let imm = mem_imm_value(imm_expr, arch, labels);
                if imm != 0 {
                    panic!("R1 format memory operand must be [rs1]");
                }
                return (rd, rs1, 0, 0);
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
            if args.len() == 3 && !is_mem_token(args[0]) && !is_mem_token(args[1]) {
                return (
                    parse_reg(args[0], arch),
                    parse_reg(args[1], arch),
                    0,
                    parse_imm(args[2], arch,labels),
                );
            }
            if args.len() == 2 {
                if is_mem_token(args[0]) && is_mem_token(args[1]) {
                    panic!("RdRsImm expects one memory operand at most");
                }
                let (rd_str, mem_str) = if is_mem_token(args[0]) {
                    (args[1], args[0])
                } else {
                    (args[0], args[1])
                };
                let rd = parse_reg(rd_str, arch);
                let (rs1, rs2, imm_expr) = parse_mem_operand(mem_str, arch);
                if rs2.is_some() {
                    panic!("RdRsImm does not allow indexed memory operand");
                }
                let imm = mem_imm_value(imm_expr, arch, labels);
                return (rd, rs1, 0, imm);
            }
            panic!("RdRsImm expects rd, rs1, imm or rd, [rs1 + imm]");
        }
        InstFormat::RdImm => {
            if args.len() != 2 {
                panic!("RdImm expects rd, imm");
            }
            (parse_reg(args[0], arch), 0, 0, parse_imm(args[1], arch, labels))
        }
        InstFormat::RdRsRsImm => {
            if args.len() == 4 && !is_mem_token(args[0]) && !is_mem_token(args[1]) {
                return (
                    parse_reg(args[0], arch),
                    parse_reg(args[1], arch),
                    parse_reg(args[2], arch),
                    parse_imm(args[3], arch, labels),
                );
            }
            if args.len() == 2 {
                if is_mem_token(args[0]) && is_mem_token(args[1]) {
                    panic!("RdRsRsImm expects one memory operand at most");
                }
                let (rd_str, mem_str) = if is_mem_token(args[0]) {
                    (args[1], args[0])
                } else {
                    (args[0], args[1])
                };
                let rd = parse_reg(rd_str, arch);
                let (rs1, rs2, imm_expr) = parse_mem_operand(mem_str, arch);
                let rs2 = rs2.expect("RdRsRsImm expects indexed memory operand");
                let imm = mem_imm_value(imm_expr, arch, labels);
                return (rd, rs1, rs2, imm);
            }
            panic!("RdRsRsImm expects rd, rs1, rs2, imm or rd, [rs1 + rs2 + imm]");
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn arch() -> ArchConfig {
        ArchConfig::new(32, "r")
    }

    #[test]
    fn parse_rd_rsimm_bracket() {
        let arch = arch();
        let labels = HashMap::new();
        let args = vec!["r1", "[r2 + 4]"];
        let (rd, rs1, rs2, imm) = parse_operands(InstFormat::RdRsImm, &args, &arch, &labels);
        assert_eq!(rd, 1);
        assert_eq!(rs1, 2);
        assert_eq!(rs2, 0);
        assert_eq!(imm, 4);
    }

    #[test]
    fn parse_rd_rsimm_bracket_no_imm() {
        let arch = arch();
        let labels = HashMap::new();
        let args = vec!["r3", "[r4]"];
        let (rd, rs1, rs2, imm) = parse_operands(InstFormat::RdRsImm, &args, &arch, &labels);
        assert_eq!(rd, 3);
        assert_eq!(rs1, 4);
        assert_eq!(rs2, 0);
        assert_eq!(imm, 0);
    }

    #[test]
    fn parse_rd_rsrsimm_bracket_indexed() {
        let arch = arch();
        let labels = HashMap::new();
        let args = vec!["r1", "[r2 + r3 + 16]"];
        let (rd, rs1, rs2, imm) = parse_operands(InstFormat::RdRsRsImm, &args, &arch, &labels);
        assert_eq!(rd, 1);
        assert_eq!(rs1, 2);
        assert_eq!(rs2, 3);
        assert_eq!(imm, 16);
    }

    #[test]
    fn parse_rd_rs_bracket_out_style() {
        let arch = arch();
        let labels = HashMap::new();
        let args = vec!["[r1]", "r2"];
        let (rd, rs1, rs2, imm) = parse_operands(InstFormat::RdRs, &args, &arch, &labels);
        assert_eq!(rd, 2);
        assert_eq!(rs1, 1);
        assert_eq!(rs2, 0);
        assert_eq!(imm, 0);
    }

    #[test]
    fn parse_legacy_rd_rsimm() {
        let arch = arch();
        let labels = HashMap::new();
        let args = vec!["r1", "r2", "8"];
        let (rd, rs1, rs2, imm) = parse_operands(InstFormat::RdRsImm, &args, &arch, &labels);
        assert_eq!(rd, 1);
        assert_eq!(rs1, 2);
        assert_eq!(rs2, 0);
        assert_eq!(imm, 8);
    }
}
