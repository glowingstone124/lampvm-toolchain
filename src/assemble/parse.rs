use std::collections::HashMap;
use crate::assemble::config::ArchConfig;
use crate::assemble::inst::InstFormat;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelocRef {
    pub symbol: String,
    pub addend: i32,
}

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

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    let first = match chars.next() {
        Some(ch) => ch,
        None => return false,
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn strip_parens(mut s: &str) -> &str {
    loop {
        let bytes = s.as_bytes();
        if bytes.len() >= 2 && bytes[0] == b'(' && bytes[bytes.len() - 1] == b')' {
            let mut depth = 0i32;
            let mut closes_at_end = false;
            for (i, ch) in s.char_indices() {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            closes_at_end = i == s.len() - 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if closes_at_end {
                s = &s[1..bytes.len() - 1];
                continue;
            }
        }
        return s;
    }
}

pub fn parse_imm_reloc(
    s: &str,
    arch: &ArchConfig,
) -> (u32, Option<RelocRef>) {
    let expr = strip_parens(s.trim());
    if let Some(&val) = arch.macros.get(expr) {
        return (val, None);
    }
    if let Some(hex) = expr.strip_prefix("0x") {
        return (u32::from_str_radix(hex, 16).expect("Invalid hex immediate"), None);
    }
    if let Ok(num) = expr.parse::<u32>() {
        return (num, None);
    }

    if expr.contains('*') || expr.contains('/') {
        let empty_labels: HashMap<String, u32> = HashMap::new();
        let val = parse_simple_expr(expr, &arch.macros, &empty_labels);
        return (val, None);
    }

    let mut symbol: Option<String> = None;
    let mut addend: i64 = 0;
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut sign = 1i64;

    let bytes = expr.as_bytes();
    if !bytes.is_empty() {
        if bytes[0] == b'+' {
            start = 1;
        } else if bytes[0] == b'-' {
            sign = -1;
            start = 1;
        }
    }

    for (i, ch) in expr.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            '+' | '-' if depth == 0 && i >= start => {
                let part = expr[start..i].trim();
                if part.is_empty() {
                    panic!("Invalid immediate expression: {}", expr);
                }
                let part = strip_parens(part);
                let part_sign = sign;
                if let Some(&val) = arch.macros.get(part) {
                    addend += part_sign * (val as i64);
                } else if let Some(hex) = part.strip_prefix("0x") {
                    let val = u32::from_str_radix(hex, 16).expect("Invalid hex immediate") as i64;
                    addend += part_sign * val;
                } else if let Ok(num) = part.parse::<i64>() {
                    addend += part_sign * num;
                } else if is_ident(part) {
                    if part_sign != 1 {
                        panic!("Negative symbol term not supported: {}", expr);
                    }
                    if symbol.is_some() {
                        panic!("Multiple symbols in immediate expression: {}", expr);
                    }
                    symbol = Some(part.to_string());
                } else {
                    let empty_labels: HashMap<String, u32> = HashMap::new();
                    let val = parse_simple_expr(part, &arch.macros, &empty_labels) as i64;
                    addend += part_sign * val;
                }
                sign = if ch == '+' { 1 } else { -1 };
                start = i + ch.len_utf8();
            }
            _ => {}
        }
    }

    let last = expr[start..].trim();
    if last.is_empty() {
        panic!("Invalid immediate expression: {}", expr);
    }
    let last = strip_parens(last);
    if let Some(&val) = arch.macros.get(last) {
        addend += sign * (val as i64);
    } else if let Some(hex) = last.strip_prefix("0x") {
        let val = u32::from_str_radix(hex, 16).expect("Invalid hex immediate") as i64;
        addend += sign * val;
    } else if let Ok(num) = last.parse::<i64>() {
        addend += sign * num;
    } else if is_ident(last) {
        if sign != 1 {
            panic!("Negative symbol term not supported: {}", expr);
        }
        if symbol.is_some() {
            panic!("Multiple symbols in immediate expression: {}", expr);
        }
        symbol = Some(last.to_string());
    } else {
        let empty_labels: HashMap<String, u32> = HashMap::new();
        let val = parse_simple_expr(last, &arch.macros, &empty_labels) as i64;
        addend += sign * val;
    }

    if let Some(sym) = symbol {
        if addend < i32::MIN as i64 || addend > i32::MAX as i64 {
            panic!("Immediate addend out of range: {}", expr);
        }
        return (0, Some(RelocRef { symbol: sym, addend: addend as i32 }));
    }

    if addend < 0 || addend > u32::MAX as i64 {
        panic!("Immediate out of range: {}", expr);
    }
    (addend as u32, None)
}

pub fn parse_operands_reloc(
    format: InstFormat,
    args: &[&str],
    arch: &ArchConfig,
) -> (u8, u8, u8, u32, Option<RelocRef>) {
    match format {
        InstFormat::None => {
            if !args.is_empty() {
                panic!("Instruction takes no operands");
            }
            (0, 0, 0, 0, None)
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
                None,
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
                if imm_expr.is_some() {
                    panic!("R1 format memory operand must be [rs1]");
                }
                return (rd, rs1, 0, 0, None);
            }
            (parse_reg(args[0], arch), parse_reg(args[1], arch), 0, 0, None)
        }

        InstFormat::I => {
            if args.len() != 1 {
                panic!("I format expects imm");
            }
            let (imm, reloc) = parse_imm_reloc(args[0], arch);
            (0, 0, 0, imm, reloc)
        }

        InstFormat::Rd => {
            if args.len() != 1 {
                panic!("Rd format expects rd");
            }
            (parse_reg(args[0], arch), 0, 0, 0, None)
        }

        InstFormat::RdRsImm => {
            if args.len() == 3 && !is_mem_token(args[0]) && !is_mem_token(args[1]) {
                let (imm, reloc) = parse_imm_reloc(args[2], arch);
                return (
                    parse_reg(args[0], arch),
                    parse_reg(args[1], arch),
                    0,
                    imm,
                    reloc,
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
                let (imm, reloc) = if let Some(expr) = imm_expr {
                    parse_imm_reloc(&expr, arch)
                } else {
                    (0, None)
                };
                return (rd, rs1, 0, imm, reloc);
            }
            panic!("RdRsImm expects rd, rs1, imm or rd, [rs1 + imm]");
        }
        InstFormat::RdImm => {
            if args.len() != 2 {
                panic!("RdImm expects rd, imm");
            }
            let (imm, reloc) = parse_imm_reloc(args[1], arch);
            (parse_reg(args[0], arch), 0, 0, imm, reloc)
        }
        InstFormat::RdRsRsImm => {
            if args.len() == 4 && !is_mem_token(args[0]) && !is_mem_token(args[1]) {
                let (imm, reloc) = parse_imm_reloc(args[3], arch);
                return (
                    parse_reg(args[0], arch),
                    parse_reg(args[1], arch),
                    parse_reg(args[2], arch),
                    imm,
                    reloc,
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
                let (imm, reloc) = if let Some(expr) = imm_expr {
                    parse_imm_reloc(&expr, arch)
                } else {
                    (0, None)
                };
                return (rd, rs1, rs2, imm, reloc);
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
