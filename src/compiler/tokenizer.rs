#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Num(i64),
    Float(f32),
    Ident(String),
    Keyword(String),
    Punct(String),
    Str(String),
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub pos: usize, // byte offset
}

const KEYWORDS: [&str; 18] = [
    "char", "int", "long", "float", "void",
    "return", "if", "else", "while", "for",
    "asm", "break", "continue", "switch", "case",
    "default", "struct", "typedef",
];

pub struct Tokenizer<'a> {
    input: &'a str,
    pos: usize, // byte offset
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        Tokenizer { input, pos: 0 }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn starts_with(&self, s: &str) -> bool {
        self.input[self.pos..].starts_with(s)
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek_char()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // whitespace
            while let Some(ch) = self.peek_char() {
                if ch.is_whitespace() {
                    self.bump();
                    continue;
                }
                break;
            }

            // line comment
            if self.starts_with("//") {
                self.pos += 2;
                while let Some(ch) = self.peek_char() {
                    self.bump();
                    if ch == '\n' {
                        break;
                    }
                }
                continue;
            }

            // block comment
            if self.starts_with("/*") {
                self.pos += 2;
                while self.pos < self.input.len() && !self.starts_with("*/") {
                    if self.bump().is_none() {
                        break;
                    }
                }
                if self.starts_with("*/") {
                    self.pos += 2;
                }
                continue;
            }

            break;
        }
    }

    fn read_string_literal(&mut self) -> String {
        // current char should be '"'
        let quote = self.bump();
        debug_assert_eq!(quote, Some('"'));

        let mut s = String::new();
        loop {
            let ch = self.peek_char().unwrap_or_else(|| panic!("Unterminated string literal"));
            if ch == '"' {
                self.bump();
                break;
            }
            if ch == '\\' {
                self.bump();
                let esc = self.peek_char().unwrap_or_else(|| panic!("Unterminated escape sequence"));
                self.bump();
                match esc {
                    'n' => s.push('\n'),
                    't' => s.push('\t'),
                    'r' => s.push('\r'),
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    _ => panic!("Unsupported escape: \\{}", esc),
                }
            } else {
                self.bump();
                s.push(ch);
            }
        }
        s
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();
        if self.pos >= self.input.len() {
            return Token { kind: TokenKind::Eof, pos: self.pos };
        }

        let c = self.peek_char().unwrap();
        let start_pos = self.pos;

        // string literal
        if c == '"' {
            let lit = self.read_string_literal();
            return Token { kind: TokenKind::Str(lit), pos: start_pos };
        }

        // number (decimal/hex, with optional float syntax for decimal)
        if c.is_ascii_digit() {
            if self.starts_with("0x") || self.starts_with("0X") {
                self.pos += 2;
                let mut hex = String::new();
                while let Some(ch) = self.peek_char() {
                    if ch.is_ascii_hexdigit() {
                        hex.push(ch);
                        self.bump();
                    } else {
                        break;
                    }
                }
                if hex.is_empty() {
                    panic!("Invalid hex literal");
                }
                let num = i64::from_str_radix(&hex, 16).expect("Invalid hex literal");
                return Token { kind: TokenKind::Num(num), pos: start_pos };
            }

            let mut s = String::new();
            while let Some(ch) = self.peek_char() {
                if ch.is_ascii_digit() {
                    s.push(ch);
                    self.bump();
                } else {
                    break;
                }
            }

            let mut is_float = false;
            if self.peek_char() == Some('.') {
                is_float = true;
                s.push('.');
                self.bump();
                while let Some(ch) = self.peek_char() {
                    if ch.is_ascii_digit() {
                        s.push(ch);
                        self.bump();
                    } else {
                        break;
                    }
                }
            }

            if matches!(self.peek_char(), Some('e') | Some('E')) {
                is_float = true;
                s.push('e');
                self.bump();
                if matches!(self.peek_char(), Some('+') | Some('-')) {
                    s.push(self.bump().unwrap());
                }
                let mut has_exp_digit = false;
                while let Some(ch) = self.peek_char() {
                    if ch.is_ascii_digit() {
                        has_exp_digit = true;
                        s.push(ch);
                        self.bump();
                    } else {
                        break;
                    }
                }
                if !has_exp_digit {
                    panic!("Invalid float literal exponent");
                }
            }

            if is_float {
                let val = s.parse::<f32>().expect("Invalid float literal");
                return Token { kind: TokenKind::Float(val), pos: start_pos };
            }

            let num = s.parse::<i64>().expect("Invalid integer literal");
            return Token { kind: TokenKind::Num(num), pos: start_pos };
        }

        // identifier / keyword
        if c.is_ascii_alphabetic() || c == '_' {
            let mut ident = String::new();
            while let Some(ch) = self.peek_char() {
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    ident.push(ch);
                    self.bump();
                } else {
                    break;
                }
            }
            let kind = if KEYWORDS.contains(&ident.as_str()) {
                TokenKind::Keyword(ident)
            } else {
                TokenKind::Ident(ident)
            };
            return Token { kind, pos: start_pos };
        }

        // two-char punctuators
        let two_char = ["==", "!=", "<=", ">=", "&&", "||", "++", "->"];
        for op in &two_char {
            if self.starts_with(op) {
                self.pos += 2;
                return Token {
                    kind: TokenKind::Punct(op.to_string()),
                    pos: start_pos,
                };
            }
        }

        // single-char punctuators
        let single = "+-*/(){}[];,<>=&!:?.";
        if single.contains(c) {
            self.bump();
            return Token {
                kind: TokenKind::Punct(c.to_string()),
                pos: start_pos,
            };
        }

        self.bump();
        Token {
            kind: TokenKind::Punct(c.to_string()),
            pos: start_pos,
        }
    }
}
