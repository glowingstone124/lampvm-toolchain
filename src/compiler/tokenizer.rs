#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Num(i64),
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

        // number (decimal only)
        if c.is_ascii_digit() {
            let mut num = 0i64;
            while let Some(ch) = self.peek_char() {
                if ch.is_ascii_digit() {
                    num = num * 10 + (ch as i64 - '0' as i64);
                    self.bump();
                } else {
                    break;
                }
            }
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
            let kind = match ident.as_str() {
                "int" | "return" | "if" | "else" | "while" | "void" | "asm" => {
                    TokenKind::Keyword(ident)
                }
                _ => TokenKind::Ident(ident),
            };
            return Token { kind, pos: start_pos };
        }

        // two-char punctuators
        let two_char = ["==", "!=", "<=", ">=", "&&", "||"];
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
        let single = "+-*/(){}[];,<>=&!";
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
