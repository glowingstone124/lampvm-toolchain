#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Num(i64),
    Ident(String),
    Keyword(String),
    Punct(String),
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub pos: usize,
}
pub struct Tokenizer<'a> {
    input: &'a str,
    pos: usize,
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

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            while self.pos < self.input.len() {
                if let Some(ch) = self.peek_char() {
                    if ch.is_whitespace() {
                        self.pos += 1;
                        continue;
                    }
                }
                break;
            }

            if self.starts_with("//") {
                while self.pos < self.input.len() {
                    let ch = self.peek_char().unwrap();
                    self.pos += 1;
                    if ch == '\n' {
                        break;
                    }
                }
                continue;
            }

            if self.starts_with("/*") {
                self.pos += 2;
                while self.pos + 1 < self.input.len() && !self.starts_with("*/") {
                    self.pos += 1;
                }
                if self.starts_with("*/") {
                    self.pos += 2;
                }
                continue;
            }
            break;
        }
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();
        if self.pos >= self.input.len() {
            return Token { kind: TokenKind::Eof, pos: self.pos };
        }

        let c = self.peek_char().unwrap();
        let start_pos = self.pos;

        if c.is_ascii_digit() {
            let mut num = 0i64;
            while self.pos < self.input.len() {
                let ch = self.peek_char().unwrap();
                if ch.is_ascii_digit() {
                    num = num * 10 + (ch as i64 - '0' as i64);
                    self.pos += 1;
                } else {
                    break;
                }
            }
            return Token { kind: TokenKind::Num(num), pos: start_pos };
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let mut ident = String::new();
            while self.pos < self.input.len() {
                let ch = self.peek_char().unwrap();
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    ident.push(ch);
                    self.pos += 1;
                } else {
                    break;
                }
            }
            let kind = match ident.as_str() {
                "int" | "return" | "if" | "else" | "while" | "void" => {
                    TokenKind::Keyword(ident)
                }
                _ => TokenKind::Ident(ident),
            };
            return Token { kind, pos: start_pos };
        }

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

        let single = "+-*/(){}[];,<>=&!";
        if single.contains(c) {
            self.pos += 1;
            return Token {
                kind: TokenKind::Punct(c.to_string()),
                pos: start_pos,
            };
        }

        self.pos += 1;
        Token {
            kind: TokenKind::Punct(c.to_string()),
            pos: start_pos,
        }
    }
}
