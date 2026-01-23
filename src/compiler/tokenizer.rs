#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Num(i64),
    Ident(String),
    Reserved(char),
    Keyword(String),
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
    fn peek_char(&self) -> char {
        self.input[self.pos..].chars().next().unwrap()
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() && self.peek_char().is_whitespace() {
            self.pos += 1;
        }
    }
    pub fn new(input: &'a str) -> Self {
        Tokenizer { input, pos: 0 }
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();
        if self.pos >= self.input.len() {
            return Token { kind: TokenKind::Eof, pos: self.pos };
        }

        let c = self.peek_char();
        let start_pos = self.pos;

        if c.is_ascii_digit() {
            let mut num = 0i64;
            while self.pos < self.input.len() && self.peek_char().is_ascii_digit() {
                num = num * 10 + (self.peek_char() as i64 - '0' as i64);
                self.pos += 1;
            }
            return Token { kind: TokenKind::Num(num), pos: start_pos };
        }
        if c.is_ascii_alphabetic() {
            let mut ident = String::new();
            while self.pos < self.input.len() {
                let ch = self.peek_char();
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    ident.push(ch);
                    self.pos += 1;
                } else {
                    break;
                }
            }
            let kind = match ident.as_str() {
                "int" | "return" | "void" => TokenKind::Keyword(ident),
                _ => TokenKind::Ident(ident),
            };
            return Token { kind, pos: start_pos };
        }
        self.pos += 1;
        Token {kind: TokenKind::Reserved(c), pos: start_pos }
    }
}