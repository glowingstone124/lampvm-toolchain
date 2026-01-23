use crate::compiler::tokenizer::{TokenKind, Tokenizer};

pub fn compile(input: &String) {
    let mut tokenizer = Tokenizer::new(input);
    
    loop {
        let token = tokenizer.next_token();
        println!("{:?}", token);
        if token.kind == TokenKind::Eof {
            break;
        }
    }
}