use std::collections::HashMap;
use crate::compiler::compiler::{assign_offsets, BinOp, ConstValue, Expr, ExprKind, FuncBuilder, Function, Global, Program, Stmt, Type, UnOp};
use crate::compiler::tokenizer::{Token, TokenKind};

const TYPE_KEYWORDS: [(&str, Type); 5] = [
    ("char", Type::Char),
    ("int", Type::Int),
    ("long", Type::Long),
    ("float", Type::Float),
    ("void", Type::Void),
];

const DECL_TYPE_KEYWORDS: [&str; 4] = ["char", "int", "long", "float"];

fn type_from_keyword(s: &str) -> Option<Type> {
    TYPE_KEYWORDS
        .iter()
        .find(|(kw, _)| *kw == s)
        .map(|(_, ty)| ty.clone())
}

fn is_decl_type_keyword(s: &str) -> bool {
    DECL_TYPE_KEYWORDS.contains(&s)
}

pub(crate) struct Parser {
    tokens: Vec<Token>,
    idx: usize,
    func_builder: Option<FuncBuilder>,
    pub(crate) func_types: HashMap<String, Type>,
    pub(crate) global_types: HashMap<String, Type>,
}
impl Parser {
    pub(crate) fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens,
            idx: 0,
            func_builder: None,
            func_types: HashMap::new(),
            global_types: HashMap::new(),
        }
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Eof)
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.idx]
    }

    fn next(&mut self) -> Token {
        let tok = self.tokens[self.idx].clone();
        self.idx += 1;
        tok
    }

    fn consume_punct(&mut self, s: &str) -> bool {
        if let TokenKind::Punct(p) = &self.peek().kind {
            if p == s {
                self.next();
                return true;
            }
        }
        false
    }

    fn expect_punct(&mut self, s: &str) {
        if !self.consume_punct(s) {
            panic!("Expected punct '{}'", s);
        }
    }
    fn peek_punct(&self, s: &str) -> bool {
        matches!(&self.peek().kind, TokenKind::Punct(p) if p == s)
    }
    fn peek_keyword(&self, s: &str) -> bool {
        matches!(&self.peek().kind, TokenKind::Keyword(k) if k == s)
    }
    fn consume_void_rparen_as_empty_params(&mut self) -> bool {
        if self.peek_keyword("void") {
            if self.idx + 1 < self.tokens.len() {
                if matches!(&self.tokens[self.idx + 1].kind, TokenKind::Punct(p) if p == ")") {
                    self.next();
                    self.next();
                    return true;
                }
            }
        }
        false
    }
    fn consume_keyword(&mut self, s: &str) -> bool {
        if let TokenKind::Keyword(k) = &self.peek().kind {
            if k == s {
                self.next();
                return true;
            }
        }
        false
    }

    fn expect_ident(&mut self) -> String {
        match &self.next().kind {
            TokenKind::Ident(name) => name.clone(),
            _ => panic!("Expected identifier"),
        }
    }

    fn parse_type_spec(&mut self) -> Type {
        match &self.peek().kind {
            TokenKind::Keyword(k) => {
                if let Some(ty) = type_from_keyword(k) {
                    self.next();
                    ty
                } else {
                    panic!("Expected type specifier");
                }
            }
            _ => panic!("Expected type specifier"),
        }
    }

    fn parse_declarator(&mut self, base: Type) -> (Type, String) {
        let mut ty = base;
        while self.consume_punct("*") {
            ty = Type::Ptr(Box::new(ty));
        }
        let name = self.expect_ident();
        while self.consume_punct("[") {
            let n = match self.next().kind {
                TokenKind::Num(v) => v as usize,
                _ => panic!("Expected array size"),
            };
            self.expect_punct("]");
            ty = Type::Array(Box::new(ty), n);
        }
        (ty, name)
    }

    pub(crate) fn parse_program(&mut self) -> Program {
        let mut funcs = Vec::new();
        let mut globals = Vec::new();
        while !self.at_eof() {
            let base_type = self.parse_type_spec();
            let (decl_type, name) = self.parse_declarator(base_type.clone());
            if self.peek_punct("(") {
                let func = self.parse_function_after_decl(decl_type, name);
                self.func_types.insert(func.name.clone(), func.ret_type.clone());
                funcs.push(func);
            } else {
                let mut decls = Vec::new();
                let init = if self.consume_punct("=") {
                    Some(self.parse_global_init())
                } else {
                    None
                };
                decls.push((decl_type, name, init));
                while self.consume_punct(",") {
                    let (ty, name) = self.parse_declarator(base_type.clone());
                    let init = if self.consume_punct("=") {
                        Some(self.parse_global_init())
                    } else {
                        None
                    };
                    decls.push((ty, name, init));
                }
                self.expect_punct(";");

                for (ty, name, init) in decls {
                    if matches!(ty, Type::Void) {
                        panic!("Global variable cannot be void: {}", name);
                    }
                    if ty.is_array() && init.is_some() {
                        panic!("Array global initializers are not supported: {}", name);
                    }
                    if self.global_types.insert(name.clone(), ty.clone()).is_some() {
                        panic!("Duplicate global definition: {}", name);
                    }
                    globals.push(Global { name, ty, init });
                }
            }
        }
        Program { funcs, globals }
    }

    fn parse_function(&mut self) -> Function {
        let ret_type = self.parse_type_spec();
        let (ret_type, name) = self.parse_declarator(ret_type);
        self.parse_function_after_decl(ret_type, name)
    }

    fn parse_function_after_decl(&mut self, ret_type: Type, name: String) -> Function {
        self.expect_punct("(");

        let mut builder = FuncBuilder::new();

        if self.consume_punct(")") {
            //NO params
        } else {
            if self.consume_void_rparen_as_empty_params() {
                //NO params
            } else {
                loop {
                    let base = self.parse_type_spec();
                    let (mut ty, name) = self.parse_declarator(base);

                    if ty.is_array() {
                        ty=ty.decay();
                    }
                    let id = builder.add_var(name, ty);
                    builder.params.push(id);

                    if self.consume_punct(")") {
                        break;
                    }

                    self.expect_punct(",");
                }
            }
        }

        self.func_builder = Some(builder);
        let body = self.parse_block_stmt();
        let builder = self.func_builder.take().unwrap();
        let mut locals = builder.locals;

        assign_offsets(&mut locals, &builder.params);

        Function {
            name,
            params: builder.params,
            locals,
            body,
            ret_type,
        }
    }

    fn parse_block_stmt(&mut self) -> Stmt {
        self.expect_punct("{");

        {
            let builder = self.func_builder.as_mut().expect("parse_block_stmt called outside of function");
            builder.enter_scope();
        }

        let mut stmts = Vec::new();
        while !self.consume_punct("}") {
            stmts.push(self.parse_stmt());
        }

        {
            let builder = self.func_builder.as_mut().expect("parse_block_stmt called outside of function");
            builder.exit_scope();
        }

        Stmt::Block(stmts)
    }

    fn parse_stmt(&mut self) -> Stmt {
        if self.consume_keyword("asm") {
            self.expect_punct("(");
            let s = match self.next().kind {
                TokenKind::Str(k) => k,
                _ => panic!("asm() expects a string literal"),
            };
            self.expect_punct(")");
            self.expect_punct(";");
            return Stmt::InlineAsm(s);
        }
        if self.consume_keyword("return") {
            if self.consume_punct(";") {
                return Stmt::Return(None);
            }
            let expr = self.parse_expr();
            self.expect_punct(";");
            return Stmt::Return(Some(expr));
        }
        if self.consume_keyword("if") {
            self.expect_punct("(");
            let cond = self.parse_expr();
            self.expect_punct(")");
            let then_branch = Box::new(self.parse_stmt());
            let else_branch = if self.consume_keyword("else") {
                Some(Box::new(self.parse_stmt()))
            } else {
                None
            };
            return Stmt::If {
                cond,
                then_branch,
                else_branch,
            };
        }
        if self.consume_keyword("while") {
            self.expect_punct("(");
            let cond = self.parse_expr();
            self.expect_punct(")");
            let body = Box::new(self.parse_stmt());
            return Stmt::While { cond, body };
        }
        if self.peek_punct("{") {
            return self.parse_block_stmt();
        }
        if self.peek_is_type() {
            return self.parse_decl_stmt();
        }
        if self.consume_punct(";") {
            return Stmt::ExprStmt(None);
        }
        let expr = self.parse_expr();
        self.expect_punct(";");
        Stmt::ExprStmt(Some(expr))
    }

    fn peek_is_type(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::Keyword(ref k) if is_decl_type_keyword(k)
        )
    }

    fn parse_decl_stmt(&mut self) -> Stmt {
        let base = self.parse_type_spec();
        let mut decls = Vec::new();
        loop {
            let (ty, name) = self.parse_declarator(base.clone());
            let id = self
                .func_builder
                .as_mut()
                .unwrap()
                .add_var(name, ty);
            let init = if self.consume_punct("=") {
                Some(self.parse_expr())
            } else {
                None
            };
            decls.push((id, init));
            if self.consume_punct(";") {
                break;
            }
            self.expect_punct(",");
        }
        Stmt::Decl(decls)
    }

    fn parse_expr(&mut self) -> Expr {
        self.parse_assign()
    }

    fn parse_assign(&mut self) -> Expr {
        let mut node = self.parse_logical_or();
        if self.consume_punct("=") {
            let rhs = self.parse_assign();
            node = Expr {
                kind: ExprKind::Assign(Box::new(node), Box::new(rhs)),
            };
        }
        node
    }

    fn parse_logical_or(&mut self) -> Expr {
        let mut node = self.parse_logical_and();
        while self.consume_punct("||") {
            let rhs = self.parse_logical_and();
            node = Expr {
                kind: ExprKind::Binary {
                    op: BinOp::Or,
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                },
            };
        }
        node
    }

    fn parse_logical_and(&mut self) -> Expr {
        let mut node = self.parse_equality();
        while self.consume_punct("&&") {
            let rhs = self.parse_equality();
            node = Expr {
                kind: ExprKind::Binary {
                    op: BinOp::And,
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                },
            };
        }
        node
    }

    fn parse_equality(&mut self) -> Expr {
        let mut node = self.parse_relational();
        loop {
            if self.consume_punct("==") {
                let rhs = self.parse_relational();
                node = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Eq,
                        lhs: Box::new(node),
                        rhs: Box::new(rhs),
                    },
                };
            } else if self.consume_punct("!=") {
                let rhs = self.parse_relational();
                node = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Ne,
                        lhs: Box::new(node),
                        rhs: Box::new(rhs),
                    },
                };
            } else {
                break;
            }
        }
        node
    }

    fn parse_relational(&mut self) -> Expr {
        let mut node = self.parse_add();
        loop {
            if self.consume_punct("<") {
                let rhs = self.parse_add();
                node = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Lt,
                        lhs: Box::new(node),
                        rhs: Box::new(rhs),
                    },
                };
            } else if self.consume_punct("<=") {
                let rhs = self.parse_add();
                node = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Le,
                        lhs: Box::new(node),
                        rhs: Box::new(rhs),
                    },
                };
            } else if self.consume_punct(">") {
                let rhs = self.parse_add();
                node = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Gt,
                        lhs: Box::new(node),
                        rhs: Box::new(rhs),
                    },
                };
            } else if self.consume_punct(">=") {
                let rhs = self.parse_add();
                node = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Ge,
                        lhs: Box::new(node),
                        rhs: Box::new(rhs),
                    },
                };
            } else {
                break;
            }
        }
        node
    }

    fn parse_add(&mut self) -> Expr {
        let mut node = self.parse_mul();
        loop {
            if self.consume_punct("+") {
                let rhs = self.parse_mul();
                node = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Add,
                        lhs: Box::new(node),
                        rhs: Box::new(rhs),
                    },
                };
            } else if self.consume_punct("-") {
                let rhs = self.parse_mul();
                node = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Sub,
                        lhs: Box::new(node),
                        rhs: Box::new(rhs),
                    },
                };
            } else {
                break;
            }
        }
        node
    }

    fn parse_mul(&mut self) -> Expr {
        let mut node = self.parse_unary();
        loop {
            if self.consume_punct("*") {
                let rhs = self.parse_unary();
                node = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Mul,
                        lhs: Box::new(node),
                        rhs: Box::new(rhs),
                    },
                };
            } else if self.consume_punct("/") {
                let rhs = self.parse_unary();
                node = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Div,
                        lhs: Box::new(node),
                        rhs: Box::new(rhs),
                    },
                };
            } else {
                break;
            }
        }
        node
    }

    fn parse_unary(&mut self) -> Expr {
        if self.consume_punct("+") {
            let expr = self.parse_unary();
            return Expr {
                kind: ExprKind::Unary {
                    op: UnOp::Pos,
                    expr: Box::new(expr),
                },
            };
        }
        if self.consume_punct("-") {
            let expr = self.parse_unary();
            return Expr {
                kind: ExprKind::Unary {
                    op: UnOp::Neg,
                    expr: Box::new(expr),
                },
            };
        }
        if self.consume_punct("&") {
            let expr = self.parse_unary();
            return Expr {
                kind: ExprKind::Unary {
                    op: UnOp::Addr,
                    expr: Box::new(expr),
                },
            };
        }
        if self.consume_punct("*") {
            let expr = self.parse_unary();
            return Expr {
                kind: ExprKind::Unary {
                    op: UnOp::Deref,
                    expr: Box::new(expr),
                },
            };
        }
        if self.consume_punct("!") {
            let expr = self.parse_unary();
            return Expr {
                kind: ExprKind::Unary {
                    op: UnOp::Not,
                    expr: Box::new(expr),
                },
            };
        }
        self.parse_cast()
    }

    fn parse_cast(&mut self) -> Expr {
        if self.consume_punct("(") {
            if self.peek_is_type() || matches!(self.peek().kind, TokenKind::Keyword(ref k) if k == "void") {
                let base = self.parse_type_spec();
                let mut ty = base;
                while self.consume_punct("*") {
                    ty = Type::Ptr(Box::new(ty));
                }
                self.expect_punct(")");
                let expr =self.parse_unary();
                return Expr {
                    kind: ExprKind::Cast {
                        ty, expr: Box::new(expr),
                    }
                }
            }
            self.idx -= 1;
        }
        self.parse_postfix()
    }
    fn parse_postfix(&mut self) -> Expr {
        let mut node = self.parse_primary();
        loop {
            if self.consume_punct("[") {
                let index = self.parse_expr();
                self.expect_punct("]");
                node = Expr {
                    kind: ExprKind::Index {
                        base: Box::new(node),
                        index: Box::new(index),
                    },
                };
                continue;
            }
            break;
        }
        node
    }

    fn parse_primary(&mut self) -> Expr {
        if self.consume_punct("(") {
            let expr = self.parse_expr();
            self.expect_punct(")");
            return expr;
        }
        if let TokenKind::Num(n) = &self.peek().kind {
            let n = *n;
            self.next();
            return Expr {
                kind: ExprKind::Num(n),
            };
        }
        if let TokenKind::Float(n) = &self.peek().kind {
            let n = *n;
            self.next();
            return Expr {
                kind: ExprKind::Float(n),
            };
        }
        match &self.peek().kind {
            TokenKind::Ident(_) => {
                let name = self.expect_ident();
                if self.consume_punct("(") {
                    let mut args = Vec::new();
                    if !self.consume_punct(")") {
                        loop {
                            args.push(self.parse_expr());
                            if self.consume_punct(")") {
                                break;
                            }
                            self.expect_punct(",");
                        }
                    }
                    return Expr {
                        kind: ExprKind::Call { name, args },
                    };
                }
                if let Some(builder) = self.func_builder.as_ref() {
                    if let Some(id) = builder.find_var(&name) {
                        return Expr {
                            kind: ExprKind::Var(id),
                        };
                    }
                }
                return Expr {
                    kind: ExprKind::GlobalVar(name),
                };
            }
            _ => panic!("Unexpected token in primary"),
        }
    }

    fn parse_global_init(&mut self) -> ConstValue {
        let expr = self.parse_expr();
        eval_const_expr(&expr).unwrap_or_else(|| panic!("Global initializer must be constant"))
    }
}

fn eval_const_expr(expr: &Expr) -> Option<ConstValue> {
    match &expr.kind {
        ExprKind::Num(n) => Some(ConstValue::Int(*n)),
        ExprKind::Float(f) => Some(ConstValue::Float(*f)),
        ExprKind::Unary { op: UnOp::Pos, expr } => eval_const_expr(expr),
        ExprKind::Unary { op: UnOp::Neg, expr } => match eval_const_expr(expr) {
            Some(ConstValue::Int(v)) => Some(ConstValue::Int(-v)),
            Some(ConstValue::Float(v)) => Some(ConstValue::Float(-v)),
            _ => None,
        },
        ExprKind::Binary { op, lhs, rhs } => {
            let lv = eval_const_expr(lhs)?;
            let rv = eval_const_expr(rhs)?;
            match (lv, rv) {
                (ConstValue::Int(a), ConstValue::Int(b)) => {
                    let v = match op {
                        BinOp::Add => a + b,
                        BinOp::Sub => a - b,
                        BinOp::Mul => a * b,
                        BinOp::Div => a / b,
                        _ => return None,
                    };
                    Some(ConstValue::Int(v))
                }
                (ConstValue::Float(a), ConstValue::Float(b)) => {
                    let v = match op {
                        BinOp::Add => a + b,
                        BinOp::Sub => a - b,
                        BinOp::Mul => a * b,
                        BinOp::Div => a / b,
                        _ => return None,
                    };
                    Some(ConstValue::Float(v))
                }
                (ConstValue::Float(a), ConstValue::Int(b)) => {
                    let b = b as f32;
                    let v = match op {
                        BinOp::Add => a + b,
                        BinOp::Sub => a - b,
                        BinOp::Mul => a * b,
                        BinOp::Div => a / b,
                        _ => return None,
                    };
                    Some(ConstValue::Float(v))
                }
                (ConstValue::Int(a), ConstValue::Float(b)) => {
                    let a = a as f32;
                    let v = match op {
                        BinOp::Add => a + b,
                        BinOp::Sub => a - b,
                        BinOp::Mul => a * b,
                        BinOp::Div => a / b,
                        _ => return None,
                    };
                    Some(ConstValue::Float(v))
                }
            }
        }
        ExprKind::Cast { ty, expr } => {
            let v = eval_const_expr(expr)?;
            match (ty, v) {
                (Type::Float, ConstValue::Int(i)) => Some(ConstValue::Float(i as f32)),
                (Type::Float, ConstValue::Float(f)) => Some(ConstValue::Float(f)),
                (_, ConstValue::Float(f)) => Some(ConstValue::Int(f as i64)),
                (_, ConstValue::Int(i)) => Some(ConstValue::Int(i)),
            }
        }
        _ => None,
    }
}
