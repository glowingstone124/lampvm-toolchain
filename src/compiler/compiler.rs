use std::collections::HashMap;
use std::fmt::format;
use std::thread::sleep;
use crate::compiler::tokenizer::{Token, TokenKind, Tokenizer};

#[derive(Debug, Clone, PartialEq)]
enum Type {
    Int,
    Void,
    Ptr(Box<Type>),
    Array(Box<Type>, usize),
}

impl Type {
    fn size(&self) -> usize {
        match self {
            Type::Int => 4,
            Type::Void => 0,
            Type::Ptr(_) => 4,
            Type::Array(elem, n) => elem.size() * *n,
        }
    }

    fn is_array(&self) -> bool {
        matches!(self, Type::Array(_, _))
    }

    fn decay(&self) -> Type {
        match self {
            Type::Array(elem, _) => Type::Ptr(elem.clone()),
            _ => self.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct Var {
    name: String,
    ty: Type,
    offset: i32,
}

#[derive(Debug, Clone)]
struct Function {
    name: String,
    params: Vec<usize>,
    locals: Vec<Var>,
    body: Stmt,
    ret_type: Type,
}

#[derive(Debug, Clone)]
struct Program {
    funcs: Vec<Function>,
}

#[derive(Debug, Clone)]
enum Stmt {
    Return(Option<Expr>),
    If {
        cond: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
    While {
        cond: Expr,
        body: Box<Stmt>,
    },
    Block(Vec<Stmt>),
    ExprStmt(Option<Expr>),
    Decl(Vec<(usize, Option<Expr>)>),
}

#[derive(Debug, Clone)]
struct Expr {
    kind: ExprKind,
}

#[derive(Debug, Clone)]
enum ExprKind {
    Num(i64),
    Var(usize),
    Assign(Box<Expr>, Box<Expr>),
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Unary {
        op: UnOp,
        expr: Box<Expr>,
    },
    Call {
        name: String,
        args: Vec<Expr>,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, Copy)]
enum UnOp {
    Pos,
    Neg,
    Addr,
    Deref,
    Not,
}

struct Parser {
    tokens: Vec<Token>,
    idx: usize,
    func_builder: Option<FuncBuilder>,
    func_types: HashMap<String, Type>,
}

struct FuncBuilder {
    locals: Vec<Var>,
    scopes: Vec<HashMap<String, usize>>,
    params: Vec<usize>,
}

impl FuncBuilder {
    fn new() -> Self {
        FuncBuilder {
            locals: Vec::new(),
            scopes: vec![HashMap::new()],
            params: Vec::new(),
        }
    }

    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn add_var(&mut self, name: String, ty: Type) -> usize {
        let id = self.locals.len();
        self.locals.push(Var {
            name: name.clone(),
            ty,
            offset: 0,
        });
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, id);
        }
        id
    }

    fn find_var(&self, name: &str) -> Option<usize> {
        for scope in self.scopes.iter().rev() {
            if let Some(id) = scope.get(name) {
                return Some(*id);
            }
        }
        None
    }
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens,
            idx: 0,
            func_builder: None,
            func_types: HashMap::new(),
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
        if self.consume_keyword("int") {
            Type::Int
        } else if self.consume_keyword("void") {
            Type::Void
        } else {
            panic!("Expected type specifier");
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

    fn parse_program(&mut self) -> Program {
        let mut funcs = Vec::new();
        while !self.at_eof() {
            let func = self.parse_function();
            self.func_types.insert(func.name.clone(), func.ret_type.clone());
            funcs.push(func);
        }
        Program { funcs }
    }

    fn parse_function(&mut self) -> Function {
        let ret_type = self.parse_type_spec();
        let (ret_type, name) = {
            let (ty, name) = self.parse_declarator(ret_type);
            (ty, name)
        };
        self.expect_punct("(");

        let mut builder = FuncBuilder::new();

        if !self.consume_punct(")") {
            if self.consume_keyword("void") && self.consume_punct(")") {
            } else {
                loop {
                    let base = self.parse_type_spec();
                    let (mut ty, name) = self.parse_declarator(base);
                    if ty.is_array() {
                        ty = ty.decay();
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
        let mut builder = self.func_builder.take().unwrap();
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
        if self.consume_punct("{") {
            self.idx -= 1;
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
            TokenKind::Keyword(ref k) if k == "int"
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
                let id = self
                    .func_builder
                    .as_ref()
                    .unwrap()
                    .find_var(&name)
                    .unwrap_or_else(|| panic!("Undefined variable: {}", name));
                Expr {
                    kind: ExprKind::Var(id),
                }
            }
            _ => panic!("Unexpected token in primary"),
        }
    }
}

fn assign_offsets(locals: &mut [Var], params: &[usize]) {
    let mut offset = 4i32;
    for &id in params {
        let size = locals[id].ty.size() as i32;
        locals[id].offset = offset;
        offset += size;
    }
    for (i, var) in locals.iter_mut().enumerate() {
        if params.contains(&i) {
            continue;
        }
        let size = var.ty.size() as i32;
        var.offset = offset;
        offset += size;
    }
}

struct Codegen<'a> {
    out: Vec<String>,
    label: usize,
    func_types: &'a HashMap<String, Type>,
}

impl<'a> Codegen<'a> {
    fn new(func_types: &'a HashMap<String, Type>) -> Self {
        Codegen {
            out: Vec::new(),
            label: 0,
            func_types,
        }
    }

    fn emit(&mut self, s: impl Into<String>) {
        self.out.push(s.into());
    }

    fn new_label(&mut self, prefix: &str) -> String {
        let l = format!("{}_{}", prefix, self.label);
        self.label += 1;
        l
    }

    fn gen_program(&mut self, prog: &Program) {
        self.emit("movi r30, DATA_STACK_BASE + DATA_STACK_SIZE");
        self.emit("call main");
        self.emit("halt");
        for func in &prog.funcs {
            self.gen_function(func);
        }
    }

    fn gen_function(&mut self, func: &Function) {
        let ret_label = self.new_label("Lret");
        self.emit(format!("{}:", func.name));

        let frame_size = self.frame_size(func);

        self.emit("mov r9, r30"); // r9 old sp
        self.emit(format!("movi r8, {}", frame_size)); // r8 framesize
        self.emit("sub r30, r30, r8");
        self.emit("store32 [r30+0], r9"); // save old sp

        for (i, &param_id) in func.params.iter().enumerate() {
            if i >= 8 {
                panic!("More than 8 parameters not supported");
            }
            let offset = func.locals[param_id].offset;
            self.emit(format!("store32 [r30 + {}], r{}", offset, i));
        }

        self.gen_stmt(&func.body, func, &ret_label);

        if func.ret_type != Type::Void {
            self.emit("movi r0, 0");
        }

        self.emit(format!("{}:", ret_label));
        self.emit("load32 r9, [r30 + 0]");
        self.emit("mov r30, r9");
        self.emit("ret");

    }

    fn frame_size(&self, func: &Function) -> i32 {
        let mut max = 4i32;
        for var in &func.locals {
            let end = var.offset + var.ty.size() as i32;
            if end > max {
                max = end;
            }
        }
        max
    }

    fn gen_stmt(&mut self, stmt: &Stmt, func: &Function, ret_label: &str) {
        match stmt {
            Stmt::Return(expr) => {
                if let Some(e) = expr {
                    self.gen_expr(e, func);
                }
                self.emit(format!("jmp {}", ret_label));
            }
            Stmt::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let else_label = self.new_label("Lelse");
                let end_label = self.new_label("Lend");
                self.gen_expr(cond, func);
                self.emit("cmpi r0, 0");
                self.emit(format!("jz {}", else_label));
                self.gen_stmt(then_branch, func, ret_label);
                self.emit(format!("jmp {}", end_label));
                self.emit(format!("{}:", else_label));
                if let Some(else_stmt) = else_branch {
                    self.gen_stmt(else_stmt, func, ret_label);
                }
                self.emit(format!("{}:", end_label));
            }
            Stmt::While { cond, body } => {
                let begin_label = self.new_label("Lbegin");
                let end_label = self.new_label("Lend");
                self.emit(format!("{}:", begin_label));
                self.gen_expr(cond, func);
                self.emit("cmpi r0, 0");
                self.emit(format!("jz {}", end_label));
                self.gen_stmt(body, func, ret_label);
                self.emit(format!("jmp {}", begin_label));
                self.emit(format!("{}:", end_label));
            }
            Stmt::Block(stmts) => {
                for s in stmts {
                    self.gen_stmt(s, func, ret_label);
                }
            }
            Stmt::ExprStmt(expr) => {
                if let Some(e) = expr {
                    self.gen_expr(e, func);
                }
            }
            Stmt::Decl(vars) => {
                for (id, init) in vars {
                    if let Some(expr) = init {
                        self.gen_expr(expr, func);
                        let offset = func.locals[*id].offset;
                        self.emit(format!("store32 [r30 + {}], r0", offset));
                    }
                }
            }
        }
    }

    fn gen_expr(&mut self, expr: &Expr, func: &Function) {
        match &expr.kind {
            ExprKind::Num(n) => {
                self.emit(format!("movi r0, {}", *n as i32));
            }
            ExprKind::Var(id) => {
                let var = &func.locals[*id];
                if var.ty.is_array() {
                    self.gen_lvalue(expr, func);
                } else {
                    self.emit(format!("load32 r0, [r30 + {}]", var.offset));
                }
            }
            ExprKind::Assign(lhs, rhs) => {
                self.gen_lvalue(lhs, func);
                self.emit("push r0");
                self.gen_expr(rhs, func);
                self.emit("pop r1");
                if let Some(lty) = self.lvalue_type(lhs, func) {
                    if lty.is_array() {
                        panic!("Cannot assign to array");
                    }
                }
                self.emit("store32 [r1], r0");
            }
            ExprKind::Binary { op, lhs, rhs } => {
                self.gen_binary(*op, lhs, rhs, func);
            }
            ExprKind::Unary { op, expr } => {
                self.gen_unary(*op, expr, func);
            }
            ExprKind::Call { name, args } => {
                if args.len() > 8 {
                    panic!("More than 8 arguments not supported");
                }
                for arg in args {
                    self.gen_expr(arg, func);
                    self.emit("push r0");
                }
                for i in (0..args.len()).rev() {
                    self.emit(format!("pop r{}", i));
                }
                self.emit(format!("call {}", name));
            }
            ExprKind::Index { base, index } => {
                self.gen_lvalue(expr, func);
                self.emit("load32 r0, [r0]");
            }
        }
    }

    fn gen_unary(&mut self, op: UnOp, expr: &Expr, func: &Function) {
        match op {
            UnOp::Pos => {
                self.gen_expr(expr, func);
            }
            UnOp::Neg => {
                self.gen_expr(expr, func);
                self.emit("movi r1, 0");
                self.emit("sub r0, r1, r0");
            }
            UnOp::Addr => {
                self.gen_lvalue(expr, func);
            }
            UnOp::Deref => {
                self.gen_expr(expr, func);
                self.emit("load32 r0, [r0]");
            }
            UnOp::Not => {
                self.gen_expr(expr, func);
                let t = self.new_label("Lnot");
                let end = self.new_label("Lend");
                self.emit("cmpi r0, 0");
                self.emit(format!("jz {}", t));
                self.emit("movi r0, 0");
                self.emit(format!("jmp {}", end));
                self.emit(format!("{}:", t));
                self.emit("movi r0, 1");
                self.emit(format!("{}:", end));
            }
        }
    }

    fn gen_binary(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr, func: &Function) {
        match op {
            BinOp::And => {
                let end = self.new_label("Land_end");
                let false_l = self.new_label("Land_false");
                self.gen_expr(lhs, func);
                self.emit("cmpi r0, 0");
                self.emit(format!("jz {}", false_l));
                self.gen_expr(rhs, func);
                self.emit("cmpi r0, 0");
                self.emit(format!("jz {}", false_l));
                self.emit("movi r0, 1");
                self.emit(format!("jmp {}", end));
                self.emit(format!("{}:", false_l));
                self.emit("movi r0, 0");
                self.emit(format!("{}:", end));
            }
            BinOp::Or => {
                let end = self.new_label("Lor_end");
                let true_l = self.new_label("Lor_true");
                self.gen_expr(lhs, func);
                self.emit("cmpi r0, 0");
                self.emit(format!("jnz {}", true_l));
                self.gen_expr(rhs, func);
                self.emit("cmpi r0, 0");
                self.emit(format!("jnz {}", true_l));
                self.emit("movi r0, 0");
                self.emit(format!("jmp {}", end));
                self.emit(format!("{}:", true_l));
                self.emit("movi r0, 1");
                self.emit(format!("{}:", end));
            }
            BinOp::Eq
            | BinOp::Ne
            | BinOp::Lt
            | BinOp::Le
            | BinOp::Gt
            | BinOp::Ge => {
                self.gen_expr(lhs, func);
                self.emit("push r0");
                self.gen_expr(rhs, func);
                self.emit("pop r1");
                self.emit("cmp r1, r0");
                let t = self.new_label("Lcmp_true");
                let end = self.new_label("Lcmp_end");
                let jmp = match op {
                    BinOp::Eq => "jz",
                    BinOp::Ne => "jnz",
                    BinOp::Lt => "jl",
                    BinOp::Le => "jle",
                    BinOp::Gt => "jg",
                    BinOp::Ge => "jge",
                    _ => unreachable!(),
                };
                self.emit(format!("{} {}", jmp, t));
                self.emit("movi r0, 0");
                self.emit(format!("jmp {}", end));
                self.emit(format!("{}:", t));
                self.emit("movi r0, 1");
                self.emit(format!("{}:", end));
            }
            BinOp::Add | BinOp::Sub => {
                let lt = self.expr_type(lhs, func);
                let rt = self.expr_type(rhs, func);
                match (lt, rt) {
                    (Type::Ptr(elem), Type::Int) => {
                        self.gen_expr(lhs, func);
                        self.emit("push r0");
                        self.gen_expr(rhs, func);
                        self.scale_reg("r0", elem.size());
                        self.emit("pop r1");
                        let inst = if matches!(op, BinOp::Add) { "add" } else { "sub" };
                        self.emit(format!("{} r0, r1, r0", inst));
                    }
                    (Type::Int, Type::Ptr(elem)) => {
                        if matches!(op, BinOp::Sub) {
                            panic!("int - ptr not supported");
                        }
                        self.gen_expr(lhs, func);
                        self.emit("push r0");
                        self.gen_expr(rhs, func);
                        self.emit("pop r1");
                        self.scale_reg("r1", elem.size());
                        self.emit("add r0, r0, r1");
                    }
                    (Type::Ptr(elem), Type::Ptr(_)) => {
                        if matches!(op, BinOp::Add) {
                            panic!("ptr + ptr not supported");
                        }
                        self.gen_expr(lhs, func);
                        self.emit("push r0");
                        self.gen_expr(rhs, func);
                        self.emit("pop r1");
                        self.emit("sub r0, r1, r0");
                        if elem.size() != 1 {
                            self.emit(format!("movi r1, {}", elem.size()));
                            self.emit("div r0, r0, r1");
                        }
                    }
                    _ => {
                        self.gen_expr(lhs, func);
                        self.emit("push r0");
                        self.gen_expr(rhs, func);
                        self.emit("pop r1");
                        let inst = if matches!(op, BinOp::Add) { "add" } else { "sub" };
                        self.emit(format!("{} r0, r1, r0", inst));
                    }
                }
            }
            BinOp::Mul | BinOp::Div => {
                self.gen_expr(lhs, func);
                self.emit("push r0");
                self.gen_expr(rhs, func);
                self.emit("pop r1");
                let inst = if matches!(op, BinOp::Mul) { "mul" } else { "div" };
                self.emit(format!("{} r0, r1, r0", inst));
            }
        }
    }

    fn scale_reg(&mut self, reg: &str, size: usize) {
        if size == 1 {
            return;
        }
        if size.is_power_of_two() {
            let shift = size.trailing_zeros();
            self.emit(format!("shl {}, {}, {}", reg, reg, shift));
        } else {
            self.emit(format!("movi r2, {}", size));
            self.emit(format!("mul {}, {}, r2", reg, reg));
        }
    }

    fn gen_lvalue(&mut self, expr: &Expr, func: &Function) {
        match &expr.kind {
            ExprKind::Var(id) => {
                let var = &func.locals[*id];
                self.emit(format!("movi r1, {}", var.offset));
                self.emit("add r0, r30, r1");
            }
            ExprKind::Unary { op: UnOp::Deref, expr } => {
                self.gen_expr(expr, func);
            }
            ExprKind::Index { base, index } => {
                let base_ty = self.expr_type(base, func);
                let elem = match base_ty {
                    Type::Ptr(elem) => elem,
                    Type::Array(elem, _) => elem,
                    _ => panic!("Indexing non-pointer/array"),
                };
                self.gen_base_ptr(base, func);
                self.emit("push r0");
                self.gen_expr(index, func);
                self.scale_reg("r0", elem.size());
                self.emit("pop r1");
                self.emit("add r0, r1, r0");
            }
            _ => panic!("Invalid lvalue"),
        }
    }

    fn gen_base_ptr(&mut self, expr: &Expr, func: &Function) {
        match &expr.kind {
            ExprKind::Var(id) => {
                let var = &func.locals[*id];
                if var.ty.is_array() {
                    self.emit(format!("movi r1, {}", var.offset));
                    self.emit("add r0, r30, r1");
                } else {
                    self.gen_expr(expr, func);
                }
            }
            _ => self.gen_expr(expr, func),
        }
    }

    fn expr_type(&self, expr: &Expr, func: &Function) -> Type {
        match &expr.kind {
            ExprKind::Num(_) => Type::Int,
            ExprKind::Var(id) => func.locals[*id].ty.decay(),
            ExprKind::Assign(lhs, _) => self.expr_type(lhs, func),
            ExprKind::Binary { op, lhs, rhs } => match op {
                BinOp::Add | BinOp::Sub => {
                    let lt = self.expr_type(lhs, func);
                    let rt = self.expr_type(rhs, func);
                    match (lt, rt) {
                        (Type::Ptr(elem), Type::Int) => Type::Ptr(elem),
                        (Type::Int, Type::Ptr(elem)) if matches!(op, BinOp::Add) => Type::Ptr(elem),
                        (Type::Ptr(_), Type::Ptr(_)) => Type::Int,
                        _ => Type::Int,
                    }
                }
                BinOp::Mul | BinOp::Div => Type::Int,
                BinOp::Eq
                | BinOp::Ne
                | BinOp::Lt
                | BinOp::Le
                | BinOp::Gt
                | BinOp::Ge
                | BinOp::And
                | BinOp::Or => Type::Int,
            },
            ExprKind::Unary { op, expr } => match op {
                UnOp::Addr => {
                    let base = self.lvalue_type(expr, func).unwrap();
                    Type::Ptr(Box::new(base))
                }
                UnOp::Deref => match self.expr_type(expr, func) {
                    Type::Ptr(elem) => (*elem).clone(),
                    _ => panic!("Cannot dereference non-pointer"),
                },
                _ => Type::Int,
            },
            ExprKind::Call { name, .. } => self
                .func_types
                .get(name)
                .cloned()
                .unwrap_or(Type::Int),
            ExprKind::Index { base, .. } => match self.expr_type(base, func) {
                Type::Ptr(elem) => (*elem).clone(),
                Type::Array(elem, _) => (*elem).clone(),
                _ => panic!("Indexing non-pointer/array"),
            },
        }
    }

    fn lvalue_type(&self, expr: &Expr, func: &Function) -> Option<Type> {
        match &expr.kind {
            ExprKind::Var(id) => Some(func.locals[*id].ty.clone()),
            ExprKind::Unary { op: UnOp::Deref, expr } => match self.expr_type(expr, func) {
                Type::Ptr(elem) => Some(*elem),
                _ => None,
            },
            ExprKind::Index { base, .. } => match self.expr_type(base, func) {
                Type::Ptr(elem) => Some(*elem),
                Type::Array(elem, _) => Some(*elem),
                _ => None,
            },
            _ => None,
        }
    }
}

pub fn compile(input: &String) -> String {
    let mut tokenizer = Tokenizer::new(input);
    let mut tokens = Vec::new();

    loop {
        let tok = tokenizer.next_token();
        let is_eof = matches!(tok.kind, TokenKind::Eof);
        tokens.push(tok);
        if is_eof {
            break;
        }
    }

    let mut parser = Parser::new(tokens);
    let prog = parser.parse_program();

    let mut codegen = Codegen::new(&parser.func_types);
    codegen.gen_program(&prog);

    let mut asm_output = String::new();
    for line in &codegen.out {
        asm_output.push_str(line);
        asm_output.push('\n');
    }
    asm_output
}

#[cfg(test)]
mod tests {
    use super::compile;

    fn compile_ok(src: &str) -> String {
        compile(&src.to_string())
    }

    #[test]
    fn compile_simple_return() {
        let asm = compile_ok("int main(){return 42;}");
        assert!(asm.contains("call main"));
        assert!(asm.contains("main:"));
        assert!(asm.contains("movi r0, 42"));
    }

    #[test]
    fn compile_if_else() {
        let asm = compile_ok("int main(){int a=1; if(a){return 2;} else {return 3;}}");
        assert!(asm.contains("Lelse_"));
        assert!(asm.contains("Lend_"));
    }

    #[test]
    fn compile_while() {
        let asm = compile_ok("int main(){int i=0; while(i<3){i=i+1;} return i;}");
        assert!(asm.contains("Lbegin_"));
        assert!(asm.contains("Lend_"));
    }

    #[test]
    fn compile_decls_and_assign() {
        let _asm = compile_ok("int main(){int a=1,b; b=a+2; return b;}");
    }

    #[test]
    fn compile_arrays_and_index() {
        let _asm = compile_ok("int main(){int a[3]; a[1]=5; return a[1];}");
    }

    #[test]
    fn compile_pointer_arith() {
        let _asm = compile_ok("int main(){int a[2]; int *p=a; return p[1];}");
    }

    #[test]
    fn compile_function_call() {
        let asm = compile_ok("int add(int a,int b){return a+b;} int main(){return add(1,2);}"); 
        assert!(asm.contains("add:"));
        assert!(asm.contains("call add"));
    }

    #[test]
    #[should_panic]
    fn reject_char_type() {
        compile_ok("int main(){char c; return 0;}");
    }

    #[test]
    #[should_panic]
    fn reject_for_loop() {
        compile_ok("int main(){for(;;){} return 0;}");
    }
}
