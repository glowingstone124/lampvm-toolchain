use std::collections::HashMap;
use crate::compiler::compiler::{
    assign_offsets, BinOp, ConstValue, Expr, ExprKind, ForInit, FuncBuilder, Function, Global,
    Program, Stmt, StructDef, StructField, SwitchItem, Type, UnOp,
};
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
    typedefs: HashMap<String, Type>,
    struct_defs: HashMap<String, StructDef>,
    anon_struct_id: usize,
}
impl Parser {
    pub(crate) fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens,
            idx: 0,
            func_builder: None,
            func_types: HashMap::new(),
            global_types: HashMap::new(),
            typedefs: HashMap::new(),
            struct_defs: HashMap::new(),
            anon_struct_id: 0,
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
                if k == "struct" {
                    self.next();
                    return self.parse_struct_spec();
                }
                if let Some(ty) = type_from_keyword(k) {
                    self.next();
                    return ty;
                }
                panic!("Expected type specifier");
            }
            TokenKind::Ident(name) => {
                if let Some(ty) = self.typedefs.get(name).cloned() {
                    self.next();
                    return ty;
                }
                panic!("Unknown type name: {}", name);
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
            let n = if self.consume_punct("]") {
                0usize
            } else {
               let expr = self.parse_expr();
                self.expect_punct("]");
                let size = eval_const_expr_int(&expr)
                    .unwrap_or_else(|| {
                        panic!("Array size must be a constant int");
                    });
                if size < 0 {
                    panic!("Array size must be non-negative");
                }
                size as usize
            };
            ty = Type::Array(Box::new(ty), n);
        }
        (ty, name)
    }

    pub(crate) fn parse_program(&mut self) -> Program {
        let mut funcs = Vec::new();
        let mut globals = Vec::new();
        while !self.at_eof() {
            if self.consume_keyword("typedef") {
                self.parse_typedef_decl();
                continue;
            }
            let base_type = self.parse_type_spec();
            let (decl_type, name) = self.parse_declarator(base_type.clone());
            if self.peek_punct("(") {
                let func = self.parse_function_after_decl(decl_type, name);
                self.func_types.insert(func.name.clone(), func.ret_type.clone());
                funcs.push(func);
            } else {
                let mut decls = Vec::new();
                let init = if self.consume_punct("=") {
                    Some(self.parse_global_init(&decl_type))
                } else {
                    None
                };
                decls.push((decl_type, name, init));
                while self.consume_punct(",") {
                    let (ty, name) = self.parse_declarator(base_type.clone());
                    let init = if self.consume_punct("=") {
                        Some(self.parse_global_init(&ty))
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
                    let ty = self.fix_unsized_array_global(ty, init.as_ref());
                    if ty.is_array() && init.is_some() {
                        if !matches!(init, Some(ConstValue::Str(_)) | Some(ConstValue::Array(_))) {
                            panic!("Array global initializers must be string literals or initializer lists: {}", name);
                        }
                    }
                    if self.global_types.insert(name.clone(), ty.clone()).is_some() {
                        panic!("Duplicate global definition: {}", name);
                    }
                    globals.push(Global { name, ty, init });
                }
            }
        }
        Program {
            funcs,
            globals,
            struct_defs: self.struct_defs.clone(),
        }
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

        assign_offsets(&mut locals, &builder.params, &self.struct_defs);

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
        if self.consume_keyword("typedef") {
            self.parse_typedef_decl();
            return Stmt::ExprStmt(None);
        }
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
        if self.consume_keyword("for") {
            return self.parse_for_stmt();
        }
        if self.consume_keyword("switch") {
            return self.parse_switch_stmt();
        }
        if self.consume_keyword("break") {
            self.expect_punct(";");
            return Stmt::Break;
        }
        if self.consume_keyword("continue") {
            self.expect_punct(";");
            return Stmt::Continue;
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
        match &self.peek().kind {
            TokenKind::Keyword(k) => is_decl_type_keyword(k) || k == "struct",
            TokenKind::Ident(name) => self.typedefs.contains_key(name),
            _ => false,
        }
    }

    fn parse_decl_stmt(&mut self) -> Stmt {
        let base = self.parse_type_spec();
        let mut decls = Vec::new();
        loop {
            let (ty, name) = self.parse_declarator(base.clone());
            let mut ty = ty;
            let id = self
                .func_builder
                .as_mut()
                .unwrap()
                .add_var(name, ty.clone());
            let init = if self.consume_punct("=") {
                Some(self.parse_expr())
            } else {
                None
            };
            ty = self.fix_unsized_array_local(ty, init.as_ref());
            self.func_builder.as_mut().unwrap().locals[id].ty = ty.clone();
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
        let mut node = self.parse_conditional();
        if self.consume_punct("=") {
            let rhs = self.parse_assign();
            node = Expr {
                kind: ExprKind::Assign(Box::new(node), Box::new(rhs)),
            };
        }
        node
    }

    fn parse_conditional(&mut self) -> Expr {
        let mut node = self.parse_logical_or();
        if self.consume_punct("?") {
            let then_expr = self.parse_assign();
            self.expect_punct(":");
            let else_expr = self.parse_conditional();
            node = Expr {
                kind: ExprKind::Conditional {
                    cond: Box::new(node),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                },
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
        if self.consume_punct("++") {
            let expr = self.parse_unary();
            return Expr {
                kind: ExprKind::Inc {
                    expr: Box::new(expr),
                    is_post: false,
                },
            };
        }
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
            if self.consume_punct(".") {
                let field = self.expect_ident();
                node = Expr {
                    kind: ExprKind::Member {
                        base: Box::new(node),
                        field,
                        is_arrow: false,
                    },
                };
                continue;
            }
            if self.consume_punct("->") {
                let field = self.expect_ident();
                node = Expr {
                    kind: ExprKind::Member {
                        base: Box::new(node),
                        field,
                        is_arrow: true,
                    },
                };
                continue;
            }
            if self.consume_punct("++") {
                node = Expr {
                    kind: ExprKind::Inc {
                        expr: Box::new(node),
                        is_post: true,
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
        if let TokenKind::Str(s) = &self.peek().kind {
            let s = s.clone();
            self.next();
            return Expr {
                kind: ExprKind::Str(s),
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

    fn parse_global_init(&mut self, ty: &Type) -> ConstValue {
        if self.peek_punct("{") {
            return self.parse_init_list(ty);
        }
        let expr = self.parse_expr();
        eval_const_expr(&expr).unwrap_or_else(|| panic!("Global initializer must be constant"))
    }

    fn parse_init_list(&mut self, ty: &Type) -> ConstValue {
        match ty {
            Type::Array(elem, _) => {
                self.expect_punct("{");
                let mut values = Vec::new();
                if !self.consume_punct("}") {
                    loop {
                        let v = if self.peek_punct("{") {
                            self.parse_init_list(elem)
                        } else {
                            self.parse_global_init(elem)
                        };
                        values.push(v);
                        if self.consume_punct("}") {
                            break;
                        }
                        self.expect_punct(",");
                    }
                }
                ConstValue::Array(values)
            }
            _ => panic!("Initializer list only allowed for array types"),
        }
    }

    fn parse_for_stmt(&mut self) -> Stmt {
        self.expect_punct("(");
        let init = if self.consume_punct(";") {
            None
        } else if self.peek_is_type() {
            let decls = self.parse_decl_in_for();
            Some(ForInit::Decl(decls))
        } else {
            let expr = self.parse_expr();
            self.expect_punct(";");
            Some(ForInit::Expr(expr))
        };
        let cond = if self.consume_punct(";") {
            None
        } else {
            let expr = self.parse_expr();
            self.expect_punct(";");
            Some(expr)
        };
        let step = if self.consume_punct(")") {
            None
        } else {
            let expr = self.parse_expr();
            self.expect_punct(")");
            Some(expr)
        };
        let body = Box::new(self.parse_stmt());
        Stmt::For {
            init,
            cond,
            step,
            body,
        }
    }

    fn parse_decl_in_for(&mut self) -> Vec<(usize, Option<Expr>)> {
        let base = self.parse_type_spec();
        let mut decls = Vec::new();
        loop {
            let (ty, name) = self.parse_declarator(base.clone());
            let mut ty = ty;
            let id = self
                .func_builder
                .as_mut()
                .unwrap()
                .add_var(name, ty.clone());
            let init = if self.consume_punct("=") {
                Some(self.parse_expr())
            } else {
                None
            };
            ty = self.fix_unsized_array_local(ty, init.as_ref());
            self.func_builder.as_mut().unwrap().locals[id].ty = ty.clone();
            decls.push((id, init));
            if self.consume_punct(";") {
                break;
            }
            self.expect_punct(",");
        }
        decls
    }

    fn parse_switch_stmt(&mut self) -> Stmt {
        self.expect_punct("(");
        let expr = self.parse_expr();
        self.expect_punct(")");
        self.expect_punct("{");
        let mut items = Vec::new();
        while !self.consume_punct("}") {
            if self.consume_keyword("case") {
                let expr = self.parse_expr();
                let value = eval_const_expr_int(&expr)
                    .unwrap_or_else(|| panic!("case label must be constant int"));
                self.expect_punct(":");
                items.push(SwitchItem::Case(value));
                continue;
            }
            if self.consume_keyword("default") {
                self.expect_punct(":");
                items.push(SwitchItem::Default);
                continue;
            }
            let stmt = self.parse_stmt();
            items.push(SwitchItem::Stmt(stmt));
        }
        Stmt::Switch { expr, items }
    }

    fn parse_struct_spec(&mut self) -> Type {
        let mut name_opt = None;
        if let TokenKind::Ident(name) = &self.peek().kind {
            name_opt = Some(name.clone());
            self.next();
        }

        if !self.peek_punct("{") {
            let name = name_opt.unwrap_or_else(|| panic!("struct name required"));
            return Type::Struct(name);
        }

        self.expect_punct("{");
        let mut fields = Vec::new();
        let mut offset = 0usize;
        let mut struct_align = 1usize;
        while !self.consume_punct("}") {
            let base = self.parse_type_spec();
            loop {
                let (ty, name) = self.parse_declarator(base.clone());
                let align = ty.align_with(&self.struct_defs);
                offset = align_up_usize(offset, align);
                fields.push(StructField {
                    name,
                    ty: ty.clone(),
                    offset,
                });
                offset += ty.size_with(&self.struct_defs);
                struct_align = struct_align.max(align);
                if self.consume_punct(";") {
                    break;
                }
                self.expect_punct(",");
            }
        }
        let size = align_up_usize(offset, struct_align);
        let name = name_opt.unwrap_or_else(|| {
            let n = format!("_anon_struct_{}", self.anon_struct_id);
            self.anon_struct_id += 1;
            n
        });
        if self.struct_defs.contains_key(&name) {
            panic!("Duplicate struct definition: {}", name);
        }
        self.struct_defs.insert(
            name.clone(),
            StructDef {
                name: name.clone(),
                fields,
                size,
                align: struct_align,
            },
        );
        Type::Struct(name)
    }

    fn parse_typedef_decl(&mut self) {
        let mut base = self.parse_type_spec();
        let mut first_typedef_name: Option<String> = None;
        loop {
            let (ty, name) = self.parse_declarator(base.clone());
            let mut ty = ty;
            if let Type::Struct(struct_name) = &ty {
                if struct_name.starts_with("_anon_struct_") {
                    if first_typedef_name.is_none() {
                        let new_name = name.clone();
                        if let Some(def) = self.struct_defs.remove(struct_name) {
                            let mut def = def;
                            def.name = new_name.clone();
                            self.struct_defs.insert(new_name.clone(), def);
                        }
                        ty = Type::Struct(new_name.clone());
                        first_typedef_name = Some(new_name);
                        base = ty.clone();
                    } else if let Some(n) = &first_typedef_name {
                        ty = Type::Struct(n.clone());
                    }
                }
            }
            self.typedefs.insert(name, ty);
            if self.consume_punct(";") {
                break;
            }
            self.expect_punct(",");
        }
    }

    fn fix_unsized_array_global(&self, ty: Type, init: Option<&ConstValue>) -> Type {
        match ty {
            Type::Array(elem, 0) => {
                if let Some(ConstValue::Str(s)) = init {
                    let n = s.as_bytes().len() + 1;
                    Type::Array(elem, n)
                } else if let Some(ConstValue::Array(vals)) = init {
                    let n = vals.len();
                    Type::Array(elem, n)
                } else {
                    panic!("Unsized array requires string literal initializer");
                }
            }
            _ => ty,
        }
    }

    fn fix_unsized_array_local(&self, ty: Type, init: Option<&Expr>) -> Type {
        match ty {
            Type::Array(elem, 0) => {
                if let Some(Expr { kind: ExprKind::Str(s) }) = init {
                    let n = s.as_bytes().len() + 1;
                    Type::Array(elem, n)
                } else {
                    panic!("Unsized array requires string literal initializer");
                }
            }
            _ => ty,
        }
    }
}

fn eval_const_expr(expr: &Expr) -> Option<ConstValue> {
    match &expr.kind {
        ExprKind::Num(n) => Some(ConstValue::Int(*n)),
        ExprKind::Float(f) => Some(ConstValue::Float(*f)),
        ExprKind::Str(s) => Some(ConstValue::Str(s.clone())),
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
                _ => None,
            }
        }
        ExprKind::Cast { ty, expr } => {
            let v = eval_const_expr(expr)?;
            match (ty, v) {
                (Type::Float, ConstValue::Int(i)) => Some(ConstValue::Float(i as f32)),
                (Type::Float, ConstValue::Float(f)) => Some(ConstValue::Float(f)),
                (_, ConstValue::Float(f)) => Some(ConstValue::Int(f as i64)),
                (_, ConstValue::Int(i)) => Some(ConstValue::Int(i)),
                (_, ConstValue::Str(s)) => Some(ConstValue::Str(s)),
                (_, ConstValue::Array(_)) => None,
            }
        }
        _ => None,
    }
}

fn eval_const_expr_int(expr: &Expr) -> Option<i64> {
    match eval_const_expr(expr)? {
        ConstValue::Int(v) => Some(v),
        _ => None,
    }
}

fn align_up_usize(v: usize, align: usize) -> usize {
    if align <= 1 {
        v
    } else {
        ((v + align - 1) / align) * align
    }
}
