use std::collections::HashMap;
use crate::assemble::config::ArchConfig;
use crate::compiler::tokenizer::{TokenKind, Tokenizer};
use crate::compiler::parser::{Parser};
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Char,
    Int,
    Long,
    Float,
    Void,
    Ptr(Box<Type>),
    Array(Box<Type>, usize),
}

impl Type {
    fn size(&self) -> usize {
        match self {
            Type::Int => 4,
            Type::Long => 4,
            Type::Float => 4,
            Type::Char => 1,
            Type::Void => 0,
            Type::Ptr(_) => 4,
            Type::Array(elem, n) => elem.size() * *n,
        }
    }

    pub(crate) fn is_array(&self) -> bool {
        matches!(self, Type::Array(_, _))
    }

    pub(crate) fn decay(&self) -> Type {
        match self {
            Type::Array(elem, _) => Type::Ptr(elem.clone()),
            _ => self.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Var {
    name: String,
    ty: Type,
    offset: i32,
}

#[derive(Debug, Clone)]
pub enum ConstValue {
    Int(i64),
    Float(f32),
}

#[derive(Debug, Clone)]
pub struct Global {
    pub(crate) name: String,
    pub(crate) ty: Type,
    pub(crate) init: Option<ConstValue>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub(crate) name: String,
    pub(crate) params: Vec<usize>,
    pub(crate) locals: Vec<Var>,
    pub(crate) body: Stmt,
    pub(crate) ret_type: Type,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub(crate) funcs: Vec<Function>,
    pub(crate) globals: Vec<Global>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
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
    InlineAsm(String),
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub(crate) kind: ExprKind,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Num(i64),
    Float(f32),
    Str(String),
    Var(usize),
    GlobalVar(String),
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
    Cast {
        ty: Type,
        expr: Box<Expr>,
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
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
pub enum UnOp {
    Pos,
    Neg,
    Addr,
    Deref,
    Not,
}


pub struct FuncBuilder {
    pub(crate) locals: Vec<Var>,
    scopes: Vec<HashMap<String, usize>>,
    pub(crate) params: Vec<usize>,
}

impl FuncBuilder {
    pub(crate) fn new() -> Self {
        FuncBuilder {
            locals: Vec::new(),
            scopes: vec![HashMap::new()],
            params: Vec::new(),
        }
    }

    pub(crate) fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub(crate) fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    pub(crate) fn add_var(&mut self, name: String, ty: Type) -> usize {
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

    pub(crate) fn find_var(&self, name: &str) -> Option<usize> {
        for scope in self.scopes.iter().rev() {
            if let Some(id) = scope.get(name) {
                return Some(*id);
            }
        }
        None
    }
}

pub fn assign_offsets(locals: &mut [Var], params: &[usize]) {
    let mut offset = 4i32;
    for &id in params {
        let size = locals[id].ty.size() as i32;
        let align = if size >= 4 { 4 } else { 1 };
        offset = align_up(offset, align);
        locals[id].offset = offset;
        offset += size;
    }
    for (i, var) in locals.iter_mut().enumerate() {
        if params.contains(&i) {
            continue;
        }
        let size = var.ty.size() as i32;
        let align = if size >= 4 { 4 } else { 1 };
        offset = align_up(offset, align);
        var.offset = offset;
        offset += size;
    }
}

fn align_up(value: i32, align: i32) -> i32 {
    if align <= 1 {
        value
    } else {
        ((value + align - 1) / align) * align
    }
}

fn align_up_u32(value: u32, align: u32) -> u32 {
    if align <= 1 {
        value
    } else {
        ((value + align - 1) / align) * align
    }
}

struct Codegen<'a> {
    out: Vec<String>,
    label: usize,
    func_types: &'a HashMap<String, Type>,
    global_types: &'a HashMap<String, Type>,
    globals: &'a [Global],
}

impl<'a> Codegen<'a> {
    fn new(
        func_types: &'a HashMap<String, Type>,
        global_types: &'a HashMap<String, Type>,
        globals: &'a [Global],
    ) -> Self {
        Codegen {
            out: Vec::new(),
            label: 0,
            func_types,
            global_types,
            globals,
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

    fn is_float_type(ty: &Type) -> bool {
        matches!(ty, Type::Float)
    }

    fn is_char_type(ty: &Type) -> bool {
        matches!(ty, Type::Char)
    }

    fn emit_load_var(&mut self, ty: &Type, offset: i32) {
        if Self::is_char_type(ty) {
            self.emit(format!("load r0, [r30 + {}]", offset));
        } else {
            self.emit(format!("load32 r0, [r30 + {}]", offset));
        }
    }

    fn emit_store_var(&mut self, ty: &Type, offset: i32) {
        if Self::is_char_type(ty) {
            self.emit(format!("store [r30 + {}], r0", offset));
        } else {
            self.emit(format!("store32 [r30 + {}], r0", offset));
        }
    }

    fn emit_load_from_addr(&mut self, ty: &Type, addr_reg: &str) {
        if Self::is_char_type(ty) {
            self.emit(format!("load r0, [{}]", addr_reg));
        } else {
            self.emit(format!("load32 r0, [{}]", addr_reg));
        }
    }

    fn emit_store_to_addr(&mut self, ty: &Type, addr_reg: &str) {
        if Self::is_char_type(ty) {
            self.emit(format!("store [{}], r0", addr_reg));
        } else {
            self.emit(format!("store32 [{}], r0", addr_reg));
        }
    }

    fn emit_cmp_zero_for_type(&mut self, ty: &Type) {
        if Self::is_float_type(ty) {
            self.emit("movi r1, 0");
            self.emit("fcmp r0, r1");
        } else {
            self.emit("cmpi r0, 0");
        }
    }

    fn cast_reg(&mut self, from: &Type, to: &Type) {
        if from == to {
            return;
        }
        match (from, to) {
            (Type::Float, Type::Float) => {}
            (Type::Ptr(_), Type::Float) | (Type::Float, Type::Ptr(_)) => {
                panic!("Pointer/float cast not supported");
            }
            (Type::Float, _) => {
                self.emit("ftoi r0, r0");
                if matches!(to, Type::Char) {
                    self.emit("movi r2, 255");
                    self.emit("and r0, r0, r2");
                }
            }
            (_, Type::Float) => {
                self.emit("itof r0, r0");
            }
            (_, Type::Char) => {
                self.emit("movi r2, 255");
                self.emit("and r0, r0, r2");
            }
            _ => {}
        }
    }
    fn try_gen_builtin_call(&mut self, name: &str, args: &[Expr], func: &Function) -> bool {
        match name {
            "__setregister" => {
                self.gen_builtin_setregister(args, func);
                return true;
            }
            _ => false
        }
    }
    fn gen_builtin_setregister(&mut self, args: &[Expr], func: &Function) {
        if args.len() != 2 {
            panic!("__setregister expects 2 arguments: (__setregister(regId, data))");
        }

        let reg_id = match args[0].kind {
            ExprKind::Num(n) => n,
            _ => panic!("__setregister: registerId must be a constant number (e.g. __setregister(1, x))"),
        };

        if reg_id < 0 || reg_id > 31 {
            panic!(
                "__setregister: registerId out of range, expected {} to {} but found {}",
                0, 31, reg_id
            );
        }

        let rid = reg_id as i32;

        if rid == 30 {
            panic!("__setregister: r30 is reserved");
        }

        /*
        We do a judgment here. If we have an immediate number, then the fastest way is to call MOVI.
        On the other side, when we get an expression, we will calculate it, save its value to r0, and do a mov.
        */
        match &args[1].kind {
            ExprKind::Num(n) => {
                self.emit(format!("movi r{}, {}", rid, *n as i32));
            }
            ExprKind::Float(f) => {
                let bits = f.to_bits();
                self.emit(format!("movi r{}, {}", rid, bits));
            }
            _ => {
                self.gen_expr(&args[1], func);
                if rid != 0 {
                    self.emit(format!("mov r{}, r0", rid));
                }
            }
        }
    }
    fn gen_program(&mut self, prog: &Program) {
        self.emit(".text");
        self.emit("movi r30, MEM_SIZE");
        self.emit("call main");
        self.emit("halt");
        for func in &prog.funcs {
            self.gen_function(func);
        }
        self.gen_globals();
    }

    fn gen_globals(&mut self) {
        if self.globals.is_empty() {
            return;
        }

        let mut data_lines: Vec<String> = Vec::new();
        let mut bss_lines: Vec<String> = Vec::new();
        let mut data_offset: u32 = 0;
        let mut bss_offset: u32 = 0;

        for g in self.globals {
            let size = g.ty.size() as u32;
            let align = if size >= 4 { 4 } else { 1 };
            if let Some(init) = &g.init {
                let aligned = align_up_u32(data_offset, align);
                if aligned > data_offset {
                    data_lines.push(format!(".zero {}", aligned - data_offset));
                    data_offset = aligned;
                }
                data_lines.push(format!("{}:", g.name));
                match (&g.ty, init) {
                    (Type::Char, ConstValue::Int(v)) => {
                        data_lines.push(format!(".byte {}", (*v as i32) & 0xFF));
                    }
                    (Type::Int | Type::Long | Type::Ptr(_), ConstValue::Int(v)) => {
                        data_lines.push(format!(".long {}", *v as i32));
                    }
                    (Type::Float, ConstValue::Float(f)) => {
                        data_lines.push(format!(".long {}", f.to_bits()));
                    }
                    (Type::Float, ConstValue::Int(v)) => {
                        let f = *v as f32;
                        data_lines.push(format!(".long {}", f.to_bits()));
                    }
                    _ => panic!("Unsupported global initializer for {}", g.name),
                }
                data_offset += size;
            } else {
                let aligned = align_up_u32(bss_offset, align);
                if aligned > bss_offset {
                    bss_lines.push(format!(".zero {}", aligned - bss_offset));
                    bss_offset = aligned;
                }
                bss_lines.push(format!("{}:", g.name));
                bss_lines.push(format!(".zero {}", size));
                bss_offset += size;
            }
        }

        if !data_lines.is_empty() {
            self.emit(".data");
            for line in data_lines {
                self.emit(line);
            }
        }
        if !bss_lines.is_empty() {
            self.emit(".bss");
            for line in bss_lines {
                self.emit(line);
            }
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
            let ty = &func.locals[param_id].ty;
            if Self::is_char_type(ty) {
                self.emit(format!("store [r30 + {}], r{}", offset, i));
            } else {
                self.emit(format!("store32 [r30 + {}], r{}", offset, i));
            }
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
        align_up(max, 4)
    }

    fn gen_stmt(&mut self, stmt: &Stmt, func: &Function, ret_label: &str) {
        match stmt {
            Stmt::InlineAsm(s) => {
                for line in s.lines() {
                    let t = line.trim();
                    if t.is_empty() {continue;}
                    self.emit(t.to_string());
                }
            }
            Stmt::Return(expr) => {
                if let Some(e) = expr {
                    self.gen_expr(e, func);
                    let rty = self.expr_type(e, func);
                    self.cast_reg(&rty, &func.ret_type);
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
                let cty = self.expr_type(cond, func);
                self.emit_cmp_zero_for_type(&cty);
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
                let cty = self.expr_type(cond, func);
                self.emit_cmp_zero_for_type(&cty);
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
            Stmt::ExprStmt(Some(e)) => {
                if let ExprKind::Call { name, args } = &e.kind {
                    if self.try_gen_builtin_call(name, args, func) {
                        return;
                    }
                }
                self.gen_expr(e, func);
            }

            Stmt::Decl(vars) => {
                for (id, init) in vars {
                    if let Some(expr) = init {
                        self.gen_expr(expr, func);
                        let rty = self.expr_type(expr, func);
                        let lty = &func.locals[*id].ty;
                        self.cast_reg(&rty, lty);
                        let offset = func.locals[*id].offset;
                        self.emit_store_var(lty, offset);
                    }
                }
            },
            Stmt::ExprStmt(None) => {
            }
        }
    }

    fn gen_expr(&mut self, expr: &Expr, func: &Function) {
        match &expr.kind {
            ExprKind::Num(n) => {
                self.emit(format!("movi r0, {}", *n as i32));
            }
            ExprKind::Float(f) => {
                let bits = f.to_bits();
                self.emit(format!("movi r0, {}", bits));
            }
            ExprKind::Str(_) => {
                panic!("String literals are not supported in expressions");
            }
            ExprKind::Var(id) => {
                let var = &func.locals[*id];
                if var.ty.is_array() {
                    self.gen_lvalue(expr, func);
                } else {
                    self.emit_load_var(&var.ty, var.offset);
                }
            }
            ExprKind::GlobalVar(name) => {
                let gty = self
                    .global_types
                    .get(name)
                    .unwrap_or_else(|| panic!("Undefined global: {}", name));
                if gty.is_array() {
                    self.emit(format!("movi r0, {}", name));
                } else {
                    self.emit(format!("movi r1, {}", name));
                    self.emit_load_from_addr(gty, "r1");
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
                    let rty = self.expr_type(rhs, func);
                    self.cast_reg(&rty, &lty);
                }
                if let Some(lty) = self.lvalue_type(lhs, func) {
                    self.emit_store_to_addr(&lty, "r1");
                } else {
                    self.emit("store32 [r1], r0");
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                self.gen_binary(*op, lhs, rhs, func);
            }
            ExprKind::Unary { op, expr } => {
                self.gen_unary(*op, expr, func);
            }
            ExprKind::Call { name, args } => {

                if self.try_gen_builtin_call(name, args, func) {
                    self.emit("movi r0, 0");
                    return;
                }
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
                let rty = self.expr_type(expr, func);
                self.emit_load_from_addr(&rty, "r0");
            }
            ExprKind::Cast { ty, expr: inner } => {
                let from = self.expr_type(inner, func);
                self.gen_expr(inner, func);
                self.cast_reg(&from, ty);
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
                let ty = self.expr_type(expr, func);
                if Self::is_float_type(&ty) {
                    self.emit("fneg r0, r0");
                } else {
                    self.emit("movi r1, 0");
                    self.emit("sub r0, r1, r0");
                }
            }
            UnOp::Addr => {
                self.gen_lvalue(expr, func);
            }
            UnOp::Deref => {
                self.gen_expr(expr, func);
                let ty = self.expr_type(expr, func);
                if let Type::Ptr(elem) = ty {
                    self.emit_load_from_addr(&elem, "r0");
                } else {
                    panic!("Cannot dereference non-pointer");
                }
            }
            UnOp::Not => {
                self.gen_expr(expr, func);
                let t = self.new_label("Lnot");
                let end = self.new_label("Lend");
                let ty = self.expr_type(expr, func);
                self.emit_cmp_zero_for_type(&ty);
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
                let lty = self.expr_type(lhs, func);
                self.emit_cmp_zero_for_type(&lty);
                self.emit(format!("jz {}", false_l));
                self.gen_expr(rhs, func);
                let rty = self.expr_type(rhs, func);
                self.emit_cmp_zero_for_type(&rty);
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
                let lty = self.expr_type(lhs, func);
                self.emit_cmp_zero_for_type(&lty);
                self.emit(format!("jnz {}", true_l));
                self.gen_expr(rhs, func);
                let rty = self.expr_type(rhs, func);
                self.emit_cmp_zero_for_type(&rty);
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
                let lt = self.expr_type(lhs, func);
                let rt = self.expr_type(rhs, func);
                self.gen_expr(lhs, func);
                self.emit("push r0");
                self.gen_expr(rhs, func);
                self.emit("pop r1");
                if Self::is_float_type(&lt) || Self::is_float_type(&rt) {
                    if matches!(lt, Type::Ptr(_)) || matches!(rt, Type::Ptr(_)) {
                        panic!("Pointer and float comparison not supported");
                    }
                    if !Self::is_float_type(&lt) {
                        self.emit("itof r1, r1");
                    }
                    if !Self::is_float_type(&rt) {
                        self.emit("itof r0, r0");
                    }
                    self.emit("fcmp r1, r0");
                    let t = self.new_label("Lfcmp_true");
                    let end = self.new_label("Lfcmp_end");
                    match op {
                        BinOp::Eq => {
                            self.emit(format!("jz {}", t));
                        }
                        BinOp::Ne => {
                            self.emit(format!("jnz {}", t));
                        }
                        BinOp::Gt => {
                            self.emit(format!("jc {}", t));
                        }
                        BinOp::Ge => {
                            let z = self.new_label("Lfcmp_ge");
                            self.emit(format!("jz {}", z));
                            self.emit(format!("jc {}", t));
                            self.emit(format!("jmp {}", end));
                            self.emit(format!("{}:", z));
                            self.emit(format!("jmp {}", t));
                        }
                        BinOp::Lt => {
                            let z = self.new_label("Lfcmp_lt_eq");
                            self.emit(format!("jz {}", z));
                            self.emit(format!("jle {}", t));
                            self.emit(format!("jmp {}", end));
                            self.emit(format!("{}:", z));
                        }
                        BinOp::Le => {
                            self.emit(format!("jle {}", t));
                        }
                        _ => unreachable!(),
                    }
                    self.emit("movi r0, 0");
                    self.emit(format!("jmp {}", end));
                    self.emit(format!("{}:", t));
                    self.emit("movi r0, 1");
                    self.emit(format!("{}:", end));
                } else {
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
            }
            BinOp::Add | BinOp::Sub => {
                let lt = self.expr_type(lhs, func);
                let rt = self.expr_type(rhs, func);
                if Self::is_float_type(&lt) || Self::is_float_type(&rt) {
                    if matches!(lt, Type::Ptr(_)) || matches!(rt, Type::Ptr(_)) {
                        panic!("Pointer and float arithmetic not supported");
                    }
                    self.gen_expr(lhs, func);
                    if !Self::is_float_type(&lt) {
                        self.emit("itof r0, r0");
                    }
                    self.emit("push r0");
                    self.gen_expr(rhs, func);
                    if !Self::is_float_type(&rt) {
                        self.emit("itof r0, r0");
                    }
                    self.emit("pop r1");
                    let inst = if matches!(op, BinOp::Add) { "fadd" } else { "fsub" };
                    self.emit(format!("{} r0, r1, r0", inst));
                    return;
                }
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
                let lt = self.expr_type(lhs, func);
                let rt = self.expr_type(rhs, func);
                if Self::is_float_type(&lt) || Self::is_float_type(&rt) {
                    if matches!(lt, Type::Ptr(_)) || matches!(rt, Type::Ptr(_)) {
                        panic!("Pointer and float arithmetic not supported");
                    }
                    self.gen_expr(lhs, func);
                    if !Self::is_float_type(&lt) {
                        self.emit("itof r0, r0");
                    }
                    self.emit("push r0");
                    self.gen_expr(rhs, func);
                    if !Self::is_float_type(&rt) {
                        self.emit("itof r0, r0");
                    }
                    self.emit("pop r1");
                    let inst = if matches!(op, BinOp::Mul) { "fmul" } else { "fdiv" };
                    self.emit(format!("{} r0, r1, r0", inst));
                } else {
                    self.gen_expr(lhs, func);
                    self.emit("push r0");
                    self.gen_expr(rhs, func);
                    self.emit("pop r1");
                    let inst = if matches!(op, BinOp::Mul) { "mul" } else { "div" };
                    self.emit(format!("{} r0, r1, r0", inst));
                }
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
            ExprKind::GlobalVar(name) => {
                self.emit(format!("movi r0, {}", name));
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
            ExprKind::GlobalVar(name) => {
                let gty = self
                    .global_types
                    .get(name)
                    .unwrap_or_else(|| panic!("Undefined global: {}", name));
                if gty.is_array() {
                    self.emit(format!("movi r0, {}", name));
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
            ExprKind::Float(_) => Type::Float,
            ExprKind::Str(_) => panic!("String literals are not supported in expressions"),
            ExprKind::Var(id) => func.locals[*id].ty.decay(),
            ExprKind::GlobalVar(name) => self
                .global_types
                .get(name)
                .unwrap_or_else(|| panic!("Undefined global: {}", name))
                .decay(),
            ExprKind::Assign(lhs, _) => self.expr_type(lhs, func),
            ExprKind::Binary { op, lhs, rhs } => match op {
                BinOp::Add | BinOp::Sub => {
                    let lt = self.expr_type(lhs, func);
                    let rt = self.expr_type(rhs, func);
                    match (lt, rt) {
                        (Type::Float, _) | (_, Type::Float) => Type::Float,
                        (Type::Ptr(elem), Type::Int) => Type::Ptr(elem),
                        (Type::Int, Type::Ptr(elem)) if matches!(op, BinOp::Add) => Type::Ptr(elem),
                        (Type::Ptr(_), Type::Ptr(_)) => Type::Int,
                        _ => Type::Int,
                    }
                }
                BinOp::Mul | BinOp::Div => {
                    let lt = self.expr_type(lhs, func);
                    let rt = self.expr_type(rhs, func);
                    if matches!(lt, Type::Float) || matches!(rt, Type::Float) {
                        Type::Float
                    } else {
                        Type::Int
                    }
                }
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
                UnOp::Neg | UnOp::Pos => {
                    let et = self.expr_type(expr, func);
                    if matches!(et, Type::Float) {
                        Type::Float
                    } else {
                        Type::Int
                    }
                }
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
            ExprKind::Cast {
                ty, ..
            } => ty.clone(),
        }
    }

    fn lvalue_type(&self, expr: &Expr, func: &Function) -> Option<Type> {
        match &expr.kind {
            ExprKind::Var(id) => Some(func.locals[*id].ty.clone()),
            ExprKind::GlobalVar(name) => self
                .global_types
                .get(name)
                .cloned(),
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

pub fn compile(input: &String, _arch: &ArchConfig) -> String {
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

    let mut codegen = Codegen::new(&parser.func_types, &parser.global_types, &prog.globals);
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
    use crate::assemble::config::ArchConfig;

    fn compile_ok(src: &str) -> String {
        compile(&src.to_string(), &ArchConfig::default())
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
    fn compile_other_scalar_types() {
        let _asm = compile_ok("int main(){char c; long l; float f; c=1; l=2; f=3; return l;}");
    }

    #[test]
    fn compile_char_byte_semantics() {
        let asm = compile_ok("int main(){char c; c=65; return c;}");
        assert!(asm.contains("store ["));
        assert!(asm.contains("load r0, [r30"));
    }

    #[test]
    fn compile_float_ops() {
        let asm = compile_ok("float add(float a,float b){return a+b;} int main(){float x=1.5; float y=2.0; float z=add(x,y); return z>0.0;}");
        assert!(asm.contains("fadd"));
        assert!(asm.contains("fcmp"));
    }

    #[test]
    #[should_panic]
    fn reject_for_loop() {
        compile_ok("int main(){for(;;){} return 0;}");
    }
}
