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
    Struct(String),
}

impl Type {
    pub(crate) fn size_with(&self, structs: &HashMap<String, StructDef>) -> usize {
        match self {
            Type::Int => 4,
            Type::Long => 4,
            Type::Float => 4,
            Type::Char => 1,
            Type::Void => 0,
            Type::Ptr(_) => 4,
            Type::Array(elem, n) => elem.size_with(structs) * *n,
            Type::Struct(name) => structs
                .get(name)
                .unwrap_or_else(|| panic!("Unknown struct type: {}", name))
                .size,
        }
    }

    pub(crate) fn is_array(&self) -> bool {
        matches!(self, Type::Array(_, _))
    }

    pub(crate) fn align_with(&self, structs: &HashMap<String, StructDef>) -> usize {
        match self {
            Type::Char => 1,
            Type::Int | Type::Long | Type::Float | Type::Ptr(_) => 4,
            Type::Array(elem, _) => elem.align_with(structs),
            Type::Struct(name) => structs
                .get(name)
                .unwrap_or_else(|| panic!("Unknown struct type: {}", name))
                .align,
            Type::Void => 1,
        }
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
    pub(crate) name: String,
    pub(crate) ty: Type,
    pub(crate) offset: i32,
}

#[derive(Debug, Clone)]
pub enum ConstValue {
    Int(i64),
    Float(f32),
    Str(String),
    Array(Vec<ConstValue>),
}

#[derive(Debug, Clone)]
pub struct StructField {
    pub name: String,
    pub ty: Type,
    pub offset: usize,
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<StructField>,
    pub size: usize,
    pub align: usize,
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
    pub(crate) struct_defs: HashMap<String, StructDef>,
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
    For {
        init: Option<ForInit>,
        cond: Option<Expr>,
        step: Option<Expr>,
        body: Box<Stmt>,
    },
    Switch {
        expr: Expr,
        items: Vec<SwitchItem>,
    },
    Block(Vec<Stmt>),
    ExprStmt(Option<Expr>),
    Decl(Vec<(usize, Option<Expr>)>),
    InlineAsm(String),
    Break,
    Continue,
}

#[derive(Debug, Clone)]
pub enum ForInit {
    Decl(Vec<(usize, Option<Expr>)>),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub enum SwitchItem {
    Case(i64),
    Default,
    Stmt(Stmt),
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
    Inc {
        expr: Box<Expr>,
        is_post: bool,
    },
    Member {
        base: Box<Expr>,
        field: String,
        is_arrow: bool,
    },
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
    },
    Conditional {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
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

pub fn assign_offsets(
    locals: &mut [Var],
    params: &[usize],
    struct_defs: &HashMap<String, StructDef>,
) {
    let mut offset = 4i32;
    for &id in params {
        let size = locals[id].ty.size_with(struct_defs) as i32;
        let align = locals[id].ty.align_with(struct_defs) as i32;
        offset = align_up(offset, align);
        locals[id].offset = offset;
        offset += size;
    }
    for (i, var) in locals.iter_mut().enumerate() {
        if params.contains(&i) {
            continue;
        }
        let size = var.ty.size_with(struct_defs) as i32;
        let align = var.ty.align_with(struct_defs) as i32;
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
    struct_defs: &'a HashMap<String, StructDef>,
    strings: Vec<(String, String)>,
    string_map: HashMap<String, String>,
    break_stack: Vec<String>,
    continue_stack: Vec<String>,
}

impl<'a> Codegen<'a> {
    fn new(
        func_types: &'a HashMap<String, Type>,
        global_types: &'a HashMap<String, Type>,
        globals: &'a [Global],
        struct_defs: &'a HashMap<String, StructDef>,
    ) -> Self {
        Codegen {
            out: Vec::new(),
            label: 0,
            func_types,
            global_types,
            globals,
            struct_defs,
            strings: Vec::new(),
            string_map: HashMap::new(),
            break_stack: Vec::new(),
            continue_stack: Vec::new(),
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

    fn string_label(&mut self, s: &str) -> String {
        if let Some(label) = self.string_map.get(s) {
            return label.clone();
        }
        let label = format!("_Lstr_{}", self.string_map.len());
        self.string_map.insert(s.to_string(), label.clone());
        self.strings.push((label.clone(), s.to_string()));
        label
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
    fn gen_program(&mut self, prog: &Program, mode: EmitMode) {
        self.emit(".text");
        if matches!(mode, EmitMode::Executable) {
            self.emit("movi r30, MEM_SIZE");
            self.emit("call main");
            self.emit("halt");
        }
        for func in &prog.funcs {
            self.emit(format!(".global {}", func.name));
            self.gen_function(func);
        }
        self.gen_globals();
        self.gen_string_literals();
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
            let size = g.ty.size_with(self.struct_defs) as u32;
            let align = g.ty.align_with(self.struct_defs) as u32;
            if let Some(init) = &g.init {
                let aligned = align_up_u32(data_offset, align);
                if aligned > data_offset {
                    data_lines.push(format!(".zero {}", aligned - data_offset));
                    data_offset = aligned;
                }
                data_lines.push(format!(".global {}", g.name));
                data_lines.push(format!("{}:", g.name));
                self.emit_const_init(&g.ty, init, &mut data_lines, &g.name);
                data_offset += size;
            } else {
                let aligned = align_up_u32(bss_offset, align);
                if aligned > bss_offset {
                    bss_lines.push(format!(".zero {}", aligned - bss_offset));
                    bss_offset = aligned;
                }
                bss_lines.push(format!(".global {}", g.name));
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

    fn emit_const_init(
        &mut self,
        ty: &Type,
        init: &ConstValue,
        out: &mut Vec<String>,
        name: &str,
    ) {
        match (ty, init) {
            (Type::Char, ConstValue::Int(v)) => {
                out.push(format!(".byte {}", (*v as i32) & 0xFF));
            }
            (Type::Int | Type::Long | Type::Ptr(_), ConstValue::Int(v)) => {
                out.push(format!(".long {}", *v as i32));
            }
            (Type::Float, ConstValue::Float(f)) => {
                out.push(format!(".long {}", f.to_bits()));
            }
            (Type::Float, ConstValue::Int(v)) => {
                let f = *v as f32;
                out.push(format!(".long {}", f.to_bits()));
            }
            (Type::Ptr(_), ConstValue::Str(s)) => {
                let label = self.string_label(s);
                out.push(format!(".long {}", label));
            }
            (Type::Array(elem, n), ConstValue::Str(s)) if matches!(&**elem, Type::Char) => {
                let bytes = s.as_bytes();
                let needed = bytes.len() + 1;
                if *n != 0 && needed > *n {
                    panic!("String literal too long for array {}", name);
                }
                let emit_len = if *n == 0 { needed } else { *n };
                out.push(format!(".ascii \"{}\"", escape_asm_string(s)));
                out.push(".byte 0".to_string());
                if emit_len > needed {
                    out.push(format!(".zero {}", emit_len - needed));
                }
            }
            (Type::Array(elem, n), ConstValue::Array(vals)) => {
                let count = if *n == 0 { vals.len() } else { *n };
                let mut i = 0usize;
                while i < count {
                    if i < vals.len() {
                        self.emit_const_init(elem, &vals[i], out, name);
                    } else {
                        let zero_size = elem.size_with(self.struct_defs);
                        out.push(format!(".zero {}", zero_size));
                    }
                    i += 1;
                }
            }
            _ => panic!("Unsupported global initializer for {}", name),
        }
    }

    fn gen_string_literals(&mut self) {
        if self.strings.is_empty() {
            return;
        }
        let strings = std::mem::take(&mut self.strings);
        self.emit(".data");
        for (label, value) in strings {
            self.emit(format!("{}:", label));
            self.emit(format!(".asciz \"{}\"", escape_asm_string(&value)));
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
            let end = var.offset + var.ty.size_with(self.struct_defs) as i32;
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
                self.break_stack.push(end_label.clone());
                self.continue_stack.push(begin_label.clone());
                self.emit(format!("{}:", begin_label));
                self.gen_expr(cond, func);
                let cty = self.expr_type(cond, func);
                self.emit_cmp_zero_for_type(&cty);
                self.emit(format!("jz {}", end_label));
                self.gen_stmt(body, func, ret_label);
                self.emit(format!("jmp {}", begin_label));
                self.emit(format!("{}:", end_label));
                self.continue_stack.pop();
                self.break_stack.pop();
            }
            Stmt::For { init, cond, step, body } => {
                if let Some(init) = init {
                    match init {
                        ForInit::Decl(vars) => {
                            self.gen_stmt(&Stmt::Decl(vars.clone()), func, ret_label);
                        }
                        ForInit::Expr(expr) => {
                            self.gen_expr(expr, func);
                        }
                    }
                }
                let begin_label = self.new_label("Lfor_begin");
                let step_label = self.new_label("Lfor_step");
                let end_label = self.new_label("Lfor_end");
                self.break_stack.push(end_label.clone());
                self.continue_stack.push(step_label.clone());
                self.emit(format!("{}:", begin_label));
                if let Some(cond) = cond {
                    self.gen_expr(cond, func);
                    let cty = self.expr_type(cond, func);
                    self.emit_cmp_zero_for_type(&cty);
                    self.emit(format!("jz {}", end_label));
                }
                self.gen_stmt(body, func, ret_label);
                self.emit(format!("{}:", step_label));
                if let Some(step) = step {
                    self.gen_expr(step, func);
                }
                self.emit(format!("jmp {}", begin_label));
                self.emit(format!("{}:", end_label));
                self.continue_stack.pop();
                self.break_stack.pop();
            }
            Stmt::Switch { expr, items } => {
                let end_label = self.new_label("Lswitch_end");
                self.break_stack.push(end_label.clone());

                self.gen_expr(expr, func);
                self.emit("mov r1, r0");

                let mut case_labels: Vec<(i64, String)> = Vec::new();
                let mut default_label: Option<String> = None;

                for item in items {
                    if let SwitchItem::Case(v) = item {
                        if case_labels.iter().any(|(val, _)| val == v) {
                            panic!("Duplicate case value {}", v);
                        }
                        case_labels.push((*v, self.new_label("Lcase")));
                    } else if let SwitchItem::Default = item {
                        if default_label.is_some() {
                            panic!("Duplicate default label in switch");
                        }
                        default_label = Some(self.new_label("Ldefault"));
                    }
                }

                for (val, label) in &case_labels {
                    self.emit(format!("cmpi r1, {}", *val as i32));
                    self.emit(format!("jz {}", label));
                }
                if let Some(def_label) = &default_label {
                    self.emit(format!("jmp {}", def_label));
                } else {
                    self.emit(format!("jmp {}", end_label));
                }

                let mut case_iter = case_labels.into_iter();
                let mut next_case_label: Option<(i64, String)> = case_iter.next();
                for item in items {
                    match item {
                        SwitchItem::Case(_) => {
                            if let Some((_, label)) = next_case_label.take() {
                                self.emit(format!("{}:", label));
                                next_case_label = case_iter.next();
                            }
                        }
                        SwitchItem::Default => {
                            if let Some(def_label) = &default_label {
                                self.emit(format!("{}:", def_label));
                            }
                        }
                        SwitchItem::Stmt(stmt) => {
                            self.gen_stmt(stmt, func, ret_label);
                        }
                    }
                }

                self.emit(format!("{}:", end_label));
                self.break_stack.pop();
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
                        let lty = &func.locals[*id].ty;
                        if let ExprKind::Str(s) = &expr.kind {
                            if let Type::Array(elem, n) = lty {
                                if !matches!(&**elem, Type::Char) {
                                    panic!("String literal can only init char arrays");
                                }
                                let bytes = s.as_bytes();
                                let needed = bytes.len() + 1;
                                if *n != 0 && needed > *n {
                                    panic!("String literal too long for array");
                                }
                                self.gen_lvalue(&Expr { kind: ExprKind::Var(*id) }, func);
                                self.emit("mov r1, r0");
                                for (i, b) in bytes.iter().enumerate() {
                                    self.emit(format!("movi r0, {}", *b as u32));
                                    self.emit(format!("store [r1 + {}], r0", i));
                                }
                                self.emit("movi r0, 0");
                                self.emit(format!("store [r1 + {}], r0", bytes.len()));
                                if *n != 0 && needed < *n {
                                    for i in needed..*n {
                                        self.emit("movi r0, 0");
                                        self.emit(format!("store [r1 + {}], r0", i));
                                    }
                                }
                                continue;
                            }
                        }
                        self.gen_expr(expr, func);
                        let rty = self.expr_type(expr, func);
                        self.cast_reg(&rty, lty);
                        let offset = func.locals[*id].offset;
                        self.emit_store_var(lty, offset);
                    }
                }
            },
            Stmt::ExprStmt(None) => {
            }
            Stmt::Break => {
                let target = self.break_stack.last().unwrap_or_else(|| {
                    panic!("break used outside of loop/switch")
                });
                self.emit(format!("jmp {}", target));
            }
            Stmt::Continue => {
                let target = self.continue_stack.last().unwrap_or_else(|| {
                    panic!("continue used outside of loop")
                });
                self.emit(format!("jmp {}", target));
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
                if let ExprKind::Str(s) = &expr.kind {
                    let label = self.string_label(s);
                    self.emit(format!("movi r0, {}", label));
                }
            }
            ExprKind::Var(id) => {
                let var = &func.locals[*id];
                if matches!(var.ty, Type::Struct(_)) {
                    panic!("Struct values cannot be used directly; access fields instead");
                }
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
                if matches!(gty, Type::Struct(_)) {
                    panic!("Struct values cannot be used directly; access fields instead");
                }
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
                    if matches!(lty, Type::Struct(_)) {
                        panic!("Cannot assign struct values");
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
            ExprKind::Inc { expr, is_post } => {
                self.gen_inc(expr, *is_post, func);
            }
            ExprKind::Member { .. } => {
                self.gen_lvalue(expr, func);
                let rty = self.expr_type(expr, func);
                self.emit_load_from_addr(&rty, "r0");
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
            ExprKind::Index { .. } => {
                self.gen_lvalue(expr, func);
                let rty = self.expr_type(expr, func);
                if !rty.is_array() {
                    self.emit_load_from_addr(&rty, "r0");
                }
            }
            ExprKind::Cast { ty, expr: inner } => {
                let from = self.expr_type(inner, func);
                self.gen_expr(inner, func);
                self.cast_reg(&from, ty);
            }
            ExprKind::Conditional { cond, then_expr, else_expr } => {
                self.gen_expr(cond, func);
                let t_else = self.new_label("Lcond_else");
                let t_end = self.new_label("Lcond_end");
                let cty = self.expr_type(cond, func);
                self.emit_cmp_zero_for_type(&cty);
                self.emit(format!("jz {}", t_else));
                self.gen_expr(then_expr, func);
                let then_ty = self.expr_type(then_expr, func);
                let else_ty = self.expr_type(else_expr, func);
                let res_ty = self.cond_result_type(&then_ty, &else_ty);
                self.cast_reg(&then_ty, &res_ty);
                self.emit(format!("jmp {}", t_end));
                self.emit(format!("{}:", t_else));
                self.gen_expr(else_expr, func);
                self.cast_reg(&else_ty, &res_ty);
                self.emit(format!("{}:", t_end));
            }
        }
    }

    fn cond_result_type(&self, a: &Type, b: &Type) -> Type {
        if a == b {
            return a.clone();
        }
        if Self::is_float_type(a) || Self::is_float_type(b) {
            return Type::Float;
        }
        Type::Int
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

    fn gen_inc(&mut self, expr: &Expr, is_post: bool, func: &Function) {
        let lty = self
            .lvalue_type(expr, func)
            .unwrap_or_else(|| panic!("++ operand is not an lvalue"));
        if lty.is_array() {
            panic!("++ not allowed on array type");
        }

        self.gen_lvalue(expr, func);
        self.emit("mov r1, r0"); // r1 = addr
        self.emit_load_from_addr(&lty, "r1");

        if is_post {
            self.emit("push r0");
        }

        match &lty {
            Type::Float => {
                self.emit("movi r2, 0x3F800000"); // 1.0f
                self.emit("fadd r0, r0, r2");
            }
            Type::Ptr(elem) => {
                let sz = elem.size_with(self.struct_defs);
                if sz == 1 {
                    self.emit("inc r0");
                } else {
                    self.emit(format!("movi r2, {}", sz));
                    self.emit("add r0, r0, r2");
                }
            }
            _ => {
                self.emit("inc r0");
            }
        }

        self.emit_store_to_addr(&lty, "r1");

        if is_post {
            self.emit("pop r0");
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
                        self.scale_reg("r0", elem.size_with(self.struct_defs));
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
                        self.scale_reg("r1", elem.size_with(self.struct_defs));
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
                        let esz = elem.size_with(self.struct_defs);
                        if esz != 1 {
                            self.emit(format!("movi r1, {}", esz));
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
            ExprKind::Member { base, field, is_arrow } => {
                let (offset, _fty) = self.member_offset(base, field, *is_arrow, func);
                if *is_arrow {
                    self.gen_expr(base, func);
                } else {
                    self.gen_lvalue(base, func);
                }
                if offset != 0 {
                    self.emit(format!("movi r1, {}", offset as i32));
                    self.emit("add r0, r0, r1");
                }
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
                self.scale_reg("r0", elem.size_with(self.struct_defs));
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
            ExprKind::Str(_) => Type::Ptr(Box::new(Type::Char)),
            ExprKind::Var(id) => func.locals[*id].ty.decay(),
            ExprKind::GlobalVar(name) => self
                .global_types
                .get(name)
                .unwrap_or_else(|| panic!("Undefined global: {}", name))
                .decay(),
            ExprKind::Assign(lhs, _) => self.expr_type(lhs, func),
            ExprKind::Inc { expr, .. } => {
                let ty = self
                    .lvalue_type(expr, func)
                    .unwrap_or_else(|| panic!("++ operand is not an lvalue"));
                if ty.is_array() {
                    panic!("++ not allowed on array type");
                }
                ty.decay()
            }
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
            ExprKind::Member { base, field, is_arrow } => {
                let (offset, ty) = self.member_offset(base, field, *is_arrow, func);
                let _ = offset;
                ty
            }
            ExprKind::Conditional { then_expr, else_expr, .. } => {
                let a = self.expr_type(then_expr, func);
                let b = self.expr_type(else_expr, func);
                self.cond_result_type(&a, &b)
            }
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
            ExprKind::Member { base, field, is_arrow } => {
                let (_, ty) = self.member_offset(base, field, *is_arrow, func);
                Some(ty)
            }
            _ => None,
        }
    }

    fn member_offset(&self, base: &Expr, field: &str, is_arrow: bool, func: &Function) -> (usize, Type) {
        let base_ty = self.expr_type(base, func);
        let struct_name = match (is_arrow, base_ty) {
            (true, Type::Ptr(inner)) => match *inner {
                Type::Struct(name) => name,
                _ => panic!("-> used on non-struct pointer"),
            },
            (false, Type::Struct(name)) => name,
            _ => panic!(". used on non-struct value"),
        };
        let def = self
            .struct_defs
            .get(&struct_name)
            .unwrap_or_else(|| panic!("Unknown struct type: {}", struct_name));
        for f in &def.fields {
            if f.name == field {
                return (f.offset, f.ty.clone());
            }
        }
        panic!("Unknown field {} in struct {}", field, struct_name);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitMode {
    Executable,
    Object,
}

pub fn compile(input: &String, _arch: &ArchConfig, mode: EmitMode) -> String {
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

    let mut codegen = Codegen::new(
        &parser.func_types,
        &parser.global_types,
        &prog.globals,
        &prog.struct_defs,
    );
    codegen.gen_program(&prog, mode);

    let mut asm_output = String::new();
    for line in &codegen.out {
        asm_output.push_str(line);
        asm_output.push('\n');
    }
    asm_output
}

fn escape_asm_string(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0' => out.push_str("\\0"),
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{compile, EmitMode};
    use crate::assemble::config::ArchConfig;

    fn compile_ok(src: &str) -> String {
        compile(&src.to_string(), &ArchConfig::default(), EmitMode::Executable)
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
    fn compile_pre_increment() {
        let asm = compile_ok("int main(){int x=1; return ++x;}");
        assert!(asm.contains("inc r0"));
    }

    #[test]
    fn compile_post_increment() {
        let asm = compile_ok("int main(){int x=1; return x++;}");
        assert!(asm.contains("inc r0"));
    }

    #[test]
    fn compile_for_break_continue() {
        let asm = compile_ok("int main(){int i=0; for(i=0;i<10;i=i+1){ if(i==3) continue; if(i==5) break; } return i;}");
        assert!(asm.contains("Lfor_begin"));
        assert!(asm.contains("Lfor_end"));
    }

    #[test]
    fn compile_switch() {
        let asm = compile_ok("int main(){int x=2; switch(x){case 1: x=3; break; case 2: x=4; default: x=5;} return x;}");
        assert!(asm.contains("Lswitch_end"));
    }

    #[test]
    fn compile_struct_and_member() {
        let asm = compile_ok("typedef struct { int a; char b; } Foo; int main(){Foo f; f.a=1; f.b=2; return f.a;}");
        assert!(asm.contains("store32"));
        assert!(asm.contains("store ["));
    }

    #[test]
    fn compile_string_literal_ptr() {
        let asm = compile_ok("int main(){char *p=\"hi\"; return p[0];}");
        assert!(asm.contains(".asciz"));
    }
}
