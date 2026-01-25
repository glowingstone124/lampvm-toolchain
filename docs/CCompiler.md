# C Compiler Syntax Specification

This document describes the syntax accepted by the current compiler implementation in `src/compiler/tokenizer.rs` and `src/compiler/compiler.rs`. It is intentionally a **small C subset**.

## 1. Lexical Structure

### 1.1 Character Set and Whitespace
- Input is processed as Rust `char`s, but only ASCII punctuation/keywords matter.
- Whitespace is skipped with `char::is_whitespace` (spaces, tabs, newlines, etc.).

### 1.2 Comments
- Line comments: `//` to end of line.
- Block comments: `/* ... */` (not nested). Unterminated block comments are not diagnosed explicitly.

### 1.3 Tokens
- **Keywords**: `int`, `void`, `return`, `if`, `else`, `while`
- **Identifiers**: `[A-Za-z_][A-Za-z0-9_]*`
- **Integer literals**: **decimal only** (`0` or `[1-9][0-9]*`)
- **Punctuators/operators**:
  - Two‑char: `==` `!=` `<=` `>=` `&&` `||`
  - Single‑char: `+ - * / ( ) { } [ ] , ; < > = & !`

Not supported by the tokenizer: character literals, string literals, hex/octal literals, `++`, `--`, `->`, `.` , `%`, bitwise operators, `?:`, `sizeof`.

## 2. Grammar (EBNF)

The grammar below reflects the parser in `src/compiler/compiler.rs`.

### 2.1 Top Level
```
program           ::= function_def*

function_def      ::= type_spec declarator "(" param_list? ")" block

param_list        ::= "void"                 // only valid as single parameter list
                    | param ("," param)*

param             ::= type_spec declarator
```

### 2.2 Declarations and Types
```
type_spec         ::= "int" | "void"

// Declarator supports pointers and fixed-size arrays.
// Arrays are only written as a suffix after the name.

declarator        ::= ptr* ident array_suffix*
ptr               ::= "*"
array_suffix      ::= "[" int_literal "]"

block             ::= "{" stmt* "}"

decl_stmt         ::= "int" declarator ("=" expr)? ("," declarator ("=" expr)?)* ";"
```

Notes:
- Only `int` declarations are allowed in statement position; `void` is not allowed for local variables.
- There are **no global variables**, only function definitions at the top level.
- Parameter arrays decay to pointers.

### 2.3 Statements
```
stmt              ::= return_stmt
                    | if_stmt
                    | while_stmt
                    | block
                    | decl_stmt
                    | expr_stmt

return_stmt       ::= "return" expr? ";"

if_stmt           ::= "if" "(" expr ")" stmt ("else" stmt)?

while_stmt        ::= "while" "(" expr ")" stmt

expr_stmt         ::= expr? ";"
```

Not supported: `for`, `do`, `switch`, `case`, `break`, `continue`, labels, `goto`.

### 2.4 Expressions
```
expr              ::= assign

assign            ::= logical_or ("=" assign)?

logical_or        ::= logical_and ("||" logical_and)*
logical_and       ::= equality ("&&" equality)*

// equality and relational

equality          ::= relational (("==" | "!=") relational)*
relational        ::= add (("<" | "<=" | ">" | ">=") add)*

// arithmetic
add               ::= mul (("+" | "-") mul)*
mul               ::= unary (("*" | "/") unary)*

unary             ::= ("+" | "-" | "&" | "*" | "!") unary
                    | postfix

postfix           ::= primary ("[" expr "]")*

primary           ::= int_literal
                    | ident
                    | ident "(" arg_list? ")"
                    | "(" expr ")"

arg_list          ::= expr ("," expr)*
```

Not supported: comma operator in expressions, `%`, bitwise ops, `++/--`, `sizeof`, casts, `?:`, member access, string/char literals.

## 3. Semantics and Limits

### 3.1 Types
- `int`, `void`, pointers (`*`), and fixed‑size arrays (`[N]`).
- Array size must be a **decimal integer literal**.
- Array types decay to pointers in parameter lists and in expression contexts.

### 3.2 Variables and Scope
- Block scope with shadowing; implemented via nested scopes.
- No global variables.

### 3.3 Functions
- Function definitions only; no prototypes.
- Max **8 parameters** and **8 call arguments**.
- Return type is `int` or `void`. If a non‑void function reaches the end without `return`, it returns `0`.

### 3.4 Lvalues and Assignment
- Assignable: variables, `*ptr`, and `base[index]`.
- Arrays cannot be assigned to directly (`int a[4]; a = ...` errors).

### 3.5 Pointer Arithmetic
- `ptr + int`, `ptr - int`, `int + ptr` are supported.
- `ptr - ptr` yields element difference (scaled by element size).
- `ptr + ptr` and `int - ptr` are rejected.

### 3.6 Boolean Semantics
- Comparisons and logical operators yield `0` or `1`.
- `&&` and `||` are short‑circuiting.

## 4. Diagnostics

Errors are raised via `panic!` with simple messages, e.g.
- unexpected token
- undefined variable
- invalid lvalue
- unsupported pointer arithmetic

## 5. Not Implemented (Explicit)

- Preprocessor (`#include`, `#define`, ...)
- `char`, `short`, `long`, `float`, `double`
- `struct`, `enum`, `typedef`, `const`, `static`, `extern`
- `for`, `do`, `switch`, `break`, `continue`
- `%`, bitwise ops, `?:`, `sizeof`, casts
- string/character literals, hex/octal literals