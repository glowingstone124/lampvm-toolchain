# C Compiler Syntax Specification

This document describes the syntax accepted by the current compiler implementation in `src/compiler/tokenizer.rs` and `src/compiler/compiler.rs`. It is intentionally a **small C subset**.

## Output Format

The compiler emits a single binary file via the shared assembler/linker pipeline.
The file layout is:

- Header: 6 little‑endian `u32` values
  1. `TEXT_BASE`
  2. `TEXT_SIZE` (bytes)
  3. `DATA_BASE`
  4. `DATA_SIZE` (bytes)
  5. `BSS_BASE`
  6. `BSS_SIZE` (bytes)
- Text section: instruction stream, `u64` little‑endian
- Data section: raw bytes

## 1. Lexical Structure

### 1.1 Character Set and Whitespace
- Input is processed as Rust `char`s, but only ASCII punctuation/keywords matter.
- Whitespace is skipped with `char::is_whitespace` (spaces, tabs, newlines, etc.).

### 1.2 Comments
- Line comments: `//` to end of line.
- Block comments: `/* ... */` (not nested). Unterminated block comments are not diagnosed explicitly.

### 1.3 Tokens
- **Keywords**: `char`, `int`, `long`, `float`, `void`, `return`, `if`, `else`, `while`, `for`,
  `break`, `continue`, `switch`, `case`, `default`, `struct`, `typedef`, `asm`
- **Identifiers**: `[A-Za-z_][A-Za-z0-9_]*`
- **Integer literals**: decimal or hex (`0x...`)
- **Float literals**: decimal with fractional and/or exponent parts (e.g. `1.0`, `3.14`, `2e3`, `4.2e-1`)
- **String literals**: double-quoted strings with basic escapes (`\n`, `\r`, `\t`, `\\`, `\"`)
- **Punctuators/operators**:
  - Two‑char: `==` `!=` `<=` `>=` `&&` `||`
  - Single‑char: `+ - * / ( ) { } [ ] , ; : < > = & ! .`
  - Other: `->`

Not supported by the tokenizer: character literals, octal literals, `--`, `%`, bitwise operators, `?:`, `sizeof`.

## 2. Grammar (EBNF)

The grammar below reflects the parser in `src/compiler/compiler.rs`.

### 2.1 Top Level
```
program           ::= (function_def | global_decl)*

function_def      ::= type_spec declarator "(" param_list? ")" block

global_decl       ::= type_spec declarator ("=" const_expr)? ("," declarator ("=" const_expr)?)* ";"

param_list        ::= "void"                 // only valid as single parameter list
                    | param ("," param)*

param             ::= type_spec declarator
```

### 2.2 Declarations and Types
```
type_spec         ::= "char" | "int" | "long" | "float" | "void"

// Declarator supports pointers and fixed-size arrays.
// Arrays are only written as a suffix after the name.

declarator        ::= ptr* ident array_suffix*
ptr               ::= "*"
array_suffix      ::= "[" int_literal "]"

block             ::= "{" stmt* "}"

decl_stmt         ::= type_spec declarator ("=" expr)? ("," declarator ("=" expr)?)* ";"
```

Notes:
- `void` is not allowed for local or global variables.
- Top-level supports **function definitions** and **global variables**.
- Parameter arrays decay to pointers.

### 2.3 Statements
```
stmt              ::= return_stmt
                    | if_stmt
                    | while_stmt
                    | for_stmt
                    | switch_stmt
                    | block
                    | decl_stmt
                    | expr_stmt
                    | break_stmt
                    | continue_stmt

return_stmt       ::= "return" expr? ";"

if_stmt           ::= "if" "(" expr ")" stmt ("else" stmt)?

while_stmt        ::= "while" "(" expr ")" stmt

for_stmt          ::= "for" "(" (decl_stmt | expr? ";") expr? ";" expr? ")" stmt

switch_stmt       ::= "switch" "(" expr ")" "{" switch_item* "}"
switch_item       ::= "case" const_expr ":" | "default" ":" | stmt

break_stmt        ::= "break" ";"
continue_stmt     ::= "continue" ";"

expr_stmt         ::= expr? ";"
```

Not supported: `do`, labels, `goto`.

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
                    | cast
                    | postfix

cast              ::= "(" type_spec ptr* ")" unary

postfix           ::= primary ("[" expr "]")*

primary           ::= int_literal
                    | float_literal
                    | string_literal
                    | ident
                    | ident "(" arg_list? ")"
                    | "(" expr ")"
                    | primary "." ident
                    | primary "->" ident

arg_list          ::= expr ("," expr)*
```

Not supported: comma operator in expressions, `%`, bitwise ops, `--`, `sizeof`, `?:`, char literals.

## 3. Semantics and Limits

### 3.1 Types
- `char`, `int`, `long`, `float`, `void`, pointers (`*`), arrays (`[N]`), and `struct` (via `typedef`).
- Array size must be a **decimal integer literal**, except `char[] = "..."` which infers size.
- Array types decay to pointers in parameter lists and in expression contexts.
 - `char` is 1 byte. `int`, `long`, `float`, pointers are 4 bytes and aligned to 4-byte boundaries.
 - `LOAD`/`STORE` are used for `char`; `LOAD32`/`STORE32` are used for 32-bit types.

### 3.2 Variables and Scope
- Block scope with shadowing; implemented via nested scopes.
- Global variables are supported at top level.
  - Initializers must be constant expressions (numeric literals with `+ - * /` and casts).
  - String literals are allowed for `char[]` and `char*`.

### 3.3 Functions
- Function definitions only; no prototypes.
- Max **8 parameters** and **8 call arguments**.
- Return type is `int` or `void`. If a non‑void function reaches the end without `return`, it returns `0`.

### 3.4 Lvalues and Assignment
- Assignable: variables, `*ptr`, `base[index]`, and struct members (`.` / `->`).
- Arrays cannot be assigned to directly (`int a[4]; a = ...` errors).

### 3.5 Pointer Arithmetic
- `ptr + int`, `ptr - int`, `int + ptr` are supported.
- `ptr - ptr` yields element difference (scaled by element size).
- `ptr + ptr` and `int - ptr` are rejected.

### 3.6 Boolean Semantics
- Comparisons and logical operators yield `0` or `1`.
- `&&` and `||` are short‑circuiting.

### 3.7 ABI and Calling Convention
- **Integer/ptr/float/char return**: returned in `r0`. If a non‑void function reaches the end, it returns `0`.
- **Struct return (sret)**: a hidden first argument is passed as a pointer to the return storage.
  - The callee writes the struct result to `*r0` and does not return it in registers.
  - User‑visible arguments are shifted by one register/stack slot when the return type is a struct.
- **Argument registers**: the first 8 arguments are passed in `r0..r7`.
- **Stack arguments**: arguments beyond 8 are passed on the caller stack; the callee reads them at
  `[r9 + 0]`, `[r9 + 4]`, ... where `r9` is the caller stack pointer saved in the prologue.
- **Struct parameters (by value)**: the caller passes the address of the source object as the argument,
  and the callee copies it into its local parameter slot using `MEMCPY` (true by‑value semantics).
- **Struct comparison**: not supported (`==`/`!=` for struct values are invalid).

## 4. Diagnostics

Errors are raised via `panic!` with simple messages, e.g.
- unexpected token
- undefined variable
- invalid lvalue
- unsupported pointer arithmetic

## 5. Not Implemented (Explicit)

- Preprocessor (`#include`, `#define`, ...)
- `short`, `double`
- `enum`, `const`, `static`, `extern`
- `do`
- `%`, bitwise ops, `?:`, `sizeof`
- character literals, octal literals
