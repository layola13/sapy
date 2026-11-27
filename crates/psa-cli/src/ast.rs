//! AST for the .psa v0.1 subset.

/// A precise source position: which source file, which line,
/// which column. P0a sourcemap: this is the single carrier of
/// statement / definition positions; the emission layer (P1)
/// reads it to build the segment table, and the parser (P0b)
/// fills it from the token stream.
///
/// Semantics (frozen by P0a; P0b depends on these exact field
/// names — renaming a field requires a coordinated change):
/// - `source` is an index into the program's sources table
///   (0 = the main .psa file; `merge_imports` assigns each
///   imported module its own index when it merges the module's
///   Func/ClassDef in).
/// - `line` is 1-based. `line == 0` marks a compiler-synthesised
///   node with no user source line (closure conversion, lambda
///   desugar, cell creation, ...). Downstream MUST treat
///   `line == 0` as origin Synthetic; it must never be presented
///   as a real line number.
/// - `col` is the 1-based character column (Unicode scalars
///   counted from the line start; see `lexer::Token::col`).
///   `col == 0` means "line known, column unknown" (a node
///   derived from a real line by a pass that only carried the
///   line, e.g. the closure rewriter's synthetic statements);
///   it is only ever 0 together with a real `line`, or together
///   with `line == 0` (synthetic). `col >= 1` is a real column.
/// - `col_byte` is the 1-based byte column, kept for
///   cross-checking against byte-oriented tools; `None` when
///   the byte column was not recorded (e.g. a node built from a
///   line number alone).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceAnchor {
    pub source: u16,
    pub line: u32,
    pub col: u32,
    pub col_byte: Option<u32>,
}

impl SourceAnchor {
    /// Full constructor.
    pub fn new(source: u16, line: u32, col: u32, col_byte: Option<u32>) -> Self {
        SourceAnchor {
            source,
            line,
            col,
            col_byte,
        }
    }

    /// Anchor for a position in the main (single) source file.
    pub fn at(line: u32, col: u32, col_byte: Option<u32>) -> Self {
        SourceAnchor::new(0, line, col, col_byte)
    }

    /// Anchor for a compiler-synthesised node: `line == 0`
    /// marks it synthetic (see the struct docs).
    pub fn synthetic() -> Self {
        SourceAnchor {
            source: 0,
            line: 0,
            col: 0,
            col_byte: None,
        }
    }

    /// True when this anchor carries no user source line.
    pub fn is_synthetic(&self) -> bool {
        self.line == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,     // Python `/` (true division)
    FloorDiv,
    Mod,
    Pow,     // Python `**`
    BitOr,   // `|`
    BitXor,  // `^`
    BitAnd,  // `&`
    Shl,     // `<<`
    Shr,     // `>>`
    Eq,
    NotEq,
    In,    // `in` (D2: dynamic containers)
    NotIn, // `not in`
    Is,    // `is` (171-B: identity, see staging-171-isnot/SEMANTICS.md)
    IsNot, // `is not`
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
    BitNot, // `~`
}

/// A source anchor: one precise position in one .psa file (P0b,
/// SOURCEMAP-DESIGN §3.2).
/// first character; the end is EXCLUSIVE — the position just past the
/// last character — as (`end_line`, `end_col`, `end_col_byte`) in the
/// same column convention as `SourceAnchor`. A one-line `xs` starting
/// at col 1 therefore has `end_col == 3`. A call whose argument list
/// spans lines has `end_line` past `start.line`; the end position is
/// meaningless without it. Synthetic spans have `start.line == 0`
/// (and `end_line == 0`, `end_col == 0`, `end_col_byte == None`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExprSpan {
    pub start: SourceAnchor,
    pub end_line: u32,
    pub end_col: u32,
    pub end_col_byte: Option<u32>,
}

impl ExprSpan {
    pub fn synthetic() -> ExprSpan {
        ExprSpan { start: SourceAnchor::synthetic(), end_line: 0, end_col: 0, end_col_byte: None }
    }

    pub fn is_synthetic(&self) -> bool {
        self.start.line == 0
    }

    /// Span of a bare name: `start` through just past the name.
    /// Names are ASCII identifiers, so char and byte lengths agree,
    /// but both are computed from the name itself.
    pub fn from_name(start: SourceAnchor, name: &str) -> ExprSpan {
        ExprSpan {
            start,
            end_line: start.line,
            end_col: start.col + name.chars().count() as u32,
            end_col_byte: start.col_byte.map(|b| b + name.len() as u32),
        }
    }
}

/// One part of an f-string: literal text or a replacement field.
#[derive(Debug, Clone)]
pub enum FStrPart {
    Lit(String),
    /// A replacement field `{expr}`, optionally with a
    /// conversion (`!r`/`!s`/`!a`), a debug `=` marker and/or
    /// a `:spec` format spec.
    Field {
        /// The value expression.
        value: Expr,
        /// Conversion character: 'r', 's' or 'a'.
        conv: Option<char>,
        /// Debug `=` marker: the verbatim literal prefix
        /// (expression text as written, the `=` and any
        /// following spaces) printed in front of the value. A
        /// debug field with no explicit conversion formats
        /// with repr (CPython).
        debug: Option<String>,
        /// The format spec: literal parts plus nested
        /// replacement fields, resolved recursively. None =
        /// no `:` was present; Some(vec![]) = an empty spec.
        spec: Option<Vec<FStrPart>>,
    },
}

impl FStrPart {
    /// True when the part carries a conversion, a debug marker
    /// or a format spec — i.e. it cannot take the plain
    /// interpolation fast paths.
    pub fn is_formatted(&self) -> bool {
        matches!(
            self,
            FStrPart::Field { conv, debug, spec, .. }
                if conv.is_some() || debug.is_some() || spec.is_some()
        )
    }

    /// Visit the field's value expression and, recursively,
    /// every expression inside its format spec.
    pub fn visit_exprs(&self, f: &mut dyn FnMut(&Expr)) {
        if let FStrPart::Field { value, spec, .. } = self {
            f(value);
            if let Some(parts) = spec {
                for p in parts {
                    p.visit_exprs(f);
                }
            }
        }
    }

    /// True when the predicate holds for the field's value
    /// expression or, recursively, for any expression inside
    /// its format spec.
    pub fn any_expr(&self, pred: &dyn Fn(&Expr) -> bool) -> bool {
        match self {
            FStrPart::Lit(_) => false,
            FStrPart::Field { value, spec, .. } => {
                pred(value)
                    || spec.as_ref().map_or(false, |parts| {
                        parts.iter().any(|p| p.any_expr(pred))
                    })
            }
        }
    }
}

/// Comprehension result kind (D5; GenExp semantics are D7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompKind {
    List,
    Set,
    Dict,
    GenExp,
}

/// One `for` or `if` clause of a comprehension, in source order.
#[derive(Debug, Clone)]
pub enum CompClause {
    For { var: String, iter: Expr },
    If(Expr),
}

/// Expression positions (P0b, design §3.2).
///
/// FALLBACK RULE: only `Name`, `Call`, `Index` and `Attr` carry their
/// own span. Every other variant carries no position of its own;
/// a consumer that needs a position for one MUST use the enclosing
/// statement's anchor (`Stmt::anchor`, precision statement-level)
/// and MUST NOT invent a column for it. `Lambda` keeps its legacy
/// `line` field only; its body expressions carry their own spans.
#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    /// Integer literal outside the i64 range. Payload is a
    /// normalized decimal digit string; it may carry a leading
    /// '-' ONLY when folded from a negative match pattern
    /// (parser) — expression literals are unsigned, sign is
    /// UnaryNeg, exactly as for Int.
    BigInt(String),
    Float(f64),
    Str(String),
    /// `b"..."` bytes literal: the raw byte content. Bytes are
    /// always dynamic values (PsaVal tag 15) riding the same
    /// Box<Vec<u8>> payload as dynamic strings (tag 6) but with
    /// CPython bytes semantics (repr with the b'' prefix,
    /// byte-wise indexing returning ints, no str methods).
    Bytes(Vec<u8>),
    Bool(bool),
    NoneLit,
    /// A name use, with its span (P0b): start is the name's first
    /// character, end is just past its last character.
    Name(String, ExprSpan),
    /// Assignment expression `(name := value)`: evaluates the
    /// value, binds it to the name (with Stmt::Assign's binding
    /// semantics) and yields the value. The parser only produces
    /// it parenthesised, or bare as the whole condition of an
    /// if/while/comprehension-if/match-guard position.
    Walrus { name: String, value: Box<Expr> },
    Bin(Box<Expr>, BinOp, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    /// Call: callee, positional args, keyword args (D3). Binding
    /// (positionals in order, then keywords by parameter name,
    /// then declared defaults) happens at emit time against the
    /// callee's parameter list. The span (P0b) covers the
    /// callee through the closing `)`.
    Call(Box<Expr>, Vec<Expr>, Vec<(String, Expr)>, ExprSpan),
    ListLit(Vec<Expr>),
    /// `{k: v, ...}` — dynamic dict literal (D2 slice C).
    DictLit(Vec<(Expr, Expr)>),
    /// `(a, b)` / `(a,)` / `()` — dynamic tuple literal (D5,
    /// tag 10). Also produced for a bare `a, b` in return
    /// position.
    TupleLit(Vec<Expr>),
    /// `{a, b, ...}` — dynamic set literal (D5, tag 11). The
    /// parser distinguishes it from DictLit by the absence of
    /// colons.
    SetLit(Vec<Expr>),
    /// Comprehension (D5): `[key for ...]`, `{key for ...}`,
    /// `{key: value for ...}`, `(key for ...)` (GenExp).
    /// Clauses nest in source order.
    Comp {
        kind: CompKind,
        key: Box<Expr>,
        value: Option<Box<Expr>>,
        clauses: Vec<CompClause>,
    },
    /// `base[index]` with its span (P0b): start is the base's
    /// start, end is just past `]`.
    Index(Box<Expr>, Box<Expr>, ExprSpan),
    /// `base.name` — attribute access (D4). The parser produces it
    /// for a `.name` postfix not followed by `(`; method calls on a
    /// plain variable keep the older MethodCall node, and a call of
    /// an Attr (`obj.m(...)`) is a Call whose callee is an Attr.
    /// `base.name` with its span (P0b): start is the base's start,
    /// end is just past the attribute name.
    Attr(Box<Expr>, String, ExprSpan),
    /// `base[lo:hi:step]` — any of lo/hi/step may be omitted.
    Slice(Box<Expr>, Option<Box<Expr>>, Option<Box<Expr>>, Option<Box<Expr>>),
    MethodCall { obj: String, method: String, args: Vec<Expr> },
    /// `then if cond else els` (Python conditional expression).
    IfExp { cond: Box<Expr>, then: Box<Expr>, els: Box<Expr> },
    /// f-string with already-parsed interpolation expressions.
    FStr(Vec<FStrPart>),
    /// `lambda a, b: body` (D3). Desugared by the emitter's
    /// pre-pass into a synthetic def plus an FnVal node.
    Lambda { params: Vec<String>, body: Box<Expr>, line: u32 },
    /// A user function used as a value (D3): a dynamic callable
    /// (PsaVal tag 7). Produced only by the lambda desugar
    /// pre-pass; the parser never emits it.
    FnVal(String),
    /// `await <expr>` (D7): runs a tag-14 coroutine value to
    /// completion at the await point (sequential semantics).
    Await(Box<Expr>),
    /// D8 closures: allocate a heap cell holding the dynamic
    /// form of the value. A cell is a one-element dynamic list
    /// (tag 4); every scope that shares the variable reads and
    /// writes through the same cell object, which is CPython's
    /// late-binding cell semantics. Produced only by the
    /// closure-conversion pass; the parser never emits it.
    CellNew(Box<Expr>),
    /// D8 closures: read the value held in a cell.
    CellGet(Box<Expr>),
    /// D8 closures: store a dynamic value into a cell;
    /// evaluates to the stored value.
    CellSet(Box<Expr>, Box<Expr>),
    /// D8 closures: a function value carrying an environment —
    /// the def key of a synthetic closure def plus the
    /// environment expression (a dynamic list of cells, in the
    /// def's capture order). Emitted as a tag-16 PsaVal whose
    /// payload mirrors the bound-method layout:
    /// [box_int(callable id), env]. Produced only by the
    /// closure-conversion pass.
    ClosureVal(String, Box<Expr>),
    /// D8 closures: the module environment — the dynamic list
    /// of the module scope's cells, in slot order, fetched at
    /// run time from the sa_std globals cell "psa.modenv"
    /// (published by Stmt::PublishModEnv at the head of main).
    /// Produced only by the closure-conversion pass.
    ModEnv,
}

/// Every statement carries `line` (unchanged, 1-based, 0 for
/// synthetic nodes) plus `anchor` (P0b): the anchor of the
/// statement's FIRST token — `source` is the `sources[]` index
/// (main file 0), `col` its 1-based character column, `col_byte`
/// its 1-based byte column. `line == 0` marks a synthetic node;
/// never present 0 as a real line number.
#[derive(Debug, Clone)]
pub enum Stmt {
    Assign { name: String, value: Expr, line: u32, anchor: SourceAnchor },
    AugAssign { name: String, op: BinOp, value: Expr, line: u32, anchor: SourceAnchor },
    IndexAssign { name: String, index: Expr, op: Option<BinOp>, value: Expr, line: u32, anchor: SourceAnchor },
    /// `base.name = value` / `base.name <augop> value` (D4). The
    /// base is an arbitrary expression (in practice a Name or a
    /// chained Attr); op mirrors IndexAssign (None = plain store).
    AttrAssign { base: Expr, name: String, op: Option<BinOp>, value: Expr, line: u32, anchor: SourceAnchor },
    /// `del base.name` (D4) — a single attribute target only.
    AttrDel { base: Expr, name: String, line: u32, anchor: SourceAnchor },
    ExprStmt { value: Expr, line: u32, anchor: SourceAnchor },
    If { arms: Vec<(Expr, Vec<Stmt>)>, els: Option<Vec<Stmt>>, line: u32, anchor: SourceAnchor },
    /// `while cond: body [else: els]` — the else clause runs
    /// when the loop ends without a `break` (CPython loop-else).
    While { cond: Expr, body: Vec<Stmt>, els: Option<Vec<Stmt>>, line: u32, anchor: SourceAnchor },
    ForRange {
        var: String,
        start: Option<Expr>,
        stop: Expr,
        body: Vec<Stmt>,
        els: Option<Vec<Stmt>>,
        line: u32,
        anchor: SourceAnchor,
    },
    /// `for var in <iterable>:` — v0.7 iterates list variables
    /// (static or dynamic); other iterables stay loud at emit.
    /// Multiple targets (D5) unpack each element. `els` is the
    /// loop-else clause, as on While.
    ForEach { vars: Vec<String>, iter: Expr, body: Vec<Stmt>, els: Option<Vec<Stmt>>, line: u32, anchor: SourceAnchor },
    /// `a, b = <sequence>` (D5): unpack a single sequence value
    /// into the names. Distinct from MultiAssign's chained form
    /// (`a = b = v`), which the parser keeps separate.
    Unpack { names: Vec<String>, value: Expr, line: u32, anchor: SourceAnchor },
    Return { value: Option<Expr>, line: u32, anchor: SourceAnchor },
    /// `assert cond` / `assert cond, "msg"` (msg is a literal; it is a
    /// diagnostic label only -- psa panics with a code, no traceback).
    Assert { cond: Expr, line: u32, msg: Option<String>, anchor: SourceAnchor },
    /// `del name` / `del name[idx]` (index form is refused at emit:
    /// no SLA-source Vec shrink shape yet).
    Del { targets: Vec<(String, Option<Expr>)>, line: u32, anchor: SourceAnchor },
    /// `a, b = v1, v2` (values.len() == names.len()) or chained
    /// `a = b = v` (values.len() == 1 < names.len()).
    MultiAssign { names: Vec<String>, values: Vec<Expr>, line: u32, anchor: SourceAnchor },
    Break { line: u32, anchor: SourceAnchor },
    Continue { line: u32, anchor: SourceAnchor },
    Pass { line: u32, anchor: SourceAnchor },
    /// `try: ... except ... else: ... finally: ...` (D6). At
    /// least one handler or a finally block is present (parser
    /// enforced). A bare `except:` has an empty `classes` list.
    Try {
        body: Vec<Stmt>,
        handlers: Vec<ExcHandler>,
        els: Option<Vec<Stmt>>,
        finally: Option<Vec<Stmt>>,
        line: u32,
        anchor: SourceAnchor,
    },
    /// `raise <expr>` / bare `raise` (D6). The bare form is
    /// only legal as the last statement of an except handler
    /// (emitter enforced).
    Raise { value: Option<Expr>, line: u32, anchor: SourceAnchor },
    /// `with <expr> [as name], ...: ...` (D6). Desugared by
    /// the emitter into the try machinery with a finally that
    /// closes the items in reverse order.
    With { items: Vec<(Expr, Option<String>)>, body: Vec<Stmt>, line: u32, anchor: SourceAnchor },
    /// `yield <expr>` / bare `yield` (D7): a statement only in
    /// v1. A def whose own body contains one is a generator
    /// def. Yield inside an expression is refused by the
    /// parser.
    Yield { value: Option<Expr>, line: u32, anchor: SourceAnchor },
    /// `match <subject>:` with `case` arms (D7). Expanded by
    /// the emitter into per-case check preludes and guarded
    /// bodies; see D7-NOTES for the pattern subset.
    Match { subject: Expr, cases: Vec<MatchCase>, line: u32, anchor: SourceAnchor },
    /// `def name(...): ...` nested inside a function body
    /// (D8). The closure-conversion pass lifts it to a
    /// synthetic top-level def and replaces this statement
    /// with the binding of its (possibly closure) value;
    /// it never reaches emission.
    Def(Func),
    /// `global a, b` (D8). Consumed by the closure-conversion
    /// pass (the names resolve to module-environment cells);
    /// never reaches emission.
    Global { names: Vec<String>, line: u32, anchor: SourceAnchor },
    /// `nonlocal a, b` (D8). Consumed by the closure-conversion
    /// pass (the names resolve to enclosing-function cells);
    /// never reaches emission.
    Nonlocal { names: Vec<String>, line: u32, anchor: SourceAnchor },
    /// D8: publish the module environment — the named cell
    /// variables (module scope's cells, in module-environment
    /// slot order) are collected into a fresh dynamic list
    /// whose box pointer is stored in the sa_std globals cell
    /// "psa.modenv". Produced only by the closure-conversion
    /// pass, at the head of the top-level statements, after
    /// the cell-creation assignments it lists.
    PublishModEnv { cells: Vec<String>, line: u32, anchor: SourceAnchor },
}

/// Position access for statements (P0a sourcemap).
///
/// Every variant carries both the legacy `line: u32` field and
/// the new `anchor: SourceAnchor`. The two always agree on the
/// line (`anchor.line == line` — the parser constructs both
/// from the same token, and every pass that rebuilds a statement
/// copies both through). `line` stays only so the existing
/// `emit.rs` / `parser.rs` readers keep compiling untouched;
/// new code must read `anchor` (it also carries the column and
/// the source index). P1 removes the `line` mirror once the
/// emitter reads `anchor` everywhere.
impl Stmt {
    pub fn anchor(&self) -> SourceAnchor {
        match self {
            Stmt::Assign { anchor, .. } => *anchor,
            Stmt::AugAssign { anchor, .. } => *anchor,
            Stmt::IndexAssign { anchor, .. } => *anchor,
            Stmt::AttrAssign { anchor, .. } => *anchor,
            Stmt::AttrDel { anchor, .. } => *anchor,
            Stmt::ExprStmt { anchor, .. } => *anchor,
            Stmt::If { anchor, .. } => *anchor,
            Stmt::While { anchor, .. } => *anchor,
            Stmt::ForRange { anchor, .. } => *anchor,
            Stmt::ForEach { anchor, .. } => *anchor,
            Stmt::Unpack { anchor, .. } => *anchor,
            Stmt::Return { anchor, .. } => *anchor,
            Stmt::Assert { anchor, .. } => *anchor,
            Stmt::Del { anchor, .. } => *anchor,
            Stmt::MultiAssign { anchor, .. } => *anchor,
            Stmt::Break { anchor, .. } => *anchor,
            Stmt::Continue { anchor, .. } => *anchor,
            Stmt::Pass { anchor, .. } => *anchor,
            Stmt::Try { anchor, .. } => *anchor,
            Stmt::Raise { anchor, .. } => *anchor,
            Stmt::With { anchor, .. } => *anchor,
            Stmt::Yield { anchor, .. } => *anchor,
            Stmt::Match { anchor, .. } => *anchor,
            Stmt::Def(f) => f.anchor,
            Stmt::Global { anchor, .. } => *anchor,
            Stmt::Nonlocal { anchor, .. } => *anchor,
            Stmt::PublishModEnv { anchor, .. } => *anchor,
        }
    }
}

/// One `case` arm of a match statement (D7).
#[derive(Debug, Clone)]
pub struct MatchCase {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Vec<Stmt>,
    pub line: u32,
    /// Anchor of the `case` keyword (P0b; same convention as `Stmt::anchor`).
    pub anchor: SourceAnchor,
}

/// A match pattern (D7 v1 subset): literals, a capture name
/// (a bare name always captures, as in Python), the wildcard,
/// sequence patterns over lists/tuples (no star) and OR
/// patterns (every alternative binds the same name set,
/// emitter enforced). Class, mapping, star and dotted value
/// patterns are refused by the parser.
#[derive(Debug, Clone)]
pub enum Pattern {
    Lit(Expr),
    Capture(String),
    Wildcard,
    Seq(Vec<Pattern>),
    Or(Vec<Pattern>),
}

/// One `except` clause of a try statement (D6). An empty
/// `classes` list is the bare `except:` catch-all.
#[derive(Debug, Clone)]
pub struct ExcHandler {
    pub classes: Vec<String>,
    pub bind: Option<String>,
    pub body: Vec<Stmt>,
    pub line: u32,
    /// Anchor of the `except` keyword (P0b; same convention as `Stmt::anchor`).
    pub anchor: SourceAnchor,
}

/// A top-level import statement (D6): `import a[.b]` (names
/// is None) or `from a[.b] import x [as y], ...` (names is
/// Some). Resolved and merged by the driver before emission.
#[derive(Debug, Clone)]
pub struct Import {
    pub module: String,
    pub names: Option<Vec<(String, Option<String>)>>,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct Func {
    pub name: String,
    /// Parameters in declaration order; the second element is the
    /// default-value expression for parameters that declare one
    /// (D3). Parser-enforced: once a parameter has a default, all
    /// following parameters must have one too (Python's rule).
    pub params: Vec<(String, Option<Expr>)>,
    /// `*args` parameter name (D3): collects surplus positional
    /// arguments into a dynamic list at the call site.
    pub vararg: Option<String>,
    /// `**kwargs` parameter name (D3): collects surplus keyword
    /// arguments into a dynamic dict (rides on str-keyed dicts).
    pub kwarg: Option<String>,
    /// `async def` (D7): calling the def returns a tag-14
    /// coroutine value; the body runs at the await point.
    pub is_async: bool,
    pub body: Vec<Stmt>,
    pub line: u32,
    /// Anchor of the `def` keyword — or of `async` for `async def`
    /// (P0b; same convention as `Stmt::anchor`); `line` mirrors
    /// `anchor.line` (P0a sourcemap).
    pub anchor: SourceAnchor,
}

impl Func {
    /// All parameter names in signature order: the fixed
    /// parameters, then `*args`, then `**kwargs`. At the SLA
    /// level the collection parameters are ordinary trailing
    /// parameters of dynamic type.
    pub fn param_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.params.iter().map(|(n, _)| n.clone()).collect();
        if let Some(v) = &self.vararg {
            names.push(v.clone());
        }
        if let Some(k) = &self.kwarg {
            names.push(k.clone());
        }
        names
    }

    pub fn param_count(&self) -> usize {
        self.params.len() + self.vararg.is_some() as usize + self.kwarg.is_some() as usize
    }
}

/// A `class Name:` / `class Name(Base):` definition (D4). The body
/// may contain only method definitions and attribute assignments;
/// the parser refuses anything else loudly. Class attributes keep
/// their source expressions; the emitter evaluates them once at
/// program start (see D4-NOTES).
#[derive(Debug, Clone)]
pub struct ClassDef {
    pub name: String,
    pub base: Option<String>,
    pub methods: Vec<Func>,
    pub attrs: Vec<(String, Expr)>,
    pub line: u32,
    /// Anchor of the `class` keyword (P0b; same convention as
    /// `Stmt::anchor`); `line` mirrors `anchor.line` (P0a sourcemap).
    pub anchor: SourceAnchor,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub funcs: Vec<Func>,
    pub classes: Vec<ClassDef>,
    pub top: Vec<Stmt>,
    /// Top-level imports (D6), filled by the parser and
    /// consumed by the driver's merge pass; empty afterwards.
    pub imports: Vec<Import>,
    /// Module literal attributes (D6 merge output): dotted key
    /// (`foo.X`) -> the literal expression, read by evaluating
    /// the literal at the read site (the D4 class-attr shape).
    pub mod_attrs: Vec<(String, Expr)>,
    /// Literal names bound by from-imports (D6 merge output):
    /// plain name -> literal expression, readable as a name.
    pub const_names: Vec<(String, Expr)>,
    /// Names of modules merged into this program (D6).
    pub modules: Vec<String>,
    /// psa_std modules bound by import (merge output): no
    /// .psa file backs these; the emitter splices the
    /// module's SLA text and resolves its entry points
    /// directly (v1: `json`).
    pub std_imports: Vec<String>,
    /// `sources[]` registry (P0b, design §3.3): display paths of the
    /// .psa files in this program, indexed by `SourceAnchor::source`.
    /// Index 0 is always the main file (filled in by the driver);
    /// each imported module file is appended once, in first-load
    /// order, by `merge_imports`. A diagnostic that needs the file
    /// for an anchor resolves it through `source_path`.
    pub sources: Vec<String>,
}

impl Program {
    /// The display path for a `SourceAnchor::source` index, if registered.
    pub fn source_path(&self, source: u16) -> Option<&str> {
        self.sources.get(source as usize).map(|s| s.as_str())
    }
}
