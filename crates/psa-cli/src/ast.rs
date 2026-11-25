//! AST for the .psa v0.1 subset.

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

/// One part of an f-string: literal text or an interpolated expression.
#[derive(Debug, Clone)]
pub enum FStrPart {
    Lit(String),
    Expr(Expr),
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

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    NoneLit,
    Name(String),
    Bin(Box<Expr>, BinOp, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    /// Call: callee, positional args, keyword args (D3). Binding
    /// (positionals in order, then keywords by parameter name,
    /// then declared defaults) happens at emit time against the
    /// callee's parameter list.
    Call(Box<Expr>, Vec<Expr>, Vec<(String, Expr)>),
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
    Index(Box<Expr>, Box<Expr>),
    /// `base.name` — attribute access (D4). The parser produces it
    /// for a `.name` postfix not followed by `(`; method calls on a
    /// plain variable keep the older MethodCall node, and a call of
    /// an Attr (`obj.m(...)`) is a Call whose callee is an Attr.
    Attr(Box<Expr>, String),
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
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Assign { name: String, value: Expr, line: u32 },
    AugAssign { name: String, op: BinOp, value: Expr, line: u32 },
    IndexAssign { name: String, index: Expr, op: Option<BinOp>, value: Expr, line: u32 },
    /// `base.name = value` / `base.name <augop> value` (D4). The
    /// base is an arbitrary expression (in practice a Name or a
    /// chained Attr); op mirrors IndexAssign (None = plain store).
    AttrAssign { base: Expr, name: String, op: Option<BinOp>, value: Expr, line: u32 },
    /// `del base.name` (D4) — a single attribute target only.
    AttrDel { base: Expr, name: String, line: u32 },
    ExprStmt { value: Expr, line: u32 },
    If { arms: Vec<(Expr, Vec<Stmt>)>, els: Option<Vec<Stmt>>, line: u32 },
    While { cond: Expr, body: Vec<Stmt>, line: u32 },
    ForRange { var: String, start: Option<Expr>, stop: Expr, body: Vec<Stmt>, line: u32 },
    /// `for var in <iterable>:` — v0.7 iterates list variables
    /// (static or dynamic); other iterables stay loud at emit.
    /// Multiple targets (D5) unpack each element.
    ForEach { vars: Vec<String>, iter: Expr, body: Vec<Stmt>, line: u32 },
    /// `a, b = <sequence>` (D5): unpack a single sequence value
    /// into the names. Distinct from MultiAssign's chained form
    /// (`a = b = v`), which the parser keeps separate.
    Unpack { names: Vec<String>, value: Expr, line: u32 },
    Return { value: Option<Expr>, line: u32 },
    /// `assert cond` / `assert cond, "msg"` (msg is a literal; it is a
    /// diagnostic label only -- psa panics with a code, no traceback).
    Assert { cond: Expr, line: u32 },
    /// `del name` / `del name[idx]` (index form is refused at emit:
    /// no SLA-source Vec shrink shape yet).
    Del { targets: Vec<(String, Option<Expr>)>, line: u32 },
    /// `a, b = v1, v2` (values.len() == names.len()) or chained
    /// `a = b = v` (values.len() == 1 < names.len()).
    MultiAssign { names: Vec<String>, values: Vec<Expr>, line: u32 },
    Break { line: u32 },
    Continue { line: u32 },
    Pass { line: u32 },
    /// `try: ... except ... else: ... finally: ...` (D6). At
    /// least one handler or a finally block is present (parser
    /// enforced). A bare `except:` has an empty `classes` list.
    Try {
        body: Vec<Stmt>,
        handlers: Vec<ExcHandler>,
        els: Option<Vec<Stmt>>,
        finally: Option<Vec<Stmt>>,
        line: u32,
    },
    /// `raise <expr>` / bare `raise` (D6). The bare form is
    /// only legal as the last statement of an except handler
    /// (emitter enforced).
    Raise { value: Option<Expr>, line: u32 },
    /// `with <expr> [as name], ...: ...` (D6). Desugared by
    /// the emitter into the try machinery with a finally that
    /// closes the items in reverse order.
    With { items: Vec<(Expr, Option<String>)>, body: Vec<Stmt>, line: u32 },
    /// `yield <expr>` / bare `yield` (D7): a statement only in
    /// v1. A def whose own body contains one is a generator
    /// def. Yield inside an expression is refused by the
    /// parser.
    Yield { value: Option<Expr>, line: u32 },
    /// `match <subject>:` with `case` arms (D7). Expanded by
    /// the emitter into per-case check preludes and guarded
    /// bodies; see D7-NOTES for the pattern subset.
    Match { subject: Expr, cases: Vec<MatchCase>, line: u32 },
}

/// One `case` arm of a match statement (D7).
#[derive(Debug, Clone)]
pub struct MatchCase {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Vec<Stmt>,
    pub line: u32,
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
}
