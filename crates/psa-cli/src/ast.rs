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

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Name(String),
    Bin(Box<Expr>, BinOp, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    ListLit(Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    MethodCall { obj: String, method: String, args: Vec<Expr> },
    /// `then if cond else els` (Python conditional expression).
    IfExp { cond: Box<Expr>, then: Box<Expr>, els: Box<Expr> },
    /// f-string with already-parsed interpolation expressions.
    FStr(Vec<FStrPart>),
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Assign { name: String, value: Expr, line: u32 },
    AugAssign { name: String, op: BinOp, value: Expr, line: u32 },
    IndexAssign { name: String, index: Expr, op: Option<BinOp>, value: Expr, line: u32 },
    ExprStmt { value: Expr, line: u32 },
    If { arms: Vec<(Expr, Vec<Stmt>)>, els: Option<Vec<Stmt>>, line: u32 },
    While { cond: Expr, body: Vec<Stmt>, line: u32 },
    ForRange { var: String, start: Option<Expr>, stop: Expr, body: Vec<Stmt>, line: u32 },
    Return { value: Option<Expr>, line: u32 },
    Break { line: u32 },
    Continue { line: u32 },
    Pass { line: u32 },
}

#[derive(Debug, Clone)]
pub struct Func {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Stmt>,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub funcs: Vec<Func>,
    pub top: Vec<Stmt>,
}
