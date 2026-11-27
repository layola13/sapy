//! Recursive-descent parser for the .psa v0.1 subset.
//! Unsupported Python constructs fail loudly here or in the emitter.

use crate::ast::*;
use crate::diag::Diagnostic;
use crate::lexer::{Tok, Token};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

fn perr(line: u32, msg: &str) -> Diagnostic {
    Diagnostic::parse(line, msg)
}

/// Anchor of one token in the file being parsed (P0b). `source` is
/// 0 here: the parser only ever sees one file at a time, and the
/// driver (`merge_imports`) rewrites `source` for imported modules.
fn anchor_of(t: &Token) -> SourceAnchor {
    SourceAnchor { source: 0, line: t.line, col: t.col, col_byte: Some(t.col_byte) }
}

/// Human wording for a token, for `expected X, found Y` messages:
/// grammar words instead of the Tok enum's Debug names (census
/// section 3: the `{:?}` leak made parse errors unreadable).
fn tok_word(t: &Tok) -> String {
    let s = match t {
        Tok::Ident(_) => "a name",
        Tok::Int(_) => "an integer",
        Tok::BigInt(_) => "an integer",
        Tok::Float(_) => "a float",
        Tok::Str(_) => "a string",
        Tok::FStr(_) => "an f-string",
        Tok::Bytes(_) => "a bytes literal",
        Tok::Walrus => "`:=`",
        Tok::Def => "`def`",
        Tok::Class => "`class`",
        Tok::Return => "`return`",
        Tok::If => "`if`",
        Tok::Elif => "`elif`",
        Tok::Else => "`else`",
        Tok::While => "`while`",
        Tok::For => "`for`",
        Tok::In => "`in`",
        Tok::Is => "`is`",
        Tok::Break => "`break`",
        Tok::Continue => "`continue`",
        Tok::Pass => "`pass`",
        Tok::And => "`and`",
        Tok::Or => "`or`",
        Tok::Not => "`not`",
        Tok::True => "`True`",
        Tok::False => "`False`",
        Tok::NoneLit => "`None`",
        Tok::Lambda => "`lambda`",
        Tok::Try => "`try`",
        Tok::Except => "`except`",
        Tok::Finally => "`finally`",
        Tok::Raise => "`raise`",
        Tok::With => "`with`",
        Tok::Import => "`import`",
        Tok::From => "`from`",
        Tok::Yield => "`yield`",
        Tok::Async => "`async`",
        Tok::Await => "`await`",
        Tok::LParen => "`(`",
        Tok::RParen => "`)`",
        Tok::LBracket => "`[`",
        Tok::RBracket => "`]`",
        Tok::LBrace => "`{`",
        Tok::RBrace => "`}`",
        Tok::Comma => "`,`",
        Tok::Colon => "`:`",
        Tok::Dot => "`.`",
        Tok::Plus => "`+`",
        Tok::Minus => "`-`",
        Tok::Star => "`*`",
        Tok::Slash => "`/`",
        Tok::FloorDiv => "`//`",
        Tok::Percent => "`%`",
        Tok::Pow => "`**`",
        Tok::BitAnd => "`&`",
        Tok::BitOr => "`|`",
        Tok::BitXor => "`^`",
        Tok::Tilde => "`~`",
        Tok::Shl => "`<<`",
        Tok::Shr => "`>>`",
        Tok::EqEq => "`==`",
        Tok::NotEq => "`!=`",
        Tok::Lt => "`<`",
        Tok::LtEq => "`<=`",
        Tok::Gt => "`>`",
        Tok::GtEq => "`>=`",
        Tok::Assign => "`=`",
        Tok::PlusEq => "`+=`",
        Tok::MinusEq => "`-=`",
        Tok::StarEq => "`*=`",
        Tok::SlashEq => "`/=`",
        Tok::FloorDivEq => "`//=`",
        Tok::PercentEq => "`%=`",
        Tok::PowEq => "`**=`",
        Tok::BitAndEq => "`&=`",
        Tok::BitOrEq => "`|=`",
        Tok::BitXorEq => "`^=`",
        Tok::ShlEq => "`<<=`",
        Tok::ShrEq => "`>>=`",
        Tok::Newline => "a newline",
        Tok::Indent => "an indented block",
        Tok::Dedent => "a dedent",
        Tok::Eof => "end of file",
    };
    s.to_string()
}

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self {
        Parser { toks, pos: 0 }
    }

    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    /// The token after the lookahead token (used for `name =`
    /// keyword-argument detection in call argument lists).
    fn peek2(&self) -> &Tok {
        if self.pos + 1 < self.toks.len() {
            &self.toks[self.pos + 1].tok
        } else {
            &self.toks[self.pos].tok
        }
    }

    fn line(&self) -> u32 {
        self.toks[self.pos].line
    }

    fn cur_tok(&self) -> &Token {
        &self.toks[self.pos]
    }

    /// Column of the current token (1-based character column, P0b).
    fn col(&self) -> u32 {
        self.cur_tok().col
    }

    /// Anchor of the current token, in the file being parsed
    /// (source 0; `merge_imports` rewrites `source` for modules).
    fn anchor(&self) -> SourceAnchor {
        anchor_of(self.cur_tok())
    }

    /// The token just consumed. Call only right after a `bump` /
    /// `expect` that consumed a real token (never Eof).
    fn prev_tok(&self) -> &Token {
        &self.toks[self.pos - 1]
    }

    /// Span from `start` to just past a single-character closing
    /// token (`)` / `]`) that was just consumed.
    fn span_past(&self, start: SourceAnchor, closer: &Token) -> ExprSpan {
        ExprSpan {
            start,
            end_line: closer.line,
            end_col: closer.col + 1,
            end_col_byte: Some(closer.col_byte + 1),
        }
    }

    /// Span from `start` to just past the identifier in `tok`
    /// (an attribute name just consumed).
    fn span_to_ident_end(&self, start: SourceAnchor, tok: &Token, name: &str) -> ExprSpan {
        ExprSpan {
            start,
            end_line: tok.line,
            end_col: tok.col + name.chars().count() as u32,
            end_col_byte: Some(tok.col_byte + name.len() as u32),
        }
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.pos].tok.clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, t: &Tok) -> Result<(), Diagnostic> {
        if self.peek() == t {
            self.bump();
            Ok(())
        } else {
            // The offending token is the current one: its column is
            // the error's column (P0b — this is what makes a_parse's
            // JSON site carry a real col).
            Err(Diagnostic::parse_col(
                self.line(),
                self.col(),
                &format!("expected {}, found {}", tok_word(t), tok_word(self.peek())),
            ))
        }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Tok::Newline) {
            self.bump();
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, Diagnostic> {
        let mut prog = Program {
            funcs: Vec::new(),
            classes: Vec::new(),
            top: Vec::new(),
            imports: Vec::new(),
            mod_attrs: Vec::new(),
            const_names: Vec::new(),
            modules: Vec::new(),
            std_imports: Vec::new(),
            // sources[0] (the main file) is filled in by the driver
            // (P0b); the parser never knows the file's path.
            sources: Vec::new(),
        };
        self.skip_newlines();
        while !matches!(self.peek(), Tok::Eof) {
            if matches!(self.peek(), Tok::Def) {
                let f = self.parse_func(false)?;
                prog.funcs.push(f);
            } else if matches!(self.peek(), Tok::Async) {
                let f = self.parse_func(true)?;
                prog.funcs.push(f);
            } else if matches!(self.peek(), Tok::Class) {
                let c = self.parse_class()?;
                prog.classes.push(c);
            } else if matches!(self.peek(), Tok::Import | Tok::From) {
                let imp = self.parse_import()?;
                prog.imports.push(imp);
            } else {
                let s = self.parse_stmt()?;
                prog.top.push(s);
            }
            self.skip_newlines();
        }
        Ok(prog)
    }

    /// `class Name:` / `class Name(Base):` at top level (D4). The
    /// body may contain only method definitions and attribute
    /// assignments (`name = expr`); anything else is refused
    /// loudly. Nested class definitions never reach here: a
    /// `class` token inside a block fails in parse_stmt.
    fn parse_class(&mut self) -> Result<ClassDef, Diagnostic> {
        let line = self.line();
        let anchor = self.anchor();
        self.expect(&Tok::Class)?;
        let name = match self.bump() {
            Tok::Ident(n) => n,
            t => {
                return Err(perr(
                    line,
                    &format!("expected a class name after `class`, found {:?}", t),
                ))
            }
        };
        let mut base = None;
        if matches!(self.peek(), Tok::LParen) {
            self.bump();
            base = Some(match self.bump() {
                Tok::Ident(n) => n,
                t => {
                    return Err(perr(
                        line,
                        &format!("expected a base class name, found {:?}", t),
                    ))
                }
            });
            if matches!(self.peek(), Tok::Comma) {
                return Err(perr(
                    line,
                    "multiple inheritance is not supported (single inheritance only, D4)",
                ));
            }
            self.expect(&Tok::RParen)?;
        }
        self.expect(&Tok::Colon)?;
        // Body: an indented block of defs and attribute assignments.
        if !matches!(self.peek(), Tok::Newline) {
            return Err(perr(
                line,
                "a class body must be an indented block in v1 (no one-line class suites)",
            ));
        }
        self.bump();
        self.expect(&Tok::Indent)?;
        let mut methods: Vec<Func> = Vec::new();
        let mut attrs: Vec<(String, Expr)> = Vec::new();
        while !matches!(self.peek(), Tok::Dedent | Tok::Eof) {
            if matches!(self.peek(), Tok::Newline) {
                self.bump();
                continue;
            }
            if matches!(self.peek(), Tok::Def) {
                methods.push(self.parse_func(false)?);
                continue;
            }
            if matches!(self.peek(), Tok::Pass) {
                self.bump();
                self.expect(&Tok::Newline)?;
                continue;
            }
            // Attribute assignment: Ident = expr, nothing else.
            let aline = self.line();
            let aname = match self.bump() {
                Tok::Ident(n) => n,
                t => {
                    return Err(perr(
                        aline,
                        &format!(
                            "a class body may contain only method definitions and attribute assignments, found {:?}",
                            t
                        ),
                    ))
                }
            };
            if !matches!(self.peek(), Tok::Assign) {
                return Err(perr(
                    aline,
                    "a class body may contain only method definitions and attribute assignments (`name = value`)",
                ));
            }
            self.bump();
            let value = self.parse_expr()?;
            self.expect(&Tok::Newline)?;
            attrs.push((aname, value));
        }
        self.expect(&Tok::Dedent)?;
        Ok(ClassDef { name, base, methods, attrs, line, anchor })
    }

    fn parse_func(&mut self, is_async: bool) -> Result<Func, Diagnostic> {
        let line = self.line();
        let anchor = self.anchor();
        if is_async {
            self.expect(&Tok::Async)?;
        }
        self.expect(&Tok::Def)?;
        let name = match self.bump() {
            Tok::Ident(n) => n,
            t => return Err(perr(line, &format!("expected function name, found {:?}", t))),
        };
        self.expect(&Tok::LParen)?;
        let mut params: Vec<(String, Option<Expr>)> = Vec::new();
        let mut vararg: Option<String> = None;
        let mut kwarg: Option<String> = None;
        let mut seen_default = false;
        if !matches!(self.peek(), Tok::RParen) {
            loop {
                if matches!(self.peek(), Tok::Star) {
                    self.bump();
                    if kwarg.is_some() {
                        return Err(perr(line, "parameter follows **kwargs parameter"));
                    }
                    match self.bump() {
                        Tok::Ident(n) => {
                            if vararg.is_some() {
                                return Err(perr(line, "duplicate *args parameter"));
                            }
                            vararg = Some(n);
                        }
                        _ => {
                            return Err(perr(
                                line,
                                "a bare * (keyword-only marker) is not supported yet (D3 later item)",
                            ))
                        }
                    }
                } else if matches!(self.peek(), Tok::Pow) {
                    self.bump();
                    match self.bump() {
                        Tok::Ident(n) => {
                            kwarg = Some(n);
                        }
                        t => {
                            return Err(perr(
                                line,
                                &format!("expected **kwargs parameter name, found {:?}", t),
                            ))
                        }
                    }
                } else {
                    if kwarg.is_some() {
                        return Err(perr(line, "parameter follows **kwargs parameter"));
                    }
                    if vararg.is_some() {
                        return Err(perr(
                            line,
                            "keyword-only parameters after *args are not supported yet (D3 later item)",
                        ));
                    }
                    let pname = match self.bump() {
                        Tok::Ident(n) => n,
                        t => {
                            return Err(perr(
                                line,
                                &format!("expected parameter name, found {:?} (annotations are not supported)", t),
                            ))
                        }
                    };
                    let default = if matches!(self.peek(), Tok::Assign) {
                        self.bump();
                        seen_default = true;
                        Some(self.parse_expr()?)
                    } else {
                        if seen_default {
                            return Err(perr(
                                line,
                                &format!("non-default parameter `{}` follows a default parameter", pname),
                            ));
                        }
                        None
                    };
                    params.push((pname, default));
                }
                if matches!(self.peek(), Tok::Comma) {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.expect(&Tok::RParen)?;
        self.expect(&Tok::Colon)?;
        let body = self.parse_block()?;
        Ok(Func { name, params, vararg, kwarg, is_async, body, line, anchor })
    }

    /// After a colon: either an indented block or a single simple statement
    /// on the same line (v0.1 supports the block form; one-liners too).
    fn parse_block(&mut self) -> Result<Vec<Stmt>, Diagnostic> {
        if matches!(self.peek(), Tok::Newline) {
            self.bump();
            self.expect(&Tok::Indent)?;
            let mut stmts = Vec::new();
            while !matches!(self.peek(), Tok::Dedent | Tok::Eof) {
                if matches!(self.peek(), Tok::Newline) {
                    self.bump();
                    continue;
                }
                stmts.push(self.parse_stmt()?);
            }
            self.expect(&Tok::Dedent)?;
            if stmts.is_empty() {
                return Err(perr(self.line(), "empty block; use pass"));
            }
            Ok(stmts)
        } else {
            let s = self.parse_simple_stmt()?;
            self.expect(&Tok::Newline)?;
            Ok(vec![s])
        }
    }

    fn parse_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        // `match` is a soft keyword: a statement starting with
        // the identifier `match` is a match statement only when
        // a subject expression and a colon follow.
        if matches!(self.peek(), Tok::Ident(n) if n == "match") && self.looks_like_match() {
            return self.parse_match();
        }
        match self.peek().clone() {
            Tok::If => self.parse_if(),
            Tok::Try => self.parse_try(),
            Tok::With => self.parse_with(),
            // D8: a def nested inside a function body. The
            // closure-conversion pass lifts it to a synthetic
            // top-level def; `async def` parses the same way
            // and is refused there (D8 v1).
            Tok::Def => {
                let f = self.parse_func(false)?;
                Ok(Stmt::Def(f))
            }
            Tok::Async => {
                let f = self.parse_func(true)?;
                Ok(Stmt::Def(f))
            }
            Tok::Import | Tok::From => Err(perr(
                self.line(),
                "import is only allowed at the top level of a module in v1 (D6)",
            )),
            Tok::While => {
                let line = self.line();
                let anchor = self.anchor();
                self.bump();
                let cond = self.parse_condition_expr()?;
                self.expect(&Tok::Colon)?;
                let body = self.parse_block()?;
                let els = self.parse_loop_else()?;
                Ok(Stmt::While { cond, body, els, line, anchor })
            }
            Tok::For => self.parse_for(),
            _ => {
                let s = self.parse_simple_stmt()?;
                self.expect(&Tok::Newline)?;
                Ok(s)
            }
        }
    }

    fn parse_if(&mut self) -> Result<Stmt, Diagnostic> {
        let line = self.line();
        let anchor = self.anchor();
        self.expect(&Tok::If)?;
        let cond = self.parse_condition_expr()?;
        self.expect(&Tok::Colon)?;
        let body = self.parse_block()?;
        let mut arms = vec![(cond, body)];
        let mut els = None;
        loop {
            // elif / else may follow after optional blank lines.
            let save = self.pos;
            self.skip_newlines();
            match self.peek().clone() {
                Tok::Elif => {
                    self.bump();
                    let c = self.parse_condition_expr()?;
                    self.expect(&Tok::Colon)?;
                    let b = self.parse_block()?;
                    arms.push((c, b));
                }
                Tok::Else => {
                    self.bump();
                    self.expect(&Tok::Colon)?;
                    els = Some(self.parse_block()?);
                    break;
                }
                _ => {
                    self.pos = save;
                    break;
                }
            }
        }
        Ok(Stmt::If { arms, els, line, anchor })
    }

    fn parse_for(&mut self) -> Result<Stmt, Diagnostic> {
        let line = self.line();
        let anchor = self.anchor();
        self.expect(&Tok::For)?;
        let mut vars = Vec::new();
        loop {
            match self.bump() {
                Tok::Ident(n) => vars.push(n),
                t => return Err(perr(line, &format!("expected loop variable, found {:?}", t))),
            }
            if matches!(self.peek(), Tok::Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(&Tok::In)?;
        // `for x in <list variable>` (v0.7, D2): anything that is not
        // the range(...) form parses as a general iteration.
        if !matches!(self.peek(), Tok::Ident(n) if n == "range") {
            let iter = self.parse_expr()?;
            self.expect(&Tok::Colon)?;
            let body = self.parse_block()?;
            if contains_continue(&body) {
                return Err(perr(
                    line,
                    "continue inside a for-each loop is not supported in v0.7 (use while)",
                ));
            }
            let els = self.parse_loop_else()?;
            return Ok(Stmt::ForEach { vars, iter, body, els, line, anchor });
        }
        if vars.len() > 1 {
            return Err(perr(
                line,
                "multiple loop targets need a sequence iterable (range() yields plain ints)",
            ));
        }
        let var = vars.into_iter().next().unwrap();
        match self.bump() {
            Tok::Ident(n) if n == "range" => {}
            t => {
                return Err(perr(
                    line,
                    &format!("for loops iterate range(...) or a list variable, found {:?}", t),
                ))
            }
        }
        self.expect(&Tok::LParen)?;
        let first = self.parse_expr()?;
        let (start, stop) = if matches!(self.peek(), Tok::Comma) {
            self.bump();
            let stop = self.parse_expr()?;
            if matches!(self.peek(), Tok::Comma) {
                return Err(perr(line, "range() with a step is not supported in v0.1"));
            }
            (Some(first), stop)
        } else {
            (None, first)
        };
        self.expect(&Tok::RParen)?;
        self.expect(&Tok::Colon)?;
        let body = self.parse_block()?;
        // `continue` inside a for body would skip the increment in the
        // while-lowering; reject loudly instead of miscompiling.
        if contains_continue(&body) {
            return Err(perr(
                line,
                "continue inside a for-range loop is not supported in v0.1 (use while)",
            ));
        }
        let els = self.parse_loop_else()?;
        Ok(Stmt::ForRange { var, start, stop, body, els, line, anchor })
    }

    /// `import a[.b]` / `from a[.b] import x [as y], ...` at
    /// the top level (D6). Module aliasing (`import a as b`)
    /// and star imports are refused loudly in v1.
    fn parse_import(&mut self) -> Result<Import, Diagnostic> {
        let line = self.line();
        if matches!(self.peek(), Tok::Import) {
            self.bump();
            let module = self.parse_module_path(line)?;
            if matches!(self.peek(), Tok::Ident(n) if n == "as") {
                return Err(perr(
                    line,
                    "module aliasing (`import a as b`) is not supported yet (D6 v1)",
                ));
            }
            self.expect(&Tok::Newline)?;
            return Ok(Import { module, names: None, line });
        }
        self.expect(&Tok::From)?;
        let module = self.parse_module_path(line)?;
        self.expect(&Tok::Import)?;
        if matches!(self.peek(), Tok::Star) {
            return Err(perr(line, "star imports (`from a import *`) are not supported (D6 v1)"));
        }
        let mut names = Vec::new();
        loop {
            let name = match self.bump() {
                Tok::Ident(n) => n,
                t => {
                    return Err(perr(
                        line,
                        &format!("expected a name in the from-import list, found {:?}", t),
                    ))
                }
            };
            let alias = if matches!(self.peek(), Tok::Ident(n) if n == "as") {
                self.bump();
                match self.bump() {
                    Tok::Ident(n) => Some(n),
                    t => {
                        return Err(perr(
                            line,
                            &format!("expected an alias name after `as`, found {:?}", t),
                        ))
                    }
                }
            } else {
                None
            };
            names.push((name, alias));
            if matches!(self.peek(), Tok::Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(&Tok::Newline)?;
        Ok(Import { module, names: Some(names), line })
    }

    /// A dotted module path: `a` or `a.b` or `a.b.c`.
    fn parse_module_path(&mut self, line: u32) -> Result<String, Diagnostic> {
        let mut parts = Vec::new();
        loop {
            match self.bump() {
                Tok::Ident(n) => parts.push(n),
                t => {
                    return Err(perr(
                        line,
                        &format!("expected a module name, found {:?}", t),
                    ))
                }
            }
            if matches!(self.peek(), Tok::Dot) {
                self.bump();
            } else {
                break;
            }
        }
        Ok(parts.join("."))
    }

    /// Soft-keyword lookahead for `match`: the current token
    /// is the identifier `match`; it is a match statement iff
    /// a subject expression and a colon follow. The token
    /// position is restored either way.
    fn looks_like_match(&mut self) -> bool {
        let save = self.pos;
        self.bump();
        let ok = match self.parse_expr() {
            Ok(_) => matches!(self.peek(), Tok::Colon),
            Err(_) => false,
        };
        self.pos = save;
        ok
    }

    /// `match <subject>: (case <pattern> [if <guard>]: block)+`
    /// (D7). Case bodies are indented blocks, as in Python.
    fn parse_match(&mut self) -> Result<Stmt, Diagnostic> {
        let line = self.line();
        let anchor = self.anchor();
        self.bump(); // the `match` identifier
        let subject = self.parse_expr()?;
        self.expect(&Tok::Colon)?;
        if !matches!(self.peek(), Tok::Newline) {
            return Err(perr(
                line,
                "a match body must be an indented block in v1 (no one-line match suites)",
            ));
        }
        self.bump();
        self.expect(&Tok::Indent)?;
        let mut cases: Vec<MatchCase> = Vec::new();
        while !matches!(self.peek(), Tok::Dedent | Tok::Eof) {
            if matches!(self.peek(), Tok::Newline) {
                self.bump();
                continue;
            }
            let cline = self.line();
            let canchor = self.anchor();
            match self.bump() {
                Tok::Ident(n) if n == "case" => {}
                t => {
                    return Err(perr(
                        cline,
                        &format!("expected `case` in a match block, found {:?}", t),
                    ))
                }
            }
            let pattern = self.parse_pattern()?;
            let guard = if matches!(self.peek(), Tok::If) {
                self.bump();
                Some(self.parse_condition_expr()?)
            } else {
                None
            };
            self.expect(&Tok::Colon)?;
            let body = self.parse_block()?;
            cases.push(MatchCase { pattern, guard, body, line: cline, anchor: canchor });
        }
        self.expect(&Tok::Dedent)?;
        if cases.is_empty() {
            return Err(perr(line, "a match statement needs at least one case"));
        }
        Ok(Stmt::Match { subject, cases, line, anchor })
    }

    /// One match pattern (D7 v1 subset). OR alternatives are
    /// split on BitOr at this level; a sequence pattern is a
    /// bracketed or parenthesised comma list; anything else is
    /// a single atom.
    fn parse_pattern(&mut self) -> Result<Pattern, Diagnostic> {
        let line = self.line();
        let first = self.parse_pattern_atom()?;
        if matches!(self.peek(), Tok::BitOr) {
            let mut alts = vec![first];
            while matches!(self.peek(), Tok::BitOr) {
                self.bump();
                alts.push(self.parse_pattern_atom()?);
            }
            return Ok(Pattern::Or(alts));
        }
        Ok(first)
    }

    fn parse_pattern_atom(&mut self) -> Result<Pattern, Diagnostic> {
        let line = self.line();
        match self.peek().clone() {
            Tok::LBracket => {
                self.bump();
                let pats = self.parse_pattern_seq(&Tok::RBracket)?;
                Ok(Pattern::Seq(pats))
            }
            Tok::LParen => {
                self.bump();
                if matches!(self.peek(), Tok::RParen) {
                    self.bump();
                    return Ok(Pattern::Seq(Vec::new()));
                }
                let first = self.parse_pattern()?;
                if matches!(self.peek(), Tok::Comma) {
                    let mut pats = vec![first];
                    while matches!(self.peek(), Tok::Comma) {
                        self.bump();
                        if matches!(self.peek(), Tok::RParen) {
                            break;
                        }
                        pats.push(self.parse_pattern()?);
                    }
                    self.expect(&Tok::RParen)?;
                    return Ok(Pattern::Seq(pats));
                }
                self.expect(&Tok::RParen)?;
                Ok(first)
            }
            Tok::Int(v) => {
                self.bump();
                Ok(Pattern::Lit(Expr::Int(v)))
            }
            Tok::BigInt(s) => {
                self.bump();
                Ok(Pattern::Lit(Expr::BigInt(s)))
            }
            Tok::Float(v) => {
                self.bump();
                Ok(Pattern::Lit(Expr::Float(v)))
            }
            Tok::Str(s) => {
                self.bump();
                Ok(Pattern::Lit(Expr::Str(s)))
            }
            Tok::Bytes(bs) => {
                self.bump();
                Ok(Pattern::Lit(Expr::Bytes(bs)))
            }
            Tok::True => {
                self.bump();
                Ok(Pattern::Lit(Expr::Bool(true)))
            }
            Tok::False => {
                self.bump();
                Ok(Pattern::Lit(Expr::Bool(false)))
            }
            Tok::NoneLit => {
                self.bump();
                Ok(Pattern::Lit(Expr::NoneLit))
            }
            Tok::Minus => {
                self.bump();
                match self.bump() {
                    Tok::Int(v) => Ok(Pattern::Lit(Expr::Int(-v))),
                    // Negative pattern: fold the sign into the digit
                    // string (the only place Expr::BigInt may carry
                    // '-'; the runtime parser __psa_big_from_psastr
                    // accepts it and the norm gate demotes -2^63 back
                    // to a tag-0 value).
                    Tok::BigInt(s) => Ok(Pattern::Lit(Expr::BigInt(format!("-{s}")))),
                    Tok::Float(v) => Ok(Pattern::Lit(Expr::Float(-v))),
                    t => Err(perr(
                        line,
                        &format!("expected a number after `-` in a pattern, found {:?}", t),
                    )),
                }
            }
            Tok::Ident(n) => {
                self.bump();
                if n == "_" {
                    return Ok(Pattern::Wildcard);
                }
                if matches!(self.peek(), Tok::Dot) {
                    return Err(perr(
                        line,
                        "value patterns (dotted names) are not supported in v1 (D7); a bare name is a capture",
                    ));
                }
                if matches!(self.peek(), Tok::LParen) {
                    return Err(perr(
                        line,
                        "class patterns are not supported in v1 (D7)",
                    ));
                }
                if matches!(self.peek(), Tok::Assign) {
                    return Err(perr(
                        line,
                        "pattern assignment (`as` patterns use the `as` keyword, and walrus patterns are not supported) in v1 (D7)",
                    ));
                }
                Ok(Pattern::Capture(n))
            }
            Tok::Star => Err(perr(
                line,
                "star patterns are not supported in v1 (D7)",
            )),
            Tok::LBrace => Err(perr(
                line,
                "mapping patterns are not supported in v1 (D7)",
            )),
            t => Err(perr(
                line,
                &format!("unsupported pattern starting with {:?}", t),
            )),
        }
    }

    fn parse_pattern_seq(&mut self, close: &Tok) -> Result<Vec<Pattern>, Diagnostic> {
        let mut pats = Vec::new();
        if matches!(self.peek(), t if t == close) {
            self.bump();
            return Ok(pats);
        }
        loop {
            pats.push(self.parse_pattern()?);
            if matches!(self.peek(), Tok::Comma) {
                self.bump();
                if matches!(self.peek(), t if t == close) {
                    break;
                }
            } else {
                break;
            }
        }
        self.expect(close)?;
        Ok(pats)
    }

    /// `try: block (except ... : block)+ [else: block]
    /// [finally: block]` (D6). A try needs at least one except
    /// or a finally; else requires an except; a bare except
    /// must be the last handler (CPython's rules).
    fn parse_try(&mut self) -> Result<Stmt, Diagnostic> {
        let line = self.line();
        let anchor = self.anchor();
        self.expect(&Tok::Try)?;
        self.expect(&Tok::Colon)?;
        let body = self.parse_block()?;
        let mut handlers: Vec<ExcHandler> = Vec::new();
        let mut els: Option<Vec<Stmt>> = None;
        let mut finally: Option<Vec<Stmt>> = None;
        loop {
            let save = self.pos;
            self.skip_newlines();
            match self.peek().clone() {
                Tok::Except if finally.is_none() && els.is_none() => {
                    let hanchor = self.anchor();
                    self.bump();
                    let hline = self.line();
                    let mut classes: Vec<String> = Vec::new();
                    if matches!(self.peek(), Tok::LParen) {
                        self.bump();
                        loop {
                            match self.bump() {
                                Tok::Ident(n) => classes.push(n),
                                t => {
                                    return Err(perr(
                                        hline,
                                        &format!(
                                            "expected an exception class name in the except tuple, found {:?}",
                                            t
                                        ),
                                    ))
                                }
                            }
                            if matches!(self.peek(), Tok::Comma) {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                        self.expect(&Tok::RParen)?;
                    } else if matches!(self.peek(), Tok::Ident(_)) {
                        match self.bump() {
                            Tok::Ident(n) => classes.push(n),
                            _ => unreachable!(),
                        }
                    }
                    let bind = if matches!(self.peek(), Tok::Ident(n) if n == "as") {
                        self.bump();
                        match self.bump() {
                            Tok::Ident(n) => Some(n),
                            t => {
                                return Err(perr(
                                    hline,
                                    &format!(
                                        "expected a name after `as` in the except clause, found {:?}",
                                        t
                                    ),
                                ))
                            }
                        }
                    } else {
                        None
                    };
                    self.expect(&Tok::Colon)?;
                    let hbody = self.parse_block()?;
                    handlers.push(ExcHandler { classes, bind, body: hbody, line: hline, anchor: hanchor });
                }
                Tok::Else if finally.is_none() && els.is_none() => {
                    self.bump();
                    self.expect(&Tok::Colon)?;
                    els = Some(self.parse_block()?);
                }
                Tok::Finally if finally.is_none() => {
                    self.bump();
                    self.expect(&Tok::Colon)?;
                    finally = Some(self.parse_block()?);
                    break;
                }
                _ => {
                    self.pos = save;
                    break;
                }
            }
        }
        if handlers.is_empty() && finally.is_none() {
            return Err(perr(line, "a try statement needs at least one except clause or a finally block"));
        }
        if els.is_some() && handlers.is_empty() {
            return Err(perr(line, "a try else clause needs at least one except clause"));
        }
        if let Some(pos) = handlers.iter().position(|h| h.classes.is_empty()) {
            if pos != handlers.len() - 1 {
                return Err(perr(line, "a bare `except:` clause must be the last except clause"));
            }
        }
        Ok(Stmt::Try { body, handlers, els, finally, line, anchor })
    }

    /// `with <expr> [as name] (, <expr> [as name])* : block`
    /// (D6).
    fn parse_with(&mut self) -> Result<Stmt, Diagnostic> {
        let line = self.line();
        let anchor = self.anchor();
        self.expect(&Tok::With)?;
        let mut items: Vec<(Expr, Option<String>)> = Vec::new();
        loop {
            let e = self.parse_expr()?;
            let bind = if matches!(self.peek(), Tok::Ident(n) if n == "as") {
                self.bump();
                match self.bump() {
                    Tok::Ident(n) => Some(n),
                    t => {
                        return Err(perr(
                            line,
                            &format!("expected a name after `as` in the with item, found {:?}", t),
                        ))
                    }
                }
            } else {
                None
            };
            items.push((e, bind));
            if matches!(self.peek(), Tok::Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(&Tok::Colon)?;
        let body = self.parse_block()?;
        Ok(Stmt::With { items, body, line, anchor })
    }

    fn parse_simple_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        let line = self.line();
        let anchor = self.anchor();
        match self.peek().clone() {
            Tok::Yield => {
                self.bump();
                if matches!(self.peek(), Tok::Newline) {
                    Ok(Stmt::Yield { value: None, line, anchor })
                } else {
                    let v = self.parse_expr()?;
                    Ok(Stmt::Yield { value: Some(v), line, anchor })
                }
            }
            Tok::Raise => {
                self.bump();
                if matches!(self.peek(), Tok::Newline) {
                    Ok(Stmt::Raise { value: None, line, anchor })
                } else {
                    let v = self.parse_expr()?;
                    if matches!(self.peek(), Tok::From) {
                        return Err(perr(
                            line,
                            "exception chaining (`raise X from Y`) is not supported yet (D6 v1)",
                        ));
                    }
                    Ok(Stmt::Raise { value: Some(v), line, anchor })
                }
            }
            Tok::Return => {
                self.bump();
                if matches!(self.peek(), Tok::Newline) {
                    Ok(Stmt::Return { value: None, line, anchor })
                } else {
                    let v = self.parse_expr()?;
                    if matches!(self.peek(), Tok::Comma) {
                        // `return a, b` returns a tuple (D5).
                        let mut elems = vec![v];
                        while matches!(self.peek(), Tok::Comma) {
                            self.bump();
                            if matches!(self.peek(), Tok::Newline) {
                                break;
                            }
                            elems.push(self.parse_expr()?);
                        }
                        Ok(Stmt::Return {
                            value: Some(Expr::TupleLit(elems)),
                            line,
                            anchor,
                        })
                    } else {
                        Ok(Stmt::Return { value: Some(v), line, anchor })
                    }
                }
            }
            Tok::Break => {
                self.bump();
                Ok(Stmt::Break { line, anchor })
            }
            Tok::Continue => {
                self.bump();
                Ok(Stmt::Continue { line, anchor })
            }
            Tok::Pass => {
                self.bump();
                Ok(Stmt::Pass { line, anchor })
            }
            // D8: `global a, b` / `nonlocal a, b` (soft
            // keywords, like assert/del). Consumed by the
            // closure-conversion pass.
            Tok::Ident(name) if name == "global" || name == "nonlocal" => {
                let is_global = name == "global";
                self.bump();
                let mut names = Vec::new();
                loop {
                    match self.bump() {
                        Tok::Ident(n) => names.push(n),
                        t => {
                            return Err(perr(
                                line,
                                &format!(
                                    "expected a name in a {} declaration, found {:?}",
                                    if is_global { "global" } else { "nonlocal" },
                                    t
                                ),
                            ))
                        }
                    }
                    if matches!(self.peek(), Tok::Comma) {
                        self.bump();
                        continue;
                    }
                    break;
                }
                if is_global {
                    Ok(Stmt::Global { names, line, anchor })
                } else {
                    Ok(Stmt::Nonlocal { names, line, anchor })
                }
            }
            Tok::Ident(name) if name == "assert" => {
                self.bump();
                let cond = self.parse_expr()?;
                // The message is a diagnostic label: it is carried
                // into the AST and written to stderr at the failure
                // site before panic(77040) (census section 0.6 --
                // it used to be parsed, validated and dropped).
                let mut msg = None;
                if matches!(self.peek(), Tok::Comma) {
                    self.bump();
                    match self.parse_expr()? {
                        Expr::Str(s) => msg = Some(s),
                        _ => {
                            return Err(perr(
                                line,
                                "assert message must be a string literal in v1 (it is a diagnostic label; psa panics with a code)",
                            ));
                        }
                    }
                }
                Ok(Stmt::Assert { cond, line, msg, anchor })
            }
            Tok::Ident(name) if name == "del" => {
                self.bump();
                // `del base.attr` (D4): a single attribute target,
                // recognised by the Ident (. Ident)+ lookahead.
                {
                    let t1 = self.toks.get(self.pos + 1).map(|t| &t.tok);
                    let t2 = self.toks.get(self.pos + 2).map(|t| &t.tok);
                    let t3 = self.toks.get(self.pos + 3).map(|t| &t.tok);
                    if matches!(self.peek(), Tok::Ident(_))
                        && matches!(t1, Some(Tok::Dot))
                        && matches!(t2, Some(Tok::Ident(_)))
                    {
                        if matches!(t3, Some(Tok::Comma)) {
                            return Err(perr(
                                line,
                                "del of an attribute cannot be combined with other del targets in v1",
                            ));
                        }
                        // The first name token, for the chain's span start (P0b).
                        let first_anchor = self.anchor();
                        let first = match self.bump() {
                            Tok::Ident(n) => n,
                            _ => unreachable!(),
                        };
                        let first_span = ExprSpan::from_name(first_anchor, &first);
                        let mut base = Expr::Name(first, first_span);
                        let mut attr: Option<String> = None;
                        while matches!(self.peek(), Tok::Dot) {
                            self.bump();
                            let attr_tok = self.cur_tok().clone();
                            let n = match self.bump() {
                                Tok::Ident(n) => n,
                                t => {
                                    return Err(perr(
                                        line,
                                        &format!(
                                            "expected an attribute name after . in del target, found {:?}",
                                            t
                                        ),
                                    ))
                                }
                            };
                            if let Some(prev) = attr.take() {
                                let span = self.span_to_ident_end(
                                    first_anchor, &attr_tok, &prev,
                                );
                                base = Expr::Attr(Box::new(base), prev, span);
                            }
                            attr = Some(n);
                        }
                        return Ok(Stmt::AttrDel {
                            base,
                            name: attr.expect("del attribute target"),
                            line,
                            anchor,
                        });
                    }
                }
                let mut targets = Vec::new();
                loop {
                    let tname = match self.bump() {
                        Tok::Ident(n) => n,
                        t => {
                            return Err(perr(
                                line,
                                &format!("del expects a name, found {:?}", t),
                            ))
                        }
                    };
                    let index = if matches!(self.peek(), Tok::LBracket) {
                        self.bump();
                        let ix = self.parse_expr()?;
                        self.expect(&Tok::RBracket)?;
                        Some(ix)
                    } else {
                        None
                    };
                    targets.push((tname, index));
                    if matches!(self.peek(), Tok::Comma) {
                        self.bump();
                        continue;
                    }
                    break;
                }
                Ok(Stmt::Del { targets, line, anchor })
            }
            Tok::Ident(name) => {
                // Multiple assignment: `a, b = v1, v2` (comma after the
                // first name) or chained `a = b = v` (name = name = ...).
                {
                    let t1 = self.toks.get(self.pos + 1).map(|t| &t.tok);
                    let t2 = self.toks.get(self.pos + 2).map(|t| &t.tok);
                    let t3 = self.toks.get(self.pos + 3).map(|t| &t.tok);
                    let multi = matches!(t1, Some(Tok::Comma));
                    let chained = matches!(t1, Some(Tok::Assign))
                        && matches!(t2, Some(Tok::Ident(_)))
                        && matches!(t3, Some(Tok::Assign));
                    if multi || chained {
                        let mut names = vec![name.clone()];
                        self.bump(); // first name
                        if multi {
                            while matches!(self.peek(), Tok::Comma) {
                                self.bump();
                                match self.bump() {
                                    Tok::Ident(n) => names.push(n),
                                    t => {
                                        return Err(perr(
                                            line,
                                            &format!(
                                                "expected a name in assignment targets, found {:?}",
                                                t
                                            ),
                                        ))
                                    }
                                }
                            }
                            self.expect(&Tok::Assign)?;
                        } else {
                            // chained: name (= name)* = value
                            while matches!(self.peek(), Tok::Assign)
                                && matches!(
                                    self.toks.get(self.pos + 1).map(|t| &t.tok),
                                    Some(Tok::Ident(_))
                                )
                                && matches!(
                                    self.toks.get(self.pos + 2).map(|t| &t.tok),
                                    Some(Tok::Assign)
                                )
                            {
                                self.bump(); // =
                                match self.bump() {
                                    Tok::Ident(n) => names.push(n),
                                    _ => unreachable!(),
                                }
                            }
                            self.expect(&Tok::Assign)?;
                        }
                        let mut values = vec![self.parse_expr()?];
                        while matches!(self.peek(), Tok::Comma) {
                            self.bump();
                            values.push(self.parse_expr()?);
                        }
                        // `a, b = <one sequence>` unpacks it (D5);
                        // the chained form keeps MultiAssign.
                        if multi && values.len() == 1 {
                            let value = values.into_iter().next().unwrap();
                            return Ok(Stmt::Unpack { names, value, line, anchor });
                        }
                        return Ok(Stmt::MultiAssign { names, values, line, anchor });
                    }
                }
                // Index assignment: name[expr] (=|+=|-=|*=) expr — try it
                // with backtracking; a bare name[expr] expression
                // statement falls through to expression parsing.
                if matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::LBracket)) {
                    let save = self.pos;
                    self.bump(); // name
                    self.bump(); // [
                    if let Ok(index) = self.parse_expr() {
                        if matches!(self.peek(), Tok::RBracket) {
                            self.bump();
                            let op = match self.peek() {
                                Tok::Assign => Some(None),
                                t => aug_op(t).map(Some),
                            };
                            if let Some(op) = op {
                                self.bump();
                                let value = self.parse_expr()?;
                                return Ok(Stmt::IndexAssign {
                                    name,
                                    index,
                                    op,
                                    value,
                                    line,
                                    anchor,
                                });
                            }
                        }
                    }
                    self.pos = save;
                }
                // Attribute assignment: name (.name)+ (=|augop) expr
                // (D4) — try with backtracking; a bare attribute
                // expression statement falls through.
                if matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::Dot)) {
                    let save = self.pos;
                    // The first name token, for the chain's span start (P0b).
                    let base_anchor = self.anchor();
                    self.bump(); // first name
                    let mut base =
                        Expr::Name(name.clone(), ExprSpan::from_name(base_anchor, &name));
                    let mut attr: Option<String> = None;
                    let mut shape_ok = true;
                    while matches!(self.peek(), Tok::Dot) {
                        self.bump();
                        let attr_tok = self.cur_tok().clone();
                        match self.bump() {
                            Tok::Ident(n) => {
                                if let Some(prev) = attr.take() {
                                    let span = self.span_to_ident_end(
                                        base_anchor, &attr_tok, &prev,
                                    );
                                    base = Expr::Attr(Box::new(base), prev, span);
                                }
                                attr = Some(n);
                            }
                            _ => {
                                shape_ok = false;
                                break;
                            }
                        }
                    }
                    if shape_ok {
                        if let Some(aname) = attr {
                            let op = match self.peek() {
                                Tok::Assign => Some(None),
                                t => aug_op(t).map(Some),
                            };
                            if let Some(op) = op {
                                self.bump();
                                let value = self.parse_expr()?;
                                return Ok(Stmt::AttrAssign {
                                    base,
                                    name: aname,
                                    op,
                                    value,
                                    line,
                                    anchor,
                                });
                            }
                        }
                    }
                    self.pos = save;
                }
                // Look ahead for assignment forms.
                let next = self.toks.get(self.pos + 1).map(|t| t.tok.clone());
                match next {
                    Some(Tok::Assign) => {
                        self.bump();
                        self.bump();
                        let value = self.parse_expr()?;
                        if matches!(self.peek(), Tok::Comma) {
                            // `x = a, b` binds a tuple (D5).
                            let mut elems = vec![value];
                            while matches!(self.peek(), Tok::Comma) {
                                self.bump();
                                if matches!(self.peek(), Tok::Newline) {
                                    break;
                                }
                                elems.push(self.parse_expr()?);
                            }
                            return Ok(Stmt::Assign {
                                name,
                                value: Expr::TupleLit(elems),
                                line,
                                anchor,
                            });
                        }
                        Ok(Stmt::Assign { name, value, line, anchor })
                    }
                    Some(t) if aug_op(&t).is_some() => {
                        self.bump(); // name
                        let optok = self.bump();
                        let op = aug_op(&optok).unwrap();
                        let value = self.parse_expr()?;
                        Ok(Stmt::AugAssign { name, op, value, line, anchor })
                    }
                    _ => {
                        let e = self.parse_expr()?;
                        Ok(Stmt::ExprStmt { value: e, line, anchor })
                    }
                }
            }
            _ => {
                let e = self.parse_expr()?;
                Ok(Stmt::ExprStmt { value: e, line, anchor })
            }
        }
    }

    pub fn parse_expr(&mut self) -> Result<Expr, Diagnostic> {
        if matches!(self.peek(), Tok::Lambda) {
            return self.parse_lambda();
        }
        self.parse_ternary()
    }

    /// A condition position (if/elif/while condition, a
    /// comprehension `if` clause, a match guard): CPython allows
    /// a BARE assignment expression here (`while line := next():`)
    /// where other expression positions demand parentheses. A
    /// bare walrus is exactly `Name := <full expression>`.
    /// Also used for call arguments and subscript parts, which
    /// are named-expression positions in CPython's grammar
    /// (`f(x := 5)`, `a[x := 5]` are legal unparenthesised).
    fn parse_condition_expr(&mut self) -> Result<Expr, Diagnostic> {
        if matches!(self.peek(), Tok::Ident(_)) && matches!(self.peek2(), Tok::Walrus) {
            let name = match self.bump() {
                Tok::Ident(n) => n,
                _ => unreachable!(),
            };
            self.bump(); // :=
            let value = self.parse_expr()?;
            return Ok(Expr::Walrus { name, value: Box::new(value) });
        }
        self.parse_expr()
    }

    /// The optional `else:` clause after a loop body (loop-else:
    /// it runs when the loop finishes without `break`). Mirrors
    /// parse_if's continuation scan: blank lines are skipped for
    /// the lookahead and the position is restored when no else
    /// follows. An else belonging to an enclosing construct is
    /// always shielded by a Dedent token, so it is never stolen.
    fn parse_loop_else(&mut self) -> Result<Option<Vec<Stmt>>, Diagnostic> {
        let save = self.pos;
        self.skip_newlines();
        if matches!(self.peek(), Tok::Else) {
            self.bump();
            self.expect(&Tok::Colon)?;
            let body = self.parse_block()?;
            Ok(Some(body))
        } else {
            self.pos = save;
            Ok(None)
        }
    }

    /// `lambda a, b: body` (D3). Plain identifier parameters
    /// only in v1: defaults and *args/**kwargs inside a lambda
    /// are refused loudly here (the desugared def could carry
    /// them, but a lambda's signature cannot be pinned from a
    /// direct call site, so they ride on a later slice).
    fn parse_lambda(&mut self) -> Result<Expr, Diagnostic> {
        let line = self.line();
        self.bump();
        let mut params: Vec<String> = Vec::new();
        if !matches!(self.peek(), Tok::Colon) {
            loop {
                match self.bump() {
                    Tok::Ident(n) => params.push(n),
                    Tok::Star | Tok::Pow => {
                        return Err(perr(
                            line,
                            "*args/**kwargs parameters in a lambda are not supported yet (D3 later item)",
                        ))
                    }
                    t => {
                        return Err(perr(
                            line,
                            &format!("expected lambda parameter name, found {:?}", t),
                        ))
                    }
                }
                if matches!(self.peek(), Tok::Assign) {
                    return Err(perr(
                        line,
                        "default values in a lambda are not supported yet (D3 later item)",
                    ));
                }
                if matches!(self.peek(), Tok::Comma) {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.expect(&Tok::Colon)?;
        let body = self.parse_expr()?;
        Ok(Expr::Lambda { params, body: Box::new(body), line })
    }

    /// Python conditional expression: `then if cond else els`. The `then`
    /// operand is an or-expression; the else branch is a full expression
    /// (so `a if c1 else b if c2 else d` chains like elif).
    fn parse_ternary(&mut self) -> Result<Expr, Diagnostic> {
        let then = self.parse_or()?;
        if matches!(self.peek(), Tok::If) {
            self.bump();
            let cond = self.parse_or()?;
            self.expect(&Tok::Else)?;
            let els = self.parse_expr()?;
            return Ok(Expr::IfExp {
                cond: Box::new(cond),
                then: Box::new(then),
                els: Box::new(els),
            });
        }
        Ok(then)
    }

    fn parse_or(&mut self) -> Result<Expr, Diagnostic> {
        let mut lhs = self.parse_and()?;
        while matches!(self.peek(), Tok::Or) {
            self.bump();
            let rhs = self.parse_and()?;
            lhs = Expr::Bin(Box::new(lhs), BinOp::Or, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr, Diagnostic> {
        let mut lhs = self.parse_not()?;
        while matches!(self.peek(), Tok::And) {
            self.bump();
            let rhs = self.parse_not()?;
            lhs = Expr::Bin(Box::new(lhs), BinOp::And, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_not(&mut self) -> Result<Expr, Diagnostic> {
        if matches!(self.peek(), Tok::Not) {
            self.bump();
            let e = self.parse_not()?;
            return Ok(Expr::Unary(UnOp::Not, Box::new(e)));
        }
        self.parse_cmp()
    }

    fn parse_cmp(&mut self) -> Result<Expr, Diagnostic> {
        let mut operands = vec![self.parse_bitor()?];
        let mut ops: Vec<BinOp> = Vec::new();
        loop {
            let not_in = matches!(self.peek(), Tok::Not)
                && matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::In));
            // `is not` is one operator (CPython): `is` directly
            // followed by `not`. The reverse order (`not is`) is
            // not an operator and stays a parse error downstream.
            let is_not = matches!(self.peek(), Tok::Is)
                && matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::Not));
            let op = match self.peek() {
                Tok::EqEq => BinOp::Eq,
                Tok::NotEq => BinOp::NotEq,
                Tok::Lt => BinOp::Lt,
                Tok::LtEq => BinOp::LtEq,
                Tok::Gt => BinOp::Gt,
                Tok::GtEq => BinOp::GtEq,
                Tok::In => BinOp::In,
                Tok::Is if is_not => {
                    self.bump();
                    BinOp::IsNot
                }
                Tok::Is => BinOp::Is,
                Tok::Not if not_in => {
                    self.bump();
                    BinOp::NotIn
                }
                _ => break,
            };
            ops.push(op);
            self.bump();
            operands.push(self.parse_bitor()?);
        }
        if ops.is_empty() {
            return Ok(operands.pop().unwrap());
        }
        // Chained comparison: `a < b < c` == `(a < b) and (b < c)`, with
        // the middle operands evaluated once in Python. psa evaluates
        // pure operands twice instead (same result); a middle operand
        // with side effects cannot be duplicated — reject it loudly.
        for mid in &operands[1..operands.len() - 1] {
            if !duplicable(mid) {
                return Err(perr(
                    self.line(),
                    "chained comparison with a call or computed value in the middle is not supported in D0 (it would be evaluated twice); bind it to a variable first",
                ));
            }
        }
        let mut acc: Option<Expr> = None;
        for i in 0..ops.len() {
            let pair = Expr::Bin(
                Box::new(operands[i].clone()),
                ops[i],
                Box::new(operands[i + 1].clone()),
            );
            acc = Some(match acc {
                None => pair,
                Some(a) => Expr::Bin(Box::new(a), BinOp::And, Box::new(pair)),
            });
        }
        Ok(acc.unwrap())
    }

    fn parse_bitor(&mut self) -> Result<Expr, Diagnostic> {
        let mut lhs = self.parse_bitxor()?;
        while matches!(self.peek(), Tok::BitOr) {
            self.bump();
            let rhs = self.parse_bitxor()?;
            lhs = Expr::Bin(Box::new(lhs), BinOp::BitOr, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_bitxor(&mut self) -> Result<Expr, Diagnostic> {
        let mut lhs = self.parse_bitand()?;
        while matches!(self.peek(), Tok::BitXor) {
            self.bump();
            let rhs = self.parse_bitand()?;
            lhs = Expr::Bin(Box::new(lhs), BinOp::BitXor, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_bitand(&mut self) -> Result<Expr, Diagnostic> {
        let mut lhs = self.parse_shift()?;
        while matches!(self.peek(), Tok::BitAnd) {
            self.bump();
            let rhs = self.parse_shift()?;
            lhs = Expr::Bin(Box::new(lhs), BinOp::BitAnd, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> Result<Expr, Diagnostic> {
        let mut lhs = self.parse_add()?;
        loop {
            let op = match self.peek() {
                Tok::Shl => BinOp::Shl,
                Tok::Shr => BinOp::Shr,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_add()?;
            lhs = Expr::Bin(Box::new(lhs), op, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_add(&mut self) -> Result<Expr, Diagnostic> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_mul()?;
            lhs = Expr::Bin(Box::new(lhs), op, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> Result<Expr, Diagnostic> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::FloorDiv => BinOp::FloorDiv,
                Tok::Percent => BinOp::Mod,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_unary()?;
            lhs = Expr::Bin(Box::new(lhs), op, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, Diagnostic> {
        if matches!(self.peek(), Tok::Await) {
            self.bump();
            let e = self.parse_unary()?;
            return Ok(Expr::Await(Box::new(e)));
        }
        if matches!(self.peek(), Tok::Minus) {
            self.bump();
            let e = self.parse_unary()?;
            return Ok(Expr::Unary(UnOp::Neg, Box::new(e)));
        }
        if matches!(self.peek(), Tok::Plus) {
            self.bump();
            return self.parse_unary();
        }
        if matches!(self.peek(), Tok::Tilde) {
            self.bump();
            let e = self.parse_unary()?;
            return Ok(Expr::Unary(UnOp::BitNot, Box::new(e)));
        }
        self.parse_power()
    }

    /// Python power: `primary ** factor`, right-associative, and binding
    /// tighter than a unary minus on its left (`-2 ** 2 == -4`), while the
    /// exponent itself may carry a sign (`2 ** -1`).
    fn parse_power(&mut self) -> Result<Expr, Diagnostic> {
        let base = self.parse_postfix()?;
        if matches!(self.peek(), Tok::Pow) {
            self.bump();
            let exp = self.parse_unary()?;
            return Ok(Expr::Bin(Box::new(base), BinOp::Pow, Box::new(exp)));
        }
        Ok(base)
    }

    fn parse_postfix(&mut self) -> Result<Expr, Diagnostic> {
        // Start anchor of the whole postfix chain (P0b): the first
        // token of the base atom. Every Call/Index/Attr built in
        // this call starts here; nested postfix calls capture their
        // own start the same way.
        let base_start = self.anchor();
        let mut e = self.parse_atom()?;
        loop {
            match self.peek() {
                Tok::LParen => {
                    self.bump();
                    let mut args = Vec::new();
                    let mut kwargs: Vec<(String, Expr)> = Vec::new();
                    if !matches!(self.peek(), Tok::RParen) {
                        loop {
                            if matches!(self.peek(), Tok::Star) || matches!(self.peek(), Tok::Pow) {
                                return Err(perr(
                                    self.line(),
                                    "*args/**kwargs unpacking at call sites is not supported yet (D3 later item)",
                                ));
                            }
                            if let Tok::Ident(kname) = self.peek().clone() {
                                if matches!(self.peek2(), Tok::Assign) {
                                    self.bump();
                                    self.bump();
                                    if kwargs.iter().any(|(k, _)| *k == kname) {
                                        return Err(perr(
                                            self.line(),
                                            &format!("keyword argument `{}` given twice", kname),
                                        ));
                                    }
                                    let v = self.parse_expr()?;
                                    kwargs.push((kname, v));
                                    if matches!(self.peek(), Tok::Comma) {
                                        self.bump();
                                    } else {
                                        break;
                                    }
                                    continue;
                                }
                            }
                            if !kwargs.is_empty() {
                                return Err(perr(
                                    self.line(),
                                    "positional argument follows keyword argument",
                                ));
                            }
                            // A positional argument is a named
                            // expression in CPython's grammar:
                            // `f(x := 5)` is legal bare. (Keyword
                            // values above stay parse_expr —
                            // `f(k=x := 5)` is a SyntaxError.)
                            args.push(self.parse_condition_expr()?);
                            if matches!(self.peek(), Tok::Comma) {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(&Tok::RParen)?;
                    let call_span = self.span_past(base_start, self.prev_tok());
                    e = Expr::Call(Box::new(e), args, kwargs, call_span);
                }
                Tok::LBracket => {
                    self.bump();
                    // Subscript parts are named expressions in
                    // CPython's grammar: `a[x := 5]` (and slice
                    // bounds) are legal unparenthesised.
                    let mut lo = None;
                    if !matches!(self.peek(), Tok::Colon) {
                        lo = Some(Box::new(self.parse_condition_expr()?));
                    }
                    if matches!(self.peek(), Tok::Colon) {
                        self.bump();
                        let mut hi = None;
                        if !matches!(self.peek(), Tok::Colon | Tok::RBracket) {
                            hi = Some(Box::new(self.parse_condition_expr()?));
                        }
                        let mut step = None;
                        if matches!(self.peek(), Tok::Colon) {
                            self.bump();
                            if !matches!(self.peek(), Tok::RBracket) {
                                step = Some(Box::new(self.parse_condition_expr()?));
                            }
                        }
                        self.expect(&Tok::RBracket)?;
                        e = Expr::Slice(Box::new(e), lo, hi, step);
                    } else {
                        self.expect(&Tok::RBracket)?;
                        let index_span = self.span_past(base_start, self.prev_tok());
                        e = Expr::Index(Box::new(e), lo.unwrap(), index_span);
                    }
                }
                Tok::Dot => {
                    self.bump();
                    // The attribute-name token, for the Attr span's
                    // end (P0b). Captured before the bump consumes it.
                    let method_tok = self.cur_tok().clone();
                    let method = match self.bump() {
                        Tok::Ident(n) => n,
                        // Keywords are legal attribute names in
                        // CPython (`obj.is`); `is` lexed as an
                        // identifier before 171-B, so keep it
                        // readable here.
                        Tok::Is => "is".to_string(),
                        t => {
                            return Err(perr(
                                self.line(),
                                &format!("expected an attribute name after ., found {:?}", t),
                            ))
                        }
                    };
                    if !matches!(self.peek(), Tok::LParen) {
                        // Attribute access (D4): `base.name`.
                        let attr_span =
                            self.span_to_ident_end(base_start, &method_tok, &method);
                        e = Expr::Attr(Box::new(e), method, attr_span);
                        continue;
                    }
                    self.expect(&Tok::LParen)?;
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Tok::RParen) {
                        loop {
                            // Method-call arguments are named
                            // expressions too (see the Call branch).
                            args.push(self.parse_condition_expr()?);
                            if matches!(self.peek(), Tok::Comma) {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(&Tok::RParen)?;
                    match e {
                        Expr::Name(n, _) => {
                            e = Expr::MethodCall { obj: n, method, args };
                        }
                        // D5: a method call on a general base
                        // (a string literal, a subscript, a call
                        // result) desugars to a call through the
                        // attribute, which the emitter routes to
                        // the builtin method set or getattr.
                        _ => {
                            let attr_span =
                                self.span_to_ident_end(base_start, &method_tok, &method);
                            let attr = Expr::Attr(Box::new(e), method, attr_span);
                            let call_span = self.span_past(base_start, self.prev_tok());
                            e = Expr::Call(Box::new(attr), args, Vec::new(), call_span);
                        }
                    }
                }
                _ => break,
            }
        }
        Ok(e)
    }

    /// Comprehension clauses after the element expression:
    /// `for v in iter [if cond] [for ...] ...` (D5). The loop
    /// target is a plain name in v1 (tuple targets are refused
    /// loudly). The iterable is an or-expression: an
    /// unparenthesised ternary there would swallow the first
    /// `if` clause, exactly as in CPython's grammar.
    fn parse_comp_clauses(&mut self) -> Result<Vec<CompClause>, Diagnostic> {
        let mut clauses = Vec::new();
        loop {
            if matches!(self.peek(), Tok::For) {
                self.bump();
                let var = match self.bump() {
                    Tok::Ident(n) => n,
                    t => {
                        return Err(perr(
                            self.line(),
                            &format!(
                                "comprehension target must be a plain name in v1, found {:?}",
                                t
                            ),
                        ))
                    }
                };
                if !matches!(self.bump(), Tok::In) {
                    return Err(perr(self.line(), "expected `in` in a comprehension clause"));
                }
                let iter = self.parse_or()?;
                clauses.push(CompClause::For { var, iter });
            } else if matches!(self.peek(), Tok::If) {
                self.bump();
                let cond = self.parse_condition_expr()?;
                clauses.push(CompClause::If(cond));
            } else {
                break;
            }
        }
        if !matches!(clauses.first(), Some(CompClause::For { .. })) {
            return Err(perr(self.line(), "a comprehension must start with a for clause"));
        }
        Ok(clauses)
    }

    fn parse_atom(&mut self) -> Result<Expr, Diagnostic> {
        let line = self.line();
        // Start anchor of this atom (P0b): the current token is the
        // atom's first token. Used for the Name span below; the
        // postfix layer captures its own start the same way.
        let start = self.anchor();
        match self.bump() {
            Tok::Int(v) => Ok(Expr::Int(v)),
            Tok::BigInt(s) => Ok(Expr::BigInt(s)),
            Tok::Float(v) => Ok(Expr::Float(v)),
            Tok::Str(s) => Ok(Expr::Str(s)),
            Tok::Bytes(bs) => Ok(Expr::Bytes(bs)),
            Tok::FStr(parts) => {
                let mut out = Vec::new();
                for p in parts {
                    out.push(parse_fstr_part(p, line)?);
                }
                Ok(Expr::FStr(out))
            }
            Tok::True => Ok(Expr::Bool(true)),
            Tok::False => Ok(Expr::Bool(false)),
            Tok::NoneLit => Ok(Expr::NoneLit),
            Tok::Ident(n) => {
                let span = ExprSpan::from_name(start, &n);
                Ok(Expr::Name(n, span))
            }
            Tok::LParen => {
                if matches!(self.peek(), Tok::RParen) {
                    self.bump();
                    return Ok(Expr::TupleLit(Vec::new()));
                }
                let first = self.parse_expr()?;
                if matches!(self.peek(), Tok::Walrus) {
                    // `(name := value)`: the parenthesised
                    // assignment expression, legal in any
                    // expression position. The target must be a
                    // plain name (CPython's rule).
                    let name = match first {
                        Expr::Name(n, _) => n,
                        _ => {
                            return Err(perr(
                                line,
                                "the target of := must be a plain name",
                            ))
                        }
                    };
                    self.bump();
                    let value = self.parse_expr()?;
                    self.expect(&Tok::RParen)?;
                    return Ok(Expr::Walrus { name, value: Box::new(value) });
                }
                if matches!(self.peek(), Tok::For) {
                    let clauses = self.parse_comp_clauses()?;
                    self.expect(&Tok::RParen)?;
                    return Ok(Expr::Comp {
                        kind: CompKind::GenExp,
                        key: Box::new(first),
                        value: None,
                        clauses,
                    });
                }
                if matches!(self.peek(), Tok::Comma) {
                    let mut elems = vec![first];
                    while matches!(self.peek(), Tok::Comma) {
                        self.bump();
                        if matches!(self.peek(), Tok::RParen) {
                            break;
                        }
                        elems.push(self.parse_expr()?);
                    }
                    self.expect(&Tok::RParen)?;
                    return Ok(Expr::TupleLit(elems));
                }
                self.expect(&Tok::RParen)?;
                Ok(first)
            }
            Tok::LBracket => {
                let mut elems = Vec::new();
                if !matches!(self.peek(), Tok::RBracket) {
                    let first = self.parse_expr()?;
                    if matches!(self.peek(), Tok::For) {
                        let clauses = self.parse_comp_clauses()?;
                        self.expect(&Tok::RBracket)?;
                        return Ok(Expr::Comp {
                            kind: CompKind::List,
                            key: Box::new(first),
                            value: None,
                            clauses,
                        });
                    }
                    elems.push(first);
                    while matches!(self.peek(), Tok::Comma) {
                        self.bump();
                        if matches!(self.peek(), Tok::RBracket) {
                            break;
                        }
                        elems.push(self.parse_expr()?);
                    }
                }
                self.expect(&Tok::RBracket)?;
                Ok(Expr::ListLit(elems))
            }
            Tok::LBrace => {
                if matches!(self.peek(), Tok::RBrace) {
                    self.bump();
                    return Ok(Expr::DictLit(Vec::new()));
                }
                let first = self.parse_expr()?;
                if matches!(self.peek(), Tok::Colon) {
                    self.bump();
                    let v = self.parse_expr()?;
                    if matches!(self.peek(), Tok::For) {
                        let clauses = self.parse_comp_clauses()?;
                        self.expect(&Tok::RBrace)?;
                        return Ok(Expr::Comp {
                            kind: CompKind::Dict,
                            key: Box::new(first),
                            value: Some(Box::new(v)),
                            clauses,
                        });
                    }
                    let mut pairs = vec![(first, v)];
                    while matches!(self.peek(), Tok::Comma) {
                        self.bump();
                        if matches!(self.peek(), Tok::RBrace) {
                            break;
                        }
                        let k = self.parse_expr()?;
                        self.expect(&Tok::Colon)?;
                        let v = self.parse_expr()?;
                        pairs.push((k, v));
                    }
                    self.expect(&Tok::RBrace)?;
                    return Ok(Expr::DictLit(pairs));
                }
                if matches!(self.peek(), Tok::For) {
                    let clauses = self.parse_comp_clauses()?;
                    self.expect(&Tok::RBrace)?;
                    return Ok(Expr::Comp {
                        kind: CompKind::Set,
                        key: Box::new(first),
                        value: None,
                        clauses,
                    });
                }
                let mut elems = vec![first];
                while matches!(self.peek(), Tok::Comma) {
                    self.bump();
                    if matches!(self.peek(), Tok::RBrace) {
                        break;
                    }
                    elems.push(self.parse_expr()?);
                }
                self.expect(&Tok::RBrace)?;
                Ok(Expr::SetLit(elems))
            }
            t => Err(perr(line, &format!("unexpected token {:?} in expression", t))),
        }
    }
}

/// Map an augmented-assignment token (`+=`, `//=`, `**=`, ...) to its
/// binary operator.
fn aug_op(t: &Tok) -> Option<BinOp> {
    Some(match t {
        Tok::PlusEq => BinOp::Add,
        Tok::MinusEq => BinOp::Sub,
        Tok::StarEq => BinOp::Mul,
        Tok::SlashEq => BinOp::Div,
        Tok::FloorDivEq => BinOp::FloorDiv,
        Tok::PercentEq => BinOp::Mod,
        Tok::PowEq => BinOp::Pow,
        Tok::BitAndEq => BinOp::BitAnd,
        Tok::BitOrEq => BinOp::BitOr,
        Tok::BitXorEq => BinOp::BitXor,
        Tok::ShlEq => BinOp::Shl,
        Tok::ShrEq => BinOp::Shr,
        _ => return None,
    })
}

/// True when an expression is pure and deterministic, so evaluating it
/// twice yields the same value with no observable difference (used by
/// chained comparisons, which duplicate their middle operands).
pub(crate) fn duplicable(e: &Expr) -> bool {
    match e {
        Expr::Int(_) | Expr::BigInt(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bool(_) | Expr::NoneLit | Expr::Name(_, _) => true,
        Expr::Index(a, b, _) => duplicable(a) && duplicable(b),
        Expr::Slice(a, lo, hi, st) => {
            duplicable(a)
                && [lo, hi, st]
                    .iter()
                    .all(|o| o.as_ref().map(|e| duplicable(e)).unwrap_or(true))
        }
        Expr::Bin(a, _, b) => duplicable(a) && duplicable(b),
        Expr::Unary(_, a) => duplicable(a),
        _ => false,
    }
}

/// Re-parse the source of one f-string `{...}` interpolation as an
/// expression of its own.
/// Annotate a diagnostic raised inside an f-string interpolation:
/// the inner lexer/parser saw only the fragment, so its line is
/// fragment-relative; re-anchor it at the f-string's own line and
/// record the context in the note instead of flattening the inner
/// diagnostic into a message string.
fn fstr_ctx(mut d: Diagnostic, line: u32) -> Diagnostic {
    if let Some(s) = d.sites.first_mut() {
        s.line = line;
    }
    d.with_note("raised inside an f-string expression".to_string())
}

fn parse_fstr_expr(src: &str, line: u32) -> Result<Expr, Diagnostic> {
    let toks = crate::lexer::lex(src).map_err(|d| fstr_ctx(d, line))?;
    let mut p = Parser::new(toks);
    let e = p.parse_expr().map_err(|d| fstr_ctx(d, line))?;
    match p.peek() {
        Tok::Newline | Tok::Eof => Ok(e),
        t => Err(perr(
            line,
            &format!("trailing token {} in f-string expression", tok_word(t)),
        )),
    }
}

/// Build one AST f-string part from a scanned part. A format
/// spec is parsed recursively: the spec source is scanned as a
/// mini f-string whose nested fields may carry their own
/// conversions and specs.
fn parse_fstr_part(p: crate::lexer::FPart, line: u32) -> Result<FStrPart, Diagnostic> {
    match p {
        crate::lexer::FPart::Lit(s) => Ok(FStrPart::Lit(s)),
        crate::lexer::FPart::Expr(src) => Ok(FStrPart::Field {
            value: parse_fstr_expr(&src, line)?,
            conv: None,
            debug: None,
            spec: None,
        }),
        crate::lexer::FPart::Fmt { src, conv, debug, spec } => {
            let spec = match spec {
                Some(spec_src) => {
                    let raw = crate::lexer::scan_fstring_fragment(&spec_src, line)
                        .map_err(|d| fstr_ctx(d, line))?;
                    let mut parts = Vec::new();
                    for sp in raw {
                        parts.push(parse_fstr_part(sp, line)?);
                    }
                    Some(parts)
                }
                None => None,
            };
            Ok(FStrPart::Field {
                value: parse_fstr_expr(&src, line)?,
                conv,
                debug,
                spec,
            })
        }
    }
}

fn contains_continue(stmts: &[Stmt]) -> bool {
    for s in stmts {
        match s {
            Stmt::Continue { .. } => return true,
            Stmt::If { arms, els, .. } => {
                for (_, b) in arms {
                    if contains_continue(b) {
                        return true;
                    }
                }
                if let Some(b) = els {
                    if contains_continue(b) {
                        return true;
                    }
                }
            }
            Stmt::While { els, .. }
            | Stmt::ForRange { els, .. }
            | Stmt::ForEach { els, .. } => {
                // A continue inside a nested loop's BODY belongs
                // to that loop and is not scanned. A continue in
                // its ELSE clause, however, targets the loop this
                // scan is protecting (the else clause sits outside
                // the nested loop), so the else body is scanned
                // like ordinary body code.
                if let Some(b) = els {
                    if contains_continue(b) {
                        return true;
                    }
                }
            }
            Stmt::Try { body, handlers, els, finally, .. } => {
                // A continue inside a try inside the loop still
                // targets the loop (D6): the flag rewrite would
                // skip the increment the same way.
                if contains_continue(body) {
                    return true;
                }
                for h in handlers {
                    if contains_continue(&h.body) {
                        return true;
                    }
                }
                if let Some(b) = els {
                    if contains_continue(b) {
                        return true;
                    }
                }
                if let Some(b) = finally {
                    if contains_continue(b) {
                        return true;
                    }
                }
            }
            Stmt::With { body, .. } => {
                if contains_continue(body) {
                    return true;
                }
            }
            Stmt::Match { cases, .. } => {
                for c in cases {
                    if contains_continue(&c.body) {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}
