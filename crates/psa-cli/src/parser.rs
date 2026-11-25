//! Recursive-descent parser for the .psa v0.1 subset.
//! Unsupported Python constructs fail loudly here or in the emitter.

use crate::ast::*;
use crate::lexer::{Tok, Token};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

fn perr(line: u32, msg: &str) -> String {
    format!("psa parse error at line {}: {}", line, msg)
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

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.pos].tok.clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, t: &Tok) -> Result<(), String> {
        if self.peek() == t {
            self.bump();
            Ok(())
        } else {
            Err(perr(self.line(), &format!("expected {:?}, found {:?}", t, self.peek())))
        }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Tok::Newline) {
            self.bump();
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, String> {
        let mut prog = Program {
            funcs: Vec::new(),
            classes: Vec::new(),
            top: Vec::new(),
            imports: Vec::new(),
            mod_attrs: Vec::new(),
            const_names: Vec::new(),
            modules: Vec::new(),
            std_imports: Vec::new(),
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
    fn parse_class(&mut self) -> Result<ClassDef, String> {
        let line = self.line();
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
        Ok(ClassDef { name, base, methods, attrs, line })
    }

    fn parse_func(&mut self, is_async: bool) -> Result<Func, String> {
        let line = self.line();
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
        Ok(Func { name, params, vararg, kwarg, is_async, body, line })
    }

    /// After a colon: either an indented block or a single simple statement
    /// on the same line (v0.1 supports the block form; one-liners too).
    fn parse_block(&mut self) -> Result<Vec<Stmt>, String> {
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

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
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
            Tok::Import | Tok::From => Err(perr(
                self.line(),
                "import is only allowed at the top level of a module in v1 (D6)",
            )),
            Tok::While => {
                let line = self.line();
                self.bump();
                let cond = self.parse_expr()?;
                self.expect(&Tok::Colon)?;
                let body = self.parse_block()?;
                Ok(Stmt::While { cond, body, line })
            }
            Tok::For => self.parse_for(),
            _ => {
                let s = self.parse_simple_stmt()?;
                self.expect(&Tok::Newline)?;
                Ok(s)
            }
        }
    }

    fn parse_if(&mut self) -> Result<Stmt, String> {
        let line = self.line();
        self.expect(&Tok::If)?;
        let cond = self.parse_expr()?;
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
                    let c = self.parse_expr()?;
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
        Ok(Stmt::If { arms, els, line })
    }

    fn parse_for(&mut self) -> Result<Stmt, String> {
        let line = self.line();
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
            return Ok(Stmt::ForEach { vars, iter, body, line });
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
        Ok(Stmt::ForRange { var, start, stop, body, line })
    }

    /// `import a[.b]` / `from a[.b] import x [as y], ...` at
    /// the top level (D6). Module aliasing (`import a as b`)
    /// and star imports are refused loudly in v1.
    fn parse_import(&mut self) -> Result<Import, String> {
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
    fn parse_module_path(&mut self, line: u32) -> Result<String, String> {
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
    fn parse_match(&mut self) -> Result<Stmt, String> {
        let line = self.line();
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
                Some(self.parse_expr()?)
            } else {
                None
            };
            self.expect(&Tok::Colon)?;
            let body = self.parse_block()?;
            cases.push(MatchCase { pattern, guard, body, line: cline });
        }
        self.expect(&Tok::Dedent)?;
        if cases.is_empty() {
            return Err(perr(line, "a match statement needs at least one case"));
        }
        Ok(Stmt::Match { subject, cases, line })
    }

    /// One match pattern (D7 v1 subset). OR alternatives are
    /// split on BitOr at this level; a sequence pattern is a
    /// bracketed or parenthesised comma list; anything else is
    /// a single atom.
    fn parse_pattern(&mut self) -> Result<Pattern, String> {
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

    fn parse_pattern_atom(&mut self) -> Result<Pattern, String> {
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
            Tok::Float(v) => {
                self.bump();
                Ok(Pattern::Lit(Expr::Float(v)))
            }
            Tok::Str(s) => {
                self.bump();
                Ok(Pattern::Lit(Expr::Str(s)))
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

    fn parse_pattern_seq(&mut self, close: &Tok) -> Result<Vec<Pattern>, String> {
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
    fn parse_try(&mut self) -> Result<Stmt, String> {
        let line = self.line();
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
                    handlers.push(ExcHandler { classes, bind, body: hbody, line: hline });
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
        Ok(Stmt::Try { body, handlers, els, finally, line })
    }

    /// `with <expr> [as name] (, <expr> [as name])* : block`
    /// (D6).
    fn parse_with(&mut self) -> Result<Stmt, String> {
        let line = self.line();
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
        Ok(Stmt::With { items, body, line })
    }

    fn parse_simple_stmt(&mut self) -> Result<Stmt, String> {
        let line = self.line();
        match self.peek().clone() {
            Tok::Yield => {
                self.bump();
                if matches!(self.peek(), Tok::Newline) {
                    Ok(Stmt::Yield { value: None, line })
                } else {
                    let v = self.parse_expr()?;
                    Ok(Stmt::Yield { value: Some(v), line })
                }
            }
            Tok::Raise => {
                self.bump();
                if matches!(self.peek(), Tok::Newline) {
                    Ok(Stmt::Raise { value: None, line })
                } else {
                    let v = self.parse_expr()?;
                    if matches!(self.peek(), Tok::From) {
                        return Err(perr(
                            line,
                            "exception chaining (`raise X from Y`) is not supported yet (D6 v1)",
                        ));
                    }
                    Ok(Stmt::Raise { value: Some(v), line })
                }
            }
            Tok::Return => {
                self.bump();
                if matches!(self.peek(), Tok::Newline) {
                    Ok(Stmt::Return { value: None, line })
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
                        })
                    } else {
                        Ok(Stmt::Return { value: Some(v), line })
                    }
                }
            }
            Tok::Break => {
                self.bump();
                Ok(Stmt::Break { line })
            }
            Tok::Continue => {
                self.bump();
                Ok(Stmt::Continue { line })
            }
            Tok::Pass => {
                self.bump();
                Ok(Stmt::Pass { line })
            }
            Tok::Ident(name) if name == "assert" => {
                self.bump();
                let cond = self.parse_expr()?;
                if matches!(self.peek(), Tok::Comma) {
                    self.bump();
                    let msg = self.parse_expr()?;
                    if !matches!(msg, Expr::Str(_)) {
                        return Err(perr(
                            line,
                            "assert message must be a string literal in v1 (it is a diagnostic label; psa panics with a code)",
                        ));
                    }
                }
                Ok(Stmt::Assert { cond, line })
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
                        let first = match self.bump() {
                            Tok::Ident(n) => n,
                            _ => unreachable!(),
                        };
                        let mut base = Expr::Name(first);
                        let mut attr: Option<String> = None;
                        while matches!(self.peek(), Tok::Dot) {
                            self.bump();
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
                                base = Expr::Attr(Box::new(base), prev);
                            }
                            attr = Some(n);
                        }
                        return Ok(Stmt::AttrDel {
                            base,
                            name: attr.expect("del attribute target"),
                            line,
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
                Ok(Stmt::Del { targets, line })
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
                            return Ok(Stmt::Unpack { names, value, line });
                        }
                        return Ok(Stmt::MultiAssign { names, values, line });
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
                    self.bump(); // first name
                    let mut base = Expr::Name(name.clone());
                    let mut attr: Option<String> = None;
                    let mut shape_ok = true;
                    while matches!(self.peek(), Tok::Dot) {
                        self.bump();
                        match self.bump() {
                            Tok::Ident(n) => {
                                if let Some(prev) = attr.take() {
                                    base = Expr::Attr(Box::new(base), prev);
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
                            });
                        }
                        Ok(Stmt::Assign { name, value, line })
                    }
                    Some(t) if aug_op(&t).is_some() => {
                        self.bump(); // name
                        let optok = self.bump();
                        let op = aug_op(&optok).unwrap();
                        let value = self.parse_expr()?;
                        Ok(Stmt::AugAssign { name, op, value, line })
                    }
                    _ => {
                        let e = self.parse_expr()?;
                        Ok(Stmt::ExprStmt { value: e, line })
                    }
                }
            }
            _ => {
                let e = self.parse_expr()?;
                Ok(Stmt::ExprStmt { value: e, line })
            }
        }
    }

    pub fn parse_expr(&mut self) -> Result<Expr, String> {
        if matches!(self.peek(), Tok::Lambda) {
            return self.parse_lambda();
        }
        self.parse_ternary()
    }

    /// `lambda a, b: body` (D3). Plain identifier parameters
    /// only in v1: defaults and *args/**kwargs inside a lambda
    /// are refused loudly here (the desugared def could carry
    /// them, but a lambda's signature cannot be pinned from a
    /// direct call site, so they ride on a later slice).
    fn parse_lambda(&mut self) -> Result<Expr, String> {
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
    fn parse_ternary(&mut self) -> Result<Expr, String> {
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

    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_and()?;
        while matches!(self.peek(), Tok::Or) {
            self.bump();
            let rhs = self.parse_and()?;
            lhs = Expr::Bin(Box::new(lhs), BinOp::Or, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_not()?;
        while matches!(self.peek(), Tok::And) {
            self.bump();
            let rhs = self.parse_not()?;
            lhs = Expr::Bin(Box::new(lhs), BinOp::And, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_not(&mut self) -> Result<Expr, String> {
        if matches!(self.peek(), Tok::Not) {
            self.bump();
            let e = self.parse_not()?;
            return Ok(Expr::Unary(UnOp::Not, Box::new(e)));
        }
        self.parse_cmp()
    }

    fn parse_cmp(&mut self) -> Result<Expr, String> {
        let mut operands = vec![self.parse_bitor()?];
        let mut ops: Vec<BinOp> = Vec::new();
        loop {
            let not_in = matches!(self.peek(), Tok::Not)
                && matches!(self.toks.get(self.pos + 1).map(|t| &t.tok), Some(Tok::In));
            let op = match self.peek() {
                Tok::EqEq => BinOp::Eq,
                Tok::NotEq => BinOp::NotEq,
                Tok::Lt => BinOp::Lt,
                Tok::LtEq => BinOp::LtEq,
                Tok::Gt => BinOp::Gt,
                Tok::GtEq => BinOp::GtEq,
                Tok::In => BinOp::In,
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

    fn parse_bitor(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_bitxor()?;
        while matches!(self.peek(), Tok::BitOr) {
            self.bump();
            let rhs = self.parse_bitxor()?;
            lhs = Expr::Bin(Box::new(lhs), BinOp::BitOr, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_bitxor(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_bitand()?;
        while matches!(self.peek(), Tok::BitXor) {
            self.bump();
            let rhs = self.parse_bitand()?;
            lhs = Expr::Bin(Box::new(lhs), BinOp::BitXor, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_bitand(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_shift()?;
        while matches!(self.peek(), Tok::BitAnd) {
            self.bump();
            let rhs = self.parse_shift()?;
            lhs = Expr::Bin(Box::new(lhs), BinOp::BitAnd, Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> Result<Expr, String> {
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

    fn parse_add(&mut self) -> Result<Expr, String> {
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

    fn parse_mul(&mut self) -> Result<Expr, String> {
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

    fn parse_unary(&mut self) -> Result<Expr, String> {
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
    fn parse_power(&mut self) -> Result<Expr, String> {
        let base = self.parse_postfix()?;
        if matches!(self.peek(), Tok::Pow) {
            self.bump();
            let exp = self.parse_unary()?;
            return Ok(Expr::Bin(Box::new(base), BinOp::Pow, Box::new(exp)));
        }
        Ok(base)
    }

    fn parse_postfix(&mut self) -> Result<Expr, String> {
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
                            args.push(self.parse_expr()?);
                            if matches!(self.peek(), Tok::Comma) {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(&Tok::RParen)?;
                    e = Expr::Call(Box::new(e), args, kwargs);
                }
                Tok::LBracket => {
                    self.bump();
                    let mut lo = None;
                    if !matches!(self.peek(), Tok::Colon) {
                        lo = Some(Box::new(self.parse_expr()?));
                    }
                    if matches!(self.peek(), Tok::Colon) {
                        self.bump();
                        let mut hi = None;
                        if !matches!(self.peek(), Tok::Colon | Tok::RBracket) {
                            hi = Some(Box::new(self.parse_expr()?));
                        }
                        let mut step = None;
                        if matches!(self.peek(), Tok::Colon) {
                            self.bump();
                            if !matches!(self.peek(), Tok::RBracket) {
                                step = Some(Box::new(self.parse_expr()?));
                            }
                        }
                        self.expect(&Tok::RBracket)?;
                        e = Expr::Slice(Box::new(e), lo, hi, step);
                    } else {
                        self.expect(&Tok::RBracket)?;
                        e = Expr::Index(Box::new(e), lo.unwrap());
                    }
                }
                Tok::Dot => {
                    self.bump();
                    let method = match self.bump() {
                        Tok::Ident(n) => n,
                        t => {
                            return Err(perr(
                                self.line(),
                                &format!("expected an attribute name after ., found {:?}", t),
                            ))
                        }
                    };
                    if !matches!(self.peek(), Tok::LParen) {
                        // Attribute access (D4): `base.name`.
                        e = Expr::Attr(Box::new(e), method);
                        continue;
                    }
                    self.expect(&Tok::LParen)?;
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Tok::RParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            if matches!(self.peek(), Tok::Comma) {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(&Tok::RParen)?;
                    match e {
                        Expr::Name(n) => {
                            e = Expr::MethodCall { obj: n, method, args };
                        }
                        // D5: a method call on a general base
                        // (a string literal, a subscript, a call
                        // result) desugars to a call through the
                        // attribute, which the emitter routes to
                        // the builtin method set or getattr.
                        _ => {
                            e = Expr::Call(
                                Box::new(Expr::Attr(Box::new(e), method)),
                                args,
                                Vec::new(),
                            );
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
    fn parse_comp_clauses(&mut self) -> Result<Vec<CompClause>, String> {
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
                let cond = self.parse_expr()?;
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

    fn parse_atom(&mut self) -> Result<Expr, String> {
        let line = self.line();
        match self.bump() {
            Tok::Int(v) => Ok(Expr::Int(v)),
            Tok::Float(v) => Ok(Expr::Float(v)),
            Tok::Str(s) => Ok(Expr::Str(s)),
            Tok::FStr(parts) => {
                let mut out = Vec::new();
                for p in parts {
                    match p {
                        crate::lexer::FPart::Lit(s) => out.push(FStrPart::Lit(s)),
                        crate::lexer::FPart::Expr(src) => {
                            out.push(FStrPart::Expr(parse_fstr_expr(&src, line)?))
                        }
                    }
                }
                Ok(Expr::FStr(out))
            }
            Tok::True => Ok(Expr::Bool(true)),
            Tok::False => Ok(Expr::Bool(false)),
            Tok::NoneLit => Ok(Expr::NoneLit),
            Tok::Ident(n) => Ok(Expr::Name(n)),
            Tok::LParen => {
                if matches!(self.peek(), Tok::RParen) {
                    self.bump();
                    return Ok(Expr::TupleLit(Vec::new()));
                }
                let first = self.parse_expr()?;
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
fn duplicable(e: &Expr) -> bool {
    match e {
        Expr::Int(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bool(_) | Expr::NoneLit | Expr::Name(_) => true,
        Expr::Index(a, b) => duplicable(a) && duplicable(b),
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
fn parse_fstr_expr(src: &str, line: u32) -> Result<Expr, String> {
    let toks = crate::lexer::lex(src)
        .map_err(|e| perr(line, &format!("in f-string: {}", e)))?;
    let mut p = Parser::new(toks);
    let e = p.parse_expr().map_err(|e| {
        perr(
            line,
            &format!(
                "in f-string expression (format specs and conversions are not supported in D0): {}",
                e
            ),
        )
    })?;
    match p.peek() {
        Tok::Newline | Tok::Eof => Ok(e),
        t => Err(perr(
            line,
            &format!("trailing token {:?} in f-string expression", t),
        )),
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
            Stmt::While { .. } => {
                // A continue inside a nested while belongs to that while.
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
