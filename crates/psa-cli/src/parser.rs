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
        let mut prog = Program { funcs: Vec::new(), top: Vec::new() };
        self.skip_newlines();
        while !matches!(self.peek(), Tok::Eof) {
            if matches!(self.peek(), Tok::Def) {
                let f = self.parse_func()?;
                prog.funcs.push(f);
            } else {
                let s = self.parse_stmt()?;
                prog.top.push(s);
            }
            self.skip_newlines();
        }
        Ok(prog)
    }

    fn parse_func(&mut self) -> Result<Func, String> {
        let line = self.line();
        self.expect(&Tok::Def)?;
        let name = match self.bump() {
            Tok::Ident(n) => n,
            t => return Err(perr(line, &format!("expected function name, found {:?}", t))),
        };
        self.expect(&Tok::LParen)?;
        let mut params = Vec::new();
        if !matches!(self.peek(), Tok::RParen) {
            loop {
                match self.bump() {
                    Tok::Ident(n) => params.push(n),
                    t => {
                        return Err(perr(
                            line,
                            &format!("expected parameter name, found {:?} (annotations/defaults are not in v0.1)", t),
                        ))
                    }
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
        Ok(Func { name, params, body, line })
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
        match self.peek().clone() {
            Tok::If => self.parse_if(),
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
        let var = match self.bump() {
            Tok::Ident(n) => n,
            t => return Err(perr(line, &format!("expected loop variable, found {:?}", t))),
        };
        self.expect(&Tok::In)?;
        // v0.1: only `for x in range(...)`.
        match self.bump() {
            Tok::Ident(n) if n == "range" => {}
            t => {
                return Err(perr(
                    line,
                    &format!("v0.1 for loops iterate range(...) only, found {:?}", t),
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

    fn parse_simple_stmt(&mut self) -> Result<Stmt, String> {
        let line = self.line();
        match self.peek().clone() {
            Tok::Return => {
                self.bump();
                if matches!(self.peek(), Tok::Newline) {
                    Ok(Stmt::Return { value: None, line })
                } else {
                    let v = self.parse_expr()?;
                    Ok(Stmt::Return { value: Some(v), line })
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
            Tok::Ident(name) => {
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
                                Tok::PlusEq => Some(Some(BinOp::Add)),
                                Tok::MinusEq => Some(Some(BinOp::Sub)),
                                Tok::StarEq => Some(Some(BinOp::Mul)),
                                _ => None,
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
                // Look ahead for assignment forms.
                let next = self.toks.get(self.pos + 1).map(|t| t.tok.clone());
                match next {
                    Some(Tok::Assign) => {
                        self.bump();
                        self.bump();
                        let value = self.parse_expr()?;
                        Ok(Stmt::Assign { name, value, line })
                    }
                    Some(Tok::PlusEq) | Some(Tok::MinusEq) | Some(Tok::StarEq) => {
                        self.bump(); // name
                        let optok = self.bump();
                        let op = match optok {
                            Tok::PlusEq => BinOp::Add,
                            Tok::MinusEq => BinOp::Sub,
                            Tok::StarEq => BinOp::Mul,
                            _ => unreachable!(),
                        };
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
        self.parse_or()
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
        let lhs = self.parse_add()?;
        let op = match self.peek() {
            Tok::EqEq => Some(BinOp::Eq),
            Tok::NotEq => Some(BinOp::NotEq),
            Tok::Lt => Some(BinOp::Lt),
            Tok::LtEq => Some(BinOp::LtEq),
            Tok::Gt => Some(BinOp::Gt),
            Tok::GtEq => Some(BinOp::GtEq),
            _ => None,
        };
        if let Some(op) = op {
            self.bump();
            let rhs = self.parse_add()?;
            // Chained comparisons (a < b < c) are not in v0.1.
            if matches!(
                self.peek(),
                Tok::EqEq | Tok::NotEq | Tok::Lt | Tok::LtEq | Tok::Gt | Tok::GtEq
            ) {
                return Err(perr(self.line(), "chained comparisons are not supported in v0.1"));
            }
            return Ok(Expr::Bin(Box::new(lhs), op, Box::new(rhs)));
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
        if matches!(self.peek(), Tok::Minus) {
            self.bump();
            let e = self.parse_unary()?;
            return Ok(Expr::Unary(UnOp::Neg, Box::new(e)));
        }
        if matches!(self.peek(), Tok::Plus) {
            self.bump();
            return self.parse_unary();
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_atom()?;
        loop {
            match self.peek() {
                Tok::LParen => {
                    self.bump();
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
                    e = Expr::Call(Box::new(e), args);
                }
                Tok::LBracket => {
                    self.bump();
                    let idx = self.parse_expr()?;
                    self.expect(&Tok::RBracket)?;
                    e = Expr::Index(Box::new(e), Box::new(idx));
                }
                Tok::Dot => {
                    self.bump();
                    let method = match self.bump() {
                        Tok::Ident(n) => n,
                        t => {
                            return Err(perr(
                                self.line(),
                                &format!("expected method name after ., found {:?}", t),
                            ))
                        }
                    };
                    let obj = match e {
                        Expr::Name(n) => n,
                        _ => {
                            return Err(perr(
                                self.line(),
                                "method calls are only supported on plain variables in v0.3",
                            ))
                        }
                    };
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
                    e = Expr::MethodCall { obj, method, args };
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn parse_atom(&mut self) -> Result<Expr, String> {
        let line = self.line();
        match self.bump() {
            Tok::Int(v) => Ok(Expr::Int(v)),
            Tok::Float(v) => Ok(Expr::Float(v)),
            Tok::Str(s) => Ok(Expr::Str(s)),
            Tok::True => Ok(Expr::Bool(true)),
            Tok::False => Ok(Expr::Bool(false)),
            Tok::Ident(n) => Ok(Expr::Name(n)),
            Tok::LParen => {
                let e = self.parse_expr()?;
                self.expect(&Tok::RParen)?;
                Ok(e)
            }
            Tok::LBracket => {
                let mut elems = Vec::new();
                if !matches!(self.peek(), Tok::RBracket) {
                    loop {
                        elems.push(self.parse_expr()?);
                        if matches!(self.peek(), Tok::Comma) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Tok::RBracket)?;
                Ok(Expr::ListLit(elems))
            }
            t => Err(perr(line, &format!("unexpected token {:?} in expression", t))),
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
            Stmt::While { .. } => {
                // A continue inside a nested while belongs to that while.
            }
            _ => {}
        }
    }
    false
}
