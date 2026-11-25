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
            return Ok(Stmt::ForEach { var, iter, body, line });
        }
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
                // Look ahead for assignment forms.
                let next = self.toks.get(self.pos + 1).map(|t| t.tok.clone());
                match next {
                    Some(Tok::Assign) => {
                        self.bump();
                        self.bump();
                        let value = self.parse_expr()?;
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
        self.parse_ternary()
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
            Tok::LBrace => {
                let mut pairs = Vec::new();
                if !matches!(self.peek(), Tok::RBrace) {
                    loop {
                        let k = self.parse_expr()?;
                        self.expect(&Tok::Colon)?;
                        let v = self.parse_expr()?;
                        pairs.push((k, v));
                        if matches!(self.peek(), Tok::Comma) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Tok::RBrace)?;
                Ok(Expr::DictLit(pairs))
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
            _ => {}
        }
    }
    false
}
