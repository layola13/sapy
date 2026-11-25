//! Type inference + SLA emission for the .psa v0.1 subset.
//!
//! The .psa layer infers concrete types and ownership-neutral scalar
//! semantics, then emits SLA source whose explicit annotations carry the
//! inferred facts; the SLA checker and the SA Referee verify them
//! downstream. Anything the subset cannot express is a loud error.

use crate::ast::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ty {
    Int,
    Float,
    Bool,
    Str,
    ListInt,
    Void,
}

impl Ty {
    fn sla(&self) -> &'static str {
        match self {
            Ty::Int => "i64",
            Ty::Float => "f64",
            Ty::Bool => "bool",
            Ty::Str => "&str",
            Ty::ListInt => "Vec<i64>",
            Ty::Void => "i32",
        }
    }
    fn default_lit(&self) -> &'static str {
        match self {
            Ty::Int => "0",
            Ty::Float => "0.0",
            Ty::Bool => "false",
            Ty::Str => "\"\"",
            Ty::ListInt => "Vec::new()",
            Ty::Void => "0",
        }
    }
}

#[derive(Debug, Clone)]
struct Sig {
    params: Vec<Ty>,
    /// Per-parameter: true when a list parameter is borrowed
    /// (SLA `&Vec<i64>`) rather than consumed (SLA `Vec<i64>`).
    param_borrow: Vec<bool>,
    ret: Option<Ty>,
}

pub struct Emitter {
    defs: HashMap<String, Func>,
    sigs: HashMap<String, Sig>,
    in_progress: Vec<String>,
    emitted: Vec<String>,
    emitted_set: Vec<String>,
    used_floordiv: bool,
    used_floormod: bool,
    used_vec: bool,
}

struct Ctx {
    vars: Vec<(String, Ty, Option<u32>)>,
    scopes: Vec<usize>,
    ret: Option<Ty>,
    fname: String,
    out: String,
    indent: usize,
    tmp: usize,
    borrowed: Vec<String>,
}

impl Ctx {
    fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }
    fn lookup(&self, name: &str) -> Option<Ty> {
        self.vars.iter().rev().find(|(n, _, _)| n == name).map(|(_, t, _)| *t)
    }
    fn moved_at(&self, name: &str) -> Option<u32> {
        self.vars
            .iter()
            .rev()
            .find(|(n, _, _)| n == name)
            .and_then(|(_, _, m)| *m)
    }
    fn mark_moved(&mut self, name: &str, line: u32) {
        if let Some(entry) = self.vars.iter_mut().rev().find(|(n, _, _)| n == name) {
            entry.2 = Some(line);
        }
    }
    fn revive(&mut self, name: &str) {
        if let Some(entry) = self.vars.iter_mut().rev().find(|(n, _, _)| n == name) {
            entry.2 = None;
        }
    }
    fn check_alive(&self, name: &str, line: u32) -> Result<(), String> {
        if let Some(at) = self.moved_at(name) {
            return Err(eerr(
                line,
                &format!(
                    "use of `{}` after it was moved at line {} (container assignment moves; the binding is dead until re-bound)",
                    name, at
                ),
            ));
        }
        Ok(())
    }
    fn declare(&mut self, name: &str, ty: Ty) -> Result<(), String> {
        // Redeclaring in the *same* scope is an error; shadowing an outer
        // scope is allowed (the inner binding wins until the scope ends).
        let mark = *self.scopes.last().unwrap_or(&0);
        if self.vars[mark..].iter().any(|(n, _, _)| n == name) {
            return Err(format!(
                "psa emit error: variable `{}` assigned twice in the same scope (v0.1 treats re-binding as an error; rename or restructure)",
                name
            ));
        }
        self.vars.push((name.to_string(), ty, None));
        Ok(())
    }
}

fn eerr(line: u32, msg: &str) -> String {
    format!("psa emit error at line {}: {}", line, msg)
}

pub fn emit_program(prog: &Program) -> Result<String, String> {
    let mut em = Emitter {
        defs: HashMap::new(),
        sigs: HashMap::new(),
        in_progress: Vec::new(),
        emitted: Vec::new(),
        emitted_set: Vec::new(),
        used_floordiv: false,
        used_floormod: false,
        used_vec: false,
    };
    for f in &prog.funcs {
        if em.defs.insert(f.name.clone(), f.clone()).is_some() {
            return Err(eerr(f.line, &format!("duplicate function definition `{}`", f.name)));
        }
    }
    // Emit user functions on demand (driven by calls from main / each
    // other), then main from the top-level statements.
    let mut main_ctx = Ctx {
        vars: Vec::new(),
        scopes: vec![0],
        ret: Some(Ty::Void),
        fname: "main".to_string(),
        out: String::new(),
        indent: 1,
        tmp: 0,
        borrowed: Vec::new(),
    };
    for s in &prog.top {
        em.emit_stmt(&mut main_ctx, s)?;
    }
    main_ctx.line("return 0;");
    let mut main_txt = String::from("fn main() -> i32 {\n");
    main_txt.push_str(&main_ctx.out);
    main_txt.push_str("}\n");

    let mut out = String::new();
    out.push_str("@import \"sa_std/io/print.sai\"\n@import \"sa_std/fmt.sai\"\n");
    if em.used_vec {
        out.push_str("@import \"sa_std/vec.sa\"\n");
    }
    out.push('\n');
    if em.used_floordiv {
        out.push_str(
            "fn __psa_floordiv(a: i64, b: i64) -> i64 {\n    let q = a / b;\n    let r = a % b;\n    if r != 0 && ((r < 0) != (b < 0)) { return q - 1; };\n    return q;\n}\n\n",
        );
    }
    if em.used_floormod {
        out.push_str(
            "fn __psa_floormod(a: i64, b: i64) -> i64 {\n    let r = a % b;\n    if r != 0 && ((r < 0) != (b < 0)) { return r + b; };\n    return r;\n}\n\n",
        );
    }
    for f in &em.emitted {
        out.push_str(f);
        out.push('\n');
    }
    out.push_str(&main_txt);
    Ok(out)
}

impl Emitter {
    fn ensure_fn(&mut self, name: &str, arg_tys: &[Ty], line: u32) -> Result<Sig, String> {
        if let Some(sig) = self.sigs.get(name) {
            if sig.params != arg_tys {
                return Err(eerr(
                    line,
                    &format!(
                        "function `{}` called with argument types {:?} but earlier inferred as {:?} (v0.1 is monomorphic: one signature per function)",
                        name, arg_tys, sig.params
                    ),
                ));
            }
            return Ok(sig.clone());
        }
        let def = match self.defs.get(name) {
            Some(d) => d.clone(),
            None => {
                return Err(eerr(line, &format!("call to undefined function `{}`", name)));
            }
        };
        if def.params.len() != arg_tys.len() {
            return Err(eerr(
                line,
                &format!(
                    "function `{}` takes {} arguments, called with {}",
                    name,
                    def.params.len(),
                    arg_tys.len()
                ),
            ));
        }
        // Ownership per parameter, from the callee body alone (spec
        // s4): a list parameter the body only reads or writes through
        // is borrowed (SLA &Vec<i64>); one it moves out of is consumed
        // (SLA Vec<i64>). Scalars are Copy and never borrow.
        let borrows: Vec<bool> = def
            .params
            .iter()
            .zip(arg_tys.iter())
            .map(|(p, t)| {
                if *t != Ty::ListInt {
                    return false;
                }
                let mut visiting = Vec::new();
                !param_usage(self, &def, p, &mut visiting)
            })
            .collect();
        // Register the signature before inferring the body so recursive
        // calls resolve; the return type starts provisional (Int) and is
        // verified when the body finishes.
        self.sigs.insert(
            name.to_string(),
            Sig { params: arg_tys.to_vec(), param_borrow: borrows.clone(), ret: Some(Ty::Int) },
        );
        self.in_progress.push(name.to_string());
        let mut ctx = Ctx {
            vars: def
                .params
                .iter()
                .cloned()
                .zip(arg_tys.iter().cloned())
                .map(|(n, t)| (n, t, None))
                .collect(),
            scopes: vec![0],
            ret: None,
            fname: name.to_string(),
            out: String::new(),
            indent: 1,
            tmp: 0,
            borrowed: def
                .params
                .iter()
                .zip(borrows.iter())
                .filter(|(_, b)| **b)
                .map(|(p, _)| p.clone())
                .collect(),
        };
        let body_res = (|| {
            for s in &def.body {
                self.emit_stmt(&mut ctx, s)?;
            }
            Ok::<(), String>(())
        })();
        self.in_progress.pop();
        if let Err(e) = body_res {
            self.sigs.remove(name);
            return Err(e);
        }
        let ret_ty = ctx.ret.unwrap_or(Ty::Void);
        // Verify the provisional return type used by recursive calls.
        if ret_ty != Ty::Int && self.in_progress.is_empty() {
            // A recursive call already consumed the provisional Int type;
            // if the true return type differs, that call site was typed
            // wrong -- but only the top-level finisher can see this, and
            // the recursion may have happened deeper. We check: if the
            // function is recursive (it called itself) and ret != Int,
            // refuse loudly rather than miscompile.
            if def_calls(&def.body, name) && ret_ty != Ty::Void {
                self.sigs.remove(name);
                return Err(eerr(
                    def.line,
                    &format!(
                        "recursive function `{}` has inferred return type {:?}; v0.1 supports recursive returns of Int only",
                        name, ret_ty
                    ),
                ));
            }
        }
        self.sigs.get_mut(name).unwrap().ret = Some(ret_ty);
        // Append a default return so all paths return (Python falls off
        // the end returning None; for value-typed functions the earlier
        // return-type unification already rejected mixed use).
        let mut txt = String::new();
        txt.push_str(&format!("fn {}(", name));
        for (i, (p, t)) in def.params.iter().zip(arg_tys.iter()).enumerate() {
            if i > 0 {
                txt.push_str(", ");
            }
            if *t == Ty::ListInt && borrows.get(i).copied().unwrap_or(false) {
                txt.push_str(&format!("{}: &Vec<i64>", p));
            } else {
                txt.push_str(&format!("{}: {}", p, t.sla()));
            }
        }
        txt.push_str(&format!(") -> {} {{\n", ret_ty.sla()));
        txt.push_str(&ctx.out);
        // Default fallthrough return (unreachable when the body always
        // returns; the SLA checker accepts it either way).
        txt.push_str(&format!("    return {};\n}}\n", ret_ty.default_lit()));
        if !self.emitted_set.contains(&name.to_string()) {
            self.emitted_set.push(name.to_string());
            self.emitted.push(txt);
        }
        Ok(self.sigs.get(name).unwrap().clone())
    }

    /// Emit `let <name>: Vec<i64> = Vec::new();` (or a plain assignment
    /// when `declare` is false) followed by one push per element.
    fn emit_list_construct(
        &mut self,
        ctx: &mut Ctx,
        name: &str,
        elems: &[Expr],
        line: u32,
        declare: bool,
    ) -> Result<(), String> {
        self.used_vec = true;
        if declare {
            ctx.line(&format!("let {}: Vec<i64> = Vec::new();", name));
        } else {
            ctx.line(&format!("{} = Vec::new();", name));
        }
        for e in elems {
            let (code, ty) = self.emit_expr(ctx, e, line)?;
            if ty != Ty::Int {
                return Err(eerr(line, "list literals hold int elements only in v0.2 (list[int])"));
            }
            ctx.line(&format!("{}.push({});", name, code));
        }
        Ok(())
    }

    fn emit_list_assign(
        &mut self,
        ctx: &mut Ctx,
        name: &str,
        elems: &[Expr],
        line: u32,
    ) -> Result<(), String> {
        match ctx.lookup(name) {
            None => {
                ctx.declare(name, Ty::ListInt).map_err(|m| eerr(line, &m))?;
                self.emit_list_construct(ctx, name, elems, line, true)
            }
            Some(Ty::ListInt) => {
                self.emit_list_construct(ctx, name, elems, line, false)?;
                ctx.revive(name);
                Ok(())
            }
            Some(other) => Err(eerr(
                line,
                &format!("variable `{}` changes type from {:?} to list[int]", name, other),
            )),
        }
    }

    fn emit_block(&mut self, ctx: &mut Ctx, stmts: &[Stmt]) -> Result<(), String> {
        ctx.scopes.push(ctx.vars.len());
        for s in stmts {
            self.emit_stmt(ctx, s)?;
        }
        let mark = ctx.scopes.pop().unwrap();
        ctx.vars.truncate(mark);
        Ok(())
    }

    fn emit_stmt(&mut self, ctx: &mut Ctx, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Pass { .. } => Ok(()),
            Stmt::Break { .. } => {
                ctx.line("break;");
                Ok(())
            }
            Stmt::Continue { .. } => {
                ctx.line("continue;");
                Ok(())
            }
            Stmt::Return { value, line } => {
                match value {
                    None => {
                        if ctx.fname == "main" {
                            ctx.line("return 0;");
                        } else {
                            unify_ret(ctx, Ty::Void, *line)?;
                            ctx.line(&format!("return {};", Ty::Void.default_lit()));
                        }
                    }
                    Some(Expr::ListLit(elems)) => {
                        // Build the list in a temp, then return it (move).
                        ctx.tmp += 1;
                        let tmp = format!("__psa_ret_{}", ctx.tmp);
                        self.emit_list_construct(ctx, &tmp, elems, *line, true)?;
                        unify_ret(ctx, Ty::ListInt, *line)?;
                        ctx.line(&format!("return {};", tmp));
                    }
                    Some(e) => {
                        if let Expr::Name(n) = e {
                            if ctx.borrowed.contains(n) {
                                return Err(eerr(
                                    *line,
                                    &format!(
                                        "cannot return `{}`: it is a borrowed parameter of `{}` and borrows may not escape (spec s7)",
                                        n, ctx.fname
                                    ),
                                ));
                            }
                        }
                        let (code, ty) = self.emit_expr(ctx, e, *line)?;
                        if ctx.fname == "main" {
                            if ty != Ty::Int {
                                return Err(eerr(*line, "main must return an int"));
                            }
                            ctx.line(&format!("return {} as i32;", code));
                        } else {
                            unify_ret(ctx, ty, *line)?;
                            ctx.line(&format!("return {};", code));
                        }
                    }
                }
                Ok(())
            }
            Stmt::Assign { name, value, line } => {
                // List literals are constructed statement-wise.
                if let Expr::ListLit(elems) = value {
                    return self.emit_list_assign(ctx, name, elems, *line);
                }
                let (code, ty) = self.emit_expr(ctx, value, *line)?;
                if ty == Ty::Void {
                    return Err(eerr(*line, "cannot assign a void value"));
                }
                // Container assignment moves the source binding (spec s3).
                let src_name = match value {
                    Expr::Name(n) if ty == Ty::ListInt => Some(n.clone()),
                    _ => None,
                };
                if let Some(src) = &src_name {
                    if ctx.borrowed.contains(src) {
                        return Err(eerr(
                            *line,
                            &format!(
                                "cannot move `{}`: it is a borrowed parameter of `{}` (spec s7)",
                                src, ctx.fname
                            ),
                        ));
                    }
                }
                match ctx.lookup(name) {
                    None => {
                        ctx.declare(name, ty)
                            .map_err(|m| eerr(*line, &m))?;
                        ctx.line(&format!("let {}: {} = {};", name, ty.sla(), code));
                    }
                    Some(existing) => {
                        if existing == ty {
                            ctx.line(&format!("{} = {};", name, code));
                            ctx.revive(name);
                        } else if existing == Ty::Float && ty == Ty::Int {
                            ctx.line(&format!("{} = {} as f64;", name, code));
                        } else {
                            return Err(eerr(
                                *line,
                                &format!(
                                    "variable `{}` changes type from {:?} to {:?} (v0.1 keeps one type per variable)",
                                    name, existing, ty
                                ),
                            ));
                        }
                    }
                }
                if let Some(src) = src_name {
                    if src != *name {
                        ctx.mark_moved(&src, *line);
                    }
                }
                Ok(())
            }
            Stmt::IndexAssign { name, index, op, value, line } => {
                let ty = ctx
                    .lookup(name)
                    .ok_or_else(|| eerr(*line, &format!("index assignment to undefined variable `{}`", name)))?;
                if ty != Ty::ListInt {
                    return Err(eerr(*line, "index assignment is only supported on list[int] in v0.2"));
                }
                ctx.check_alive(name, *line)?;
                let (icode, ity) = self.emit_expr(ctx, index, *line)?;
                if ity != Ty::Int {
                    return Err(eerr(*line, "list index must be int"));
                }
                let (vcode, vty) = self.emit_expr(ctx, value, *line)?;
                if vty != Ty::Int {
                    return Err(eerr(*line, "list[int] elements must be int"));
                }
                match op {
                    None => ctx.line(&format!("{}[{}] = {};", name, icode, vcode)),
                    Some(bop) => {
                        let place = format!("{}[{}]", name, icode);
                        let (code, _) = self.emit_binop(place.clone(), Ty::Int, *bop, vcode, Ty::Int, *line)?;
                        ctx.line(&format!("{} = {};", place, code));
                    }
                }
                Ok(())
            }
            Stmt::AugAssign { name, op, value, line } => {
                let existing = ctx
                    .lookup(name)
                    .ok_or_else(|| eerr(*line, &format!("augmented assignment to undefined variable `{}`", name)))?;
                let fake = Expr::Bin(
                    Box::new(Expr::Name(name.clone())),
                    *op,
                    Box::new(value.clone()),
                );
                let (code, ty) = self.emit_expr(ctx, &fake, *line)?;
                if ty != existing {
                    if existing == Ty::Float && ty == Ty::Int {
                        ctx.line(&format!("{} = {} as f64;", name, code));
                        return Ok(());
                    }
                    return Err(eerr(
                        *line,
                        &format!("augmented assignment changes `{}` from {:?} to {:?}", name, existing, ty),
                    ));
                }
                ctx.line(&format!("{} = {};", name, code));
                Ok(())
            }
            Stmt::ExprStmt { value, line } => {
                // print(...) is a builtin statement-level call.
                if let Expr::Call(callee, args) = value {
                    if let Expr::Name(n) = &**callee {
                        if n == "print" {
                            return self.emit_print(ctx, args, *line);
                        }
                    }
                }
                let (code, _ty) = self.emit_expr(ctx, value, *line)?;
                ctx.line(&format!("{};", code));
                Ok(())
            }
            Stmt::If { arms, els, line } => {
                for (i, (cond, body)) in arms.iter().enumerate() {
                    let (ccode, cty) = self.emit_expr(ctx, cond, *line)?;
                    if cty != Ty::Bool {
                        return Err(eerr(*line, "if condition must be bool"));
                    }
                    if i == 0 {
                        ctx.line(&format!("if {} {{", ccode));
                    } else {
                        ctx.line(&format!("}} else if {} {{", ccode));
                    }
                    ctx.indent += 1;
                    self.emit_block(ctx, body)?;
                    ctx.indent -= 1;
                }
                if let Some(body) = els {
                    ctx.line("} else {");
                    ctx.indent += 1;
                    self.emit_block(ctx, body)?;
                    ctx.indent -= 1;
                }
                ctx.line("};");
                Ok(())
            }
            Stmt::While { cond, body, line } => {
                let (ccode, cty) = self.emit_expr(ctx, cond, *line)?;
                if cty != Ty::Bool {
                    return Err(eerr(*line, "while condition must be bool"));
                }
                ctx.line(&format!("while {} {{", ccode));
                ctx.indent += 1;
                self.emit_block(ctx, body)?;
                ctx.indent -= 1;
                ctx.line("}");
                Ok(())
            }
            Stmt::ForRange { var, start, stop, body, line } => {
                let (stop_code, stop_ty) = self.emit_expr(ctx, stop, *line)?;
                if stop_ty != Ty::Int {
                    return Err(eerr(*line, "range() bounds must be int"));
                }
                let (start_code, start_ty) = match start {
                    Some(e) => self.emit_expr(ctx, e, *line)?,
                    None => ("0".to_string(), Ty::Int),
                };
                if start_ty != Ty::Int {
                    return Err(eerr(*line, "range() bounds must be int"));
                }
                ctx.line("{");
                ctx.indent += 1;
                ctx.scopes.push(ctx.vars.len());
                ctx.declare(var, Ty::Int).map_err(|m| eerr(*line, &m))?;
                // Materialise the stop value once (Python evaluates the
                // range bound once; a bare expression in the while
                // condition would re-evaluate every iteration).
                let stop_var = format!("__psa_stop_{}", line);
                ctx.line(&format!("let {}: i64 = {};", stop_var, stop_code));
                ctx.line(&format!("let {}: i64 = {};", var, start_code));
                ctx.line(&format!("while {} < {} {{", var, stop_var));
                ctx.indent += 1;
                // Body statements share the loop scope (the loop variable
                // stays visible); nested blocks scope themselves.
                for s in body {
                    self.emit_stmt(ctx, s)?;
                }
                ctx.line(&format!("{} = {} + 1;", var, var));
                ctx.indent -= 1;
                ctx.line("}");
                let mark = ctx.scopes.pop().unwrap();
                ctx.vars.truncate(mark);
                ctx.indent -= 1;
                ctx.line("}");
                Ok(())
            }
        }
    }

    fn emit_print(&mut self, ctx: &mut Ctx, args: &[Expr], line: u32) -> Result<(), String> {
        if args.is_empty() {
            ctx.line("println(\"\");");
            return Ok(());
        }
        // Single bool argument: Python prints True/False.
        if args.len() == 1 {
            let (code, ty) = self.emit_expr(ctx, &args[0], line)?;
            if ty == Ty::Float {
                return Err(eerr(
                    line,
                    "printing a float is blocked in v0.1: the Rust sla build-exe route truncates f64 in println (3.5 prints as 3.0000000000) while the value itself computes correctly; psa refuses to print a wrong value (upstream defect, see README)",
                ));
            }
            if ty == Ty::ListInt {
                return Err(eerr(line, "printing a whole list is not supported in v0.2 (print its elements or len)"));
            }
            if ty == Ty::Bool {
                ctx.line(&format!("if {} {{ println(\"True\"); }} else {{ println(\"False\"); }};", code));
                return Ok(());
            }
            if let Expr::Str(s) = &args[0] {
                if s.contains('{') || s.contains('}') {
                    return Err(eerr(line, "string literals containing braces are not supported in print in v0.1"));
                }
                ctx.line(&format!("println(\"{}\");", escape_sla_str(s)));
                return Ok(());
            }
            ctx.line(&format!("println(\"{{}}\", {});", code));
            return Ok(());
        }
        // Multiple arguments: space-joined, no bools in v0.1.
        let mut fmt = String::new();
        let mut codes = Vec::new();
        for (i, a) in args.iter().enumerate() {
            let (code, ty) = self.emit_expr(ctx, a, line)?;
            if ty == Ty::Bool {
                return Err(eerr(line, "bool values in multi-argument print are not supported in v0.1"));
            }
            if ty == Ty::Float {
                return Err(eerr(
                    line,
                    "printing a float is blocked in v0.1: the Rust sla build-exe route truncates f64 in println (upstream defect, see README)",
                ));
            }
            if ty == Ty::ListInt {
                return Err(eerr(line, "printing a whole list is not supported in v0.2 (print its elements or len)"));
            }
            if let Expr::Str(s) = a {
                if s.contains('{') || s.contains('}') {
                    return Err(eerr(line, "string literals containing braces are not supported in print in v0.1"));
                }
                if i > 0 {
                    fmt.push(' ');
                }
                fmt.push_str(&escape_sla_str(s));
            } else {
                if i > 0 {
                    fmt.push(' ');
                }
                fmt.push_str("{}");
                codes.push(code);
            }
        }
        if codes.is_empty() {
            ctx.line(&format!("println(\"{}\");", fmt));
        } else {
            ctx.line(&format!("println(\"{}\", {});", fmt, codes.join(", ")));
        }
        Ok(())
    }

    fn emit_expr(&mut self, ctx: &mut Ctx, e: &Expr, line: u32) -> Result<(String, Ty), String> {
        match e {
            Expr::Int(v) => Ok((v.to_string(), Ty::Int)),
            Expr::Float(v) => Ok((fmt_f64(*v), Ty::Float)),
            Expr::Str(s) => Ok((format!("\"{}\"", escape_sla_str(s)), Ty::Str)),
            Expr::Bool(b) => Ok((b.to_string(), Ty::Bool)),
            Expr::Name(n) => match ctx.lookup(n) {
                Some(t) => {
                    ctx.check_alive(n, line)?;
                    Ok((n.clone(), t))
                }
                None => Err(eerr(line, &format!("undefined variable `{}`", n))),
            },
            Expr::ListLit(_) => Err(eerr(
                line,
                "list literals are only supported as an assignment value or a return value in v0.2",
            )),
            Expr::Index(base, idx) => {
                let obj = match &**base {
                    Expr::Name(n) => n.clone(),
                    _ => {
                        return Err(eerr(
                            line,
                            "indexing is only supported on plain variables in v0.2",
                        ))
                    }
                };
                let ty = ctx
                    .lookup(&obj)
                    .ok_or_else(|| eerr(line, &format!("undefined variable `{}`", obj)))?;
                if ty != Ty::ListInt {
                    return Err(eerr(line, "indexing is only supported on list[int] in v0.2"));
                }
                ctx.check_alive(&obj, line)?;
                let (icode, ity) = self.emit_expr(ctx, idx, line)?;
                if ity != Ty::Int {
                    return Err(eerr(line, "list index must be int"));
                }
                // Scalar element: copied out of the borrow (spec s5).
                Ok((format!("{}[{}]", obj, icode), Ty::Int))
            }
            Expr::MethodCall { obj, method, args } => {
                let ty = ctx
                    .lookup(obj)
                    .ok_or_else(|| eerr(line, &format!("undefined variable `{}`", obj)))?;
                if ty != Ty::ListInt || method != "append" {
                    return Err(eerr(
                        line,
                        &format!(
                            "method `{}` is not supported in v0.2 (only list.append)",
                            method
                        ),
                    ));
                }
                ctx.check_alive(obj, line)?;
                if args.len() != 1 {
                    return Err(eerr(line, "append takes exactly one argument"));
                }
                let (acode, aty) = self.emit_expr(ctx, &args[0], line)?;
                if aty != Ty::Int {
                    return Err(eerr(line, "list[int] elements must be int"));
                }
                self.used_vec = true;
                Ok((format!("{}.push({})", obj, acode), Ty::Void))
            }
            Expr::Unary(op, inner) => {
                let (code, ty) = self.emit_expr(ctx, inner, line)?;
                match op {
                    UnOp::Neg => {
                        if ty == Ty::Int || ty == Ty::Float {
                            Ok((format!("-({})", code), ty))
                        } else {
                            Err(eerr(line, "unary - needs a number"))
                        }
                    }
                    UnOp::Not => {
                        if ty == Ty::Bool {
                            Ok((format!("!({})", code), Ty::Bool))
                        } else {
                            Err(eerr(line, "not needs a bool"))
                        }
                    }
                }
            }
            Expr::Call(callee, args) => {
                let fname = match &**callee {
                    Expr::Name(n) => n.clone(),
                    _ => {
                        return Err(eerr(
                            line,
                            "v0.1 calls only plain function names (no methods, no first-class functions)",
                        ))
                    }
                };
                if fname == "print" {
                    return Err(eerr(line, "print() is a statement in v0.1, not an expression"));
                }
                if fname == "range" {
                    return Err(eerr(line, "range() is only valid in a for loop header in v0.1"));
                }
                if fname == "len" {
                    if args.len() != 1 {
                        return Err(eerr(line, "len() takes exactly one argument"));
                    }
                    let (c, t) = self.emit_expr(ctx, &args[0], line)?;
                    if t != Ty::ListInt {
                        return Err(eerr(line, "len() is only supported on list[int] in v0.2"));
                    }
                    return Ok((format!("(len({}) as i64)", c), Ty::Int));
                }
                let mut arg_codes = Vec::new();
                let mut arg_tys = Vec::new();
                for a in args {
                    let (c, t) = self.emit_expr(ctx, a, line)?;
                    arg_codes.push(c);
                    arg_tys.push(t);
                }
                let sig = self.ensure_fn(&fname, &arg_tys, line)?;
                let ret = sig.ret.unwrap_or(Ty::Void);
                // Ownership at the call site (spec s4/s6): a borrowed
                // parameter takes `&name`; a consumed parameter moves
                // the argument binding. A borrowed parameter of the
                // caller can be re-borrowed onward but never moved.
                for (i, a) in args.iter().enumerate() {
                    if arg_tys.get(i) != Some(&Ty::ListInt) {
                        continue;
                    }
                    let borrow = sig.param_borrow.get(i).copied().unwrap_or(false);
                    match a {
                        Expr::Name(n) => {
                            let caller_borrowed = ctx.borrowed.contains(n);
                            if borrow {
                                if !caller_borrowed {
                                    arg_codes[i] = format!("&{}", arg_codes[i]);
                                }
                            } else {
                                if caller_borrowed {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "cannot move `{}` into `{}`: it is a borrowed parameter of `{}` (spec s7: borrows never transfer ownership)",
                                            n, fname, ctx.fname
                                        ),
                                    ));
                                }
                                ctx.mark_moved(n, line);
                            }
                        }
                        _ => {
                            if borrow {
                                return Err(eerr(
                                    line,
                                    "passing a temporary list to a borrowing parameter is not supported in v0.2; bind it to a variable first",
                                ));
                            }
                        }
                    }
                }
                Ok((format!("{}({})", fname, arg_codes.join(", ")), ret))
            }
            Expr::Bin(l, op, r) => {
                let (lc, lt) = self.emit_expr(ctx, l, line)?;
                let (rc, rt) = self.emit_expr(ctx, r, line)?;
                self.emit_binop(lc, lt, *op, rc, rt, line)
            }
        }
    }

    fn emit_binop(
        &mut self,
        lc: String,
        lt: Ty,
        op: BinOp,
        rc: String,
        rt: Ty,
        line: u32,
    ) -> Result<(String, Ty), String> {
        match op {
            BinOp::And | BinOp::Or => {
                if lt != Ty::Bool || rt != Ty::Bool {
                    return Err(eerr(line, "and/or need bool operands"));
                }
                let sym = if op == BinOp::And { "&&" } else { "||" };
                Ok((format!("({} {} {})", lc, sym, rc), Ty::Bool))
            }
            BinOp::Eq | BinOp::NotEq => {
                let sym = if op == BinOp::Eq { "==" } else { "!=" };
                match (lt, rt) {
                    (Ty::Int, Ty::Int) | (Ty::Bool, Ty::Bool) => {
                        Ok((format!("({} {} {})", lc, sym, rc), Ty::Bool))
                    }
                    (Ty::Float, Ty::Float) => Ok((format!("({} {} {})", lc, sym, rc), Ty::Bool)),
                    (Ty::Int, Ty::Float) => {
                        Ok((format!("(({} as f64) {} {})", lc, sym, rc), Ty::Bool))
                    }
                    (Ty::Float, Ty::Int) => {
                        Ok((format!("({} {} ({} as f64))", lc, sym, rc), Ty::Bool))
                    }
                    (Ty::Str, Ty::Str) => Err(eerr(line, "string comparison is not supported in v0.1")),
                    _ => Err(eerr(line, &format!("cannot compare {:?} with {:?}", lt, rt))),
                }
            }
            BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq => {
                let sym = match op {
                    BinOp::Lt => "<",
                    BinOp::LtEq => "<=",
                    BinOp::Gt => ">",
                    _ => ">=",
                };
                match (lt, rt) {
                    (Ty::Int, Ty::Int) => Ok((format!("({} {} {})", lc, sym, rc), Ty::Bool)),
                    (Ty::Float, Ty::Float) => Ok((format!("({} {} {})", lc, sym, rc), Ty::Bool)),
                    (Ty::Int, Ty::Float) => {
                        Ok((format!("(({} as f64) {} {})", lc, sym, rc), Ty::Bool))
                    }
                    (Ty::Float, Ty::Int) => {
                        Ok((format!("({} {} ({} as f64))", lc, sym, rc), Ty::Bool))
                    }
                    _ => Err(eerr(line, "ordering comparison needs numbers")),
                }
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul => {
                let sym = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    _ => "*",
                };
                match (lt, rt) {
                    (Ty::Int, Ty::Int) => Ok((format!("({} {} {})", lc, sym, rc), Ty::Int)),
                    (Ty::Float, Ty::Float) => Ok((format!("({} {} {})", lc, sym, rc), Ty::Float)),
                    (Ty::Int, Ty::Float) => {
                        Ok((format!("(({} as f64) {} {})", lc, sym, rc), Ty::Float))
                    }
                    (Ty::Float, Ty::Int) => {
                        Ok((format!("({} {} ({} as f64))", lc, sym, rc), Ty::Float))
                    }
                    (Ty::Str, _) | (_, Ty::Str) => {
                        Err(eerr(line, "string arithmetic is not supported in v0.1"))
                    }
                    _ => Err(eerr(line, "arithmetic needs numbers")),
                }
            }
            BinOp::Div => match (lt, rt) {
                (Ty::Int, Ty::Int) => Ok((format!("(({} as f64) / ({} as f64))", lc, rc), Ty::Float)),
                (Ty::Float, Ty::Float) => Ok((format!("({} / {})", lc, rc), Ty::Float)),
                (Ty::Int, Ty::Float) => Ok((format!("(({} as f64) / {})", lc, rc), Ty::Float)),
                (Ty::Float, Ty::Int) => Ok((format!("({} / ({} as f64))", lc, rc), Ty::Float)),
                _ => Err(eerr(line, "division needs numbers")),
            },
            BinOp::FloorDiv => match (lt, rt) {
                (Ty::Int, Ty::Int) => {
                    self.used_floordiv = true;
                    Ok((format!("__psa_floordiv({}, {})", lc, rc), Ty::Int))
                }
                _ => Err(eerr(line, "// on non-integers is not supported in v0.1")),
            },
            BinOp::Mod => match (lt, rt) {
                (Ty::Int, Ty::Int) => {
                    self.used_floormod = true;
                    Ok((format!("__psa_floormod({}, {})", lc, rc), Ty::Int))
                }
                _ => Err(eerr(line, "% on non-integers is not supported in v0.1")),
            },
        }
    }
}

/// Does function `def` consume (move out of) its parameter `param`?
/// True when the parameter is returned bare, bound into another
/// variable, or passed to a call that itself consumes it. Reads,
/// index reads/writes, len(), and append() are borrows. Cycles in
/// the call graph resolve to "consumed" — the loud, safe direction
/// (spec s4: never guess a borrow to make a program pass).
fn param_usage(em: &Emitter, def: &Func, param: &str, visiting: &mut Vec<(String, String)>) -> bool {
    fn expr_consumes(
        em: &Emitter,
        e: &Expr,
        param: &str,
        visiting: &mut Vec<(String, String)>,
    ) -> bool {
        match e {
            Expr::Call(callee, args) => {
                let gname = match &**callee {
                    Expr::Name(n) => Some(n.clone()),
                    _ => None,
                };
                for (i, a) in args.iter().enumerate() {
                    if let Expr::Name(n) = a {
                        if n == param {
                            if let (Some(g), Some(gdef)) =
                                (gname.clone(), gname.as_ref().and_then(|g| em.defs.get(g)))
                            {
                                if let Some(gparam) = gdef.params.get(i) {
                                    let key = (g.clone(), gparam.clone());
                                    if visiting.contains(&key) {
                                        return true;
                                    }
                                    visiting.push(key);
                                    let used =
                                        param_usage(em, gdef, gparam, visiting);
                                    visiting.pop();
                                    if used {
                                        return true;
                                    }
                                }
                            }
                            // Builtins (len) borrow; unknown callees were
                            // already rejected by ensure_fn upstream.
                            continue;
                        }
                    }
                    if expr_consumes(em, a, param, visiting) {
                        return true;
                    }
                }
                false
            }
            Expr::Bin(l, _, r) => {
                expr_consumes(em, l, param, visiting) || expr_consumes(em, r, param, visiting)
            }
            Expr::Unary(_, i) => expr_consumes(em, i, param, visiting),
            Expr::Index(base, idx) => {
                // Indexing the parameter itself is a read (borrow).
                let _ = base;
                expr_consumes(em, idx, param, visiting)
            }
            Expr::MethodCall { args, .. } => {
                args.iter().any(|a| expr_consumes(em, a, param, visiting))
            }
            Expr::ListLit(elems) => elems.iter().any(|a| expr_consumes(em, a, param, visiting)),
            _ => false,
        }
    }

    fn stmts_consume(
        em: &Emitter,
        stmts: &[Stmt],
        param: &str,
        visiting: &mut Vec<(String, String)>,
    ) -> bool {
        for s in stmts {
            let hit = match s {
                Stmt::Assign { value, .. } => {
                    matches!(value, Expr::Name(n) if n == param)
                        || expr_consumes(em, value, param, visiting)
                }
                Stmt::AugAssign { value, .. } => expr_consumes(em, value, param, visiting),
                Stmt::ExprStmt { value, .. } => expr_consumes(em, value, param, visiting),
                Stmt::Return { value, .. } => match value {
                    Some(Expr::Name(n)) => n == param,
                    Some(e) => expr_consumes(em, e, param, visiting),
                    None => false,
                },
                Stmt::If { arms, els, .. } => {
                    arms.iter().any(|(c, b)| {
                        expr_consumes(em, c, param, visiting)
                            || stmts_consume(em, b, param, visiting)
                    }) || els
                        .as_ref()
                        .map(|b| stmts_consume(em, b, param, visiting))
                        .unwrap_or(false)
                }
                Stmt::While { cond, body, .. } => {
                    expr_consumes(em, cond, param, visiting)
                        || stmts_consume(em, body, param, visiting)
                }
                Stmt::ForRange { start, stop, body, .. } => {
                    start
                        .as_ref()
                        .map(|e| expr_consumes(em, e, param, visiting))
                        .unwrap_or(false)
                        || expr_consumes(em, stop, param, visiting)
                        || stmts_consume(em, body, param, visiting)
                }
                Stmt::IndexAssign { value, index, .. } => {
                    expr_consumes(em, value, param, visiting)
                        || expr_consumes(em, index, param, visiting)
                }
                _ => false,
            };
            if hit {
                return true;
            }
        }
        false
    }

    stmts_consume(em, &def.body, param, visiting)
}

fn unify_ret(ctx: &mut Ctx, ty: Ty, line: u32) -> Result<(), String> {
    match ctx.ret {
        None => {
            ctx.ret = Some(ty);
            Ok(())
        }
        Some(existing) => {
            if existing == ty {
                Ok(())
            } else if existing == Ty::Float && ty == Ty::Int {
                Ok(())
            } else if existing == Ty::Int && ty == Ty::Float {
                ctx.ret = Some(Ty::Float);
                Ok(())
            } else {
                Err(eerr(
                    line,
                    &format!("return type mismatch in `{}`: {:?} vs {:?}", ctx.fname, existing, ty),
                ))
            }
        }
    }
}

fn def_calls(stmts: &[Stmt], name: &str) -> bool {
    fn expr_calls(e: &Expr, name: &str) -> bool {
        match e {
            Expr::Call(callee, args) => {
                if let Expr::Name(n) = &**callee {
                    if n == name {
                        return true;
                    }
                }
                args.iter().any(|a| expr_calls(a, name))
            }
            Expr::Bin(l, _, r) => expr_calls(l, name) || expr_calls(r, name),
            Expr::Unary(_, i) => expr_calls(i, name),
            _ => false,
        }
    }
    for s in stmts {
        let hit = match s {
            Stmt::Assign { value, .. } => expr_calls(value, name),
            Stmt::AugAssign { value, .. } => expr_calls(value, name),
            Stmt::ExprStmt { value, .. } => expr_calls(value, name),
            Stmt::Return { value, .. } => value.as_ref().map(|v| expr_calls(v, name)).unwrap_or(false),
            Stmt::If { arms, els, .. } => {
                arms.iter().any(|(c, b)| expr_calls(c, name) || def_calls(b, name))
                    || els.as_ref().map(|b| def_calls(b, name)).unwrap_or(false)
            }
            Stmt::While { cond, body, .. } => expr_calls(cond, name) || def_calls(body, name),
            Stmt::ForRange { start, stop, body, .. } => {
                start.as_ref().map(|e| expr_calls(e, name)).unwrap_or(false)
                    || expr_calls(stop, name)
                    || def_calls(body, name)
            }
            _ => false,
        };
        if hit {
            return true;
        }
    }
    false
}

fn fmt_f64(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

fn escape_sla_str(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}
