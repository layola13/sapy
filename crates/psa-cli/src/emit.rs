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
    ListFloat,
    ListBool,
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
            Ty::ListFloat => "Vec<f64>",
            Ty::ListBool => "Vec<bool>",
            Ty::Void => "i32",
        }
    }
    fn default_lit(&self) -> &'static str {
        match self {
            Ty::Int => "0",
            Ty::Float => "0.0",
            Ty::Bool => "false",
            Ty::Str => "\"\"",
            Ty::ListInt | Ty::ListFloat | Ty::ListBool => "Vec::new()",
            Ty::Void => "0",
        }
    }
    fn is_list(&self) -> bool {
        matches!(self, Ty::ListInt | Ty::ListFloat | Ty::ListBool)
    }
    /// Element type of a list type (scalar elements only in v0.3).
    fn elem(&self) -> Option<Ty> {
        match self {
            Ty::ListInt => Some(Ty::Int),
            Ty::ListFloat => Some(Ty::Float),
            Ty::ListBool => Some(Ty::Bool),
            _ => None,
        }
    }
    /// List type for a scalar element type; None for non-scalars.
    fn list_of(elem: Ty) -> Option<Ty> {
        match elem {
            Ty::Int => Some(Ty::ListInt),
            Ty::Float => Some(Ty::ListFloat),
            Ty::Bool => Some(Ty::ListBool),
            _ => None,
        }
    }
    /// Python-style spelling for diagnostics, e.g. `list[int]`.
    fn py_name(&self) -> String {
        match self {
            Ty::ListInt => "list[int]".to_string(),
            Ty::ListFloat => "list[float]".to_string(),
            Ty::ListBool => "list[bool]".to_string(),
            other => format!("{:?}", other),
        }
    }
}

#[derive(Debug, Clone)]
struct Sig {
    params: Vec<Ty>,
    /// Per-parameter: true when a list parameter is borrowed
    /// (SLA `&Vec<T>`) rather than consumed (SLA `Vec<T>`).
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
    used_ipow: bool,
    used_math: bool,
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
    /// True while emitting a while-condition: if-expressions are rejected
    /// there because their materialised temp would only be computed once,
    /// before the loop, not on every evaluation of the condition.
    in_while_cond: bool,
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
        used_ipow: false,
        used_math: false,
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
        in_while_cond: false,
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
    if em.used_math {
        out.push_str("@import \"sa_std/math.sai\"\n");
    }
    out.push('\n');
    if em.used_ipow {
        out.push_str(
            "fn __psa_ipow(a: i64, e: i64) -> i64 {\n    let r = 1;\n    let b = a;\n    let n = e;\n    while n > 0 {\n        if (n & 1) == 1 { r = r * b; };\n        b = b * b;\n        n = n >> 1;\n    }\n    return r;\n}\n\n",
        );
    }
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
        if arg_tys.iter().any(|t| *t == Ty::Str) {
            return Err(eerr(
                line,
                "passing a string value to a function is blocked by upstream defect (xxv): string bindings cannot be lowered on the Rust sla build-exe route (see README)",
            ));
        }
        // Ownership per parameter, from the callee body alone (spec
        // s4): a list parameter the body only reads or writes through
        // is borrowed (SLA &Vec<T>); one it moves out of is consumed
        // (SLA Vec<T>). Scalars are Copy and never borrow.
        let borrows: Vec<bool> = def
            .params
            .iter()
            .zip(arg_tys.iter())
            .map(|(p, t)| {
                if !t.is_list() {
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
            in_while_cond: false,
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
            if t.is_list() && borrows.get(i).copied().unwrap_or(false) {
                txt.push_str(&format!("{}: &{}", p, t.sla()));
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

    /// Type-check a list literal's elements and emit their codes.
    /// Returns the list type and one code string per element. Element
    /// types must agree on one scalar type (int/float/bool in v0.3;
    /// string elements are heap values and stay out of scope).
    fn emit_list_literal(
        &mut self,
        ctx: &mut Ctx,
        elems: &[Expr],
        line: u32,
    ) -> Result<(Ty, Vec<String>), String> {
        if elems.is_empty() {
            return Err(eerr(
                line,
                "cannot infer the element type of an empty list literal in v0.3; bind a non-empty list first",
            ));
        }
        let mut codes = Vec::new();
        let mut elem_ty: Option<Ty> = None;
        for e in elems {
            let (code, ty) = self.emit_expr(ctx, e, line)?;
            match elem_ty {
                None => {
                    if ty == Ty::Float {
                        return Err(eerr(
                            line,
                            "list[float] is blocked in v0.3: Vec<f64> element access is miscompiled on both sla routes (hand-written SLA probes independent of psa read pushed f64 elements back wrong on the Rust route and the Zig route alike) — psa refuses to emit silently wrong programs (upstream defect (xxiv), see README)",
                        ));
                    }
                    if Ty::list_of(ty).is_none() {
                        return Err(eerr(
                            line,
                            &format!(
                                "list elements of type {:?} are not supported in v0.3 (scalar int/float/bool elements only)",
                                ty
                            ),
                        ));
                    }
                    elem_ty = Some(ty);
                }
                Some(want) => {
                    if ty != want {
                        return Err(eerr(
                            line,
                            &format!(
                                "list literal mixes element types {:?} and {:?} (one element type per list)",
                                want, ty
                            ),
                        ));
                    }
                }
            }
            codes.push(code);
        }
        Ok((Ty::list_of(elem_ty.unwrap()).unwrap(), codes))
    }

    /// Emit `let <name>: Vec<T> = Vec::new();` (or a plain assignment
    /// when `declare` is false) followed by one push per element code.
    fn emit_list_construct(
        &mut self,
        ctx: &mut Ctx,
        name: &str,
        list_ty: Ty,
        codes: &[String],
        declare: bool,
    ) {
        self.used_vec = true;
        if declare {
            ctx.line(&format!("let {}: {} = Vec::new();", name, list_ty.sla()));
        } else {
            ctx.line(&format!("{} = Vec::new();", name));
        }
        for code in codes {
            ctx.line(&format!("{}.push({});", name, code));
        }
    }

    fn emit_list_assign(
        &mut self,
        ctx: &mut Ctx,
        name: &str,
        elems: &[Expr],
        line: u32,
    ) -> Result<(), String> {
        let (list_ty, codes) = self.emit_list_literal(ctx, elems, line)?;
        match ctx.lookup(name) {
            None => {
                ctx.declare(name, list_ty).map_err(|m| eerr(line, &m))?;
                self.emit_list_construct(ctx, name, list_ty, &codes, true);
                Ok(())
            }
            Some(existing) if existing == list_ty => {
                self.emit_list_construct(ctx, name, list_ty, &codes, false);
                ctx.revive(name);
                Ok(())
            }
            Some(other) => Err(eerr(
                line,
                &format!(
                    "variable `{}` changes type from {} to {}",
                    name,
                    other.py_name(),
                    list_ty.py_name()
                ),
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
                        let (list_ty, codes) = self.emit_list_literal(ctx, elems, *line)?;
                        ctx.tmp += 1;
                        let tmp = format!("__psa_ret_{}", ctx.tmp);
                        self.emit_list_construct(ctx, &tmp, list_ty, &codes, true);
                        unify_ret(ctx, list_ty, *line)?;
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
                        if ty == Ty::Str {
                            return Err(eerr(
                                *line,
                                "returning a string value is blocked by upstream defect (xxv): string bindings cannot be lowered on the Rust sla build-exe route (see README)",
                            ));
                        }
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
                if ty == Ty::Str {
                    return Err(eerr(
                        *line,
                        "binding a string value to a variable is blocked by upstream defect (xxv): string bindings cannot be lowered on the Rust sla build-exe route (see README); string literals in print() work",
                    ));
                }
                // Container assignment moves the source binding (spec s3).
                let src_name = match value {
                    Expr::Name(n) if ty.is_list() => Some(n.clone()),
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
                            let nt = expr_mentions(value, name) && expr_has_call(value);
                            assign_line(ctx, name, &code, ty, false, nt)
                                .map_err(|m| eerr(*line, &m))?;
                            ctx.revive(name);
                        } else if existing == Ty::Float && ty == Ty::Int {
                            let nt = expr_mentions(value, name) && expr_has_call(value);
                            assign_line(ctx, name, &code, ty, true, nt)
                                .map_err(|m| eerr(*line, &m))?;
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
                let elem = match ty.elem() {
                    Some(e) => e,
                    None => {
                        return Err(eerr(
                            *line,
                            &format!("index assignment is only supported on lists, not on {}", ty.py_name()),
                        ))
                    }
                };
                ctx.check_alive(name, *line)?;
                let (icode, ity) = self.emit_expr(ctx, index, *line)?;
                if ity != Ty::Int {
                    return Err(eerr(*line, "list index must be int"));
                }
                let (vcode, vty) = self.emit_expr(ctx, value, *line)?;
                if vty != elem {
                    return Err(eerr(
                        *line,
                        &format!("{} elements must be {:?}, assigned {:?}", ty.py_name(), elem, vty),
                    ));
                }
                match op {
                    None => ctx.line(&format!("{}[{}] = {};", name, icode, vcode)),
                    Some(bop) => {
                        let place = format!("{}[{}]", name, icode);
                        let (code, _) = self.emit_binop(place.clone(), elem, *bop, vcode, elem, *line)?;
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
                let needs_temp = matches!(op, BinOp::FloorDiv | BinOp::Mod | BinOp::Pow)
                    || expr_has_call(value);
                if ty != existing {
                    if existing == Ty::Float && ty == Ty::Int {
                        assign_line(ctx, name, &code, ty, true, needs_temp)
                            .map_err(|m| eerr(*line, &m))?;
                        return Ok(());
                    }
                    return Err(eerr(
                        *line,
                        &format!("augmented assignment changes `{}` from {:?} to {:?}", name, existing, ty),
                    ));
                }
                assign_line(ctx, name, &code, ty, false, needs_temp)
                    .map_err(|m| eerr(*line, &m))?;
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
                let saved = ctx.in_while_cond;
                ctx.in_while_cond = true;
                let res = self.emit_expr(ctx, cond, *line);
                ctx.in_while_cond = saved;
                let (ccode, cty) = res?;
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
        // An f-string as the sole argument lowers to one println whose
        // format string is assembled from its literal parts.
        if args.len() == 1 {
            if let Expr::FStr(parts) = &args[0] {
                return self.emit_print_fstr(ctx, parts, line);
            }
        } else if args.iter().any(|a| matches!(a, Expr::FStr(_))) {
            return Err(eerr(line, "an f-string must be the only print argument in D0"));
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
            if ty.is_list() {
                return Err(eerr(line, "printing a whole list is not supported in v0.3 (print its elements or len)"));
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
            if ty.is_list() {
                return Err(eerr(line, "printing a whole list is not supported in v0.3 (print its elements or len)"));
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

    /// print(f"...") — assemble println output from the f-string's parts.
    /// Literal parts become format text (braces are rejected: SLA's
    /// println has no brace escaping, as a probe showed `{{}}` printing
    /// a lone `{`); interpolated ints become `{}` holes. A single bool
    /// interpolation is rendered by emitting both println variants under
    /// an if/else (there is no string temp to hold "True"/"False" —
    /// string bindings are blocked upstream, defect (xxv)).
    fn emit_print_fstr(
        &mut self,
        ctx: &mut Ctx,
        parts: &[FStrPart],
        line: u32,
    ) -> Result<(), String> {
        let mut fmt = String::new();
        let mut codes: Vec<String> = Vec::new();
        let mut bool_hole: Option<(String, usize)> = None;
        for part in parts {
            match part {
                FStrPart::Lit(s) => {
                    if s.contains('{') || s.contains('}') {
                        return Err(eerr(
                            line,
                            "f-string literal text containing braces is not supported in D0 (SLA println has no brace escaping)",
                        ));
                    }
                    fmt.push_str(&escape_sla_str(s));
                }
                FStrPart::Expr(e) => {
                    if let Expr::Str(s) = e {
                        if s.contains('{') || s.contains('}') {
                            return Err(eerr(
                                line,
                                "f-string literal text containing braces is not supported in D0 (SLA println has no brace escaping)",
                            ));
                        }
                        fmt.push_str(&escape_sla_str(s));
                        continue;
                    }
                    let (code, ty) = self.emit_expr(ctx, e, line)?;
                    match ty {
                        Ty::Int => {
                            fmt.push_str("{}");
                            codes.push(code);
                        }
                        Ty::Bool => {
                            if bool_hole.is_some() {
                                return Err(eerr(
                                    line,
                                    "f-strings with more than one bool interpolation are not supported in D0",
                                ));
                            }
                            bool_hole = Some((code, fmt.len()));
                        }
                        Ty::Str => {
                            return Err(eerr(
                                line,
                                "interpolating a string value in an f-string is blocked by upstream defect (xxv): string values cannot be held in variables on the Rust sla build-exe route (see README); interpolate a literal or a number",
                            ));
                        }
                        Ty::Float => {
                            return Err(eerr(
                                line,
                                "interpolating a float in an f-string is blocked: the Rust sla build-exe route truncates f64 in println (upstream defect, see README)",
                            ));
                        }
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "values of type {} cannot be interpolated in an f-string in D0",
                                    ty.py_name()
                                ),
                            ));
                        }
                    }
                }
            }
        }
        let args = if codes.is_empty() {
            String::new()
        } else {
            format!(", {}", codes.join(", "))
        };
        match bool_hole {
            None => {
                ctx.line(&format!("println(\"{}\"{}) ;", fmt, args));
            }
            Some((bcode, pos)) => {
                let fmt_true = format!("{}True{}", &fmt[..pos], &fmt[pos..]);
                let fmt_false = format!("{}False{}", &fmt[..pos], &fmt[pos..]);
                ctx.line(&format!(
                    "if {} {{ println(\"{}\"{}) ; }} else {{ println(\"{}\"{}) ; }};",
                    bcode, fmt_true, args, fmt_false, args
                ));
            }
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
                "list literals are only supported as an assignment value or a return value in v0.3",
            )),
            Expr::Index(base, idx) => {
                let obj = match &**base {
                    Expr::Name(n) => n.clone(),
                    _ => {
                        return Err(eerr(
                            line,
                            "indexing is only supported on plain variables in v0.3",
                        ))
                    }
                };
                let ty = ctx
                    .lookup(&obj)
                    .ok_or_else(|| eerr(line, &format!("undefined variable `{}`", obj)))?;
                let elem = match ty.elem() {
                    Some(e) => e,
                    None => {
                        return Err(eerr(
                            line,
                            &format!("indexing is only supported on lists, not on {}", ty.py_name()),
                        ))
                    }
                };
                ctx.check_alive(&obj, line)?;
                let (icode, ity) = self.emit_expr(ctx, idx, line)?;
                if ity != Ty::Int {
                    return Err(eerr(line, "list index must be int"));
                }
                // Scalar element: copied out of the borrow (spec s5).
                Ok((format!("{}[{}]", obj, icode), elem))
            }
            Expr::MethodCall { obj, method, args } => {
                let ty = ctx
                    .lookup(obj)
                    .ok_or_else(|| eerr(line, &format!("undefined variable `{}`", obj)))?;
                let elem = match ty.elem() {
                    Some(e) if method == "append" => e,
                    _ => {
                        return Err(eerr(
                            line,
                            &format!(
                                "method `{}` is not supported in v0.3 (only list.append)",
                                method
                            ),
                        ))
                    }
                };
                ctx.check_alive(obj, line)?;
                if args.len() != 1 {
                    return Err(eerr(line, "append takes exactly one argument"));
                }
                let (acode, aty) = self.emit_expr(ctx, &args[0], line)?;
                if aty != elem {
                    return Err(eerr(
                        line,
                        &format!("{} elements must be {:?}, appended {:?}", ty.py_name(), elem, aty),
                    ));
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
                    UnOp::BitNot => {
                        if ty == Ty::Int {
                            // ~x == -x - 1 exactly; no SLA operator needed.
                            Ok((format!("(-({}) - 1)", code), Ty::Int))
                        } else {
                            Err(eerr(line, "~ needs an int operand"))
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
                    if !t.is_list() {
                        return Err(eerr(line, "len() is only supported on lists in v0.3"));
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
                    if !arg_tys.get(i).map(|t| t.is_list()).unwrap_or(false) {
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
                                    "passing a temporary list to a borrowing parameter is not supported in v0.3; bind it to a variable first",
                                ));
                            }
                        }
                    }
                }
                Ok((format!("{}({})", fname, arg_codes.join(", ")), ret))
            }
            Expr::Bin(l, op, r) => {
                if *op == BinOp::Pow {
                    return self.emit_pow(ctx, l, r, line);
                }
                let (lc, lt) = self.emit_expr(ctx, l, line)?;
                let (rc, rt) = self.emit_expr(ctx, r, line)?;
                self.emit_binop(lc, lt, *op, rc, rt, line)
            }
            Expr::IfExp { cond, then, els } => {
                if ctx.in_while_cond {
                    return Err(eerr(
                        line,
                        "if-expressions are not supported in while conditions in D0 (the condition is re-evaluated every iteration); compute the value before the loop",
                    ));
                }
                let (cc, ct) = self.emit_expr(ctx, cond, line)?;
                if ct != Ty::Bool {
                    return Err(eerr(line, "if-expression condition must be bool"));
                }
                // Emit each branch into a scratch buffer so any lines a
                // nested if-expression produces land *inside* the branch
                // block, not before the whole expression.
                let saved_out = std::mem::take(&mut ctx.out);
                ctx.indent += 1;
                let then_res = self.emit_expr(ctx, then, line);
                let then_lines = std::mem::replace(&mut ctx.out, saved_out);
                let (tc, tt) = then_res?;
                let saved_out = std::mem::take(&mut ctx.out);
                let els_res = self.emit_expr(ctx, els, line);
                let els_lines = std::mem::replace(&mut ctx.out, saved_out);
                ctx.indent -= 1;
                let (ec, et) = els_res?;
                if tt.is_list() || et.is_list() {
                    return Err(eerr(
                        line,
                        "if-expression branches cannot be lists in D0 (a branch value would move out of its scope)",
                    ));
                }
                if tt == Ty::Str || et == Ty::Str {
                    return Err(eerr(
                        line,
                        "if-expressions with string branches are blocked by upstream defect (xxv): the value would have to live in a string temp, and string bindings cannot be lowered on the Rust sla build-exe route (see README)",
                    ));
                }
                let (ty, tc, ec) = match (tt, et) {
                    (a, b) if a == b => (a, tc, ec),
                    (Ty::Float, Ty::Int) => (Ty::Float, tc, format!("{} as f64", ec)),
                    (Ty::Int, Ty::Float) => (Ty::Float, format!("{} as f64", tc), ec),
                    _ => {
                        return Err(eerr(
                            line,
                            &format!(
                                "if-expression branches have different types ({} and {}); Python would make this value dynamic — the dynamic face is D1+",
                                tt.py_name(),
                                et.py_name()
                            ),
                        ))
                    }
                };
                if ty == Ty::Void {
                    return Err(eerr(line, "if-expression branches produce no value"));
                }
                ctx.tmp += 1;
                let name = format!("__psa_ifexp_{}", ctx.tmp);
                ctx.declare(&name, ty).map_err(|m| eerr(line, &m))?;
                ctx.line(&format!("let {}: {} = {};", name, ty.sla(), ty.default_lit()));
                ctx.line(&format!("if {} {{", cc));
                ctx.out.push_str(&then_lines);
                ctx.indent += 1;
                ctx.line(&format!("{} = {};", name, tc));
                ctx.indent -= 1;
                ctx.line("} else {");
                ctx.out.push_str(&els_lines);
                ctx.indent += 1;
                ctx.line(&format!("{} = {};", name, ec));
                ctx.indent -= 1;
                ctx.line("};");
                Ok((name, ty))
            }
            Expr::FStr(_) => Err(eerr(
                line,
                "f-strings are supported as the sole argument of print() in D0; a general string value needs the string runtime (D2)",
            )),
        }
    }

    /// `**`: int ** int with a literal non-negative exponent uses the
    /// exact integer helper; anything involving a float (or a negative
    /// literal int exponent, whose Python result is a float) goes through
    /// sa_std's `sa_math_pow`. A non-literal int exponent is rejected:
    /// its Python result type depends on the runtime sign.
    fn emit_pow(
        &mut self,
        ctx: &mut Ctx,
        l: &Expr,
        r: &Expr,
        line: u32,
    ) -> Result<(String, Ty), String> {
        let (lc, lt) = self.emit_expr(ctx, l, line)?;
        let (rc, rt) = self.emit_expr(ctx, r, line)?;
        match (lt, rt) {
            (Ty::Int, Ty::Int) => match int_literal(r) {
                Some(e) if e >= 0 => {
                    self.used_ipow = true;
                    Ok((format!("__psa_ipow({}, {})", lc, rc), Ty::Int))
                }
                Some(_) => {
                    self.used_math = true;
                    Ok((
                        format!("sa_math_pow({} as f64, {} as f64)", lc, rc),
                        Ty::Float,
                    ))
                }
                None => Err(eerr(
                    line,
                    "int ** needs a literal exponent in D0: with a computed exponent the Python result type (int vs float) depends on its sign — use a float base, or wait for the dynamic face (D1+)",
                )),
            },
            (Ty::Float, Ty::Float)
            | (Ty::Int, Ty::Float)
            | (Ty::Float, Ty::Int) => {
                self.used_math = true;
                let lf = if lt == Ty::Int {
                    format!("{} as f64", lc)
                } else {
                    lc
                };
                let rf = if rt == Ty::Int {
                    format!("{} as f64", rc)
                } else {
                    rc
                };
                Ok((format!("sa_math_pow({}, {})", lf, rf), Ty::Float))
            }
            _ => Err(eerr(line, "** needs numeric operands")),
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
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                if lt != Ty::Int || rt != Ty::Int {
                    return Err(eerr(
                        line,
                        "bitwise operators need int operands in D0 (Python also allows them on bools; that arrives with the dynamic face)",
                    ));
                }
                let sym = match op {
                    BinOp::BitAnd => "&",
                    BinOp::BitOr => "|",
                    BinOp::BitXor => "^",
                    BinOp::Shl => "<<",
                    _ => ">>",
                };
                Ok((format!("({} {} {})", lc, sym, rc), Ty::Int))
            }
            BinOp::Pow => Err(eerr(line, "internal error: ** is handled by emit_pow")),
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

/// Emit `name = code;` (with an `as f64` widening cast when asked).
/// When the value is a call that mentions the target itself, the direct
/// shape `x = f(x, ...)` leaves a live register at function exit in the
/// Rust sla build-exe lowering (defect (xxvi)); routing the value
/// through a fresh temp first is the sound shape, so do that.
fn assign_line(
    ctx: &mut Ctx,
    name: &str,
    code: &str,
    ty: Ty,
    widen: bool,
    needs_temp: bool,
) -> Result<(), String> {
    if needs_temp {
        ctx.tmp += 1;
        let tmp = format!("__psa_asn_{}", ctx.tmp);
        ctx.declare(&tmp, ty)?;
        ctx.line(&format!("let {}: {} = {};", tmp, ty.sla(), code));
        if widen {
            ctx.line(&format!("{} = {} as f64;", name, tmp));
        } else {
            ctx.line(&format!("{} = {};", name, tmp));
        }
    } else if widen {
        ctx.line(&format!("{} = {} as f64;", name, code));
    } else {
        ctx.line(&format!("{} = {};", name, code));
    }
    Ok(())
}

/// Does the expression mention variable `name` anywhere?
fn expr_mentions(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Name(n) => n == name,
        Expr::Bin(a, _, b) => expr_mentions(a, name) || expr_mentions(b, name),
        Expr::Unary(_, a) => expr_mentions(a, name),
        Expr::Call(f, args) => {
            expr_mentions(f, name) || args.iter().any(|a| expr_mentions(a, name))
        }
        Expr::Index(a, b) => expr_mentions(a, name) || expr_mentions(b, name),
        Expr::MethodCall { obj, args, .. } => {
            obj == name || args.iter().any(|a| expr_mentions(a, name))
        }
        Expr::ListLit(elems) => elems.iter().any(|a| expr_mentions(a, name)),
        Expr::IfExp { cond, then, els } => {
            expr_mentions(cond, name)
                || expr_mentions(then, name)
                || expr_mentions(els, name)
        }
        Expr::FStr(parts) => parts
            .iter()
            .any(|p| matches!(p, FStrPart::Expr(x) if expr_mentions(x, name))),
        _ => false,
    }
}

/// Does the expression contain a call — including the helper calls that
/// `//`, `%` (on ints) and `**` lower to?
fn expr_has_call(e: &Expr) -> bool {
    match e {
        Expr::Call(..) | Expr::MethodCall { .. } => true,
        Expr::Bin(a, op, b) => {
            matches!(op, BinOp::FloorDiv | BinOp::Mod | BinOp::Pow)
                || expr_has_call(a)
                || expr_has_call(b)
        }
        Expr::Unary(_, a) => expr_has_call(a),
        Expr::Index(a, b) => expr_has_call(a) || expr_has_call(b),
        Expr::ListLit(elems) => elems.iter().any(expr_has_call),
        Expr::IfExp { cond, then, els } => {
            expr_has_call(cond) || expr_has_call(then) || expr_has_call(els)
        }
        Expr::FStr(parts) => parts
            .iter()
            .any(|p| matches!(p, FStrPart::Expr(x) if expr_has_call(x))),
        _ => false,
    }
}

/// The value of an integer literal expression, including a negated one
/// (`-3` parses as unary minus over a literal).
fn int_literal(e: &Expr) -> Option<i64> {
    match e {
        Expr::Int(v) => Some(*v),
        Expr::Unary(UnOp::Neg, inner) => int_literal(inner).map(|v| -v),
        _ => None,
    }
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
