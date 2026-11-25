//! psa — the SA-Py compiler driver (v0.1).
//!
//! Pipeline: .psa source -> lexer/parser (subset) -> type inference ->
//! SLA source -> the `sla` binary's pipeline (check / SA emission) ->
//! the `sa` toolchain (build-exe). The psa layer never claims a feature
//! it cannot carry through that whole chain.

mod ast;
mod emit;
mod lexer;
mod parser;

use std::path::{Path, PathBuf};
use std::process::{exit, Command};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
        exit(2);
    }
    let cmd = args[1].as_str();
    match cmd {
        "emit" | "check" | "build-exe" => {
            if args.len() < 3 {
                eprintln!("psa {}: missing .psa file", cmd);
                exit(2);
            }
            let file = PathBuf::from(&args[2]);
            let out_path = parse_out(&args[3..]);
            if let Err(msg) = run(cmd, &file, out_path.as_deref()) {
                eprintln!("{}", msg);
                exit(1);
            }
        }
        "version" | "--version" | "-V" => {
            println!("psa 0.8.0 (sapy v0.8, D2 slices A+B+C: dynamic lists, for-each, in, dynamic dicts, list repr, dynamic strings, nested containers, slices)");
        }
        "help" | "--help" | "-h" => usage(),
        other => {
            eprintln!("psa: unknown command `{}`", other);
            usage();
            exit(2);
        }
    }
}

fn usage() {
    eprintln!(
        "usage: psa <command> <file.psa> [options]\n\nCommands:\n  emit      print the generated SLA source\n  check     type-check through the sla pipeline\n  build-exe build a native executable (delegates to sla + sa)\n            [-o <output>] (default: ./<stem>)\n  version   print the psa version"
    );
}

fn parse_out(args: &[String]) -> Option<PathBuf> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "-o" && i + 1 < args.len() {
            return Some(PathBuf::from(&args[i + 1]));
        }
        i += 1;
    }
    None
}

fn run(cmd: &str, file: &Path, out_path: Option<&Path>) -> Result<(), String> {
    if file.extension().map(|e| e != "psa").unwrap_or(true) {
        return Err(format!("psa: {} is not a .psa file", file.display()));
    }
    let src = std::fs::read_to_string(file)
        .map_err(|e| format!("psa: cannot read {}: {}", file.display(), e))?;
    let toks = lexer::lex(&src)?;
    let mut p = parser::Parser::new(toks);
    let mut prog = p.parse_program()?;
    let base_dir = file
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    merge_imports(&mut prog, &base_dir)?;
    let sla_src = emit::emit_program(&prog)?;
    if cmd == "emit" {
        print!("{}", sla_src);
        return Ok(());
    }
    // Write the generated SLA next to the source and delegate.
    let stem = file.file_stem().unwrap().to_string_lossy().into_owned();
    let dir = file.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
    let sla_path = dir.join(format!("{}.sla", stem));
    std::fs::write(&sla_path, &sla_src)
        .map_err(|e| format!("psa: cannot write {}: {}", sla_path.display(), e))?;
    let sla_bin = std::env::var("PSA_SLA").unwrap_or_else(|_| "sla".to_string());
    let mut command = Command::new(&sla_bin);
    command.arg(cmd).arg(&sla_path);
    // Help the sla pipeline find sa_std when the caller has not pinned
    // it: walk ancestors of the source file for sci/sa_std or sa_std.
    if std::env::var_os("SA_STD_DIR").is_none() {
        // Candidates, in order: ancestors of the source file, ancestors
        // of the sla binary (the toolchain tree layout), the cwd.
        let mut found = find_sa_std(&dir);
        if found.is_none() && Path::new(&sla_bin).is_absolute() {
            if let Some(parent) = Path::new(&sla_bin).parent() {
                found = find_sa_std(parent);
            }
        }
        if found.is_none() {
            if let Ok(cwd) = std::env::current_dir() {
                found = find_sa_std(&cwd);
            }
        }
        if let Some(std_dir) = found {
            command.env("SA_STD_DIR", std_dir);
        }
    }
    command.env("SLA_NO_FRESHNESS_CACHE", "1");
    let status = command
        .status()
        .map_err(|e| format!("psa: cannot run `{}` (set PSA_SLA to the sla binary): {}", sla_bin, e))?;
    if !status.success() {
        return Err(format!("psa: sla {} failed with {}", cmd, status));
    }
    if cmd == "build-exe" {
        // The sla pipeline leaves the binary at
        // <dir>/.sla-cache/sab/.sa_cache/build-exe/<hash>/output.bin
        let cache = dir.join(".sla-cache").join("sab").join(".sa_cache").join("build-exe");
        let bin = newest_output_bin(&cache).ok_or_else(|| {
            "psa: sla reported success but no output.bin was found under .sla-cache".to_string()
        })?;
        let target = out_path.map(|p| p.to_path_buf()).unwrap_or_else(|| {
            std::env::current_dir().unwrap().join(&stem)
        });
        std::fs::copy(&bin, &target)
            .map_err(|e| format!("psa: cannot copy binary to {}: {}", target.display(), e))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&target).unwrap().permissions();
            perms.set_mode(0o755);
            let _ = std::fs::set_permissions(&target, perms);
        }
        println!("psa: built {}", target.display());
    }
    Ok(())
}

fn find_sa_std(start: &Path) -> Option<PathBuf> {
    let mut cur = Some(start.to_path_buf());
    while let Some(d) = cur {
        let cand = d.join("sci").join("sa_std");
        if cand.is_dir() {
            return Some(cand);
        }
        let cand2 = d.join("sa_std");
        if cand2.is_dir() {
            return Some(cand2);
        }
        cur = d.parent().map(|p| p.to_path_buf());
    }
    None
}

/// D6 import resolution: merge imported module files into the
/// program before emission. A module file lives next to the
/// importing file (dotted names map to subdirectories). Only
/// defs, classes and literal assignments may appear at a
/// module's top level; anything else is refused loudly.
/// Module defs are renamed to dotted keys (`foo.bar`), module
/// classes join the shared class registry under their plain
/// names, and module literal attributes are recorded under
/// dotted keys for read-site evaluation. Circular imports are
/// refused. From-import bindings are program-wide in v1 (a
/// documented simplification of CPython's module-local
/// bindings).
fn merge_imports(prog: &mut ast::Program, base_dir: &Path) -> Result<(), String> {
    use std::collections::HashSet;
    let imports = std::mem::take(&mut prog.imports);
    if imports.is_empty() {
        return Ok(());
    }
    let mut visiting: HashSet<String> = HashSet::new();
    let mut done: HashSet<String> = HashSet::new();
    for imp in &imports {
        load_module(prog, base_dir, &imp.module, &mut visiting, &mut done, imp.line)?;
        if let Some(names) = &imp.names {
            bind_from_import(prog, &imp.module, names, imp.line)?;
        }
    }
    Ok(())
}

fn load_module(
    prog: &mut ast::Program,
    base_dir: &Path,
    key: &str,
    visiting: &mut std::collections::HashSet<String>,
    done: &mut std::collections::HashSet<String>,
    line: u32,
) -> Result<(), String> {
    use ast::Stmt;
    if done.contains(key) {
        return Ok(());
    }
    if !visiting.insert(key.to_string()) {
        return Err(format!(
            "psa: circular import of module `{}` (line {})",
            key, line
        ));
    }
    let rel = key.replace('.', "/") + ".psa";
    let path = base_dir.join(&rel);
    let src = std::fs::read_to_string(&path).map_err(|e| {
        format!("psa: cannot import module `{}` ({}): {}", key, path.display(), e)
    })?;
    let toks = lexer::lex(&src)?;
    let mut p = parser::Parser::new(toks);
    let mut sub = p
        .parse_program()
        .map_err(|e| format!("psa: in module `{}`: {}", key, e))?;
    // The module's own imports resolve relative to its own file.
    let sub_dir = path
        .parent()
        .map(|x| x.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let sub_imports = std::mem::take(&mut sub.imports);
    for imp in &sub_imports {
        load_module(prog, &sub_dir, &imp.module, visiting, done, imp.line)?;
        if let Some(names) = &imp.names {
            bind_from_import(prog, &imp.module, names, imp.line)?;
        }
    }
    // Validate the module top level: statements may only be
    // literal assignments (the module attributes).
    let mut attrs: Vec<(String, ast::Expr)> = Vec::new();
    for st in &sub.top {
        match st {
            Stmt::Assign { name, value, .. } if is_literal(value) => {
                attrs.push((name.clone(), value.clone()));
            }
            _ => {
                return Err(format!(
                    "psa: module `{}` top level may only contain defs, classes and literal assignments (line {})",
                    key, line
                ));
            }
        }
    }
    // Class name collisions in the shared registry are loud.
    for c in &sub.classes {
        if prog.classes.iter().any(|x| x.name == c.name) {
            return Err(format!(
                "psa: class `{}` of module `{}` collides with another class of the same name (the class registry is program-wide in v1)",
                c.name, key
            ));
        }
    }
    let def_names: Vec<String> = sub.funcs.iter().map(|f| f.name.clone()).collect();
    for mut f in sub.funcs {
        rename_module_refs(&mut f, key, &def_names, &attrs);
        f.name = format!("{}.{}", key, f.name);
        prog.funcs.push(f);
    }
    for c in &mut sub.classes {
        for m in &mut c.methods {
            rename_module_refs(m, key, &def_names, &attrs);
        }
    }
    prog.classes.extend(sub.classes);
    for (n, lit) in &attrs {
        prog.mod_attrs.push((format!("{}.{}", key, n), lit.clone()));
    }
    prog.modules.push(key.to_string());
    visiting.remove(key);
    done.insert(key.to_string());
    Ok(())
}

fn bind_from_import(
    prog: &mut ast::Program,
    key: &str,
    names: &[(String, Option<String>)],
    line: u32,
) -> Result<(), String> {
    for (name, alias) in names {
        let target = alias.clone().unwrap_or_else(|| name.clone());
        let dotted = format!("{}.{}", key, name);
        if let Some(f) = prog.funcs.iter().find(|f| f.name == dotted).cloned() {
            if let Some(existing) = prog.funcs.iter().find(|x| x.name == target) {
                // Re-importing the same def under the same name
                // is a no-op; anything else is a collision.
                if format!("{:?}", existing.body) == format!("{:?}", f.body)
                    && existing.params.len() == f.params.len()
                {
                    continue;
                }
                return Err(format!(
                    "psa: from-import of `{}` as `{}` collides with an existing definition (line {})",
                    dotted, target, line
                ));
            }
            let mut f2 = f;
            f2.name = target.clone();
            prog.funcs.push(f2);
            continue;
        }
        if prog.classes.iter().any(|c| &c.name == name) {
            if target != *name {
                return Err(format!(
                    "psa: cannot alias class `{}` as `{}` in v1 (line {})",
                    name, target, line
                ));
            }
            continue;
        }
        if let Some((_, lit)) = prog.mod_attrs.iter().find(|(k, _)| k == &dotted) {
            let lit = lit.clone();
            if prog.const_names.iter().any(|(n, _)| n == &target) {
                return Err(format!(
                    "psa: from-import of `{}` as `{}` collides with an existing imported name (line {})",
                    dotted, target, line
                ));
            }
            prog.const_names.push((target, lit));
            continue;
        }
        return Err(format!(
            "psa: cannot import name `{}` from module `{}` (line {})",
            name, key, line
        ));
    }
    Ok(())
}

fn is_literal(e: &ast::Expr) -> bool {
    use ast::Expr;
    matches!(
        e,
        Expr::Int(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bool(_) | Expr::NoneLit
    )
}

/// Rewrite a merged module's function body: references to
/// sibling defs take their dotted keys, and reads of the
/// module's literal attributes are replaced by the literal
/// itself (the read-site evaluation shape). Names shadowed by
/// parameters or local assignments are left alone.
fn rename_module_refs(
    f: &mut ast::Func,
    key: &str,
    def_names: &[String],
    attrs: &[(String, ast::Expr)],
) {
    use std::collections::HashSet;
    let mut shadow: HashSet<String> = HashSet::new();
    for n in f.param_names() {
        shadow.insert(n);
    }
    collect_assigned(&f.body, &mut shadow);
    rename_stmts(&mut f.body, key, def_names, attrs, &shadow);
}

fn collect_assigned(stmts: &[ast::Stmt], out: &mut std::collections::HashSet<String>) {
    use ast::Stmt;
    for st in stmts {
        match st {
            Stmt::Assign { name, .. }
            | Stmt::AugAssign { name, .. }
            | Stmt::IndexAssign { name, .. } => {
                out.insert(name.clone());
            }
            Stmt::MultiAssign { names, .. } | Stmt::Unpack { names, .. } => {
                for n in names {
                    out.insert(n.clone());
                }
            }
            Stmt::ForRange { var, body, .. } => {
                out.insert(var.clone());
                collect_assigned(body, out);
            }
            Stmt::ForEach { vars, body, .. } => {
                for n in vars {
                    out.insert(n.clone());
                }
                collect_assigned(body, out);
            }
            Stmt::If { arms, els, .. } => {
                for (_, b) in arms {
                    collect_assigned(b, out);
                }
                if let Some(b) = els {
                    collect_assigned(b, out);
                }
            }
            Stmt::While { body, .. } => collect_assigned(body, out),
            Stmt::With { items, body, .. } => {
                for (_, bind) in items {
                    if let Some(n) = bind {
                        out.insert(n.clone());
                    }
                }
                collect_assigned(body, out);
            }
            Stmt::Try {
                body,
                handlers,
                els,
                finally,
                ..
            } => {
                collect_assigned(body, out);
                for h in handlers {
                    if let Some(n) = &h.bind {
                        out.insert(n.clone());
                    }
                    collect_assigned(&h.body, out);
                }
                if let Some(b) = els {
                    collect_assigned(b, out);
                }
                if let Some(b) = finally {
                    collect_assigned(b, out);
                }
            }
            Stmt::Del { targets, .. } => {
                for (n, _) in targets {
                    out.insert(n.clone());
                }
            }
            Stmt::Match { cases, .. } => {
                for c in cases {
                    collect_pattern_names(&c.pattern, out);
                    collect_assigned(&c.body, out);
                }
            }
            _ => {}
        }
    }
}

fn collect_pattern_names(
    p: &ast::Pattern,
    out: &mut std::collections::HashSet<String>,
) {
    match p {
        ast::Pattern::Capture(n) => {
            out.insert(n.clone());
        }
        ast::Pattern::Seq(pats) | ast::Pattern::Or(pats) => {
            for x in pats {
                collect_pattern_names(x, out);
            }
        }
        ast::Pattern::Lit(_) | ast::Pattern::Wildcard => {}
    }
}

fn rename_stmts(
    stmts: &mut [ast::Stmt],
    key: &str,
    def_names: &[String],
    attrs: &[(String, ast::Expr)],
    shadow: &std::collections::HashSet<String>,
) {
    use ast::Stmt;
    for st in stmts.iter_mut() {
        match st {
            Stmt::Assign { value, .. }
            | Stmt::AugAssign { value, .. }
            | Stmt::ExprStmt { value, .. } => {
                rename_expr(value, key, def_names, attrs, shadow)
            }
            Stmt::IndexAssign {
                index, value, ..
            } => {
                rename_expr(index, key, def_names, attrs, shadow);
                rename_expr(value, key, def_names, attrs, shadow);
            }
            Stmt::AttrAssign { base, value, .. } => {
                rename_expr(base, key, def_names, attrs, shadow);
                rename_expr(value, key, def_names, attrs, shadow);
            }
            Stmt::AttrDel { base, .. } => rename_expr(base, key, def_names, attrs, shadow),
            Stmt::If { arms, els, .. } => {
                for (c, b) in arms.iter_mut() {
                    rename_expr(c, key, def_names, attrs, shadow);
                    rename_stmts(b, key, def_names, attrs, shadow);
                }
                if let Some(b) = els {
                    rename_stmts(b, key, def_names, attrs, shadow);
                }
            }
            Stmt::While { cond, body, .. } => {
                rename_expr(cond, key, def_names, attrs, shadow);
                rename_stmts(body, key, def_names, attrs, shadow);
            }
            Stmt::ForRange {
                start, stop, body, ..
            } => {
                if let Some(s) = start {
                    rename_expr(s, key, def_names, attrs, shadow);
                }
                rename_expr(stop, key, def_names, attrs, shadow);
                rename_stmts(body, key, def_names, attrs, shadow);
            }
            Stmt::ForEach { iter, body, .. } => {
                rename_expr(iter, key, def_names, attrs, shadow);
                rename_stmts(body, key, def_names, attrs, shadow);
            }
            Stmt::Unpack { value, .. } => rename_expr(value, key, def_names, attrs, shadow),
            Stmt::Return { value, .. } => {
                if let Some(v) = value {
                    rename_expr(v, key, def_names, attrs, shadow);
                }
            }
            Stmt::Assert { cond, .. } => rename_expr(cond, key, def_names, attrs, shadow),
            Stmt::Del { targets, .. } => {
                for (_, idx) in targets.iter_mut() {
                    if let Some(e) = idx {
                        rename_expr(e, key, def_names, attrs, shadow);
                    }
                }
            }
            Stmt::MultiAssign { values, .. } => {
                for v in values.iter_mut() {
                    rename_expr(v, key, def_names, attrs, shadow);
                }
            }
            Stmt::Try {
                body,
                handlers,
                els,
                finally,
                ..
            } => {
                rename_stmts(body, key, def_names, attrs, shadow);
                for h in handlers.iter_mut() {
                    rename_stmts(&mut h.body, key, def_names, attrs, shadow);
                }
                if let Some(b) = els {
                    rename_stmts(b, key, def_names, attrs, shadow);
                }
                if let Some(b) = finally {
                    rename_stmts(b, key, def_names, attrs, shadow);
                }
            }
            Stmt::Raise { value, .. } => {
                if let Some(v) = value {
                    rename_expr(v, key, def_names, attrs, shadow);
                }
            }
            Stmt::With { items, body, .. } => {
                for (e, _) in items.iter_mut() {
                    rename_expr(e, key, def_names, attrs, shadow);
                }
                rename_stmts(body, key, def_names, attrs, shadow);
            }
            Stmt::Yield { value, .. } => {
                if let Some(v) = value {
                    rename_expr(v, key, def_names, attrs, shadow);
                }
            }
            Stmt::Match { subject, cases, .. } => {
                rename_expr(subject, key, def_names, attrs, shadow);
                for c in cases.iter_mut() {
                    rename_pattern(&mut c.pattern, key, def_names, attrs, shadow);
                    if let Some(g) = &mut c.guard {
                        rename_expr(g, key, def_names, attrs, shadow);
                    }
                    rename_stmts(&mut c.body, key, def_names, attrs, shadow);
                }
            }
            Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Pass { .. } => {}
        }
    }
}

fn rename_expr(
    e: &mut ast::Expr,
    key: &str,
    def_names: &[String],
    attrs: &[(String, ast::Expr)],
    shadow: &std::collections::HashSet<String>,
) {
    use ast::Expr;
    match e {
        Expr::Name(n) => {
            if shadow.contains(n) {
                return;
            }
            if def_names.iter().any(|d| d == n) {
                *e = Expr::Name(format!("{}.{}", key, n));
            } else if let Some((_, lit)) = attrs.iter().find(|(a, _)| a == n) {
                *e = lit.clone();
            }
        }
        Expr::Bin(a, _, b) => {
            rename_expr(a, key, def_names, attrs, shadow);
            rename_expr(b, key, def_names, attrs, shadow);
        }
        Expr::Unary(_, a) => rename_expr(a, key, def_names, attrs, shadow),
        Expr::Call(f, args, kwargs) => {
            rename_expr(f, key, def_names, attrs, shadow);
            for a in args.iter_mut() {
                rename_expr(a, key, def_names, attrs, shadow);
            }
            for (_, v) in kwargs.iter_mut() {
                rename_expr(v, key, def_names, attrs, shadow);
            }
        }
        Expr::ListLit(xs) | Expr::TupleLit(xs) | Expr::SetLit(xs) => {
            for x in xs.iter_mut() {
                rename_expr(x, key, def_names, attrs, shadow);
            }
        }
        Expr::DictLit(pairs) => {
            for (k, v) in pairs.iter_mut() {
                rename_expr(k, key, def_names, attrs, shadow);
                rename_expr(v, key, def_names, attrs, shadow);
            }
        }
        Expr::Comp {
            key: k,
            value,
            clauses,
            ..
        } => {
            let mut inner = shadow.clone();
            for cl in clauses.iter() {
                if let ast::CompClause::For { var, .. } = cl {
                    inner.insert(var.clone());
                }
            }
            for cl in clauses.iter_mut() {
                match cl {
                    ast::CompClause::For { iter, .. } => {
                        rename_expr(iter, key, def_names, attrs, shadow)
                    }
                    ast::CompClause::If(c) => {
                        rename_expr(c, key, def_names, attrs, &inner)
                    }
                }
            }
            rename_expr(k, key, def_names, attrs, &inner);
            if let Some(v) = value {
                rename_expr(v, key, def_names, attrs, &inner);
            }
        }
        Expr::Index(a, b) => {
            rename_expr(a, key, def_names, attrs, shadow);
            rename_expr(b, key, def_names, attrs, shadow);
        }
        Expr::Attr(a, _) => rename_expr(a, key, def_names, attrs, shadow),
        Expr::Slice(a, lo, hi, step) => {
            rename_expr(a, key, def_names, attrs, shadow);
            for part in [lo, hi, step] {
                if let Some(x) = part {
                    rename_expr(x, key, def_names, attrs, shadow);
                }
            }
        }
        Expr::MethodCall { args, .. } => {
            for a in args.iter_mut() {
                rename_expr(a, key, def_names, attrs, shadow);
            }
        }
        Expr::IfExp { cond, then, els } => {
            rename_expr(cond, key, def_names, attrs, shadow);
            rename_expr(then, key, def_names, attrs, shadow);
            rename_expr(els, key, def_names, attrs, shadow);
        }
        Expr::FStr(parts) => {
            for p in parts.iter_mut() {
                if let ast::FStrPart::Expr(x) = p {
                    rename_expr(x, key, def_names, attrs, shadow);
                }
            }
        }
        Expr::Lambda { params, body, .. } => {
            let mut inner = shadow.clone();
            for p in params.iter() {
                inner.insert(p.clone());
            }
            rename_expr(body, key, def_names, attrs, &inner);
        }
        Expr::Await(a) => rename_expr(a, key, def_names, attrs, shadow),
        Expr::Int(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bool(_) | Expr::NoneLit
        | Expr::FnVal(_) => {}
    }
}

fn rename_pattern(
    p: &mut ast::Pattern,
    key: &str,
    def_names: &[String],
    attrs: &[(String, ast::Expr)],
    shadow: &std::collections::HashSet<String>,
) {
    match p {
        ast::Pattern::Lit(e) => rename_expr(e, key, def_names, attrs, shadow),
        ast::Pattern::Seq(pats) | ast::Pattern::Or(pats) => {
            for x in pats.iter_mut() {
                rename_pattern(x, key, def_names, attrs, shadow);
            }
        }
        ast::Pattern::Capture(_) | ast::Pattern::Wildcard => {}
    }
}

fn newest_output_bin(root: &Path) -> Option<PathBuf> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    let entries = std::fs::read_dir(root).ok()?;
    for e in entries.flatten() {
        let cand = e.path().join("output.bin");
        if cand.is_file() {
            if let Ok(meta) = cand.metadata() {
                let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                if best.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) {
                    best = Some((mtime, cand));
                }
            }
        }
    }
    best.map(|(_, p)| p)
}
