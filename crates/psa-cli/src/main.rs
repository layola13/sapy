//! psa — the SA-Py compiler driver (v0.1).
//!
//! Pipeline: .psa source -> lexer/parser (subset) -> type inference ->
//! SLA source -> the `sla` binary's pipeline (check / SA emission) ->
//! the `sa` toolchain (build-exe). The psa layer never claims a feature
//! it cannot carry through that whole chain.

mod ast;
mod diag;
mod diag_registry;
mod emit;
mod hostdiag;
mod lexer;
mod minijson;
mod panic_codes;
mod parser;
mod srcmap;

use diag::{json_str, Diagnostic, ErrFormat};
use std::path::{Path, PathBuf};
use std::process::{exit, Command};

fn main() {
    let raw: Vec<String> = std::env::args().collect();
    // Global --error-format flag (design section 4.1): accepted
    // before or after the subcommand, as `--error-format=json` or
    // `--error-format json`; PSA_ERROR_FORMAT is the env fallback.
    // Every diagnostic psa prints goes through it.
    let mut format = std::env::var("PSA_ERROR_FORMAT")
        .ok()
        .and_then(|s| ErrFormat::parse(&s))
        .unwrap_or(ErrFormat::Text);
    let mut args: Vec<String> = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let a = raw[i].clone();
        if a == "--error-format" {
            match raw.get(i + 1).and_then(|v| ErrFormat::parse(v)) {
                Some(f) => {
                    format = f;
                    i += 2;
                }
                None => {
                    eprintln!("psa: --error-format needs a value (text|json)");
                    exit(2);
                }
            }
            continue;
        }
        if let Some(v) = a.strip_prefix("--error-format=") {
            match ErrFormat::parse(v) {
                Some(f) => format = f,
                None => {
                    eprintln!("psa: bad --error-format value `{}` (text|json)", v);
                    exit(2);
                }
            }
            i += 1;
            continue;
        }
        args.push(a);
        i += 1;
    }
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
            // P2: `emit` writes a map only when asked
            // (--map-out); `check` / `build-exe` always write one
            // next to the generated .sla. --embed-sources embeds
            // the source text in the map instead of only its hash.
            let map_out = parse_map_out(&args[3..]);
            let embed_sources = has_flag(&args[3..], "--embed-sources");
            if let Err(mut d) = run(cmd, &file, out_path.as_deref(), map_out.as_deref(), embed_sources) {
                d.fill_file(&file.display().to_string());
                d.fill_snippets();
                eprintln!("{}", d.render(format));
                exit(1);
            }
        }
        "map-loc" => {
            // `psa map-loc <map.json> --psa <file>:<line> [--col N]`
            // or `--sla <line>` (SOURCEMAP-DESIGN §3.3): prints
            // JSON on stdout; exit 0 for a query result (including
            // an explicit unmapped result), 1 for map/lookup
            // errors, 2 for bad arguments.
            exit(map_loc(&args[2..], format));
        }
        "explain" => {
            if args.len() < 3 {
                eprintln!("psa explain: missing code (e.g. `psa explain 77069`)");
                exit(2);
            }
            exit(explain(&args[2], format));
        }
        "version" | "--version" | "-V" => {
            println!("psa 1.0.0 (sapy v1: D0-D7 complete: static + dynamic faces, kwargs/defaults, ordered dict, lambdas, classes with runtime class-attribute registry, tuples/sets/comprehensions, methods, exceptions with cross-function propagation, with/files, imports incl. psa_std json via `import json`, generators/coroutines, match, overflow guards; unified batch green on the SLA->SA text route; dual-face bench beats CPython)");
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
        "usage: psa [--error-format=text|json] <command> <file.psa> [options]\n\nCommands:\n  emit      print the generated SLA source\n            [--map-out <path>] write the source map there (default: no map file)\n            [--embed-sources] embed source text in the map\n  check     type-check through the sla pipeline (writes <stem>.sla + <stem>.psa-map.json)\n  build-exe build a native executable (delegates to sla + sa)\n            [-o <output>] (default: ./<stem>)\n            (also copies <output>.psa-map.json next to the binary)\n  map-loc   query a .psa-map.json: psa map-loc <map.json> --psa <file>:<line> [--col N]\n            | --sla <line>  (prints JSON; see SOURCEMAP-DESIGN §3.3)\n  explain   explain a diagnostic or panic code (e.g. psa explain 77069)\n  version   print the psa version"
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

/// `--map-out <path>` (or `--map-out=<path>`): where `emit`
/// writes the source map when explicitly asked for one.
fn parse_map_out(args: &[String]) -> Option<PathBuf> {
    parse_flag_value(args, "--map-out").map(PathBuf::from)
}

/// The value of `--flag <value>` or `--flag=<value>`, if present.
fn parse_flag_value(args: &[String], flag: &str) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
        if let Some(v) = args[i].strip_prefix(&format!("{}=", flag)) {
            return Some(v.to_string());
        }
        i += 1;
    }
    None
}

/// Whether a bare `--flag` is present in the arguments.
fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn run(
    cmd: &str,
    file: &Path,
    out_path: Option<&Path>,
    map_out: Option<&Path>,
    embed_sources: bool,
) -> Result<(), Diagnostic> {
    if file.extension().map(|e| e != "psa").unwrap_or(true) {
        return Err(Diagnostic::host(
            "PSA-H002",
            "not-a-psa-file",
            format!("{} is not a .psa file", file.display()),
        ));
    }
    let src = std::fs::read_to_string(file).map_err(|e| {
        Diagnostic::host(
            "PSA-H002",
            "cannot-read-file",
            format!("cannot read {}: {}", file.display(), e),
        )
    })?;
    let toks = lexer::lex(&src)?;
    let mut p = parser::Parser::new(toks);
    let mut prog = p.parse_program()?;
    // sources[0] is always the main file (P0b); merge_imports
    // appends each imported module file in first-load order.
    prog.sources.push(file.display().to_string());
    let base_dir = file
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    merge_imports(&mut prog, &base_dir)?;
    // P2: the map's `sources` header — one entry per .psa file
    // in the program (main first, merged modules after, in
    // first-load order; P0b `prog.sources`). Entry 0 reuses the
    // exact bytes already read; each module file is read and
    // hashed here (merge_imports read the same paths a moment
    // ago, so a read failure now is a loud driver error, never
    // a silent omission). Content is embedded only under
    // --embed-sources (default: null, path + sha only).
    let mut sources: Vec<srcmap::SourceEntry> = Vec::new();
    for (i, spath) in prog.sources.iter().enumerate() {
        let text = if i == 0 {
            src.clone()
        } else {
            std::fs::read_to_string(spath).map_err(|e| {
                Diagnostic::host(
                    "PSA-H002",
                    "cannot-read-file",
                    format!("cannot read {}: {}", spath, e),
                )
            })?
        };
        sources.push(srcmap::SourceEntry {
            path: spath.clone(),
            sha256: srcmap::sha256_hex(text.as_bytes()),
            content: if embed_sources { Some(text) } else { None },
        });
    }
    // The generated .sla and its map live next to the source,
    // same stem, different suffix (design §3.3; §6 un-decided 4:
    // the map copy travels with the product, not with the host
    // cache — see RESULT.md for the alignment point).
    let stem = file.file_stem().unwrap().to_string_lossy().into_owned();
    let dir = file.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
    let sla_path = dir.join(format!("{}.sla", stem));
    let map_path = dir.join(format!("{}.psa-map.json", stem));
    let (sla_src, map) =
        emit::emit_program_with_map(&prog, sources, &sla_path.display().to_string())?;
    if cmd == "emit" {
        // `emit` keeps its stdout-pipeline semantics: the SLA
        // goes to stdout and no map file is written unless
        // --map-out explicitly asks for one.
        print!("{}", sla_src);
        if let Some(mp) = map_out {
            srcmap::write_map_file(mp, &map).map_err(|e| {
                Diagnostic::host(
                    "PSA-H002",
                    "cannot-write-file",
                    format!("cannot write {}: {}", mp.display(), e),
                )
            })?;
        }
        return Ok(());
    }
    // Write the generated SLA next to the source and delegate.
    // `check` / `build-exe` always write the map alongside it
    // (same segment table, zero extra emission cost).
    std::fs::write(&sla_path, &sla_src).map_err(|e| {
        Diagnostic::host(
            "PSA-H002",
            "cannot-write-file",
            format!("cannot write {}: {}", sla_path.display(), e),
        )
    })?;
    srcmap::write_map_file(&map_path, &map).map_err(|e| {
        Diagnostic::host(
            "PSA-H002",
            "cannot-write-file",
            format!("cannot write {}: {}", map_path.display(), e),
        )
    })?;
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
    // P3 (sourcemap design §3.5): capture the host's output
    // instead of inheriting stdio — psa can only map a failure
    // it can read. `output()` keeps the exit code semantics of
    // the old `status()` call; only the stdio routing changes.
    let output = command.output().map_err(|e| {
        Diagnostic::host(
            "PSA-H003",
            "cannot-run-host",
            format!("cannot run `{}`: {}", sla_bin, e),
        )
        .with_hint("set PSA_SLA to the sla binary".to_string())
    })?;
    if output.status.success() {
        // Success: pass the host output straight through, byte
        // for byte (stdout to stdout, stderr to stderr), so
        // visible behaviour — e.g. deprecation notes — is
        // unchanged. (Interleaving between the two streams is
        // not preserved: capture cannot know it.)
        use std::io::Write;
        let stdout = std::io::stdout();
        let mut so = stdout.lock();
        let _ = so.write_all(&output.stdout);
        let _ = so.flush();
        let stderr = std::io::stderr();
        let mut se = stderr.lock();
        let _ = se.write_all(&output.stderr);
        let _ = se.flush();
    } else {
        // Failure: parse the captured output and present the
        // dual diagnostic (host verbatim + psa mapping).
        // (P2's run() already defined `map_path` above; P3's
        // copy of that line is dropped here.)
        return Err(host_failure_diagnostic(
            cmd,
            &output,
            &prog,
            &src,
            &file.display().to_string(),
            &sla_src,
            &sla_path,
            &map_path,
        ));
    }
    if cmd == "build-exe" {
        // R1379: the text-road host leaves the binary at
        // <dir>/.sa_cache/build-exe/<hash>/output.bin, next to the
        // generated .sla. The previous lookup only searched the
        // SAB-era nested layout (.sla-cache/sab/.sa_cache/...),
        // which the text road never writes — build-exe reported
        // success and then failed to find its own product. The old
        // layout stays as a fallback for older hosts.
        let cache = dir.join(".sa_cache").join("build-exe");
        let legacy = dir.join(".sla-cache").join("sab").join(".sa_cache").join("build-exe");
        let bin = newest_output_bin(&cache).or_else(|| newest_output_bin(&legacy)).ok_or_else(|| {
            Diagnostic::host(
                "PSA-H003",
                "host-output-missing",
                "sla reported success but no output.bin was found under .sa_cache/build-exe".to_string(),
            )
        })?;
        let target = out_path.map(|p| p.to_path_buf()).unwrap_or_else(|| {
            std::env::current_dir().unwrap().join(&stem)
        });
        std::fs::copy(&bin, &target).map_err(|e| {
            Diagnostic::host(
                "PSA-H002",
                "cannot-copy-file",
                format!("cannot copy binary to {}: {}", target.display(), e),
            )
        })?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&target).unwrap().permissions();
            perms.set_mode(0o755);
            let _ = std::fs::set_permissions(&target, perms);
        }
        // P2: the map travels with the product — a copy sits next
        // to the executable so a debugger can find it from the
        // binary path alone (design §3.3). Same stem as the
        // binary, `.psa-map.json` suffix. (The copy inside the
        // host cache dir, next to output.bin, is deliberately NOT
        // made here: whether a map belongs in the cache layout is
        // design §6 un-decided 4, to align with the host side —
        // see RESULT.md.)
        let exe_map_path = PathBuf::from(format!("{}.psa-map.json", target.display()));
        std::fs::copy(&map_path, &exe_map_path).map_err(|e| {
            Diagnostic::host(
                "PSA-H002",
                "cannot-copy-file",
                format!(
                    "cannot copy source map to {}: {}",
                    exe_map_path.display(),
                    e
                ),
            )
        })?;
        println!("psa: built {}", target.display());
    }
    Ok(())
}

/// P3: build the dual (host verbatim + psa mapping) diagnostic
/// for a failed host invocation (design §3.4/§3.5).
///
/// 1. Parse the captured host output into diagnostics + residual
///    (`hostdiag::parse_host_output` — adapters eat what they
///    recognise; the rest stays verbatim in `host.raw`).
/// 2. Resolve each diagnostic against the P2 segment table when
///    `<stem>.psa-map.json` exists, else against the name bridge
///    (function-level) or to an honest unmapped outcome.
/// 3. Attach the report + top-level mapping to the PSA-H003; the
///    psa-source site mirrors the first mapping so JSON `sites`
///    and the text `psa -->` line agree.
///
/// The old H003 note ("the messages above come from … and refer
/// to the generated file …") is retired: the verbatim block is
/// now labelled, and the generated file + map file are named on
/// the `sla -->` / `map:` lines instead. Message, code, rule
/// and hint text are unchanged from the pre-P3 wording.
#[allow(clippy::too_many_arguments)]
fn host_failure_diagnostic(
    cmd: &str,
    output: &std::process::Output,
    prog: &ast::Program,
    psa_src: &str,
    psa_name: &str,
    sla_src: &str,
    sla_path: &Path,
    map_path: &Path,
) -> Diagnostic {
    let mut report = hostdiag::parse_host_output(
        "sla",
        cmd,
        output.status.code(),
        &output.stdout,
        &output.stderr,
    );
    // The P2 segment table, when one exists next to the
    // generated .sla (loaded + sha-verified by `SourceMap::load`;
    // a stale/corrupt map is a loud load error, never a silent
    // downgrade — see `hostdiag::reason::STALE_MAP` /
    // `BAD_MAP`). A missing map file is not an error: row-level
    // resolution then degrades explicitly (`MAP_UNAVAILABLE`).
    let (map, map_error) = if map_path.exists() {
        match srcmap::SourceMap::load(map_path) {
            Ok(m) => (Some(m), None),
            Err(e) => (None, Some(e)),
        }
    } else {
        (None, None)
    };
    let ctx = hostdiag::MapCtx {
        prog,
        psa_src,
        psa_file: psa_name.to_string(),
        sla_src,
        sla_file: sla_path.display().to_string(),
        map_file: map_path,
        map: map.as_ref(),
        map_error: map_error.as_ref(),
    };
    let mut top_mapping: Option<hostdiag::MappingOutcome> = None;
    if report.diagnostics.is_empty() {
        top_mapping = Some(hostdiag::resolve_empty_report(&report.raw, map_path));
    } else {
        for d in report.diagnostics.iter_mut() {
            let m = hostdiag::resolve_mapping(d, &ctx);
            if top_mapping.is_none() {
                top_mapping = Some(m.clone());
            }
            d.mapping = Some(m);
        }
    }
    let mut diag = Diagnostic::host(
        "PSA-H003",
        "host-pipeline-failed",
        format!(
            "the sla pipeline failed while running `{}` ({})",
            cmd, output.status
        ),
    );
    if let Some(e) = &map_error {
        diag = diag.with_note(format!(
            "source map {} could not be used for row-level mapping ({}); the host output below is still verbatim",
            map_path.display(),
            e
        ));
    }
    diag = diag
    .with_host(report)
    .with_hint(
        "run `psa emit` on the source to inspect the generated code; if it looks right, the failure is below psa"
            .to_string(),
    );
    if let Some(m) = top_mapping {
        if let (Some(line), Some(col)) = (m.psa_line, m.psa_col) {
            diag = diag.with_psa_site(line, col, m.snippet.clone());
        }
        diag = diag.with_mapping(m);
    }
    diag
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
fn merge_imports(prog: &mut ast::Program, base_dir: &Path) -> Result<(), Diagnostic> {
    use std::collections::HashSet;
    let imports = std::mem::take(&mut prog.imports);
    if imports.is_empty() {
        return Ok(());
    }
    let mut visiting: HashSet<String> = HashSet::new();
    let mut done: HashSet<String> = HashSet::new();
    for imp in &imports {
        // psa_std fallback: a module with no .psa file on disk
        // whose name is a known psa_std module binds to the
        // standard library instead (the emitter splices its
        // SLA text). A .psa file of the same name wins, as a
        // local module shadows the stdlib. from-imports of
        // std modules are refused in v1.
        let rel = imp.module.replace('.', "/") + ".psa";
        if !base_dir.join(&rel).exists() && is_std_module(&imp.module) {
            if imp.names.is_some() {
                return Err(Diagnostic::host(
                    "PSA-H010",
                    "from-import-std-module",
                    format!(
                        "`from {} import ...` is not supported for psa_std modules in v1; use `import {}` and call its functions through the module",
                        imp.module, imp.module
                    ),
                )
                .with_site(imp.line, "import"));
            }
            if !prog.std_imports.contains(&imp.module) {
                prog.std_imports.push(imp.module.clone());
            }
            continue;
        }
        load_module(prog, base_dir, &imp.module, &mut visiting, &mut done, imp.line)?;
        if let Some(names) = &imp.names {
            bind_from_import(prog, &imp.module, names, imp.line)?;
        }
    }
    Ok(())
}

/// The psa_std modules this psa build ships (json, math,
/// string, and the batch-3 modules csv, logging, argparse,
/// collections, itertools, functools, bisect, heapq,
/// base64, random, re, socket, urllib -- see
/// psa-std/BATCH-WIRING.md).
fn is_std_module(key: &str) -> bool {
    matches!(
        key,
        "json" | "math" | "string" | "csv" | "logging" | "argparse" | "collections"
            | "itertools" | "functools" | "bisect" | "heapq" | "base64" | "random" | "re" | "socket" | "urllib" | "pickle"
    )
}

fn load_module(
    prog: &mut ast::Program,
    base_dir: &Path,
    key: &str,
    visiting: &mut std::collections::HashSet<String>,
    done: &mut std::collections::HashSet<String>,
    line: u32,
) -> Result<(), Diagnostic> {
    use ast::Stmt;
    if done.contains(key) {
        return Ok(());
    }
    if !visiting.insert(key.to_string()) {
        return Err(Diagnostic::host(
            "PSA-H004",
            "circular-import",
            format!("circular import of module `{}`", key),
        )
        .with_site(line, "import"));
    }
    let rel = key.replace('.', "/") + ".psa";
    let path = base_dir.join(&rel);
    let src = std::fs::read_to_string(&path).map_err(|e| {
        Diagnostic::host(
            "PSA-H002",
            "cannot-read-module",
            format!("cannot import module `{}` ({}): {}", key, path.display(), e),
        )
    })?;
    let toks = lexer::lex(&src).map_err(|mut d| {
        d.fill_file(&path.display().to_string());
        d
    })?;
    let mut p = parser::Parser::new(toks);
    let mut sub = p.parse_program().map_err(|mut d| {
        d.fill_file(&path.display().to_string());
        d.with_note(format!("in module `{}`", key))
    })?;
    // Register this module's file in sources[] (P0b) and remember
    // its index: every anchor inside the merged defs/classes is
    // re-stamped to it below, so a diagnostic inside a module
    // function resolves to the module file, not the main file.
    let src_idx = register_source(prog, &path.display().to_string());
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
                return Err(
                    Diagnostic::host(
                        "PSA-H005",
                        "module-top-level",
                        format!(
                            "module `{}` top level may only contain defs, classes and literal assignments",
                            key
                        ),
                    )
                    .with_site(line, "import"),
                );
            }
        }
    }
    // Class name collisions in the shared registry are loud.
    for c in &sub.classes {
        if prog.classes.iter().any(|x| x.name == c.name) {
            return Err(Diagnostic::host(
                "PSA-H006",
                "module-class-collision",
                format!(
                    "class `{}` of module `{}` collides with another class of the same name",
                    c.name, key
                ),
            )
            .with_note("the class registry is program-wide in v1".to_string()));
        }
    }
    let def_names: Vec<String> = sub.funcs.iter().map(|f| f.name.clone()).collect();
    for mut f in sub.funcs {
        rename_module_refs(&mut f, key, &def_names, &attrs);
        f.name = format!("{}.{}", key, f.name);
        // Re-stamp every anchor in the merged function to this
        // module's sources[] index (P0b).
        stamp_source_func(&mut f, src_idx);
        prog.funcs.push(f);
    }
    for c in &mut sub.classes {
        for m in &mut c.methods {
            rename_module_refs(m, key, &def_names, &attrs);
        }
        // Re-stamp the class, its methods and its attributes (P0b).
        stamp_source_class(c, src_idx);
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
) -> Result<(), Diagnostic> {
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
                return Err(Diagnostic::host(
                    "PSA-H007",
                    "from-import-collision",
                    format!(
                        "from-import of `{}` as `{}` collides with an existing definition",
                        dotted, target
                    ),
                )
                .with_site(line, "import"));
            }
            let mut f2 = f;
            f2.name = target.clone();
            prog.funcs.push(f2);
            continue;
        }
        if prog.classes.iter().any(|c| &c.name == name) {
            if target != *name {
                return Err(Diagnostic::host(
                    "PSA-H008",
                    "from-import-class-alias",
                    format!("cannot alias class `{}` as `{}` in v1", name, target),
                )
                .with_site(line, "import"));
            }
            continue;
        }
        if let Some((_, lit)) = prog.mod_attrs.iter().find(|(k, _)| k == &dotted) {
            let lit = lit.clone();
            if prog.const_names.iter().any(|(n, _)| n == &target) {
                return Err(Diagnostic::host(
                    "PSA-H009",
                    "from-import-name-collision",
                    format!(
                        "from-import of `{}` as `{}` collides with an existing imported name",
                        dotted, target
                    ),
                )
                .with_site(line, "import"));
            }
            prog.const_names.push((target, lit));
            continue;
        }
        return Err(Diagnostic::host(
            "PSA-H011",
            "cannot-import-name",
            format!("cannot import name `{}` from module `{}`", name, key),
        )
        .with_site(line, "import")
        .with_hint("check that the module defines this name".to_string()));
    }
    Ok(())
}

fn is_literal(e: &ast::Expr) -> bool {
    use ast::Expr;
    matches!(
        e,
        Expr::Int(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bytes(_) | Expr::Bool(_)
            | Expr::NoneLit
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

/// Walrus targets an expression binds in the scope that
/// contains it. (A comprehension's walrus targets bind in
/// the containing scope too, so comprehensions are entered;
/// lambda bodies are not — a walrus there binds in the
/// lambda's own scope.)
/// Walrus targets inside f-string replacement fields, including
/// fields nested inside a format spec (fstring batch: FStrPart::Field
/// replaced the old FStrPart::Expr).
fn collect_fstr_walrus(parts: &[ast::FStrPart], out: &mut std::collections::HashSet<String>) {
    for p in parts {
        if let ast::FStrPart::Field { value, spec, .. } = p {
            collect_expr_walrus(value, out);
            if let Some(spec) = spec {
                collect_fstr_walrus(spec, out);
            }
        }
    }
}

fn collect_expr_walrus(e: &ast::Expr, out: &mut std::collections::HashSet<String>) {
    use ast::Expr;
    match e {
        Expr::Walrus { name, value } => {
            out.insert(name.clone());
            collect_expr_walrus(value, out);
        }
        Expr::Bin(a, _, b) => {
            collect_expr_walrus(a, out);
            collect_expr_walrus(b, out);
        }
        Expr::Unary(_, a) => collect_expr_walrus(a, out),
        Expr::Call(f, args, kwargs, _) => {
            collect_expr_walrus(f, out);
            for a in args {
                collect_expr_walrus(a, out);
            }
            for (_, v) in kwargs {
                collect_expr_walrus(v, out);
            }
        }
        Expr::ListLit(es) | Expr::TupleLit(es) | Expr::SetLit(es) => {
            for a in es {
                collect_expr_walrus(a, out);
            }
        }
        Expr::DictLit(pairs) => {
            for (k, v) in pairs {
                collect_expr_walrus(k, out);
                collect_expr_walrus(v, out);
            }
        }
        Expr::Index(a, b, _) => {
            collect_expr_walrus(a, out);
            collect_expr_walrus(b, out);
        }
        Expr::Slice(a, lo, hi, st) => {
            collect_expr_walrus(a, out);
            for o in [lo, hi, st] {
                if let Some(x) = o {
                    collect_expr_walrus(x, out);
                }
            }
        }
        Expr::MethodCall { args, .. } => {
            for a in args {
                collect_expr_walrus(a, out);
            }
        }
        Expr::Attr(b, _, _) => collect_expr_walrus(b, out),
        Expr::Comp { key, value, clauses, .. } => {
            collect_expr_walrus(key, out);
            if let Some(v) = value {
                collect_expr_walrus(v, out);
            }
            for c in clauses {
                match c {
                    ast::CompClause::For { iter, .. } => collect_expr_walrus(iter, out),
                    ast::CompClause::If(cond) => collect_expr_walrus(cond, out),
                }
            }
        }
        Expr::IfExp { cond, then, els } => {
            collect_expr_walrus(cond, out);
            collect_expr_walrus(then, out);
            collect_expr_walrus(els, out);
        }
        Expr::FStr(parts) => collect_fstr_walrus(parts, out),
        Expr::Await(a) => collect_expr_walrus(a, out),
        _ => {}
    }
}

fn collect_pattern_walrus(p: &ast::Pattern, out: &mut std::collections::HashSet<String>) {
    match p {
        ast::Pattern::Lit(e) => collect_expr_walrus(e, out),
        ast::Pattern::Seq(ps) | ast::Pattern::Or(ps) => {
            for x in ps {
                collect_pattern_walrus(x, out);
            }
        }
        _ => {}
    }
}

/// Walrus targets bound by a statement list, recursing into
/// child statement lists (same scopes as collect_assigned)
/// but never into a nested def's body — that is a child
/// scope, whose own collect_assigned call covers it.
fn collect_stmt_walrus(stmts: &[ast::Stmt], out: &mut std::collections::HashSet<String>) {
    use ast::Stmt;
    for st in stmts {
        match st {
            Stmt::Assign { value, .. }
            | Stmt::AugAssign { value, .. }
            | Stmt::ExprStmt { value, .. }
            | Stmt::Unpack { value, .. } => collect_expr_walrus(value, out),
            Stmt::MultiAssign { values, .. } => {
                for v in values {
                    collect_expr_walrus(v, out);
                }
            }
            Stmt::IndexAssign { index, value, .. } => {
                collect_expr_walrus(index, out);
                collect_expr_walrus(value, out);
            }
            Stmt::AttrAssign { base, value, .. } => {
                collect_expr_walrus(base, out);
                collect_expr_walrus(value, out);
            }
            Stmt::AttrDel { base, .. } => collect_expr_walrus(base, out),
            Stmt::Del { targets, .. } => {
                for (_, idx) in targets {
                    if let Some(i) = idx {
                        collect_expr_walrus(i, out);
                    }
                }
            }
            Stmt::Return { value, .. } | Stmt::Yield { value, .. } => {
                if let Some(v) = value {
                    collect_expr_walrus(v, out);
                }
            }
            Stmt::Assert { cond, .. } => collect_expr_walrus(cond, out),
            Stmt::If { arms, els, .. } => {
                for (c, b) in arms {
                    collect_expr_walrus(c, out);
                    collect_stmt_walrus(b, out);
                }
                if let Some(b) = els {
                    collect_stmt_walrus(b, out);
                }
            }
            Stmt::While { cond, body, els, .. } => {
                collect_expr_walrus(cond, out);
                collect_stmt_walrus(body, out);
                if let Some(b) = els {
                    collect_stmt_walrus(b, out);
                }
            }
            Stmt::ForRange { start, stop, body, els, .. } => {
                if let Some(s) = start {
                    collect_expr_walrus(s, out);
                }
                collect_expr_walrus(stop, out);
                collect_stmt_walrus(body, out);
                if let Some(b) = els {
                    collect_stmt_walrus(b, out);
                }
            }
            Stmt::ForEach { iter, body, els, .. } => {
                collect_expr_walrus(iter, out);
                collect_stmt_walrus(body, out);
                if let Some(b) = els {
                    collect_stmt_walrus(b, out);
                }
            }
            Stmt::With { items, body, .. } => {
                for (e, _) in items {
                    collect_expr_walrus(e, out);
                }
                collect_stmt_walrus(body, out);
            }
            Stmt::Try { body, handlers, els, finally, .. } => {
                collect_stmt_walrus(body, out);
                for h in handlers {
                    collect_stmt_walrus(&h.body, out);
                }
                if let Some(b) = els {
                    collect_stmt_walrus(b, out);
                }
                if let Some(b) = finally {
                    collect_stmt_walrus(b, out);
                }
            }
            Stmt::Match { subject, cases, .. } => {
                collect_expr_walrus(subject, out);
                for c in cases {
                    collect_pattern_walrus(&c.pattern, out);
                    if let Some(g) = &c.guard {
                        collect_expr_walrus(g, out);
                    }
                    collect_stmt_walrus(&c.body, out);
                }
            }
            _ => {}
        }
    }
}

fn collect_assigned(stmts: &[ast::Stmt], out: &mut std::collections::HashSet<String>) {
    use ast::Stmt;
    // A name bound only by an assignment expression is just
    // as assigned as one bound by a statement (without this,
    // the rename pass would take a walrus-bound local that
    // shares its name with a module literal attribute for a
    // read of that attribute).
    collect_stmt_walrus(stmts, out);
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
            Stmt::ForRange { var, body, els, .. } => {
                out.insert(var.clone());
                collect_assigned(body, out);
                if let Some(b) = els {
                    collect_assigned(b, out);
                }
            }
            Stmt::ForEach { vars, body, els, .. } => {
                for n in vars {
                    out.insert(n.clone());
                }
                collect_assigned(body, out);
                if let Some(b) = els {
                    collect_assigned(b, out);
                }
            }
            Stmt::If { arms, els, .. } => {
                for (_, b) in arms {
                    collect_assigned(b, out);
                }
                if let Some(b) = els {
                    collect_assigned(b, out);
                }
            }
            Stmt::While { body, els, .. } => {
                collect_assigned(body, out);
                if let Some(b) = els {
                    collect_assigned(b, out);
                }
            }
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
            Stmt::While { cond, body, els, .. } => {
                rename_expr(cond, key, def_names, attrs, shadow);
                rename_stmts(body, key, def_names, attrs, shadow);
                if let Some(b) = els {
                    rename_stmts(b, key, def_names, attrs, shadow);
                }
            }
            Stmt::ForRange {
                start, stop, body, els, ..
            } => {
                if let Some(s) = start {
                    rename_expr(s, key, def_names, attrs, shadow);
                }
                rename_expr(stop, key, def_names, attrs, shadow);
                rename_stmts(body, key, def_names, attrs, shadow);
                if let Some(b) = els {
                    rename_stmts(b, key, def_names, attrs, shadow);
                }
            }
            Stmt::ForEach { iter, body, els, .. } => {
                rename_expr(iter, key, def_names, attrs, shadow);
                rename_stmts(body, key, def_names, attrs, shadow);
                if let Some(b) = els {
                    rename_stmts(b, key, def_names, attrs, shadow);
                }
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
            Stmt::Def(f) => {
                // A nested def can appear in an imported
                // module's function body: this pass runs on the
                // raw parsed AST, before closure conversion.
                // Rename inside it; its parameters shadow the
                // enclosing scope, its defaults evaluate in it.
                let mut inner = shadow.clone();
                for (p, _) in f.params.iter() {
                    inner.insert(p.clone());
                }
                if let Some(v) = &f.vararg {
                    inner.insert(v.clone());
                }
                if let Some(k) = &f.kwarg {
                    inner.insert(k.clone());
                }
                for (_, d) in f.params.iter_mut() {
                    if let Some(d) = d {
                        rename_expr(d, key, def_names, attrs, shadow);
                    }
                }
                rename_stmts(&mut f.body, key, def_names, attrs, &inner);
            }
            Stmt::Global { .. } | Stmt::Nonlocal { .. } | Stmt::PublishModEnv { .. } => {}
            Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Pass { .. } => {}
        }
    }
}

/// Rename inside one f-string part: the field's value
/// expression and, recursively, the expressions inside its
/// format spec.
fn rename_fstr_part(
    p: &mut ast::FStrPart,
    key: &str,
    def_names: &[String],
    attrs: &[(String, ast::Expr)],
    shadow: &std::collections::HashSet<String>,
) {
    if let ast::FStrPart::Field { value, spec, .. } = p {
        rename_expr(value, key, def_names, attrs, shadow);
        if let Some(parts) = spec {
            for sp in parts.iter_mut() {
                rename_fstr_part(sp, key, def_names, attrs, shadow);
            }
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
        Expr::Name(n, span) => {
            if shadow.contains(n) {
                return;
            }
            if def_names.iter().any(|d| d == n) {
                // The reference keeps its original span (P0b): the
                // span identifies WHERE the name was written, and the
                // rewritten dotted name is longer than the source text.
                *e = Expr::Name(format!("{}.{}", key, n), *span);
            } else if let Some((_, lit)) = attrs.iter().find(|(a, _)| a == n) {
                *e = lit.clone();
            }
        }
        Expr::Bin(a, _, b) => {
            rename_expr(a, key, def_names, attrs, shadow);
            rename_expr(b, key, def_names, attrs, shadow);
        }
        Expr::Unary(_, a) => rename_expr(a, key, def_names, attrs, shadow),
        Expr::Call(f, args, kwargs, _) => {
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
        Expr::Index(a, b, _) => {
            rename_expr(a, key, def_names, attrs, shadow);
            rename_expr(b, key, def_names, attrs, shadow);
        }
        Expr::Attr(a, _, _) => rename_expr(a, key, def_names, attrs, shadow),
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
                rename_fstr_part(p, key, def_names, attrs, shadow);
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
        Expr::Walrus { value, .. } => {
            // The target binds a plain local name in the merged
            // module (like an assignment target, which this pass
            // leaves alone); the value is evaluated before the
            // binding, so it renames in the current shadow set.
            rename_expr(value, key, def_names, attrs, shadow)
        }
        Expr::CellNew(a) | Expr::CellGet(a) => {
            // Produced only by closure conversion, which runs
            // after this pass; recurse defensively so the pass
            // stays total if it ever runs post-conversion.
            rename_expr(a, key, def_names, attrs, shadow)
        }
        Expr::CellSet(a, b) => {
            rename_expr(a, key, def_names, attrs, shadow);
            rename_expr(b, key, def_names, attrs, shadow);
        }
        Expr::ClosureVal(_, env) => rename_expr(env, key, def_names, attrs, shadow),
        Expr::ModEnv => {}
        Expr::Int(_) | Expr::BigInt(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bytes(_) | Expr::Bool(_)
        | Expr::NoneLit | Expr::FnVal(_) => {}
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

/// Append `path` to `prog.sources` unless already present, and
/// return its index (P0b). Index 0 is the main file (filled in by
/// the driver); module files follow in first-load order. Paths are
/// display strings as constructed by the driver (not canonicalised);
/// the map writer (P2) owns any canonicalisation.
fn register_source(prog: &mut ast::Program, path: &str) -> u16 {
    if let Some(i) = prog.sources.iter().position(|s| s == path) {
        return i as u16;
    }
    prog.sources.push(path.to_string());
    (prog.sources.len() - 1) as u16
}

/// Re-stamp every anchor/span inside a merged module function to
/// `source` (P0b): the function's own anchor, its parameter
/// defaults, and every statement/expression anchor in its body.
/// `line` values are per-file line numbers and stay untouched —
/// only `source` changes, which is what disambiguates the files.
fn stamp_source_func(f: &mut ast::Func, source: u16) {
    f.anchor.source = source;
    for (_, d) in f.params.iter_mut() {
        if let Some(d) = d {
            stamp_source_expr(d, source);
        }
    }
    stamp_source_stmts(&mut f.body, source);
}

/// Re-stamp a merged module class: its own anchor, its methods
/// (via `stamp_source_func`) and its attribute expressions (P0b).
fn stamp_source_class(c: &mut ast::ClassDef, source: u16) {
    c.anchor.source = source;
    for m in &mut c.methods {
        stamp_source_func(m, source);
    }
    for (_, e) in &mut c.attrs {
        stamp_source_expr(e, source);
    }
}

fn stamp_source_stmts(stmts: &mut [ast::Stmt], source: u16) {
    use ast::Stmt;
    for st in stmts.iter_mut() {
        match st {
            Stmt::Assign { value, anchor, .. }
            | Stmt::AugAssign { value, anchor, .. }
            | Stmt::ExprStmt { value, anchor, .. } => {
                anchor.source = source;
                stamp_source_expr(value, source);
            }
            Stmt::IndexAssign { index, value, anchor, .. } => {
                anchor.source = source;
                stamp_source_expr(index, source);
                stamp_source_expr(value, source);
            }
            Stmt::AttrAssign { base, value, anchor, .. } => {
                anchor.source = source;
                stamp_source_expr(base, source);
                stamp_source_expr(value, source);
            }
            Stmt::AttrDel { base, anchor, .. } => {
                anchor.source = source;
                stamp_source_expr(base, source);
            }
            Stmt::If { arms, els, anchor, .. } => {
                anchor.source = source;
                for (c, b) in arms.iter_mut() {
                    stamp_source_expr(c, source);
                    stamp_source_stmts(b, source);
                }
                if let Some(b) = els {
                    stamp_source_stmts(b, source);
                }
            }
            Stmt::While { cond, body, els, anchor, .. } => {
                anchor.source = source;
                stamp_source_expr(cond, source);
                stamp_source_stmts(body, source);
                if let Some(b) = els {
                    stamp_source_stmts(b, source);
                }
            }
            Stmt::ForRange { start, stop, body, els, anchor, .. } => {
                anchor.source = source;
                if let Some(s) = start {
                    stamp_source_expr(s, source);
                }
                stamp_source_expr(stop, source);
                stamp_source_stmts(body, source);
                if let Some(b) = els {
                    stamp_source_stmts(b, source);
                }
            }
            Stmt::ForEach { iter, body, els, anchor, .. } => {
                anchor.source = source;
                stamp_source_expr(iter, source);
                stamp_source_stmts(body, source);
                if let Some(b) = els {
                    stamp_source_stmts(b, source);
                }
            }
            Stmt::Unpack { value, anchor, .. } => {
                anchor.source = source;
                stamp_source_expr(value, source);
            }
            Stmt::Return { value, anchor, .. } => {
                anchor.source = source;
                if let Some(v) = value {
                    stamp_source_expr(v, source);
                }
            }
            Stmt::Assert { cond, anchor, .. } => {
                anchor.source = source;
                stamp_source_expr(cond, source);
            }
            Stmt::Del { targets, anchor, .. } => {
                anchor.source = source;
                for (_, idx) in targets.iter_mut() {
                    if let Some(e) = idx {
                        stamp_source_expr(e, source);
                    }
                }
            }
            Stmt::MultiAssign { values, anchor, .. } => {
                anchor.source = source;
                for v in values.iter_mut() {
                    stamp_source_expr(v, source);
                }
            }
            Stmt::Try { body, handlers, els, finally, anchor, .. } => {
                anchor.source = source;
                stamp_source_stmts(body, source);
                for h in handlers.iter_mut() {
                    h.anchor.source = source;
                    stamp_source_stmts(&mut h.body, source);
                }
                if let Some(b) = els {
                    stamp_source_stmts(b, source);
                }
                if let Some(b) = finally {
                    stamp_source_stmts(b, source);
                }
            }
            Stmt::Raise { value, anchor, .. } => {
                anchor.source = source;
                if let Some(v) = value {
                    stamp_source_expr(v, source);
                }
            }
            Stmt::With { items, body, anchor, .. } => {
                anchor.source = source;
                for (e, _) in items.iter_mut() {
                    stamp_source_expr(e, source);
                }
                stamp_source_stmts(body, source);
            }
            Stmt::Yield { value, anchor, .. } => {
                anchor.source = source;
                if let Some(v) = value {
                    stamp_source_expr(v, source);
                }
            }
            Stmt::Match { subject, cases, anchor, .. } => {
                anchor.source = source;
                stamp_source_expr(subject, source);
                for c in cases.iter_mut() {
                    c.anchor.source = source;
                    stamp_source_pattern(&mut c.pattern, source);
                    if let Some(g) = &mut c.guard {
                        stamp_source_expr(g, source);
                    }
                    stamp_source_stmts(&mut c.body, source);
                }
            }
            Stmt::Def(f) => {
                // A nested def inside a merged module function: its
                // own anchor and body are stamped by the same rule.
                stamp_source_func(f, source);
            }
            Stmt::Global { anchor, .. }
            | Stmt::Nonlocal { anchor, .. }
            | Stmt::PublishModEnv { anchor, .. } => {
                anchor.source = source;
            }
            Stmt::Break { anchor, .. }
            | Stmt::Continue { anchor, .. }
            | Stmt::Pass { anchor, .. } => {
                anchor.source = source;
            }
        }
    }
}

/// Re-stamp every span inside an expression to `source` (P0b).
/// Only `Name` / `Call` / `Index` / `Attr` carry spans; the rest
/// is pure recursion into children (their positions, if ever
/// needed, come from the enclosing statement's anchor).
/// Source stamping inside f-string replacement fields, including
/// fields nested inside a format spec (fstring batch:
/// FStrPart::Field replaced the old FStrPart::Expr).
fn stamp_source_fstr(parts: &mut [ast::FStrPart], source: u16) {
    for p in parts.iter_mut() {
        if let ast::FStrPart::Field { value, spec, .. } = p {
            stamp_source_expr(value, source);
            if let Some(spec) = spec {
                stamp_source_fstr(spec, source);
            }
        }
    }
}

fn stamp_source_expr(e: &mut ast::Expr, source: u16) {
    use ast::Expr;
    match e {
        Expr::Name(_, span) => {
            span.start.source = source;
        }
        Expr::Bin(a, _, b) => {
            stamp_source_expr(a, source);
            stamp_source_expr(b, source);
        }
        Expr::Unary(_, a) => stamp_source_expr(a, source),
        Expr::Call(f, args, kwargs, span) => {
            span.start.source = source;
            stamp_source_expr(f, source);
            for a in args.iter_mut() {
                stamp_source_expr(a, source);
            }
            for (_, v) in kwargs.iter_mut() {
                stamp_source_expr(v, source);
            }
        }
        Expr::ListLit(xs) | Expr::TupleLit(xs) | Expr::SetLit(xs) => {
            for x in xs.iter_mut() {
                stamp_source_expr(x, source);
            }
        }
        Expr::DictLit(pairs) => {
            for (k, v) in pairs.iter_mut() {
                stamp_source_expr(k, source);
                stamp_source_expr(v, source);
            }
        }
        Expr::Comp { key, value, clauses, .. } => {
            for cl in clauses.iter_mut() {
                match cl {
                    ast::CompClause::For { iter, .. } => stamp_source_expr(iter, source),
                    ast::CompClause::If(c) => stamp_source_expr(c, source),
                }
            }
            stamp_source_expr(key, source);
            if let Some(v) = value {
                stamp_source_expr(v, source);
            }
        }
        Expr::Index(a, b, span) => {
            span.start.source = source;
            stamp_source_expr(a, source);
            stamp_source_expr(b, source);
        }
        Expr::Attr(a, _, span) => {
            span.start.source = source;
            stamp_source_expr(a, source);
        }
        Expr::Slice(a, lo, hi, step) => {
            stamp_source_expr(a, source);
            for part in [lo, hi, step] {
                if let Some(x) = part {
                    stamp_source_expr(x, source);
                }
            }
        }
        Expr::MethodCall { args, .. } => {
            for a in args.iter_mut() {
                stamp_source_expr(a, source);
            }
        }
        Expr::IfExp { cond, then, els } => {
            stamp_source_expr(cond, source);
            stamp_source_expr(then, source);
            stamp_source_expr(els, source);
        }
        Expr::FStr(parts) => stamp_source_fstr(parts, source),
        Expr::Lambda { body, .. } => stamp_source_expr(body, source),
        Expr::Await(a) => stamp_source_expr(a, source),
        Expr::Walrus { value, .. } => stamp_source_expr(value, source),
        Expr::CellNew(a) | Expr::CellGet(a) => stamp_source_expr(a, source),
        Expr::CellSet(a, b) => {
            stamp_source_expr(a, source);
            stamp_source_expr(b, source);
        }
        Expr::ClosureVal(_, env) => stamp_source_expr(env, source),
        Expr::ModEnv => {}
        Expr::Int(_) | Expr::BigInt(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bytes(_) | Expr::Bool(_)
        | Expr::NoneLit | Expr::FnVal(_) => {}
    }
}

fn stamp_source_pattern(p: &mut ast::Pattern, source: u16) {
    match p {
        ast::Pattern::Lit(e) => stamp_source_expr(e, source),
        ast::Pattern::Seq(pats) | ast::Pattern::Or(pats) => {
            for x in pats.iter_mut() {
                stamp_source_pattern(x, source);
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

// ---------------------------------------------------------------------
// psa explain <code>
// ---------------------------------------------------------------------

/// Codes built by hand rather than from registry rows: the two
/// multi-site diagnostics, the unclassified buckets and the host
/// family. (code, rule, one-line summary)
const SPECIAL_CODES: &[(&str, &str, &str)] = &[
    ("PSA-E1001", "use-after-move", "use of a name after it was moved; the diagnostic carries two sites (the use and the move)"),
    ("PSA-E1002", "use-after-del", "use of a name after `del`; the diagnostic carries two sites (the use and the del)"),
    ("PSA-E6009", "internal-unclassified", "an unregistered psa internal error (a psa bug, not a problem in your code)"),
    ("PSA-E9000", "unclassified-emit-error", "an emit error whose message is not in the registry yet"),
    ("PSA-L900", "unclassified-lex-error", "a lexer error whose message is not in the registry yet"),
    ("PSA-P900", "unclassified-parse-error", "a parse error whose message is not in the registry yet"),
    ("PSA-H002", "host-file-error", "a file could not be read, written or copied"),
    ("PSA-H003", "host-pipeline-error", "the sla/sa host pipeline failed or produced no output"),
    ("PSA-H004", "circular-import", "modules import each other in a cycle"),
    ("PSA-H005", "module-top-level", "a module top level holds something other than defs, classes and literal assignments"),
    ("PSA-H006", "module-class-collision", "two modules define a class of the same name (the class registry is program-wide in v1)"),
    ("PSA-H007", "from-import-collision", "a from-import collides with an existing definition"),
    ("PSA-H008", "from-import-class-alias", "a class cannot be aliased by a from-import in v1"),
    ("PSA-H009", "from-import-name-collision", "a from-import collides with an existing imported name"),
    ("PSA-H010", "from-import-std-module", "from-imports of psa_std modules are not supported; use `import` and call through the module"),
    ("PSA-H011", "cannot-import-name", "the module does not define the imported name"),
];

/// `psa explain <code>`: look a code up in the panic table or the
/// compile registry and print what it means and how to fix it.
/// Accepts 77069, PSA-R77069, PSA-E1001, PSA-L014, PSA-H003 forms.
/// Unknown codes fail loudly (exit 1).
fn explain(arg: &str, format: ErrFormat) -> i32 {
    let a = arg.trim().to_ascii_uppercase();
    let digits: Option<String> = if let Some(rest) = a.strip_prefix("PSA-R") {
        Some(rest.to_string())
    } else if !a.is_empty() && a.chars().all(|c| c.is_ascii_digit()) {
        Some(a.clone())
    } else {
        None
    };
    if let Some(d) = digits {
        return match d.parse::<u32>() {
            Ok(n) => explain_panic(n, format),
            Err(_) => {
                eprintln!("psa explain: `{}` is not a valid panic code", arg);
                1
            }
        };
    }
    explain_compile(&a, arg, format)
}

fn explain_panic(n: u32, format: ErrFormat) -> i32 {
    match panic_codes::lookup_panic(n) {
        Some(info) if info.reserved => {
            println!("panic {} - reserved: {}", n, info.summary);
            0
        }
        Some(info) => {
            if format == ErrFormat::Json {
                let d = Diagnostic {
                    code: format!("PSA-R{}", n),
                    rule: info.rule.to_string(),
                    kind: diag::Kind::Runtime,
                    file: None,
                    sites: Vec::new(),
                    message: info.summary.to_string(),
                    note: None,
                    hint: info.fix.map(|s| s.to_string()),
                    host: None,
                    mapping: None,
                };
                println!("{}", d.render_json());
            } else {
                println!("panic {} - {} (PSA-R{})", n, info.summary, n);
                println!("  category: {}", info.category);
                if let Some(cp) = info.cp_surface {
                    println!("  CPython surface: {}", cp);
                }
                if let Some(fix) = info.fix {
                    println!("  fix: {}", fix);
                }
            }
            0
        }
        None => {
            eprintln!("psa explain: unknown panic code {}", n);
            1
        }
    }
}

fn explain_compile(code: &str, original: &str, format: ErrFormat) -> i32 {
    if let Some((_, rule, summary)) = SPECIAL_CODES.iter().find(|(c, _, _)| *c == code) {
        if format == ErrFormat::Json {
            let d = Diagnostic {
                code: code.to_string(),
                rule: rule.to_string(),
                kind: if code.starts_with("PSA-H") {
                    diag::Kind::Host
                } else if code.starts_with("PSA-L") {
                    diag::Kind::Lex
                } else if code.starts_with("PSA-P") {
                    diag::Kind::Parse
                } else {
                    diag::Kind::Emit
                },
                file: None,
                sites: Vec::new(),
                message: summary.to_string(),
                note: None,
                hint: None,
                host: None,
                mapping: None,
            };
            println!("{}", d.render_json());
        } else {
            println!("{} ({})", code, rule);
            println!("  {}", summary);
        }
        return 0;
    }
    match diag::lookup_compile_code(code) {
        Some(e) => {
            if format == ErrFormat::Json {
                let d = Diagnostic {
                    code: e.code.to_string(),
                    rule: e.rule.to_string(),
                    kind: if e.internal {
                        diag::Kind::Internal
                    } else if e.code.starts_with("PSA-L") {
                        diag::Kind::Lex
                    } else if e.code.starts_with("PSA-P") {
                        diag::Kind::Parse
                    } else {
                        diag::Kind::Emit
                    },
                    file: None,
                    sites: Vec::new(),
                    message: e.pattern.to_string(),
                    note: None,
                    hint: e.hint.map(|s| s.to_string()),
                    host: None,
                    mapping: None,
                };
                println!("{}", d.render_json());
            } else {
                println!("{} ({})", e.code, e.rule);
                println!("  message template: {}", e.pattern);
                if let Some(clean) = e.clean {
                    println!("  shown as: {}", clean);
                }
                if let Some(h) = e.hint {
                    println!("  hint: {}", h);
                }
            }
            0
        }
        None => {
            eprintln!("psa explain: unknown code `{}`", original);
            1
        }
    }
}

// ---------------------------------------------------------------------
// psa map-loc <map.json> --psa <file>:<line> [--col N] | --sla <line>
// (SOURCEMAP-DESIGN §3.3; implementation package P2.)
//
// Prints one JSON object on stdout. Exit codes: 0 for a query
// result — including an explicit `unmapped` result, which is an
// answer, not a failure; 1 for map errors (unreadable /
// unparseable / stale sha / source not in map); 2 for bad
// arguments. The map's sources are sha-verified before any query
// is answered, so a stale map is a loud refusal, never a
// degraded answer.
// ---------------------------------------------------------------------

fn report_map_error(e: &srcmap::MapError, format: ErrFormat) {
    match format {
        ErrFormat::Json => {
            eprintln!(
                "{{\"error\":{},\"kind\":{},\"detail\":{}}}",
                json_str("map-loc failed"),
                json_str(e.kind_str()),
                json_str(&e.detail)
            );
        }
        ErrFormat::Text => {
            eprintln!("psa map-loc: {}", e);
        }
    }
}

fn map_loc(args: &[String], format: ErrFormat) -> i32 {
    if args.is_empty() {
        eprintln!("psa map-loc: missing <map.json>");
        usage();
        return 2;
    }
    let map_path = PathBuf::from(&args[0]);
    let rest = &args[1..];
    let psa_spec = parse_flag_value(rest, "--psa");
    let sla_spec = parse_flag_value(rest, "--sla");
    let col_spec = parse_flag_value(rest, "--col");
    match (&psa_spec, &sla_spec) {
        (Some(_), Some(_)) => {
            eprintln!("psa map-loc: --psa and --sla are mutually exclusive (one query per invocation)");
            return 2;
        }
        (None, None) => {
            eprintln!("psa map-loc: need exactly one of --psa <file>:<line> or --sla <line>");
            usage();
            return 2;
        }
        _ => {}
    }
    let col: Option<u32> = match col_spec {
        Some(c) => match c.parse::<u32>() {
            Ok(n) if n >= 1 => Some(n),
            _ => {
                eprintln!("psa map-loc: --col needs a positive integer, got `{}`", c);
                return 2;
            }
        },
        None => None,
    };
    if col.is_some() && sla_spec.is_some() {
        eprintln!("psa map-loc: --col only applies to --psa queries");
        return 2;
    }
    let map = match srcmap::SourceMap::read_from(&map_path) {
        Ok(m) => m,
        Err(e) => {
            report_map_error(&e, format);
            return 1;
        }
    };
    let map_dir = map_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    // A stale map is a loud refusal, never a degraded answer.
    if let Err(e) = map.verify_sources(&map_dir) {
        report_map_error(&e, format);
        return 1;
    }
    let map_display = map_path.display().to_string();
    if let Some(spec) = psa_spec {
        // <file>:<line>, split at the last ':'.
        let (file, line_s) = match spec.rsplit_once(':') {
            Some((f, l)) if !f.is_empty() => (f, l),
            _ => {
                eprintln!("psa map-loc: --psa needs <file>:<line>, got `{}`", spec);
                return 2;
            }
        };
        let line: u32 = match line_s.parse() {
            Ok(n) if n >= 1 => n,
            _ => {
                eprintln!("psa map-loc: bad psa line in `{}`", spec);
                return 2;
            }
        };
        let source_idx = match map.resolve_source(file, &map_dir) {
            Ok(i) => i,
            Err(e) => {
                report_map_error(&e, format);
                return 1;
            }
        };
        if source_idx > u16::MAX as usize {
            eprintln!("psa map-loc: source index {} exceeds the map format's range", source_idx);
            return 1;
        }
        let source_path = map.sources[source_idx].path.clone();
        let col_json = match col {
            Some(c) => c.to_string(),
            None => "null".to_string(),
        };
        let query_json = format!(
            "\"query\":{{\"kind\":\"psa-to-sla\",\"source\":{},\"source_index\":{},\"line\":{},\"col\":{}}}",
            json_str(&source_path),
            source_idx,
            line,
            col_json
        );
        match map.lookup_psa(source_idx as u16, line, col) {
            srcmap::PsaLookup::Mapped {
                segments,
                approximate,
                note,
            } => {
                // All user segments for the line — never just the
                // first (design §3.3).
                let mut segs = String::new();
                for (i, &si) in segments.iter().enumerate() {
                    if i > 0 {
                        segs.push(',');
                    }
                    segs.push_str(&srcmap::segment_to_json(&map.segments[si]));
                }
                let mut out = String::new();
                out.push('{');
                out.push_str(&query_json);
                out.push_str(&format!(
                    ",\"status\":\"mapped\",\"approximate\":{},\"segments\":[{}],\"map\":{}",
                    approximate,
                    segs,
                    json_str(&map_display)
                ));
                if let Some(n) = &note {
                    out.push_str(&format!(",\"note\":{}", json_str(n)));
                }
                out.push('}');
                println!("{}", out);
            }
            srcmap::PsaLookup::Unmapped { reason } => {
                println!(
                    "{{{},\"status\":\"unmapped\",\"approximate\":false,\"reason\":{},\"map\":{}}}",
                    query_json,
                    json_str(&reason),
                    json_str(&map_display)
                );
            }
        }
        return 0;
    }
    // --sla <line>: generated -> psa.
    let line: u32 = match sla_spec.unwrap().parse() {
        Ok(n) if n >= 1 => n,
        _ => {
            eprintln!("psa map-loc: --sla needs a positive integer line number");
            return 2;
        }
    };
    let query_json = format!(
        "\"query\":{{\"kind\":\"sla-to-psa\",\"line\":{}}}",
        line
    );
    match map.lookup_gen(line) {
        srcmap::GenLookup::Mapped {
            segment,
            approximate,
            note,
        } => {
            let seg = &map.segments[segment];
            let status = if approximate { "approximate" } else { "mapped" };
            let mut out = String::new();
            out.push('{');
            out.push_str(&query_json);
            out.push_str(&format!(
                ",\"status\":\"{}\",\"approximate\":{},\"segment\":{}",
                status,
                approximate,
                srcmap::segment_to_json(seg)
            ));
            match (seg.src.source, seg.src.line) {
                (source, l) if l != 0 => {
                    let path = map
                        .sources
                        .get(source as usize)
                        .map(|s| s.path.clone())
                        .unwrap_or_default();
                    let snippet = map
                        .source_line_text(source, l, &map_dir)
                        .map(|t| json_str(&t))
                        .unwrap_or_else(|| "null".to_string());
                    let col_start = match seg.src.col_start {
                        Some(c) => c.to_string(),
                        None => "null".to_string(),
                    };
                    let col_end = match seg.src.col_end {
                        Some(c) => c.to_string(),
                        None => "null".to_string(),
                    };
                    out.push_str(&format!(
                        ",\"source\":{{\"path\":{},\"source_index\":{},\"line\":{},\"col_start\":{},\"col_end\":{},\"snippet\":{}}}",
                        json_str(&path),
                        source,
                        l,
                        col_start,
                        col_end,
                        snippet
                    ));
                }
                // Prelude / unanchored synthetic: there is no psa
                // line to report, and the segment itself says so
                // (origin + precision none) — no invented source.
                _ => out.push_str(",\"source\":null"),
            }
            out.push_str(&format!(",\"map\":{}", json_str(&map_display)));
            if let Some(n) = &note {
                out.push_str(&format!(",\"note\":{}", json_str(n)));
            }
            out.push('}');
            println!("{}", out);
        }
        srcmap::GenLookup::Unmapped { reason } => {
            println!(
                "{{{},\"status\":\"unmapped\",\"approximate\":false,\"reason\":{},\"map\":{}}}",
                query_json,
                json_str(&reason),
                json_str(&map_display)
            );
        }
    }
    0
}
