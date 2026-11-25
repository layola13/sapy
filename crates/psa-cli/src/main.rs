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
            println!("psa 0.4.0 (sapy v0.4, D0)");
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
    let prog = p.parse_program()?;
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
