//! Structured diagnostics for psa (clear-error-feedback project,
//! stage 1). One `Diagnostic` object per error; the human text and
//! the NDJSON form are both renderings of it, so they can never
//! drift apart (ERROR-FORMAT-DESIGN.md sections 2-4).
//!
//! Pipeline stages build diagnostics through `Diagnostic::lex`,
//! `::parse` and `::emit`, which classify the historical message
//! text against the template registry (diag_registry.rs) to attach
//! a stable code, a rule name, and -- where written -- a note, a
//! hint and a cleaned-up message. Codes are stable and never
//! reused; unregistered messages fall back to a per-stage
//! unclassified code rather than losing the diagnostic.

use std::fmt;

/// Which pipeline stage produced the diagnostic. `Internal` is a
/// psa bug, not a user error (design section 2); `Runtime` is used
/// only by `psa explain` renderings of panic codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Lex,
    Parse,
    Emit,
    Runtime,
    Host,
    Internal,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Lex => "lex",
            Kind::Parse => "parse",
            Kind::Emit => "emit",
            Kind::Runtime => "runtime",
            Kind::Host => "host",
            Kind::Internal => "internal",
        }
    }
}

/// One source position. `col` is `Some` when the reporting
/// stage knew the column (P0a sourcemap: the lexer and AST now
/// carry real columns; P0b: parse errors raised by
/// `Parser::expect` carry the offending token's column); it
/// stays `None` for positions built from a line number alone,
/// and the text rendering keeps its old `:line` form in that
/// case. `snippet` is the source line text at the position
/// when the caller had the source at hand (filled by the
/// driver, which owns the file text); the text rendering prints
/// it with a `^` caret under `col` when both are present.
/// The JSON shape is unchanged. (P3: `snippet` is also the psa
/// side of a dual host diagnostic; `None` when no line text was
/// available, e.g. a multi-file segment — see hostdiag.)
#[derive(Debug, Clone)]
pub struct Site {
    pub line: u32,
    pub col: Option<u32>,
    pub role: String,
    pub snippet: Option<String>,
}

impl Site {
    pub fn new(line: u32, role: &str) -> Site {
        Site {
            line,
            col: None,
            role: role.to_string(),
            snippet: None,
        }
    }

    /// A site with a known column (P0a sourcemap).
    pub fn at(line: u32, col: u32, role: &str) -> Site {
        Site {
            line,
            col: Some(col),
            role: role.to_string(),
            snippet: None,
        }
    }

    /// Attach the source line text for the text rendering's
    /// excerpt + caret. The caret points at `col`; when the
    /// site has no column the excerpt prints without a caret.
    pub fn with_snippet(mut self, snippet: String) -> Site {
        self.snippet = Some(snippet);
        self
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub code: String,
    pub rule: String,
    pub kind: Kind,
    pub file: Option<String>,
    pub sites: Vec<Site>,
    pub message: String,
    pub note: Option<String>,
    pub hint: Option<String>,
    /// P3: the captured host invocation behind a PSA-H003
    /// (its parsed diagnostics carry their own mappings).
    /// `None` for every non-host diagnostic.
    pub host: Option<crate::hostdiag::HostReport>,
    /// P3: the top-level mapping of the host failure — the
    /// first diagnostic's mapping, or the whole-report outcome
    /// when the host produced no parsable diagnostic. `None`
    /// for every non-host diagnostic.
    pub mapping: Option<crate::hostdiag::MappingOutcome>,
}

/// One row of the message-template registry (generated; see
/// diag_registry.rs). `pattern` is the verbatim format template of
/// an err/perr/eerr call site, with `{...}` placeholders.
pub struct RegistryEntry {
    pub kind: Kind,
    pub pattern: &'static str,
    pub code: &'static str,
    pub rule: &'static str,
    pub note: Option<&'static str>,
    pub hint: Option<&'static str>,
    pub clean: Option<&'static str>,
    pub internal: bool,
}

/// Split a template into literal segments around `{...}`
/// placeholders.
fn segments(pattern: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = pattern;
    loop {
        match rest.find('{') {
            None => {
                out.push(rest);
                break;
            }
            Some(i) => match rest[i..].find('}') {
                Some(j) => {
                    out.push(&rest[..i]);
                    rest = &rest[i + j + 1..];
                }
                None => {
                    out.push(rest);
                    break;
                }
            },
        }
    }
    out
}

/// Match `msg` against a registry `pattern`. On success returns the
/// captured placeholder fillings in order. A pattern without
/// placeholders requires exact equality; otherwise the first/last
/// literal segments anchor at the start/end when non-empty and the
/// middle segments must appear in order.
fn match_template(pattern: &str, msg: &str) -> Option<Vec<String>> {
    let segs = segments(pattern);
    if segs.len() == 1 {
        return if segs[0] == msg {
            Some(Vec::new())
        } else {
            None
        };
    }
    let mut caps: Vec<String> = Vec::new();
    let mut pos = 0usize;
    for (idx, seg) in segs.iter().enumerate() {
        if seg.is_empty() {
            if idx == segs.len() - 1 {
                caps.push(msg[pos..].to_string());
            }
            continue;
        }
        if idx == 0 {
            if !msg.starts_with(seg) {
                return None;
            }
            pos = seg.len();
            continue;
        }
        if idx == segs.len() - 1 && !pattern.ends_with('}') {
            if !msg.ends_with(seg) {
                return None;
            }
            let end = msg.len() - seg.len();
            if end < pos {
                return None;
            }
            caps.push(msg[pos..end].to_string());
            pos = msg.len();
            continue;
        }
        match msg[pos..].find(seg) {
            Some(off) => {
                caps.push(msg[pos..pos + off].to_string());
                pos += off + seg.len();
            }
            None => return None,
        }
    }
    if pos != msg.len() && !pattern.ends_with('}') {
        return None;
    }
    Some(caps)
}

/// Fill a `clean` template's placeholders with captured values, in
/// order. The generator guarantees the counts agree.
fn fill(template: &str, caps: &[String]) -> String {
    let segs = segments(template);
    if segs.len() == 1 {
        return template.to_string();
    }
    let mut out = String::new();
    for (i, seg) in segs.iter().enumerate() {
        out.push_str(seg);
        if i < caps.len() {
            out.push_str(&caps[i]);
        }
    }
    out
}

/// Classify a pipeline message: find its registry entry and build
/// the diagnostic. Falls back to a per-stage unclassified code
/// (the message is preserved verbatim) when nothing matches.
fn classify(kind: Kind, line: u32, role: &str, msg: &str) -> Diagnostic {
    classify_at(kind, line, None, role, msg)
}

/// `classify` with a known column: the primary site is built
/// with `col` (P0a sourcemap).
fn classify_at(kind: Kind, line: u32, col: Option<u32>, role: &str, msg: &str) -> Diagnostic {
    let site = match col {
        Some(c) => Site::at(line, c, role),
        None => Site::new(line, role),
    };
    for e in crate::diag_registry::REGISTRY.iter() {
        if e.kind != kind {
            continue;
        }
        if let Some(caps) = match_template(e.pattern, msg) {
            let message = match e.clean {
                Some(clean) => fill(clean, &caps),
                None => msg.to_string(),
            };
            return Diagnostic {
                code: e.code.to_string(),
                rule: e.rule.to_string(),
                kind: if e.internal { Kind::Internal } else { kind },
                file: None,
                sites: vec![site.clone()],
                message,
                note: e.note.map(|s| s.to_string()),
                hint: e.hint.map(|s| s.to_string()),
                host: None,
                mapping: None,
            };
        }
    }
    // Unclassified fallbacks. A message that calls itself internal
    // is a psa bug even when its exact text is unregistered.
    if kind == Kind::Emit && (msg.starts_with("internal error:") || msg.starts_with("internal:")) {
        return Diagnostic {
            code: "PSA-E6009".to_string(),
            rule: "internal-unclassified".to_string(),
            kind: Kind::Internal,
            file: None,
            sites: vec![site.clone()],
            message: msg.to_string(),
            note: Some("this is a psa internal error, not a problem in your code".to_string()),
            hint: Some("please report this with the source file".to_string()),
            host: None,
            mapping: None,
        };
    }
    let (code, rule) = match kind {
        Kind::Lex => ("PSA-L900", "unclassified-lex-error"),
        Kind::Parse => ("PSA-P900", "unclassified-parse-error"),
        _ => ("PSA-E9000", "unclassified-emit-error"),
    };
    Diagnostic {
        code: code.to_string(),
        rule: rule.to_string(),
        kind,
        file: None,
        sites: vec![site],
        message: msg.to_string(),
        note: None,
        hint: None,
        host: None,
        mapping: None,
    }
}

impl Diagnostic {
    pub fn lex(line: u32, msg: &str) -> Diagnostic {
        classify(Kind::Lex, line, "offending-token", msg)
    }

    /// `lex` with a known column (P0a sourcemap).
    pub fn lex_at(line: u32, col: u32, msg: &str) -> Diagnostic {
        classify_at(Kind::Lex, line, Some(col), "offending-token", msg)
    }

    pub fn parse(line: u32, msg: &str) -> Diagnostic {
        classify(Kind::Parse, line, "found-token", msg)
    }

    /// Parse error at a known column (P0b): same as `parse`, but
    /// the primary site carries the offending token's 1-based
    /// character column. Used by `Parser::expect`. (P0a's
    /// `parse_at` alias was dropped in the merge — no callers;
    /// `classify_at` stays as the shared helper.)
    pub fn parse_col(line: u32, col: u32, msg: &str) -> Diagnostic {
        let mut d = classify(Kind::Parse, line, "found-token", msg);
        if let Some(s) = d.sites.first_mut() {
            s.col = Some(col);
        }
        d
    }

    pub fn emit(line: u32, msg: &str) -> Diagnostic {
        classify(Kind::Emit, line, "error", msg)
    }

    /// `emit` with a known column (P0a sourcemap).
    pub fn emit_at(line: u32, col: u32, msg: &str) -> Diagnostic {
        classify_at(Kind::Emit, line, Some(col), "error", msg)
    }

    /// Host / CLI / IO diagnostic with an explicit code (main.rs).
    pub fn host(code: &str, rule: &str, message: String) -> Diagnostic {
        Diagnostic {
            code: code.to_string(),
            rule: rule.to_string(),
            kind: Kind::Host,
            file: None,
            sites: Vec::new(),
            message,
            note: None,
            hint: None,
            host: None,
            mapping: None,
        }
    }

    /// Attach a captured host report (P3: main.rs host failure).
    pub fn with_host(mut self, host: crate::hostdiag::HostReport) -> Diagnostic {
        self.host = Some(host);
        self
    }

    /// Attach the top-level mapping of a host failure (P3).
    pub fn with_mapping(mut self, mapping: crate::hostdiag::MappingOutcome) -> Diagnostic {
        self.mapping = Some(mapping);
        self
    }

    /// Append a psa-source site with column + snippet (P3: the
    /// psa side of a dual host diagnostic).
    pub fn with_psa_site(mut self, line: u32, col: u32, snippet: Option<String>) -> Diagnostic {
        let mut site = Site::new(line, "psa-source");
        site.col = Some(col);
        site.snippet = snippet;
        self.sites.push(site);
        self
    }

    /// use-after-move (emit.rs check_alive): the one diagnostic
    /// with two structured sites in v1 (design section 3).
    /// Ruling #8 (sourcemap P0): the use and move sites are full
    /// `Site`s, so the caller supplies the name-span columns; the
    /// `file` is filled in by the caller (emit's `check_alive`
    /// resolves it from `Ctx.sources`).
    pub fn use_after_move(name: &str, use_site: Site, move_site: Site) -> Diagnostic {
        let (use_line, move_line) = (use_site.line, move_site.line);
        Diagnostic {
            code: "PSA-E1001".to_string(),
            rule: "use-after-move".to_string(),
            kind: Kind::Emit,
            file: None,
            sites: vec![use_site, move_site],
            message: format!("use of `{}` after it was moved", name),
            note: Some(
                "container assignment moves the value; the binding is dead until it is re-bound"
                    .to_string(),
            ),
            hint: Some(format!(
                "re-bind `{}` before line {}, or bind a copy at line {}",
                name, use_line, move_line
            )),
            host: None,
            mapping: None,
        }
    }

    /// use-after-del (emit.rs check_not_deleted).
    pub fn use_after_del(name: &str, use_line: u32, del_line: u32) -> Diagnostic {
        Diagnostic {
            code: "PSA-E1002".to_string(),
            rule: "use-after-del".to_string(),
            kind: Kind::Emit,
            file: None,
            sites: vec![Site::new(use_line, "use"), Site::new(del_line, "del")],
            message: format!("use of `{}` after `del`", name),
            note: Some("Python raises NameError for a name used after `del`".to_string()),
            hint: Some(format!("re-bind `{}` before line {}", name, use_line)),
            host: None,
            mapping: None,
        }
    }

    /// Append a site (e.g. the import statement for module errors).
    pub fn with_site(mut self, line: u32, role: &str) -> Diagnostic {
        self.sites.push(Site::new(line, role));
        self
    }

    /// Append a site with a known column (P0a sourcemap).
    pub fn with_site_at(mut self, line: u32, col: u32, role: &str) -> Diagnostic {
        self.sites.push(Site::at(line, col, role));
        self
    }

    /// Fill the primary site's column (P0a sourcemap): for a
    /// diagnostic built line-only by a stage that later learns
    /// the column (e.g. the parser re-anchoring an f-string
    /// diagnostic). No-op when there is no site.
    pub fn with_col(mut self, col: u32) -> Diagnostic {
        if let Some(s) = self.sites.first_mut() {
            s.col = Some(col);
        }
        self
    }

    /// Attach the primary site's source line text (P0a
    /// sourcemap): the text rendering prints it with a caret
    /// under the column. No-op when there is no site.
    pub fn with_snippet(mut self, snippet: String) -> Diagnostic {
        if let Some(s) = self.sites.first_mut() {
            s.snippet = Some(snippet);
        }
        self
    }

    pub fn with_note(mut self, note: String) -> Diagnostic {
        self.note = Some(match self.note {
            Some(old) => format!("{}; {}", old, note),
            None => note,
        });
        self
    }

    pub fn with_hint(mut self, hint: String) -> Diagnostic {
        self.hint = Some(hint);
        self
    }

    /// Attach the source file name when the pipeline did not know
    /// it (main.rs knows the path; lex/parse/emit do not).
    pub fn fill_file(&mut self, file: &str) {
        if self.file.is_none() {
            self.file = Some(file.to_string());
        }
    }

    /// Attach the source line text to every site that lacks one.
    /// Lex errors already carry their snippet from
    /// `lexer::err_at`; parse/emit errors only know (file, line,
    /// col), so the driver fills the excerpt here (P0a corpus
    /// README: "parse diagnostics get their excerpt only once
    /// main.rs attaches it (P3 wiring)" — the wiring lands with
    /// this integration). Reads the file named by `self.file`
    /// once; silently leaves snippets empty when the file cannot
    /// be read (synthetic diagnostic, removed file). Never
    /// overwrites a snippet already present, and never invents
    /// text for a line-0 (synthetic) site.
    pub fn fill_snippets(&mut self) {
        let Some(f) = self.file.clone() else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(&f) else {
            return;
        };
        for site in &mut self.sites {
            if site.snippet.is_none() && site.line >= 1 {
                if let Some(line) = text.lines().nth((site.line as usize).saturating_sub(1)) {
                    site.snippet = Some(line.to_string());
                }
            }
        }
    }

    /// Human rendering (design section 3):
    ///   error[PSA-E1001]: <message>
    ///     --> file:line:col          (col only when known)
    ///      | <source line text>      (only when a snippet is attached)
    ///      |  ^                      (caret under col; single caret —
    ///                                 Site carries no span end)
    ///      = site: <role> (line N[:C])   (for each extra site)
    ///   note: ...
    ///   hint: ...
    /// Without a column the header keeps the old `file:line`
    /// form and no excerpt is printed: the rendering never
    /// invents a column. A snippet counts one character per
    /// Unicode scalar for caret alignment (tabs are not
    /// expanded).
    ///
    /// P3 host rendering (design §3.4 dual presentation) replaces
    /// the `-->` block with one psa/sla/map block per host
    /// diagnostic, then the host output verbatim:
    ///   error[PSA-H003]: <message>
    ///   psa --> demo.psa:2:1
    ///   | def pick(flag):
    ///   sla --> demo.sla:2581 (fn __py_pick)
    ///   map: demo.psa-map.json (mapping: function-level — …)
    ///   host output (verbatim):
    ///   <host stdout+stderr, verbatim>
    ///   hint: ...
    /// The host text is never edited, translated or summarised;
    /// only trailing newlines are stripped for framing.
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("error[{}]: {}\n", self.code, self.message));
        if let Some(host) = &self.host {
            if host.diagnostics.is_empty() {
                self.render_mapping_block(&mut out, self.mapping.as_ref());
            } else {
                for d in &host.diagnostics {
                    self.render_mapping_block(&mut out, d.mapping.as_ref());
                }
            }
            let verbatim = host.verbatim();
            if !verbatim.is_empty() {
                out.push_str("host output (verbatim):\n");
                out.push_str(&verbatim);
                out.push('\n');
            }
            if let Some(note) = &self.note {
                out.push_str(&format!("note: {}\n", note));
            }
            if let Some(hint) = &self.hint {
                out.push_str(&format!("hint: {}\n", hint));
            }
            return out.trim_end_matches('\n').to_string();
        }
        if let Some(primary) = self.sites.first() {
            match (&self.file, primary.col) {
                (Some(f), Some(c)) => {
                    out.push_str(&format!("  --> {}:{}:{}\n", f, primary.line, c))
                }
                (Some(f), None) => out.push_str(&format!("  --> {}:{}\n", f, primary.line)),
                (None, Some(c)) => {
                    out.push_str(&format!("  --> line {}:{}\n", primary.line, c))
                }
                (None, None) => out.push_str(&format!("  --> line {}\n", primary.line)),
            }
            render_site_excerpt(&mut out, primary);
            for s in &self.sites[1..] {
                match s.col {
                    Some(c) => {
                        out.push_str(&format!("   = site: {} (line {}:{})\n", s.role, s.line, c))
                    }
                    None => out.push_str(&format!("   = site: {} (line {})\n", s.role, s.line)),
                }
                render_site_excerpt(&mut out, s);
            }
        }
        if let Some(note) = &self.note {
            out.push_str(&format!("note: {}\n", note));
        }
        if let Some(hint) = &self.hint {
            out.push_str(&format!("hint: {}\n", hint));
        }
        out.trim_end_matches('\n').to_string()
    }

    /// One psa/sla/map block of the dual presentation. `mapping`
    /// is the diagnostic's own outcome; when it is missing (a
    /// caller forgot to resolve) nothing psa-side is printed —
    /// the verbatim host output below still carries the error,
    /// and no position is ever invented.
    fn render_mapping_block(
        &self,
        out: &mut String,
        mapping: Option<&crate::hostdiag::MappingOutcome>,
    ) {
        let Some(m) = mapping else {
            return;
        };
        use crate::hostdiag::MappingStatus;
        match m.status {
            MappingStatus::Mapped | MappingStatus::FunctionLevel => {
                if let Some(line) = m.psa_line {
                    // The psa file comes from the mapping itself
                    // (a module hit names the module file); the
                    // diagnostic's own file is the main-file
                    // fallback (name-bridge hits).
                    let psa_file = m.psa_file.as_ref().or(self.file.as_ref());
                    match (psa_file, m.psa_col) {
                        (Some(f), Some(c)) => {
                            out.push_str(&format!("psa --> {}:{}:{}\n", f, line, c))
                        }
                        (Some(f), None) => {
                            out.push_str(&format!("psa --> {}:{}\n", f, line))
                        }
                        (None, _) => out.push_str(&format!("psa --> line {}\n", line)),
                    }
                    if let Some(snippet) = &m.snippet {
                        out.push_str(&format!("| {}\n", snippet));
                    }
                } else {
                    out.push_str("psa --> unmapped (reason: no-psa-line)\n");
                }
            }
            MappingStatus::Unmapped => {
                let reason = m.reason.unwrap_or("no-host-location");
                out.push_str(&format!("psa --> unmapped (reason: {})\n", reason));
            }
        }
        if let (Some(f), Some(l)) = (&m.generated_file, m.generated_line) {
            // Label by the generated file's kind: a `.sa`
            // position is not a `.sla` position, and the label
            // must not pretend otherwise.
            let label = if f.ends_with(".sa") { "sa" } else { "sla" };
            match &m.generated_fn {
                Some(name) => match &m.segment_origin {
                    Some(origin) => out.push_str(&format!(
                        "{} --> {}:{} (fn {}, segment origin={})\n",
                        label, f, l, name, origin
                    )),
                    None => {
                        out.push_str(&format!("{} --> {}:{} (fn {})\n", label, f, l, name))
                    }
                },
                None => out.push_str(&format!("{} --> {}:{}\n", label, f, l)),
            }
        }
        out.push_str(&format!(
            "map: {} (mapping: {} — {})\n",
            m.map_file,
            m.status.as_str(),
            m.note
        ));
    }

    /// NDJSON rendering (design section 4.2): one object, fields
    /// in schema order; optional fields are omitted when absent.
    /// P3 adds the design §3.4 `host` / `mapping` objects; old
    /// fields keep their meaning (sites carries the psa-source
    /// site when a mapping resolved, and stays empty when the
    /// host failure is unmapped).
    pub fn render_json(&self) -> String {
        let mut out = String::new();
        out.push('{');
        out.push_str(&format!("\"code\":{}", json_str(&self.code)));
        out.push_str(&format!(",\"rule\":{}", json_str(&self.rule)));
        out.push_str(&format!(",\"kind\":{}", json_str(self.kind.as_str())));
        if let Some(f) = &self.file {
            out.push_str(&format!(",\"file\":{}", json_str(f)));
        }
        out.push_str(",\"sites\":[");
        for (i, s) in self.sites.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let col = match s.col {
                Some(c) => c.to_string(),
                None => "null".to_string(),
            };
            out.push_str(&format!(
                "{{\"line\":{},\"col\":{},\"role\":{}",
                s.line,
                col,
                json_str(&s.role)
            ));
            if let Some(snippet) = &s.snippet {
                out.push_str(&format!(",\"snippet\":{}", json_str(snippet)));
            }
            out.push('}');
        }
        out.push(']');
        out.push_str(&format!(",\"message\":{}", json_str(&self.message)));
        if let Some(host) = &self.host {
            out.push_str(&format!(",\"host\":{}", host.render_json()));
        }
        if let Some(mapping) = &self.mapping {
            out.push_str(&format!(",\"mapping\":{}", mapping.render_json()));
        }
        if let Some(note) = &self.note {
            out.push_str(&format!(",\"note\":{}", json_str(note)));
        }
        if let Some(hint) = &self.hint {
            out.push_str(&format!(",\"hint\":{}", json_str(hint)));
        }
        out.push('}');
        out
    }

    pub fn render(&self, format: ErrFormat) -> String {
        match format {
            ErrFormat::Text => self.render_text(),
            ErrFormat::Json => self.render_json(),
        }
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render_text())
    }
}

/// Print a site's source excerpt + caret for the text
/// rendering (P0a sourcemap). Nothing prints when the site
/// carries no snippet; the caret prints only when the site
/// also carries a column, positioned under that column (one
/// space per preceding Unicode scalar; tabs not expanded).
fn render_site_excerpt(out: &mut String, s: &Site) {
    if let Some(snippet) = &s.snippet {
        out.push_str(&format!("   | {}\n", snippet));
        if let Some(c) = s.col {
            let pad = " ".repeat(c.saturating_sub(1) as usize);
            out.push_str(&format!("   | {}^\n", pad));
        }
    }
}

impl std::error::Error for Diagnostic {}

/// The `--error-format` / `PSA_ERROR_FORMAT` selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrFormat {
    Text,
    Json,
}

impl ErrFormat {
    pub fn parse(s: &str) -> Option<ErrFormat> {
        match s {
            "text" => Some(ErrFormat::Text),
            "json" => Some(ErrFormat::Json),
            _ => None,
        }
    }
}

/// Look up a compile-time code (PSA-L/P/E/H...) in the registry,
/// for `psa explain`. Returns the registry row, if any.
pub fn lookup_compile_code(code: &str) -> Option<&'static RegistryEntry> {
    crate::diag_registry::REGISTRY.iter().find(|e| e.code == code)
}

/// Minimal JSON string escaping (no serde dependency in psa).
/// Shared with srcmap.rs (map serialization uses the same escaper
/// so both JSON writers stay identical) and with `hostdiag`'s
/// renderers.
pub(crate) fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
