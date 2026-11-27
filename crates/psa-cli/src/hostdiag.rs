//! Host-output capture, parsing and dual (psa + host) mapping
//! for the psa -> sla/sa delegation (sourcemap design §3.4/§3.5,
//! implementation package P3).
//!
//! Rules this module enforces (violating any of them is rework):
//!
//! * Host text is **never rewritten**: every parsed diagnostic
//!   keeps its verbatim `raw`; anything no adapter understands
//!   lands whole in `HostReport.raw`; the text renderer prints
//!   the combined host output verbatim under one header.
//! * No fabricated positions: when the map / name bridge cannot
//!   resolve a host location, the outcome is `Unmapped` with an
//!   explicit reason code — never a neighbouring line passed off
//!   as exact. Guesses only ever surface as `FunctionLevel`,
//!   labelled as such.
//! * The Rust-sla adapter is a **placeholder** (design §6
//!   unres. 2): its format has never been measured on any box
//!   available to this line, so nothing about it is invented
//!   here. Unparsed Rust-sla output falls through to `host.raw`
//!   with reason `unparsed-host-output`.
//!
//! Adapter chain (design §3.5), in order:
//! 1. SA trap JSON — one JSON object per line, with a `trap` key.
//! 2. sla/sa text diagnostics — `error[CODE]: msg` blocks, with
//!    `in function …`, `line N (expanded M): …`, `help: …` and
//!    `--> file:line[:col]` continuations.
//! 3. Rust sla — placeholder (see `try_rust_sla_adapter`).
//! A trap-JSON line and a text block describing the *same* host
//! error (equal code + message + function) are merged into one
//! diagnostic whose `raw` concatenates both verbatim forms in
//! original order — the host prints both forms of one error, and
//! presenting it twice would double-report a single failure.

use crate::ast;
use crate::minijson::{self, Value};
use crate::srcmap::{GenLookup, MapErrorKind, Origin, SourceMap};
use std::path::Path;

/// One parsed host diagnostic. `file`/`line`/`source_line` keep
/// the host's own coordinates: for a trap, `file` is the `.sa`
/// file, `line` the SA line and `source_line` the expanded line;
/// for a `--> file:line` text diagnostic, `file`/`line` are
/// whatever the host printed (a `.sla` file when the host points
/// at generated SLA). `upstream_*` carries the trap's
/// `upstream_loc` when the host fills it in future (design §6
/// unres. 3): a `.sla` upstream location upgrades a
/// function-level outcome to row-level automatically.
#[derive(Debug, Clone)]
pub struct HostDiagnostic {
    pub code: String,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub source_line: Option<u32>,
    pub column: Option<u32>,
    pub function: Option<String>,
    pub hint: Option<String>,
    pub upstream_file: Option<String>,
    pub upstream_line: Option<u32>,
    /// The diagnostic's verbatim host text (block lines joined
    /// with `\n`; a merged trap+text diagnostic concatenates
    /// both verbatim forms in original order).
    pub raw: String,
    /// Appearance order in the combined host output (for merges).
    pub order: usize,
    /// P3 mapping outcome for this diagnostic, filled by the
    /// driver after parsing (see `resolve_mapping`). `None`
    /// until resolved; renderers print no psa-side block then.
    pub mapping: Option<MappingOutcome>,
}

/// The whole captured host invocation (design §3.4 `host`).
#[derive(Debug, Clone)]
pub struct HostReport {
    pub tool: String,
    pub stage: String,
    pub exit_status: Option<i32>,
    pub diagnostics: Vec<HostDiagnostic>,
    /// Unparsed host output, verbatim (empty when everything
    /// was consumed by an adapter; omitted from JSON then).
    pub raw: String,
    /// Full host stdout/stderr, verbatim, for the text-mode
    /// "host output (verbatim):" dump. Trailing newlines are
    /// stripped only for rendering framing; every other byte
    /// is preserved. Printed stderr-first, then stdout (the
    /// two streams have no true interleaving once captured).
    pub verbatim_stderr: String,
    pub verbatim_stdout: String,
}

impl HostReport {
    pub fn verbatim(&self) -> String {
        let err = self.verbatim_stderr.trim_end_matches('\n');
        let out = self.verbatim_stdout.trim_end_matches('\n');
        match (err.is_empty(), out.is_empty()) {
            (true, true) => String::new(),
            (false, true) => err.to_string(),
            (true, false) => out.to_string(),
            (false, false) => format!("{}\n{}", err, out),
        }
    }
}

/// Mapping precision, design §3.4 (three grades, always
/// self-declared in the presentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingStatus {
    /// Row-level: segment-table hit on a generated `.sla` line.
    Mapped,
    /// Function-level: host gave only a generated-SA position
    /// or a function name; resolved to the psa `def` line via
    /// the name bridge. Explicitly NOT the error line.
    FunctionLevel,
    /// No psa position could be established honestly.
    Unmapped,
}

impl MappingStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            MappingStatus::Mapped => "mapped",
            MappingStatus::FunctionLevel => "function-level",
            MappingStatus::Unmapped => "unmapped",
        }
    }
}

/// Unmapped reason codes (design §3.7, plus one P3 addition).
pub mod reason {
    /// The host diagnostic carries no location at all.
    pub const NO_HOST_LOCATION: &str = "no-host-location";
    /// The host location is a generated-SA position only
    /// (`upstream_loc` null) and no usable function name came
    /// with it, or the generated line is prelude/std code.
    pub const SA_POSITION_ONLY: &str = "sa-position-only";
    /// A map file was present but its source sha no longer
    /// matches what the map records (`SourceMap::load` refused
    /// it; integration wiring in main.rs, P2's reader owns the
    /// check). The host output is still presented verbatim;
    /// only row-level mapping is lost.
    pub const STALE_MAP: &str = "stale-map";
    /// A map file was present but could not be read/parsed
    /// (corrupt JSON, unsupported version, I/O error). Distinct
    /// from `MAP_UNAVAILABLE` (no map file at all) and from
    /// `STALE_MAP` (well-formed but out of date).
    pub const BAD_MAP: &str = "bad-map";
    /// Host output matched no known adapter (includes the
    /// Rust-sla placeholder: an unmeasured format).
    pub const UNPARSED_HOST_OUTPUT: &str = "unparsed-host-output";
    /// P3 addition: the host gave a generated `.sla` line but
    /// no `<stem>.psa-map.json` exists to resolve it (P2 has
    /// not landed yet). Not one of design §3.7's four codes,
    /// because none of them covers a missing table.
    pub const MAP_UNAVAILABLE: &str = "map-unavailable";
}

/// One diagnostic's mapping outcome (design §3.4 `mapping`).
#[derive(Debug, Clone)]
pub struct MappingOutcome {
    pub status: MappingStatus,
    pub reason: Option<&'static str>,
    /// The .psa file the psa position belongs to (the map's
    /// `sources[]` path for row-level hits; the main file for
    /// name-bridge hits). `None` when no psa position exists.
    /// Integration addition: P3's shape had only a line, so a
    /// module hit would have been printed against the main file.
    pub psa_file: Option<String>,
    pub psa_line: Option<u32>,
    pub psa_col: Option<u32>,
    pub snippet: Option<String>,
    pub generated_file: Option<String>,
    pub generated_line: Option<u32>,
    pub generated_fn: Option<String>,
    /// Segment origin (`user` / `prelude` / …) when the outcome
    /// came from a segment-table hit; `None` for name-bridge
    /// (function-level) outcomes, where no segment was consulted.
    pub segment_origin: Option<String>,
    pub segment: Option<usize>,
    pub approximate: bool,
    pub map_file: String,
    pub note: String,
}

impl MappingOutcome {
    fn unmapped(
        reason: &'static str,
        map_file: String,
        note: String,
    ) -> MappingOutcome {
        MappingOutcome {
            status: MappingStatus::Unmapped,
            reason: Some(reason),
            psa_file: None,
            psa_line: None,
            psa_col: None,
            snippet: None,
            generated_file: None,
            generated_line: None,
            generated_fn: None,
            segment_origin: None,
            segment: None,
            approximate: false,
            map_file,
            note,
        }
    }
}

/// Rust-sla adapter status (design §6 unres. 2). The production
/// host is the Rust sla binary, but its diagnostic shape has
/// never been measured on any machine this line could reach, so
/// this adapter is a deliberate placeholder: it recognises
/// nothing, and everything it would have eaten stays in
/// `host.raw` with reason `unparsed-host-output`. When the
/// format is measured, this function is where the adapter plugs
/// in — before the residual step, after the trap/text adapters
/// — and this constant is replaced by the measured grammar.
/// Nothing about the Rust sla format is asserted anywhere else
/// in this file.
pub const RUST_SLA_ADAPTER_STATUS: &str =
    "placeholder — Rust sla diagnostic format unmeasured (design §6 unres. 2); output falls through to host.raw";

fn try_rust_sla_adapter(_line: &str) -> Option<HostDiagnostic> {
    // Placeholder: no measured Rust-sla grammar exists yet.
    // (The status string is referenced so the placeholder stays
    // visible in the build, not silently dead.)
    let _ = RUST_SLA_ADAPTER_STATUS;
    None
}

// ---------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------

/// Split host text into lines, dropping the single empty piece a
/// trailing newline leaves, so `lines.join("\n")` reproduces the
/// input modulo that trailing newline.
fn split_lines(text: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = text.split('\n').collect();
    if let Some(last) = lines.last() {
        if last.is_empty() {
            lines.pop();
        }
    }
    // A trailing '\r' (CRLF host output) is stripped per line so
    // adapters see clean text; `raw` is built from the stripped
    // lines. (No CRLF host output has been observed; this is
    // defensive.)
    lines
        .iter()
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect()
}

/// Parse captured host stdout/stderr into diagnostics + residual.
/// `tool`/`stage`/`exit_status` are psa-side facts about the
/// invocation, carried through into the report unchanged.
pub fn parse_host_output(
    tool: &str,
    stage: &str,
    exit_status: Option<i32>,
    stdout: &[u8],
    stderr: &[u8],
) -> HostReport {
    let stdout_text = String::from_utf8_lossy(stdout).into_owned();
    let stderr_text = String::from_utf8_lossy(stderr).into_owned();
    let mut diagnostics: Vec<HostDiagnostic> = Vec::new();
    let mut consumed: Vec<bool> = Vec::new();
    let mut all_lines: Vec<String> = Vec::new();
    // Diagnostics live on stderr in every observed host; stdout
    // is scanned second, so a diagnostic's `order` still reflects
    // its stream (stderr before stdout).
    for text in [&stderr_text, &stdout_text] {
        for line in split_lines(text) {
            all_lines.push(line.to_string());
            consumed.push(false);
        }
    }
    // `order` is the diagnostic's line index in the combined
    // stream (stderr lines first, then stdout), so merged
    // duplicates concatenate their verbatim raws in original
    // appearance order even though the trap pass runs before
    // the text pass.
    // Pass 1: SA trap JSON lines.
    for (i, line) in all_lines.iter().enumerate() {
        if consumed[i] {
            continue;
        }
        if let Some(d) = parse_trap_line(line, i) {
            diagnostics.push(d);
            consumed[i] = true;
        }
    }
    // Pass 2: sla/sa text diagnostic blocks.
    let mut i = 0usize;
    while i < all_lines.len() {
        if consumed[i] {
            i += 1;
            continue;
        }
        if let Some((diag, end)) = parse_text_block(&all_lines, i, &consumed, i) {
            diagnostics.push(diag);
            for j in i..end {
                consumed[j] = true;
            }
            i = end;
        } else {
            // Rust-sla placeholder adapter gets its look at the
            // line before it is left as residual.
            if let Some(mut d) = try_rust_sla_adapter(&all_lines[i]) {
                d.order = i;
                diagnostics.push(d);
                consumed[i] = true;
            }
            i += 1;
        }
    }
    // Pass 3: merge trap+text forms of the same host error.
    let diagnostics = merge_duplicate_forms(diagnostics);
    // Residual: every line no adapter ate, in original order.
    let mut residual: Vec<&str> = Vec::new();
    for (i, line) in all_lines.iter().enumerate() {
        if !consumed[i] {
            residual.push(line.as_str());
        }
    }
    HostReport {
        tool: tool.to_string(),
        stage: stage.to_string(),
        exit_status,
        diagnostics,
        raw: residual.join("\n"),
        verbatim_stderr: stderr_text,
        verbatim_stdout: stdout_text,
    }
}

/// One SA trap-JSON line (evidence B2 verbatim shape):
/// `{"trap":"FallthroughForbidden","trap_code":1014,
/// "file":"b15_ret_arms.sa","line":27287,
/// "source_line":78486,"column":null, …,
/// "upstream_loc":null,
/// "function":"@sla____py_pick(flag: u8) -> ptr:", …,
/// "message":"…","hint":"…"}`.
/// Returns `None` when the line is not a trap object.
fn parse_trap_line(line: &str, order: usize) -> Option<HostDiagnostic> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('{') || !trimmed.contains("\"trap\"") {
        return None;
    }
    let value = minijson::parse(trimmed)?;
    let trap = value.get("trap")?.as_str()?;
    let message = value
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or(trap)
        .to_string();
    let (upstream_file, upstream_line) = match value.get("upstream_loc") {
        Some(Value::Obj(_)) => {
            let loc = value.get("upstream_loc").unwrap();
            let f = loc.get("file").and_then(|v| v.as_str()).map(|s| s.to_string());
            let l = loc.get("line").and_then(|v| v.as_u32());
            (f, l)
        }
        _ => (None, None),
    };
    Some(HostDiagnostic {
        code: trap.to_string(),
        message,
        file: value
            .get("file")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        line: value.get("line").and_then(|v| v.as_u32()),
        source_line: value.get("source_line").and_then(|v| v.as_u32()),
        column: value.get("column").and_then(|v| v.as_u32()),
        function: value
            .get("function")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        hint: value
            .get("hint")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        upstream_file,
        upstream_line,
        raw: line.to_string(),
        order,
        mapping: None,
    })
}

/// Parse one `error[CODE]: message` text block starting at
/// `start`. A block runs until the next `error[` line, the next
/// trap-JSON line, or the end of the stream. Returns the
/// diagnostic and the index just past the block, or `None` when
/// line `start` does not open a diagnostic.
fn parse_text_block(
    lines: &[String],
    start: usize,
    consumed: &[bool],
    order: usize,
) -> Option<(HostDiagnostic, usize)> {
    let first = lines[start].trim_start();
    if !first.starts_with("error[") {
        return None;
    }
    let close = first.find(']')?;
    let code = first["error[".len()..close].to_string();
    let rest = first[close + 1..].trim_start();
    let message = rest.strip_prefix(':').unwrap_or(rest).trim().to_string();
    let mut diag = HostDiagnostic {
        code,
        message,
        file: None,
        line: None,
        source_line: None,
        column: None,
        function: None,
        hint: None,
        upstream_file: None,
        upstream_line: None,
        raw: String::new(),
        order,
        mapping: None,
    };
    let mut end = start + 1;
    while end < lines.len() {
        if consumed[end] {
            break;
        }
        let line = lines[end].trim_start();
        if line.starts_with("error[") {
            break;
        }
        if line.starts_with('{') && line.contains("\"trap\"") {
            break;
        }
        if let Some(f) = line.strip_prefix("in function ") {
            diag.function = Some(f.trim().to_string());
        } else if let Some(after) = line.strip_prefix("line ") {
            parse_line_coords(after, &mut diag);
        } else if let Some(after) = line.strip_prefix("--> ") {
            parse_arrow_loc(after, &mut diag);
        } else if let Some(after) = line
            .strip_prefix("help:")
            .or_else(|| line.strip_prefix("hint:"))
        {
            let text = after.trim().to_string();
            diag.hint = Some(match diag.hint.take() {
                Some(old) => format!("{}; {}", old, text),
                None => text,
            });
        }
        end += 1;
    }
    diag.raw = lines[start..end].join("\n");
    Some((diag, end))
}

/// `line 78486 (expanded 27287): <text>` — the first number is
/// the expanded/source line, the parenthesised one the SA line
/// (evidence B2: text `line 78486 (expanded 27287)` matches trap
/// `line: 27287, source_line: 78486`). Also accepts a bare
/// `line N: …` form, which sets only `line`. Only the text
/// before the first `:` is scanned, so numbers in the source
/// text after the colon can never be mistaken for coordinates.
fn parse_line_coords(after: &str, diag: &mut HostDiagnostic) {
    let head = after.split(':').next().unwrap_or(after);
    let mut nums: Vec<u32> = Vec::new();
    let mut cur = String::new();
    for ch in head.chars() {
        if ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            if let Ok(n) = cur.parse::<u32>() {
                nums.push(n);
            }
            cur.clear();
        }
    }
    if !cur.is_empty() {
        if let Ok(n) = cur.parse::<u32>() {
            nums.push(n);
        }
    }
    if nums.len() >= 2 {
        diag.source_line = Some(nums[0]);
        diag.line = Some(nums[1]);
    } else if nums.len() == 1 {
        diag.line = Some(nums[0]);
    }
}

/// `--> file:line[:col]` — the second diagnostic shape of
/// design §3.5. Split from the right: the last segment is the
/// column (three-part shape) or the line (two-part shape).
fn parse_arrow_loc(after: &str, diag: &mut HostDiagnostic) {
    let text = after.trim();
    let parts: Vec<&str> = text.rsplit(':').collect();
    if parts.len() < 2 {
        return;
    }
    // Three-part shape `file:line:col`.
    if parts.len() >= 3 {
        if let (Ok(line), Ok(col)) = (parts[1].parse::<u32>(), parts[0].parse::<u32>()) {
            let file: Vec<&str> = parts[2..].iter().rev().copied().collect();
            diag.file = Some(file.join(":"));
            diag.line = Some(line);
            diag.column = Some(col);
            return;
        }
    }
    // Two-part shape `file:line`.
    if let Ok(line) = parts[0].parse::<u32>() {
        let file: Vec<&str> = parts[1..].iter().rev().copied().collect();
        diag.file = Some(file.join(":"));
        diag.line = Some(line);
    }
}

/// Merge a trap-JSON diagnostic with the text diagnostic of the
/// same host error (equal code + message + function, both forms
/// present). The merged diagnostic keeps the trap's structured
/// coordinates, fills gaps from the text form, takes the text
/// form's `help:` when the trap has no hint, and concatenates
/// both verbatim raws in original order. Diagnostics that do not
/// match stay separate — in particular, two genuinely different
/// host errors are never merged.
fn merge_duplicate_forms(mut diags: Vec<HostDiagnostic>) -> Vec<HostDiagnostic> {
    diags.sort_by_key(|d| d.order);
    let mut out: Vec<HostDiagnostic> = Vec::new();
    for d in diags {
        let mut merged = false;
        for prev in out.iter_mut() {
            let same_error = prev.code == d.code
                && prev.message == d.message
                && prev.function.is_some()
                && prev.function == d.function;
            if !same_error {
                continue;
            }
            // Keep the first-appearing form's coordinates (the
            // two forms agree on them in every observed host);
            // fill only what it lacks from the other form.
            if prev.file.is_none() {
                prev.file = d.file.clone();
            }
            if prev.line.is_none() {
                prev.line = d.line;
            }
            if prev.source_line.is_none() {
                prev.source_line = d.source_line;
            }
            if prev.column.is_none() {
                prev.column = d.column;
            }
            if prev.hint.is_none() {
                prev.hint = d.hint.clone();
            }
            if prev.upstream_file.is_none() {
                prev.upstream_file = d.upstream_file.clone();
                prev.upstream_line = d.upstream_line;
            }
            // Concatenate raws in original appearance order
            // (`out` is built in ascending `order`, so `prev`
            // always appeared first).
            prev.raw = format!("{}\n{}", prev.raw, d.raw);
            merged = true;
            break;
        }
        if !merged {
            out.push(d);
        }
    }
    out.sort_by_key(|d| d.order);
    out
}

// ---------------------------------------------------------------------
// Mapping (design §3.4 three grades)
// ---------------------------------------------------------------------

/// Everything the resolver needs. `map` is the P2 segment table
/// when `<stem>.psa-map.json` exists next to the generated `.sla`
/// (loaded once by the caller); `None` until P2 lands, in which
/// case row-level resolution degrades explicitly (see
/// `reason::MAP_UNAVAILABLE`).
pub struct MapCtx<'a> {
    pub prog: &'a ast::Program,
    pub psa_src: &'a str,
    pub psa_file: String,
    pub sla_src: &'a str,
    pub sla_file: String,
    pub map_file: &'a Path,
    pub map: Option<&'a SourceMap>,
    /// The `SourceMap::load` error when a map file existed but
    /// could not be used (stale sha / corrupt / unreadable).
    /// `None` when there was no map file at all (then the
    /// `MAP_UNAVAILABLE` reason applies) or the load succeeded.
    pub map_error: Option<&'a crate::srcmap::MapError>,
}

/// Resolve one host diagnostic to a psa position (or an honest
/// unmapped outcome). Pure function of the diagnostic + context:
/// no I/O beyond the already-loaded map.
pub fn resolve_mapping(diag: &HostDiagnostic, ctx: &MapCtx) -> MappingOutcome {
    let map_file = ctx.map_file.display().to_string();
    // 1. A generated `.sla` line: direct row-level candidate.
    //    Sources, in order: the trap's `upstream_loc` (the
    //    design §6 unres. 3 upgrade path), then a `--> x.sla`
    //    location printed by the host itself.
    let sla_line: Option<u32> = if let (Some(f), Some(l)) =
        (&diag.upstream_file, diag.upstream_line)
    {
        if f.ends_with(".sla") {
            Some(l)
        } else {
            None
        }
    } else {
        None
    }
    .or_else(|| match (&diag.file, diag.line) {
        (Some(f), Some(l)) if f.ends_with(".sla") => Some(l),
        _ => None,
    });
    if let Some(gen_line) = sla_line {
        return resolve_sla_line(diag, gen_line, ctx, map_file);
    }
    // 2. A function name: the name bridge to the psa `def` line.
    if let Some(func) = &diag.function {
        return resolve_function(func, ctx, map_file);
    }
    // 3. Only a generated-SA position, no function to bridge on.
    if diag.file.as_deref().map(|f| f.ends_with(".sa")).unwrap_or(false)
        || diag.line.is_some()
        || diag.source_line.is_some()
    {
        let mut out = MappingOutcome::unmapped(
            reason::SA_POSITION_ONLY,
            map_file,
            "host reported only a generated-SA position (upstream_loc is null); no function name to bridge on".to_string(),
        );
        out.generated_file = diag.file.clone();
        out.generated_line = diag.line;
        return out;
    }
    // 4. No location at all in this diagnostic.
    MappingOutcome::unmapped(
        reason::NO_HOST_LOCATION,
        map_file,
        "host diagnostic carries no source location".to_string(),
    )
}

/// The whole-report outcome when the host produced no parsable
/// diagnostic at all: one unmapped outcome for the report. The
/// reason distinguishes "the host said nothing locatable" from
/// "the host said something no adapter understands".
pub fn resolve_empty_report(raw: &str, map_file: &Path) -> MappingOutcome {
    let map_file = map_file.display().to_string();
    if raw.to_lowercase().contains("error") {
        MappingOutcome::unmapped(
            reason::UNPARSED_HOST_OUTPUT,
            map_file,
            "host output matched no known diagnostic adapter (see Rust sla placeholder, design §6 unres. 2); the raw output is preserved below".to_string(),
        )
    } else {
        MappingOutcome::unmapped(
            reason::NO_HOST_LOCATION,
            map_file,
            "host output carries no source location".to_string(),
        )
    }
}

fn resolve_sla_line(
    diag: &HostDiagnostic,
    gen_line: u32,
    ctx: &MapCtx,
    map_file: String,
) -> MappingOutcome {
    let gen_file = diag
        .upstream_file
        .clone()
        .or_else(|| diag.file.clone())
        .unwrap_or_else(|| ctx.sla_file.to_string());
    let Some(map) = ctx.map else {
        let (reason_code, note) = match ctx.map_error {
            Some(e) if e.kind == MapErrorKind::StaleMap => (
                reason::STALE_MAP,
                format!(
                    "host reported generated line {} but {} is stale ({}); regenerate it with `psa check`",
                    gen_line,
                    ctx.map_file.display(),
                    e.detail
                ),
            ),
            Some(e) => (
                reason::BAD_MAP,
                format!(
                    "host reported generated line {} but {} could not be read ({}); regenerate it with `psa check`",
                    gen_line,
                    ctx.map_file.display(),
                    e.detail
                ),
            ),
            None => (
                reason::MAP_UNAVAILABLE,
                format!(
                    "host reported generated line {} but no {} exists to resolve it (P2 map not produced yet)",
                    gen_line,
                    ctx.map_file.display()
                ),
            ),
        };
        let mut out = MappingOutcome::unmapped(reason_code, map_file, note);
        out.generated_file = Some(gen_file);
        out.generated_line = Some(gen_line);
        out.generated_fn = diag.function.clone();
        return out;
    };
    // Canonical `lookup_gen`: exact covering segment first
    // (any origin), else the nearest preceding segment marked
    // approximate, else unmapped. Only user-origin segments map
    // to a psa position — a prelude/std hit is SA_POSITION_ONLY
    // (P3's exact-branch rule), and an approximate hit on a
    // non-user segment falls through to the no-user-segment
    // outcome below (P3's approx-branch rule).
    match map.lookup_gen(gen_line) {
        GenLookup::Mapped { segment: idx, approximate, note: _ } => {
            let seg = &map.segments[idx];
            if seg.origin == Origin::User {
                return mapped_from_segment(diag, idx, seg, ctx, gen_file, gen_line, map_file, approximate);
            }
            if !approximate {
                let mut out = MappingOutcome::unmapped(
                    reason::SA_POSITION_ONLY,
                    map_file,
                    format!(
                        "generated line {} is {} code, not user code",
                        gen_line,
                        seg.origin.as_str()
                    ),
                );
                out.generated_file = Some(gen_file);
                out.generated_line = Some(gen_line);
                return out;
            }
        }
        GenLookup::Unmapped { .. } => {}
    }
    let mut out = MappingOutcome::unmapped(
        reason::SA_POSITION_ONLY,
        map_file,
        format!(
            "generated line {} has no user segment in the map",
            gen_line
        ),
    );
    out.generated_file = Some(gen_file);
    out.generated_line = Some(gen_line);
    out
}

fn mapped_from_segment(
    diag: &HostDiagnostic,
    idx: usize,
    seg: &crate::srcmap::Segment,
    ctx: &MapCtx,
    gen_file: String,
    gen_line: u32,
    map_file: String,
    approximate: bool,
) -> MappingOutcome {
    // The psa file is the map's `sources[]` path for this
    // segment's source index (integration: P3 had no file here,
    // so a module hit would print against the main file). The
    // snippet reads from the main source text for source 0 and
    // from the recorded file (via the map, relative to the map
    // directory) for module sources; a missing file still
    // reports the line honestly, with no snippet.
    let psa_file = map_source_path(ctx, seg.src.source);
    let snippet = if seg.src.source == 0 {
        source_line_text(ctx.psa_src, seg.src.line)
    } else {
        ctx.map.and_then(|m| {
            let dir = ctx.map_file.parent().unwrap_or(Path::new("."));
            m.source_line_text(seg.src.source, seg.src.line, dir)
        })
    };
    let note = if approximate {
        format!(
            "nearest preceding user segment for generated line {} (approximate — the reported line falls between segments)",
            gen_line
        )
    } else {
        format!("exact segment match for generated line {}", gen_line)
    };
    MappingOutcome {
        status: MappingStatus::Mapped,
        reason: None,
        psa_file,
        psa_line: Some(seg.src.line),
        psa_col: seg.src.col_start,
        snippet,
        generated_file: Some(gen_file),
        generated_line: Some(gen_line),
        generated_fn: diag
            .function
            .as_ref()
            .map(|f| host_fn_name(f))
            .or(seg.name.clone()),
        segment_origin: Some(seg.origin.as_str().to_string()),
        segment: Some(idx),
        approximate,
        map_file,
        note,
    }
}

/// The recorded path for a map source index (`None` when
/// there is no map, or the index is out of range — the caller
/// then falls back to the main file).
fn map_source_path(ctx: &MapCtx, source: u16) -> Option<String> {
    ctx.map
        .and_then(|m| m.sources.get(source as usize))
        .map(|e| e.path.clone())
}

fn resolve_function(
    func: &str,
    ctx: &MapCtx,
    map_file: String,
) -> MappingOutcome {
    let sla_name = host_fn_name(func);
    // Generated `.sla` definition line for the `sla -->` line:
    // a scan of the generated text for `fn <name>(`.
    let generated_line = find_fn_line(ctx.sla_src, &sla_name);
    let mut base = MappingOutcome {
        status: MappingStatus::FunctionLevel,
        reason: None,
        psa_file: Some(ctx.psa_file.clone()),
        psa_line: None,
        psa_col: None,
        snippet: None,
        generated_file: Some(ctx.sla_file.to_string()),
        generated_line,
        generated_fn: Some(sla_name.clone()),
        segment_origin: None,
        segment: None,
        approximate: false,
        map_file: map_file.clone(),
        note: "host reported a generated-SA position; the psa line is the definition site of the generated function".to_string(),
    };
    match bridge_fn_name(&sla_name, ctx.prog) {
        Bridge::Def { line, col, what } => {
            base.psa_line = Some(line);
            // The def keyword's real column (P0b anchor; methods
            // sit indented). Column 1 by fallback, never 0.
            base.psa_col = Some(col.max(1));
            base.snippet = source_line_text(ctx.psa_src, line);
            base.note = format!(
                "host reported a generated-SA position; the psa line is the definition site of the generated function ({})",
                what
            );
            base
        }
        Bridge::Main { line, col } => {
            base.psa_line = Some(line);
            base.psa_col = Some(col.max(1));
            base.snippet = source_line_text(ctx.psa_src, line);
            base.note = "host reported a generated-SA position; the psa line is the first top-level statement of the generated main entry (function-level, not the error line)".to_string();
            base
        }
        Bridge::NoDef { why } => {
            let mut out = MappingOutcome::unmapped(reason::SA_POSITION_ONLY, map_file, why);
            out.generated_file = Some(ctx.sla_file.to_string());
            out.generated_line = generated_line;
            out.generated_fn = Some(sla_name);
            out
        }
    }
}

/// Strip the SA host's `@sla__` prefix and signature suffix:
/// `@sla____py_pick(flag: u8) -> ptr:` -> `__py_pick`.
fn host_fn_name(func: &str) -> String {
    let mut name = func.trim();
    name = name.strip_prefix("@sla__").unwrap_or(name);
    name = name.strip_suffix(':').unwrap_or(name);
    let end = name
        .find('(')
        .unwrap_or_else(|| name.find(' ').unwrap_or(name.len()));
    name[..end].trim().to_string()
}

/// Scan generated `.sla` text for `fn <name>(` (1-based line).
/// `main` is `fn main() -> i32` in the generated text.
fn find_fn_line(sla_src: &str, sla_name: &str) -> Option<u32> {
    let needle = format!("fn {}(", sla_name);
    for (i, line) in sla_src.lines().enumerate() {
        if line.trim_start().starts_with(&needle) {
            return Some((i + 1) as u32);
        }
    }
    None
}

/// Number of builtin exception classes the emitter registers
/// before user classes (emit.rs: ids 0..=8), so a user class's
/// `__py_cls_<id>_` id is 9 + its index in `prog.classes`.
/// P3-assumed from the emitter comment; to be verified on the
/// box (see RESULT.md).
const BUILTIN_CLASS_COUNT: usize = 9;

enum Bridge {
    /// A psa `def` (or method) line was found. `col` is the
    /// def keyword's column from its P0b anchor (integration:
    /// P3 hardcoded 1; methods are indented, so the anchor
    /// column is the honest value).
    Def { line: u32, col: u32, what: String },
    /// The generated `main` entry: anchored at the first
    /// top-level statement (`col` is that statement's anchor
    /// column).
    Main { line: u32, col: u32 },
    /// No psa definition corresponds to this generated name.
    NoDef { why: String },
}

/// The name bridge (design §3.1/§3.5): generated symbol ->
/// psa definition line, using only the emitter's deterministic
/// mangling (emit.rs `sla_fn_name` / `__py_mod_` /
/// `__py_cls_<id>_`).
fn bridge_fn_name(sla_name: &str, prog: &ast::Program) -> Bridge {
    if sla_name == "main" {
        for st in &prog.top {
            let line = stmt_line(st);
            if line > 0 {
                return Bridge::Main { line, col: st.anchor().col };
            }
        }
        return Bridge::NoDef {
            why: "generated main entry has no top-level statement to anchor to".to_string(),
        };
    }
    if let Some(rest) = sla_name.strip_prefix("__py_cls_") {
        // `__py_cls_<id>_<method>`
        if let Some((id_s, method)) = rest.split_once('_') {
            if let Ok(id) = id_s.parse::<usize>() {
                if let Some(idx) = id.checked_sub(BUILTIN_CLASS_COUNT) {
                    if let Some(class) = prog.classes.get(idx) {
                        if let Some(m) = class.methods.iter().find(|m| m.name == method) {
                            return Bridge::Def {
                                line: m.line,
                                col: m.anchor.col,
                                what: format!("method {}.{} (class id {})", class.name, method, id),
                            };
                        }
                    }
                }
            }
        }
        return Bridge::NoDef {
            why: format!(
                "host function {} is a class method with no matching psa method (class-id bridge missed)",
                sla_name
            ),
        };
    }
    if let Some(rest) = sla_name.strip_prefix("__py_mod_") {
        // `__py_mod_<i>`: `i` is the index into `prog.funcs`
        // (emit.rs: merged-module defs, enumerated in order).
        if let Ok(i) = rest.parse::<usize>() {
            if let Some(f) = prog.funcs.get(i) {
                return Bridge::Def {
                    line: f.line,
                    col: f.anchor.col,
                    what: format!("module function {} (index {})", f.name, i),
                };
            }
        }
        return Bridge::NoDef {
            why: format!(
                "host function {} is a module function with no matching psa definition",
                sla_name
            ),
        };
    }
    if let Some(rest) = sla_name.strip_prefix("__py_") {
        if let Some(f) = prog.funcs.iter().find(|f| f.name == rest) {
            return Bridge::Def {
                line: f.line,
                col: f.anchor.col,
                what: format!("function {}", rest),
            };
        }
        return Bridge::NoDef {
            why: format!(
                "host function {} names no psa function (no def {})",
                sla_name, rest
            ),
        };
    }
    if sla_name.starts_with("__psa_") {
        return Bridge::NoDef {
            why: format!(
                "host function {} is a psa runtime helper with no psa definition",
                sla_name
            ),
        };
    }
    Bridge::NoDef {
        why: format!("host function {} has no psa definition", sla_name),
    }
}

/// First line of a top-level statement (0 = synthetic).
/// Every struct variant carries `line`; only `Def(Func)` needs
/// its inner function line.
fn stmt_line(st: &ast::Stmt) -> u32 {
    use ast::Stmt;
    match st {
        Stmt::Def(f) => f.line,
        Stmt::Assign { line, .. }
        | Stmt::AugAssign { line, .. }
        | Stmt::IndexAssign { line, .. }
        | Stmt::AttrAssign { line, .. }
        | Stmt::AttrDel { line, .. }
        | Stmt::ExprStmt { line, .. }
        | Stmt::If { line, .. }
        | Stmt::While { line, .. }
        | Stmt::ForRange { line, .. }
        | Stmt::ForEach { line, .. }
        | Stmt::Unpack { line, .. }
        | Stmt::Return { line, .. }
        | Stmt::Assert { line, .. }
        | Stmt::Del { line, .. }
        | Stmt::MultiAssign { line, .. }
        | Stmt::Break { line, .. }
        | Stmt::Continue { line, .. }
        | Stmt::Pass { line, .. }
        | Stmt::Try { line, .. }
        | Stmt::Raise { line, .. }
        | Stmt::With { line, .. }
        | Stmt::Yield { line, .. }
        | Stmt::Match { line, .. }
        | Stmt::Global { line, .. }
        | Stmt::Nonlocal { line, .. }
        | Stmt::PublishModEnv { line, .. } => *line,
    }
}

/// Verbatim source line (1-based), without its newline.
pub fn source_line_text(src: &str, line: u32) -> Option<String> {
    src.lines().nth((line as usize).checked_sub(1)?).map(|s| s.to_string())
}

// ---------------------------------------------------------------------
// JSON rendering (design §3.4; field order follows the design.
// Optional fields are omitted when absent; old fields keep
// their meaning.)
// ---------------------------------------------------------------------

use crate::diag::json_str;

impl MappingOutcome {
    pub fn render_json(&self) -> String {
        let mut out = String::new();
        out.push('{');
        out.push_str(&format!("\"status\":{}", json_str(self.status.as_str())));
        if let Some(f) = &self.psa_file {
            out.push_str(&format!(",\"psa_file\":{}", json_str(f)));
        }
        if let Some(l) = self.psa_line {
            out.push_str(&format!(",\"psa_line\":{}", l));
        }
        if let Some(c) = self.psa_col {
            out.push_str(&format!(",\"psa_col\":{}", c));
        }
        out.push_str(&format!(",\"map_file\":{}", json_str(&self.map_file)));
        if let (Some(f), Some(l)) = (&self.generated_file, self.generated_line) {
            out.push_str(&format!(
                ",\"generated\":{{\"file\":{},\"line\":{}}}",
                json_str(f),
                l
            ));
        }
        if let Some(idx) = self.segment {
            out.push_str(&format!(",\"segment\":{}", idx));
        }
        if self.approximate {
            out.push_str(",\"approximate\":true");
        }
        if let Some(reason) = self.reason {
            out.push_str(&format!(",\"reason\":{}", json_str(reason)));
        }
        out.push_str(&format!(",\"note\":{}", json_str(&self.note)));
        out.push('}');
        out
    }
}

impl HostDiagnostic {
    pub fn render_json(&self) -> String {
        let mut out = String::new();
        out.push('{');
        out.push_str(&format!("\"code\":{}", json_str(&self.code)));
        out.push_str(&format!(",\"message\":{}", json_str(&self.message)));
        if let Some(f) = &self.file {
            out.push_str(&format!(",\"file\":{}", json_str(f)));
        }
        if let Some(l) = self.line {
            out.push_str(&format!(",\"line\":{}", l));
        }
        if let Some(l) = self.source_line {
            out.push_str(&format!(",\"source_line\":{}", l));
        }
        if let Some(c) = self.column {
            out.push_str(&format!(",\"column\":{}", c));
        }
        if let Some(f) = &self.function {
            out.push_str(&format!(",\"function\":{}", json_str(f)));
        }
        if let Some(h) = &self.hint {
            out.push_str(&format!(",\"hint\":{}", json_str(h)));
        }
        out.push_str(&format!(",\"raw\":{}", json_str(&self.raw)));
        if let Some(m) = &self.mapping {
            out.push_str(&format!(",\"mapping\":{}", m.render_json()));
        }
        out.push('}');
        out
    }
}

impl HostReport {
    pub fn render_json(&self) -> String {
        let mut out = String::new();
        out.push('{');
        out.push_str(&format!("\"tool\":{}", json_str(&self.tool)));
        out.push_str(&format!(",\"stage\":{}", json_str(&self.stage)));
        match self.exit_status {
            Some(code) => out.push_str(&format!(",\"exit_status\":{}", code)),
            None => out.push_str(",\"exit_status\":null"),
        }
        out.push_str(",\"diagnostics\":[");
        for (i, d) in self.diagnostics.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&d.render_json());
        }
        out.push(']');
        if !self.raw.is_empty() {
            out.push_str(&format!(",\"raw\":{}", json_str(&self.raw)));
        }
        out.push('}');
        out
    }
}

// ---------------------------------------------------------------------
// Unit tests (integration): the defensive map-error branches of
// `resolve_sla_line` — a missing map, a stale map and a corrupt
// map must degrade to distinct, honest reason codes instead of
// panicking or inventing a position.
// ---------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::srcmap::{MapError, MapErrorKind};

    fn empty_prog() -> ast::Program {
        ast::Program {
            funcs: Vec::new(),
            classes: Vec::new(),
            top: Vec::new(),
            imports: Vec::new(),
            mod_attrs: Vec::new(),
            const_names: Vec::new(),
            modules: Vec::new(),
            std_imports: Vec::new(),
            sources: Vec::new(),
        }
    }

    fn sla_diag() -> HostDiagnostic {
        HostDiagnostic {
            code: "TypeMismatch".to_string(),
            message: "expected i64, found str".to_string(),
            file: Some("x.sla".to_string()),
            line: Some(2581),
            source_line: None,
            column: Some(3),
            function: None,
            hint: None,
            upstream_file: None,
            upstream_line: None,
            raw: "error[TypeMismatch]: expected i64, found str\n  --> x.sla:2581:3".to_string(),
            order: 0,
            mapping: None,
        }
    }

    fn ctx<'a>(
        prog: &'a ast::Program,
        map_error: Option<&'a MapError>,
    ) -> MapCtx<'a> {
        MapCtx {
            prog,
            psa_src: "def f(x):\n    return x + 1\n",
            psa_file: "x.psa".to_string(),
            sla_src: "",
            sla_file: "x.sla".to_string(),
            map_file: Path::new("x.psa-map.json"),
            map: None,
            map_error,
        }
    }

    #[test]
    fn no_map_file_is_map_unavailable() {
        let prog = empty_prog();
        let c = ctx(&prog, None);
        let out = resolve_mapping(&sla_diag(), &c);
        assert_eq!(out.status, MappingStatus::Unmapped);
        assert_eq!(out.reason, Some(reason::MAP_UNAVAILABLE));
        assert_eq!(out.generated_line, Some(2581));
    }

    #[test]
    fn stale_map_is_stale_map_reason() {
        let prog = empty_prog();
        let e = MapError::new(MapErrorKind::StaleMap, "source 'x.psa' sha256 mismatch".to_string());
        let c = ctx(&prog, Some(&e));
        let out = resolve_mapping(&sla_diag(), &c);
        assert_eq!(out.status, MappingStatus::Unmapped);
        assert_eq!(out.reason, Some(reason::STALE_MAP));
        assert!(out.note.contains("stale"));
    }

    #[test]
    fn corrupt_map_is_bad_map_reason() {
        let prog = empty_prog();
        let e = MapError::new(MapErrorKind::Parse, "\"generated\" is missing a \"sha256\" string".to_string());
        let c = ctx(&prog, Some(&e));
        let out = resolve_mapping(&sla_diag(), &c);
        assert_eq!(out.status, MappingStatus::Unmapped);
        assert_eq!(out.reason, Some(reason::BAD_MAP));
    }
}
