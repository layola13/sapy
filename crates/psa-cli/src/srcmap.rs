//! srcmap.rs — psa source map: the emission-time segment table,
//! its persistence (`<stem>.psa-map.json`, design §3.3 v1) and the
//! bidirectional lookups `psa map-loc` and P3's host diagnostics
//! share.
//!
//! Canonical merge of the four implementation packages (P1–P4),
//! decided at integration (see INTEGRATION-RESULT-R2.md §3 item 1):
//!
//! * **P1 owns the shape**: the nested `Segment` (`gen: GenPos`,
//!   `src: SrcPos`), `SegMeta`, `SegmentWriter` and the
//!   full-coverage assertion. P1's own `SourceAnchor` definition is
//!   dropped: anchors are P0's landed `ast::SourceAnchor`
//!   (`source: u16, line: u32, col: u32, col_byte: Option<u32>`;
//!   `col >= 1` is a real column, `col == 0` means no column,
//!   `line == 0` means synthetic). `SegMeta::stmt` upgrades a user
//!   segment to `Exact` when its anchor carries a column.
//! * **P2 owns persistence**: `SourceEntry` / `GeneratedInfo`,
//!   `MAP_VERSION` / `GENERATOR`, the hand-rolled JSON writer and
//!   reader, `finish()` (sort / remap / index / hash), the
//!   `lookup_psa` / `lookup_gen` semantics, `verify_sources` /
//!   `resolve_source`, and the hand-rolled SHA-256. P2's
//!   `SourceMapBuilder` is retired: the writer feeds segments
//!   straight into the table and `finish()` finalises it. The
//!   wire format is P2's: a std-splice segment writes
//!   `"origin": "std-splice"` plus a separate `"module"` field
//!   (P4's `"std-splice:<module>"` spelling is NOT used); an
//!   unanchored `src.line` writes as JSON `null` and reads back
//!   as `0`.
//! * **P3 owns the reader contract**: `SourceMap::load` (read +
//!   verify, loud `StaleMap` on mismatch) is what the host
//!   diagnostic layer calls; the minimal P3 `srcmap.rs` loader is
//!   retired.
//! * **P4 owns the panic view**: `Segment::panic_code` (in-memory
//!   only — the v1 format has no such field; the code travels in
//!   the `panic_sites` view), `USER_PANIC_EMITTERS` and
//!   `lookup_panic_code`. The view itself is the writer-recorded
//!   one P1 builds (remapped by `finish()`), not rebuilt from
//!   segments at read time.
//!
//! Sources-table layout (integration decision; the design does not
//! pin it): `[user .psa files in prog.sources order (main first),
//! prelude identifier, psa-std/<module>.sla in splice order]`.
//! `SRC_PRELUDE` is therefore NOT a fixed index in multi-file
//! programs: each writer records its own `prelude_source`, and
//! prelude-origin segments are remapped to it when recorded or
//! appended (`SegmentWriter::push_line` / `push_raw` /
//! `append_writer` / `append_chunk`). User anchors map 1:1 onto
//! the table (chunk writers only ever reference user indices and
//! the prelude index, so the remap is unambiguous — it keys on
//! `origin == Prelude`, never on the raw index). Std-splice
//! sources join the table with the sha256 of the spliced bytes
//! (P2 open point (a), option 1); merged module files join it
//! with the sha256 of their bytes (P2 open point (b), closed at
//! integration: main.rs reads every `prog.sources` entry).
//!
//! Column conventions (design §3.1 / P0): generated columns are
//! BYTE columns, 1-based, `col_end` exclusive; source columns are
//! CHARACTER columns, 1-based, `col_end` exclusive; a source
//! column pair of `None` means line-level only.

use crate::ast::SourceAnchor;
use crate::diag::{json_str, Diagnostic};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Source index of the main `.psa` file (by convention).
pub const SRC_MAIN: u16 = 0;
/// Source index of the psa version identifier in a single-file
/// (chunk-local) writer. The assembly writer's prelude index is
/// `user_sources.len()` instead — see the module docs.
pub const SRC_PRELUDE: u16 = 1;
/// The `sources[]` entry for the prelude: the psa version string.
/// Its `sha256` is empty (it names no file; verification skips it).
pub const PRELUDE_SOURCE_ID: &str = "psa 1.0.0 (emitter prelude)";

/// Map format version (design §3.3). Readers reject anything else.
pub const MAP_VERSION: u32 = 1;

/// The `generator` string stamped on every map we write.
pub const GENERATOR: &str = "psa 1.0.0";

/// The `sources` table a fresh (chunk-local) map / writer starts
/// with. Index 0 is the main `.psa` file (path unknown to the
/// emitter; the driver fills it in via the seam's `sources`
/// argument). Index 1 is the prelude identifier. Std-splice
/// sources are appended from index 2 up in single-file programs,
/// after the user files in multi-file ones, by the assembly.
pub fn default_sources() -> Vec<SourceEntry> {
    vec![
        SourceEntry { path: String::new(), sha256: String::new(), content: None },
        SourceEntry {
            path: PRELUDE_SOURCE_ID.to_string(),
            sha256: String::new(),
            content: None,
        },
    ]
}

/// The `SrcPos` an anchor denotes. The column span is the anchor
/// column alone when one is known (`col_end` stays `None`: P0
/// anchors are points, not spans; the expression-span work in
/// `ExprSpan` carries ends, not anchors). `col == 0` means "no
/// column" and maps to `None` on both ends.
fn anchor_src_pos(anchor: &SourceAnchor) -> SrcPos {
    SrcPos {
        source: anchor.source,
        line: anchor.line,
        col_start: if anchor.col >= 1 { Some(anchor.col) } else { None },
        col_end: None,
    }
}

/// One entry of the map's `sources` array: a .psa file the map was
/// built from. `content` is `None` by default (only path + sha256
/// are stored); `--embed-sources` fills it with the source text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceEntry {
    pub path: String,
    pub sha256: String,
    pub content: Option<String>,
}

/// The map's `generated` object: the .sla file the map describes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedInfo {
    pub path: String,
    pub sha256: String,
    pub lines: u32,
}

/// Where a generated span comes from (design §3.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin {
    /// Direct translation of user code.
    User,
    /// psa's own prelude / runtime constants. `src` points at the
    /// prelude identifier, never at a user line.
    Prelude,
    /// A splice of a psa_std module's SLA text. The module name is
    /// carried so a consumer can find the module file.
    StdSplice { module: String },
    /// Synthesized code (dispatcher, class registry seed, closure
    /// conversion output). Anchored to the producing psa definition
    /// when known (precision Function), otherwise unanchored.
    Synthetic,
}

impl Origin {
    /// The JSON spelling (design §3.3 example uses `"user"`).
    /// Std-splice writes as plain `"std-splice"`; the module name
    /// travels in the segment's separate `"module"` field.
    pub fn as_str(&self) -> &'static str {
        match self {
            Origin::User => "user",
            Origin::Prelude => "prelude",
            Origin::StdSplice { .. } => "std-splice",
            Origin::Synthetic => "synthetic",
        }
    }

    pub fn parse(s: &str) -> Option<Origin> {
        match s {
            "user" => Some(Origin::User),
            "prelude" => Some(Origin::Prelude),
            "std-splice" => Some(Origin::StdSplice { module: String::new() }),
            "synthetic" => Some(Origin::Synthetic),
            _ => None,
        }
    }
}

/// What a segment is (design §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegKind {
    Stmt,
    Expr,
    PanicSite,
    CallSite,
    FnHeader,
}

impl SegKind {
    /// The JSON spelling (design §3.3 example uses `"fn-header"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            SegKind::Stmt => "stmt",
            SegKind::Expr => "expr",
            SegKind::PanicSite => "panic-site",
            SegKind::CallSite => "call-site",
            SegKind::FnHeader => "fn-header",
        }
    }

    pub fn parse(s: &str) -> Option<SegKind> {
        match s {
            "stmt" => Some(SegKind::Stmt),
            "expr" => Some(SegKind::Expr),
            "panic-site" => Some(SegKind::PanicSite),
            "call-site" => Some(SegKind::CallSite),
            "fn-header" => Some(SegKind::FnHeader),
            _ => None,
        }
    }
}

/// How precise a segment's psa anchor is (design §3.4's three
/// tiers, plus `None` for unanchorable synthetic/prelude spans).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Exact psa line + column.
    Exact,
    /// Exact psa line, column unknown / line-level only.
    LineOnly,
    /// Anchored to the enclosing definition (function-level).
    Function,
    /// No psa anchor at all (`src.line == 0`).
    None,
}

impl Precision {
    pub fn as_str(&self) -> &'static str {
        match self {
            Precision::Exact => "exact",
            Precision::LineOnly => "line-only",
            Precision::Function => "function",
            Precision::None => "none",
        }
    }

    pub fn parse(s: &str) -> Option<Precision> {
        match s {
            "exact" => Some(Precision::Exact),
            "line-only" => Some(Precision::LineOnly),
            "function" => Some(Precision::Function),
            "none" => Some(Precision::None),
            _ => None,
        }
    }
}

/// Position of a segment inside the generated `.sla` file.
/// Columns are BYTE columns, 1-based, `col_end` exclusive
/// (`col_start = 1`, `col_end = 1 + byte length` for a whole line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenPos {
    pub line: u32,
    pub col_start: u32,
    pub col_end: u32,
    pub line_end: u32,
}

/// Position of a segment inside a source file. Columns are
/// CHARACTER columns (P0 convention); both are `None` when the
/// anchor has no column. `line == 0` means "no source line"
/// (synthetic / prelude); a segment with `line == 0` MUST carry
/// `precision == Precision::None` and MUST NOT be presented as a
/// user location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SrcPos {
    pub source: u16,
    pub line: u32,
    pub col_start: Option<u32>,
    pub col_end: Option<u32>,
}

/// One mapping segment (design §3.1, exact shape, plus P4's
/// `panic_code`: `Some` iff `kind == SegKind::PanicSite` and the
/// segment covers a `panic(N)` literal, holding that `N`. The v1
/// file format has no field for it — it is in-memory only, and
/// the code travels in the `panic_sites` view — so a map read
/// back from disk has `panic_code == None` on every segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub gen: GenPos,
    pub src: SrcPos,
    pub origin: Origin,
    pub kind: SegKind,
    /// Generated symbol this span belongs to, when one exists
    /// (e.g. `__py_add`, `__psa_adapt_0`). The name bridge
    /// (`SourceMap::name_bridge`) resolves it to a psa definition.
    pub name: Option<String>,
    pub precision: Precision,
    pub panic_code: Option<u32>,
}

impl Segment {
    /// A single-line segment spanning the whole line text.
    pub fn whole_line(
        line: u32,
        byte_len: u32,
        src: SrcPos,
        origin: Origin,
        kind: SegKind,
        name: Option<String>,
        precision: Precision,
    ) -> Self {
        Segment {
            gen: GenPos { line, col_start: 1, col_end: 1 + byte_len, line_end: line },
            src,
            origin,
            kind,
            name,
            precision,
            panic_code: None,
        }
    }

    /// A multi-line spanning segment (one per hand-built chunk).
    /// `line_end` is inclusive; `col_end` is the byte length of
    /// the last line + 1 (exclusive).
    pub fn span(
        first_line: u32,
        last_line: u32,
        last_len: u32,
        src: SrcPos,
        origin: Origin,
        kind: SegKind,
        name: Option<String>,
        precision: Precision,
    ) -> Self {
        Segment {
            gen: GenPos { line: first_line, col_start: 1, col_end: 1 + last_len, line_end: last_line },
            src,
            origin,
            kind,
            name,
            precision,
            panic_code: None,
        }
    }

    /// A user statement segment: the backbone granularity
    /// (design §3.1). `src_col_end` is exclusive; pass `None`
    /// for both columns only when the anchor genuinely has none.
    pub fn user_stmt(
        gen_line: u32,
        gen_col_start: u32,
        gen_col_end: u32,
        source: u16,
        src_line: u32,
        src_col_start: Option<u32>,
        src_col_end: Option<u32>,
    ) -> Segment {
        Segment {
            gen: GenPos { line: gen_line, col_start: gen_col_start, col_end: gen_col_end, line_end: gen_line },
            src: SrcPos { source, line: src_line, col_start: src_col_start, col_end: src_col_end },
            origin: Origin::User,
            kind: SegKind::Stmt,
            name: None,
            precision: Precision::Exact,
            panic_code: None,
        }
    }

    /// A user panic-site segment: the generated `panic(N)` literal
    /// emitted for one user statement/expression. The segment's
    /// `gen` range covers the generated line holding the literal;
    /// `src` is that statement's anchor.
    pub fn user_panic_site(
        gen_line: u32,
        gen_col_start: u32,
        gen_col_end: u32,
        source: u16,
        src_line: u32,
        src_col_start: Option<u32>,
        src_col_end: Option<u32>,
        code: u32,
    ) -> Segment {
        Segment {
            gen: GenPos { line: gen_line, col_start: gen_col_start, col_end: gen_col_end, line_end: gen_line },
            src: SrcPos { source, line: src_line, col_start: src_col_start, col_end: src_col_end },
            origin: Origin::User,
            kind: SegKind::PanicSite,
            name: None,
            precision: Precision::Exact,
            panic_code: Some(code),
        }
    }
}

/// The per-write metadata a `SegmentWriter` call carries: the
/// anchor the lines belong to, plus the segment's classification.
/// Built by the small constructors below so call sites cannot
/// assemble an inconsistent combination (e.g. a `User` segment
/// with `src.line == 0`, or `precision = Exact` without a column).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegMeta {
    pub anchor: SourceAnchor,
    pub origin: Origin,
    pub kind: SegKind,
    pub name: Option<String>,
    pub precision: Precision,
}

impl SegMeta {
    /// The segment metadata for one statement's lines: `User` /
    /// `Exact` when the anchor names a psa line with a column,
    /// `User` / `LineOnly` when it names a line without one,
    /// `Synthetic` / `None` when it names no line (`line == 0`).
    pub fn stmt(anchor: SourceAnchor) -> Self {
        if anchor.is_synthetic() {
            SegMeta { anchor, origin: Origin::Synthetic, kind: SegKind::Stmt, name: None, precision: Precision::None }
        } else if anchor.col >= 1 {
            SegMeta { anchor, origin: Origin::User, kind: SegKind::Stmt, name: None, precision: Precision::Exact }
        } else {
            SegMeta { anchor, origin: Origin::User, kind: SegKind::Stmt, name: None, precision: Precision::LineOnly }
        }
    }

    /// The segment metadata for a generated function's frame lines
    /// (header / closer): anchored to the psa definition,
    /// `precision = Function`; `Synthetic` when unanchored.
    pub fn fn_header(anchor: SourceAnchor, name: String) -> Self {
        if anchor.is_synthetic() {
            SegMeta { anchor, origin: Origin::Synthetic, kind: SegKind::FnHeader, name: Some(name), precision: Precision::None }
        } else {
            SegMeta { anchor, origin: Origin::User, kind: SegKind::FnHeader, name: Some(name), precision: Precision::Function }
        }
    }

    /// A synthetic statement anchored to a psa definition (e.g. a
    /// psa-added default return): `Synthetic` / `Function`.
    pub fn synthetic_stmt(anchor: SourceAnchor) -> Self {
        SegMeta {
            anchor,
            origin: Origin::Synthetic,
            kind: SegKind::Stmt,
            name: None,
            precision: if anchor.is_synthetic() { Precision::None } else { Precision::Function },
        }
    }

    /// A synthetic single-function chunk (adapter, generator/async
    /// wrapper): anchored to the psa definition it was generated
    /// from, `precision = Function`, but `Synthetic` — the function
    /// itself is psa-generated, not user-written.
    pub fn synthetic_fn(anchor: SourceAnchor, name: String) -> Self {
        SegMeta {
            anchor,
            origin: Origin::Synthetic,
            kind: SegKind::FnHeader,
            name: Some(name),
            precision: if anchor.is_synthetic() { Precision::None } else { Precision::Function },
        }
    }

    /// A synthetic statement with no anchor at all.
    pub fn synthetic_unanchored(kind: SegKind) -> Self {
        SegMeta { anchor: SourceAnchor::synthetic(), origin: Origin::Synthetic, kind, name: None, precision: Precision::None }
    }

    /// Handwritten runtime text: `Prelude`, anchored to the psa
    /// version identifier, never to a user line. The anchor's
    /// source index is the chunk-local `SRC_PRELUDE`; the writer
    /// remaps prelude-origin segments to its own prelude index
    /// when recording them (see the module docs).
    pub fn prelude(kind: SegKind) -> Self {
        SegMeta {
            anchor: SourceAnchor::new(SRC_PRELUDE, 0, 0, None),
            origin: Origin::Prelude,
            kind,
            name: None,
            precision: Precision::None,
        }
    }

    /// A handwritten runtime function (`__psa_ipow`, ...): Prelude
    /// frame, named.
    pub fn prelude_fn(name: &str) -> Self {
        let mut m = SegMeta::prelude(SegKind::FnHeader);
        m.name = Some(name.to_string());
        m
    }

    /// Rebuild the metadata a recorded segment was written with
    /// (used for the blank separator line after a chunk: it
    /// inherits the chunk's last segment's classification).
    pub fn from_segment(seg: &Segment) -> Self {
        SegMeta {
            anchor: SourceAnchor::new(
                seg.src.source,
                seg.src.line,
                seg.src.col_start.unwrap_or(0),
                None,
            ),
            origin: seg.origin.clone(),
            kind: seg.kind,
            name: seg.name.clone(),
            precision: seg.precision,
        }
    }
}

/// One entry of the `panic_sites` index view (design §3.3):
/// a user-origin `panic(N)` emission point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanicSite {
    pub code: u32,
    pub gen_line: u32,
    /// Index into `SourceMap::segments`.
    pub segment: usize,
}

/// One entry of the name bridge (design §3.1): a generated symbol
/// resolved to the psa definition it was emitted from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameBinding {
    /// Generated symbol, e.g. `__py_add`, `__py_cls_3_init`,
    /// `__psa_adapt_0`, `__py_gen__prod`.
    pub generated: String,
    /// psa definition name, e.g. `add`, `Point.init`,
    /// `gen` (for wrappers the def name, not the wrapper name).
    pub psa_name: String,
    /// The definition's psa line (the `Function`-precision anchor).
    pub def_line: u32,
    /// `sources` index of the file the definition lives in
    /// (`SRC_MAIN` in P1; P0 multi-file fills this in).
    pub source: u16,
}

/// The assembled segment table (design §3.1/§3.3).
///
/// Invariants: `segments` is sorted by `(gen.line,
/// gen.col_start)` after [`SourceMap::finish`]; `by_gen_line`
/// and `by_src_line` index into that order and are rebuilt by
/// `finish` (readers rebuild them from the segments they parse,
/// ignoring the file's stored indexes); `panic_sites` is the
/// writer-recorded view, remapped by `finish` to the sorted order.
/// The name bridge is in-memory only: the v1 format has no field
/// for it, so it never survives a round-trip (P3 reads it from the
/// in-memory table of the same process run).
#[derive(Debug, Clone)]
pub struct SourceMap {
    pub version: u32,
    pub generator: String,
    /// Source files, by index. `[0]` = main `.psa`; `[1..]` =
    /// merged module files in first-load order (multi-file
    /// programs); then the prelude identifier; then
    /// `psa-std/<module>.sla` in splice order.
    pub sources: Vec<SourceEntry>,
    /// The generated file the segments describe.
    pub generated: GeneratedInfo,
    /// Segments in generated-line order (after `finish`).
    pub segments: Vec<Segment>,
    /// Generated line -> segment indices starting on that line.
    pub by_gen_line: HashMap<u32, Vec<usize>>,
    /// (source, psa line) -> segment indices anchored there.
    /// Segments with `src.line == 0` (prelude / unanchored
    /// synthetic) have no psa line to index under and are absent.
    pub by_src_line: HashMap<(u16, u32), Vec<usize>>,
    /// User-origin panic points (code + generated line + segment).
    pub panic_sites: Vec<PanicSite>,
    /// Generated symbol -> psa definition.
    pub name_bridge: Vec<NameBinding>,
}

impl SourceMap {
    pub fn new() -> Self {
        SourceMap {
            version: MAP_VERSION,
            generator: GENERATOR.to_string(),
            sources: default_sources(),
            generated: GeneratedInfo { path: String::new(), sha256: String::new(), lines: 0 },
            segments: Vec::new(),
            by_gen_line: HashMap::new(),
            by_src_line: HashMap::new(),
            panic_sites: Vec::new(),
            name_bridge: Vec::new(),
        }
    }

    /// Build a map from already-finalised segments (chunk maps).
    /// Indexes are built here (chunk segments arrive in gen.line
    /// order); `finish` is still the final step for a map that
    /// will be serialised.
    pub fn from_segments(segments: Vec<Segment>, panic_sites: Vec<PanicSite>) -> Self {
        let (by_gen_line, by_src_line) = build_indexes(&segments);
        SourceMap {
            version: MAP_VERSION,
            generator: GENERATOR.to_string(),
            sources: default_sources(),
            generated: GeneratedInfo { path: String::new(), sha256: String::new(), lines: 0 },
            segments,
            by_gen_line,
            by_src_line,
            panic_sites,
            name_bridge: Vec::new(),
        }
    }

    /// A one-segment map spanning a hand-built chunk's whole text
    /// (dispatch, adapters, machinery, wrappers): the segment
    /// covers every line of `text`.
    pub fn span(text: &str, meta: &SegMeta) -> Self {
        let mut m = SourceMap::new();
        if text.is_empty() {
            return m;
        }
        let lines: Vec<&str> = text.split('\n').collect();
        // `text` is \n-terminated (all chunk builders end with
        // "}\n"), so the split leaves a trailing empty piece that
        // is not a line.
        let nlines = lines.len().saturating_sub(if text.ends_with('\n') { 1 } else { 0 });
        let last_len = if nlines == 0 { 0 } else { lines[nlines - 1].len() as u32 };
        if nlines > 0 {
            m.segments.push(Segment::span(
                1,
                nlines as u32,
                last_len,
                anchor_src_pos(&meta.anchor),
                meta.origin.clone(),
                meta.kind,
                meta.name.clone(),
                meta.precision,
            ));
            let (by_gen_line, by_src_line) = build_indexes(&m.segments);
            m.by_gen_line = by_gen_line;
            m.by_src_line = by_src_line;
        }
        m
    }

    /// Append a source path, deduplicated; returns its index.
    /// The caller supplies the sha256 of the file's bytes (the
    /// assembly passes the sha of each spliced module's text).
    pub fn add_source(&mut self, path: String, sha256: String) -> u16 {
        if let Some(i) = self.sources.iter().position(|s| s.path == path) {
            return i as u16;
        }
        self.sources.push(SourceEntry { path, sha256, content: None });
        (self.sources.len() - 1) as u16
    }

    /// Append one segment; returns its index.
    pub fn push_segment(&mut self, seg: Segment) -> usize {
        self.segments.push(seg);
        self.segments.len() - 1
    }

    /// Append another map's segments and panic sites, rebased so
    /// their generated lines start at `line_offset + 1`
    /// (`line_offset` = lines already in this map). Panic-site
    /// segment indices are rebased too.
    /// P2 API (map merging); the P1 emitter rebases through
    /// `SegmentWriter::append_chunk` instead.
    #[allow(dead_code)]
    pub fn append_rebased(&mut self, other: &SourceMap, line_offset: u32) {
        let base_seg = self.segments.len();
        for seg in &other.segments {
            let mut s = seg.clone();
            s.gen.line += line_offset;
            s.gen.line_end += line_offset;
            self.segments.push(s);
        }
        for p in &other.panic_sites {
            self.panic_sites.push(PanicSite {
                code: p.code,
                gen_line: p.gen_line + line_offset,
                segment: base_seg + p.segment,
            });
        }
    }

    /// Name-bridge lookup for a generated symbol.
    /// P2/P3 API; unused by the P1 emitter itself.
    #[allow(dead_code)]
    pub fn lookup_name(&self, generated: &str) -> Option<&NameBinding> {
        self.name_bridge.iter().find(|b| b.generated == generated)
    }

    /// Full-coverage assertion (design §3.1/§3.7): every generated
    /// line must belong to at least one segment. An unmapped line
    /// is a psa internal error (PSA-E6009 via
    /// `Diagnostic::emit`'s `internal error:` rule), raised loudly
    /// — never silently skipped. Out-of-range segments are the
    /// same error: the table contradicts the text it describes.
    pub fn assert_full_coverage(&self, generated: &str) -> Result<(), Diagnostic> {
        let newlines = generated.matches('\n').count() as u32;
        let total = if generated.is_empty() {
            0
        } else if generated.ends_with('\n') {
            newlines
        } else {
            newlines + 1
        };
        let mut covered = vec![false; total as usize];
        for seg in &self.segments {
            if seg.gen.line == 0 || seg.gen.line_end < seg.gen.line || seg.gen.line_end > total {
                return Err(Diagnostic::emit(
                    0,
                    &format!(
                        "internal error: source-map segment out of range (gen lines {}..={}, generated file has {} lines)",
                        seg.gen.line, seg.gen.line_end, total
                    ),
                ));
            }
            for l in seg.gen.line..=seg.gen.line_end {
                covered[(l - 1) as usize] = true;
            }
        }
        if let Some(i) = covered.iter().position(|c| !*c) {
            return Err(Diagnostic::emit(
                0,
                &format!(
                    "internal error: unmapped generated line {} (no source-map segment covers it; generated file has {} lines, table has {} segments)",
                    i + 1,
                    total,
                    self.segments.len()
                ),
            ));
        }
        Ok(())
    }

    /// Finish the map once emission is complete: hash the
    /// generated text, count its lines, sort segments by
    /// `gen.line` (stable), remap `panic_sites` indices, and
    /// build both lookup indexes from the segments. Call exactly
    /// once, after `assert_full_coverage`, before any lookup or
    /// serialisation.
    pub fn finish(&mut self, generated_src: &str, generated_path: String) {
        let (segments, panic_sites) = sort_segments_and_remap(
            std::mem::take(&mut self.segments),
            std::mem::take(&mut self.panic_sites),
        );
        self.segments = segments;
        self.panic_sites = panic_sites;
        let (by_gen_line, by_src_line) = build_indexes(&self.segments);
        self.by_gen_line = by_gen_line;
        self.by_src_line = by_src_line;
        self.generated = GeneratedInfo {
            path: generated_path,
            sha256: sha256_hex(generated_src.as_bytes()),
            lines: generated_src.lines().count() as u32,
        };
    }
}

/// Stable-sort segments by `gen.line` and remap `panic_sites`
/// segment indices to the new order. The single place that owns
/// the "segments are in gen.line order" invariant, used by both
/// `SourceMap::finish` and `SourceMap::from_json`.
fn sort_segments_and_remap(
    mut segments: Vec<Segment>,
    panic_sites: Vec<PanicSite>,
) -> (Vec<Segment>, Vec<PanicSite>) {
    let mut order: Vec<usize> = (0..segments.len()).collect();
    order.sort_by_key(|&i| segments[i].gen.line);
    let mut remap = vec![0usize; segments.len()];
    let mut sorted: Vec<Segment> = Vec::with_capacity(segments.len());
    for (new_i, &old_i) in order.iter().enumerate() {
        remap[old_i] = new_i;
        sorted.push(segments[old_i].clone());
    }
    segments = sorted;
    let panic_sites = panic_sites
        .into_iter()
        .map(|p| PanicSite {
            code: p.code,
            gen_line: p.gen_line,
            segment: remap[p.segment],
        })
        .collect();
    (segments, panic_sites)
}

/// Build both lookup indexes from the segments, mechanically.
/// The single function that generates `by_gen_line` /
/// `by_src_line`; writers and readers both go through it, so the
/// indexes can never drift from the segment table. Segments with
/// `src.line == 0` (prelude / unanchored synthetic) have no psa
/// line to index under and are absent from `by_src_line`.
/// `by_gen_line` keys on the segment's FIRST generated line only
/// (a multi-line segment is found through its start; the lookups
/// below scan ranges, they do not rely on per-line keys).
fn build_indexes(
    segments: &[Segment],
) -> (
    HashMap<u32, Vec<usize>>,
    HashMap<(u16, u32), Vec<usize>>,
) {
    let mut by_gen: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut by_src: HashMap<(u16, u32), Vec<usize>> = HashMap::new();
    for (i, s) in segments.iter().enumerate() {
        by_gen.entry(s.gen.line).or_default().push(i);
        if s.src.line != 0 {
            by_src.entry((s.src.source, s.src.line)).or_default().push(i);
        }
    }
    (by_gen, by_src)
}

/// A hand-built emitted chunk: its text plus the map of that text
/// (chunk-local generated lines starting at 1). The assembly
/// rebases the map when splicing the chunk into the final file.
/// Every `emitted.push` site pushes one of these, so text and map
/// can never drift apart.
#[derive(Debug, Clone)]
pub struct EmittedChunk {
    pub text: String,
    pub map: SourceMap,
}

impl EmittedChunk {
    /// A chunk built as a plain string (dispatch, adapters,
    /// machinery, wrappers): one spanning segment over its text.
    pub fn span(text: String, meta: &SegMeta) -> Self {
        let map = SourceMap::span(&text, meta);
        EmittedChunk { text, map }
    }
}

/// Find the first `panic(<digits>)` code in a generated line.
/// Only the first code on the line is returned: P1 records one
/// panic per segment, and a generated line holds at most one
/// panic site in practice (guard lines are single-panic).
fn find_panic_code(text: &str) -> Option<u32> {
    let mut rest = text;
    while let Some(i) = rest.find("panic(") {
        let after = &rest[i + "panic(".len()..];
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            if let Ok(code) = digits.parse::<u32>() {
                return Some(code);
            }
        }
        rest = after;
    }
    None
}

/// The recording writer every emission buffer goes through.
///
/// Invariants (the byte-identical red line depends on them):
///
/// * Every method appends exactly the text it is given (plus the
///   single `'\n'` `push_line` adds, mirroring the old
///   `Ctx::line`). Recording only reads the text; it never edits,
///   reorders or drops a byte.
/// * `text` is always empty or `\n`-terminated, and `next_line`
///   is the 1-based number of the next line to be written, so a
///   segment's `gen.line` is exact at record time. All push sites
///   satisfy this (every emitted line ends with `'\n'`).
/// * `push_line` / `push_raw` record segments for the lines they
///   append, in order; `append_writer` / `append_chunk` rebase
///   the appended writer's segments by the lines already here.
///   A segment's line therefore stays correct across the
///   scratch-buffer swap dance in `emit_expr`.
/// * Segments reference only user sources and the prelude index
///   while in a writer (std-splice sources are assigned by the
///   assembly, which owns the writer being appended into).
///   Prelude-origin segments are remapped to this writer's
///   `prelude_source` on record / append (chunk-local writers
///   have it at 1; the assembly writer has it after the user
///   files), so a chunk's prelude segments land on the final
///   table's prelude entry instead of colliding with the first
///   merged module file.
#[derive(Debug, Clone)]
pub struct SegmentWriter {
    pub text: String,
    /// The `sources` table these segments index into (seeded with
    /// `default_sources()` by `new`, or with the driver's user
    /// files plus the prelude entry by `with_sources`; the
    /// assembly appends std-splice sources from the prelude index
    /// up). Chunk maps built from a writer never reference beyond
    /// the prelude index, so their tables stay consistent with the
    /// assembly's (see `append_chunk`).
    pub sources: Vec<SourceEntry>,
    pub segments: Vec<Segment>,
    pub panic_sites: Vec<PanicSite>,
    /// Index of the prelude entry in `sources` (1 for a
    /// chunk-local writer, `user_sources.len()` for the assembly
    /// writer). Prelude-origin segments are recorded against it.
    pub prelude_source: u16,
    next_line: u32,
}

impl Default for SegmentWriter {
    /// NOTE: `Default` must start at line 1, not 0 — the
    /// scratch-buffer dance does `std::mem::take(&mut ctx.out)`,
    /// which installs a fresh default writer mid-function; its
    /// first recorded line is line 1 of the scratch buffer.
    fn default() -> Self {
        SegmentWriter {
            text: String::new(),
            sources: default_sources(),
            segments: Vec::new(),
            panic_sites: Vec::new(),
            prelude_source: SRC_PRELUDE,
            next_line: 1,
        }
    }
}

impl SegmentWriter {
    pub fn new() -> Self {
        SegmentWriter::default()
    }

    /// A writer seeded with the driver's user sources (main file
    /// first, merged modules after): the prelude entry is appended
    /// after them, so `prelude_source == user_sources.len()`.
    /// Used by the assembly writer in `emit_program_with_map`;
    /// every other writer is chunk-local (`new()`).
    pub fn with_sources(mut user_sources: Vec<SourceEntry>) -> Self {
        let prelude_source = user_sources.len() as u16;
        user_sources.push(SourceEntry {
            path: PRELUDE_SOURCE_ID.to_string(),
            sha256: String::new(),
            content: None,
        });
        SegmentWriter {
            text: String::new(),
            sources: user_sources,
            segments: Vec::new(),
            panic_sites: Vec::new(),
            prelude_source,
            next_line: 1,
        }
    }

    /// Lines written so far (= `text`'s newline count).
    pub fn line_count(&self) -> u32 {
        self.next_line - 1
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Append a source path, deduplicated; returns its index
    /// (assembly use: std-splice sources, with the sha256 of the
    /// spliced text).
    pub fn add_source(&mut self, path: String, sha256: String) -> u16 {
        if let Some(i) = self.sources.iter().position(|s| s.path == path) {
            return i as u16;
        }
        self.sources.push(SourceEntry { path, sha256, content: None });
        (self.sources.len() - 1) as u16
    }

    /// The `SrcPos` a recorded segment gets for `meta`: the
    /// anchor's position, with prelude-origin segments pointed
    /// at this writer's prelude entry.
    fn record_src_pos(&self, meta: &SegMeta) -> SrcPos {
        let mut src = anchor_src_pos(&meta.anchor);
        if meta.origin == Origin::Prelude {
            src.source = self.prelude_source;
        }
        src
    }

    /// Append one logical line (no `'\n'` in `text`) and record
    /// its segment. `col_end` is exclusive: `1 + byte length`.
    /// User-origin lines are scanned for `panic(N)`: on a hit the
    /// segment kind becomes `PanicSite` and the panic is indexed
    /// (see the module docs for why only `User` lines are scanned
    /// here — multi-line raw spans are never `PanicSite`).
    pub fn push_line(&mut self, text: &str, meta: &SegMeta) {
        debug_assert!(!text.contains('\n'), "push_line takes a single line without newline");
        let line = self.next_line;
        let mut kind = meta.kind;
        let mut panic_code = None;
        if meta.origin == Origin::User {
            if let Some(code) = find_panic_code(text) {
                kind = SegKind::PanicSite;
                panic_code = Some(code);
                self.panic_sites.push(PanicSite { code, gen_line: line, segment: self.segments.len() });
            }
        }
        let mut seg = Segment::whole_line(
            line,
            text.len() as u32,
            self.record_src_pos(meta),
            meta.origin.clone(),
            kind,
            meta.name.clone(),
            meta.precision,
        );
        seg.panic_code = panic_code;
        self.segments.push(seg);
        self.text.push_str(text);
        self.text.push('\n');
        self.next_line += 1;
    }

    /// Append `text` as-is (must be `\n`-terminated or empty) and
    /// record ONE spanning segment over all its lines. Used for
    /// handwritten runtime constants, the import block, and
    /// prologue lines (each of which is a single line in practice).
    /// A single-line user-origin `text` is panic-scanned like
    /// `push_line`; multi-line spans are never `PanicSite`
    /// (the panic cannot be attributed to one line of the span).
    pub fn push_raw(&mut self, text: &str, meta: &SegMeta) {
        if text.is_empty() {
            return;
        }
        debug_assert!(text.ends_with('\n'), "push_raw takes \\n-terminated text");
        let nlines = text.matches('\n').count() as u32;
        let first = self.next_line;
        let last = first + nlines - 1;
        let last_len = text.rsplit('\n').nth(1).map(|l| l.len() as u32).unwrap_or(0);
        let mut kind = meta.kind;
        let mut panic_code = None;
        if nlines == 1 && meta.origin == Origin::User {
            if let Some(code) = find_panic_code(text) {
                kind = SegKind::PanicSite;
                panic_code = Some(code);
                self.panic_sites.push(PanicSite { code, gen_line: first, segment: self.segments.len() });
            }
        }
        let mut seg = Segment::span(
            first,
            last,
            last_len,
            self.record_src_pos(meta),
            meta.origin.clone(),
            kind,
            meta.name.clone(),
            meta.precision,
        );
        seg.panic_code = panic_code;
        self.segments.push(seg);
        self.text.push_str(text);
        self.next_line += nlines;
    }

    /// Append a psa_std module splice: one segment per generated
    /// line, `src.line` advancing with the module file's own line
    /// numbers (`module` is the short name, e.g. `"json"`; the
    /// `source` index is the assembly's `add_source` result for
    /// `psa-std/<module>.sla`). Then appends the assembly's
    /// trailing `"\n"` separator: when `text` ends with `'\n'` the
    /// separator is a blank line, recorded as its own segment
    /// (source = the module's last line); when it does not, the
    /// separator merely terminates the last line and records
    /// nothing. All spliced modules end with `'\n'` (verified
    /// 2026-11-26 over `psa-std/*.sla`); the other branch is a
    /// guard, not a live path.
    pub fn push_std_splice(&mut self, text: &str, module: &str, source: u16) {
        let mut src_line: u32 = 0;
        for part in text.split_inclusive('\n') {
            src_line += 1;
            let line_text = part.strip_suffix('\n').unwrap_or(part);
            let meta = SegMeta {
                anchor: SourceAnchor::new(source, src_line, 0, None),
                origin: Origin::StdSplice { module: module.to_string() },
                kind: SegKind::Stmt,
                name: None,
                precision: Precision::LineOnly,
            };
            // `part` is \n-terminated except possibly the last
            // one; record the line and append it verbatim.
            let line = self.next_line;
            self.segments.push(Segment::whole_line(
                line,
                line_text.len() as u32,
                anchor_src_pos(&meta.anchor),
                meta.origin.clone(),
                meta.kind,
                meta.name.clone(),
                meta.precision,
            ));
            self.text.push_str(part);
            self.next_line += 1;
            if !part.ends_with('\n') {
                // Unterminated last line: the separator below
                // terminates it; the writer invariant is restored
                // there, and no extra segment is recorded.
                self.text.push('\n');
                return;
            }
        }
        // Separator blank line after the splice (the old
        // `out.push('\n')`): its own segment, source = the
        // module's last line — it is a separator, not code.
        let meta = SegMeta {
            anchor: SourceAnchor::new(source, src_line.max(1), 0, None),
            origin: Origin::StdSplice { module: module.to_string() },
            kind: SegKind::Stmt,
            name: None,
            precision: Precision::LineOnly,
        };
        self.push_line("", &meta);
    }

    /// Append another writer's text and segments, rebasing the
    /// segments (and panic sites) to their final lines. This is
    /// the splice point for scratch buffers (`ctx.out` dance),
    /// function prologues/bodies, and `main_txt`. Prelude-origin
    /// segments are remapped to this writer's prelude index.
    pub fn append_writer(&mut self, mut other: SegmentWriter) {
        let offset = self.next_line - 1;
        let base_seg = self.segments.len();
        for seg in other.segments.drain(..) {
            let mut s = seg;
            s.gen.line += offset;
            s.gen.line_end += offset;
            if s.origin == Origin::Prelude {
                s.src.source = self.prelude_source;
            }
            self.segments.push(s);
        }
        for p in other.panic_sites.drain(..) {
            self.panic_sites.push(PanicSite {
                code: p.code,
                gen_line: p.gen_line + offset,
                segment: base_seg + p.segment,
            });
        }
        self.text.push_str(&other.text);
        self.next_line += other.line_count();
    }

    /// Append an emitted chunk (text + its map), then the blank
    /// separator line the old assembly wrote after every chunk
    /// (`out.push('\n')`). The blank line inherits the chunk's
    /// last segment's classification, so it is covered without
    /// inventing a source position for it. Prelude-origin
    /// segments are remapped to this writer's prelude index.
    pub fn append_chunk(&mut self, chunk: &EmittedChunk) {
        debug_assert!(
            chunk
                .map
                .segments
                .iter()
                .all(|s| !matches!(s.origin, Origin::StdSplice { .. })),
            "chunk maps may only reference user sources / the prelude (std sources are assembly-assigned)"
        );
        let offset = self.next_line - 1;
        let base_seg = self.segments.len();
        for seg in &chunk.map.segments {
            let mut s = seg.clone();
            s.gen.line += offset;
            s.gen.line_end += offset;
            if s.origin == Origin::Prelude {
                s.src.source = self.prelude_source;
            }
            self.segments.push(s);
        }
        for p in &chunk.map.panic_sites {
            self.panic_sites.push(PanicSite {
                code: p.code,
                gen_line: p.gen_line + offset,
                segment: base_seg + p.segment,
            });
        }
        self.text.push_str(&chunk.text);
        self.next_line += chunk.text.matches('\n').count() as u32;
        // Separator blank line after the chunk.
        let meta = chunk
            .map
            .segments
            .last()
            .map(SegMeta::from_segment)
            .unwrap_or_else(|| SegMeta::synthetic_unanchored(SegKind::Stmt));
        self.push_line("", &meta);
    }

    /// Split into text, segments and panic sites. Sources for the
    /// segments live in the map built around them (see
    /// `SourceMap::from_segments` / `SourceMap::append_rebased`).
    pub fn into_parts(self) -> (String, Vec<Segment>, Vec<PanicSite>) {
        (self.text, self.segments, self.panic_sites)
    }

    /// Consume the writer into its text plus the final `SourceMap`
    /// (assembly use): the writer's own sources table travels with
    /// its segments, so std-splice indices stay valid.
    pub fn into_text_and_map(self, name_bridge: Vec<NameBinding>) -> (String, SourceMap) {
        let map = SourceMap {
            version: MAP_VERSION,
            generator: GENERATOR.to_string(),
            sources: self.sources,
            generated: GeneratedInfo { path: String::new(), sha256: String::new(), lines: 0 },
            segments: self.segments,
            by_gen_line: HashMap::new(),
            by_src_line: HashMap::new(),
            panic_sites: self.panic_sites,
            name_bridge,
        };
        (self.text, map)
    }
}

// ---------------------------------------------------------------------
// SHA-256 (FIPS 180-4), std-only: the psa tree has no hashing
// dependency and this package adds none. Test vectors are in the
// unit tests below (empty string, "abc").
// ---------------------------------------------------------------------

/// SHA-256 of `data`, as 64 lowercase hex characters.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bitlen = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = String::with_capacity(64);
    for v in h {
        out.push_str(&format!("{:08x}", v));
    }
    out
}

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

/// Why a map could not be read, parsed, or trusted. `StaleMap` is
/// the loud refusal to use a map whose sources no longer match
/// (design §3.3/§3.7): never answer from a stale map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapErrorKind {
    Io,
    Parse,
    UnsupportedVersion,
    StaleMap,
    NotInMap,
}

#[derive(Debug, Clone)]
pub struct MapError {
    pub kind: MapErrorKind,
    pub detail: String,
}

impl MapError {
    pub fn new(kind: MapErrorKind, detail: String) -> MapError {
        MapError { kind, detail }
    }

    pub fn kind_str(&self) -> &'static str {
        match self.kind {
            MapErrorKind::Io => "io-error",
            MapErrorKind::Parse => "bad-map",
            MapErrorKind::UnsupportedVersion => "unsupported-version",
            MapErrorKind::StaleMap => "stale-map",
            MapErrorKind::NotInMap => "not-in-map",
        }
    }
}

impl std::fmt::Display for MapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind_str(), self.detail)
    }
}

impl std::error::Error for MapError {}

// ---------------------------------------------------------------------
// Serialization (design §3.3 version-1 format). Hand-rolled JSON in
// the diag.rs style (same `json_str` escaper, reused here).
// Field order follows the design's example exactly. An unanchored
// `src.line` (0) writes as JSON `null`, matching P2's corpus
// fixtures; `panic_code` is NOT serialised (in-memory only).
// ---------------------------------------------------------------------

/// Render one segment exactly as it appears in `segments[]`.
pub fn segment_to_json(s: &Segment) -> String {
    let mut out = String::new();
    out.push('{');
    out.push_str(&format!(
        "\"gen\":{{\"line\":{},\"col_start\":{},\"col_end\":{},\"line_end\":{}}}",
        s.gen.line, s.gen.col_start, s.gen.col_end, s.gen.line_end
    ));
    let src_line = if s.src.line == 0 { "null".to_string() } else { s.src.line.to_string() };
    let src_col_start = match s.src.col_start {
        Some(c) => c.to_string(),
        None => "null".to_string(),
    };
    let src_col_end = match s.src.col_end {
        Some(c) => c.to_string(),
        None => "null".to_string(),
    };
    out.push_str(&format!(
        ",\"src\":{{\"source\":{},\"line\":{},\"col_start\":{},\"col_end\":{}}}",
        s.src.source, src_line, src_col_start, src_col_end
    ));
    out.push_str(&format!(",\"origin\":{}", json_str(s.origin.as_str())));
    if let Origin::StdSplice { module } = &s.origin {
        out.push_str(&format!(",\"module\":{}", json_str(module)));
    }
    out.push_str(&format!(",\"kind\":{}", json_str(s.kind.as_str())));
    match &s.name {
        Some(n) => out.push_str(&format!(",\"name\":{}", json_str(n))),
        None => out.push_str(",\"name\":null"),
    }
    out.push_str(&format!(
        ",\"precision\":{}}}",
        json_str(s.precision.as_str())
    ));
    out
}

/// Render the whole map. `by_gen_line` / `by_src_line` are the
/// stored indexes (always built by `build_indexes`); keys are
/// emitted in ascending numeric order so output is deterministic.
pub fn map_to_json(map: &SourceMap) -> String {
    let mut out = String::new();
    out.push('{');
    out.push_str(&format!("\"version\":{}", map.version));
    out.push_str(&format!(",\"generator\":{}", json_str(&map.generator)));
    out.push_str(",\"sources\":[");
    for (i, s) in map.sources.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"path\":{},\"sha256\":{}",
            json_str(&s.path),
            json_str(&s.sha256)
        ));
        match &s.content {
            Some(c) => out.push_str(&format!(",\"content\":{}}}", json_str(c))),
            None => out.push_str(",\"content\":null}"),
        }
    }
    out.push(']');
    out.push_str(&format!(
        ",\"generated\":{{\"path\":{},\"sha256\":{},\"lines\":{}}}",
        json_str(&map.generated.path),
        json_str(&map.generated.sha256),
        map.generated.lines
    ));
    out.push_str(",\"segments\":[");
    for (i, s) in map.segments.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&segment_to_json(s));
    }
    out.push(']');
    // by_gen_line: keys ascending by generated line.
    let mut gen_keys: Vec<u32> = map.by_gen_line.keys().cloned().collect();
    gen_keys.sort_unstable();
    out.push_str(",\"by_gen_line\":{");
    for (i, k) in gen_keys.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let idxs = &map.by_gen_line[k];
        let list: Vec<String> = idxs.iter().map(|x| x.to_string()).collect();
        out.push_str(&format!("\"{}\":[{}]", k, list.join(",")));
    }
    out.push('}');
    // by_src_line: keys ascending by (source, line), spelled "s:l".
    let mut src_keys: Vec<(u16, u32)> = map.by_src_line.keys().cloned().collect();
    src_keys.sort_unstable();
    out.push_str(",\"by_src_line\":{");
    for (i, k) in src_keys.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let idxs = &map.by_src_line[k];
        let list: Vec<String> = idxs.iter().map(|x| x.to_string()).collect();
        out.push_str(&format!("\"{}:{}\":[{}]", k.0, k.1, list.join(",")));
    }
    out.push('}');
    out.push_str(",\"panic_sites\":[");
    for (i, p) in map.panic_sites.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"code\":{},\"gen_line\":{},\"segment\":{}}}",
            p.code, p.gen_line, p.segment
        ));
    }
    out.push_str("]}");
    out
}

impl SourceMap {
    /// Serialize this map (design §3.3 v1).
    pub fn to_json(&self) -> String {
        map_to_json(self)
    }

    /// Write this map to `path` (with a trailing newline).
    pub fn write_to(&self, path: &Path) -> Result<(), MapError> {
        std::fs::write(path, format!("{}\n", self.to_json())).map_err(|e| {
            MapError::new(
                MapErrorKind::Io,
                format!("cannot write {}: {}", path.display(), e),
            )
        })
    }

    /// Read and parse a map file. The file's stored `by_*`
    /// indexes are ignored and rebuilt from its segments (see the
    /// `SourceMap` docs); a version other than 1 is refused.
    /// Sources are NOT verified here — use `load` for the
    /// verifying read.
    pub fn read_from(path: &Path) -> Result<SourceMap, MapError> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            MapError::new(
                MapErrorKind::Io,
                format!("cannot read {}: {}", path.display(), e),
            )
        })?;
        SourceMap::from_json(&text)
    }

    /// Read, parse and verify a map file: every source that can
    /// be checked must still match its recorded sha256 (see
    /// `verify_sources`). The verifying read P3's host layer and
    /// `map-loc` use.
    pub fn load(path: &Path) -> Result<SourceMap, MapError> {
        let map = SourceMap::read_from(path)?;
        let map_dir = path.parent().unwrap_or(Path::new("."));
        map.verify_sources(map_dir)?;
        Ok(map)
    }
}

/// Write `map` to `path` (convenience wrapper for main.rs).
pub fn write_map_file(path: &Path, map: &SourceMap) -> Result<(), MapError> {
    map.write_to(path)
}

// ---------------------------------------------------------------------
// Deserialization: a small JSON parser (std-only; the map format
// needs the full JSON value grammar, so a hand parser beats ad-hoc
// string matching and stays honest about malformed input).
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_u32(&self) -> Option<u32> {
        match self {
            Json::Int(n) if *n >= 0 && *n <= u32::MAX as i64 => Some(*n as u32),
            _ => None,
        }
    }

    fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Int(n) if *n >= 0 => Some(*n as usize),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
}

struct JsonParser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(text: &'a str) -> JsonParser<'a> {
        JsonParser {
            src: text.as_bytes(),
            pos: 0,
        }
    }

    fn err(&self, msg: &str) -> MapError {
        MapError::new(
            MapErrorKind::Parse,
            format!("map JSON parse error at byte {}: {}", self.pos, msg),
        )
    }

    fn skip_ws(&mut self) {
        while self.pos < self.src.len() && matches!(self.src[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).cloned()
    }

    fn parse_value(&mut self) -> Result<Json, MapError> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Json::Str(self.parse_string()?)),
            Some(b't') => self.parse_lit("true", Json::Bool(true)),
            Some(b'f') => self.parse_lit("false", Json::Bool(false)),
            Some(b'n') => self.parse_lit("null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(self.err(&format!("unexpected character '{}'", c as char))),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn parse_lit(&mut self, lit: &str, value: Json) -> Result<Json, MapError> {
        if self.src.len() - self.pos < lit.len() || &self.src[self.pos..self.pos + lit.len()] != lit.as_bytes() {
            return Err(self.err(&format!("expected `{}`", lit)));
        }
        self.pos += lit.len();
        Ok(value)
    }

    fn parse_object(&mut self) -> Result<Json, MapError> {
        self.pos += 1; // '{'
        let mut pairs: Vec<(String, Json)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected an object key string"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(self.err("expected ':' after object key"));
            }
            self.pos += 1;
            let value = self.parse_value()?;
            pairs.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(pairs));
                }
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
    }

    fn parse_array(&mut self) -> Result<Json, MapError> {
        self.pos += 1; // '['
        let mut items: Vec<Json> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            let value = self.parse_value()?;
            items.push(value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, MapError> {
        // Caller guarantees src[pos] == '"'.
        self.pos += 1;
        let mut out = String::new();
        loop {
            let c = match self.peek() {
                Some(c) => c,
                None => return Err(self.err("unterminated string")),
            };
            match c {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let e = match self.peek() {
                        Some(e) => e,
                        None => return Err(self.err("unterminated escape")),
                    };
                    self.pos += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let cp = self.parse_hex4()?;
                            if (0xD800..0xDC00).contains(&cp) {
                                // High surrogate: expect \uDC00-\uDFFF next.
                                if self.peek() == Some(b'\\')
                                    && self.src.get(self.pos + 1) == Some(&b'u')
                                {
                                    self.pos += 2;
                                    let lo = self.parse_hex4()?;
                                    if (0xDC00..0xE000).contains(&lo) {
                                        let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                        out.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
                                    } else {
                                        return Err(self.err("bad low surrogate in \\u escape"));
                                    }
                                } else {
                                    return Err(self.err("unpaired high surrogate in \\u escape"));
                                }
                            } else if (0xDC00..0xE000).contains(&cp) {
                                return Err(self.err("unpaired low surrogate in \\u escape"));
                            } else {
                                out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                            }
                        }
                        _ => return Err(self.err("bad escape in string")),
                    }
                }
                _ => {
                    // Copy one UTF-8 character verbatim.
                    let rest = &self.src[self.pos..];
                    let s = std::str::from_utf8(rest)
                        .map_err(|_| self.err("invalid UTF-8 in string"))?;
                    let ch = s.chars().next().unwrap();
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, MapError> {
        if self.pos + 4 > self.src.len() {
            return Err(self.err("truncated \\u escape"));
        }
        let mut v: u32 = 0;
        for _ in 0..4 {
            let c = self.src[self.pos];
            let d = match c {
                b'0'..=b'9' => (c - b'0') as u32,
                b'a'..=b'f' => (c - b'a' + 10) as u32,
                b'A'..=b'F' => (c - b'A' + 10) as u32,
                _ => return Err(self.err("bad hex digit in \\u escape")),
            };
            v = (v << 4) | d;
            self.pos += 1;
        }
        Ok(v)
    }

    fn parse_number(&mut self) -> Result<Json, MapError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let mut is_float = false;
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'9' => self.pos += 1,
                b'.' | b'e' | b'E' | b'+' | b'-' => {
                    is_float = true;
                    self.pos += 1;
                }
                _ => break,
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| self.err("invalid number"))?;
        if is_float {
            text.parse::<f64>()
                .map(Json::Float)
                .map_err(|_| self.err("invalid number"))
        } else {
            text.parse::<i64>()
                .map(Json::Int)
                .map_err(|_| self.err("integer out of range"))
        }
    }

    fn finish(&mut self) -> Result<(), MapError> {
        self.skip_ws();
        if self.pos != self.src.len() {
            return Err(self.err("trailing characters after JSON value"));
        }
        Ok(())
    }
}

/// Parse a map JSON document (the inverse of `map_to_json`).
/// The stored `by_*` indexes are ignored and rebuilt from the
/// parsed segments; `panic_sites` segment indices refer to the
/// file's segment order and are remapped if sorting reorders them.
/// `panic_code` is not in the format: read-back segments have it
/// as `None`.
impl SourceMap {
    pub fn from_json(text: &str) -> Result<SourceMap, MapError> {
        let mut p = JsonParser::new(text);
        let root = p.parse_value()?;
        p.finish()?;
        let version = root
            .get("version")
            .and_then(|v| v.as_u32())
            .ok_or_else(|| {
                MapError::new(
                    MapErrorKind::Parse,
                    "map is missing a numeric \"version\" field".to_string(),
                )
            })?;
        if version != MAP_VERSION {
            return Err(MapError::new(
                MapErrorKind::UnsupportedVersion,
                format!(
                    "unsupported map version {} (this psa reads version {})",
                    version, MAP_VERSION
                ),
            ));
        }
        let generator = root
            .get("generator")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let sources = root
            .get("sources")
            .and_then(|v| v.as_arr())
            .ok_or_else(|| {
                MapError::new(
                    MapErrorKind::Parse,
                    "map is missing a \"sources\" array".to_string(),
                )
            })?
            .iter()
            .map(|s| {
                let path = s
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        MapError::new(
                            MapErrorKind::Parse,
                            "a sources[] entry is missing a \"path\" string".to_string(),
                        )
                    })?
                    .to_string();
                let sha256 = s
                    .get("sha256")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        MapError::new(
                            MapErrorKind::Parse,
                            "a sources[] entry is missing a \"sha256\" string".to_string(),
                        )
                    })?
                    .to_string();
                let content = match s.get("content") {
                    Some(Json::Str(c)) => Some(c.clone()),
                    Some(Json::Null) | None => None,
                    _ => {
                        return Err(MapError::new(
                            MapErrorKind::Parse,
                            "a sources[] entry has a non-string \"content\"".to_string(),
                        ))
                    }
                };
                Ok(SourceEntry {
                    path,
                    sha256,
                    content,
                })
            })
            .collect::<Result<Vec<_>, MapError>>()?;
        let generated = root.get("generated").ok_or_else(|| {
            MapError::new(
                MapErrorKind::Parse,
                "map is missing a \"generated\" object".to_string(),
            )
        })?;
        let generated = GeneratedInfo {
            path: generated
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    MapError::new(
                        MapErrorKind::Parse,
                        "\"generated\" is missing a \"path\" string".to_string(),
                    )
                })?
                .to_string(),
            sha256: generated
                .get("sha256")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    MapError::new(
                        MapErrorKind::Parse,
                        "\"generated\" is missing a \"sha256\" string".to_string(),
                    )
                })?
                .to_string(),
            lines: generated
                .get("lines")
                .and_then(|v| v.as_u32())
                .ok_or_else(|| {
                    MapError::new(
                        MapErrorKind::Parse,
                        "\"generated\" is missing a numeric \"lines\" field".to_string(),
                    )
                })?,
        };
        let segments = root
            .get("segments")
            .and_then(|v| v.as_arr())
            .ok_or_else(|| {
                MapError::new(
                    MapErrorKind::Parse,
                    "map is missing a \"segments\" array".to_string(),
                )
            })?
            .iter()
            .map(segment_from_json)
            .collect::<Result<Vec<_>, MapError>>()?;
        let panic_sites = match root.get("panic_sites") {
            Some(Json::Arr(a)) => a
                .iter()
                .map(|p| {
                    let code = p
                        .get("code")
                        .and_then(|v| v.as_u32())
                        .ok_or_else(|| {
                            MapError::new(
                                MapErrorKind::Parse,
                                "a panic_sites[] entry is missing a numeric \"code\"".to_string(),
                            )
                        })?;
                    let gen_line = p
                        .get("gen_line")
                        .and_then(|v| v.as_u32())
                        .ok_or_else(|| {
                            MapError::new(
                                MapErrorKind::Parse,
                                "a panic_sites[] entry is missing a numeric \"gen_line\""
                                    .to_string(),
                            )
                        })?;
                    let segment = p
                        .get("segment")
                        .and_then(|v| v.as_usize())
                        .ok_or_else(|| {
                            MapError::new(
                                MapErrorKind::Parse,
                                "a panic_sites[] entry is missing a numeric \"segment\"".to_string(),
                            )
                        })?;
                    if segment >= segments.len() {
                        return Err(MapError::new(
                            MapErrorKind::Parse,
                            format!(
                                "panic_sites[] entry refers to segment {} but the map has only {} segments",
                                segment,
                                segments.len()
                            ),
                        ));
                    }
                    Ok(PanicSite {
                        code,
                        gen_line,
                        segment,
                    })
                })
                .collect::<Result<Vec<_>, MapError>>()?,
            Some(_) => {
                return Err(MapError::new(
                    MapErrorKind::Parse,
                    "\"panic_sites\" is not an array".to_string(),
                ))
            }
            None => Vec::new(),
        };
        let (segments, panic_sites) = sort_segments_and_remap(segments, panic_sites);
        let (by_gen_line, by_src_line) = build_indexes(&segments);
        Ok(SourceMap {
            version,
            generator,
            sources,
            generated,
            segments,
            by_gen_line,
            by_src_line,
            panic_sites,
            name_bridge: Vec::new(),
        })
    }
}

fn opt_u32(v: Option<&Json>) -> Result<Option<u32>, MapError> {
    match v {
        None | Some(Json::Null) => Ok(None),
        Some(j) => j.as_u32().ok_or_else(|| {
            MapError::new(
                MapErrorKind::Parse,
                "expected a number or null for a column/line field".to_string(),
            )
        }).map(Some),
    }
}

fn segment_from_json(v: &Json) -> Result<Segment, MapError> {
    let missing = |field: &str| {
        MapError::new(
            MapErrorKind::Parse,
            format!("a segments[] entry is missing \"{}\"", field),
        )
    };
    let gen = v.get("gen").ok_or_else(|| missing("gen"))?;
    let gen_line = gen.get("line").and_then(|x| x.as_u32()).ok_or_else(|| missing("gen.line"))?;
    let gen_col_start = gen
        .get("col_start")
        .and_then(|x| x.as_u32())
        .ok_or_else(|| missing("gen.col_start"))?;
    let gen_col_end = gen
        .get("col_end")
        .and_then(|x| x.as_u32())
        .ok_or_else(|| missing("gen.col_end"))?;
    let gen_line_end = gen
        .get("line_end")
        .and_then(|x| x.as_u32())
        .ok_or_else(|| missing("gen.line_end"))?;
    if gen_col_end < gen_col_start || gen_line_end < gen_line {
        return Err(MapError::new(
            MapErrorKind::Parse,
            format!(
                "segment at generated line {} has an inverted span (col {}..{} / lines {}..{})",
                gen_line, gen_col_start, gen_col_end, gen_line, gen_line_end
            ),
        ));
    }
    let src = v.get("src").ok_or_else(|| missing("src"))?;
    let src_source_raw = src
        .get("source")
        .and_then(|x| x.as_u32())
        .ok_or_else(|| missing("src.source"))?;
    if src_source_raw > u16::MAX as u32 {
        return Err(MapError::new(
            MapErrorKind::Parse,
            format!(
                "segment at generated line {} has src.source {} (exceeds the u16 source-index range)",
                gen_line, src_source_raw
            ),
        ));
    }
    let src_source = src_source_raw as u16;
    // `src.line: null` (unanchored) reads back as 0.
    let src_line = opt_u32(src.get("line"))?.unwrap_or(0);
    let src_col_start = opt_u32(src.get("col_start"))?;
    let src_col_end = opt_u32(src.get("col_end"))?;
    let origin_str = v
        .get("origin")
        .and_then(|x| x.as_str())
        .ok_or_else(|| missing("origin"))?;
    let mut origin = Origin::parse(origin_str).ok_or_else(|| {
        MapError::new(
            MapErrorKind::Parse,
            format!("unknown segment origin \"{}\"", origin_str),
        )
    })?;
    if let Origin::StdSplice { module } = &mut origin {
        *module = v
            .get("module")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
    }
    let kind_str = v
        .get("kind")
        .and_then(|x| x.as_str())
        .ok_or_else(|| missing("kind"))?;
    let kind = SegKind::parse(kind_str).ok_or_else(|| {
        MapError::new(
            MapErrorKind::Parse,
            format!("unknown segment kind \"{}\"", kind_str),
        )
    })?;
    let name = match v.get("name") {
        Some(Json::Str(n)) => Some(n.clone()),
        Some(Json::Null) | None => None,
        _ => {
            return Err(MapError::new(
                MapErrorKind::Parse,
                "a segments[] entry has a non-string \"name\"".to_string(),
            ))
        }
    };
    let precision_str = v
        .get("precision")
        .and_then(|x| x.as_str())
        .ok_or_else(|| missing("precision"))?;
    let precision = Precision::parse(precision_str).ok_or_else(|| {
        MapError::new(
            MapErrorKind::Parse,
            format!("unknown segment precision \"{}\"", precision_str),
        )
    })?;
    Ok(Segment {
        gen: GenPos { line: gen_line, col_start: gen_col_start, col_end: gen_col_end, line_end: gen_line_end },
        src: SrcPos { source: src_source, line: src_line, col_start: src_col_start, col_end: src_col_end },
        origin,
        kind,
        name,
        precision,
        panic_code: None,
    })
}

// ---------------------------------------------------------------------
// Bidirectional lookup (design §3.3 query semantics, shared by
// `psa map-loc` and future debugger consumers).
// ---------------------------------------------------------------------

/// Result of a psa -> generated query: **all** user segments for
/// the psa line (never just the first — one psa line can generate
/// several segments, including multiple monomorphized instances).
#[derive(Debug, Clone)]
pub enum PsaLookup {
    Mapped {
        segments: Vec<usize>,
        approximate: bool,
        note: Option<String>,
    },
    Unmapped {
        reason: String,
    },
}

/// Result of a generated -> psa query.
#[derive(Debug, Clone)]
pub enum GenLookup {
    Mapped {
        segment: usize,
        approximate: bool,
        note: Option<String>,
    },
    Unmapped {
        reason: String,
    },
}

impl SourceMap {
    /// psa -> generated: all **user-origin** segments anchored to
    /// (`source`, `line`). With `col`, only segments whose psa
    /// column span contains it are exact; line-level segments
    /// (no column info) are returned instead, marked approximate.
    pub fn lookup_psa(&self, source: u16, line: u32, col: Option<u32>) -> PsaLookup {
        let idxs = match self.by_src_line.get(&(source, line)) {
            Some(idxs) => idxs,
            None => {
                return PsaLookup::Unmapped {
                    reason: format!(
                        "no segment maps source {} line {} (the line has no user code, or the map is partial)",
                        source, line
                    ),
                }
            }
        };
        let user: Vec<usize> = idxs
            .iter()
            .cloned()
            .filter(|&i| self.segments[i].origin == Origin::User)
            .collect();
        if user.is_empty() {
            return PsaLookup::Unmapped {
                reason: format!(
                    "source {} line {} is only covered by non-user segments (prelude / synthetic / std-splice)",
                    source, line
                ),
            };
        }
        match col {
            None => PsaLookup::Mapped {
                segments: user,
                approximate: false,
                note: None,
            },
            Some(c) => {
                let covering: Vec<usize> = user
                    .iter()
                    .cloned()
                    .filter(|&i| {
                        let s = &self.segments[i];
                        match (s.src.col_start, s.src.col_end) {
                            (Some(a), Some(b)) => a <= c && c < b,
                            _ => false,
                        }
                    })
                    .collect();
                if !covering.is_empty() {
                    return PsaLookup::Mapped {
                        segments: covering,
                        approximate: false,
                        note: None,
                    };
                }
                let line_level: Vec<usize> = user
                    .iter()
                    .cloned()
                    .filter(|&i| {
                        let s = &self.segments[i];
                        s.src.col_start.is_none() || s.src.col_end.is_none()
                    })
                    .collect();
                if !line_level.is_empty() {
                    return PsaLookup::Mapped {
                        segments: line_level,
                        approximate: true,
                        note: Some(format!(
                            "column {} is not resolved exactly: only line-level segments cover source {} line {}",
                            c, source, line
                        )),
                    };
                }
                PsaLookup::Unmapped {
                    reason: format!(
                        "no segment covers column {} on source {} line {}",
                        c, source, line
                    ),
                }
            }
        }
    }

    /// generated -> psa: the segment covering `line` (a line
    /// inside a multi-line segment hits that segment). A line in
    /// a gap between segments resolves to the nearest preceding
    /// segment in the same file, marked approximate. A line before
    /// the first segment is unmapped (unreachable in a
    /// full-coverage map — the design's full-coverage assertion
    /// is what makes this case "should not happen").
    pub fn lookup_gen(&self, line: u32) -> GenLookup {
        // Covering segment with the latest start (most specific
        // when spans overlap).
        let mut best: Option<usize> = None;
        for (i, s) in self.segments.iter().enumerate() {
            if s.gen.line <= line && line <= s.gen.line_end {
                let better = match best {
                    None => true,
                    Some(b) => s.gen.line > self.segments[b].gen.line,
                };
                if better {
                    best = Some(i);
                }
            }
        }
        if let Some(i) = best {
            return GenLookup::Mapped {
                segment: i,
                approximate: false,
                note: None,
            };
        }
        // Nearest preceding segment (greatest gen_line_end below
        // the queried line).
        let mut prev: Option<usize> = None;
        for (i, s) in self.segments.iter().enumerate() {
            if s.gen.line_end < line {
                let better = match prev {
                    None => true,
                    Some(b) => s.gen.line_end > self.segments[b].gen.line_end,
                };
                if better {
                    prev = Some(i);
                }
            }
        }
        match prev {
            Some(i) => GenLookup::Mapped {
                segment: i,
                approximate: true,
                note: Some(format!(
                    "generated line {} falls in a gap between segments; nearest preceding segment ends at line {}",
                    line, self.segments[i].gen.line_end
                )),
            },
            None => GenLookup::Unmapped {
                reason: format!(
                    "generated line {} precedes the first segment (starts at line {})",
                    line,
                    self.segments.first().map(|s| s.gen.line).unwrap_or(0)
                ),
            },
        }
    }

    /// Panic sites for one code (P4 consumes this).
    pub fn panic_sites_for_code(&self, code: u32) -> Vec<&PanicSite> {
        self.panic_sites.iter().filter(|p| p.code == code).collect()
    }

    /// Name-bridge lookup (in-memory only, P3's function-level
    /// mapping): the psa definition site of a generated symbol.
    pub fn lookup_name_bridge(&self, generated_name: &str) -> Option<(u16, u32)> {
        self.name_bridge
            .iter()
            .find(|b| b.generated == generated_name)
            .map(|b| (b.source, b.def_line))
    }

    // -- source verification (design §3.3: sha256 mismatch is a
    // loud stale-map refusal, never a silent fallback) -----------

    /// Resolve one source entry to a file on disk: the stored
    /// path first as given, then relative to the map file's
    /// directory. `None` when it is in neither place.
    fn source_disk_path(&self, idx: usize, map_dir: &Path) -> Option<PathBuf> {
        let p = Path::new(&self.sources[idx].path);
        if p.is_file() {
            Some(p.to_path_buf())
        } else {
            let q = map_dir.join(p);
            if q.is_file() {
                Some(q)
            } else {
                None
            }
        }
    }

    /// Read one source's bytes: the embedded `content` when the
    /// map carries it, else the file on disk (the stored path
    /// first as given, then relative to the map file's directory).
    fn read_source_bytes(&self, idx: usize, map_dir: &Path) -> Result<Vec<u8>, MapError> {
        let entry = &self.sources[idx];
        if let Some(c) = &entry.content {
            return Ok(c.as_bytes().to_vec());
        }
        match self.source_disk_path(idx, map_dir) {
            Some(path) => std::fs::read(&path).map_err(|e| {
                MapError::new(
                    MapErrorKind::Io,
                    format!("cannot read source {}: {}", path.display(), e),
                )
            }),
            None => Err(MapError::new(
                MapErrorKind::StaleMap,
                format!(
                    "source '{}' is not on disk and the map carries no embedded content, so its sha256 cannot be verified (the map may be stale or the sources moved)",
                    entry.path
                ),
            )),
        }
    }

    /// Verify every source against its recorded sha256. Any
    /// mismatch — or any user source that cannot be checked — is a
    /// `StaleMap` error: the map must not be used to answer
    /// queries about different source text. Two entries are
    /// unverifiable by construction and are skipped, never failed:
    /// the prelude entry (empty sha256 — it names a version
    /// identifier, not a file) and std-splice distribution files
    /// (`psa-std/*.sla`) that are not on disk next to the map
    /// (their sha was recorded at build time from the spliced
    /// bytes; a map whose std region went stale is still honest
    /// about its user segments, which is what queries answer).
    pub fn verify_sources(&self, map_dir: &Path) -> Result<(), MapError> {
        for (i, entry) in self.sources.iter().enumerate() {
            if entry.sha256.is_empty() {
                continue;
            }
            if entry.content.is_none()
                && self.source_disk_path(i, map_dir).is_none()
                && entry.path.starts_with("psa-std/")
            {
                continue;
            }
            let bytes = self.read_source_bytes(i, map_dir)?;
            let actual = sha256_hex(&bytes);
            if actual != entry.sha256 {
                return Err(MapError::new(
                    MapErrorKind::StaleMap,
                    format!(
                        "source '{}' sha256 mismatch: the map was built for a different revision (map says {}, file is {})",
                        entry.path, entry.sha256, actual
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The text of one psa source line (1-based), for the
    /// `snippet` field of `map-loc` output. `None` when the
    /// source cannot be read or the line is out of range.
    pub fn source_line_text(&self, source: u16, line: u32, map_dir: &Path) -> Option<String> {
        let idx = source as usize;
        if idx >= self.sources.len() {
            return None;
        }
        let bytes = self.read_source_bytes(idx, map_dir).ok()?;
        let text = String::from_utf8(bytes).ok()?;
        text.lines().nth((line as usize).saturating_sub(1)).map(|s| s.to_string())
    }

    /// Resolve a `--psa <file>` argument to a sources[] index:
    /// exact path match, then unique basename match, then (when
    /// the file exists on disk) a sha256 match. Ambiguous or
    /// absent files are loud `NotInMap` errors, never guesses.
    pub fn resolve_source(&self, file: &str, map_dir: &Path) -> Result<usize, MapError> {
        if let Some(i) = self.sources.iter().position(|s| s.path == file) {
            return Ok(i);
        }
        let base = file.rsplit('/').next().unwrap_or(file);
        let mut base_hits: Vec<usize> = self
            .sources
            .iter()
            .enumerate()
            .filter(|(_, s)| s.path.rsplit('/').next().unwrap_or(&s.path) == base)
            .map(|(i, _)| i)
            .collect();
        if base_hits.len() == 1 {
            return Ok(base_hits.remove(0));
        }
        if base_hits.len() > 1 {
            return Err(MapError::new(
                MapErrorKind::NotInMap,
                format!(
                    "source '{}' matches {} entries in the map ({}); use the full path from sources[]",
                    file,
                    base_hits.len(),
                    base_hits
                        .iter()
                        .map(|&i| self.sources[i].path.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }
        // The file itself may be on disk: match by content hash.
        let p = Path::new(file);
        let cand: Option<PathBuf> = if p.is_file() {
            Some(p.to_path_buf())
        } else {
            let q = map_dir.join(p);
            if q.is_file() {
                Some(q)
            } else {
                None
            }
        };
        if let Some(path) = cand {
            if let Ok(bytes) = std::fs::read(&path) {
                let actual = sha256_hex(&bytes);
                if let Some(i) = self.sources.iter().position(|s| s.sha256 == actual) {
                    return Ok(i);
                }
            }
        }
        Err(MapError::new(
            MapErrorKind::NotInMap,
            format!(
                "source '{}' is not in this map (sources: {})",
                file,
                self.sources
                    .iter()
                    .map(|s| s.path.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ))
    }

    /// Look up a panic code in the psa code table (delegates to
    /// `panic_codes::lookup_panic`). Returns `None` for codes the
    /// table does not know — the caller must render those as
    /// "unknown code", never guess a meaning. Known gap at the
    /// base snapshot: `panic(77083)` is emitted by
    /// `CLOSURE_RUNTIME` (emit.rs:12935, the modenv fetch) but
    /// has NO row in `panic_codes.rs` (the file is generated;
    /// the gap belongs to its generator, not to this module).
    pub fn lookup_panic_code(code: u32) -> Option<&'static crate::panic_codes::PanicInfo> {
        crate::panic_codes::lookup_panic(code)
    }
}

// ---------------------------------------------------------------------
// P4: the user panic emitters (design §3.6).
// ---------------------------------------------------------------------

/// Where a `panic(N)` literal is emitted for user code in
/// `emit.rs` (base snapshot: `impl/base-sha256.txt`, emit.rs
/// `b0edc6ae…`). This is the complete set: a grep over base
/// `emit.rs` finds 189 `panic(` occurrences, of which 183 sit
/// inside prelude text (the `__psa_*` helper builders around
/// lines 680–1140 and the `DYN_RUNTIME` / `CLOSURE_RUNTIME` /
/// `D6_EXC_RUNTIME` raw-string constants, lines 12901–15766)
/// and are therefore `origin == Prelude`, excluded from the
/// view by rule. The 6 sites below are the only ones emitted
/// per user statement, inside `emit_stmt`/`emit_expr` with the
/// statement's psa `line` in scope; the writer's line scan
/// (not this table) is what records them.
///
/// `emit_rs_line` is a snapshot line number and WILL drift as
/// `emit.rs` evolves — re-grep at integration and keep this table
/// in sync; the `construct` + `code` columns are the stable
/// identity, the line number is a locator.
#[derive(Debug, Clone, Copy)]
pub struct UserPanicEmitter {
    /// Base-snapshot `emit.rs` line number (locator only).
    pub emit_rs_line: u32,
    pub code: u32,
    pub construct: &'static str,
}

pub static USER_PANIC_EMITTERS: &[UserPanicEmitter] = &[
    UserPanicEmitter {
        emit_rs_line: 4250,
        code: 77074,
        construct: "method-call __init__ returns non-None (emit_expr, Ty::Dyn check)",
    },
    UserPanicEmitter {
        emit_rs_line: 6107,
        code: 77100,
        construct: "exc_unwind_action: uncaught exception in main (raise / trailing check)",
    },
    UserPanicEmitter {
        emit_rs_line: 6792,
        code: 77040,
        construct: "assert with message (stderr label first, then panic)",
    },
    UserPanicEmitter {
        emit_rs_line: 6796,
        code: 77040,
        construct: "assert without message",
    },
    UserPanicEmitter {
        emit_rs_line: 6938,
        code: 77077,
        construct: "static list unpack length check",
    },
    UserPanicEmitter {
        emit_rs_line: 7070,
        code: 77069,
        construct: "del of a class attribute that is not in the class dict",
    },
];

// ---------------------------------------------------------------------
// Unit tests (run on the box: `cargo test` in psa-src; no local Rust
// toolchain on this VM, so these are written, not executed, here).
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_map() -> SourceMap {
        // c_panic.psa, hand-computed (see corpus/c_panic.psa-map.json):
        // seg 0 = prelude imports (gen 1..12, unanchored);
        // seg 1 = `xs = [1, 2, 3]` (gen 3156..3159, psa line 1);
        // seg 2 = `print(xs[10])` (gen 3160, psa line 2).
        let mut m = SourceMap::new();
        m.sources = vec![SourceEntry {
            path: "c_panic.psa".to_string(),
            sha256: "f1828b62abc05112ec5153d8f1cd33c5a4ac4cf550992dd70ee56db6d278a2d9".to_string(),
            content: None,
        }];
        m.push_segment(Segment {
            gen: GenPos { line: 1, col_start: 1, col_end: 26, line_end: 12 },
            src: SrcPos { source: 0, line: 0, col_start: None, col_end: None },
            origin: Origin::Prelude,
            kind: SegKind::Stmt,
            name: None,
            precision: Precision::None,
            panic_code: None,
        });
        m.push_segment(Segment {
            gen: GenPos { line: 3156, col_start: 5, col_end: 16, line_end: 3159 },
            src: SrcPos { source: 0, line: 1, col_start: Some(1), col_end: Some(15) },
            origin: Origin::User,
            kind: SegKind::Stmt,
            name: None,
            precision: Precision::Exact,
            panic_code: None,
        });
        m.push_segment(Segment {
            gen: GenPos { line: 3160, col_start: 5, col_end: 28, line_end: 3160 },
            src: SrcPos { source: 0, line: 2, col_start: Some(1), col_end: Some(14) },
            origin: Origin::User,
            kind: SegKind::Stmt,
            name: None,
            precision: Precision::Exact,
            panic_code: None,
        });
        m.finish("line1\nline2\n", "c_panic.sla".to_string());
        m
    }

    #[test]
    fn sha256_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn round_trip_rebuilds_indexes() {
        let map = sample_map();
        let json = map.to_json();
        let back = SourceMap::from_json(&json).expect("round-trip parse");
        assert_eq!(back.segments.len(), 3);
        assert_eq!(back.segments, map.segments);
        assert_eq!(back.by_gen_line, map.by_gen_line);
        assert_eq!(back.by_src_line, map.by_src_line);
        assert_eq!(back.panic_sites, map.panic_sites);
        // The file's stored indexes are ignored on read: corrupting
        // them must not change the rebuilt map.
        let tampered = json.replace("\"by_gen_line\":{\"1\":[0]", "\"by_gen_line\":{\"1\":[2]");
        let back2 = SourceMap::from_json(&tampered).expect("tampered-index parse");
        assert_eq!(back2.by_gen_line, map.by_gen_line);
    }

    #[test]
    fn psa_lookup_returns_all_user_segments() {
        let map = sample_map();
        match map.lookup_psa(0, 1, None) {
            PsaLookup::Mapped {
                segments,
                approximate,
                ..
            } => {
                assert_eq!(segments, vec![1]);
                assert!(!approximate);
            }
            other => panic!("expected mapped, got {:?}", other),
        }
        match map.lookup_psa(0, 3, None) {
            PsaLookup::Unmapped { .. } => {}
            other => panic!("expected unmapped for psa line 3, got {:?}", other),
        }
    }

    #[test]
    fn gen_lookup_gap_is_approximate() {
        let map = sample_map();
        // Line 3160 is covered exactly.
        match map.lookup_gen(3160) {
            GenLookup::Mapped {
                segment,
                approximate,
                ..
            } => {
                assert_eq!(segment, 2);
                assert!(!approximate);
            }
            other => panic!("expected exact, got {:?}", other),
        }
        // Line 3157 is inside the multi-line segment 1.
        match map.lookup_gen(3157) {
            GenLookup::Mapped {
                segment,
                approximate,
                ..
            } => {
                assert_eq!(segment, 1);
                assert!(!approximate);
            }
            other => panic!("expected exact (multi-line), got {:?}", other),
        }
        // Line 3161 is in a gap: nearest preceding segment, flagged.
        match map.lookup_gen(3161) {
            GenLookup::Mapped {
                segment,
                approximate,
                note,
            } => {
                assert_eq!(segment, 2);
                assert!(approximate);
                assert!(note.is_some());
            }
            other => panic!("expected approximate, got {:?}", other),
        }
    }

    #[test]
    fn col_filter_falls_back_to_line_level() {
        let mut map = sample_map();
        // Add a line-level user segment on psa line 2 (no columns).
        map.segments.push(Segment {
            gen: GenPos { line: 3161, col_start: 5, col_end: 15, line_end: 3161 },
            src: SrcPos { source: 0, line: 2, col_start: None, col_end: None },
            origin: Origin::User,
            kind: SegKind::Stmt,
            name: None,
            precision: Precision::LineOnly,
            panic_code: None,
        });
        let (by_gen, by_src) = build_indexes(&map.segments);
        map.by_gen_line = by_gen;
        map.by_src_line = by_src;
        match map.lookup_psa(0, 2, Some(5)) {
            PsaLookup::Mapped {
                segments,
                approximate,
                ..
            } => {
                assert_eq!(segments, vec![2]);
                assert!(!approximate);
            }
            other => panic!("expected exact col hit, got {:?}", other),
        }
        // Column 99 is covered by no exact span, so the query
        // falls back to the line-level segment, marked approximate
        // (it must not claim the column was resolved exactly).
        match map.lookup_psa(0, 2, Some(99)) {
            PsaLookup::Mapped {
                segments,
                approximate,
                note,
            } => {
                assert_eq!(segments, vec![3]);
                assert!(approximate);
                assert!(note.is_some());
            }
            other => panic!("expected approximate fallback for col 99, got {:?}", other),
        }
    }

    #[test]
    fn prelude_segments_remap_to_assembly_prelude_index() {
        // Two user files: the prelude lands at index 2, not 1.
        let user = vec![
            SourceEntry { path: "main.psa".to_string(), sha256: "a".to_string(), content: None },
            SourceEntry { path: "mod.psa".to_string(), sha256: "b".to_string(), content: None },
        ];
        let mut asm = SegmentWriter::with_sources(user);
        assert_eq!(asm.prelude_source, 2);
        asm.push_raw("@import \"sa_std/io/print.sai\"\n", &SegMeta::prelude(SegKind::Stmt));
        assert_eq!(asm.segments[0].src.source, 2);
        assert_eq!(asm.segments[0].origin, Origin::Prelude);
        // A user line anchored to the module file keeps index 1.
        asm.push_line(
            "x = 1;",
            &SegMeta {
                anchor: SourceAnchor::new(1, 3, 1, None),
                origin: Origin::User,
                kind: SegKind::Stmt,
                name: None,
                precision: Precision::Exact,
            },
        );
        assert_eq!(asm.segments[1].src.source, 1);
        assert_eq!(asm.segments[1].src.line, 3);
        // A chunk-local prelude segment remaps on append.
        let chunk = EmittedChunk::span(
            "fn __psa_ipow() -> i32 {\n}\n".to_string(),
            &SegMeta::prelude_fn("__psa_ipow"),
        );
        asm.append_chunk(&chunk);
        assert!(asm
            .segments
            .iter()
            .filter(|s| s.origin == Origin::Prelude)
            .all(|s| s.src.source == 2));
    }

    /// The c_panic shape, built by hand from the measured
    /// evidence (EVIDENCE.md §C): c_panic.sla is 3,162 lines,
    /// user code starts at generated line 3154, and generated
    /// line 3160 (`println("{}", xs[10]);`) is the statement
    /// for psa line 2. A prelude segment stands in for the
    /// runtime region. No segment here is a user PanicSite:
    /// `xs[10]` is a native SLA Vec index, not a `panic(N)`
    /// literal, so `panic_sites` must come out EMPTY.
    fn c_panic_map() -> SourceMap {
        let segments = vec![
            Segment {
                gen: GenPos { line: 426, col_start: 1, col_end: 45, line_end: 426 },
                src: SrcPos { source: 0, line: 0, col_start: None, col_end: None },
                origin: Origin::Prelude,
                kind: SegKind::PanicSite,
                name: Some("__psa_vec_get".to_string()),
                precision: Precision::None,
                panic_code: Some(77050),
            },
            Segment::user_stmt(3160, 1, 27, 0, 2, Some(1), Some(14)),
        ];
        SourceMap::from_segments(segments, Vec::new())
    }

    #[test]
    fn c_panic_gen_line_3160_resolves_to_psa_line_2() {
        let m = c_panic_map();
        match m.lookup_gen(3160) {
            GenLookup::Mapped { segment, approximate, .. } => {
                assert!(!approximate);
                assert_eq!(m.segments[segment].src.line, 2);
                assert_eq!(m.segments[segment].src.source, 0);
            }
            GenLookup::Unmapped { reason } => panic!("expected a hit, got unmapped: {}", reason),
        }
    }

    #[test]
    fn c_panic_panic_sites_is_empty() {
        // The honest answer for c_panic: no user segment holds a
        // panic(N) literal, so the view is empty. The prelude
        // 77050 site above must NOT leak into it.
        let m = c_panic_map();
        assert!(m.panic_sites.is_empty());
        assert_eq!(m.to_json().matches("\"panic_sites\":[]").count(), 1);
    }

    #[test]
    fn c_panic_src_line_2_lists_its_segment() {
        let m = c_panic_map();
        match m.lookup_psa(0, 2, None) {
            PsaLookup::Mapped { segments, .. } => {
                assert_eq!(segments.len(), 1);
                assert_eq!(m.segments[segments[0]].gen.line, 3160);
            }
            PsaLookup::Unmapped { reason } => panic!("expected mapped, got: {}", reason),
        }
    }

    #[test]
    fn gap_line_returns_previous_segment_as_approximate() {
        let m = c_panic_map();
        // Generated line 3161 (`return 0;`) has no segment of its
        // own in this hand-built table: the previous segment
        // (3160) is the answer, labelled approximate.
        match m.lookup_gen(3161) {
            GenLookup::Mapped { segment, approximate, .. } => {
                assert!(approximate);
                assert_eq!(m.segments[segment].gen.line, 3160);
            }
            GenLookup::Unmapped { reason } => panic!("expected approximate hit, got: {}", reason),
        }
    }

    #[test]
    fn line_before_first_segment_is_unmapped_not_a_guess() {
        let m = c_panic_map();
        match m.lookup_gen(1) {
            GenLookup::Unmapped { .. } => {}
            GenLookup::Mapped { .. } => panic!("line 1 must not resolve to a neighbour"),
        }
    }

    /// A user assert: the one user-emitted panic literal shape
    /// P4 can exhibit without running the compiler. The prelude
    /// 77050 site must stay out; the user 77040 site must be
    /// the view's only entry, with its code, generated line
    /// and segment index.
    #[test]
    fn user_assert_panic_site_is_listed_prelude_is_not() {
        let segments = vec![
            Segment {
                gen: GenPos { line: 426, col_start: 1, col_end: 45, line_end: 426 },
                src: SrcPos { source: 0, line: 0, col_start: None, col_end: None },
                origin: Origin::Prelude,
                kind: SegKind::PanicSite,
                name: Some("__psa_vec_get".to_string()),
                precision: Precision::None,
                panic_code: Some(77050),
            },
            Segment::user_panic_site(3160, 1, 80, 0, 2, Some(1), Some(20), 77040),
        ];
        let panics = vec![PanicSite { code: 77040, gen_line: 3160, segment: 1 }];
        let m = SourceMap::from_segments(segments, panics);
        assert_eq!(m.panic_sites.len(), 1);
        assert_eq!(
            m.panic_sites[0],
            PanicSite {
                code: 77040,
                gen_line: 3160,
                segment: 1,
            }
        );
        // The code resolves in the table; the prelude code would
        // too, but it is never listed, so it can never be
        // attributed to a user line through this view.
        assert!(SourceMap::lookup_panic_code(77040).is_some());
        assert_eq!(
            SourceMap::lookup_panic_code(77040).unwrap().rule,
            "assert-failed"
        );
    }

    #[test]
    fn unknown_panic_code_stays_unknown() {
        // 77083 is emitted by CLOSURE_RUNTIME but has no row in
        // panic_codes.rs; the view must not invent a meaning.
        assert!(SourceMap::lookup_panic_code(77083).is_none());
    }

    #[test]
    fn user_panic_emitters_cover_all_six_sites() {
        // The wiring table must stay complete: six sites, and
        // every code in it exists in the panic-code table.
        assert_eq!(USER_PANIC_EMITTERS.len(), 6);
        for e in USER_PANIC_EMITTERS {
            assert!(
                SourceMap::lookup_panic_code(e.code).is_some(),
                "emitter code {} has no panic_codes row",
                e.code
            );
        }
    }
}
