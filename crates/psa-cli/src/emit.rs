//! Type inference + SLA emission for the .psa v0.1 subset.
//!
//! The .psa layer infers concrete types and ownership-neutral scalar
//! semantics, then emits SLA source whose explicit annotations carry the
//! inferred facts; the SLA checker and the SA Referee verify them
//! downstream. Anything the subset cannot express is a loud error.

use crate::ast::*;
use crate::diag::Diagnostic;
use crate::srcmap::{
    sha256_hex, EmittedChunk, NameBinding, Origin, Precision, SegKind, SegMeta, SourceEntry,
    SourceMap, SegmentWriter, SRC_MAIN,
};
use std::collections::HashMap;
use std::collections::HashSet;

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
    /// A dynamic (tagged) value: the D1 PsaVal runtime. Arises when
    /// static inference cannot pin one scalar type (mixed return
    /// branches, mixed conditional-expression branches, a signature
    /// first inferred from a dynamic argument). D1 dynamic values are
    /// the scalar kinds int/float/bool/None; dynamic containers and
    /// strings arrive in D2.
    Dyn,
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
            Ty::Dyn => "PsaVal",
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
            // The None value of the dynamic runtime (tag 3).
            Ty::Dyn => "PsaVal { tag: 3, ipay: 0, fpay: 0.0 }",
        }
    }
    /// The scalar kinds a dynamic value can hold in D1.
    fn is_scalar(&self) -> bool {
        matches!(self, Ty::Int | Ty::Float | Ty::Bool)
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
            Ty::Dyn => "dynamic".to_string(),
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

/// D4 class registry entry: a class's methods and attributes
/// flattened over its base chain at registry time (single
/// inheritance; own definitions win). Method values are def keys
/// (`ClassName.method`) into the ordinary defs table.
struct ClassInfo {
    id: i64,
    name: String,
    flat_methods: Vec<(String, String)>,
    flat_attrs: Vec<(String, Expr)>,
    /// This class's OWN attribute literals, in body order
    /// (the runtime-registry seed writes exactly these into
    /// the class's own dict; reads walk the ancestor chain).
    attrs: Vec<(String, Expr)>,
    /// This class's id plus every ancestor's id (D5, backing
    /// isinstance: an instance of a subclass carries all of
    /// them in its ancestry).
    ancestors: Vec<i64>,
}

pub struct Emitter {
    defs: HashMap<String, Func>,
    sigs: HashMap<String, Sig>,
    in_progress: Vec<String>,
    emitted: Vec<EmittedChunk>,
    emitted_set: Vec<String>,
    /// D3 function values: source name -> callable id, plus the
    /// id-ordered callable names. The dispatch (`__psa_call_value`
    /// and its per-function adapters) is generated at assembly
    /// time from `callables`.
    fn_ids: HashMap<String, i64>,
    callables: Vec<String>,
    /// D8 closures: the same registry for closure defs (defs
    /// whose first parameter is the hidden environment).
    /// Closure callables get their OWN id space and their own
    /// adapters (`__psa_cadapt_<j>`), generated at assembly
    /// time from `closure_callables`; `used_closure` gates
    /// the CLOSURE_RUNTIME text (and the tag-16 branch of
    /// `__psa_call_value`).
    fn_cids: HashMap<String, i64>,
    closure_callables: Vec<String>,
    used_closure: bool,
    /// D4 classes: the registry in definition order, a name ->
    /// index map, and def key -> emitted SLA name for methods
    /// (the dotted def key is not a valid SLA identifier; the
    /// assigned `__py_cls_<id>_<method>` names are injective).
    classes: Vec<ClassInfo>,
    class_ids: HashMap<String, usize>,
    sla_names: HashMap<String, String>,
    /// D6 imports: module literal attributes by dotted key
    /// (`foo.X`), literal names bound by from-imports, and the
    /// set of merged module names (for `foo.Class(...)` and
    /// `foo.CONST` resolution through attribute chains).
    mod_attrs: HashMap<String, Expr>,
    const_names: HashMap<String, Expr>,
    modules: Vec<String>,
    /// Source file names in source-id order (Program.sources),
    /// handed down to each Ctx for move-error file resolution.
    sources: Vec<String>,
    used_floordiv: bool,
    used_floormod: bool,
    used_vec: bool,
    used_ipow: bool,
    used_math: bool,
    used_dyn: bool,
    used_file: bool,
    /// D7: generator / coroutine bookkeeping. `gen_names` and
    /// `coro_names` are filled by the emit_program pre-pass
    /// (before any body is emitted, so iteration sites know
    /// whether `__psa_iter_source` exists); `used_gen` /
    /// `used_coro` gate the D7 runtime text and the chan
    /// imports; `genexp_counter` names synthetic generator
    /// defs built for generator expressions.
    gen_names: Vec<String>,
    coro_names: Vec<String>,
    used_gen: bool,
    used_coro: bool,
    genexp_counter: usize,
    /// psa_std: `import json` was bound to the standard
    /// library by the driver's merge pass; `used_psa_json`
    /// is set when a json entry point is actually called,
    /// which splices the module's SLA text at assembly.
    std_json: bool,
    used_psa_json: bool,
    /// psa_std: the same wiring for `pickle` (thin data-only
    /// wrapper; dump/load also need the file runtime).
    std_pickle: bool,
    used_psa_pickle: bool,
    /// psa_std L1 batch 2: the same wiring for `math` and
    /// `string` -- `std_*` records the merge-pass binding,
    /// `used_psa_*` is set when an entry point is actually
    /// called and gates the SLA splice at assembly.
    std_math: bool,
    used_psa_math: bool,
    std_string: bool,
    used_psa_string: bool,
    /// psa_std batch 3: the same wiring for csv, logging,
    /// argparse, collections, itertools, functools, bisect,
    /// heapq, base64, random and re (see
    /// psa-std/BATCH-WIRING.md).
    std_csv: bool,
    used_psa_csv: bool,
    std_logging: bool,
    used_psa_logging: bool,
    std_argparse: bool,
    used_psa_argparse: bool,
    std_collections: bool,
    used_psa_collections: bool,
    std_itertools: bool,
    used_psa_itertools: bool,
    std_functools: bool,
    used_psa_functools: bool,
    std_bisect: bool,
    used_psa_bisect: bool,
    std_heapq: bool,
    used_psa_heapq: bool,
    std_base64: bool,
    used_psa_base64: bool,
    std_random: bool,
    used_psa_random: bool,
    std_re: bool,
    used_psa_re: bool,
    std_socket: bool,
    used_psa_socket: bool,
    std_urllib: bool,
    used_psa_urllib: bool,
    /// D6: the program uses try/raise/with anywhere (pre-scan
    /// in emit_program). Gates the trailing pending-exception
    /// check after call-containing statements (emit_stmt),
    /// which is what unwinds a callee's exception through
    /// frames that have no try of their own.
    program_exc: bool,
    /// Loop-else support: a stack with one entry per loop
    /// currently being emitted (innermost last). The entry is
    /// the loop's break-flag variable name when the loop has an
    /// `else:` clause, or None when it does not. emit_break_
    /// through sets the innermost loop's flag (when it has one)
    /// before emitting the break itself, so the loop can run
    /// its else clause exactly when no break fired.
    loop_else_stack: Vec<Option<String>>,
}

struct Ctx {
    vars: Vec<(String, Ty, Option<SourceAnchor>)>,
    /// The program's source file names, in source-id order
    /// (Program.sources); used to fill Diagnostic.file for
    /// move errors. Mirrors the Emitter's copy.
    sources: Vec<String>,
    scopes: Vec<usize>,
    ret: Option<Ty>,
    fname: String,
    out: SegmentWriter,
    /// Function-prologue lines (hoisted declarations). On the Rust
    /// sla build-exe route a block-scoped aggregate local is never
    /// retired (hand-written probes q16/q29: MemoryLeak at function
    /// exit), so a dynamic variable or temp first materialised inside
    /// a block has its `let` hoisted here, to function scope, and is
    /// filled in place via the stage pattern (see emit_dyn_store).
    prologue: SegmentWriter,
    indent: usize,
    tmp: usize,
    borrowed: Vec<String>,
    /// True while emitting a while-condition: if-expressions are rejected
    /// there because their materialised temp would only be computed once,
    /// before the loop, not on every evaluation of the condition.
    in_while_cond: bool,
    /// True when a prescan of the function body showed its returns
    /// unify to Dyn: every return value is boxed at emission time.
    dyn_ret: bool,
    /// Names removed by `del` (with the del line). Function-scoped,
    /// like Python's own name binding; re-binding clears the entry.
    /// The emitted SLA variable still exists and drops at scope end;
    /// psa refuses later references, mirroring Python's NameError.
    deleted: Vec<(String, u32)>,
    /// D6 exception machinery: the stack of try frames enclosing
    /// the current emission point (innermost last), the guard
    /// depth (statements emitted while > 0 get a trailing
    /// pending-exception guard that breaks the innermost guard
    /// wrapper), the user-loop depth (a break/continue whose
    /// depth equals the innermost frame's entry depth targets a
    /// loop outside that frame and is flag-rewritten), whether
    /// the pending-exception machinery (a process-global
    /// slot, see ensure_exc_slot) is in use in this function,
    /// and the variable holding the exception currently being
    /// handled (for a bare `raise`).
    try_stack: Vec<TryFrame>,
    guard_depth: usize,
    loop_depth: usize,
    exc_slot: bool,
    exc_bare: Option<String>,
    /// D6: nesting depth of finally-block emission. The
    /// trailing pending-exception check (emit_stmt) is
    /// suppressed inside finally bodies: a finally must run
    /// to completion even with an exception pending; the try
    /// epilogue's own check propagates it afterwards.
    in_finally: usize,
    /// D7: inside a generator producer body, the name of the
    /// channel variable each `yield` sends its wrap on. None
    /// everywhere else (a `yield` there is a loud error).
    gen_chan: Option<String>,
    /// P1 source map: the anchor (psa position) the lines currently
    /// being written belong to, and how they are classified.
    /// `emit_stmt` / `emit_expr` set this on entry and restore the
    /// enclosing anchor on exit; `Ctx::line` records every line
    /// under it. `anchor.line == 0` means synthetic (no psa line).
    anchor: SourceAnchor,
    anchor_kind: SegKind,
    anchor_precision: Precision,
    anchor_name: Option<String>,
    anchor_origin: Origin,
    /// P1 source map: a call boundary noted by `emit_expr`'s Call
    /// arm (`note_call`). The next `Ctx::line` whose text contains
    /// the callee name becomes a `CallSite` segment; see `line`.
    pending_call: bool,
    pending_call_name: Option<String>,
}

/// D6: one try statement's flag machinery. The flags and the
/// return slot live in the function prologue under unique
/// `__psa_tf<id>_*` names; the *_used fields are emitter-side
/// records of whether any break/continue/return was rewritten
/// into this frame (the epilogue only emits the branches that
/// can actually fire). The return slot is declared lazily, at
/// the first rewritten return, with that return's type.
struct TryFrame {
    brk_flag: String,
    cont_flag: String,
    ret_flag: String,
    slot_name: String,
    loop_depth: usize,
    brk_used: bool,
    cont_used: bool,
    ret_used: bool,
    ret_void: bool,
    ret_ty: Option<Ty>,
}

impl Ctx {
    /// Write one generated line and record its source-map segment
    /// under the current anchor (P1). The bytes written are exactly
    /// the old ones (indent + `s` + newline); recording only reads
    /// them. A pending call (see `note_call`) turns this line into
    /// a `CallSite` segment when the line's text contains the
    /// callee's call (`<name>(`); a `panic(N)` in the text turns
    /// it into a `PanicSite` segment instead (the panic scan in
    /// `SegmentWriter::push_line` takes precedence: the panic code
    /// is the more precise signal).
    fn line(&mut self, s: &str) {
        let mut full = String::with_capacity(self.indent * 4 + s.len());
        for _ in 0..self.indent {
            full.push_str("    ");
        }
        full.push_str(s);
        let mut meta = SegMeta {
            anchor: self.anchor,
            origin: self.anchor_origin.clone(),
            kind: self.anchor_kind,
            name: self.anchor_name.clone(),
            precision: self.anchor_precision,
        };
        if self.pending_call {
            let consume = match &self.pending_call_name {
                Some(n) => full.contains(&format!("{}(", n)),
                None => {
                    full.contains("__psa_call_value(")
                        || full.contains("__psa_call_closure(")
                        || full.contains("__psa_call_id(")
                }
            };
            if consume {
                meta.kind = SegKind::CallSite;
                meta.name = self.pending_call_name.clone();
                self.pending_call = false;
                self.pending_call_name = None;
            }
        }
        self.out.push_line(&full, &meta);
    }

    /// Set the anchor under which subsequent lines are recorded.
    /// The origin is derived from the anchor (a psa line anchors
    /// `User`, `line == 0` anchors `Synthetic`); override it with
    /// `set_anchor_origin` for psa-added lines anchored to a user
    /// line (e.g. main's fallthrough `return 0;`).
    fn set_anchor(&mut self, anchor: SourceAnchor, kind: SegKind, precision: Precision, name: Option<String>) {
        self.anchor_origin = if anchor.is_synthetic() { Origin::Synthetic } else { Origin::User };
        self.anchor = anchor;
        self.anchor_kind = kind;
        self.anchor_precision = precision;
        self.anchor_name = name;
    }

    /// Override the origin of the current anchor (see `set_anchor`).
    fn set_anchor_origin(&mut self, origin: Origin) {
        self.anchor_origin = origin;
    }

    /// Save the current anchor (for `emit_stmt` / `emit_expr`
    /// entry/exit discipline).
    fn save_anchor(&self) -> (SourceAnchor, SegKind, Precision, Option<String>, Origin) {
        (
            self.anchor,
            self.anchor_kind,
            self.anchor_precision,
            self.anchor_name.clone(),
            self.anchor_origin.clone(),
        )
    }

    /// Restore an anchor saved by `save_anchor`.
    fn restore_anchor(&mut self, saved: (SourceAnchor, SegKind, Precision, Option<String>, Origin)) {
        self.anchor = saved.0;
        self.anchor_kind = saved.1;
        self.anchor_precision = saved.2;
        self.anchor_name = saved.3;
        self.anchor_origin = saved.4;
    }

    /// Note a call boundary (called by `emit_expr`'s Call arm):
    /// the next line containing the callee's call becomes a
    /// `CallSite` segment. First call wins when calls nest on one
    /// line (the outermost call names the line); `None` means a
    /// dynamic callee (dispatch through `__psa_call_value`).
    fn note_call(&mut self, name: Option<String>) {
        if self.pending_call {
            return;
        }
        self.pending_call = true;
        self.pending_call_name = name;
    }

    /// Drop a pending call note without consuming it (called when
    /// a statement finishes: a note that survived its statement
    /// must not leak into a later statement's lines).
    fn clear_pending_call(&mut self) {
        self.pending_call = false;
        self.pending_call_name = None;
    }

    /// Push raw (newline-terminated) text into the function
    /// prologue and record its segment(s) under the current anchor:
    /// a hoisted declaration belongs to the statement that first
    /// materialised the variable. Every current call site pushes
    /// exactly one line.
    fn prologue_push(&mut self, text: &str) {
        let meta = SegMeta {
            anchor: self.anchor,
            origin: self.anchor_origin.clone(),
            kind: SegKind::Stmt,
            name: None,
            precision: self.anchor_precision,
        };
        self.prologue.push_raw(text, &meta);
    }
    fn lookup(&self, name: &str) -> Option<Ty> {
        self.vars.iter().rev().find(|(n, _, _)| n == name).map(|(_, t, _)| *t)
    }
    /// Ruling #8: the move slot records the move site's full
    /// anchor (line + name-span column + source id), not just a
    /// line, so the E1001 diagnostic can point at both ends.
    fn moved_at(&self, name: &str) -> Option<SourceAnchor> {
        self.vars
            .iter()
            .rev()
            .find(|(n, _, _)| n == name)
            .and_then(|(_, _, m)| *m)
    }
    fn mark_moved(&mut self, name: &str, anchor: SourceAnchor) {
        if let Some(entry) = self.vars.iter_mut().rev().find(|(n, _, _)| n == name) {
            entry.2 = Some(anchor);
        }
    }
    fn revive(&mut self, name: &str) {
        if let Some(entry) = self.vars.iter_mut().rev().find(|(n, _, _)| n == name) {
            entry.2 = None;
        }
    }
    /// Ruling #8: `anchor` is the use site's name span when the
    /// caller has it, or None for a line-only report (MethodCall
    /// objects carry no span; emit_expr has no statement anchor
    /// in scope there -- see the P0b fallback note in
    /// INTEGRATION-RESULT-R2.md). The diagnostic's `file` comes
    /// from the use anchor's source, falling back to the move
    /// anchor's source for line-only uses.
    fn check_alive(
        &self,
        name: &str,
        line: u32,
        anchor: Option<SourceAnchor>,
    ) -> Result<(), Diagnostic> {
        if let Some(moved) = self.moved_at(name) {
            let use_site = match anchor {
                Some(a) => crate::diag::Site::at(a.line, a.col, "use"),
                None => crate::diag::Site::new(line, "use"),
            };
            let mut d = Diagnostic::use_after_move(
                name,
                use_site,
                crate::diag::Site::at(moved.line, moved.col, "move"),
            );
            let src: u16 = anchor.map(|a| a.source).unwrap_or(moved.source);
            if let Some(path) = self.sources.get(src as usize) {
                d.file = Some(path.clone());
            }
            return Err(d);
        }
        Ok(())
    }
    fn check_not_deleted(&self, name: &str, line: u32) -> Result<(), Diagnostic> {
        if let Some((_, at)) = self.deleted.iter().rev().find(|(n, _)| n == name) {
            return Err(Diagnostic::use_after_del(name, line, *at));
        }
        Ok(())
    }
    fn clear_deleted(&mut self, name: &str) {
        self.deleted.retain(|(n, _)| n != name);
    }
    fn declare(&mut self, name: &str, ty: Ty) -> Result<(), String> {
        // Redeclaring in the *same* scope is an error; shadowing an outer
        // scope is allowed (the inner binding wins until the scope ends).
        let mark = *self.scopes.last().unwrap_or(&0);
        if self.vars[mark..].iter().any(|(n, _, _)| n == name) {
            return Err(format!(
                "variable `{}` assigned twice in the same scope (v0.1 treats re-binding as an error; rename or restructure)",
                name
            ));
        }
        self.vars.push((name.to_string(), ty, None));
        Ok(())
    }

    /// Declare `name` in the *function* scope and emit its `let`
    /// (with the type's default literal) into the prologue, wherever
    /// the current emission point is. Used for dynamic bindings first
    /// materialised inside a block: the binding must live at function
    /// scope for the SAB lowering to retire it (probe q29), while
    /// Ctx keeps it visible for the rest of the function (Python's
    /// scoping is function-level anyway).
    fn hoist_declare(&mut self, name: &str, ty: Ty) {
        let pos = if self.scopes.len() > 1 {
            self.scopes[1]
        } else {
            self.vars.len()
        };
        self.vars.insert(pos, (name.to_string(), ty, None));
        for mark in self.scopes.iter_mut().skip(1) {
            if *mark >= pos {
                *mark += 1;
            }
        }
        self.prologue_push(&format!(
            "    let {}: {} = {};\n",
            name,
            ty.sla(),
            ty.default_lit()
        ));
    }
}

fn eerr(line: u32, msg: &str) -> Diagnostic {
    Diagnostic::emit(line, msg)
}

pub fn emit_program(prog: &Program) -> Result<String, Diagnostic> {
    // Plain-text entry point: the map headers are empty here (no
    // file paths / source bytes were supplied); the emitted text
    // is identical. Callers that need a map use
    // `emit_program_with_map`.
    let sources = vec![SourceEntry {
        path: String::new(),
        sha256: String::new(),
        content: None,
    }];
    Ok(emit_program_with_map(prog, sources, "")?.0)
}

/// Emit the program and return the generated SLA together with
/// its source map (the in-memory segment table). The map's
/// `sources` / `generated` headers come from the caller (main.rs,
/// which owns the file paths and source bytes); the `segments`
/// table is populated by the SegmentWriter during emission and
/// finished here once emission is complete. `emit_program` stays
/// the plain-text entry point; `psa emit` output is byte-identical
/// either way.
pub fn emit_program_with_map(
    prog: &Program,
    sources: Vec<SourceEntry>,
    generated_path: &str,
) -> Result<(String, SourceMap), Diagnostic> {
    // D3: lambdas desugar into synthetic defs before anything
    // else runs; the analysis and emission passes never see a
    // Lambda node.
    let mut prog = prog.clone();
    desugar_program(&mut prog)?;
    let prog = &prog;
    let mut em = Emitter {
        defs: HashMap::new(),
        sigs: HashMap::new(),
        in_progress: Vec::new(),
        emitted: Vec::new(),
        emitted_set: Vec::new(),
        fn_ids: HashMap::new(),
        callables: Vec::new(),
        fn_cids: HashMap::new(),
        closure_callables: Vec::new(),
        used_closure: false,
        classes: Vec::new(),
        class_ids: HashMap::new(),
        sla_names: HashMap::new(),
        mod_attrs: prog.mod_attrs.iter().cloned().collect(),
        const_names: prog.const_names.iter().cloned().collect(),
        modules: prog.modules.clone(),
        sources: prog.sources.clone(),
        used_floordiv: false,
        used_floormod: false,
        used_vec: false,
        used_ipow: false,
        used_math: false,
        used_dyn: false,
        used_file: false,
        gen_names: Vec::new(),
        coro_names: Vec::new(),
        used_gen: false,
        used_coro: false,
        genexp_counter: 0,
        std_json: prog.std_imports.iter().any(|m| m == "json"),
        // psa_std: argparse's sys.argv path calls
        // psa_json_parse, and the assembly splice reads
        // used_psa_json (json block) before the argparse
        // block runs, so the dependency is primed here at
        // construction (psa-std/BATCH-WIRING.md, R5).
        used_psa_json: prog.std_imports.iter().any(|m| m == "argparse"),
        std_pickle: prog.std_imports.iter().any(|m| m == "pickle"),
        used_psa_pickle: false,
        std_math: prog.std_imports.iter().any(|m| m == "math"),
        used_psa_math: false,
        std_string: prog.std_imports.iter().any(|m| m == "string"),
        used_psa_string: false,
        std_csv: prog.std_imports.iter().any(|m| m == "csv"),
        used_psa_csv: false,
        std_logging: prog.std_imports.iter().any(|m| m == "logging"),
        used_psa_logging: false,
        std_argparse: prog.std_imports.iter().any(|m| m == "argparse"),
        used_psa_argparse: false,
        std_collections: prog.std_imports.iter().any(|m| m == "collections"),
        used_psa_collections: false,
        std_itertools: prog.std_imports.iter().any(|m| m == "itertools"),
        used_psa_itertools: false,
        std_functools: prog.std_imports.iter().any(|m| m == "functools"),
        used_psa_functools: false,
        std_bisect: prog.std_imports.iter().any(|m| m == "bisect"),
        used_psa_bisect: false,
        std_heapq: prog.std_imports.iter().any(|m| m == "heapq"),
        used_psa_heapq: false,
        std_base64: prog.std_imports.iter().any(|m| m == "base64"),
        used_psa_base64: false,
        std_random: prog.std_imports.iter().any(|m| m == "random"),
        used_psa_random: false,
        std_re: prog.std_imports.iter().any(|m| m == "re"),
        used_psa_re: false,
        std_socket: prog.std_imports.iter().any(|m| m == "socket"),
        used_psa_socket: false,
        std_urllib: prog.std_imports.iter().any(|m| m == "urllib"),
        used_psa_urllib: false,
        program_exc: false,
        loop_else_stack: Vec::new(),
    };
    em.program_exc = prog.funcs.iter().any(|f| stmts_use_exc(&f.body))
        || stmts_use_exc(&prog.top);
    for (i, f) in prog.funcs.iter().enumerate() {
        if em.defs.insert(f.name.clone(), f.clone()).is_some() {
            return Err(eerr(f.line, &format!("duplicate function definition `{}`", f.name)));
        }
        // D6: a merged module def's key is dotted (`foo.bar`),
        // which is not a valid SLA identifier; assign it an
        // id-based emitted name (the D4 method-name precedent).
        if f.name.contains('.') {
            em.sla_names.insert(f.name.clone(), format!("__py_mod_{}", i));
        }
    }
    // D6: the builtin exception classes are registered before
    // every user class (ids 0..=8), so user classes may derive
    // from them and handler matching is plain ancestry. They
    // have no methods and no attributes; construction is
    // special-cased in the emitter (__psa_exc_new).
    let mut prog = prog.clone();
    {
        let mut synth: Vec<ClassDef> = vec![ClassDef {
            name: "Exception".to_string(),
            base: None,
            methods: Vec::new(),
            attrs: Vec::new(),
            line: 0, anchor: SourceAnchor::synthetic(),
        }];
        for n in [
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "ZeroDivisionError",
            "RuntimeError",
            "NameError",
        ] {
            synth.push(ClassDef {
                name: n.to_string(),
                base: Some("Exception".to_string()),
                methods: Vec::new(),
                attrs: Vec::new(),
                line: 0, anchor: SourceAnchor::synthetic(),
            });
        }
        synth.extend(prog.classes.iter().cloned());
        prog.classes = synth;
    }
    let prog = &prog;
    // D4: register classes and their methods before any emission,
    // so construction and attribute paths resolve during main.
    build_class_registry(&mut em, prog)?;
    // D7: generator and async defs are not emitted on demand.
    // Their callable face is a hand-written wrapper installed
    // here, before any body emission (a generator wrapper spawns
    // the producer thread; an async wrapper packs a lazy
    // coroutine value), and the all-Dyn signature is
    // pre-registered so ensure_fn short-circuits. Class methods
    // may not be generators or async in v1 (loud).
    {
        let tops: Vec<Func> = prog.funcs.clone();
        // Pass 1: pre-register every generator / async def's
        // all-Dyn signature, so a producer or worker body that
        // calls a later sibling resolves to its wrapper face
        // instead of triggering on-demand body emission.
        for f in &tops {
            let is_gen = func_is_generator(f);
            if f.is_async && is_gen {
                return Err(eerr(
                    f.line,
                    &format!(
                        "async generator function `{}` is not supported in psa v1 (D7)",
                        f.name
                    ),
                ));
            }
            if is_gen {
                em.sigs.insert(
                    f.name.clone(),
                    Sig {
                        params: vec![Ty::Dyn; f.param_count()],
                        param_borrow: vec![false; f.param_count()],
                        ret: Some(Ty::Dyn),
                    },
                );
                em.gen_names.push(f.name.clone());
            } else if f.is_async {
                em.sigs.insert(
                    f.name.clone(),
                    Sig {
                        params: vec![Ty::Dyn; f.param_count()],
                        param_borrow: vec![false; f.param_count()],
                        ret: Some(Ty::Dyn),
                    },
                );
                em.coro_names.push(f.name.clone());
            }
        }
        // Pass 2: emit the wrappers, producers and workers.
        for f in &tops {
            if func_is_generator(f) {
                em.emit_generator_def(f)?;
            } else if f.is_async {
                em.emit_async_def(f)?;
            }
        }
        let top_names: std::collections::HashSet<&str> =
            prog.funcs.iter().map(|f| f.name.as_str()).collect();
        for (key, d) in &em.defs {
            if !top_names.contains(key.as_str()) && (d.is_async || func_is_generator(d)) {
                return Err(eerr(
                    d.line,
                    &format!(
                        "method `{}` may not be a generator or async in psa v1 (D7)",
                        key
                    ),
                ));
            }
        }
    }
    // Emit user functions on demand (driven by calls from main / each
    // other), then main from the top-level statements.
    let mut main_ctx = Ctx {
        vars: Vec::new(),
        sources: prog.sources.clone(),
        scopes: vec![0],
        ret: Some(Ty::Void),
        fname: "main".to_string(),
        out: SegmentWriter::new(),
        prologue: SegmentWriter::new(),
        indent: 1,
        tmp: 0,
        borrowed: Vec::new(),
        in_while_cond: false,
        dyn_ret: false,
                deleted: Vec::new(),
        try_stack: Vec::new(),
        guard_depth: 0,
        loop_depth: 0,
        exc_slot: false,
        in_finally: 0,
        exc_bare: None,
        gen_chan: None,
        anchor: SourceAnchor::synthetic(),
        anchor_kind: SegKind::Stmt,
        anchor_precision: Precision::None,
        anchor_name: None,
        anchor_origin: Origin::Synthetic,
        pending_call: false,
        pending_call_name: None,
    };
    // D4 runtime registry: seed every class's literal
    // attributes into its registry dict before any user code
    // runs (CPython evaluates class bodies at definition
    // time; program start is psa's equivalent point).
    if !em.classes.is_empty() {
        // P1: the registry-seed call is synthetic runtime wiring,
        // not a user statement (no user line anchors it).
        main_ctx.set_anchor(
            SourceAnchor::synthetic(),
            SegKind::CallSite,
            Precision::None,
            Some("__psa_class_attrs_seed".to_string()),
        );
        main_ctx.line("__psa_class_attrs_seed();");
    }
    for s in &prog.top {
        em.emit_stmt(&mut main_ctx, s)?;
    }
    // P1: main's fallthrough return is psa-added, anchored to the
    // last top-level statement (definition-site precision).
    if let Some(last) = prog.top.last() {
        main_ctx.set_anchor(stmt_anchor(last), SegKind::Stmt, Precision::Function, None);
        main_ctx.set_anchor_origin(Origin::Synthetic);
    }
    main_ctx.line("return 0;");
    // P1: main's text is a segment writer: header (anchored to the
    // first top-level statement), prologue, body, closer.
    let main_anchor = prog.top.first().map(stmt_anchor).unwrap_or_else(SourceAnchor::synthetic);
    let mut main_txt = SegmentWriter::new();
    main_txt.push_line("fn main() -> i32 {", &SegMeta::fn_header(main_anchor, "main".to_string()));
    main_txt.append_writer(std::mem::take(&mut main_ctx.prologue));
    main_txt.append_writer(std::mem::take(&mut main_ctx.out));
    main_txt.push_line("}", &SegMeta::fn_header(main_anchor, "main".to_string()));

    // D4: pin every method of every class as a callable -- the
    // per-class getattr generators below bind methods by id.
    // Defining a class therefore emits all of its methods, even
    // unused ones (a v1 tightening in CPython's whole-module
    // compile spirit, and loud rather than silent).
    {
        let mut pins: Vec<(String, u32)> = Vec::new();
        for ci in &em.classes {
            for (_, key) in &ci.flat_methods {
                let l = em.defs.get(key).map(|d| d.line).unwrap_or(0);
                pins.push((key.clone(), l));
            }
        }
        for (key, l) in &pins {
            em.method_callable_id(key, *l)?;
        }
    }

    // The dispatch and the object machinery speak PsaVal
    // throughout; if emission registered callables or classes
    // without ever touching a dynamic value (a kwargs-only def
    // suite, say), the dynamic runtime and the PsaVal struct
    // must still be assembled, or the checker meets field
    // accesses on an unknown struct.
    if !em.callables.is_empty() || !em.classes.is_empty() || !em.closure_callables.is_empty() {
        em.used_dyn = true;
    }

    // D3 function values + D4 bound methods: generate the call
    // dispatch and one adapter per callable function, from the
    // registry built during emission. The dispatch splits into
    // __psa_call_value (tag check: 7 = plain callable, 9 = bound
    // method) and __psa_call_id (per-id adapter dispatch), so a
    // bound method can prepend its receiver and reuse it.
    // D8: when the program has closure callables, the dispatch
    // gains a tag-16 branch and a parallel closure dispatch
    // (__psa_call_closure_id + __psa_cadapt_<j>, whose adapters
    // pass the environment as the callee's first argument);
    // without closures the generated text is unchanged.
    if !em.callables.is_empty() || !em.classes.is_empty() || !em.closure_callables.is_empty() {
        let mut disp = String::from(
            if em.closure_callables.is_empty() {
                "fn __psa_call_value(f: PsaVal, args: PsaVal) -> PsaVal {\n    if f.tag == 9 { return __psa_call_bound(f, args); };\n    if f.tag != 7 { panic(77067); };\n    return __psa_call_id(f.ipay, args);\n}\n"
            } else {
                "fn __psa_call_value(f: PsaVal, args: PsaVal) -> PsaVal {\n    if f.tag == 9 { return __psa_call_bound(f, args); };\n    if f.tag == 16 { return __psa_call_closure(f, args); };\n    if f.tag != 7 { panic(77067); };\n    return __psa_call_id(f.ipay, args);\n}\n"
            },
        );
        disp.push_str("fn __psa_call_id(id: i64, args: PsaVal) -> PsaVal {\n");
        for i in 0..em.callables.len() {
            disp.push_str(&format!(
                "    if id == {} {{ return __psa_adapt_{}(args); }};\n",
                i, i
            ));
        }
        disp.push_str("    panic(77067);\n}\n");
        disp.push_str(
            "fn __psa_call_bound(f: PsaVal, args: PsaVal) -> PsaVal {\n    let mid = __psa_method_id(f);\n    let mself = __psa_method_self(f);\n    let a2 = __psa_list_new();\n    __psa_list_push(a2, mself);\n    let n = __psa_list_len(args);\n    let i: i64 = 0;\n    while i < n {\n        __psa_list_push(a2, __psa_list_get(args, i));\n        i = i + 1;\n    }\n    return __psa_call_id(mid, a2);\n}\n",
        );
        // P1: the dispatch is one synthetic multi-function chunk
        // (no single def anchors it).
        em.emitted.push(EmittedChunk::span(disp, &SegMeta::synthetic_unanchored(SegKind::Stmt)));
        for (i, name) in em.callables.iter().enumerate() {
            let n = em
                .defs
                .get(name.as_str())
                .expect("a callable is always a def")
                .param_count();
            let mut call = format!("{}(", em.sla_name(name));
            for j in 0..n {
                if j > 0 {
                    call.push_str(", ");
                }
                call.push_str(&format!("__psa_list_get(args, {})", j));
            }
            call.push(')');
            // Adapters normalise the callee's return to a dynamic
            // value: dynamic returns pass through, void becomes
            // None, static scalars box (methods may be pinned
            // with any of these; plain function values are
            // dynamic-only, enforced by fn_val_code).
            let ret = em.sigs.get(name.as_str()).and_then(|s| s.ret);
            let body = match ret {
                Some(Ty::Void) => format!("    {};\n    return {};\n", call, Ty::Dyn.default_lit()),
                Some(Ty::Int) => format!("    return __psa_box_int({});\n", call),
                Some(Ty::Float) => format!("    return __psa_box_float({});\n", call),
                Some(Ty::Bool) => format!("    return __psa_box_bool({});\n", call),
                _ => format!("    return {};\n", call),
            };
            let ad = format!(
                "fn __psa_adapt_{}(args: PsaVal) -> PsaVal {{\n    if __psa_list_len(args) != {} {{ panic(77068); }};\n{}}}\n",
                i, n, body
            );
            // P1: one synthetic function chunk per adapter, anchored
            // to the callable def it adapts (precision Function).
            let ad_anchor = em
                .defs
                .get(name.as_str())
                .expect("a callable is always a def")
                .anchor;
            em.emitted.push(EmittedChunk::span(
                ad,
                &SegMeta::synthetic_fn(ad_anchor, format!("__psa_adapt_{}", i)),
            ));
        }
        // D8: the closure dispatch and one adapter per
        // closure callable. The adapter passes the captured
        // environment as the callee's first argument; the
        // arity check counts only the user-visible
        // parameters (param_count includes the env param).
        if !em.closure_callables.is_empty() {
            let mut cdisp = String::from(
                "fn __psa_call_closure(f: PsaVal, args: PsaVal) -> PsaVal {\n    return __psa_call_closure_id(__psa_closure_id(f), __psa_closure_env(f), args);\n}\n",
            );
            cdisp.push_str("fn __psa_call_closure_id(id: i64, env: PsaVal, args: PsaVal) -> PsaVal {\n");
            for j in 0..em.closure_callables.len() {
                cdisp.push_str(&format!(
                    "    if id == {} {{ return __psa_cadapt_{}(env, args); }};\n",
                    j, j
                ));
            }
            cdisp.push_str("    panic(77067);\n}\n");
            // P1: the closure dispatch is one synthetic
            // multi-function chunk (no single def anchors it).
            em.emitted.push(EmittedChunk::span(cdisp, &SegMeta::synthetic_unanchored(SegKind::Stmt)));
            for (j, name) in em.closure_callables.iter().enumerate() {
                let n = em
                    .defs
                    .get(name.as_str())
                    .expect("a closure callable is always a def")
                    .param_count();
                let mut call = format!("{}(env", em.sla_name(name));
                for k in 0..n - 1 {
                    call.push_str(&format!(", __psa_list_get(args, {})", k));
                }
                call.push(')');
                // Return normalisation mirrors the plain
                // adapters (see above).
                let ret = em.sigs.get(name.as_str()).and_then(|s| s.ret);
                let body = match ret {
                    Some(Ty::Void) => format!("    {};\n    return {};\n", call, Ty::Dyn.default_lit()),
                    Some(Ty::Int) => format!("    return __psa_box_int({});\n", call),
                    Some(Ty::Float) => format!("    return __psa_box_float({});\n", call),
                    Some(Ty::Bool) => format!("    return __psa_box_bool({});\n", call),
                    _ => format!("    return {};\n", call),
                };
                let ad = format!(
                    "fn __psa_cadapt_{}(env: PsaVal, args: PsaVal) -> PsaVal {{\n    if __psa_list_len(args) != {} {{ panic(77068); }};\n{}}}\n",
                    j,
                    n - 1,
                    body
                );
                // P1: one synthetic function chunk per closure
                // adapter, anchored to the callable def it adapts.
                let ad_anchor = em
                    .defs
                    .get(name.as_str())
                    .expect("a closure callable is always a def")
                    .anchor;
                em.emitted.push(EmittedChunk::span(
                    ad,
                    &SegMeta::synthetic_fn(ad_anchor, format!("__psa_cadapt_{}", j)),
                ));
            }
        }
    }

    // D4: the generated object machinery (getattr dispatch,
    // per-class getattr, object str/repr/truthiness). The static
    // runtime text references these whenever dynamic values are
    // in play, so they are generated for every dynamic program,
    // with panic-only bodies when no class exists (tag 8 values
    // cannot arise then).
    if em.used_dyn {
        let machinery = gen_obj_machinery(&em);
        // P1: the object machinery is one synthetic chunk (its
        // lines serve every class; no single def anchors it).
        em.emitted.push(EmittedChunk::span(machinery, &SegMeta::synthetic_unanchored(SegKind::Stmt)));
    }

    // P1 source map: the assembly buffer is a segment writer; every
    // push below records the segments for the lines it appends, so
    // the final table covers the whole file. The text appended is
    // unchanged (byte-identical red line). The writer is seeded
    // with the caller's user sources (main first, merged modules
    // after); the prelude entry lands after them.
    let mut asm = SegmentWriter::with_sources(sources);
    let prelude_meta = SegMeta::prelude(SegKind::Stmt);
    asm.push_raw("@import \"sa_std/io/print.sai\"\n@import \"sa_std/fmt.sai\"\n", &prelude_meta);
    if em.used_vec || em.used_dyn {
        asm.push_raw("@import \"sa_std/vec.sa\"\n", &prelude_meta);
    }
    if em.used_dyn {
        asm.push_raw("@import \"sa_std/globals.sai\"\n", &prelude_meta);
        asm.push_raw("@import \"sa_std/core/box.sa\"\n", &prelude_meta);
        asm.push_raw("@import \"sa_std/hashmap.sa\"\n", &prelude_meta);
        asm.push_raw("@import \"sa_std/core/option.sa\"\n", &prelude_meta);
        asm.push_raw("@import \"sa_std/string.sa\"\n", &prelude_meta);
        asm.push_raw("@import \"sa_std/ptr.sa\"\n", &prelude_meta);
        asm.push_raw("@import \"sla_std/wrap/io.sla\"\n", &prelude_meta);
        asm.push_raw("@import \"sla_std/wrap/fmt.sla\"\n", &prelude_meta);
        asm.push_raw("@import \"sla_std/bigint.sla\"\n", &prelude_meta);
    }
    if em.used_math || em.used_dyn {
        asm.push_raw("@import \"sa_std/math.sai\"\n", &prelude_meta);
    }
    if em.used_file {
        asm.push_raw("@import \"sa_std/fs.sai\"\n", &prelude_meta);
    }
    if em.used_gen {
        asm.push_raw("@import \"sa_std/task.sai\"\n", &prelude_meta);
        asm.push_raw("@import \"sa_std/thread.sa\"\n", &prelude_meta);
        asm.push_raw("@import \"sa_std/core/result.sa\"\n", &prelude_meta);
    }
    if em.std_random {
        // psa_std: random.sla calls sa_crypto_random_bytes
        // for entropy seeding; emitted whenever `import
        // random` bound to the standard library, regardless
        // of used_dyn (WIRING-random.md).
        asm.push_raw("@import \"sa_std/crypto.sai\"\n", &prelude_meta);
    }
    asm.push_raw("\n", &prelude_meta);
    if em.used_dyn {
        asm.push_raw("@derive(copy)\nstruct PsaVal { tag: i64, ipay: i64, fpay: f64 }\n\nstruct PsaStage { tag: i64, ipay: i64, fpay: f64 }\n\n", &prelude_meta);
    }
    if em.used_ipow || em.used_dyn {
        asm.push_raw(
            "fn __psa_ipow(a: i64, e: i64) -> i64 {\n    let r = 1;\n    let b = a;\n    let n = e;\n    while n > 0 {\n        if (n & 1) == 1 { r = r * b; };\n        b = b * b;\n        n = n >> 1;\n    }\n    return r;\n}\n\n",
            &SegMeta::prelude_fn("__psa_ipow"),
        );
    }
    if em.used_floordiv || em.used_dyn {
        asm.push_raw(
            "fn __psa_floordiv(a: i64, b: i64) -> i64 {\n    if a == -9223372036854775807 - 1 && b == -1 { panic(77080); };\n    let q = a / b;\n    let r = a % b;\n    if r != 0 && ((r < 0) != (b < 0)) { return q - 1; };\n    return q;\n}\n\n",
            &SegMeta::prelude_fn("__psa_floordiv"),
        );
    }
    if em.used_floormod || em.used_dyn {
        asm.push_raw(
            "fn __psa_floormod(a: i64, b: i64) -> i64 {\n    if a == -9223372036854775807 - 1 && b == -1 { return 0; };\n    let r = a % b;\n    if r != 0 && ((r < 0) != (b < 0)) { return r + b; };\n    return r;\n}\n\n",
            &SegMeta::prelude_fn("__psa_floormod"),
        );
    }
    if em.used_dyn {
        asm.push_raw(DYN_RUNTIME, &prelude_meta);
        asm.push_raw(BIG_RUNTIME, &prelude_meta);
        asm.push_raw(D6_EXC_RUNTIME, &prelude_meta);
    }
    if em.used_closure {
        asm.push_raw(CLOSURE_RUNTIME, &prelude_meta);
    }
    if em.used_file {
        asm.push_raw(D6_FILE_RUNTIME, &prelude_meta);
    }
    if em.used_dyn && !em.used_gen {
        // psa_std collections / itertools / functools call
        // __psa_iter_source unconditionally; the draining
        // definition lives in D7_GEN_RUNTIME. In a program
        // with no generators no tag-13 value can be
        // constructed (generator defs/exprs are the only
        // source, and no std module creates one), so the
        // identity is the same function here.
        asm.push_raw("fn __psa_iter_source(v: PsaVal) -> PsaVal {
    return v;
}

", &SegMeta::prelude_fn("__psa_iter_source"));
    }
    if em.used_gen {
        asm.push_raw(D7_GEN_RUNTIME, &prelude_meta);
    }
    if em.used_coro {
        asm.push_raw(D7_CORO_RUNTIME, &prelude_meta);
    }
    if em.used_psa_json {
        let src = asm.add_source("psa-std/json.sla".to_string(), sha256_hex(PSA_JSON_SLA.as_bytes()));
        asm.push_std_splice(PSA_JSON_SLA, "json", src);
    }
    if em.used_psa_pickle {
        let src = asm.add_source("psa-std/pickle.sla".to_string(), sha256_hex(PSA_PICKLE_SLA.as_bytes()));
        asm.push_std_splice(PSA_PICKLE_SLA, "pickle", src);
    }
    if em.used_psa_math {
        let src = asm.add_source("psa-std/math.sla".to_string(), sha256_hex(PSA_MATH_SLA.as_bytes()));
        asm.push_std_splice(PSA_MATH_SLA, "math", src);
    }
    if em.used_psa_string {
        let src = asm.add_source("psa-std/string.sla".to_string(), sha256_hex(PSA_STRING_SLA.as_bytes()));
        asm.push_std_splice(PSA_STRING_SLA, "string", src);
    }
    if em.used_psa_csv {
        let src = asm.add_source("psa-std/csv.sla".to_string(), sha256_hex(PSA_CSV_SLA.as_bytes()));
        asm.push_std_splice(PSA_CSV_SLA, "csv", src);
    }
    if em.used_psa_logging {
        let src = asm.add_source("psa-std/logging.sla".to_string(), sha256_hex(PSA_LOGGING_SLA.as_bytes()));
        asm.push_std_splice(PSA_LOGGING_SLA, "logging", src);
    }
    if em.used_psa_argparse {
        let src = asm.add_source("psa-std/argparse.sla".to_string(), sha256_hex(PSA_ARGPARSE_SLA.as_bytes()));
        asm.push_std_splice(PSA_ARGPARSE_SLA, "argparse", src);
    }
    if em.used_psa_collections {
        let src = asm.add_source("psa-std/collections.sla".to_string(), sha256_hex(PSA_COLLECTIONS_SLA.as_bytes()));
        asm.push_std_splice(PSA_COLLECTIONS_SLA, "collections", src);
    }
    if em.used_psa_itertools {
        let src = asm.add_source("psa-std/itertools.sla".to_string(), sha256_hex(PSA_ITERTOOLS_SLA.as_bytes()));
        asm.push_std_splice(PSA_ITERTOOLS_SLA, "itertools", src);
    }
    if em.used_psa_functools {
        let src = asm.add_source("psa-std/functools.sla".to_string(), sha256_hex(PSA_FUNCTOOLS_SLA.as_bytes()));
        asm.push_std_splice(PSA_FUNCTOOLS_SLA, "functools", src);
    }
    if em.used_psa_bisect {
        let src = asm.add_source("psa-std/bisect.sla".to_string(), sha256_hex(PSA_BISECT_SLA.as_bytes()));
        asm.push_std_splice(PSA_BISECT_SLA, "bisect", src);
    }
    if em.used_psa_heapq {
        let src = asm.add_source("psa-std/heapq.sla".to_string(), sha256_hex(PSA_HEAPQ_SLA.as_bytes()));
        asm.push_std_splice(PSA_HEAPQ_SLA, "heapq", src);
    }
    if em.used_psa_base64 {
        let src = asm.add_source("psa-std/base64.sla".to_string(), sha256_hex(PSA_BASE64_SLA.as_bytes()));
        asm.push_std_splice(PSA_BASE64_SLA, "base64", src);
    }
    if em.used_psa_random {
        let src = asm.add_source("psa-std/random.sla".to_string(), sha256_hex(PSA_RANDOM_SLA.as_bytes()));
        asm.push_std_splice(PSA_RANDOM_SLA, "random", src);
    }
    if em.used_psa_re {
        let src = asm.add_source("psa-std/re.sla".to_string(), sha256_hex(PSA_RE_SLA.as_bytes()));
        asm.push_std_splice(PSA_RE_SLA, "re", src);
    }
    if em.used_psa_socket {
        let src = asm.add_source("psa-std/socket.sla".to_string(), sha256_hex(PSA_SOCKET_SLA.as_bytes()));
        asm.push_std_splice(PSA_SOCKET_SLA, "socket", src);
    }
    if em.used_psa_urllib {
        let src = asm.add_source("psa-std/urllib.sla".to_string(), sha256_hex(PSA_URLLIB_SLA.as_bytes()));
        asm.push_std_splice(PSA_URLLIB_SLA, "urllib", src);
    }
    for chunk in &em.emitted {
        asm.append_chunk(chunk);
    }
    asm.append_writer(main_txt);
    let (text, mut map) = asm.into_text_and_map(build_name_bridge(&em));
    // P1 red line: the table must cover every generated line, or
    // emission fails loudly as an internal error (never silently).
    map.assert_full_coverage(&text)?;
    // P2: hash the generated text, sort the segments, remap the
    // panic-site indices, build the lookup indexes.
    map.finish(&text, generated_path.to_string());
    Ok((text, map))
}

/// P1 source map: the anchor of a statement — its psa line, on the
/// main source. This is the single seam P0 touches: when `Stmt`
/// carries a `SourceAnchor` (design §3.2), this function returns
/// it (with its column and multi-file source) instead of rebuilding
/// an anchor from the line. Every `Stmt` variant carries a line;
/// `Def` carries its `Func`'s line.
/// P1 source map: the anchor of a statement — its psa position,
/// read off the statement itself (P0b `Stmt::anchor`, post-rewrite
/// spans per P0d: the column and multi-file source travel with the
/// statement through the closure rewrite). `Def` carries its `Func`'s
/// anchor the same way.
fn stmt_anchor(stmt: &Stmt) -> SourceAnchor {
    stmt.anchor()
}

/// P0b span carriers: only `Name` / `Call` / `Index` / `Attr` carry
/// their own span; every other variant falls back to the enclosing
/// statement's anchor (see the `Expr` docs in ast.rs). Returns the
/// span's start anchor, or `None`.
fn expr_span_start(e: &Expr) -> Option<SourceAnchor> {
    match e {
        Expr::Name(_, s) | Expr::Call(_, _, _, s) | Expr::Index(_, _, s) | Expr::Attr(_, _, s) => {
            Some(s.start)
        }
        _ => None,
    }
}

/// P1 source map: build the name bridge (design §3.1) from the
/// emitter's registries — every generated function symbol that
/// resolves to a psa definition. Plain defs resolve through
/// `sla_name` (dotted module defs are `__py_mod_<i>` and methods
/// `__py_cls_<id>_<m>` via `sla_names`); adapters resolve by their
/// registry index (the same index the dispatch uses); generator
/// and async wrappers resolve to the def they shell. Sorted by
/// generated name, deduplicated: the bridge is deterministic
/// (the defs table itself iterates in hash order).
fn build_name_bridge(em: &Emitter) -> Vec<NameBinding> {
    let mut bridge: Vec<NameBinding> = Vec::new();
    for (name, def) in &em.defs {
        bridge.push(NameBinding {
            generated: em.sla_name(name),
            psa_name: name.clone(),
            def_line: def.line,
            source: SRC_MAIN,
        });
    }
    for (i, name) in em.callables.iter().enumerate() {
        if let Some(def) = em.defs.get(name) {
            bridge.push(NameBinding {
                generated: format!("__psa_adapt_{}", i),
                psa_name: name.clone(),
                def_line: def.line,
                source: SRC_MAIN,
            });
        }
    }
    for (j, name) in em.closure_callables.iter().enumerate() {
        if let Some(def) = em.defs.get(name) {
            bridge.push(NameBinding {
                generated: format!("__psa_cadapt_{}", j),
                psa_name: name.clone(),
                def_line: def.line,
                source: SRC_MAIN,
            });
        }
    }
    for name in &em.gen_names {
        if let Some(def) = em.defs.get(name) {
            let base = em.sla_name(name);
            bridge.push(NameBinding {
                generated: format!("{}__prod", base),
                psa_name: name.clone(),
                def_line: def.line,
                source: SRC_MAIN,
            });
            bridge.push(NameBinding {
                generated: format!("{}__drive", base),
                psa_name: name.clone(),
                def_line: def.line,
                source: SRC_MAIN,
            });
        }
    }
    for name in &em.coro_names {
        if let Some(def) = em.defs.get(name) {
            bridge.push(NameBinding {
                generated: format!("{}__coro", em.sla_name(name)),
                psa_name: name.clone(),
                def_line: def.line,
                source: SRC_MAIN,
            });
        }
    }
    bridge.sort_by(|a, b| a.generated.cmp(&b.generated));
    bridge.dedup_by(|a, b| a.generated == b.generated);
    bridge
}

/// Build a dynamic string from a compile-time literal inside a
/// generated runtime function: the same byte-by-byte shape
/// emit_str_temp produces in user code.
fn sla_str_build(var: &str, s: &str) -> String {
    let mut t = format!("    let {} = __psa_str_new();\n", var);
    for b in s.bytes() {
        t.push_str(&format!("    {} = __psa_str_push({}, {}u8);\n", var, var, b));
    }
    t
}

/// D4 generated text: one class's class-side getattr. Method
/// names resolve to bound methods (tag 9); class attributes
/// resolve through the runtime registry (per-class dicts
/// seeded at program start, walked over the ancestor chain
/// by __psa_class_<id>_cgetattr); anything else is an
/// AttributeError (panic 77069).
fn gen_class_getattr(em: &Emitter, ci: &ClassInfo) -> String {
    let mut t = format!(
        "fn __psa_class_{}_getattr(inst: PsaVal, name: PsaVal) -> PsaVal {{\n",
        ci.id
    );
    for (i, (mname, _)) in ci.flat_methods.iter().enumerate() {
        t.push_str(&sla_str_build(&format!("nm_{}", i), mname));
    }
    for (i, (_, key)) in ci.flat_methods.iter().enumerate() {
        let id = em.fn_ids.get(key).copied().unwrap_or(-1);
        t.push_str(&format!(
            "    if __psa_str_eq(name, nm_{}) {{ return __psa_box_method({}, inst); }};\n",
            i, id
        ));
    }
    t.push_str(&format!(
        "    return __psa_class_{}_cgetattr(name);\n}}\n",
        ci.id
    ));
    t
}

/// D4 generated text: one class's attribute-only lookup over
/// the runtime registry: its own dict first, then each
/// ancestor's, in flattened order. Seeded at program start
/// by __psa_class_attrs_seed; assignments and deletions at
/// runtime mutate the dicts, so reads observe them.
fn gen_class_cgetattr(ci: &ClassInfo) -> String {
    let mut t = format!(
        "fn __psa_class_{}_cgetattr(name: PsaVal) -> PsaVal {{\n",
        ci.id
    );
    for aid in &ci.ancestors {
        t.push_str(&format!(
            "    let __d{} = __psa_class_attrs_raw({});\n    if __psa_dict_find(__d{}, name) >= 0 {{ return __psa_dict_get(__d{}, name); }};\n",
            aid, aid, aid, aid
        ));
    }
    t.push_str("    panic(77069);\n}\n");
    t
}

/// D4 generated text: the class-attribute registry root and
/// per-class dict accessor, plus the program-start seed that
/// writes every class's own literal attributes into its
/// dict. The registry root is a tag-4 list (indexed by class
/// id) whose box pointer lives in the sa_std globals cell
/// "psa.clsattrs" (the mutable-global shape proven by
/// probe_globals); each slot holds that class's tag-5 dict.
fn gen_class_registry(em: &Emitter) -> String {
    let mut t = String::new();
    t.push_str("fn __psa_clsreg() -> PsaVal {\n    let out = [0u64; 1];\n    let rc = sa_global_u64_get(STR_PTR(\"psa.clsattrs\"), STR_LEN(\"psa.clsattrs\"), &out[0]);\n    if rc == 0 { return PsaVal { tag: 4, ipay: out[0] as i64, fpay: 0.0 }; };\n    let reg = __psa_list_new();\n    let already = [0u64; 1];\n    let rc2 = sa_global_u64_set_once(STR_PTR(\"psa.clsattrs\"), STR_LEN(\"psa.clsattrs\"), reg.ipay as u64, &already[0]);\n    if rc2 == 0 {\n        if already[0] == 0 { return reg; };\n    };\n    let out2 = [0u64; 1];\n    let rc3 = sa_global_u64_get(STR_PTR(\"psa.clsattrs\"), STR_LEN(\"psa.clsattrs\"), &out2[0]);\n    if rc3 != 0 { return reg; };\n    return PsaVal { tag: 4, ipay: out2[0] as i64, fpay: 0.0 };\n}\n");
    t.push_str("fn __psa_class_attrs_raw(cid: i64) -> PsaVal {\n    let reg = __psa_clsreg();\n    while __psa_vec_len_h(reg.ipay as ptr) <= cid { __psa_list_push(reg, PsaVal { tag: 3, ipay: 0, fpay: 0.0 }); }\n    let slot = __psa_vec_get_h(reg.ipay as ptr, cid);\n    if slot.tag == 3 {\n        let d = __psa_dict_new();\n        __psa_vec_set_h(reg.ipay as ptr, cid, d);\n        return d;\n    };\n    return slot;\n}\n");
    t.push_str("fn __psa_class_attrs_seed() {\n");
    for ci in &em.classes {
        if ci.attrs.is_empty() {
            continue;
        }
        t.push_str(&format!(
            "    let __sd{} = __psa_class_attrs_raw({});\n",
            ci.id, ci.id
        ));
        for (i, (aname, ae)) in ci.attrs.iter().enumerate() {
            t.push_str(&sla_str_build(&format!("sk_{}_{}", ci.id, i), aname));
            if let Expr::Str(s) = ae {
                t.push_str(&sla_str_build(&format!("sv_{}_{}", ci.id, i), s));
            }
            let rc = match ae {
                Expr::Int(v) => format!("__psa_box_int({})", v),
                Expr::BigInt(d) => {
                    t.push_str(&sla_str_build(&format!("sv_{}_{}", ci.id, i), d));
                    format!("__psa_big_from_psastr(sv_{}_{})", ci.id, i)
                }
                Expr::Float(v) => format!("__psa_box_float({})", fmt_f64(*v)),
                Expr::Bool(b) => format!("__psa_box_bool({})", b),
                Expr::Str(_) => format!("sv_{}_{}", ci.id, i),
                _ => Ty::Dyn.default_lit().to_string(),
            };
            t.push_str(&format!(
                "    __psa_dict_set(__sd{}, sk_{}_{}, {});\n",
                ci.id, ci.id, i, rc
            ));
        }
    }
    t.push_str("}\n");
    t
}

/// D4 generated text: the object machinery the static runtime
/// references -- getattr (instance entries, then the class
/// side), the per-class getattr functions, and the object
/// str/repr/truthiness dispatchers honouring __str__, __repr__
/// and __bool__. With no classes registered the bodies degrade
/// to their unreachable defaults.
fn gen_obj_machinery(em: &Emitter) -> String {
    let mut t = String::new();
    t.push_str("fn __psa_getattr(obj: PsaVal, name: PsaVal) -> PsaVal {\n    if obj.tag != 8 { panic(77069); };\n    let j = __psa_obj_find(obj, name);\n    if j >= 0 { return __psa_vec_get_h(obj.ipay as ptr, j + 1); };\n    return __psa_class_getattr(obj, name);\n}\n");
    t.push_str("fn __psa_class_getattr(obj: PsaVal, name: PsaVal) -> PsaVal {\n    let cid = __psa_obj_class(obj);\n");
    for ci in &em.classes {
        t.push_str(&format!(
            "    if cid == {} {{ return __psa_class_{}_getattr(obj, name); }};\n",
            ci.id, ci.id
        ));
    }
    t.push_str("    panic(77069);\n}\n");
    for ci in &em.classes {
        t.push_str(&gen_class_getattr(em, ci));
        t.push_str(&gen_class_cgetattr(ci));
    }
    if !em.classes.is_empty() {
        t.push_str(&gen_class_registry(em));
    }
    // The default object string: `<Name object>` (no address --
    // CPython's address form is non-deterministic).
    t.push_str("fn __psa_obj_default_str(v: PsaVal) -> PsaVal {\n    let cid = __psa_obj_class(v);\n");
    for ci in &em.classes {
        t.push_str(&format!("    if cid == {} {{\n", ci.id));
        t.push_str(&sla_str_build("s", &format!("<{} object>", ci.name)));
        t.push_str("        return s;\n    };\n");
    }
    t.push_str(&sla_str_build("s", "<object>"));
    t.push_str("    return s;\n}\n");
    // repr uses __repr__ only; str uses __str__, then __repr__.
    let plans: [(&str, &[&str]); 2] = [
        ("__psa_obj_repr", &["__repr__"]),
        ("__psa_obj_str", &["__str__", "__repr__"]),
    ];
    for (gname, prefs) in plans {
        let mut body = format!("fn {}(v: PsaVal) -> PsaVal {{\n    let cid = __psa_obj_class(v);\n", gname);
        let mut used: Vec<&str> = Vec::new();
        let mut picks: Vec<Option<&str>> = Vec::new();
        for ci in &em.classes {
            let pick = prefs
                .iter()
                .find(|p| ci.flat_methods.iter().any(|(n, _)| n.as_str() == **p))
                .copied();
            picks.push(pick);
            if let Some(p) = pick {
                if !used.contains(&p) {
                    used.push(p);
                }
            }
        }
        for (i, u) in used.iter().enumerate() {
            body.push_str(&sla_str_build(&format!("dn_{}", i), u));
        }
        for (ci, pick) in em.classes.iter().zip(picks.iter()) {
            if let Some(p) = pick {
                let ti = used.iter().position(|u| *u == *p).unwrap();
                body.push_str(&format!(
                    "    if cid == {} {{\n        let m = __psa_class_{}_getattr(v, dn_{});\n        let r = __psa_call_value(m, __psa_list_new());\n        if r.tag == 6 {{ return r; }};\n        panic(77030);\n    }};\n",
                    ci.id, ci.id, ti
                ));
            } else if ci.ancestors.contains(&0) {
                // D6: an exception class without its own dunder
                // renders from its stored message.
                let helper = if gname == "__psa_obj_repr" {
                    "__psa_exc_repr"
                } else {
                    "__psa_exc_str"
                };
                body.push_str(&format!(
                    "    if cid == {} {{ return {}(v); }};\n",
                    ci.id, helper
                ));
            }
        }
        body.push_str("    return __psa_obj_default_str(v);\n}\n");
        t.push_str(&body);
    }
    // Truthiness: __bool__ when the class defines it, else true.
    let mut body =
        String::from("fn __psa_obj_truthy(v: PsaVal) -> bool {\n    let cid = __psa_obj_class(v);\n");
    let bool_classes: Vec<&ClassInfo> = em
        .classes
        .iter()
        .filter(|ci| ci.flat_methods.iter().any(|(n, _)| n.as_str() == "__bool__"))
        .collect();
    if !bool_classes.is_empty() {
        body.push_str(&sla_str_build("nb", "__bool__"));
        for ci in bool_classes {
            body.push_str(&format!(
                "    if cid == {} {{\n        let m = __psa_class_{}_getattr(v, nb);\n        let r = __psa_call_value(m, __psa_list_new());\n        if r.tag == 2 {{ return r.ipay != 0; }};\n        if r.tag == 0 {{ return r.ipay != 0; }};\n        panic(77010);\n    }};\n",
                ci.id, ci.id
            ));
        }
    }
    body.push_str("    return true;\n}\n");
    t.push_str(&body);
    // The class name as a dynamic string (D5, backing type()).
    let mut body =
        String::from("fn __psa_class_name(v: PsaVal) -> PsaVal {\n    let cid = __psa_obj_class(v);\n");
    for ci in &em.classes {
        body.push_str(&format!("    if cid == {} {{\n", ci.id));
        body.push_str(&sla_str_build("s", &ci.name));
        body.push_str("        return s;\n    };\n");
    }
    body.push_str(&sla_str_build("s", "object"));
    body.push_str("    return s;\n}\n");
    t.push_str(&body);
    // D6: which registry classes derive from Exception (id 0).
    let mut body = String::from("fn __psa_is_exc_class(cid: i64) -> bool {\n");
    for ci in &em.classes {
        if ci.ancestors.contains(&0) {
            body.push_str(&format!("    if cid == {} {{ return true; }};\n", ci.id));
        }
    }
    body.push_str("    return false;\n}\n");
    t.push_str(&body);
    t
}

/// The SLA name a user-defined Python function is emitted under.
/// User functions are mangled with a `__py_` prefix (D3): a user
/// function named like a std method the runtime calls (`get`,
/// `insert`, `push`, ...) otherwise hijacks the SLA checker's
/// method resolution -- `(*b).get(k)` inside __psa_map_get_h was
/// checked against the user's 0-arg `get` (D2 exit notes). The
/// prefix is injective and cannot collide with the `__psa_*`
/// runtime helpers or the generated `fn main` entry. Internal
/// bookkeeping (defs/sigs maps) stays keyed by the source name;
/// only emitted text is mangled, at the definition site and at
/// the one call site that reaches a user function.
fn sla_fn_name(name: &str) -> String {
    format!("__py_{}", name)
}

/// Names a lambda body may reference without capturing: the
/// expression-level builtins the emitter special-cases in call
/// position.
const LAMBDA_BUILTINS: [&str; 16] = [
    "print", "range", "len", "int", "float", "str", "bool", "list", "tuple", "set", "type",
    "isinstance", "open", "next", "repr", "bytes",
];

/// Collect every name an expression mentions: variable names,
/// call callee names and method-call object names.
fn collect_names(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Name(n, _) => out.push(n.clone()),
        Expr::Bin(a, _, b) => {
            collect_names(a, out);
            collect_names(b, out);
        }
        Expr::Unary(_, a) => collect_names(a, out),
        Expr::Walrus { value, .. } => {
            // The walrus target is bound by the expression
            // itself, not a free name; the value's names are.
            collect_names(value, out);
        }
        Expr::Call(f, args, kwargs, _) => {
            collect_names(f, out);
            for a in args {
                collect_names(a, out);
            }
            for (_, v) in kwargs {
                collect_names(v, out);
            }
        }
        Expr::ListLit(elems) => {
            for a in elems {
                collect_names(a, out);
            }
        }
        Expr::DictLit(pairs) => {
            for (k, v) in pairs {
                collect_names(k, out);
                collect_names(v, out);
            }
        }
        Expr::Index(a, b, _) => {
            collect_names(a, out);
            collect_names(b, out);
        }
        Expr::Slice(a, lo, hi, st) => {
            collect_names(a, out);
            for o in [lo, hi, st] {
                if let Some(x) = o {
                    collect_names(x, out);
                }
            }
        }
        Expr::MethodCall { obj, args, .. } => {
            out.push(obj.clone());
            for a in args {
                collect_names(a, out);
            }
        }
        Expr::Attr(base, _, _) => {
            // The attribute name is a field name, not a variable.
            collect_names(base, out);
        }
        Expr::TupleLit(elems) | Expr::SetLit(elems) => {
            for a in elems {
                collect_names(a, out);
            }
        }
        Expr::Comp { key, value, clauses, .. } => {
            // Comprehension loop variables are bound inside the
            // comprehension; only the other names are free.
            let bound: Vec<String> = clauses
                .iter()
                .filter_map(|c| match c {
                    CompClause::For { var, .. } => Some(var.clone()),
                    _ => None,
                })
                .collect();
            let mut inner: Vec<String> = Vec::new();
            collect_names(key, &mut inner);
            if let Some(v) = value {
                collect_names(v, &mut inner);
            }
            for c in clauses {
                match c {
                    CompClause::For { iter, .. } => collect_names(iter, &mut inner),
                    CompClause::If(cond) => collect_names(cond, &mut inner),
                }
            }
            for n in inner {
                if !bound.contains(&n) {
                    out.push(n);
                }
            }
        }
        Expr::IfExp { cond, then, els } => {
            collect_names(cond, out);
            collect_names(then, out);
            collect_names(els, out);
        }
        // D8: cell and closure nodes (produced by the
        // closure-conversion pass) mention the cell variable
        // or the environment expression they wrap.
        Expr::CellNew(a) | Expr::CellGet(a) => collect_names(a, out),
        Expr::CellSet(a, b) => {
            collect_names(a, out);
            collect_names(b, out);
        }
        Expr::ClosureVal(_, env) => collect_names(env, out),
        Expr::ModEnv => {}
        Expr::FStr(parts) => {
            for p in parts {
                p.visit_exprs(&mut |x| collect_names(x, &mut *out));
            }
        }
        _ => {}
    }
}


/// D8 closure conversion (supersedes the D3 lambda
/// desugaring; the retired D3 pass — desugar_expr /
/// desugar_stmt / desugar_pattern — was deleted at
/// integration).
///
/// A scope-aware rewrite that runs before any emission.
/// Every body (the module top level is the module scope) is
/// analysed for the names it binds. Lambdas and nested defs
/// become synthetic top-level defs. When an inner function
/// references a name bound in an enclosing function scope,
/// that variable becomes a heap cell in the enclosing scope
/// (Expr::CellNew — a one-element dynamic list), the inner
/// def gains a hidden leading environment parameter
/// (`__psa_env`), and its value is a tag-16 closure
/// (Expr::ClosureVal) carrying the cells. Names that resolve
/// to the module scope go through the module environment,
/// which main publishes into the sa_std globals cell
/// "psa.modenv" (Expr::ModEnv / Stmt::PublishModEnv).
/// Capture-free inner functions keep the exact D3 shape
/// (Expr::FnVal, tag 7), so programs without captures emit
/// the same text as before.
///
/// Semantics follow CPython: one cell per variable per scope
/// activation, created at scope entry, so closures made in a
/// loop share the loop variable's final value (late binding)
/// and `nonlocal` writes land in the very cell the enclosing
/// scope reads. Sub-forms that stay loudly refused and the
/// registered divergences are listed in
/// staging-closure/REPORT.md.
fn desugar_program(prog: &mut Program) -> Result<(), Diagnostic> {
    let mut conv = Conv {
        lambda_counter: 0,
        nested_counter: 0,
        tmp_counter: 0,
        new_funcs: Vec::new(),
        modenv_slots: HashMap::new(),
        modenv_names: Vec::new(),
    };
    // The module scope's bound names: everything the top
    // level binds, plus every name declared `global`
    // anywhere (a global declaration creates the module
    // binding when the declaring function first assigns).
    let mut module_bound: std::collections::HashSet<String> = HashSet::new();
    collect_bound(&prog.top, &mut module_bound);
    collect_global_decls(&prog.top, &mut module_bound);
    for f in &prog.funcs {
        collect_global_decls(&f.body, &mut module_bound);
    }
    for c in &prog.classes {
        for m in &c.methods {
            collect_global_decls(&m.body, &mut module_bound);
        }
    }
    let module_chain: Vec<std::collections::HashSet<String>> = vec![module_bound.clone()];
    // 0. Analysis pass (throwaway): learn every module name
    // any top-level def or method touches through the
    // module environment. The module scope's own conversion
    // needs that set before it starts — a module variable
    // read or written from a function must live in a cell
    // at module level too, with the module's own statements
    // rewritten to the same cell — but the real conversion
    // below keeps the D3 scope order (module, defs,
    // methods) so capture-free programs keep their exact
    // synthetic-name numbering. The probe converts the same
    // functions with a scratch Conv and discards the
    // results; a conversion error surfaces identically in
    // the real pass.
    let force_cells: std::collections::HashSet<String> = {
        let mut probe = Conv {
            lambda_counter: 0,
            nested_counter: 0,
            tmp_counter: 0,
            new_funcs: Vec::new(),
            modenv_slots: HashMap::new(),
            modenv_names: Vec::new(),
        };
        let mut g: std::collections::HashSet<String> = HashSet::new();
        for f in &prog.funcs {
            let (_, _, gf) = convert_function(&mut probe, f.clone(), &module_chain, &module_bound)?;
            g.extend(gf);
        }
        for c in &prog.classes {
            for m in &c.methods {
                let (_, _, gf) =
                    convert_function(&mut probe, m.clone(), &module_chain, &module_bound)?;
                g.extend(gf);
            }
        }
        g
    };
    // 1. The module scope itself (children resolve against
    // an empty chain; the module bound set is passed apart).
    let (top, module_out) = convert_scope(
        &mut conv,
        &[],
        prog.top.clone(),
        &[],
        &module_bound,
        true,
        &force_cells,
    )?;
    prog.top = top;
    // 2. Top-level defs. A free name resolves to the module
    // environment or stays free (a def, a builtin, or an
    // error the emitter reports); a top-level def never
    // captures an enclosing function scope, so the returned
    // `needed` is empty by construction.
    let mut funcs: Vec<Func> = Vec::new();
    for f in &prog.funcs {
        let (nf, _, _) = convert_function(&mut conv, f.clone(), &module_chain, &module_bound)?;
        funcs.push(nf);
    }
    prog.funcs = funcs;
    // 3. Class methods, same treatment.
    for c in &mut prog.classes {
        for m in &mut c.methods {
            let (nm, _, _) = convert_function(&mut conv, m.clone(), &module_chain, &module_bound)?;
            *m = nm;
        }
    }
    let new_funcs = std::mem::take(&mut conv.new_funcs);
    prog.funcs.extend(new_funcs);
    // 4. Module-environment fix-up: conversions above may
    // have slotted module-env names the module scope never
    // cellified (a `global` name no top-level statement
    // binds). Create their cells at the head of main and
    // point PublishModEnv at the final slot order.
    finish_modenv(&mut prog.top, &mut conv, &module_out.cell_of);
    Ok(())
}

/// D8: the hidden environment parameter every closure def
/// takes as its first parameter.
const ENV_PARAM: &str = "__psa_env";

/// D8: closure-conversion state shared across all scopes.
struct Conv {
    lambda_counter: u32,
    nested_counter: u32,
    tmp_counter: u32,
    new_funcs: Vec<Func>,
    modenv_slots: HashMap<String, usize>,
    modenv_names: Vec<String>,
}

impl Conv {
    fn fresh(&mut self, prefix: &str) -> String {
        let n = self.tmp_counter;
        self.tmp_counter += 1;
        format!("__psa_{}_{}", prefix, n)
    }
    fn modenv_slot(&mut self, name: &str) -> usize {
        match self.modenv_slots.get(name) {
            Some(i) => *i,
            None => {
                let i = self.modenv_names.len();
                self.modenv_slots.insert(name.to_string(), i);
                self.modenv_names.push(name.to_string());
                i
            }
        }
    }
}

/// D8: what one scope's conversion produced.
struct ScopeOutcome {
    /// Ordered names this scope receives through its
    /// environment parameter (empty unless the scope is a
    /// closure def).
    needed: Vec<String>,
    /// Own variables converted to cells: source name -> the
    /// cell variable's name in this scope.
    cell_of: HashMap<String, String>,
    /// Module-scope names this scope (or a descendant)
    /// accesses through the module environment.
    globals_used: std::collections::HashSet<String>,
}

/// D8: every name a statement list binds in its own scope
/// (CPython's rule: an assignment anywhere in the body makes
/// the name local to the whole body). Nested def interiors
/// are child scopes and are not entered; a nested def's NAME
/// is bound here.
fn collect_bound(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
    for s in stmts {
        match s {
            Stmt::Assign { name, .. } | Stmt::AugAssign { name, .. } => {
                out.insert(name.clone());
            }
            Stmt::MultiAssign { names, .. } | Stmt::Unpack { names, .. } => {
                for n in names {
                    out.insert(n.clone());
                }
            }
            Stmt::ForRange { var, body, els, .. } => {
                out.insert(var.clone());
                collect_bound(body, out);
                if let Some(b) = els {
                    collect_bound(b, out);
                }
            }
            Stmt::ForEach { vars, body, els, .. } => {
                for v in vars {
                    out.insert(v.clone());
                }
                collect_bound(body, out);
                if let Some(b) = els {
                    collect_bound(b, out);
                }
            }
            Stmt::If { arms, els, .. } => {
                for (_, b) in arms {
                    collect_bound(b, out);
                }
                if let Some(b) = els {
                    collect_bound(b, out);
                }
            }
            Stmt::While { body, els, .. } => {
                collect_bound(body, out);
                if let Some(b) = els {
                    collect_bound(b, out);
                }
            }
            Stmt::Try { body, handlers, els, finally, .. } => {
                collect_bound(body, out);
                for h in handlers {
                    if let Some(b) = &h.bind {
                        out.insert(b.clone());
                    }
                    collect_bound(&h.body, out);
                }
                if let Some(b) = els {
                    collect_bound(b, out);
                }
                if let Some(b) = finally {
                    collect_bound(b, out);
                }
            }
            Stmt::With { items, body, .. } => {
                for (_, b) in items {
                    if let Some(n) = b {
                        out.insert(n.clone());
                    }
                }
                collect_bound(body, out);
            }
            Stmt::Match { cases, .. } => {
                for c in cases {
                    for n in pattern_captures(&c.pattern) {
                        out.insert(n);
                    }
                    collect_bound(&c.body, out);
                }
            }
            Stmt::Def(f) => {
                out.insert(f.name.clone());
            }
            Stmt::Del { targets, .. } => {
                for (n, idx) in targets {
                    if idx.is_none() {
                        out.insert(n.clone());
                    }
                }
            }
            _ => {}
        }
        // Assignment expressions bind in this scope too
        // (CPython: a name bound anywhere in the body —
        // including inside an expression — is local to the
        // whole body), but the match above only sees
        // statement-level targets. Walk the statement's own
        // expressions for walrus targets; walk_stmts_exprs
        // does not enter child scopes (lambda bodies, nested
        // defs), whose interiors are their own scopes, and a
        // comprehension's walrus targets bind HERE, so
        // entering comprehensions is correct.
        walk_stmts_exprs(std::slice::from_ref(s), &mut |e| {
            if let Expr::Walrus { name, .. } = e {
                out.insert(name.clone());
            }
        });
    }
}

/// D8: the names a statement list declares `global` or
/// `nonlocal` in its own scope (child scopes not entered).
fn collect_declared(
    stmts: &[Stmt],
    dglobal: &mut std::collections::HashSet<String>,
    dnonlocal: &mut std::collections::HashSet<String>,
) {
    for s in stmts {
        match s {
            Stmt::Global { names, .. } => {
                for n in names {
                    dglobal.insert(n.clone());
                }
            }
            Stmt::Nonlocal { names, .. } => {
                for n in names {
                    dnonlocal.insert(n.clone());
                }
            }
            Stmt::If { arms, els, .. } => {
                for (_, b) in arms {
                    collect_declared(b, dglobal, dnonlocal);
                }
                if let Some(b) = els {
                    collect_declared(b, dglobal, dnonlocal);
                }
            }
            Stmt::While { body, els, .. }
            | Stmt::ForRange { body, els, .. }
            | Stmt::ForEach { body, els, .. } => {
                collect_declared(body, dglobal, dnonlocal);
                if let Some(b) = els {
                    collect_declared(b, dglobal, dnonlocal);
                }
            }
            Stmt::With { body, .. } => collect_declared(body, dglobal, dnonlocal),
            Stmt::Try { body, handlers, els, finally, .. } => {
                collect_declared(body, dglobal, dnonlocal);
                for h in handlers {
                    collect_declared(&h.body, dglobal, dnonlocal);
                }
                if let Some(b) = els {
                    collect_declared(b, dglobal, dnonlocal);
                }
                if let Some(b) = finally {
                    collect_declared(b, dglobal, dnonlocal);
                }
            }
            Stmt::Match { cases, .. } => {
                for c in cases {
                    collect_declared(&c.body, dglobal, dnonlocal);
                }
            }
            _ => {}
        }
    }
}

/// D8: like collect_declared, but appends the declared
/// names to a Vec in source order (duplicates kept; the
/// caller dedups). `want_nonlocal` selects which
/// declaration kind to collect. Slot assignment must be
/// deterministic, so environment slots are handed out in
/// this order rather than HashSet order.
fn collect_declared_ordered(stmts: &[Stmt], want_nonlocal: bool, out: &mut Vec<String>) {
    for s in stmts {
        match s {
            Stmt::Global { names, .. } if !want_nonlocal => out.extend(names.iter().cloned()),
            Stmt::Nonlocal { names, .. } if want_nonlocal => out.extend(names.iter().cloned()),
            Stmt::If { arms, els, .. } => {
                for (_, b) in arms {
                    collect_declared_ordered(b, want_nonlocal, out);
                }
                if let Some(b) = els {
                    collect_declared_ordered(b, want_nonlocal, out);
                }
            }
            Stmt::While { body, els, .. }
            | Stmt::ForRange { body, els, .. }
            | Stmt::ForEach { body, els, .. } => {
                collect_declared_ordered(body, want_nonlocal, out);
                if let Some(b) = els {
                    collect_declared_ordered(b, want_nonlocal, out);
                }
            }
            Stmt::With { body, .. } => collect_declared_ordered(body, want_nonlocal, out),
            Stmt::Try { body, handlers, els, finally, .. } => {
                collect_declared_ordered(body, want_nonlocal, out);
                for h in handlers {
                    collect_declared_ordered(&h.body, want_nonlocal, out);
                }
                if let Some(b) = els {
                    collect_declared_ordered(b, want_nonlocal, out);
                }
                if let Some(b) = finally {
                    collect_declared_ordered(b, want_nonlocal, out);
                }
            }
            Stmt::Match { cases, .. } => {
                for c in cases {
                    collect_declared_ordered(&c.body, want_nonlocal, out);
                }
            }
            _ => {}
        }
    }
}

/// D8: every name declared `global` in a body or in any
/// nested def inside it (such a name is a module binding,
/// created when the declaring function first assigns it).
fn collect_global_decls(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
    let mut g: std::collections::HashSet<String> = HashSet::new();
    let mut n: std::collections::HashSet<String> = HashSet::new();
    collect_declared(stmts, &mut g, &mut n);
    out.extend(g);
    for s in stmts {
        match s {
            Stmt::Def(f) => collect_global_decls(&f.body, out),
            Stmt::If { arms, els, .. } => {
                for (_, b) in arms {
                    collect_global_decls(b, out);
                }
                if let Some(b) = els {
                    collect_global_decls(b, out);
                }
            }
            Stmt::While { body, els, .. }
            | Stmt::ForRange { body, els, .. }
            | Stmt::ForEach { body, els, .. } => {
                collect_global_decls(body, out);
                if let Some(b) = els {
                    collect_global_decls(b, out);
                }
            }
            Stmt::With { body, .. } => collect_global_decls(body, out),
            Stmt::Try { body, handlers, els, finally, .. } => {
                collect_global_decls(body, out);
                for h in handlers {
                    collect_global_decls(&h.body, out);
                }
                if let Some(b) = els {
                    collect_global_decls(b, out);
                }
                if let Some(b) = finally {
                    collect_global_decls(b, out);
                }
            }
            Stmt::Match { cases, .. } => {
                for c in cases {
                    collect_global_decls(&c.body, out);
                }
            }
            _ => {}
        }
    }
}

/// D8: how a free name resolves against the enclosing
/// scopes (chain[0] is the module scope): 0 = stays free
/// (a def, a builtin, or an error for the emitter),
/// 1 = an enclosing function scope binds it (a capture),
/// 2 = the module scope binds it (a global).
fn resolve_chain(chain: &[std::collections::HashSet<String>], n: &str) -> u8 {
    for (i, s) in chain.iter().enumerate().rev() {
        if s.contains(n) {
            return if i == 0 { 2 } else { 1 };
        }
    }
    0
}

/// D8: a child-function site inside a scope: a lambda
/// expression or a nested def statement. The traversal in
/// scan_stmt / scan_expr visits sites in exactly the order
/// the pass-B rewriter (Rw) encounters them; a site's
/// interior is NOT scanned here — converting the child
/// scans its own body.
enum Site<'a> {
    Lambda { params: &'a [String], body: &'a Expr, line: u32 },
    Def(&'a Func),
}

/// D8: a converted child function, in site order.
struct ChildResult {
    key: String,
    needed: Vec<String>,
    is_def: bool,
    src_name: String,
    line: u32,
}

fn scan_stmts(
    stmts: &[Stmt],
    conv: &mut Conv,
    cb: &mut dyn FnMut(&mut Conv, Site) -> Result<(), Diagnostic>,
) -> Result<(), Diagnostic> {
    for s in stmts {
        scan_stmt(s, conv, cb)?;
    }
    Ok(())
}

fn scan_stmt(
    s: &Stmt,
    conv: &mut Conv,
    cb: &mut dyn FnMut(&mut Conv, Site) -> Result<(), Diagnostic>,
) -> Result<(), Diagnostic> {
    match s {
        Stmt::Assign { value, .. }
        | Stmt::AugAssign { value, .. }
        | Stmt::ExprStmt { value, .. }
        | Stmt::Unpack { value, .. } => scan_expr(value, conv, cb),
        Stmt::IndexAssign { index, value, .. } => {
            scan_expr(index, conv, cb)?;
            scan_expr(value, conv, cb)
        }
        Stmt::AttrAssign { base, value, .. } => {
            scan_expr(base, conv, cb)?;
            scan_expr(value, conv, cb)
        }
        Stmt::AttrDel { base, .. } => scan_expr(base, conv, cb),
        Stmt::If { arms, els, .. } => {
            for (c, b) in arms {
                scan_expr(c, conv, cb)?;
                scan_stmts(b, conv, cb)?;
            }
            if let Some(b) = els {
                scan_stmts(b, conv, cb)?;
            }
            Ok(())
        }
        Stmt::While { cond, body, els, .. } => {
            scan_expr(cond, conv, cb)?;
            scan_stmts(body, conv, cb)?;
            if let Some(b) = els {
                scan_stmts(b, conv, cb)?;
            }
            Ok(())
        }
        Stmt::ForRange { start, stop, body, els, .. } => {
            if let Some(x) = start {
                scan_expr(x, conv, cb)?;
            }
            scan_expr(stop, conv, cb)?;
            scan_stmts(body, conv, cb)?;
            if let Some(b) = els {
                scan_stmts(b, conv, cb)?;
            }
            Ok(())
        }
        Stmt::ForEach { iter, body, els, .. } => {
            scan_expr(iter, conv, cb)?;
            scan_stmts(body, conv, cb)?;
            if let Some(b) = els {
                scan_stmts(b, conv, cb)?;
            }
            Ok(())
        }
        Stmt::Return { value, .. } | Stmt::Yield { value, .. } | Stmt::Raise { value, .. } => {
            if let Some(x) = value {
                scan_expr(x, conv, cb)?;
            }
            Ok(())
        }
        Stmt::Assert { cond, .. } => scan_expr(cond, conv, cb),
        Stmt::Del { targets, .. } => {
            for (_, idx) in targets {
                if let Some(x) = idx {
                    scan_expr(x, conv, cb)?;
                }
            }
            Ok(())
        }
        Stmt::MultiAssign { values, .. } => {
            for x in values {
                scan_expr(x, conv, cb)?;
            }
            Ok(())
        }
        Stmt::Try { body, handlers, els, finally, .. } => {
            scan_stmts(body, conv, cb)?;
            for h in handlers {
                scan_stmts(&h.body, conv, cb)?;
            }
            if let Some(b) = els {
                scan_stmts(b, conv, cb)?;
            }
            if let Some(b) = finally {
                scan_stmts(b, conv, cb)?;
            }
            Ok(())
        }
        Stmt::With { items, body, .. } => {
            for (e, _) in items {
                scan_expr(e, conv, cb)?;
            }
            scan_stmts(body, conv, cb)
        }
        Stmt::Match { subject, cases, .. } => {
            scan_expr(subject, conv, cb)?;
            for c in cases {
                scan_pattern(&c.pattern, conv, cb)?;
                if let Some(g) = &c.guard {
                    scan_expr(g, conv, cb)?;
                }
                scan_stmts(&c.body, conv, cb)?;
            }
            Ok(())
        }
        Stmt::Def(f) => cb(conv, Site::Def(f)),
        _ => Ok(()),
    }
}

fn scan_pattern(
    p: &Pattern,
    conv: &mut Conv,
    cb: &mut dyn FnMut(&mut Conv, Site) -> Result<(), Diagnostic>,
) -> Result<(), Diagnostic> {
    match p {
        Pattern::Lit(e) => scan_expr(e, conv, cb),
        Pattern::Seq(pats) | Pattern::Or(pats) => {
            for x in pats {
                scan_pattern(x, conv, cb)?;
            }
            Ok(())
        }
        Pattern::Capture(_) | Pattern::Wildcard => Ok(()),
    }
}

fn scan_expr(
    e: &Expr,
    conv: &mut Conv,
    cb: &mut dyn FnMut(&mut Conv, Site) -> Result<(), Diagnostic>,
) -> Result<(), Diagnostic> {
    match e {
        Expr::Lambda { params, body, line } => cb(
            conv,
            Site::Lambda { params, body, line: *line },
        ),
        Expr::Bin(a, _, b) => {
            scan_expr(a, conv, cb)?;
            scan_expr(b, conv, cb)
        }
        Expr::Unary(_, a) => scan_expr(a, conv, cb),
        Expr::Call(f, args, kwargs, _) => {
            scan_expr(f, conv, cb)?;
            for a in args {
                scan_expr(a, conv, cb)?;
            }
            for (_, v) in kwargs {
                scan_expr(v, conv, cb)?;
            }
            Ok(())
        }
        Expr::ListLit(elems) | Expr::TupleLit(elems) | Expr::SetLit(elems) => {
            for a in elems {
                scan_expr(a, conv, cb)?;
            }
            Ok(())
        }
        Expr::DictLit(pairs) => {
            for (k, v) in pairs {
                scan_expr(k, conv, cb)?;
                scan_expr(v, conv, cb)?;
            }
            Ok(())
        }
        Expr::Index(a, b, _) => {
            scan_expr(a, conv, cb)?;
            scan_expr(b, conv, cb)
        }
        Expr::Slice(a, lo, hi, st) => {
            scan_expr(a, conv, cb)?;
            for o in [lo, hi, st] {
                if let Some(x) = o {
                    scan_expr(x, conv, cb)?;
                }
            }
            Ok(())
        }
        Expr::MethodCall { args, .. } => {
            for a in args {
                scan_expr(a, conv, cb)?;
            }
            Ok(())
        }
        Expr::Attr(base, _, _) => scan_expr(base, conv, cb),
        Expr::Comp { key, value, clauses, .. } => {
            scan_expr(key, conv, cb)?;
            if let Some(v) = value {
                scan_expr(v, conv, cb)?;
            }
            for c in clauses {
                match c {
                    CompClause::For { iter, .. } => scan_expr(iter, conv, cb)?,
                    CompClause::If(cond) => scan_expr(cond, conv, cb)?,
                }
            }
            Ok(())
        }
        Expr::IfExp { cond, then, els } => {
            scan_expr(cond, conv, cb)?;
            scan_expr(then, conv, cb)?;
            scan_expr(els, conv, cb)
        }
        Expr::FStr(parts) => {
            scan_fstr_parts(parts, conv, cb)?;
            Ok(())
        }
        Expr::Await(a) => scan_expr(a, conv, cb),
        Expr::Walrus { value, .. } => scan_expr(value, conv, cb),
        _ => Ok(()),
    }
}

/// Scan the expressions of f-string parts: each field's value
/// expression and, recursively, the expressions inside its
/// format spec (a spec is a mini f-string).
fn scan_fstr_parts(
    parts: &[FStrPart],
    conv: &mut Conv,
    cb: &mut dyn FnMut(&mut Conv, Site) -> Result<(), Diagnostic>,
) -> Result<(), Diagnostic> {
    for p in parts {
        if let FStrPart::Field { value, spec, .. } = p {
            scan_expr(value, conv, cb)?;
            if let Some(spec_parts) = spec {
                scan_fstr_parts(spec_parts, conv, cb)?;
            }
        }
    }
    Ok(())
}

/// D8: visit every expression in a statement list without
/// entering child scopes (lambda bodies, nested defs).
/// Used by the genexp and match-capture scans.
fn walk_stmts_exprs(stmts: &[Stmt], f: &mut dyn FnMut(&Expr)) {
    for s in stmts {
        match s {
            Stmt::Assign { value, .. }
            | Stmt::AugAssign { value, .. }
            | Stmt::ExprStmt { value, .. }
            | Stmt::Unpack { value, .. } => walk_expr(value, f),
            Stmt::IndexAssign { index, value, .. } => {
                walk_expr(index, f);
                walk_expr(value, f);
            }
            Stmt::AttrAssign { base, value, .. } => {
                walk_expr(base, f);
                walk_expr(value, f);
            }
            Stmt::AttrDel { base, .. } => walk_expr(base, f),
            Stmt::If { arms, els, .. } => {
                for (c, b) in arms {
                    walk_expr(c, f);
                    walk_stmts_exprs(b, f);
                }
                if let Some(b) = els {
                    walk_stmts_exprs(b, f);
                }
            }
            Stmt::While { cond, body, els, .. } => {
                walk_expr(cond, f);
                walk_stmts_exprs(body, f);
                if let Some(b) = els {
                    walk_stmts_exprs(b, f);
                }
            }
            Stmt::ForRange { start, stop, body, els, .. } => {
                if let Some(x) = start {
                    walk_expr(x, f);
                }
                walk_expr(stop, f);
                walk_stmts_exprs(body, f);
                if let Some(b) = els {
                    walk_stmts_exprs(b, f);
                }
            }
            Stmt::ForEach { iter, body, els, .. } => {
                walk_expr(iter, f);
                walk_stmts_exprs(body, f);
                if let Some(b) = els {
                    walk_stmts_exprs(b, f);
                }
            }
            Stmt::Return { value, .. } | Stmt::Yield { value, .. } | Stmt::Raise { value, .. } => {
                if let Some(x) = value {
                    walk_expr(x, f);
                }
            }
            Stmt::Assert { cond, .. } => walk_expr(cond, f),
            Stmt::Del { targets, .. } => {
                for (_, i) in targets {
                    if let Some(x) = i {
                        walk_expr(x, f);
                    }
                }
            }
            Stmt::MultiAssign { values, .. } => {
                for v in values {
                    walk_expr(v, f);
                }
            }
            Stmt::Try { body, handlers, els, finally, .. } => {
                walk_stmts_exprs(body, f);
                for h in handlers {
                    walk_stmts_exprs(&h.body, f);
                }
                if let Some(b) = els {
                    walk_stmts_exprs(b, f);
                }
                if let Some(b) = finally {
                    walk_stmts_exprs(b, f);
                }
            }
            Stmt::With { items, body, .. } => {
                for (e, _) in items {
                    walk_expr(e, f);
                }
                walk_stmts_exprs(body, f);
            }
            Stmt::Match { subject, cases, .. } => {
                walk_expr(subject, f);
                for c in cases {
                    walk_pattern_exprs(&c.pattern, f);
                    if let Some(g) = &c.guard {
                        walk_expr(g, f);
                    }
                    walk_stmts_exprs(&c.body, f);
                }
            }
            _ => {}
        }
    }
}

fn walk_pattern_exprs(p: &Pattern, f: &mut dyn FnMut(&Expr)) {
    match p {
        Pattern::Lit(e) => walk_expr(e, f),
        Pattern::Seq(ps) | Pattern::Or(ps) => {
            for x in ps {
                walk_pattern_exprs(x, f);
            }
        }
        _ => {}
    }
}

fn walk_expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    match e {
        Expr::Bin(a, _, b) => {
            walk_expr(a, f);
            walk_expr(b, f);
        }
        Expr::Unary(_, a) => walk_expr(a, f),
        Expr::Call(c, args, kwargs, _) => {
            walk_expr(c, f);
            for a in args {
                walk_expr(a, f);
            }
            for (_, v) in kwargs {
                walk_expr(v, f);
            }
        }
        Expr::ListLit(es) | Expr::TupleLit(es) | Expr::SetLit(es) => {
            for a in es {
                walk_expr(a, f);
            }
        }
        Expr::DictLit(ps) => {
            for (k, v) in ps {
                walk_expr(k, f);
                walk_expr(v, f);
            }
        }
        Expr::Index(a, b, _) => {
            walk_expr(a, f);
            walk_expr(b, f);
        }
        Expr::Slice(a, lo, hi, st) => {
            walk_expr(a, f);
            for o in [lo, hi, st] {
                if let Some(x) = o {
                    walk_expr(x, f);
                }
            }
        }
        Expr::MethodCall { args, .. } => {
            for a in args {
                walk_expr(a, f);
            }
        }
        Expr::Attr(b, _, _) => walk_expr(b, f),
        Expr::Comp { key, value, clauses, .. } => {
            walk_expr(key, f);
            if let Some(v) = value {
                walk_expr(v, f);
            }
            for c in clauses {
                match c {
                    CompClause::For { iter, .. } => walk_expr(iter, f),
                    CompClause::If(c2) => walk_expr(c2, f),
                }
            }
        }
        Expr::IfExp { cond, then, els } => {
            walk_expr(cond, f);
            walk_expr(then, f);
            walk_expr(els, f);
        }
        Expr::FStr(parts) => {
            for p in parts {
                p.visit_exprs(&mut |x| walk_expr(x, &mut *f));
            }
        }
        Expr::Await(a) => walk_expr(a, f),
        Expr::Walrus { value, .. } => walk_expr(value, f),
        Expr::CellNew(a) | Expr::CellGet(a) => walk_expr(a, f),
        Expr::CellSet(a, b) => {
            walk_expr(a, f);
            walk_expr(b, f);
        }
        Expr::ClosureVal(_, env) => walk_expr(env, f),
        _ => {}
    }
}

/// D8: convert one scope (a function body, or the module
/// top level when `is_module`). `chain` holds the enclosing
/// scopes' bound sets, module first; it is empty for the
/// module scope itself. Returns the rewritten body — cell
/// creations (and, for the module scope, the PublishModEnv
/// placeholder) prepended — and the scope's outcome.
fn convert_scope(
    conv: &mut Conv,
    params: &[String],
    body: Vec<Stmt>,
    chain: &[std::collections::HashSet<String>],
    module_bound: &std::collections::HashSet<String>,
    is_module: bool,
    force_cells: &std::collections::HashSet<String>,
) -> Result<(Vec<Stmt>, ScopeOutcome), Diagnostic> {
    // ---- scope analysis (CPython's rules: a name assigned
    // anywhere in the body is local everywhere in it).
    let mut bound: std::collections::HashSet<String> = params.iter().cloned().collect();
    collect_bound(&body, &mut bound);
    let mut dglobal: std::collections::HashSet<String> = HashSet::new();
    let mut dnonlocal: std::collections::HashSet<String> = HashSet::new();
    collect_declared(&body, &mut dglobal, &mut dnonlocal);
    for n in &dglobal {
        if dnonlocal.contains(n) {
            return Err(eerr(
                0,
                &format!("name `{}` is both global and nonlocal (CPython raises SyntaxError)", n),
            ));
        }
        if params.iter().any(|p| p == n) {
            return Err(eerr(
                0,
                &format!("name `{}` is a parameter and declared global (CPython raises SyntaxError)", n),
            ));
        }
    }
    for n in &dnonlocal {
        if params.iter().any(|p| p == n) {
            return Err(eerr(
                0,
                &format!("name `{}` is a parameter and declared nonlocal (CPython raises SyntaxError)", n),
            ));
        }
    }
    if is_module {
        if !dnonlocal.is_empty() {
            return Err(eerr(
                0,
                "nonlocal declaration not allowed at module level (CPython raises SyntaxError)",
            ));
        }
        // `global` at module level is a no-op in CPython.
        dglobal.clear();
    }
    for n in dglobal.iter().chain(dnonlocal.iter()) {
        bound.remove(n);
    }
    for n in &dnonlocal {
        if resolve_chain(chain, n) != 1 {
            return Err(eerr(
                0,
                &format!(
                    "no binding for nonlocal `{}` found in any enclosing function scope (CPython raises SyntaxError)",
                    n
                ),
            ));
        }
    }
    // ---- pass A: convert the child functions (lambdas and
    // nested defs) in traversal order. Learn which of this
    // scope's variables must become cells, which names this
    // scope must itself receive through its environment, and
    // which module names the subtree touches.
    let mut sites: Vec<ChildResult> = Vec::new();
    let mut needed: Vec<String> = Vec::new();
    let mut cellified: Vec<String> = Vec::new();
    let mut globals_used: std::collections::HashSet<String> = HashSet::new();
    {
        let mut child_chain: Vec<std::collections::HashSet<String>> = chain.to_vec();
        // The module scope's children resolve against the
        // full module namespace (which also holds names
        // that exist only through a `global` declaration);
        // a function scope's children resolve against its
        // own bound set.
        child_chain.push(if is_module {
            module_bound.clone()
        } else {
            bound.clone()
        });
        let bound_ref = &bound;
        let mut cb = |conv: &mut Conv, site: Site| -> Result<(), Diagnostic> {
            let (raw, src_name, is_def, line) = match site {
                Site::Lambda { params, body, line } => {
                    let f = Func {
                        name: String::new(),
                        params: params.iter().map(|p| (p.clone(), None)).collect(),
                        vararg: None,
                        kwarg: None,
                        is_async: false,
                        body: vec![Stmt::Return { value: Some((*body).clone()), line, anchor: SourceAnchor::at(line, 0, None)}],
                        line, anchor: SourceAnchor::at(line, 0, None),
                    };
                    (f, String::new(), false, line)
                }
                Site::Def(f) => {
                    if f.is_async {
                        return Err(eerr(
                            f.line,
                            &format!(
                                "nested async def `{}` is not supported yet (D8 v1 closures)",
                                f.name
                            ),
                        ));
                    }
                    if func_is_generator(f) {
                        return Err(eerr(
                            f.line,
                            &format!(
                                "nested generator def `{}` cannot capture yet (D8 v1 closures)",
                                f.name
                            ),
                        ));
                    }
                    (f.clone(), f.name.clone(), true, f.line)
                }
            };
            let (mut cfunc, cneeded, cglobals) =
                convert_function(conv, raw, &child_chain, module_bound)?;
            for n in cglobals {
                globals_used.insert(n);
            }
            // The def key is assigned only now — after the
            // interior was converted — which preserves the
            // D3 lambda numbering (post-order).
            let key = if is_def {
                let k = format!("__psa_nested_{}", conv.nested_counter);
                conv.nested_counter += 1;
                k
            } else {
                let k = format!("__psa_lambda_{}", conv.lambda_counter);
                conv.lambda_counter += 1;
                k
            };
            cfunc.name = key.clone();
            conv.new_funcs.push(cfunc);
            sites.push(ChildResult {
                key,
                needed: cneeded.clone(),
                is_def,
                src_name,
                line,
            });
            for n in &cneeded {
                if bound_ref.contains(n) {
                    if !cellified.contains(n) {
                        cellified.push(n.clone());
                    }
                } else if resolve_chain(chain, n) == 1 {
                    if !needed.contains(n) {
                        needed.push(n.clone());
                    }
                } else {
                    return Err(eerr(
                        line,
                        "internal error: a nested function needs a name that resolves nowhere",
                    ));
                }
            }
            Ok(())
        };
        scan_stmts(&body, conv, &mut cb)?;
    }
    // Names this scope binds that the subtree (or, for the
    // module scope, any function in the program — the
    // caller passes those in `force_cells`) accesses
    // through the module environment must live in cells
    // here: the module scope's own reads and writes have
    // to go through the very cell the environment
    // publishes, or the two sides would see different
    // variables. (For a function scope the filter is
    // empty by construction — globals are never its own
    // bound names.)
    {
        let mut extra: Vec<String> = globals_used
            .iter()
            .chain(force_cells.iter())
            .filter(|n| bound.contains(*n) && !cellified.contains(*n))
            .cloned()
            .collect();
        extra.sort();
        extra.dedup();
        for n in extra {
            cellified.push(n);
        }
    }
    // Generator expressions in this scope reference its
    // variables from a synthetic def (see emit_genexp), so
    // every bound name their key / condition expressions
    // mention must live in a cell too. The clause
    // iterables are NOT collected: psa's genexp shape
    // evaluates them in the enclosing scope and passes
    // them in, so their names never enter the synthetic
    // def (collecting them would needlessly cellify, say,
    // a statically-typed list parameter — whose cell
    // initialiser then could not box it).
    {
        let mut gnames: Vec<String> = Vec::new();
        walk_stmts_exprs(&body, &mut |e| {
            if let Expr::Comp { kind: CompKind::GenExp, key, clauses, .. } = e {
                let cvars: Vec<&String> = clauses
                    .iter()
                    .filter_map(|c| match c {
                        CompClause::For { var, .. } => Some(var),
                        _ => None,
                    })
                    .collect();
                let mut inner: Vec<String> = Vec::new();
                collect_names(key, &mut inner);
                for c in clauses {
                    if let CompClause::If(cond) = c {
                        collect_names(cond, &mut inner);
                    }
                }
                for n in inner {
                    if !cvars.contains(&&n) {
                        gnames.push(n);
                    }
                }
            }
        });
        // Sorted + deduped: collect_names appends in
        // traversal order with duplicates, and cell-creation
        // order is visible in the emitted text, so it must
        // be deterministic.
        gnames.sort();
        gnames.dedup();
        for n in gnames {
            if bound.contains(&n) && !cellified.contains(&n) {
                cellified.push(n);
            }
        }
    }
    // A match pattern may not bind a cell variable (the
    // match machinery binds captures through its own temps,
    // with no cell write-back path).
    {
        let mut caps: std::collections::HashSet<String> = HashSet::new();
        for s in &body {
            if let Stmt::Match { cases, .. } = s {
                for c in cases {
                    for n in pattern_captures(&c.pattern) {
                        caps.insert(n);
                    }
                }
            }
        }
        walk_stmts_match_captures(&body, &mut caps);
        for n in &caps {
            if cellified.contains(n) {
                return Err(eerr(
                    0,
                    &format!(
                        "match pattern captures `{}`, which is also captured by a nested function; binding a cell variable through a match pattern is not supported yet (D8 v1)",
                        n
                    ),
                ));
            }
        }
    }
    // `nonlocal` names receive their environment slots
    // first (they are known statically), in declaration
    // order — HashSet order is not stable across runs and
    // slot order is visible in the emitted text. Further
    // captures append in encounter order during the
    // rewrite.
    {
        let mut ordered: Vec<String> = Vec::new();
        collect_declared_ordered(&body, true, &mut ordered);
        for n in ordered {
            if !needed.contains(&n) {
                needed.push(n);
            }
        }
    }
    let mut cell_of: HashMap<String, String> = HashMap::new();
    for n in &cellified {
        let cv = conv.fresh("cell");
        cell_of.insert(n.clone(), cv);
    }
    // ---- pass B: rewrite the body.
    let mut rw = Rw {
        conv,
        bound: &bound,
        cell_of: &cell_of,
        dglobal: &dglobal,
        dnonlocal: &dnonlocal,
        chain,
        module_bound,
        is_module,
        needed,
        globals_used,
        sites: &sites,
        site_idx: 0,
        shadow: Vec::new(),
        hoist_forbidden: false,
    };
    let new_body = rw.rewrite_stmts(body)?;
    let needed = rw.needed;
    let globals_used = rw.globals_used;
    if rw.site_idx != sites.len() {
        return Err(eerr(
            0,
            "internal error: closure conversion site count mismatch (scanner/rewriter traversal diverged)",
        ));
    }
    // ---- prepend the cell creations: one cell per captured
    // variable, created at scope entry (CPython's cell
    // lifetime). Parameters initialise their cell from the
    // argument; other variables start as None and are filled
    // by their first (rewritten) assignment.
    let mut head: Vec<Stmt> = Vec::new();
    for n in &cellified {
        let init = if !is_module && params.iter().any(|p| p == n) {
            Expr::Name(n.clone(), ExprSpan::synthetic())
        } else {
            Expr::NoneLit
        };
        head.push(Stmt::Assign {
            name: cell_of[n].clone(),
            value: Expr::CellNew(Box::new(init)),
            line: 0, anchor: SourceAnchor::synthetic(),
        });
    }
    if is_module {
        // Placeholder: finish_modenv points it at the final
        // slot order, or removes it when no module-env name
        // was ever slotted.
        head.push(Stmt::PublishModEnv { cells: Vec::new(), line: 0, anchor: SourceAnchor::synthetic()});
    }
    head.extend(new_body);
    Ok((head, ScopeOutcome { needed, cell_of, globals_used }))
}

/// D8: convert one function (a top-level def, a method, or
/// a synthetic child). When the converted body needs names
/// from an enclosing function scope, the def gains the
/// hidden environment parameter in first position.
fn convert_function(
    conv: &mut Conv,
    f: Func,
    chain: &[std::collections::HashSet<String>],
    module_bound: &std::collections::HashSet<String>,
) -> Result<(Func, Vec<String>, std::collections::HashSet<String>), Diagnostic> {
    let mut nf = f;
    // Parameter defaults are evaluated in the ENCLOSING
    // scope at def time; the converter does not rewrite
    // them, so a lambda there would reach emission
    // unconverted. Refuse it loudly instead.
    for (_, d) in &nf.params {
        if let Some(dx) = d {
            let mut has_lambda = false;
            walk_expr(dx, &mut |e| {
                if matches!(e, Expr::Lambda { .. }) {
                    has_lambda = true;
                }
            });
            if has_lambda {
                return Err(eerr(
                    nf.line,
                    "a lambda in a parameter default value is not supported yet (D8 v1 closures)",
                ));
            }
        }
    }
    let params = nf.param_names();
    let body = std::mem::take(&mut nf.body);
    let no_force: std::collections::HashSet<String> = HashSet::new();
    let (body, out) =
        convert_scope(conv, &params, body, chain, module_bound, false, &no_force)?;
    nf.body = body;
    if !out.needed.is_empty() {
        if nf.params.iter().any(|(n, _)| n == ENV_PARAM) {
            return Err(eerr(
                nf.line,
                "a function that captures cannot have a parameter named `__psa_env` (reserved for the closure environment)",
            ));
        }
        let mut ps = vec![(ENV_PARAM.to_string(), None)];
        ps.extend(nf.params.iter().cloned());
        nf.params = ps;
    }
    Ok((nf, out.needed, out.globals_used))
}

/// D8: after every scope is converted, point the module
/// scope's PublishModEnv at the final module-environment
/// slot order, creating cells (at the head of main) for
/// slotted names the module scope never cellified — a
/// `global` name no top-level statement binds. When no
/// module-env name was ever slotted, the placeholder is
/// removed instead.
fn finish_modenv(
    top: &mut Vec<Stmt>,
    conv: &mut Conv,
    module_cells: &HashMap<String, String>,
) {
    if conv.modenv_names.is_empty() {
        top.retain(|s| !matches!(s, Stmt::PublishModEnv { .. }));
        return;
    }
    let names = conv.modenv_names.clone();
    let mut cells: Vec<String> = Vec::new();
    let mut missing: Vec<Stmt> = Vec::new();
    for n in &names {
        match module_cells.get(n) {
            Some(cv) => cells.push(cv.clone()),
            None => {
                let cv = conv.fresh("cell");
                cells.push(cv.clone());
                missing.push(Stmt::Assign {
                    name: cv,
                    value: Expr::CellNew(Box::new(Expr::NoneLit)),
                    line: 0, anchor: SourceAnchor::synthetic(),
                });
            }
        }
    }
    let mut new_top: Vec<Stmt> = missing;
    new_top.append(top);
    *top = new_top;
    for s in top.iter_mut() {
        if let Stmt::PublishModEnv { cells: c, .. } = s {
            *c = cells.clone();
            break;
        }
    }
}

/// D8: how one name resolves inside the scope being
/// rewritten.
#[derive(Clone)]
enum Cl {
    /// An ordinary local (or a name that stays free: a def,
    /// a builtin, a comprehension variable).
    Plain,
    /// A cell variable of this scope.
    Cell(String),
    /// A slot of this scope's environment parameter.
    Env(usize),
    /// A slot of the module environment.
    Global(usize),
}

/// D8 pass B: rewrite one scope's body against the
/// classification computed in pass A. Reads of cell /
/// environment / global names become CellGet; writes become
/// CellSet; child-function sites (visited in the same
/// traversal order as the pass-A scanner) become FnVal /
/// ClosureVal bindings. Method calls on a cell-held value
/// need the value in a plain temporary (MethodCall names
/// its object), so the read is hoisted to just before the
/// enclosing statement — except in positions where hoisting
/// would change semantics (a while condition re-evaluates;
/// match guards and pattern literals run under the match
/// machinery), which refuse loudly instead.
struct Rw<'a> {
    conv: &'a mut Conv,
    bound: &'a std::collections::HashSet<String>,
    cell_of: &'a HashMap<String, String>,
    dglobal: &'a std::collections::HashSet<String>,
    dnonlocal: &'a std::collections::HashSet<String>,
    chain: &'a [std::collections::HashSet<String>],
    module_bound: &'a std::collections::HashSet<String>,
    is_module: bool,
    needed: Vec<String>,
    globals_used: std::collections::HashSet<String>,
    sites: &'a [ChildResult],
    site_idx: usize,
    shadow: Vec<std::collections::HashSet<String>>,
    hoist_forbidden: bool,
}

impl<'a> Rw<'a> {
    fn classify(&mut self, name: &str) -> Cl {
        if self.shadow.iter().any(|f| f.contains(name)) {
            return Cl::Plain;
        }
        if self.dnonlocal.contains(name) {
            return Cl::Env(self.env_slot(name));
        }
        if self.dglobal.contains(name) {
            self.globals_used.insert(name.to_string());
            return Cl::Global(self.conv.modenv_slot(name));
        }
        if self.bound.contains(name) {
            return match self.cell_of.get(name) {
                Some(cv) => Cl::Cell(cv.clone()),
                None => Cl::Plain,
            };
        }
        if self.is_module && self.module_bound.contains(name) {
            // A module name this scope never binds itself:
            // it exists only because some function declared
            // it `global`. It lives in the module
            // environment (finish_modenv creates its cell).
            self.globals_used.insert(name.to_string());
            return Cl::Global(self.conv.modenv_slot(name));
        }
        match resolve_chain(self.chain, name) {
            1 => Cl::Env(self.env_slot(name)),
            2 => {
                self.globals_used.insert(name.to_string());
                Cl::Global(self.conv.modenv_slot(name))
            }
            _ => Cl::Plain,
        }
    }

    fn env_slot(&mut self, name: &str) -> usize {
        match self.needed.iter().position(|n| n == name) {
            Some(i) => i,
            None => {
                self.needed.push(name.to_string());
                self.needed.len() - 1
            }
        }
    }

    fn cell_expr(cl: &Cl) -> Expr {
        match cl {
            Cl::Cell(cv) => Expr::Name(cv.clone(), ExprSpan::synthetic()),
            Cl::Env(i) => Expr::Index(
                Box::new(Expr::Name(ENV_PARAM.to_string(), ExprSpan::synthetic())),
                Box::new(Expr::Int(*i as i64)),
            ExprSpan::synthetic()),
            Cl::Global(i) => Expr::Index(
                Box::new(Expr::ModEnv),
                Box::new(Expr::Int(*i as i64)),
            ExprSpan::synthetic()),
            Cl::Plain => unreachable!("cell_expr of a plain name"),
        }
    }

    /// P0d (sourcemap integration): the plain-name path keeps the
    /// parsed span, so E1001 and other name-site diagnostics can
    /// point at the name instead of the statement. Captured names
    /// become CellGet, which carries no span (pre-existing D8
    /// limitation: move checks on captured names are line-only
    /// at best).
    fn read_name(&mut self, name: &str, span: ExprSpan) -> Expr {
        match self.classify(name) {
            Cl::Plain => Expr::Name(name.to_string(), span),
            cl => Expr::CellGet(Box::new(Self::cell_expr(&cl))),
        }
    }

    fn bind_stmt(&mut self, name: &str, value: Expr, anchor: SourceAnchor) -> Stmt {
        // The statement is a rewrite of a user statement, so it
        // keeps that statement's anchor; `line` still mirrors
        // `anchor.line` (see ast.rs).
        let line = anchor.line;
        match self.classify(name) {
            Cl::Plain => Stmt::Assign { name: name.to_string(), value, line, anchor },
            cl => Stmt::ExprStmt {
                value: Expr::CellSet(Box::new(Self::cell_expr(&cl)), Box::new(value)),
                line,
                anchor,
            },
        }
    }

    /// The expression that builds a child site's function
    /// value: a plain tag-7 FnVal when the child captures
    /// nothing (the exact D3 shape), otherwise a tag-16
    /// ClosureVal whose environment list holds this scope's
    /// cell for each name the child needs, in the child's
    /// capture order.
    fn site_value(&mut self, cr: &ChildResult) -> Result<Expr, Diagnostic> {
        if cr.needed.is_empty() {
            return Ok(Expr::FnVal(cr.key.clone()));
        }
        let mut elems: Vec<Expr> = Vec::new();
        for n in &cr.needed {
            if let Some(cv) = self.cell_of.get(n) {
                elems.push(Expr::Name(cv.clone(), ExprSpan::synthetic()));
            } else if let Some(i) = self.needed.iter().position(|x| x == n) {
                elems.push(Expr::Index(
                    Box::new(Expr::Name(ENV_PARAM.to_string(), ExprSpan::synthetic())),
                    Box::new(Expr::Int(i as i64)),
                ExprSpan::synthetic()));
            } else {
                return Err(eerr(
                    cr.line,
                    "internal error: a nested function's capture resolves in no enclosing scope",
                ));
            }
        }
        Ok(Expr::ClosureVal(cr.key.clone(), Box::new(Expr::ListLit(elems))))
    }

    fn take_site(&mut self, is_def: bool, line: u32) -> Result<ChildResult, Diagnostic> {
        let cr = self.sites.get(self.site_idx).ok_or_else(|| {
            eerr(line, "internal error: closure conversion ran out of child sites")
        })?;
        if cr.is_def != is_def || cr.line != line {
            return Err(eerr(
                line,
                "internal error: closure conversion site order mismatch (scanner/rewriter traversal diverged)",
            ));
        }
        self.site_idx += 1;
        Ok(ChildResult {
            key: cr.key.clone(),
            needed: cr.needed.clone(),
            is_def: cr.is_def,
            src_name: cr.src_name.clone(),
            line: cr.line,
        })
    }

    fn rewrite_stmts(&mut self, stmts: Vec<Stmt>) -> Result<Vec<Stmt>, Diagnostic> {
        let mut out: Vec<Stmt> = Vec::new();
        for s in stmts {
            out.extend(self.rewrite_stmt(s)?);
        }
        Ok(out)
    }

    fn rewrite_stmt(&mut self, s: Stmt) -> Result<Vec<Stmt>, Diagnostic> {
        let mut pre: Vec<Stmt> = Vec::new();
        let mut main: Vec<Stmt> = Vec::new();
        match s {
            Stmt::Assign { name, value, line, anchor} => {
                let v = self.rewrite_expr(value, &mut pre)?;
                main.push(self.bind_stmt(&name, v, anchor));
            }
            Stmt::AugAssign { name, op, value, line, anchor} => {
                let v = self.rewrite_expr(value, &mut pre)?;
                match self.classify(&name) {
                    Cl::Plain => main.push(Stmt::AugAssign { name, op, value: v, line, anchor}),
                    cl => {
                        let ce = Self::cell_expr(&cl);
                        main.push(Stmt::ExprStmt {
                            value: Expr::CellSet(
                                Box::new(ce.clone()),
                                Box::new(Expr::Bin(
                                    Box::new(Expr::CellGet(Box::new(ce))),
                                    op,
                                    Box::new(v),
                                )),
                            ),
                            line, anchor,
                        });
                    }
                }
            }
            Stmt::IndexAssign { name, index, op, value, line, anchor} => {
                let idx = self.rewrite_expr(index, &mut pre)?;
                let v = self.rewrite_expr(value, &mut pre)?;
                match self.classify(&name) {
                    Cl::Plain => main.push(Stmt::IndexAssign { name, index: idx, op, value: v, line, anchor}),
                    cl => {
                        let tmp = self.conv.fresh("ix");
                        pre.push(Stmt::Assign {
                            name: tmp.clone(),
                            value: Expr::CellGet(Box::new(Self::cell_expr(&cl))),
                            line, anchor,
                        });
                        main.push(Stmt::IndexAssign { name: tmp, index: idx, op, value: v, line, anchor});
                    }
                }
            }
            Stmt::AttrAssign { base, name, op, value, line, anchor} => {
                let b = self.rewrite_expr(base, &mut pre)?;
                let v = self.rewrite_expr(value, &mut pre)?;
                main.push(Stmt::AttrAssign { base: b, name, op, value: v, line, anchor});
            }
            Stmt::AttrDel { base, name, line, anchor} => {
                let b = self.rewrite_expr(base, &mut pre)?;
                main.push(Stmt::AttrDel { base: b, name, line, anchor});
            }
            Stmt::ExprStmt { value, line, anchor} => {
                let v = self.rewrite_expr(value, &mut pre)?;
                main.push(Stmt::ExprStmt { value: v, line, anchor});
            }
            Stmt::If { arms, els, line, anchor} => {
                let mut new_arms: Vec<(Expr, Vec<Stmt>)> = Vec::new();
                for (c, b) in arms {
                    let c2 = self.rewrite_expr(c, &mut pre)?;
                    new_arms.push((c2, self.rewrite_stmts(b)?));
                }
                let els = match els {
                    Some(b) => Some(self.rewrite_stmts(b)?),
                    None => None,
                };
                main.push(Stmt::If { arms: new_arms, els, line, anchor});
            }
            Stmt::While { cond, body, els, line, anchor} => {
                let saved = self.hoist_forbidden;
                self.hoist_forbidden = true;
                let c = self.rewrite_expr(cond, &mut pre);
                self.hoist_forbidden = saved;
                let c = c?;
                let b = self.rewrite_stmts(body)?;
                let els = match els {
                    Some(b) => Some(self.rewrite_stmts(b)?),
                    None => None,
                };
                main.push(Stmt::While { cond: c, body: b, els, line, anchor});
            }
            Stmt::ForRange { var, start, stop, body, els, line, anchor} => {
                let start = match start {
                    Some(x) => Some(self.rewrite_expr(x, &mut pre)?),
                    None => None,
                };
                let stop = self.rewrite_expr(stop, &mut pre)?;
                let els = match els {
                    Some(b) => Some(self.rewrite_stmts(b)?),
                    None => None,
                };
                match self.classify(&var) {
                    Cl::Plain => {
                        let b = self.rewrite_stmts(body)?;
                        main.push(Stmt::ForRange { var, start, stop, body: b, els, line, anchor});
                    }
                    cl => {
                        let tmp = self.conv.fresh("fv");
                        let mut b: Vec<Stmt> = vec![Stmt::ExprStmt {
                            value: Expr::CellSet(
                                Box::new(Self::cell_expr(&cl)),
                                Box::new(Expr::Name(tmp.clone(), ExprSpan::synthetic())),
                            ),
                            line, anchor,
                        }];
                        b.extend(self.rewrite_stmts(body)?);
                        main.push(Stmt::ForRange { var: tmp, start, stop, body: b, els, line, anchor});
                    }
                }
            }
            Stmt::ForEach { vars, iter, body, els, line, anchor} => {
                let it = self.rewrite_expr(iter, &mut pre)?;
                let mut new_vars: Vec<String> = Vec::new();
                let mut binds: Vec<Stmt> = Vec::new();
                for v in &vars {
                    match self.classify(v) {
                        Cl::Plain => new_vars.push(v.clone()),
                        cl => {
                            let tmp = self.conv.fresh("fv");
                            new_vars.push(tmp.clone());
                            binds.push(Stmt::ExprStmt {
                                value: Expr::CellSet(
                                    Box::new(Self::cell_expr(&cl)),
                                    Box::new(Expr::Name(tmp, ExprSpan::synthetic())),
                                ),
                                line, anchor,
                            });
                        }
                    }
                }
                let mut b = binds;
                b.extend(self.rewrite_stmts(body)?);
                let els = match els {
                    Some(b) => Some(self.rewrite_stmts(b)?),
                    None => None,
                };
                main.push(Stmt::ForEach { vars: new_vars, iter: it, body: b, els, line, anchor});
            }
            Stmt::Unpack { names, value, line, anchor} => {
                let v = self.rewrite_expr(value, &mut pre)?;
                let mut new_names: Vec<String> = Vec::new();
                let mut binds: Vec<(String, Cl)> = Vec::new();
                for n in &names {
                    match self.classify(n) {
                        Cl::Plain => new_names.push(n.clone()),
                        cl => {
                            let tmp = self.conv.fresh("up");
                            new_names.push(tmp.clone());
                            binds.push((tmp, cl));
                        }
                    }
                }
                main.push(Stmt::Unpack { names: new_names, value: v, line, anchor});
                for (tmp, cl) in binds {
                    main.push(Stmt::ExprStmt {
                        value: Expr::CellSet(
                            Box::new(Self::cell_expr(&cl)),
                            Box::new(Expr::Name(tmp, ExprSpan::synthetic())),
                        ),
                        line, anchor,
                    });
                }
            }
            Stmt::MultiAssign { names, values, line, anchor} => {
                let mut vs: Vec<Expr> = Vec::new();
                for x in values {
                    vs.push(self.rewrite_expr(x, &mut pre)?);
                }
                if names.iter().all(|n| matches!(self.classify(n), Cl::Plain)) {
                    main.push(Stmt::MultiAssign { names, values: vs, line, anchor});
                } else {
                    // Evaluate every value into a temp first
                    // (CPython evaluates the right-hand side
                    // before any binding), then bind each
                    // name from its temp. The chained form
                    // (one value, many names) binds them all
                    // from the single temp.
                    let mut tmps: Vec<String> = Vec::new();
                    for v in vs {
                        let t = self.conv.fresh("ma");
                        pre.push(Stmt::Assign { name: t.clone(), value: v, line, anchor});
                        tmps.push(t);
                    }
                    if tmps.len() == 1 {
                        for n in &names {
                            let st = self.bind_stmt(n, Expr::Name(tmps[0].clone(), ExprSpan::synthetic()), anchor);
                            main.push(st);
                        }
                    } else {
                        for (n, t) in names.iter().zip(tmps.iter()) {
                            let st = self.bind_stmt(n, Expr::Name(t.clone(), ExprSpan::synthetic()), anchor);
                            main.push(st);
                        }
                    }
                }
            }
            Stmt::Return { value, line, anchor} => {
                let v = match value {
                    Some(x) => Some(self.rewrite_expr(x, &mut pre)?),
                    None => None,
                };
                main.push(Stmt::Return { value: v, line, anchor});
            }
            Stmt::Yield { value, line, anchor} => {
                let v = match value {
                    Some(x) => Some(self.rewrite_expr(x, &mut pre)?),
                    None => None,
                };
                main.push(Stmt::Yield { value: v, line, anchor});
            }
            Stmt::Raise { value, line, anchor} => {
                let v = match value {
                    Some(x) => Some(self.rewrite_expr(x, &mut pre)?),
                    None => None,
                };
                main.push(Stmt::Raise { value: v, line, anchor});
            }
            Stmt::Assert { cond, line, anchor, msg } => {
                let c = self.rewrite_expr(cond, &mut pre)?;
                main.push(Stmt::Assert { cond: c, line, anchor, msg });
            }
            Stmt::Del { targets, line, anchor} => {
                let mut new_targets: Vec<(String, Option<Expr>)> = Vec::new();
                for (n, idx) in targets {
                    if !matches!(self.classify(&n), Cl::Plain) {
                        return Err(eerr(
                            line,
                            &format!(
                                "`del {}` on a captured, global or nonlocal variable is not supported yet (D8 v1 closures)",
                                n
                            ),
                        ));
                    }
                    let idx = match idx {
                        Some(x) => Some(self.rewrite_expr(x, &mut pre)?),
                        None => None,
                    };
                    new_targets.push((n, idx));
                }
                main.push(Stmt::Del { targets: new_targets, line, anchor});
            }
            Stmt::Try { body, handlers, els, finally, line, anchor} => {
                let b = self.rewrite_stmts(body)?;
                let mut hs: Vec<ExcHandler> = Vec::new();
                for h in handlers {
                    let (bind, hb) = match h.bind {
                        Some(n) => match self.classify(&n) {
                            Cl::Plain => (Some(n), self.rewrite_stmts(h.body)?),
                            cl => {
                                let tmp = self.conv.fresh("ex");
                                let mut nb: Vec<Stmt> = vec![Stmt::ExprStmt {
                                    value: Expr::CellSet(
                                        Box::new(Self::cell_expr(&cl)),
                                        Box::new(Expr::Name(tmp.clone(), ExprSpan::synthetic())),
                                    ),
                                    line: h.line, anchor: SourceAnchor::at(h.line, 0, None),
                                }];
                                nb.extend(self.rewrite_stmts(h.body)?);
                                (Some(tmp), nb)
                            }
                        },
                        None => (None, self.rewrite_stmts(h.body)?),
                    };
                    hs.push(ExcHandler { classes: h.classes, bind, body: hb, line: h.line, anchor: h.anchor });
                }
                let els = match els {
                    Some(b2) => Some(self.rewrite_stmts(b2)?),
                    None => None,
                };
                let finally = match finally {
                    Some(b2) => Some(self.rewrite_stmts(b2)?),
                    None => None,
                };
                main.push(Stmt::Try { body: b, handlers: hs, els, finally, line, anchor});
            }
            Stmt::With { items, body, line, anchor} => {
                let mut new_items: Vec<(Expr, Option<String>)> = Vec::new();
                let mut binds: Vec<Stmt> = Vec::new();
                for (e, bind) in items {
                    let e2 = self.rewrite_expr(e, &mut pre)?;
                    match bind {
                        Some(n) => match self.classify(&n) {
                            Cl::Plain => new_items.push((e2, Some(n))),
                            cl => {
                                let tmp = self.conv.fresh("wv");
                                new_items.push((e2, Some(tmp.clone())));
                                binds.push(Stmt::ExprStmt {
                                    value: Expr::CellSet(
                                        Box::new(Self::cell_expr(&cl)),
                                        Box::new(Expr::Name(tmp, ExprSpan::synthetic())),
                                    ),
                                    line, anchor,
                                });
                            }
                        },
                        None => new_items.push((e2, None)),
                    }
                }
                let mut b = binds;
                b.extend(self.rewrite_stmts(body)?);
                main.push(Stmt::With { items: new_items, body: b, line, anchor});
            }
            Stmt::Match { subject, cases, line, anchor} => {
                let subj = self.rewrite_expr(subject, &mut pre)?;
                let mut new_cases: Vec<MatchCase> = Vec::new();
                for c in cases {
                    let saved = self.hoist_forbidden;
                    self.hoist_forbidden = true;
                    let pat = self.rewrite_pattern(c.pattern, &mut pre);
                    let guard = match c.guard {
                        Some(g) => match self.rewrite_expr(g, &mut pre) {
                            Ok(g2) => Some(g2),
                            Err(e) => {
                                self.hoist_forbidden = saved;
                                return Err(e);
                            }
                        },
                        None => None,
                    };
                    self.hoist_forbidden = saved;
                    let pat = pat?;
                    let b = self.rewrite_stmts(c.body)?;
                    new_cases.push(MatchCase { pattern: pat, guard, body: b, line: c.line, anchor: c.anchor });
                }
                main.push(Stmt::Match { subject: subj, cases: new_cases, line, anchor});
            }
            Stmt::Def(f) => {
                let cr = self.take_site(true, f.line)?;
                if cr.src_name != f.name {
                    return Err(eerr(
                        f.line,
                        "internal error: closure conversion def site name mismatch",
                    ));
                }
                let v = self.site_value(&cr)?;
                main.push(self.bind_stmt(&f.name, v, f.anchor));
            }
            Stmt::Global { .. } | Stmt::Nonlocal { .. } => {
                // Consumed by the scope analysis; the
                // declaration itself emits nothing.
            }
            Stmt::Break { line, anchor} => main.push(Stmt::Break { line, anchor}),
            Stmt::Continue { line, anchor} => main.push(Stmt::Continue { line, anchor}),
            Stmt::Pass { line, anchor} => main.push(Stmt::Pass { line, anchor}),
            Stmt::PublishModEnv { .. } => {
                return Err(eerr(
                    0,
                    "internal error: PublishModEnv reached the closure rewriter (it is produced after rewriting)",
                ));
            }
        }
        pre.extend(main);
        Ok(pre)
    }

    fn rewrite_pattern(&mut self, p: Pattern, pre: &mut Vec<Stmt>) -> Result<Pattern, Diagnostic> {
        Ok(match p {
            Pattern::Lit(e) => Pattern::Lit(self.rewrite_expr(e, pre)?),
            Pattern::Seq(pats) => {
                let mut out: Vec<Pattern> = Vec::new();
                for x in pats {
                    out.push(self.rewrite_pattern(x, pre)?);
                }
                Pattern::Seq(out)
            }
            Pattern::Or(pats) => {
                let mut out: Vec<Pattern> = Vec::new();
                for x in pats {
                    out.push(self.rewrite_pattern(x, pre)?);
                }
                Pattern::Or(out)
            }
            other => other,
        })
    }

    /// Rewrite one f-string part: the field's value expression
    /// and, recursively, the expressions inside its format
    /// spec. Conversion, debug prefix and literal parts pass
    /// through unchanged.
    fn rewrite_fstr_part(
        &mut self,
        p: FStrPart,
        pre: &mut Vec<Stmt>,
    ) -> Result<FStrPart, Diagnostic> {
        match p {
            FStrPart::Lit(s) => Ok(FStrPart::Lit(s)),
            FStrPart::Field { value, conv, debug, spec } => {
                let spec = match spec {
                    Some(parts) => {
                        let mut out = Vec::new();
                        for sp in parts {
                            out.push(self.rewrite_fstr_part(sp, pre)?);
                        }
                        Some(out)
                    }
                    None => None,
                };
                Ok(FStrPart::Field {
                    value: self.rewrite_expr(value, pre)?,
                    conv,
                    debug,
                    spec,
                })
            }
        }
    }

    fn rewrite_expr(&mut self, e: Expr, pre: &mut Vec<Stmt>) -> Result<Expr, Diagnostic> {
        Ok(match e {
            Expr::Name(n, span) => self.read_name(&n, span),
            Expr::Bin(a, op, b) => Expr::Bin(
                Box::new(self.rewrite_expr(*a, pre)?),
                op,
                Box::new(self.rewrite_expr(*b, pre)?),
            ),
            Expr::Unary(op, a) => Expr::Unary(op, Box::new(self.rewrite_expr(*a, pre)?)),
            Expr::Call(f, args, kwargs, span) => {
                let f2 = self.rewrite_expr(*f, pre)?;
                let mut a2: Vec<Expr> = Vec::new();
                for a in args {
                    a2.push(self.rewrite_expr(a, pre)?);
                }
                let mut k2: Vec<(String, Expr)> = Vec::new();
                for (k, v) in kwargs {
                    k2.push((k, self.rewrite_expr(v, pre)?));
                }
                Expr::Call(Box::new(f2), a2, k2, span)
            }
            Expr::ListLit(es) => {
                let mut out: Vec<Expr> = Vec::new();
                for x in es {
                    out.push(self.rewrite_expr(x, pre)?);
                }
                Expr::ListLit(out)
            }
            Expr::TupleLit(es) => {
                let mut out: Vec<Expr> = Vec::new();
                for x in es {
                    out.push(self.rewrite_expr(x, pre)?);
                }
                Expr::TupleLit(out)
            }
            Expr::SetLit(es) => {
                let mut out: Vec<Expr> = Vec::new();
                for x in es {
                    out.push(self.rewrite_expr(x, pre)?);
                }
                Expr::SetLit(out)
            }
            Expr::DictLit(pairs) => {
                let mut out: Vec<(Expr, Expr)> = Vec::new();
                for (k, v) in pairs {
                    out.push((self.rewrite_expr(k, pre)?, self.rewrite_expr(v, pre)?));
                }
                Expr::DictLit(out)
            }
            Expr::Index(a, b, span) => Expr::Index(
                Box::new(self.rewrite_expr(*a, pre)?),
                Box::new(self.rewrite_expr(*b, pre)?),
                span,
            ),
            Expr::Slice(a, lo, hi, st) => {
                let a2 = self.rewrite_expr(*a, pre)?;
                let mut parts: Vec<Option<Box<Expr>>> = Vec::new();
                for o in [lo, hi, st] {
                    parts.push(match o {
                        Some(x) => Some(Box::new(self.rewrite_expr(*x, pre)?)),
                        None => None,
                    });
                }
                let mut it = parts.into_iter();
                Expr::Slice(
                    Box::new(a2),
                    it.next().unwrap(),
                    it.next().unwrap(),
                    it.next().unwrap(),
                )
            }
            Expr::MethodCall { obj, method, args } => {
                // The object is classified (and, when it
                // lives in a cell, read into a temp) BEFORE
                // the arguments are rewritten: CPython
                // evaluates the object before the arguments,
                // and the hoisted read must observe the same
                // binding.
                let obj2 = match self.classify(&obj) {
                    Cl::Plain => obj,
                    cl => {
                        if self.hoist_forbidden {
                            return Err(eerr(
                                0,
                                &format!(
                                    "method call on captured variable `{}` in a while condition, match guard or match pattern is not supported yet (D8 v1 closures); bind it to a plain local first",
                                    obj
                                ),
                            ));
                        }
                        let tmp = self.conv.fresh("mc");
                        pre.push(Stmt::Assign {
                            name: tmp.clone(),
                            value: Expr::CellGet(Box::new(Self::cell_expr(&cl))),
                            line: 0, anchor: SourceAnchor::synthetic(),
                        });
                        tmp
                    }
                };
                let mut a2: Vec<Expr> = Vec::new();
                for a in args {
                    a2.push(self.rewrite_expr(a, pre)?);
                }
                Expr::MethodCall { obj: obj2, method, args: a2 }
            }
            Expr::Attr(b, n, span) => Expr::Attr(Box::new(self.rewrite_expr(*b, pre)?), n, span),
            Expr::Comp { kind, key, value, clauses } => {
                let mut frame: std::collections::HashSet<String> = HashSet::new();
                for c in &clauses {
                    if let CompClause::For { var, .. } = c {
                        frame.insert(var.clone());
                    }
                }
                self.shadow.push(frame);
                let r: Result<Expr, Diagnostic> = (|| {
                    let k2 = self.rewrite_expr(*key, pre)?;
                    let v2 = match value {
                        Some(v) => Some(Box::new(self.rewrite_expr(*v, pre)?)),
                        None => None,
                    };
                    let mut c2: Vec<CompClause> = Vec::new();
                    for c in clauses {
                        c2.push(match c {
                            CompClause::For { var, iter } => {
                                CompClause::For { var, iter: self.rewrite_expr(iter, pre)? }
                            }
                            CompClause::If(cond) => {
                                CompClause::If(self.rewrite_expr(cond, pre)?)
                            }
                        });
                    }
                    Ok(Expr::Comp { kind, key: Box::new(k2), value: v2, clauses: c2 })
                })();
                self.shadow.pop();
                r?
            }
            Expr::IfExp { cond, then, els } => Expr::IfExp {
                cond: Box::new(self.rewrite_expr(*cond, pre)?),
                then: Box::new(self.rewrite_expr(*then, pre)?),
                els: Box::new(self.rewrite_expr(*els, pre)?),
            },
            Expr::FStr(parts) => {
                let mut out: Vec<FStrPart> = Vec::new();
                for p in parts {
                    out.push(self.rewrite_fstr_part(p, pre)?);
                }
                Expr::FStr(out)
            }
            Expr::Await(a) => Expr::Await(Box::new(self.rewrite_expr(*a, pre)?)),
            Expr::Lambda { line, .. } => {
                let cr = self.take_site(false, line)?;
                self.site_value(&cr)?
            }
            Expr::CellNew(a) => Expr::CellNew(Box::new(self.rewrite_expr(*a, pre)?)),
            Expr::CellGet(a) => Expr::CellGet(Box::new(self.rewrite_expr(*a, pre)?)),
            Expr::CellSet(a, b) => Expr::CellSet(
                Box::new(self.rewrite_expr(*a, pre)?),
                Box::new(self.rewrite_expr(*b, pre)?),
            ),
            Expr::ClosureVal(k, env) => {
                Expr::ClosureVal(k, Box::new(self.rewrite_expr(*env, pre)?))
            }
            Expr::Walrus { name, value } => {
                // The value rewrites first: its reads observe
                // the binding state before the store, as in
                // CPython. A plain target keeps the Walrus node
                // (emission binds it through the plain
                // assignment machinery). A cell / environment /
                // global target must NOT stay a Walrus —
                // emission would bind a plain local of that
                // name and every closure would keep reading
                // the stale cell — so the write becomes a
                // CellSet, which evaluates to the stored value:
                // exactly the walrus's yield. (A target shadowed
                // by a comprehension frame classifies Plain and
                // stays a Walrus; emit_comp refuses it if it
                // names the iteration variable.)
                let v = self.rewrite_expr(*value, pre)?;
                match self.classify(&name) {
                    Cl::Plain => Expr::Walrus { name, value: Box::new(v) },
                    cl => Expr::CellSet(Box::new(Self::cell_expr(&cl)), Box::new(v)),
                }
            }
            other => other,
        })
    }
}

/// D8: match-pattern capture names anywhere in a statement
/// list (child scopes not entered), for the cell check in
/// convert_scope.
fn walk_stmts_match_captures(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
    for s in stmts {
        match s {
            Stmt::Match { cases, .. } => {
                for c in cases {
                    for n in pattern_captures(&c.pattern) {
                        out.insert(n);
                    }
                    walk_stmts_match_captures(&c.body, out);
                }
            }
            Stmt::If { arms, els, .. } => {
                for (_, b) in arms {
                    walk_stmts_match_captures(b, out);
                }
                if let Some(b) = els {
                    walk_stmts_match_captures(b, out);
                }
            }
            Stmt::While { body, els, .. }
            | Stmt::ForRange { body, els, .. }
            | Stmt::ForEach { body, els, .. } => {
                walk_stmts_match_captures(body, out);
                if let Some(b) = els {
                    walk_stmts_match_captures(b, out);
                }
            }
            Stmt::With { body, .. } => walk_stmts_match_captures(body, out),
            Stmt::Try { body, handlers, els, finally, .. } => {
                walk_stmts_match_captures(body, out);
                for h in handlers {
                    walk_stmts_match_captures(&h.body, out);
                }
                if let Some(b) = els {
                    walk_stmts_match_captures(b, out);
                }
                if let Some(b) = finally {
                    walk_stmts_match_captures(b, out);
                }
            }
            _ => {}
        }
    }
}

/// D4 class registry: validate the class definitions, flatten
/// each class's methods and attributes over its base chain (the
/// base must be defined before the subclass, as in Python's
/// execution order), and register every method as an ordinary
/// def under the key `ClassName.method` with an assigned SLA
/// name. Class attributes are restricted to literals in v1:
/// the fallback shape evaluates them at each class-side read,
/// which is exact for literals; computed attributes wait for
/// the runtime-registry shape (mutable-global probe pending,
/// see D4-NOTES).
fn build_class_registry(em: &mut Emitter, prog: &Program) -> Result<(), Diagnostic> {
    for c in &prog.classes {
        if em.class_ids.contains_key(&c.name) {
            return Err(eerr(c.line, &format!("duplicate class definition `{}`", c.name)));
        }
        if em.defs.contains_key(&c.name) {
            return Err(eerr(
                c.line,
                &format!("class `{}` collides with a function of the same name (D4 v1)", c.name),
            ));
        }
        if matches!(c.name.as_str(), "print" | "range" | "len") {
            return Err(eerr(
                c.line,
                &format!("class name `{}` collides with a builtin (D4 v1)", c.name),
            ));
        }
        let (base_methods, base_attrs): (Vec<(String, String)>, Vec<(String, Expr)>) =
            match &c.base {
                Some(b) => {
                    let bid = em.class_ids.get(b).ok_or_else(|| {
                        eerr(
                            c.line,
                            &format!(
                                "base class `{}` of class `{}` is not defined yet (define the base before the subclass)",
                                b, c.name
                            ),
                        )
                    })?;
                    let bi = &em.classes[*bid];
                    (bi.flat_methods.clone(), bi.flat_attrs.clone())
                }
                None => (Vec::new(), Vec::new()),
            };
        for (an, ae) in &c.attrs {
            match ae {
                Expr::Int(_) | Expr::BigInt(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bool(_) | Expr::NoneLit => {}
                _ => {
                    return Err(eerr(
                        c.line,
                        &format!(
                            "class attribute `{}` of class `{}` must be a scalar literal in v1 (the runtime registry stores literals; computed class attributes are refused)",
                            an, c.name
                        ),
                    ))
                }
            }
        }
        let id = em.classes.len() as i64;
        // Flattened attributes: own first (a repeated assignment
        // keeps the last value, as in Python), then the base's
        // attributes this class does not shadow.
        let mut flat_attrs: Vec<(String, Expr)> = Vec::new();
        for (an, ae) in &c.attrs {
            if let Some(slot) = flat_attrs.iter_mut().find(|(n, _)| n == an) {
                slot.1 = ae.clone();
            } else {
                flat_attrs.push((an.clone(), ae.clone()));
            }
        }
        for (an, ae) in &base_attrs {
            if !flat_attrs.iter().any(|(n, _)| n == an) {
                flat_attrs.push((an.clone(), ae.clone()));
            }
        }
        // Flattened methods: the base's table with own methods
        // overriding by name (a repeated method keeps the last
        // definition, as in Python).
        let mut flat_methods = base_methods;
        for m in &c.methods {
            let key = format!("{}.{}", c.name, m.name);
            let mut mf = m.clone();
            mf.name = key.clone();
            em.defs.insert(key.clone(), mf);
            em.sla_names
                .insert(key.clone(), format!("__py_cls_{}_{}", id, m.name));
            if let Some(slot) = flat_methods.iter_mut().find(|(n, _)| *n == m.name) {
                slot.1 = key;
            } else {
                flat_methods.push((m.name.clone(), key));
            }
        }
        let mut ancestors: Vec<i64> = vec![id];
        if let Some(b) = &c.base {
            if let Some(&bid) = em.class_ids.get(b) {
                ancestors.extend_from_slice(&em.classes[bid].ancestors);
            }
        }
        em.class_ids.insert(c.name.clone(), em.classes.len());
        em.classes.push(ClassInfo { id, name: c.name.clone(), flat_methods, flat_attrs, attrs: c.attrs.clone(), ancestors });
    }
    Ok(())
}

/// A list literal's emission plan: a monomorphic static Vec, or a
/// dynamic (tagged) heap list when the elements are dynamic, of
/// mixed scalar types, floats (Vec<f64> is miscompiled upstream,
/// defect (xxiv); the tagged payload carries f64 exactly), or the
/// literal is empty (no static element type to infer).
enum ListLit {
    Static(Ty, Vec<String>),
    Dyn(Vec<String>),
}

/// D6: does this statement list use try/raise/with anywhere
/// (nested blocks included)? Drives Emitter::program_exc.
fn stmts_use_exc(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Try { .. } | Stmt::Raise { .. } | Stmt::With { .. } => true,
        Stmt::If { arms, els, .. } => {
            arms.iter().any(|(_, b)| stmts_use_exc(b))
                || els.as_ref().map_or(false, |b| stmts_use_exc(b))
        }
        Stmt::While { body, els, .. }
        | Stmt::ForRange { body, els, .. }
        | Stmt::ForEach { body, els, .. } => {
            stmts_use_exc(body)
                || els.as_ref().map_or(false, |b| stmts_use_exc(b))
        }
        _ => false,
    })
}

/// D6: does executing this statement evaluate a call?
/// (expr_has_call already exists below for the hoisting
/// analysis; it is a superset of what propagation needs —
/// it also counts the //, %, ** helper lowerings, which
/// only ever adds a harmless extra pending check.)
/// (Try/With/Raise are handled by their own machinery and
/// are excluded by the caller.)
fn stmt_may_call(s: &Stmt) -> bool {
    match s {
        Stmt::Assign { value, .. } => expr_has_call(value),
        Stmt::AugAssign { value, .. } => expr_has_call(value),
        Stmt::IndexAssign { index, value, .. } => {
            expr_has_call(index) || expr_has_call(value)
        }
        Stmt::AttrAssign { base, value, .. } => {
            expr_has_call(base) || expr_has_call(value)
        }
        Stmt::AttrDel { base, .. } => expr_has_call(base),
        Stmt::ExprStmt { value, .. } => expr_has_call(value),
        Stmt::If { arms, .. } => arms.iter().any(|(c, _)| expr_has_call(c)),
        Stmt::While { cond, .. } => expr_has_call(cond),
        Stmt::ForRange { start, stop, .. } => {
            start.as_ref().map_or(false, expr_has_call) || expr_has_call(stop)
        }
        Stmt::ForEach { iter, .. } => expr_has_call(iter),
        Stmt::Unpack { value, .. } => expr_has_call(value),
        Stmt::Assert { cond, .. } => expr_has_call(cond),
        Stmt::Del { targets, .. } => targets
            .iter()
            .any(|(_, i)| i.as_ref().map_or(false, expr_has_call)),
        Stmt::MultiAssign { values, .. } => values.iter().any(expr_has_call),
        Stmt::Match { subject, .. } => expr_has_call(subject),
        _ => false,
    }
}

impl Emitter {
    /// D3 function values: the code for using function `name` as
    /// a value — a dynamic callable (PsaVal tag 7) carrying the
    /// function's callable id. The function is pinned to an
    /// all-Dyn signature: if it has not been emitted yet it is
    /// emitted now with Dyn parameters; if it was already
    /// emitted with a non-dynamic signature, the value use is a
    /// loud error. Functions with defaults or collection
    /// parameters cannot be values in v1 (an adapter could not
    /// bind them); direct calls to them are unaffected.
    fn fn_val_code(&mut self, name: &str, line: u32) -> Result<(String, Ty), Diagnostic> {
        let def = match self.defs.get(name) {
            Some(d) => d.clone(),
            None => return Err(eerr(line, &format!("undefined function `{}`", name))),
        };
        if def.vararg.is_some()
            || def.kwarg.is_some()
            || def.params.iter().any(|(_, d)| d.is_some())
        {
            return Err(eerr(
                line,
                &format!(
                    "function `{}` has defaults or *args/**kwargs parameters and cannot be used as a value yet (D3 v1 callable slice)",
                    name
                ),
            ));
        }
        let n = def.param_count();
        self.ensure_fn(name, &vec![Ty::Dyn; n], line)?;
        let sig = self.sigs.get(name).expect("ensure_fn registers a signature");
        // Parameters must be all-dynamic: the generated
        // adapter passes PsaVal arguments straight through.
        // The return may be dynamic, a static scalar or
        // void: the adapter normalises it (box int / float
        // / bool, None for void), exactly as it does for
        // method callables.
        let ret_ok = matches!(
            sig.ret,
            Some(Ty::Dyn)
                | Some(Ty::Int)
                | Some(Ty::Float)
                | Some(Ty::Bool)
                | Some(Ty::Void)
        );
        if sig.params.iter().any(|t| *t != Ty::Dyn) || !ret_ok {
            return Err(eerr(
                line,
                &format!(
                    "function `{}` cannot be used as a value: callable functions are pinned to all-dynamic parameters, but it was already emitted with a different one",
                    name
                ),
            ));
        }
        self.used_dyn = true;
        let id = self.register_callable(name);
        Ok((format!("__psa_box_fn({})", id), Ty::Dyn))
    }

    /// The SLA name a def is emitted under: the D3 `__py_`
    /// mangling, except class methods, whose registry-assigned
    /// names (`__py_cls_<id>_<method>`) are injective where the
    /// dotted def key is not a valid identifier.
    fn sla_name(&self, name: &str) -> String {
        match self.sla_names.get(name) {
            Some(s) => s.clone(),
            None => sla_fn_name(name),
        }
    }

    /// Assign (or fetch) the callable id of a def.
    fn register_callable(&mut self, name: &str) -> i64 {
        match self.fn_ids.get(name) {
            Some(i) => *i,
            None => {
                let i = self.callables.len() as i64;
                self.fn_ids.insert(name.to_string(), i);
                self.callables.push(name.to_string());
                i
            }
        }
    }

    /// D8 closures: the code for a closure value — a tag-16
    /// PsaVal carrying the def's closure-callable id and the
    /// environment (a dynamic list of cells). Mirrors
    /// fn_val_code: the def is pinned to an all-Dyn signature
    /// (the hidden environment parameter included) and
    /// defaults / *args / **kwargs are refused loudly,
    /// because the generated adapter binds positionally.
    fn closure_val_code(
        &mut self,
        name: &str,
        env_code: &str,
        line: u32,
    ) -> Result<(String, Ty), Diagnostic> {
        let def = match self.defs.get(name) {
            Some(d) => d.clone(),
            None => return Err(eerr(line, &format!("undefined function `{}`", name))),
        };
        if def.vararg.is_some()
            || def.kwarg.is_some()
            || def.params.iter().any(|(_, d)| d.is_some())
        {
            return Err(eerr(
                line,
                &format!(
                    "function `{}` has defaults or *args/**kwargs parameters and cannot be used as a closure value yet (D8 v1 closure slice)",
                    name
                ),
            ));
        }
        let n = def.param_count();
        self.ensure_fn(name, &vec![Ty::Dyn; n], line)?;
        let sig = self.sigs.get(name).expect("ensure_fn registers a signature");
        let ret_ok = matches!(
            sig.ret,
            Some(Ty::Dyn)
                | Some(Ty::Int)
                | Some(Ty::Float)
                | Some(Ty::Bool)
                | Some(Ty::Void)
        );
        if sig.params.iter().any(|t| *t != Ty::Dyn) || !ret_ok {
            return Err(eerr(
                line,
                &format!(
                    "function `{}` cannot be used as a closure value: closure functions are pinned to all-dynamic parameters, but it was already emitted with a different one",
                    name
                ),
            ));
        }
        self.used_dyn = true;
        self.used_closure = true;
        let id = self.register_closure_callable(name);
        Ok((format!("__psa_box_closure({}, {})", id, env_code), Ty::Dyn))
    }

    /// Assign (or fetch) the closure-callable id of a def
    /// (a separate id space from the plain callables).
    fn register_closure_callable(&mut self, name: &str) -> i64 {
        match self.fn_cids.get(name) {
            Some(i) => *i,
            None => {
                let i = self.closure_callables.len() as i64;
                self.fn_cids.insert(name.to_string(), i);
                self.closure_callables.push(name.to_string());
                i
            }
        }
    }

    /// D4: pin a method def as a callable and return its id.
    /// Like fn_val_code, the method is pinned to an all-Dyn
    /// signature and defaults/*args/**kwargs are refused (an
    /// adapter cannot bind them; explicit base calls still can,
    /// through the ordinary def-call path). Unlike fn_val_code,
    /// the return may be Void or a static scalar -- the adapter
    /// normalises it to a dynamic value -- because methods are
    /// routinely written for effect (`__init__`, mutators).
    fn method_callable_id(&mut self, key: &str, line: u32) -> Result<i64, Diagnostic> {
        let def = match self.defs.get(key) {
            Some(d) => d.clone(),
            None => return Err(eerr(line, &format!("undefined method `{}`", key))),
        };
        if def.vararg.is_some()
            || def.kwarg.is_some()
            || def.params.iter().any(|(_, d)| d.is_some())
        {
            return Err(eerr(
                line,
                &format!(
                    "method `{}` has defaults or *args/**kwargs parameters and cannot be bound as a method value yet (D4 v1 method slice; explicit base calls still work)",
                    key
                ),
            ));
        }
        let n = def.param_count();
        self.ensure_fn(key, &vec![Ty::Dyn; n], line)?;
        let sig = self.sigs.get(key).expect("ensure_fn registers a signature");
        if sig.params.iter().any(|t| *t != Ty::Dyn) {
            return Err(eerr(
                line,
                &format!(
                    "method `{}` cannot be bound: it was already emitted with a non-dynamic signature",
                    key
                ),
            ));
        }
        match sig.ret {
            Some(Ty::Dyn) | Some(Ty::Void) | Some(Ty::Int) | Some(Ty::Float) | Some(Ty::Bool) => {}
            other => {
                return Err(eerr(
                    line,
                    &format!(
                        "method `{}` returns {:?}, which cannot be normalised to a dynamic value (D4 v1)",
                        key, other
                    ),
                ))
            }
        }
        self.used_dyn = true;
        Ok(self.register_callable(key))
    }

    /// Emit call arguments as dynamic values (the packing the
    /// callable paths share): string literals become dynamic
    /// strings, container literals become container temps,
    /// everything else is boxed at the boundary.
    fn emit_dyn_args(
        &mut self,
        ctx: &mut Ctx,
        args: &[Expr],
        line: u32,
    ) -> Result<Vec<String>, Diagnostic> {
        let mut codes: Vec<String> = Vec::new();
        for a in args {
            codes.push(self.emit_dyn_value(ctx, a, line)?);
        }
        Ok(codes)
    }

    /// D7: install a generator def's callable face. The def's
    /// own SLA name becomes a thin wrapper: create the
    /// rendezvous channel, spawn the drive function with the
    /// arguments, and return the boxed state. The drive
    /// function runs the producer body and then sends the end
    /// wrap, so every return path of the body (including the
    /// D6 flag machinery unwinding through finally blocks)
    /// terminates the generator. The producer is the def body
    /// itself, emitted with all-Dyn parameters and the channel
    /// in scope as `__psa_gench`.
    fn emit_generator_def(&mut self, def: &Func) -> Result<(), Diagnostic> {
        let name = def.name.clone();
        gen_check_returns(&def.body, &name, def.line)?;
        let params = def.param_names();
        let n = def.param_count();
        self.sigs.insert(
            name.clone(),
            Sig {
                params: vec![Ty::Dyn; n],
                param_borrow: vec![false; n],
                ret: Some(Ty::Dyn),
            },
        );
        self.used_dyn = true;
        self.used_gen = true;
        let base = self.sla_name(&name);
        let prod_name = format!("{}__prod", base);
        let drive_name = format!("{}__drive", base);
        let args_joined = params.join(", ");
        let mut w = String::new();
        w.push_str(&format!("fn {}(", base));
        for (i, p) in params.iter().enumerate() {
            if i > 0 {
                w.push_str(", ");
            }
            w.push_str(&format!("{}: PsaVal", p));
        }
        w.push_str(") -> PsaVal {\n");
        w.push_str("    let __psa_ch: Chan<i64> = Chan::new(0);\n");
        w.push_str(&format!("    let __psa_h = thread::spawn(^|| {}(__psa_ch", drive_name));
        if !params.is_empty() {
            w.push_str(", ");
            w.push_str(&args_joined);
        }
        w.push_str("));\n");
        w.push_str("    return __psa_gen_new(__psa_ch, __psa_h);\n}\n");
        // P1: the wrapper is one synthetic function chunk anchored
        // to the generator def (precision Function).
        self.emitted.push(EmittedChunk::span(
            w,
            &SegMeta::synthetic_fn(def.anchor, base.clone()),
        ));
        self.emitted_set.push(name.clone());
        let mut d = String::new();
        d.push_str(&format!("fn {}(__psa_ch: Chan<i64>", drive_name));
        for p in &params {
            d.push_str(&format!(", {}: PsaVal", p));
        }
        d.push_str(") -> PsaVal {\n");
        // Laziness: the producer body must not run before the
        // first pull (CPython runs nothing at call time), so
        // the drive waits for the start token __psa_gen_pull
        // sends on the state's first pull.
        d.push_str("    let (__psa_start, __psa_sclosed) = __psa_ch.recv();\n");
        d.push_str(&format!("    let __psa_r = {}(__psa_ch", prod_name));
        if !params.is_empty() {
            d.push_str(", ");
            d.push_str(&args_joined);
        }
        d.push_str(");\n");
        d.push_str("    let __psa_endw = __psa_gen_wrap_end();\n");
        d.push_str("    __psa_ch.send(__psa_endw.ipay);\n");
        d.push_str("    return __psa_r;\n}\n");
        // P1: the drive wrapper is one synthetic function chunk
        // anchored to the generator def (precision Function).
        self.emitted.push(EmittedChunk::span(
            d,
            &SegMeta::synthetic_fn(def.anchor, drive_name.clone()),
        ));
        let (prod_txt, prod_map) = self.emit_gen_producer(def, &prod_name)?;
        self.emitted.push(EmittedChunk { text: prod_txt, map: prod_map });
        Ok(())
    }

    /// D7: emit a generator def's body as its producer function.
    /// Parameters are dynamic values; `yield` statements send
    /// their wrap on `__psa_gench` (Ctx::gen_chan). Returns may
    /// only be bare / None (checked by gen_check_returns).
    fn emit_gen_producer(&mut self, def: &Func, prod_sla: &str) -> Result<(String, SourceMap), Diagnostic> {
        let param_vars: Vec<(String, Ty, Option<SourceAnchor>)> = def
            .param_names()
            .into_iter()
            .map(|n| (n, Ty::Dyn, None))
            .collect();
        let mut ctx = Ctx {
            vars: param_vars,
            sources: self.sources.clone(),
            scopes: vec![0],
            ret: None,
            fname: def.name.clone(),
            out: SegmentWriter::new(),
            prologue: SegmentWriter::new(),
            indent: 1,
            tmp: 0,
            borrowed: Vec::new(),
            in_while_cond: false,
            dyn_ret: true,
            deleted: Vec::new(),
            try_stack: Vec::new(),
            guard_depth: 0,
            loop_depth: 0,
            exc_slot: false,
            in_finally: 0,
            exc_bare: None,
            gen_chan: Some("__psa_gench".to_string()),
            anchor: SourceAnchor::synthetic(),
            anchor_kind: SegKind::Stmt,
            anchor_precision: Precision::None,
            anchor_name: None,
            anchor_origin: Origin::Synthetic,
            pending_call: false,
            pending_call_name: None,
        };
        for s in &def.body {
            self.emit_stmt(&mut ctx, s)?;
        }
        // P1 source map: same shape as `ensure_fn` — the producer is
        // the user generator def's body under its `__prod` name.
        let last_is_return = ctx
            .out
            .text
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .map(|l| l.trim_start().starts_with("return "))
            .unwrap_or(false);
        let mut head = format!("fn {}(__psa_gench: Chan<i64>", prod_sla);
        for p in def.param_names() {
            head.push_str(&format!(", {}: PsaVal", p));
        }
        head.push_str(") -> PsaVal {");
        let def_anchor = def.anchor;
        let mut txt = SegmentWriter::new();
        txt.push_line(&head, &SegMeta::fn_header(def_anchor, prod_sla.to_string()));
        txt.append_writer(std::mem::take(&mut ctx.prologue));
        txt.append_writer(std::mem::take(&mut ctx.out));
        if !last_is_return {
            txt.push_line(
                &format!("    return {};", Ty::Dyn.default_lit()),
                &SegMeta::synthetic_stmt(def_anchor),
            );
        }
        txt.push_line("}", &SegMeta::fn_header(def_anchor, prod_sla.to_string()));
        let (text, segments, panics) = txt.into_parts();
        Ok((text, SourceMap::from_segments(segments, panics)))
    }

    /// D7: install an async def's callable face. The body is
    /// registered as a worker def under a `::coro` key and
    /// pinned all-Dyn as a callable; the def's own SLA name
    /// becomes a wrapper that packs the arguments into a list
    /// and returns a lazy coroutine value (tag 14). Calling an
    /// async def never runs the body; `await` does
    /// (__psa_coro_run). Defaults and *args/**kwargs are
    /// refused: the callable adapter cannot bind them.
    fn emit_async_def(&mut self, def: &Func) -> Result<(), Diagnostic> {
        let name = def.name.clone();
        if def.vararg.is_some()
            || def.kwarg.is_some()
            || def.params.iter().any(|(_, d)| d.is_some())
        {
            return Err(eerr(
                def.line,
                &format!(
                    "async function `{}` uses defaults or *args/**kwargs parameters, which the coroutine adapter cannot bind in psa v1 (D7)",
                    name
                ),
            ));
        }
        let params = def.param_names();
        let n = def.param_count();
        let worker_key = format!("{}::coro", name);
        let mut worker = def.clone();
        worker.is_async = false;
        worker.name = worker_key.clone();
        self.defs.insert(worker_key.clone(), worker);
        self.sla_names
            .insert(worker_key.clone(), format!("{}__coro", self.sla_name(&name)));
        self.ensure_fn(&worker_key, &vec![Ty::Dyn; n], def.line)?;
        let id = self.register_callable(&worker_key);
        self.sigs.insert(
            name.clone(),
            Sig {
                params: vec![Ty::Dyn; n],
                param_borrow: vec![false; n],
                ret: Some(Ty::Dyn),
            },
        );
        self.used_dyn = true;
        self.used_coro = true;
        let mut w = String::new();
        w.push_str(&format!("fn {}(", self.sla_name(&name)));
        for (i, p) in params.iter().enumerate() {
            if i > 0 {
                w.push_str(", ");
            }
            w.push_str(&format!("{}: PsaVal", p));
        }
        w.push_str(") -> PsaVal {\n");
        w.push_str("    let __psa_args = __psa_list_new();\n");
        for p in &params {
            w.push_str(&format!("    __psa_list_push(__psa_args, {});\n", p));
        }
        w.push_str(&format!("    return __psa_coro_new({}, __psa_args);\n}}\n", id));
        // P1: the async wrapper is one synthetic function chunk
        // anchored to the async def (precision Function).
        self.emitted.push(EmittedChunk::span(
            w,
            &SegMeta::synthetic_fn(def.anchor, self.sla_name(&name)),
        ));
        self.emitted_set.push(name.clone());
        Ok(())
    }

    /// D7: match statement. The subject is evaluated once to a
    /// dynamic temp; each case is tried in order under a `done`
    /// flag. Pattern checks write a per-case ok flag and store
    /// captures into prologue-hoisted per-case temps; on a
    /// pattern match the temps are bound to the real names
    /// BEFORE the guard runs (CPython leaves those bindings in
    /// place when a guard fails, and so does psa), the guard is
    /// evaluated against the real names, and the body runs as a
    /// plain block, so break/continue/return keep their
    /// enclosing meaning (no try-frame rewriting applies --
    /// match is not a control boundary in CPython either).
    fn emit_match(
        &mut self,
        ctx: &mut Ctx,
        subject: &Expr,
        cases: &[MatchCase],
        line: u32,
    ) -> Result<(), Diagnostic> {
        self.used_dyn = true;
        ctx.tmp += 1;
        let mid = ctx.tmp;
        let subj = format!("__psa_msubj_{}", mid);
        let scode = self.emit_dyn_value(ctx, subject, line)?;
        ctx.declare(&subj, Ty::Dyn).map_err(|m| eerr(line, &m))?;
        ctx.line(&format!("let {}: PsaVal = {};", subj, scode));
        let done = format!("__psa_mdone_{}", mid);
        ctx.declare(&done, Ty::Bool).map_err(|m| eerr(line, &m))?;
        ctx.line(&format!("let {}: bool = false;", done));
        for case in cases {
            let caps = pattern_captures(&case.pattern);
            let mut seen: Vec<&String> = Vec::new();
            for c in &caps {
                if seen.contains(&c) {
                    return Err(eerr(
                        case.line,
                        &format!(
                            "match pattern binds `{}` more than once (CPython refuses this too, D7)",
                            c
                        ),
                    ));
                }
                seen.push(c);
            }
            ctx.tmp += 1;
            let cid = ctx.tmp;
            let ok = format!("__psa_mok_{}", cid);
            let mut binds: Vec<(String, String)> = Vec::new();
            for cap in &caps {
                let temp = format!("__psa_mbind_{}_{}", cid, cap);
                ctx.hoist_declare(&temp, Ty::Dyn);
                binds.push((cap.clone(), temp));
            }
            ctx.line(&format!("if !({}) {{", done));
            ctx.indent += 1;
            ctx.declare(&ok, Ty::Bool).map_err(|m| eerr(case.line, &m))?;
            ctx.line(&format!("let {}: bool = true;", ok));
            self.match_check(ctx, &case.pattern, &subj, &ok, &binds, case.line)?;
            ctx.line(&format!("if {} {{", ok));
            ctx.indent += 1;
            for (name, temp) in &binds {
                let t = temp.clone();
                if ctx.lookup(name).is_none() {
                    // The bind runs inside the emitted
                    // `if ok` block; a fresh `let` there is
                    // invisible to the guard and the body
                    // under SLA block scoping (match1's
                    // UndefinedVariable `x`). Hoist the real
                    // name into the function prologue first
                    // so emit_bind_dyn_code takes its store
                    // path instead of declaring in-block.
                    ctx.hoist_declare(name, Ty::Dyn);
                }
                self.emit_bind_dyn_code(ctx, name, &t, case.line)?;
            }
            ctx.indent -= 1;
            ctx.line("};");
            if let Some(guard) = &case.guard {
                // A walrus in the guard binds through the
                // synthetic-Assign machinery, which declares a
                // fresh `let` at the emission point — inside
                // the `if ok` block below, where SLA block
                // scoping makes it invisible to the case body
                // (w09's UndefinedVariable, the same trap the
                // pattern captures above already avoid).
                // Hoist each unbound guard walrus target into
                // the function prologue as a dynamic variable
                // so the binding takes the store path and the
                // body can read it.
                {
                    let mut targets: Vec<String> = Vec::new();
                    collect_walrus_targets(guard, &mut targets);
                    for t in &targets {
                        if ctx.lookup(t).is_none() {
                            ctx.hoist_declare(t, Ty::Dyn);
                        }
                    }
                }
                ctx.line(&format!("if {} {{", ok));
                ctx.indent += 1;
                let (gc, gt) = self.emit_expr(ctx, guard, case.line)?;
                let gc = self.cond_code(gc, gt, case.line, "match guard must be bool")?;
                ctx.line(&format!("if !({}) {{ {} = false; }};", gc, ok));
                ctx.indent -= 1;
                ctx.line("};");
            }
            ctx.line(&format!("if {} {{", ok));
            ctx.indent += 1;
            ctx.line(&format!("{} = true;", done));
            self.emit_block(ctx, &case.body)?;
            ctx.indent -= 1;
            ctx.line("};");
            ctx.indent -= 1;
            ctx.line("};");
        }
        Ok(())
    }

    /// D7: emit the checks for one pattern against the dynamic
    /// subject code `subj`, clearing `ok` on any mismatch and
    /// storing captures into their temps from `binds`.
    fn match_check(
        &mut self,
        ctx: &mut Ctx,
        pat: &Pattern,
        subj: &str,
        ok: &str,
        binds: &[(String, String)],
        line: u32,
    ) -> Result<(), Diagnostic> {
        match pat {
            Pattern::Lit(e) => {
                let lc = self.emit_dyn_value(ctx, e, line)?;
                ctx.line(&format!(
                    "if !(__psa_dyn_eq({}, {})) {{ {} = false; }};",
                    subj, lc, ok
                ));
                Ok(())
            }
            Pattern::Capture(name) => {
                let temp = binds
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, t)| t.clone())
                    .ok_or_else(|| eerr(line, "internal: match capture temp missing"))?;
                self.emit_dyn_store(ctx, &temp, subj)
                    .map_err(|m| eerr(line, &m))?;
                Ok(())
            }
            Pattern::Wildcard => Ok(()),
            Pattern::Seq(pats) => {
                ctx.line(&format!(
                    "if !({}.tag == 4 || {}.tag == 10) {{ {} = false; }};",
                    subj, subj, ok
                ));
                ctx.line(&format!("if {} {{", ok));
                ctx.indent += 1;
                ctx.line(&format!(
                    "if __psa_dyn_len({}) != {} {{ {} = false; }};",
                    subj,
                    pats.len(),
                    ok
                ));
                for (i, p) in pats.iter().enumerate() {
                    ctx.tmp += 1;
                    let el = format!("__psa_mel_{}", ctx.tmp);
                    ctx.line(&format!("if {} {{", ok));
                    ctx.indent += 1;
                    ctx.declare(&el, Ty::Dyn).map_err(|m| eerr(line, &m))?;
                    ctx.line(&format!(
                        "let {}: PsaVal = __psa_iter_get({}, {});",
                        el, subj, i
                    ));
                    self.match_check(ctx, p, &el, ok, binds, line)?;
                    ctx.indent -= 1;
                    ctx.line("};");
                }
                ctx.indent -= 1;
                ctx.line("};");
                Ok(())
            }
            Pattern::Or(alts) => {
                let mut want = pattern_captures(&alts[0]);
                want.sort();
                want.dedup();
                for a in &alts[1..] {
                    let mut got = pattern_captures(a);
                    got.sort();
                    got.dedup();
                    if got != want {
                        return Err(eerr(
                            line,
                            "match alternatives must bind the same names (CPython rule, D7)",
                        ));
                    }
                }
                ctx.tmp += 1;
                let aid = ctx.tmp;
                let any = format!("__psa_mor_{}", aid);
                ctx.declare(&any, Ty::Bool).map_err(|m| eerr(line, &m))?;
                ctx.line(&format!("let {}: bool = false;", any));
                for a in alts {
                    ctx.tmp += 1;
                    let aok = format!("__psa_mok_{}", ctx.tmp);
                    let mut abinds: Vec<(String, String)> = Vec::new();
                    for cap in pattern_captures(a) {
                        let t = format!("__psa_mbind_{}_{}", ctx.tmp, cap);
                        ctx.hoist_declare(&t, Ty::Dyn);
                        abinds.push((cap, t));
                    }
                    ctx.line(&format!("if !({}) {{", any));
                    ctx.indent += 1;
                    ctx.declare(&aok, Ty::Bool).map_err(|m| eerr(line, &m))?;
                    ctx.line(&format!("let {}: bool = true;", aok));
                    self.match_check(ctx, a, subj, &aok, &abinds, line)?;
                    ctx.line(&format!("if {} {{", aok));
                    ctx.indent += 1;
                    for (name, atemp) in &abinds {
                        let target = binds
                            .iter()
                            .find(|(n, _)| n == name)
                            .map(|(_, t)| t.clone())
                            .ok_or_else(|| {
                                eerr(line, "internal: match alt capture temp missing")
                            })?;
                        self.emit_dyn_store(ctx, &target, atemp)
                            .map_err(|m| eerr(line, &m))?;
                    }
                    ctx.line(&format!("{} = true;", any));
                    ctx.indent -= 1;
                    ctx.line("};");
                    ctx.indent -= 1;
                    ctx.line("};");
                }
                ctx.line(&format!("if !({}) {{ {} = false; }};", any, ok));
                Ok(())
            }
        }
    }

    /// D4 class construction (also D6 module classes and the
    /// builtin exception classes): allocate the instance, then
    /// run the flattened __init__ (if any) directly on it.
    /// Exception-derived classes with no user-defined __init__
    /// take the __psa_exc_new path instead: zero or one
    /// message argument, stored as the instance's `msg` entry.
    fn emit_construct(
        &mut self,
        ctx: &mut Ctx,
        fname: &str,
        cid: usize,
        args: &[Expr],
        kwargs: &[(String, Expr)],
        line: u32,
    ) -> Result<(String, Ty), Diagnostic> {
        if !kwargs.is_empty() {
            return Err(eerr(
                line,
                "keyword arguments in class construction are not supported yet (D4 v1)",
            ));
        }
        self.used_dyn = true;
        let init_key = self.classes[cid]
            .flat_methods
            .iter()
            .find(|(n, _)| n.as_str() == "__init__")
            .map(|(_, k)| k.clone());
        if init_key.is_none() && self.classes[cid].ancestors.contains(&0) {
            if args.len() > 1 {
                return Err(eerr(
                    line,
                    &format!(
                        "exception class `{}` takes at most one argument in v1 (the message)",
                        fname
                    ),
                ));
            }
            let msg = if args.is_empty() {
                Ty::Dyn.default_lit().to_string()
            } else {
                self.emit_dyn_value(ctx, &args[0], line)?
            };
            ctx.tmp += 1;
            let inst = format!("__psa_obj_{}", ctx.tmp);
            ctx.line(&format!("let {} = __psa_exc_new({}, {});", inst, cid, msg));
            return Ok((inst, Ty::Dyn));
        }
        let codes = self.emit_dyn_args(ctx, args, line)?;
        ctx.tmp += 1;
        let inst = format!("__psa_obj_{}", ctx.tmp);
        ctx.line(&format!("let {} = __psa_obj_new({});", inst, cid));
        match init_key {
            Some(key) => {
                let def = self
                    .defs
                    .get(&key)
                    .expect("a method is always a def")
                    .clone();
                if def.vararg.is_some()
                    || def.kwarg.is_some()
                    || def.params.iter().any(|(_, d)| d.is_some())
                {
                    return Err(eerr(
                        line,
                        &format!(
                            "__init__ of class `{}` has defaults or *args/**kwargs parameters, which construction cannot bind yet (D4 v1)",
                            fname
                        ),
                    ));
                }
                if def.param_count() != codes.len() + 1 {
                    return Err(eerr(
                        line,
                        &format!(
                            "class `{}` constructor takes {} argument(s), called with {}",
                            fname,
                            def.param_count().saturating_sub(1),
                            codes.len()
                        ),
                    ));
                }
                self.method_callable_id(&key, line)?;
                let ret = self.sigs.get(&key).and_then(|s| s.ret);
                let mut call = format!("{}({}", self.sla_name(&key), inst);
                for c in &codes {
                    call.push_str(", ");
                    call.push_str(c);
                }
                call.push(')');
                match ret {
                    Some(Ty::Dyn) => {
                        ctx.tmp += 1;
                        let r = format!("__psa_init_{}", ctx.tmp);
                        ctx.line(&format!("let {} = {};", r, call));
                        ctx.line(&format!("if {}.tag != 3 {{ panic(77074); }};", r));
                    }
                    Some(Ty::Void) => {
                        ctx.line(&format!("{};", call));
                    }
                    other => {
                        return Err(eerr(
                            line,
                            &format!(
                                "__init__ of class `{}` must return None, but it returns {:?} (Python raises TypeError)",
                                fname, other
                            ),
                        ))
                    }
                }
            }
            None => {
                if !codes.is_empty() {
                    ctx.line(&format!("__psa_obj_check_noargs({});", codes.len()));
                }
            }
        }
        Ok((inst, Ty::Dyn))
    }

    /// D5 conversion / constructor / reflection builtins in
    /// call position: int, float, str, bool, list, tuple, set,
    /// bytes, type, isinstance. Returns None when `fname` is
    /// none of them; the caller has already established that
    /// no user def or variable shadows the name (a shadowed
    /// builtin is an ordinary call, as in Python).
    fn emit_builtin_call(
        &mut self,
        ctx: &mut Ctx,
        fname: &str,
        args: &[Expr],
        kwargs: &[(String, Expr)],
        line: u32,
    ) -> Result<Option<(String, Ty)>, Diagnostic> {
        if !matches!(
            fname,
            "int" | "float" | "str" | "bool" | "list" | "tuple" | "set" | "bytes" | "type" | "isinstance"
                | "open"
                | "next"
                | "repr"
        ) {
            return Ok(None);
        }
        if !kwargs.is_empty() {
            return Err(eerr(
                line,
                &format!("{}() takes no keyword arguments in v1", fname),
            ));
        }
        match fname {
            "repr" => {
                // repr() of a dynamic value (D5/D6 surface):
                // strings quote, objects ride __repr__, scalars
                // format exactly; containers and floats stay
                // blocked (77030) with the missing repr-to-string
                // builders, exactly like str().
                if args.len() != 1 {
                    return Err(eerr(line, "repr() takes exactly one argument"));
                }
                let (c, t) = self.emit_expr(ctx, &args[0], line)?;
                if t != Ty::Dyn {
                    return Err(eerr(
                        line,
                        "repr() requires a dynamic value in v1 (bind the value to a dynamic name first)",
                    ));
                }
                self.used_dyn = true;
                Ok(Some((format!("__psa_repr_of({})", c), Ty::Dyn)))
            }
            "next" => {
                // D7: next(gen) pulls one item; exhaustion is
                // panic 77081 (psa has no exception values a
                // caller could catch a StopIteration with in the
                // dynamic face -- try/except StopIteration is
                // refused loudly instead, see D7-NOTES). A
                // two-argument next(x, default) is not in v1.
                if args.len() != 1 {
                    return Err(eerr(
                        line,
                        "next() takes exactly one argument in psa v1 (D7); the two-argument default form is not supported",
                    ));
                }
                let (c, t) = self.emit_expr(ctx, &args[0], line)?;
                if t != Ty::Dyn {
                    return Err(eerr(
                        line,
                        "next() requires a dynamic value (a generator) in psa v1 (D7)",
                    ));
                }
                self.used_dyn = true;
                self.used_gen = true;
                Ok(Some((format!("__psa_gen_next({})", c), Ty::Dyn)))
            }
            "open" => {
                // D6: open(path[, mode]) -> a tag-12 file object.
                // Text modes r/w/a only (a trailing t is a no-op);
                // the mode must be a string literal in v1 so the
                // runtime receives a plain mode code.
                if args.is_empty() || args.len() > 2 {
                    return Err(eerr(line, "open() takes a path and an optional mode in v1"));
                }
                let pcode = match &args[0] {
                    Expr::Str(s) => self.emit_str_temp(ctx, s, line)?,
                    _ => {
                        let (c, t) = self.emit_expr(ctx, &args[0], line)?;
                        if t != Ty::Dyn {
                            return Err(eerr(line, "open() path must be a string"));
                        }
                        c
                    }
                };
                let mode: i64 = if args.len() == 2 {
                    match &args[1] {
                        Expr::Str(s) => match s.as_str() {
                            "r" | "rt" => 0,
                            "w" | "wt" => 1,
                            "a" | "at" => 2,
                            other => {
                                return Err(eerr(
                                    line,
                                    &format!(
                                        "open() mode `{}` is not supported in v1 (text modes r, w and a only; binary and + modes wait)",
                                        other
                                    ),
                                ))
                            }
                        },
                        _ => {
                            return Err(eerr(
                                line,
                                "open() mode must be a string literal in v1",
                            ))
                        }
                    }
                } else {
                    0
                };
                self.used_dyn = true;
                self.used_file = true;
                Ok(Some((format!("__psa_open({}, {})", pcode, mode), Ty::Dyn)))
            }
            "int" => {
                if args.len() > 1 {
                    return Err(eerr(line, "int() with a base argument is not supported in v1"));
                }
                if args.is_empty() {
                    return Ok(Some(("0".to_string(), Ty::Int)));
                }
                if let Expr::Str(s) = &args[0] {
                    return match s.trim().parse::<i64>() {
                        Ok(v) => Ok(Some((v.to_string(), Ty::Int))),
                        Err(_) => {
                            // A valid integer literal past i64
                            // promotes to a dynamic bigint instead of
                            // failing (CPython parses it fine);
                            // anything else keeps the loud refusal.
                            let t = s.trim().replace('_', "");
                            let digits = t.strip_prefix(['+', '-']).unwrap_or(&t);
                            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
                                self.used_dyn = true;
                                let tmp = self.emit_str_temp(ctx, &t, line)?;
                                return Ok(Some((format!("__psa_big_from_psastr({})", tmp), Ty::Dyn)));
                            }
                            Err(eerr(
                                line,
                                &format!(
                                    "int() cannot parse the literal {:?} (CPython raises ValueError)",
                                    s
                                ),
                            ))
                        }
                    };
                }
                let (c, t) = self.emit_expr(ctx, &args[0], line)?;
                match t {
                    Ty::Int => Ok(Some((c, Ty::Int))),
                    Ty::Bool => {
                        ctx.tmp += 1;
                        let tmp = format!("__psa_cv_{}", ctx.tmp);
                        ctx.line(&format!("let {}: i64 = 0;", tmp));
                        ctx.line(&format!("if {} {{ {} = 1; }};", c, tmp));
                        Ok(Some((tmp, Ty::Int)))
                    }
                    Ty::Float => Ok(Some((format!("({} as i64)", c), Ty::Int))),
                    Ty::Dyn => {
                        // int() of a dynamic value stays dynamic: a
                        // tag-17 bigint passes through untouched and
                        // an overflowing numeric string promotes
                        // instead of wrapping (__psa_dyn_int lives in
                        // BIG_RUNTIME).
                        self.used_dyn = true;
                        Ok(Some((format!("__psa_dyn_int({})", c), Ty::Dyn)))
                    }
                    _ => Err(eerr(
                        line,
                        &format!("int() cannot convert a {} value in v1", t.py_name()),
                    )),
                }
            }
            "float" => {
                if args.len() > 1 {
                    return Err(eerr(line, "float() takes at most one argument"));
                }
                if args.is_empty() {
                    return Ok(Some((fmt_f64(0.0), Ty::Float)));
                }
                if let Expr::Str(s) = &args[0] {
                    return match s.trim().parse::<f64>() {
                        Ok(v) => Ok(Some((fmt_f64(v), Ty::Float))),
                        Err(_) => Err(eerr(
                            line,
                            &format!(
                                "float() cannot parse the literal {:?} (CPython raises ValueError)",
                                s
                            ),
                        )),
                    };
                }
                let (c, t) = self.emit_expr(ctx, &args[0], line)?;
                match t {
                    Ty::Float => Ok(Some((c, Ty::Float))),
                    Ty::Int => Ok(Some((format!("({} as f64)", c), Ty::Float))),
                    Ty::Bool => {
                        ctx.tmp += 1;
                        let tmp = format!("__psa_cv_{}", ctx.tmp);
                        ctx.line(&format!("let {}: f64 = 0.0;", tmp));
                        ctx.line(&format!("if {} {{ {} = 1.0; }};", c, tmp));
                        Ok(Some((tmp, Ty::Float)))
                    }
                    Ty::Dyn => {
                        self.used_dyn = true;
                        Ok(Some((format!("__psa_to_float({})", c), Ty::Float)))
                    }
                    _ => Err(eerr(
                        line,
                        &format!("float() cannot convert a {} value in v1", t.py_name()),
                    )),
                }
            }
            "str" => {
                if args.len() > 1 {
                    return Err(eerr(
                        line,
                        "str() with encoding arguments is not supported in v1",
                    ));
                }
                if args.is_empty() {
                    let t = self.emit_str_temp(ctx, "", line)?;
                    return Ok(Some((t, Ty::Dyn)));
                }
                if let Expr::Str(s) = &args[0] {
                    let t = self.emit_str_temp(ctx, s, line)?;
                    return Ok(Some((t, Ty::Dyn)));
                }
                let (c, t) = self.emit_expr(ctx, &args[0], line)?;
                match t {
                    Ty::Dyn => {
                        self.used_dyn = true;
                        Ok(Some((format!("__psa_str_of({})", c), Ty::Dyn)))
                    }
                    Ty::Int => {
                        self.used_dyn = true;
                        Ok(Some((format!("__psa_str_from_i64({})", c), Ty::Dyn)))
                    }
                    Ty::Bool => {
                        self.used_dyn = true;
                        Ok(Some((format!("__psa_str_from_bool({})", c), Ty::Dyn)))
                    }
                    Ty::Float => Err(eerr(
                        line,
                        "str() of a float is blocked: f64 text is unavailable until the shortest-roundtrip formatter lands (upstream defect (xxiii) family)",
                    )),
                    _ => Err(eerr(
                        line,
                        &format!("str() of a {} value is not supported in v1", t.py_name()),
                    )),
                }
            }
            "bytes" => {
                // The bytes() constructor, minimal forms:
                // bytes() is the empty bytes value, and
                // bytes(x) of a dynamic value hands a bytes
                // value back unchanged (CPython returns the
                // same object for an exact bytes argument,
                // and psa bytes are immutable, so the alias
                // is exact). Every other form -- an int
                // count, an iterable of ints, a str with an
                // encoding -- is a registered gap (bytes
                // constructor forms) and refuses loudly
                // rather than guessing a conversion.
                if args.len() > 1 {
                    return Err(eerr(
                        line,
                        "bytes() with an encoding argument is not supported in v1 (registered gap: bytes constructor forms)",
                    ));
                }
                if args.is_empty() {
                    let t = self.emit_bytes_temp(ctx, &[], line)?;
                    return Ok(Some((t, Ty::Dyn)));
                }
                if let Expr::Bytes(bs) = &args[0] {
                    let t = self.emit_bytes_temp(ctx, bs, line)?;
                    return Ok(Some((t, Ty::Dyn)));
                }
                let (c, t) = self.emit_expr(ctx, &args[0], line)?;
                match t {
                    Ty::Dyn => {
                        self.used_dyn = true;
                        Ok(Some((format!("__psa_bytes_convert({})", c), Ty::Dyn)))
                    }
                    _ => Err(eerr(
                        line,
                        &format!(
                            "bytes() of a {} value is not supported in v1 (registered gap: bytes constructor forms)",
                            t.py_name()
                        ),
                    )),
                }
            }
            "bool" => {
                if args.len() > 1 {
                    return Err(eerr(line, "bool() takes at most one argument"));
                }
                if args.is_empty() {
                    return Ok(Some(("false".to_string(), Ty::Bool)));
                }
                if let Expr::Str(s) = &args[0] {
                    return Ok(Some((
                        if s.is_empty() { "false" } else { "true" }.to_string(),
                        Ty::Bool,
                    )));
                }
                let (c, t) = self.emit_expr(ctx, &args[0], line)?;
                match t {
                    Ty::Bool => Ok(Some((c, Ty::Bool))),
                    Ty::Int => Ok(Some((format!("({} != 0)", c), Ty::Bool))),
                    Ty::Float => Ok(Some((format!("({} != 0.0)", c), Ty::Bool))),
                    Ty::Dyn => {
                        self.used_dyn = true;
                        Ok(Some((format!("__psa_dyn_truthy({})", c), Ty::Bool)))
                    }
                    _ if t.is_list() => {
                        Ok(Some((format!("((len({}) as i64) != 0)", c), Ty::Bool)))
                    }
                    _ => Err(eerr(
                        line,
                        &format!("bool() of a {} value is not supported in v1", t.py_name()),
                    )),
                }
            }
            "list" | "tuple" | "set" => {
                if args.len() > 1 {
                    return Err(eerr(
                        line,
                        &format!("{}() takes at most one argument", fname),
                    ));
                }
                self.used_dyn = true;
                if args.is_empty() {
                    let t = match fname {
                        "list" => self.emit_container_temp(ctx, &Expr::ListLit(vec![]), line)?,
                        "tuple" => self.emit_tuple_temp(ctx, &[], line)?,
                        _ => self.emit_set_temp(ctx, &[], line)?,
                    };
                    return Ok(Some((t, Ty::Dyn)));
                }
                // range(...) materialises through a synthetic
                // list comprehension, then converts.
                if let Expr::Call(callee, _, _, _) = &args[0] {
                    if matches!(&**callee, Expr::Name(n, _) if n == "range") {
                        let comp = Expr::Comp {
                            kind: CompKind::List,
                            key: Box::new(Expr::Name("__psa_rx".to_string(), ExprSpan::synthetic())),
                            value: None,
                            clauses: vec![CompClause::For {
                                var: "__psa_rx".to_string(),
                                iter: args[0].clone(),
                            }],
                        };
                        let lt = self.emit_comp(ctx, &comp, line)?;
                        return Ok(Some((
                            match fname {
                                "list" => lt,
                                "tuple" => format!("__psa_tuple_from({})", lt),
                                _ => format!("__psa_set_from({})", lt),
                            },
                            Ty::Dyn,
                        )));
                    }
                }
                let c = self.emit_dyn_value(ctx, &args[0], line)?;
                // D7: a generator argument is drained first
                // (list(gen) / tuple(gen) / set(gen)).
                let c = if self.used_gen {
                    format!("__psa_iter_source({})", c)
                } else {
                    c
                };
                let f = match fname {
                    "list" => "__psa_list_from",
                    "tuple" => "__psa_tuple_from",
                    _ => "__psa_set_from",
                };
                Ok(Some((format!("{}({})", f, c), Ty::Dyn)))
            }
            "type" => {
                if args.len() != 1 {
                    return Err(eerr(
                        line,
                        "type() takes exactly one argument in v1 (the three-argument class-creation form is not supported)",
                    ));
                }
                self.used_dyn = true;
                let c = self.emit_dyn_value(ctx, &args[0], line)?;
                Ok(Some((format!("__psa_type_name({})", c), Ty::Dyn)))
            }
            "isinstance" => {
                if args.len() != 2 {
                    return Err(eerr(line, "isinstance() takes exactly two arguments"));
                }
                self.used_dyn = true;
                let c = self.emit_dyn_value(ctx, &args[0], line)?;
                let code = self.emit_isinstance_check(&c, &args[1], line)?;
                Ok(Some((code, Ty::Bool)))
            }
            _ => unreachable!(),
        }
    }

    /// One isinstance() target: a class name (the class plus
    /// every subclass, via the registry's ancestor lists), a
    /// builtin type name, or a tuple of targets.
    fn emit_isinstance_check(
        &self,
        code: &str,
        t: &Expr,
        line: u32,
    ) -> Result<String, Diagnostic> {
        match t {
            Expr::TupleLit(elems) => {
                if elems.is_empty() {
                    return Ok("false".to_string());
                }
                let mut parts = Vec::new();
                for e in elems {
                    parts.push(self.emit_isinstance_check(code, e, line)?);
                }
                Ok(parts.join(" || "))
            }
            Expr::Name(n, _) => {
                if let Some(&cid) = self.class_ids.get(n) {
                    let target = cid as i64;
                    let mut parts = Vec::new();
                    for ci in &self.classes {
                        if ci.ancestors.contains(&target) {
                            parts.push(format!("__psa_isinst_class({}, {})", code, ci.id));
                        }
                    }
                    return Ok(parts.join(" || "));
                }
                let bcode = match n.as_str() {
                    "int" => Some(0),
                    "float" => Some(1),
                    "bool" => Some(2),
                    "str" => Some(3),
                    "list" => Some(4),
                    "dict" => Some(5),
                    "tuple" => Some(6),
                    "set" => Some(7),
                    "object" => Some(8),
                    "NoneType" => Some(9),
                    "bytes" => Some(10),
                    _ => None,
                };
                match bcode {
                    Some(bc) => Ok(format!("__psa_is_type({}, {})", code, bc)),
                    None => Err(eerr(
                        line,
                        &format!(
                            "isinstance() target `{}` is not a class or a builtin type name in v1",
                            n
                        ),
                    )),
                }
            }
            _ => Err(eerr(
                line,
                "isinstance() target must be a class, a builtin type name, or a tuple of them in v1",
            )),
        }
    }

    /// D5 builtin method dispatch: for a dynamic base, a method
    /// whose name is in the builtin table goes to the runtime fn
    /// `__psa_m_<name>(base, name_temp, packed_args)` (the
    /// runtime defers tag 8 to getattr, so an object's own
    /// method still wins); any other name returns None here and
    /// the caller falls back to the getattr path.
    fn emit_builtin_method(
        &mut self,
        ctx: &mut Ctx,
        base_code: &str,
        method: &str,
        args: &[Expr],
        line: u32,
    ) -> Result<Option<String>, Diagnostic> {
        const BUILTIN_METHODS: &[&str] = &[
            "append", "extend", "insert", "pop", "remove", "clear", "copy", "index", "count",
            "reverse", "sort", "get", "keys", "values", "items", "setdefault", "update", "upper",
            "lower", "strip", "lstrip", "rstrip", "split", "join", "replace", "startswith",
            "endswith", "find", "isdigit", "isalpha", "add", "discard", "union", "intersection",
            "difference", "read", "readline", "write", "close",
        ];
        if !BUILTIN_METHODS.contains(&method) {
            return Ok(None);
        }
        self.used_dyn = true;
        if matches!(method, "read" | "readline" | "write" | "close") {
            self.used_file = true;
        }
        let ntmp = self.emit_str_temp(ctx, method, line)?;
        let codes = self.emit_dyn_args(ctx, args, line)?;
        let tmp = self.pack_dyn_args(ctx, &codes);
        Ok(Some(format!("__psa_m_{}({}, {}, {})", method, base_code, ntmp, tmp)))
    }

    /// Pack emitted dynamic argument codes into a fresh dynamic
    /// list temp (the argument vector of the callable dispatch).
    fn pack_dyn_args(&mut self, ctx: &mut Ctx, codes: &[String]) -> String {
        ctx.tmp += 1;
        let tmp = format!("__psa_ctmp_{}", ctx.tmp);
        ctx.line(&format!("let {} = __psa_list_new();", tmp));
        for c in codes {
            ctx.line(&format!("__psa_list_push({}, {});", tmp, c));
        }
        tmp
    }

    fn ensure_fn(&mut self, name: &str, arg_tys: &[Ty], line: u32) -> Result<Sig, Diagnostic> {
        if let Some(sig) = self.sigs.get(name) {
            let compatible = sig.params.len() == arg_tys.len()
                && sig
                    .params
                    .iter()
                    .zip(arg_tys.iter())
                    .all(|(p, a)| sig_param_compatible(*p, *a));
            if !compatible {
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
        if def.param_count() != arg_tys.len() {
            return Err(eerr(
                line,
                &format!(
                    "function `{}` takes {} arguments, called with {}",
                    name,
                    def.param_count(),
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
            .param_names()
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
        let param_vars: Vec<(String, Ty, Option<SourceAnchor>)> = def
            .param_names()
            .into_iter()
            .zip(arg_tys.iter().cloned())
            .map(|(n, t)| (n, t, None))
            .collect();
        let borrowed_params: Vec<String> = def
            .param_names()
            .iter()
            .zip(borrows.iter())
            .filter(|(_, b)| **b)
            .map(|(p, _)| p.clone())
            .collect();
        // Prescan the body into a scratch context to learn the unified
        // return type before emitting for real: when the returns unify
        // to Dyn, every return value must be boxed, including returns
        // that textually precede the branch that forced the upgrade.
        // A prescan error is ignored here; the real emission below
        // reproduces it identically and reports it.
        let prescan_ret = {
            let mut sctx = Ctx {
                vars: param_vars.clone(),
                sources: self.sources.clone(),
                scopes: vec![0],
                ret: None,
                fname: name.to_string(),
                out: SegmentWriter::new(),
                prologue: SegmentWriter::new(),
                indent: 1,
                tmp: 0,
                borrowed: borrowed_params.clone(),
                in_while_cond: false,
                dyn_ret: false,
                deleted: Vec::new(),
                try_stack: Vec::new(),
                guard_depth: 0,
                loop_depth: 0,
                exc_slot: false,
                in_finally: 0,
                exc_bare: None,
                gen_chan: None,
                anchor: SourceAnchor::synthetic(),
                anchor_kind: SegKind::Stmt,
                anchor_precision: Precision::None,
                anchor_name: None,
                anchor_origin: Origin::Synthetic,
                pending_call: false,
                pending_call_name: None,
            };
            let r = (|| {
                for s in &def.body {
                    self.emit_stmt(&mut sctx, s)?;
                }
                Ok::<(), Diagnostic>(())
            })();
            match r {
                Ok(()) => sctx.ret,
                Err(_) => None,
            }
        };
        if let Some(pr) = prescan_ret {
            self.sigs.get_mut(name).unwrap().ret = Some(pr);
        }
        let mut ctx = Ctx {
            vars: param_vars,
            sources: self.sources.clone(),
            scopes: vec![0],
            ret: None,
            fname: name.to_string(),
            out: SegmentWriter::new(),
            prologue: SegmentWriter::new(),
            indent: 1,
            tmp: 0,
            borrowed: borrowed_params,
            in_while_cond: false,
            dyn_ret: prescan_ret == Some(Ty::Dyn),
            deleted: Vec::new(),
            try_stack: Vec::new(),
            guard_depth: 0,
            loop_depth: 0,
            exc_slot: false,
            in_finally: 0,
            exc_bare: None,
            gen_chan: None,
            anchor: SourceAnchor::synthetic(),
            anchor_kind: SegKind::Stmt,
            anchor_precision: Precision::None,
            anchor_name: None,
            anchor_origin: Origin::Synthetic,
            pending_call: false,
            pending_call_name: None,
        };
        let body_res = (|| {
            for s in &def.body {
                self.emit_stmt(&mut ctx, s)?;
            }
            Ok::<(), Diagnostic>(())
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
        // P1 source map: the function's text is a segment writer.
        // The header is built as a plain string first (byte-identical
        // to the old construction), then recorded as the function's
        // `FnHeader` segment; prologue and body follow as recorded
        // by `Ctx::line` / `prologue_push`, rebased behind the
        // header; the psa-added default return is a `Synthetic`
        // statement anchored to the def; the closer re-states the
        // header's anchor. The `last_is_return` check reads the
        // body writer's text, as before (computed before the takes).
        // Default fallthrough return, appended only when the body's
        // last top-level statement is not already a return. A second,
        // dead return after a live one is accepted by the SLA checker
        // but the SA-text flattener rejects its expansion: the dead
        // block materialises the return temp and then falls through
        // with no terminator (error[FallthroughForbidden], fib/list/
        // dyn1 on the text route, 2026-11-25).
        let last_is_return = ctx
            .out
            .text
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .map(|l| l.starts_with("    return "))
            .unwrap_or(false);
        let mut head = format!("fn {}(", self.sla_name(name));
        for (i, (p, t)) in def.param_names().iter().zip(arg_tys.iter()).enumerate() {
            if i > 0 {
                head.push_str(", ");
            }
            if t.is_list() && borrows.get(i).copied().unwrap_or(false) {
                head.push_str(&format!("{}: &{}", p, t.sla()));
            } else {
                head.push_str(&format!("{}: {}", p, t.sla()));
            }
        }
        head.push_str(&format!(") -> {} {{", ret_ty.sla()));
        let def_anchor = def.anchor;
        let mut txt = SegmentWriter::new();
        txt.push_line(&head, &SegMeta::fn_header(def_anchor, self.sla_name(name)));
        txt.append_writer(std::mem::take(&mut ctx.prologue));
        txt.append_writer(std::mem::take(&mut ctx.out));
        if !last_is_return {
            txt.push_line(
                &format!("    return {};", ret_ty.default_lit()),
                &SegMeta::synthetic_stmt(def_anchor),
            );
        }
        txt.push_line("}", &SegMeta::fn_header(def_anchor, self.sla_name(name)));
        if !self.emitted_set.contains(&name.to_string()) {
            self.emitted_set.push(name.to_string());
            let (text, segments, panics) = txt.into_parts();
            self.emitted.push(EmittedChunk {
                text,
                map: SourceMap::from_segments(segments, panics),
            });
        }
        Ok(self.sigs.get(name).unwrap().clone())
    }

    /// Wrap a statically-typed value's code as a dynamic value
    /// (boxing at a static/dynamic boundary). Dynamic values pass
    /// through unchanged; kinds that cannot be dynamic in D1 fail
    /// loudly with the phase that will carry them.
    fn box_code(&mut self, code: String, ty: Ty, line: u32) -> Result<String, Diagnostic> {
        match ty {
            Ty::Dyn => Ok(code),
            Ty::Int => {
                self.used_dyn = true;
                Ok(format!("__psa_box_int({})", code))
            }
            Ty::Float => {
                self.used_dyn = true;
                Ok(format!("__psa_box_float({})", code))
            }
            Ty::Bool => {
                self.used_dyn = true;
                Ok(format!("__psa_box_bool({})", code))
            }
            Ty::Str => Err(eerr(
                line,
                "a string expression cannot become a dynamic value here; only plain string literals become dynamic strings (D2 slice C)",
            )),
            t if t.is_list() => Err(eerr(
                line,
                "lists cannot become dynamic values in D1 (dynamic containers arrive in D2)",
            )),
            _ => Err(eerr(line, "a void value cannot become a dynamic value")),
        }
    }

    /// Construct a dynamic string (tag 6) from a compile-time-known
    /// literal, byte by byte, in a fresh temp. Dynamic strings are
    /// how psa binds Python str values: there is no static Str
    /// variable on the text route.
    fn emit_str_temp(&mut self, ctx: &mut Ctx, s: &str, line: u32) -> Result<String, Diagnostic> {
        self.used_dyn = true;
        ctx.tmp += 1;
        let tmp = format!("__psa_str_{}", ctx.tmp);
        ctx.declare(&tmp, Ty::Dyn).map_err(|m| eerr(line, &m))?;
        ctx.line(&format!("let {} = __psa_str_new();", tmp));
        for b in s.bytes() {
            ctx.line(&format!("{} = __psa_str_push({}, {});", tmp, tmp, b));
        }
        Ok(tmp)
    }

    /// A bytes literal as a dynamic value in a temp (tag 15):
    /// the same Box<Vec<u8>> payload as a dynamic string (tag
    /// 6), built byte-by-byte through the shared push helper,
    /// which rebuilds the payload and preserves the value's
    /// tag.
    fn emit_bytes_temp(&mut self, ctx: &mut Ctx, bs: &[u8], line: u32) -> Result<String, Diagnostic> {
        self.used_dyn = true;
        ctx.tmp += 1;
        let tmp = format!("__psa_bytes_{}", ctx.tmp);
        ctx.declare(&tmp, Ty::Dyn).map_err(|m| eerr(line, &m))?;
        ctx.line(&format!("let {} = __psa_bytes_new();", tmp));
        for b in bs {
            ctx.line(&format!("{} = __psa_str_push({}, {});", tmp, tmp, b));
        }
        Ok(tmp)
    }

    /// One slice bound as a dynamic value: omitted is None (tag 3),
    /// ints and bools box, dynamic values pass through.
    fn slice_bound_code(
        &mut self,
        ctx: &mut Ctx,
        b: &Option<Box<Expr>>,
        line: u32,
    ) -> Result<String, Diagnostic> {
        match b {
            None => Ok(Ty::Dyn.default_lit().to_string()),
            Some(e) => {
                let (c, t) = self.emit_expr(ctx, e, line)?;
                match t {
                    Ty::Dyn => Ok(c),
                    Ty::Int | Ty::Bool => self.box_code(c, t, line),
                    _ => Err(eerr(line, "slice indices must be integers or omitted")),
                }
            }
        }
    }

    /// Construct a container literal as a dynamic value in a temp,
    /// for nesting inside another container, a subscript store or
    /// an append.
    fn emit_container_temp(&mut self, ctx: &mut Ctx, e: &Expr, line: u32) -> Result<String, Diagnostic> {
        match e {
            Expr::ListLit(elems) => {
                let codes = self.emit_list_literal_dyn(ctx, elems, line)?;
                ctx.tmp += 1;
                let tmp = format!("__psa_ctmp_{}", ctx.tmp);
                self.emit_dyn_list_construct(ctx, &tmp, &codes, true);
                Ok(tmp)
            }
            Expr::DictLit(pairs) => {
                let codes = self.emit_dict_literal(ctx, pairs, line)?;
                ctx.tmp += 1;
                let tmp = format!("__psa_ctmp_{}", ctx.tmp);
                self.emit_dyn_dict_construct(ctx, &tmp, &codes, true);
                Ok(tmp)
            }
            Expr::TupleLit(elems) => self.emit_tuple_temp(ctx, elems, line),
            Expr::SetLit(elems) => self.emit_set_temp(ctx, elems, line),
            Expr::Comp { .. } => self.emit_comp(ctx, e, line),
            _ => Err(eerr(line, "internal error: not a container literal")),
        }
    }

    /// One expression evaluated as a dynamic value (D5 helper
    /// shared by argument packing, tuple/set literals and
    /// comprehensions): literals and containers become temps,
    /// scalars box at the boundary.
    fn emit_dyn_value(
        &mut self,
        ctx: &mut Ctx,
        e: &Expr,
        line: u32,
    ) -> Result<String, Diagnostic> {
        let (c, t) = if let Expr::Str(s) = e {
            (self.emit_str_temp(ctx, s, line)?, Ty::Dyn)
        } else if matches!(
            e,
            Expr::ListLit(_)
                | Expr::DictLit(_)
                | Expr::TupleLit(_)
                | Expr::SetLit(_)
                | Expr::Comp { .. }
        ) {
            (self.emit_container_temp(ctx, e, line)?, Ty::Dyn)
        } else {
            self.emit_expr(ctx, e, line)?
        };
        if t == Ty::Dyn {
            Ok(c)
        } else {
            self.box_code(c, t, line)
        }
    }

    /// A tuple literal materialised in a temp (D5). Tuples are
    /// immutable, so the temp shape is safe in any position.
    fn emit_tuple_temp(
        &mut self,
        ctx: &mut Ctx,
        elems: &[Expr],
        line: u32,
    ) -> Result<String, Diagnostic> {
        self.used_dyn = true;
        let mut codes = Vec::new();
        for e in elems {
            codes.push(self.emit_dyn_value(ctx, e, line)?);
        }
        ctx.tmp += 1;
        let tmp = format!("__psa_ctmp_{}", ctx.tmp);
        ctx.line(&format!("let {} = __psa_tuple_new();", tmp));
        for c in &codes {
            ctx.line(&format!("__psa_tuple_push({}, {});", tmp, c));
        }
        Ok(tmp)
    }

    /// A set literal materialised in a temp (D5).
    fn emit_set_temp(
        &mut self,
        ctx: &mut Ctx,
        elems: &[Expr],
        line: u32,
    ) -> Result<String, Diagnostic> {
        self.used_dyn = true;
        let mut codes = Vec::new();
        for e in elems {
            codes.push(self.emit_dyn_value(ctx, e, line)?);
        }
        ctx.tmp += 1;
        let tmp = format!("__psa_ctmp_{}", ctx.tmp);
        ctx.line(&format!("let {} = __psa_set_new();", tmp));
        for c in &codes {
            ctx.line(&format!("__psa_set_add({}, {});", tmp, c));
        }
        Ok(tmp)
    }

    /// D7: a generator expression becomes a synthetic generator
    /// def whose parameters are the clause iterables (evaluated
    /// eagerly here, as CPython evaluates the outermost
    /// iterable eagerly) and whose body is the clause nest with
    /// the element expression yielded at the centre. Element
    /// and condition expressions may only reference the clause
    /// variables, defs, classes and builtins — plus, since
    /// D8, variables of the enclosing scope: the closure
    /// conversion pass has already rewritten those references
    /// (a cell read, or an environment slot read), and the
    /// cell / environment variable itself is passed into the
    /// synthetic def as an extra parameter under its own
    /// name, so the rewritten body resolves unchanged.
    /// Anything else stays a loud refusal. Iterable
    /// expressions themselves are evaluated in the enclosing
    /// scope, so they are not subject to the check.
    fn emit_genexp(
        &mut self,
        ctx: &mut Ctx,
        key: &Expr,
        value: &Option<Box<Expr>>,
        clauses: &[CompClause],
        line: u32,
    ) -> Result<String, Diagnostic> {
        if value.is_some() {
            return Err(eerr(
                line,
                "a parenthesised comprehension yields single values; there is no dict generator expression (D7)",
            ));
        }
        // A walrus in the element or an `if` condition would
        // be cloned into the synthetic generator def built
        // below and bind THERE; CPython binds it in the
        // containing scope, at consumption time. Refuse
        // loudly rather than bind in the wrong scope (a list
        // comprehension binds eagerly in scope; an explicit
        // loop always works). A walrus rebinding one of the
        // generator's own iteration variables is a CPython
        // SyntaxError wherever it appears, iterables included.
        if expr_contains_walrus(key)
            || expr_contains_cellset(key)
            || clauses.iter().any(|c| {
                matches!(c, CompClause::If(cond) if expr_contains_walrus(cond) || expr_contains_cellset(cond))
            })
        {
            return Err(eerr(
                line,
                "an assignment expression inside a generator expression's element or condition is not supported in v1 (it would bind inside the generator, not in the enclosing scope); use a list comprehension or an explicit loop",
            ));
        }
        {
            let mut targets: Vec<String> = Vec::new();
            for c in clauses {
                if let CompClause::For { iter, .. } = c {
                    collect_walrus_targets(iter, &mut targets);
                }
            }
            for t in &targets {
                if clauses
                    .iter()
                    .any(|c| matches!(c, CompClause::For { var, .. } if var == t))
                {
                    return Err(eerr(
                        line,
                        &format!(
                            "assignment expression cannot rebind comprehension iteration variable `{}` (CPython raises SyntaxError); rename the walrus target",
                            t
                        ),
                    ));
                }
            }
        }
        let mut fors: Vec<(&str, &Expr)> = Vec::new();
        for c in clauses {
            if let CompClause::For { var, iter } = c {
                fors.push((var.as_str(), iter));
            }
        }
        if fors.is_empty() {
            return Err(eerr(line, "a generator expression needs at least one for clause (D7)"));
        }
        let clause_vars: std::collections::HashSet<String> =
            fors.iter().map(|(v, _)| v.to_string()).collect();
        let mut names: Vec<String> = Vec::new();
        collect_names(key, &mut names);
        for c in clauses {
            if let CompClause::If(cond) = c {
                collect_names(cond, &mut names);
            }
        }
        // D8: a name that resolves to a variable of the
        // enclosing scope at this point (a cell variable or
        // the environment parameter, after closure
        // conversion) is passed into the synthetic def as an
        // extra parameter under its own name.
        let mut extra: Vec<String> = Vec::new();
        for n in &names {
            if clause_vars.contains(n)
                || self.defs.contains_key(n)
                || self.class_ids.contains_key(n)
                || LAMBDA_BUILTINS.contains(&n.as_str())
            {
                continue;
            }
            if ctx.lookup(n).is_some() {
                if !extra.contains(n) {
                    extra.push(n.clone());
                }
                continue;
            }
            return Err(eerr(
                line,
                &format!(
                    "generator expression references enclosing name `{}`, which would need a closure capture; psa does not support captures in v1 -- bind it through a def parameter or use a list comprehension (D7)",
                    n
                ),
            ));
        }
        self.genexp_counter += 1;
        let gname = format!("__psa_genexp_{}", self.genexp_counter);
        let total_fors = fors.len();
        let mut body: Vec<Stmt> = vec![Stmt::Yield {
            value: Some(key.clone()),
            line, anchor: SourceAnchor::at(line, 0, None),
        }];
        let mut fors_from_end = 0usize;
        for c in clauses.iter().rev() {
            match c {
                CompClause::If(cond) => {
                    body = vec![Stmt::If {
                        arms: vec![(cond.clone(), body)],
                        els: None,
                        line, anchor: SourceAnchor::at(line, 0, None),
                    }];
                }
                CompClause::For { var, .. } => {
                    let pi = total_fors - 1 - fors_from_end;
                    fors_from_end += 1;
                    body = vec![Stmt::ForEach {
                        vars: vec![var.to_string()],
                        iter: Expr::Name(format!("__psa_gx_{}", pi), ExprSpan::synthetic()),
                        body,
                        els: None,
                        line, anchor: SourceAnchor::at(line, 0, None),
                    }];
                }
            }
        }
        let mut gparams: Vec<(String, Option<Expr>)> = (0..total_fors)
            .map(|i| (format!("__psa_gx_{}", i), None))
            .collect();
        for n in &extra {
            gparams.push((n.clone(), None));
        }
        let gfunc = Func {
            name: gname.clone(),
            params: gparams,
            vararg: None,
            kwarg: None,
            is_async: false,
            body,
            line, anchor: SourceAnchor::at(line, 0, None),
        };
        self.defs.insert(gname.clone(), gfunc.clone());
        self.emit_generator_def(&gfunc)?;
        self.gen_names.push(gname.clone());
        let mut codes: Vec<String> = Vec::new();
        for (_, iter) in &fors {
            match self.emit_dyn_value(ctx, iter, line) {
                Ok(c) => codes.push(c),
                Err(_) => {
                    // A static iterable (a static list variable
                    // or literal) cannot box directly; copy it
                    // into a dynamic list through a synthetic
                    // list comprehension, whose own iteration
                    // paths cover the static face.
                    let comp = Expr::Comp {
                        kind: CompKind::List,
                        key: Box::new(Expr::Name("__psa_gxe".to_string(), ExprSpan::synthetic())),
                        value: None,
                        clauses: vec![CompClause::For {
                            var: "__psa_gxe".to_string(),
                            iter: (*iter).clone(),
                        }],
                    };
                    codes.push(self.emit_comp(ctx, &comp, line)?);
                }
            }
        }
        // D8: the extra parameters are the enclosing scope's
        // own variables (cells / the environment), passed by
        // name; emit_dyn_value boxes a statically-typed one
        // if it ever occurs.
        for n in &extra {
            codes.push(self.emit_dyn_value(ctx, &Expr::Name(n.clone(), ExprSpan::synthetic()), line)?);
        }
        Ok(format!("{}({})", self.sla_name(&gname), codes.join(", ")))
    }

    /// A comprehension materialised in a temp (D5): the result
    /// container is built by the clause nest emitted as
    /// statements. Loop variables live in a pushed scope
    /// (CPython 3 comprehension scoping: they do not leak).
    fn emit_comp(&mut self, ctx: &mut Ctx, e: &Expr, line: u32) -> Result<String, Diagnostic> {
        let (kind, key, value, clauses) = match e {
            Expr::Comp { kind, key, value, clauses } => (*kind, key, value, clauses),
            _ => unreachable!(),
        };
        if kind == CompKind::GenExp {
            return self.emit_genexp(ctx, key, value, clauses, line);
        }
        // Walrus targets inside a comprehension bind in the
        // CONTAINING scope in CPython, but the clause nest
        // below is emitted under a pushed scope mark that
        // truncates every declaration made inside it — a
        // statically-typed first binding would be forgotten
        // at the end of the comprehension (and its SLA `let`
        // is trapped inside the loop text). Materialise each
        // not-yet-bound target as a dynamic variable BEFORE
        // the mark: hoist_declare puts it in the function
        // prologue, and the assignment machinery then boxes
        // static values into it like any dynamic binding.
        // Targets the enclosing scope already holds are left
        // alone — the assignment rebinds that variable, which
        // is exactly CPython's rule. A target naming one of
        // the comprehension's own iteration variables is a
        // CPython SyntaxError, refused here where the key and
        // the clauses are both in view.
        {
            let mut targets: Vec<String> = Vec::new();
            collect_walrus_targets(key, &mut targets);
            if let Some(v) = value.as_deref() {
                collect_walrus_targets(v, &mut targets);
            }
            for c in clauses {
                match c {
                    CompClause::For { iter, .. } => collect_walrus_targets(iter, &mut targets),
                    CompClause::If(cond) => collect_walrus_targets(cond, &mut targets),
                }
            }
            for t in &targets {
                if clauses
                    .iter()
                    .any(|c| matches!(c, CompClause::For { var, .. } if var == t))
                {
                    return Err(eerr(
                        line,
                        &format!(
                            "assignment expression cannot rebind comprehension iteration variable `{}` (CPython raises SyntaxError); rename the walrus target",
                            t
                        ),
                    ));
                }
            }
            for t in &targets {
                if ctx.lookup(t).is_none() {
                    ctx.hoist_declare(t, Ty::Dyn);
                }
            }
        }
        self.used_dyn = true;
        ctx.tmp += 1;
        let res = format!("__psa_ctmp_{}", ctx.tmp);
        match kind {
            CompKind::List => ctx.line(&format!("let {} = __psa_list_new();", res)),
            CompKind::Set => ctx.line(&format!("let {} = __psa_set_new();", res)),
            CompKind::Dict => ctx.line(&format!("let {} = __psa_dict_new();", res)),
            CompKind::GenExp => unreachable!(),
        }
        ctx.scopes.push(ctx.vars.len());
        let r = self.emit_comp_clauses(ctx, clauses, 0, &res, kind, key, value.as_deref(), line);
        let mark = ctx.scopes.pop().unwrap();
        ctx.vars.truncate(mark);
        r?;
        Ok(res)
    }

    fn emit_comp_clauses(
        &mut self,
        ctx: &mut Ctx,
        clauses: &[CompClause],
        i: usize,
        res: &str,
        kind: CompKind,
        key: &Expr,
        value: Option<&Expr>,
        line: u32,
    ) -> Result<(), Diagnostic> {
        if i == clauses.len() {
            match kind {
                CompKind::List => {
                    let c = self.emit_dyn_value(ctx, key, line)?;
                    ctx.line(&format!("__psa_list_push({}, {});", res, c));
                }
                CompKind::Set => {
                    let c = self.emit_dyn_value(ctx, key, line)?;
                    ctx.line(&format!("__psa_set_add({}, {});", res, c));
                }
                CompKind::Dict => {
                    let k = self.emit_dyn_value(ctx, key, line)?;
                    let v = self.emit_dyn_value(ctx, value.expect("dict comp value"), line)?;
                    ctx.line(&format!("__psa_dict_set({}, {}, {});", res, k, v));
                }
                CompKind::GenExp => unreachable!(),
            }
            return Ok(());
        }
        match &clauses[i] {
            CompClause::If(cond) => {
                let (ccode, cty) = self.emit_expr(ctx, cond, line)?;
                let ccode = self.cond_code(ccode, cty, line, "comprehension condition")?;
                ctx.line(&format!("if {} {{", ccode));
                ctx.indent += 1;
                let r = self.emit_comp_clauses(ctx, clauses, i + 1, res, kind, key, value, line);
                ctx.indent -= 1;
                ctx.line("};");
                r
            }
            CompClause::For { var, iter } => {
                self.emit_comp_for(ctx, clauses, i, var, iter, res, kind, key, value, line)
            }
        }
    }

    fn emit_comp_for(
        &mut self,
        ctx: &mut Ctx,
        clauses: &[CompClause],
        i: usize,
        var: &str,
        iter: &Expr,
        res: &str,
        kind: CompKind,
        key: &Expr,
        value: Option<&Expr>,
        line: u32,
    ) -> Result<(), Diagnostic> {
        ctx.tmp += 1;
        let n = ctx.tmp;
        let idx = format!("__psa_ci_{}", n);
        // range(...) iterates numerically.
        if let Expr::Call(callee, args, kwargs, _) = iter {
            if let Expr::Name(f, _) = &**callee {
                if f == "range" && kwargs.is_empty() {
                    if args.len() > 2 {
                        return Err(eerr(
                            line,
                            "range() with a step is not supported in comprehensions in v1",
                        ));
                    }
                    let (start_e, stop_e) = match args.len() {
                        1 => (None, &args[0]),
                        2 => (Some(&args[0]), &args[1]),
                        _ => return Err(eerr(line, "range() takes 1 or 2 arguments in v1")),
                    };
                    let mut stop_code = self.emit_comp_bound(ctx, stop_e, line)?;
                    ctx.tmp += 1;
                    let stop_t = format!("__psa_cs_{}", ctx.tmp);
                    ctx.line(&format!("let {}: i64 = {};", stop_t, stop_code));
                    stop_code = stop_t;
                    let start_code = match start_e {
                        Some(e) => self.emit_comp_bound(ctx, e, line)?,
                        None => "0".to_string(),
                    };
                    ctx.declare(var, Ty::Int).map_err(|m| eerr(line, &m))?;
                    ctx.line(&format!("let {}: i64 = {};", idx, start_code));
                    ctx.line(&format!("while {} < {} {{", idx, stop_code));
                    ctx.indent += 1;
                    ctx.line(&format!("let {} = {};", var, idx));
                    let r = self.emit_comp_clauses(ctx, clauses, i + 1, res, kind, key, value, line);
                    ctx.line(&format!("{} = {} + 1;", idx, idx));
                    ctx.indent -= 1;
                    ctx.line("}");
                    return r;
                }
            }
        }
        // A static list variable iterates by index.
        if let Expr::Name(name, span) = iter {
            if let Some(t) = ctx.lookup(name) {
                if let Some(elem) = t.elem() {
                    ctx.check_alive(name, line, Some(span.start))?;
                    ctx.declare(var, elem).map_err(|m| eerr(line, &m))?;
                    ctx.line(&format!("let {}: i64 = 0;", idx));
                    ctx.line(&format!("while {} < (len({}) as i64) {{", idx, name));
                    ctx.indent += 1;
                    ctx.line(&format!("let {} = {}[{}];", var, name, idx));
                    let r = self.emit_comp_clauses(ctx, clauses, i + 1, res, kind, key, value, line);
                    ctx.line(&format!("{} = {} + 1;", idx, idx));
                    ctx.indent -= 1;
                    ctx.line("}");
                    return r;
                }
            }
        }
        // Everything else iterates through the dynamic protocol.
        let (icode, ity) = self.emit_expr(ctx, iter, line)?;
        if ity != Ty::Dyn {
            return Err(eerr(
                line,
                &format!(
                    "a comprehension can only iterate a dynamic value, a static list or range(...), not {}",
                    ity.py_name()
                ),
            ));
        }
        let icode = self.dyn_clone_wrap(icode, Ty::Dyn);
        // D7: a generator iterable is drained into a list first
        // (same rule as the for statement).
        let icode = if self.used_gen {
            format!("__psa_iter_source({})", icode)
        } else {
            icode
        };
        ctx.tmp += 1;
        let it = format!("__psa_cit_{}", ctx.tmp);
        ctx.line(&format!("let {} = {};", it, icode));
        ctx.declare(var, Ty::Dyn).map_err(|m| eerr(line, &m))?;
        ctx.line(&format!("let {}: i64 = 0;", idx));
        ctx.line(&format!("while {} < __psa_dyn_len({}) {{", idx, it));
        ctx.indent += 1;
        ctx.line(&format!("let {} = __psa_iter_get({}, {});", var, it, idx));
        let r = self.emit_comp_clauses(ctx, clauses, i + 1, res, kind, key, value, line);
        ctx.line(&format!("{} = {} + 1;", idx, idx));
        ctx.indent -= 1;
        ctx.line("}");
        r
    }

    /// A range bound in a comprehension: an int expression, or a
    /// dynamic value unboxed with the runtime type assertion.
    fn emit_comp_bound(
        &mut self,
        ctx: &mut Ctx,
        e: &Expr,
        line: u32,
    ) -> Result<String, Diagnostic> {
        let (code, ty) = self.emit_expr(ctx, e, line)?;
        match ty {
            Ty::Int => Ok(code),
            Ty::Dyn => self.unbox_code(code, Ty::Int, line),
            _ => Err(eerr(line, "range() bounds must be int")),
        }
    }

    /// Element codes and types for a list literal, shared by the
    /// static/dynamic decision and the forced-dynamic path used for
    /// nesting. String literals become dynamic strings and nested
    /// container literals become dynamic container temps.
    fn list_elem_codes(
        &mut self,
        ctx: &mut Ctx,
        elems: &[Expr],
        line: u32,
    ) -> Result<(Vec<String>, Vec<Ty>), Diagnostic> {
        let mut codes = Vec::new();
        let mut tys = Vec::new();
        for e in elems {
            if let Expr::Str(s) = e {
                codes.push(self.emit_str_temp(ctx, s, line)?);
                tys.push(Ty::Dyn);
                continue;
            }
            if matches!(e, Expr::ListLit(_) | Expr::DictLit(_)) {
                codes.push(self.emit_container_temp(ctx, e, line)?);
                tys.push(Ty::Dyn);
                continue;
            }
            let (code, ty) = self.emit_expr(ctx, e, line)?;
            codes.push(code);
            tys.push(ty);
        }
        Ok((codes, tys))
    }

    /// A list literal forced onto the dynamic path (used when the
    /// list itself is an element or value of another container).
    fn emit_list_literal_dyn(
        &mut self,
        ctx: &mut Ctx,
        elems: &[Expr],
        line: u32,
    ) -> Result<Vec<String>, Diagnostic> {
        let (codes, tys) = self.list_elem_codes(ctx, elems, line)?;
        if tys.iter().any(|t| !matches!(t, Ty::Int | Ty::Float | Ty::Bool | Ty::Dyn)) {
            return Err(eerr(
                line,
                "list elements of this type cannot become dynamic values",
            ));
        }
        self.used_dyn = true;
        let mut boxed = Vec::new();
        for (code, ty) in codes.into_iter().zip(tys.iter()) {
            boxed.push(match ty {
                Ty::Dyn => code,
                t => self.box_code(code, *t, line)?,
            });
        }
        Ok(boxed)
    }

    /// Extract a static value from a dynamic one at a boundary; the
    /// runtime checks the tag and panics TypeError-style on mismatch
    /// (design s3: unboxing carries a runtime type assertion).
    fn unbox_code(&mut self, code: String, want: Ty, line: u32) -> Result<String, Diagnostic> {
        self.used_dyn = true;
        match want {
            Ty::Int => Ok(format!("__psa_unbox_int({})", code)),
            Ty::Float => Ok(format!("__psa_unbox_float({})", code)),
            Ty::Bool => Ok(format!("__psa_unbox_bool({})", code)),
            Ty::Str => Err(eerr(
                line,
                "a dynamic value cannot be unboxed to a string in D1 (the string runtime is D2)",
            )),
            t if t.is_list() => Err(eerr(
                line,
                "a dynamic value cannot be unboxed to a list in D1 (dynamic containers arrive in D2)",
            )),
            _ => Err(eerr(
                line,
                &format!("a dynamic value cannot be unboxed to {}", want.py_name()),
            )),
        }
    }

    /// A dynamic value read into a *binding* (assignment target,
    /// materialised temp): on the Rust sla build-exe route a plain
    /// `let y = v;` moves a struct variable even when the struct is
    /// `@derive(copy)` (the checker honours the derive, the SAB
    /// lowering's assign path does not; hand-written probes q1/q2/q3
    /// pin calls as copying and bindings as moving). Reading a user
    /// variable into a binding therefore goes through __psa_clone,
    /// whose call copies. psa's own temps (the `__psa_` namespace)
    /// are exempt: they are dead after the binding, so moving them
    /// is correct and avoids leaving live registers behind.
    fn dyn_clone_wrap(&mut self, code: String, ty: Ty) -> String {
        let plain_ident = !code.is_empty()
            && code.chars().all(|c| c.is_alphanumeric() || c == '_')
            && !code.chars().next().unwrap().is_numeric();
        if ty == Ty::Dyn && plain_ident && !code.starts_with("__psa_") {
            self.used_dyn = true;
            format!("__psa_clone({})", code)
        } else {
            code
        }
    }

    /// Store a dynamic value into an existing PsaVal variable
    /// (assignment, augmented assignment, branch join). A whole-value
    /// `dst = v;` is not available on the Rust sla build-exe route:
    /// the SAB lowering emits every aggregate assignment as a fresh
    /// register definition and traps RegisterRedefinition (probes
    /// q4/q5/q6/q9), and a block-scoped aggregate temp is never
    /// retired (q16/q29). The proven shape (probe q28) is: capture
    /// the value once in a non-Copy stage, copy its scalar fields
    /// into the destination, then move the stage into a void sink,
    /// which retires it in the same block.
    fn emit_dyn_store(
        &mut self,
        ctx: &mut Ctx,
        target: &str,
        psaval_code: &str,
    ) -> Result<(), String> {
        self.used_dyn = true;
        ctx.tmp += 1;
        let stg = format!("__psa_stg_{}", ctx.tmp);
        ctx.declare(&stg, Ty::Dyn)?;
        ctx.line(&format!("let {}: PsaStage = __psa_stage_of({});", stg, psaval_code));
        ctx.line(&format!("{}.tag = {}.tag;", target, stg));
        ctx.line(&format!("{}.ipay = {}.ipay;", target, stg));
        ctx.line(&format!("{}.fpay = {}.fpay;", target, stg));
        ctx.line(&format!("__psa_drop_stage({});", stg));
        Ok(())
    }

    /// Bind `name` to an already-emitted dynamic value code
    /// (D5 unpacking and multi-target loop variables): mirrors
    /// the dynamic path of Stmt::Assign -- a fresh binding
    /// inside a block is hoisted to the function prologue and
    /// filled through the stage pattern; an existing dynamic
    /// binding is stored into; any other existing binding is a
    /// loud type-pinning error.
    fn emit_bind_dyn_code(
        &mut self,
        ctx: &mut Ctx,
        name: &str,
        code: &str,
        line: u32,
    ) -> Result<(), Diagnostic> {
        ctx.clear_deleted(name);
        match ctx.lookup(name) {
            None => {
                if ctx.scopes.len() > 1 {
                    ctx.hoist_declare(name, Ty::Dyn);
                } else {
                    ctx.declare(name, Ty::Dyn).map_err(|m| eerr(line, &m))?;
                    ctx.line(&format!(
                        "let {}: {} = {};",
                        name,
                        Ty::Dyn.sla(),
                        Ty::Dyn.default_lit()
                    ));
                }
                self.emit_dyn_store(ctx, name, code).map_err(|m| eerr(line, &m))
            }
            Some(Ty::Dyn) => {
                self.emit_dyn_store(ctx, name, code).map_err(|m| eerr(line, &m))?;
                ctx.revive(name);
                Ok(())
            }
            Some(existing) => Err(eerr(
                line,
                &format!(
                    "variable `{}` was inferred as {} from its first binding and cannot receive a dynamic unpacked value (psa pins one type per variable)",
                    name,
                    existing.py_name()
                ),
            )),
        }
    }

    /// A condition position: bools pass through, dynamic values go
    /// through the runtime truthiness check, anything else keeps the
    /// caller's historical error message.
    fn cond_code(&mut self, code: String, ty: Ty, line: u32, what: &str) -> Result<String, Diagnostic> {
        match ty {
            Ty::Bool => Ok(code),
            Ty::Dyn => {
                self.used_dyn = true;
                Ok(format!("__psa_dyn_truthy({})", code))
            }
            _ => Err(eerr(line, what)),
        }
    }

    /// Type-check a list literal's elements and emit their codes.
    /// Homogeneous int/bool literals stay monomorphic static Vecs;
    /// dynamic elements, mixed scalar element types, float elements
    /// and the empty literal become dynamic heap lists (D2): every
    /// element code is boxed to a PsaVal. String elements are heap
    /// values whose dynamic kind lands later in D2 and stay refused.
    fn emit_list_literal(
        &mut self,
        ctx: &mut Ctx,
        elems: &[Expr],
        line: u32,
    ) -> Result<ListLit, Diagnostic> {
        let (codes, tys) = self.list_elem_codes(ctx, elems, line)?;
        let dyn_list = tys.is_empty()
            || tys.iter().any(|t| *t == Ty::Dyn || *t == Ty::Float)
            || tys.iter().any(|t| *t != tys[0]);
        if dyn_list {
            if tys.iter().any(|t| !matches!(t, Ty::Int | Ty::Float | Ty::Bool | Ty::Dyn)) {
                return Err(eerr(
                    line,
                    "a static container value cannot become a dynamic element (only container literals nest)",
                ));
            }
            self.used_dyn = true;
            let mut boxed = Vec::new();
            for (code, ty) in codes.into_iter().zip(tys.iter()) {
                boxed.push(match ty {
                    Ty::Dyn => code,
                    t => self.box_code(code, *t, line)?,
                });
            }
            return Ok(ListLit::Dyn(boxed));
        }
        let elem_ty = tys[0];
        if Ty::list_of(elem_ty).is_none() {
            return Err(eerr(
                line,
                &format!(
                    "list elements of type {:?} are not supported (scalar int/bool elements, or a dynamic list)",
                    elem_ty
                ),
            ));
        }
        Ok(ListLit::Static(Ty::list_of(elem_ty).unwrap(), codes))
    }

    /// Emit the construction of a dynamic heap list bound to `name`:
    /// a fresh handle plus one runtime push per element code.
    fn emit_dyn_list_construct(
        &mut self,
        ctx: &mut Ctx,
        name: &str,
        codes: &[String],
        declare: bool,
    ) {
        self.used_dyn = true;
        if declare {
            ctx.line(&format!("let {} = __psa_list_new();", name));
        } else {
            ctx.line(&format!("{} = __psa_list_new();", name));
        }
        for code in codes {
            ctx.line(&format!("__psa_list_push({}, {});", name, code));
        }
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
    ) -> Result<(), Diagnostic> {
        match self.emit_list_literal(ctx, elems, line)? {
            ListLit::Dyn(codes) => match ctx.lookup(name) {
                None => {
                    ctx.declare(name, Ty::Dyn).map_err(|m| eerr(line, &m))?;
                    self.emit_dyn_list_construct(ctx, name, &codes, true);
                    Ok(())
                }
                Some(Ty::Dyn) => {
                    ctx.tmp += 1;
                    let tmp = format!("__psa_ltmp_{}", ctx.tmp);
                    ctx.declare(&tmp, Ty::Dyn).map_err(|m| eerr(line, &m))?;
                    self.emit_dyn_list_construct(ctx, &tmp, &codes, true);
                    self.emit_dyn_store(ctx, name, &tmp).map_err(|m| eerr(line, &m))?;
                    ctx.revive(name);
                    Ok(())
                }
                Some(other) => Err(eerr(
                    line,
                    &format!(
                        "variable `{}` changes type from {} to a dynamic list",
                        name,
                        other.py_name()
                    ),
                )),
            },
            ListLit::Static(list_ty, codes) => match ctx.lookup(name) {
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
            },
        }
    }

    /// Type-check a dict literal's pairs and emit key/value codes.
    /// Keys are int/bool/None expressions or dynamic values (the
    /// runtime normalises bool keys onto int keys and None onto a
    /// sentinel, matching Python key identity for those kinds).
    /// Float keys stay loud (Python float key identity needs
    /// value-based hashing) and string keys land with the ordered-dict
    /// string runtime. Values are any D1 scalar or dynamic value,
    /// boxed on the way in.
    fn emit_dict_literal(
        &mut self,
        ctx: &mut Ctx,
        pairs: &[(Expr, Expr)],
        line: u32,
    ) -> Result<Vec<(String, String)>, Diagnostic> {
        self.used_dyn = true;
        let mut out = Vec::new();
        for (ke, ve) in pairs {
            // A string literal key becomes a dynamic string temp
            // (D3: str keys are first-class under key equivalence).
            let (kcode, kty) = if let Expr::Str(s) = ke {
                (self.emit_str_temp(ctx, s, line)?, Ty::Dyn)
            } else {
                self.emit_expr(ctx, ke, line)?
            };
            let kcode = match kty {
                Ty::Dyn => kcode,
                Ty::Int | Ty::Bool | Ty::Float => self.box_code(kcode, kty, line)?,
                _ => {
                    return Err(eerr(
                        line,
                        "dict keys must be int, float, bool, None, str or dynamic values",
                    ))
                }
            };
            let (mut vcode, vty) = if let Expr::Str(s) = ve {
                (self.emit_str_temp(ctx, s, line)?, Ty::Dyn)
            } else if matches!(ve, Expr::ListLit(_) | Expr::DictLit(_)) {
                (self.emit_container_temp(ctx, ve, line)?, Ty::Dyn)
            } else {
                self.emit_expr(ctx, ve, line)?
            };
            if vty != Ty::Dyn {
                vcode = self.box_code(vcode, vty, line)?;
            }
            out.push((kcode, vcode));
        }
        Ok(out)
    }

    /// Emit the construction of a dynamic dict bound to `name`: a
    /// fresh handle plus one runtime set per key/value code pair.
    fn emit_dyn_dict_construct(
        &mut self,
        ctx: &mut Ctx,
        name: &str,
        pairs: &[(String, String)],
        declare: bool,
    ) {
        self.used_dyn = true;
        if declare {
            ctx.line(&format!("let {} = __psa_dict_new();", name));
        } else {
            ctx.line(&format!("{} = __psa_dict_new();", name));
        }
        for (k, v) in pairs {
            ctx.line(&format!("__psa_dict_set({}, {}, {});", name, k, v));
        }
    }

    fn emit_dict_assign(
        &mut self,
        ctx: &mut Ctx,
        name: &str,
        pairs: &[(Expr, Expr)],
        line: u32,
    ) -> Result<(), Diagnostic> {
        let codes = self.emit_dict_literal(ctx, pairs, line)?;
        match ctx.lookup(name) {
            None => {
                ctx.declare(name, Ty::Dyn).map_err(|m| eerr(line, &m))?;
                self.emit_dyn_dict_construct(ctx, name, &codes, true);
                Ok(())
            }
            Some(Ty::Dyn) => {
                ctx.tmp += 1;
                let tmp = format!("__psa_ltmp_{}", ctx.tmp);
                ctx.declare(&tmp, Ty::Dyn).map_err(|m| eerr(line, &m))?;
                self.emit_dyn_dict_construct(ctx, &tmp, &codes, true);
                self.emit_dyn_store(ctx, name, &tmp).map_err(|m| eerr(line, &m))?;
                ctx.revive(name);
                Ok(())
            }
            Some(other) => Err(eerr(
                line,
                &format!(
                    "variable `{}` changes type from {} to a dynamic dict",
                    name,
                    other.py_name()
                ),
            )),
        }
    }

    /// Compute the (code, type) of a return value, emitting any
    /// construction lines first: container literals and string
    /// literals build a temp, exactly as the plain return path
    /// has always done. Shared by the plain return path and the
    /// D6 try-intercepted return path.
    fn return_value_code(
        &mut self,
        ctx: &mut Ctx,
        e: &Expr,
        line: u32,
    ) -> Result<(String, Ty), Diagnostic> {
        match e {
            Expr::ListLit(elems) => match self.emit_list_literal(ctx, elems, line)? {
                ListLit::Static(list_ty, codes) => {
                    ctx.tmp += 1;
                    let tmp = format!("__psa_ret_{}", ctx.tmp);
                    self.emit_list_construct(ctx, &tmp, list_ty, &codes, true);
                    Ok((tmp, list_ty))
                }
                ListLit::Dyn(codes) => {
                    ctx.tmp += 1;
                    let tmp = format!("__psa_ret_{}", ctx.tmp);
                    self.emit_dyn_list_construct(ctx, &tmp, &codes, true);
                    Ok((tmp, Ty::Dyn))
                }
            },
            Expr::DictLit(pairs) => {
                let codes = self.emit_dict_literal(ctx, pairs, line)?;
                ctx.tmp += 1;
                let tmp = format!("__psa_ret_{}", ctx.tmp);
                self.emit_dyn_dict_construct(ctx, &tmp, &codes, true);
                Ok((tmp, Ty::Dyn))
            }
            Expr::Str(s) => {
                if ctx.fname == "main" {
                    return Err(eerr(line, "main must return an int"));
                }
                let tmp = self.emit_str_temp(ctx, s, line)?;
                Ok((tmp, Ty::Dyn))
            }
            _ => {
                if let Expr::Name(n, _) = e {
                    if ctx.borrowed.contains(n) {
                        return Err(eerr(
                            line,
                            &format!(
                                "cannot return `{}`: it is a borrowed parameter of `{}` and borrows may not escape (spec s7)",
                                n, ctx.fname
                            ),
                        ));
                    }
                }
                let (code, ty) = self.emit_expr(ctx, e, line)?;
                if ty == Ty::Str {
                    return Err(eerr(
                        line,
                        "returning a string expression is not supported yet: only plain string literals return (as dynamic strings); f-string results do not in this slice",
                    ));
                }
                Ok((code, ty))
            }
        }
    }

    /// D6: a return statement inside at least one try frame.
    /// The value is computed now (CPython evaluates it before
    /// the finally runs), stored in the innermost frame's slot,
    /// and the frame's return flag is set; the frame epilogues
    /// then propagate it outward frame by frame, running every
    /// finally, until the outermost epilogue performs the real
    /// return.
    fn emit_return_intercepted(
        &mut self,
        ctx: &mut Ctx,
        value: &Option<Expr>,
        line: u32,
    ) -> Result<(), Diagnostic> {
        let (code, slot_ty): (Option<String>, Ty) = match value {
            None => {
                if ctx.fname != "main" {
                    if ctx.dyn_ret {
                        unify_ret(ctx, Ty::Dyn, line)?;
                    } else {
                        unify_ret(ctx, Ty::Void, line)?;
                    }
                }
                (None, Ty::Void)
            }
            Some(e) => {
                let (mut code, ty) = self.return_value_code(ctx, e, line)?;
                if ctx.fname == "main" {
                    if ty != Ty::Int {
                        return Err(eerr(line, "main must return an int"));
                    }
                    (Some(code), Ty::Int)
                } else {
                    unify_ret(ctx, ty, line)?;
                    let sty = if ctx.dyn_ret && ty != Ty::Dyn {
                        code = self.box_code(code, ty, line)?;
                        Ty::Dyn
                    } else {
                        ty
                    };
                    (Some(code), sty)
                }
            }
        };
        let (slot_name, ret_flag, need_decl) = {
            let frame = ctx.try_stack.last_mut().expect("intercepted return needs a frame");
            frame.ret_used = true;
            let need_decl = match &code {
                None => {
                    frame.ret_void = true;
                    false
                }
                Some(_) => {
                    if frame.ret_ty.is_none() {
                        frame.ret_ty = Some(slot_ty);
                        true
                    } else if frame.ret_ty != Some(slot_ty) {
                        return Err(eerr(
                            line,
                            "returns crossing a try block must agree on one type in v1 (the deferred-return slot is monomorphic; make the returns dynamic or restructure)",
                        ));
                    } else {
                        false
                    }
                }
            };
            (frame.slot_name.clone(), frame.ret_flag.clone(), need_decl)
        };
        if need_decl {
            ctx.prologue_push(&format!(
                "    let {}: {} = {};\n",
                slot_name,
                slot_ty.sla(),
                slot_ty.default_lit()
            ));
        }
        if let Some(code) = &code {
            if slot_ty == Ty::Dyn {
                self.emit_dyn_store(ctx, &slot_name, code)
                    .map_err(|m| eerr(line, &m))?;
            } else {
                ctx.line(&format!("{} = {};", slot_name, code));
            }
        }
        ctx.line(&format!("{} = true;", ret_flag));
        ctx.line("break;");
        Ok(())
    }

    /// D6: mark this function as using the pending-exception
    /// machinery, once. The pending slot is process-global —
    /// the sa_std globals cells behind __psa_exc_get /
    /// __psa_exc_store / __psa_exc_tag in the D6 runtime —
    /// because an exception raised in a callee must be
    /// visible to the caller's handler dispatch. The original
    /// per-function slot could never cross a call boundary:
    /// a raise in a helper panicked 77100 at its own raise
    /// site even when the caller had a matching handler
    /// (exc1's div/outer cases, batch 6). Pending <=> the
    /// stored tag is not 3 (None); exception values are
    /// always tag 8 with fpay 0.0, so storing (tag, ipay)
    /// is lossless.
    fn ensure_exc_slot(&mut self, ctx: &mut Ctx) {
        if !ctx.exc_slot {
            ctx.exc_slot = true;
            self.used_dyn = true;
        }
    }

    /// D6: the statement that unwinds the current function
    /// when a pending exception reaches a point no try in
    /// this function will catch. In main there is no caller,
    /// so the exception is uncaught (panic 77100); anywhere
    /// else the function returns its default value and the
    /// caller's own trailing check keeps the unwinding
    /// going, frame by frame, until a handler matches.
    fn exc_unwind_action(&self, ctx: &Ctx) -> String {
        if ctx.fname == "main" {
            return "panic(77100);".to_string();
        }
        match ctx.ret {
            // No inferred return: the SLA signature falls
            // back to i32 (Ty::Void's SLA type is also i32,
            // default 0), never a void function.
            None => "return 0;".to_string(),
            Some(t) => format!("return {};", t.default_lit()),
        }
    }

    /// D6: open a try frame: declare its flags in the prologue
    /// and push it on the frame stack.
    fn begin_try(&mut self, ctx: &mut Ctx) {
        self.ensure_exc_slot(ctx);
        ctx.tmp += 1;
        let id = ctx.tmp;
        let frame = TryFrame {
            brk_flag: format!("__psa_tf{}_brk", id),
            cont_flag: format!("__psa_tf{}_cont", id),
            ret_flag: format!("__psa_tf{}_ret", id),
            slot_name: format!("__psa_tf{}_slot", id),
            loop_depth: ctx.loop_depth,
            brk_used: false,
            cont_used: false,
            ret_used: false,
            ret_void: false,
            ret_ty: None,
        };
        for f in [&frame.brk_flag, &frame.cont_flag, &frame.ret_flag] {
            ctx.prologue_push(&format!("    let {}: bool = false;\n", f));
        }
        ctx.try_stack.push(frame);
    }

    /// D6: emit a `break` for the current emission point. When
    /// the innermost try frame was entered at the current loop
    /// depth, the break targets a loop outside that frame: set
    /// its flag and break the frame's guard wrapper instead
    /// (the epilogue re-emits the real break after the finally).
    fn emit_break_through(&mut self, ctx: &mut Ctx) {
        // Loop-else: a break always targets the innermost
        // enclosing loop; when that loop has an else clause,
        // record that it fired. This also runs on the try
        // epilogue's re-emission of a deferred break (the flag
        // assignment is idempotent, and the epilogue still sits
        // inside the target loop's emission region).
        if let Some(Some(flag)) = self.loop_else_stack.last() {
            let flag = flag.clone();
            ctx.line(&format!("{} = true;", flag));
        }
        let target = match ctx.try_stack.last_mut() {
            Some(f) if ctx.loop_depth <= f.loop_depth => {
                f.brk_used = true;
                Some(f.brk_flag.clone())
            }
            _ => None,
        };
        match target {
            Some(flag) => {
                ctx.line(&format!("{} = true;", flag));
                ctx.line("break;");
            }
            None => ctx.line("break;"),
        }
    }

    /// D6: the continue twin of emit_break_through.
    fn emit_continue_through(&mut self, ctx: &mut Ctx) {
        let target = match ctx.try_stack.last_mut() {
            Some(f) if ctx.loop_depth <= f.loop_depth => {
                f.cont_used = true;
                Some(f.cont_flag.clone())
            }
            _ => None,
        };
        match target {
            Some(flag) => {
                ctx.line(&format!("{} = true;", flag));
                ctx.line("break;");
            }
            None => ctx.line("continue;"),
        }
    }

    /// Loop-else: open the emission of a loop that may carry an
    /// `else:` clause. When a clause exists, declare its
    /// break-flag variable at the current position (a plain
    /// `let`, like the other loop temps) and return its name;
    /// the loop's entry is pushed on the loop-else stack either
    /// way, and the caller pops it right after the loop body,
    /// before the else clause itself is emitted (a break inside
    /// the else clause targets the NEXT loop out, as in
    /// CPython).
    fn loop_else_begin(
        &mut self,
        ctx: &mut Ctx,
        els: &Option<Vec<Stmt>>,
    ) -> Option<String> {
        let flag = if els.is_some() {
            ctx.tmp += 1;
            let f = format!("__psa_broke_{}", ctx.tmp);
            ctx.line(&format!("let {}: bool = false;", f));
            Some(f)
        } else {
            None
        };
        self.loop_else_stack.push(flag.clone());
        flag
    }

    /// Loop-else: emit a loop's else clause after its body. The
    /// clause runs only on a clean, break-free exit: the break
    /// flag is clear, no exception is pending, and no control
    /// flag is set on the try frame enclosing the loop (the
    /// same condition shape as try's own else clause; all
    /// operands are side-effect-free, so SLA's eager `&&` is
    /// safe here).
    fn emit_loop_else_body(
        &mut self,
        ctx: &mut Ctx,
        els: &[Stmt],
        flag: &str,
    ) -> Result<(), Diagnostic> {
        let mut cond = format!("!{}", flag);
        if self.program_exc {
            cond = format!("__psa_exc_tag() == 3 && {}", cond);
            if let Some(f) = ctx.try_stack.last() {
                let (rf, bf, cf) =
                    (f.ret_flag.clone(), f.brk_flag.clone(), f.cont_flag.clone());
                cond = format!("{} && !{} && !{} && !{}", cond, rf, bf, cf);
            }
        }
        ctx.line(&format!("if {} {{", cond));
        ctx.indent += 1;
        let r = self.emit_block(ctx, els);
        ctx.indent -= 1;
        ctx.line("};");
        r
    }

    /// D6: the epilogue of a try frame (the frame has already
    /// been popped; any finally text has already been emitted).
    /// Deferred control flow propagates into the next frame or
    /// takes effect for real; a still-pending exception with no
    /// frame left is uncaught (panic 77100).
    fn emit_try_epilogue(
        &mut self,
        ctx: &mut Ctx,
        frame: TryFrame,
        line: u32,
    ) -> Result<(), Diagnostic> {
        if frame.ret_used {
            ctx.line(&format!("if {} {{", frame.ret_flag));
            ctx.indent += 1;
            if ctx.try_stack.is_empty() {
                if frame.ret_void {
                    if ctx.fname == "main" {
                        ctx.line("return 0;");
                    } else if ctx.dyn_ret {
                        ctx.line(&format!("return {};", Ty::Dyn.default_lit()));
                    } else {
                        ctx.line(&format!("return {};", Ty::Void.default_lit()));
                    }
                } else if ctx.fname == "main" {
                    ctx.line(&format!("return {} as i32;", frame.slot_name));
                } else {
                    ctx.line(&format!("return {};", frame.slot_name));
                }
            } else {
                let (oslot, oflag, need_decl) = {
                    let outer = ctx.try_stack.last_mut().expect("outer frame");
                    outer.ret_used = true;
                    if frame.ret_void {
                        outer.ret_void = true;
                    }
                    let need_decl = if frame.ret_void {
                        false
                    } else if outer.ret_ty.is_none() {
                        outer.ret_ty = frame.ret_ty;
                        true
                    } else if outer.ret_ty != frame.ret_ty {
                        return Err(eerr(
                            line,
                            "returns crossing nested try blocks must agree on one type in v1 (the deferred-return slot is monomorphic)",
                        ));
                    } else {
                        false
                    };
                    (outer.slot_name.clone(), outer.ret_flag.clone(), need_decl)
                };
                if need_decl {
                    let ty = frame.ret_ty.expect("a value return carries a slot type");
                    ctx.prologue_push(&format!(
                        "    let {}: {} = {};\n",
                        oslot,
                        ty.sla(),
                        ty.default_lit()
                    ));
                }
                if !frame.ret_void {
                    if frame.ret_ty == Some(Ty::Dyn) {
                        self.emit_dyn_store(ctx, &oslot, &frame.slot_name)
                            .map_err(|m| eerr(line, &m))?;
                    } else {
                        ctx.line(&format!("{} = {};", oslot, frame.slot_name));
                    }
                }
                ctx.line(&format!("{} = true;", oflag));
                ctx.line("break;");
            }
            ctx.indent -= 1;
            ctx.line("};");
        }
        if frame.brk_used {
            ctx.line(&format!("if {} {{", frame.brk_flag));
            ctx.indent += 1;
            self.emit_break_through(ctx);
            ctx.indent -= 1;
            ctx.line("};");
        }
        if frame.cont_used {
            ctx.line(&format!("if {} {{", frame.cont_flag));
            ctx.indent += 1;
            self.emit_continue_through(ctx);
            ctx.indent -= 1;
            ctx.line("};");
        }
        if ctx.try_stack.is_empty() {
            // No outer try in this function: a still-pending
            // exception propagates to the caller (in main it
            // is uncaught).
            let act = self.exc_unwind_action(ctx);
            ctx.line(&format!("if __psa_exc_tag() != 3 {{ {} }};", act));
        }
        Ok(())
    }

    /// D6: the handler match condition for the pending
    /// exception: the instance's class is the handler class or
    /// one of its descendants. Handler classes must derive from
    /// Exception (checked here, loudly).
    fn handler_cond(&mut self, h: &ExcHandler) -> Result<String, Diagnostic> {
        if h.classes.is_empty() {
            return Ok("true".to_string());
        }
        let mut parts: Vec<String> = Vec::new();
        for cname in &h.classes {
            let cid = match self.class_ids.get(cname) {
                Some(&i) => i as i64,
                None => {
                    return Err(eerr(
                        h.line,
                        &format!("except clause names unknown class `{}`", cname),
                    ))
                }
            };
            if !self.classes[cid as usize].ancestors.contains(&0) {
                return Err(eerr(
                    h.line,
                    &format!(
                        "except clause class `{}` does not derive from Exception",
                        cname
                    ),
                ));
            }
            for ci in &self.classes {
                if ci.ancestors.contains(&cid) {
                    parts.push(format!("__psa_obj_class(__psa_exc_get()) == {}", ci.id));
                }
            }
        }
        Ok(parts.join(" || "))
    }

    /// Flatten an attribute chain `a.b.c` into its segments
    /// (root first), or None when the chain is not rooted at a
    /// plain name.
    fn attr_chain(e: &Expr) -> Option<Vec<String>> {
        let mut segs = Vec::new();
        let mut cur = e;
        loop {
            match cur {
                Expr::Attr(base, name, _) => {
                    segs.push(name.clone());
                    cur = base;
                }
                Expr::Name(n, _) => {
                    segs.push(n.clone());
                    segs.reverse();
                    return Some(segs);
                }
                _ => return None,
            }
        }
    }

    /// D6: if `callee` is an attribute chain spelling a merged
    /// module def key (`foo.bar`, `foo.bar.baz`), return the
    /// key. A chain rooted at a bound variable never resolves
    /// to a module.
    fn module_def_key(&self, ctx: &Ctx, callee: &Expr) -> Option<String> {
        let segs = Self::attr_chain(callee)?;
        if segs.len() < 2 || ctx.lookup(&segs[0]).is_some() {
            return None;
        }
        let key = segs.join(".");
        if self.defs.contains_key(&key) {
            Some(key)
        } else {
            None
        }
    }

    /// D6: if `callee` is `<module>.ClassName` for a merged
    /// module, return the class name and registry index.
    fn module_class(&self, ctx: &Ctx, callee: &Expr) -> Option<(String, usize)> {
        let segs = Self::attr_chain(callee)?;
        if segs.len() < 2 || ctx.lookup(&segs[0]).is_some() {
            return None;
        }
        let (prefix, last) = segs.split_at(segs.len() - 1);
        let module = prefix.join(".");
        if !self.modules.iter().any(|m| m == &module) {
            return None;
        }
        let cname = last[0].clone();
        self.class_ids.get(&cname).map(|&cid| (cname, cid))
    }

    /// The dynamic code for a literal expression (the read
    /// shape of D4 class attributes, D6 module attributes and
    /// from-imported literal names), or None for a non-literal.
    fn literal_dyn_code(
        &mut self,
        ctx: &mut Ctx,
        e: &Expr,
        line: u32,
    ) -> Option<Result<(String, Ty), Diagnostic>> {
        match e {
            Expr::Int(v) => {
                self.used_dyn = true;
                Some(Ok((format!("__psa_box_int({})", v), Ty::Dyn)))
            }
            Expr::BigInt(d) => {
                self.used_dyn = true;
                Some(
                    self.emit_str_temp(ctx, d, line)
                        .map(|t| (format!("__psa_big_from_psastr({})", t), Ty::Dyn)),
                )
            }
            Expr::Float(v) => {
                self.used_dyn = true;
                Some(Ok((format!("__psa_box_float({})", fmt_f64(*v)), Ty::Dyn)))
            }
            Expr::Bool(b) => {
                self.used_dyn = true;
                Some(Ok((format!("__psa_box_bool({})", b), Ty::Dyn)))
            }
            Expr::Str(s) => Some(self.emit_str_temp(ctx, s, line).map(|t| (t, Ty::Dyn))),
            Expr::Bytes(bs) => Some(self.emit_bytes_temp(ctx, bs, line).map(|t| (t, Ty::Dyn))),
            Expr::NoneLit => {
                self.used_dyn = true;
                Some(Ok((Ty::Dyn.default_lit().to_string(), Ty::Dyn)))
            }
            _ => None,
        }
    }

    fn emit_block(&mut self, ctx: &mut Ctx, stmts: &[Stmt]) -> Result<(), Diagnostic> {
        ctx.scopes.push(ctx.vars.len());
        for s in stmts {
            self.emit_stmt(ctx, s)?;
        }
        let mark = ctx.scopes.pop().unwrap();
        ctx.vars.truncate(mark);
        Ok(())
    }

    /// Statement emission with the D6 exception guard: while
    /// emitting a try body (or a handler / else body), every
    /// statement is followed by a check that breaks the
    /// innermost guard wrapper when an exception is pending or
    /// a control-flow flag was set on the innermost try frame.
    /// The chain of guards is what unwinds a raise out of
    /// arbitrarily nested loops and blocks to the wrapper.
    /// Returns and plain control exits skip the guard: a return
    /// inside a finally discards a pending exception (CPython),
    /// and a break/continue is itself the exit.
    fn emit_stmt(&mut self, ctx: &mut Ctx, stmt: &Stmt) -> Result<(), Diagnostic> {
        // P1 source map: anchor every line this statement emits to
        // the statement's psa position (kind Stmt; Exact when the
        // post-rewrite anchor carries a column, else LineOnly).
        // Restored on exit so a nested statement cannot leak its
        // anchor into the enclosing statement's trailing lines.
        let saved = ctx.save_anchor();
        let stmt_a = stmt_anchor(stmt);
        let stmt_p = if stmt_a.col >= 1 { Precision::Exact } else { Precision::LineOnly };
        ctx.set_anchor(stmt_a, SegKind::Stmt, stmt_p, None);
        let r = self.emit_stmt_inner(ctx, stmt);
        if r.is_ok() && ctx.guard_depth > 0 {
            match stmt {
                Stmt::Return { .. } | Stmt::Break { .. } | Stmt::Continue { .. } => {}
                _ => match ctx.try_stack.last() {
                    Some(f) => {
                        let (rf, bf, cf) =
                            (f.ret_flag.clone(), f.brk_flag.clone(), f.cont_flag.clone());
                        ctx.line(&format!(
                            "if __psa_exc_tag() != 3 || {} || {} || {} {{ break; }};",
                            rf, bf, cf
                        ));
                    }
                    None => {
                        ctx.line("if __psa_exc_tag() != 3 { break; };");
                    }
                },
            }
        }
        // D6 cross-function propagation: outside any guard
        // wrapper, a statement that evaluated a call may have
        // left a pending exception behind (raised in a
        // callee). Unwind this frame immediately; a matching
        // handler in some caller will see the global slot.
        // Suppressed inside finally bodies (they must run to
        // completion; the try epilogue propagates after).
        if r.is_ok() && ctx.guard_depth == 0 && ctx.in_finally == 0 && self.program_exc {
            match stmt {
                Stmt::Return { .. }
                | Stmt::Break { .. }
                | Stmt::Continue { .. }
                | Stmt::Try { .. }
                | Stmt::Raise { .. }
                | Stmt::With { .. } => {}
                _ => {
                    if stmt_may_call(stmt) {
                        let act = self.exc_unwind_action(ctx);
                        ctx.line(&format!("if __psa_exc_tag() != 3 {{ {} }};", act));
                    }
                }
            }
        }
        // P1: a call noted while evaluating this statement must not
        // leak into a later statement's lines: the consuming line
        // was written inside this statement (or there was no such
        // line, e.g. a temp-materialised call — the marker is then
        // dropped rather than misattributed).
        ctx.clear_pending_call();
        ctx.restore_anchor(saved);
        r
    }

    fn emit_stmt_inner(&mut self, ctx: &mut Ctx, stmt: &Stmt) -> Result<(), Diagnostic> {
        match stmt {
            // D8: nested defs and global/nonlocal declarations
            // are consumed by the closure-conversion pass and
            // must never reach emission; reaching one is an
            // internal error, reported loudly.
            Stmt::Def(f) => Err(eerr(
                f.line,
                &format!(
                    "internal error: nested def `{}` reached emission without closure conversion",
                    f.name
                ),
            )),
            Stmt::Global { line, .. } | Stmt::Nonlocal { line, .. } => Err(eerr(
                *line,
                "internal error: a global/nonlocal declaration reached emission without closure conversion",
            )),
            // D8: publish the module environment at the head
            // of main: collect the module scope's cell
            // variables (already created by the preceding
            // assignments) into a fresh dynamic list and store
            // its pointer in the sa_std globals cell.
            Stmt::PublishModEnv { cells, .. } => {
                self.used_dyn = true;
                self.used_closure = true;
                ctx.line("let __psa_me = __psa_list_new();");
                for c in cells {
                    ctx.check_alive(c, 0, Some(SourceAnchor::synthetic()))?;
                    ctx.line(&format!("__psa_list_push(__psa_me, {});", c));
                }
                ctx.line("__psa_modenv_set(__psa_me);");
                Ok(())
            }
            Stmt::Pass { .. } => Ok(()),
            Stmt::Break { .. } => {
                self.emit_break_through(ctx);
                Ok(())
            }
            Stmt::Continue { .. } => {
                self.emit_continue_through(ctx);
                Ok(())
            }
            Stmt::Return { value, line, anchor: _} => {
                // D6: a return crossing a try frame is deferred
                // through the frame's flag/slot machinery, so
                // every enclosing finally runs before the return
                // takes effect (innermost first, by propagation).
                if !ctx.try_stack.is_empty() {
                    return self.emit_return_intercepted(ctx, value, *line);
                }
                match value {
                    None => {
                        if ctx.fname == "main" {
                            ctx.line("return 0;");
                        } else if ctx.dyn_ret {
                            unify_ret(ctx, Ty::Dyn, *line)?;
                            ctx.line(&format!("return {};", Ty::Dyn.default_lit()));
                        } else {
                            unify_ret(ctx, Ty::Void, *line)?;
                            ctx.line(&format!("return {};", Ty::Void.default_lit()));
                        }
                    }
                    Some(e) => {
                        let (mut code, ty) = self.return_value_code(ctx, e, *line)?;
                        if ctx.fname == "main" {
                            if ty != Ty::Int {
                                return Err(eerr(*line, "main must return an int"));
                            }
                            ctx.line(&format!("return {} as i32;", code));
                        } else {
                            unify_ret(ctx, ty, *line)?;
                            if ctx.dyn_ret && ty != Ty::Dyn {
                                code = self.box_code(code, ty, *line)?;
                            }
                            ctx.line(&format!("return {};", code));
                        }
                    }
                }
                Ok(())
            }
            Stmt::Assign { name, value, line, anchor: _} => {
                ctx.clear_deleted(name);
                // List literals are constructed statement-wise.
                if let Expr::ListLit(elems) = value {
                    return self.emit_list_assign(ctx, name, elems, *line);
                }
                // Dict literals likewise (always dynamic, D2 slice C).
                if let Expr::DictLit(pairs) = value {
                    return self.emit_dict_assign(ctx, name, pairs, *line);
                }
                // A plain string literal binds as a dynamic string
                // (tag 6, D2 slice C); other string expressions
                // (f-string results) do not bind yet.
                let (mut code, mut ty) = if let Expr::Str(s) = value {
                    (self.emit_str_temp(ctx, s, *line)?, Ty::Dyn)
                } else {
                    self.emit_expr(ctx, value, *line)?
                };
                if ty == Ty::Void {
                    return Err(eerr(*line, "cannot assign a void value"));
                }
                if ty == Ty::Str {
                    return Err(eerr(
                        *line,
                        "binding a string expression is not supported yet: only plain string literals bind (as dynamic strings); f-string results do not bind in this slice",
                    ));
                }
                code = self.dyn_clone_wrap(code, ty);
                // A dynamic variable accepts any D1 scalar value,
                // boxed on the way in. The reverse (a statically
                // inferred variable receiving a dynamic value) stays
                // a loud error in the match below: psa pins one type
                // per variable from its first binding.
                if ty != Ty::Dyn && ctx.lookup(name) == Some(Ty::Dyn) {
                    code = self.box_code(code, ty, *line)?;
                    ty = Ty::Dyn;
                }
                // Container assignment moves the source binding (spec s3).
                let src_name = match value {
                    Expr::Name(n, span) if ty.is_list() => Some((n.clone(), span.start)),
                    _ => None,
                };
                if let Some((src, _)) = &src_name {
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
                        if ty == Ty::Dyn {
                            // A dynamic binding is declared with the
                            // None literal (hoisted to the prologue
                            // when inside a block) and filled through
                            // the stage pattern: a direct `let x =
                            // <call>;` inside a block is never
                            // retired on this route (probes q16/q24).
                            if ctx.scopes.len() > 1 {
                                ctx.hoist_declare(name, ty);
                            } else {
                                ctx.declare(name, ty)
                                    .map_err(|m| eerr(*line, &m))?;
                                ctx.line(&format!(
                                    "let {}: {} = {};",
                                    name,
                                    ty.sla(),
                                    ty.default_lit()
                                ));
                            }
                            self.emit_dyn_store(ctx, name, &code)
                                .map_err(|m| eerr(*line, &m))?;
                        } else {
                            ctx.declare(name, ty)
                                .map_err(|m| eerr(*line, &m))?;
                            if matches!(value, Expr::Name(_, _))
                                && matches!(ty, Ty::Int | Ty::Float | Ty::Bool)
                            {
                                // A fresh `let q = p;` MOVES p on the
                                // SAB route even for scalars ((xxvii)(a)
                                // family: let-from-name consumes).
                                // Declare with the default, then assign:
                                // the assignment form reads p by copy.
                                ctx.line(&format!(
                                    "let {}: {} = {};",
                                    name,
                                    ty.sla(),
                                    ty.default_lit()
                                ));
                                ctx.line(&format!("{} = {};", name, code));
                            } else {
                                ctx.line(&format!(
                                    "let {}: {} = {};",
                                    name,
                                    ty.sla(),
                                    code
                                ));
                            }
                        }
                    }
                    Some(existing) => {
                        if existing == ty && existing == Ty::Dyn {
                            self.emit_dyn_store(ctx, name, &code)
                                .map_err(|m| eerr(*line, &m))?;
                            ctx.revive(name);
                        } else if existing == ty {
                            let nt = expr_mentions(value, name) && expr_has_call(value);
                            assign_line(ctx, name, &code, ty, false, nt)
                                .map_err(|m| eerr(*line, &m))?;
                            ctx.revive(name);
                        } else if existing == Ty::Float && ty == Ty::Int {
                            let nt = expr_mentions(value, name) && expr_has_call(value);
                            assign_line(ctx, name, &code, ty, true, nt)
                                .map_err(|m| eerr(*line, &m))?;
                        } else if ty == Ty::Dyn {
                            return Err(eerr(
                                *line,
                                &format!(
                                    "variable `{}` was inferred as {} from its first binding and cannot become dynamic later (psa pins one type per variable; make the first binding dynamic, e.g. through a conditional expression, or use a new name)",
                                    name,
                                    existing.py_name()
                                ),
                            ));
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
                if let Some((src, src_anchor)) = src_name {
                    if src != *name {
                        ctx.mark_moved(&src, src_anchor);
                    }
                }
                Ok(())
            }
            Stmt::Assert { cond, line, anchor: _, msg } => {
                let (code, ty) = self.emit_expr(ctx, cond, *line)?;
                let code = match ty {
                    Ty::Bool => code,
                    Ty::Dyn => self.unbox_code(code, Ty::Bool, *line)?,
                    _ => {
                        return Err(eerr(
                            *line,
                            "assert condition must be a bool (or a dynamic value)",
                        ))
                    }
                };
                match msg {
                    // The label reaches the failure site at last
                    // (census section 0.6): written to stderr just
                    // before the panic, the argparse 77161/77162
                    // precedent. Assert without a message emits
                    // exactly what it always did.
                    Some(label) => {
                        // eprintln does NOT exist at the SLA layer
                        // (probe 2026-11-26 on the box: `Undefined
                        // call: eprintln` at check time). Write the
                        // label to stderr inline via sa_io_stderr +
                        // sa_io_write_all (argparse 77161/77162
                        // precedent); sa_std/io.sai arrives through
                        // the generated import block's
                        // sla_std/wrap/io.sla, so no extra import.
                        // The trailing newline is baked into the
                        // literal — eprintln would have added one.
                        let text = format!("AssertionError: {}\n", label);
                        let esc = escape_sla_str(&text);
                        ctx.line(&format!(
                            "if !({}) {{ let __psa_assert_h = sa_io_stderr(); let __psa_assert_rc: i32 = sa_io_write_all(__psa_assert_h, STR_PTR(\"{}\"), STR_LEN(\"{}\")); panic(77040); }};",
                            code, esc, esc
                        ));
                    }
                    None => ctx.line(&format!("if !({}) {{ panic(77040); }};", code)),
                }
                Ok(())
            }
            Stmt::Del { targets, line, anchor } => {
                for (name, index) in targets {
                    if let Some(idx) = index {
                        // del dict[key] (D3 ordered dict): dynamic
                        // values only -- the runtime checks the tag
                        // (a dict removes the key, anything else
                        // panics 77052; a missing key panics 77054,
                        // Python's KeyError). Static containers stay
                        // refused: they have no SLA-source shrink
                        // shape yet.
                        let ty = ctx.lookup(name).ok_or_else(|| {
                            eerr(
                                *line,
                                &format!("del of undefined variable `{}` (Python raises NameError)", name),
                            )
                        })?;
                        if ty != Ty::Dyn {
                            return Err(eerr(
                                *line,
                                "del name[index] is only supported on dynamic dicts in D3: static containers have no SLA-source shrink shape yet",
                            ));
                        }
                        ctx.check_alive(name, *line, Some(*anchor))?;
                        let (mut kcode, kty) = if let Expr::Str(s) = idx {
                            (self.emit_str_temp(ctx, s, *line)?, Ty::Dyn)
                        } else {
                            self.emit_expr(ctx, idx, *line)?
                        };
                        if kty != Ty::Dyn {
                            kcode = self.box_code(kcode, kty, *line)?;
                        }
                        ctx.line(&format!("__psa_dict_del({}, {});", name, kcode));
                        continue;
                    }
                    if ctx.lookup(name).is_none() {
                        return Err(eerr(
                            *line,
                            &format!("del of undefined variable `{}` (Python raises NameError)", name),
                        ));
                    }
                    ctx.check_alive(name, *line, Some(*anchor))?;
                    if !ctx.deleted.iter().any(|(n, _)| n == name) {
                        ctx.deleted.push((name.clone(), *line));
                    }
                }
                Ok(())
            }
            Stmt::MultiAssign { names, values, line, anchor} => {
                // Evaluate every RHS value into its own temp first
                // (Python evaluates the whole RHS before binding, so
                // `a, b = b, a` swaps correctly).
                let mut temps: Vec<String> = Vec::new();
                for v in values {
                    ctx.tmp += 1;
                    let tname = format!("__psa_ma_{}", ctx.tmp);
                    self.emit_stmt(
                        ctx,
                        &Stmt::Assign { name: tname.clone(), value: v.clone(), line: *line, anchor: *anchor},
                    )?;
                    temps.push(tname);
                }
                if values.len() == 1 && names.len() > 1 {
                    // Chained: every target binds the same value. For a
                    // list that would alias in Python; psa ownership
                    // cannot alias, so refuse loudly.
                    if let Some(t) = ctx.lookup(&temps[0]) {
                        if t.is_list() {
                            return Err(eerr(
                                *line,
                                "chained assignment of a list would alias it in Python; psa ownership cannot alias a container -- bind the targets separately",
                            ));
                        }
                    }
                    for n in names {
                        self.emit_stmt(
                            ctx,
                            &Stmt::Assign {
                                name: n.clone(),
                                value: Expr::Name(temps[0].clone(), ExprSpan::synthetic()),
                                line: *line, anchor: *anchor,
                            },
                        )?;
                    }
                } else if values.len() == names.len() {
                    for (n, t) in names.iter().zip(temps.iter()) {
                        self.emit_stmt(
                            ctx,
                            &Stmt::Assign {
                                name: n.clone(),
                                value: Expr::Name(t.clone(), ExprSpan::synthetic()),
                                line: *line, anchor: *anchor,
                            },
                        )?;
                    }
                } else {
                    return Err(eerr(
                        *line,
                        &format!(
                            "cannot assign {} value(s) to {} name(s)",
                            values.len(),
                            names.len()
                        ),
                    ));
                }
                Ok(())
            }
            Stmt::Unpack { names, value, line, anchor} => {
                // A literal sequence of exactly matching length
                // unpacks pairwise, like a multi-assignment (the
                // RHS is fully evaluated into temps first, so
                // `a, b = b, a` swaps).
                if let Expr::ListLit(elems) | Expr::TupleLit(elems) = value {
                    if elems.len() != names.len() {
                        return Err(eerr(
                            *line,
                            &format!(
                                "cannot unpack {} value(s) into {} name(s)",
                                elems.len(),
                                names.len()
                            ),
                        ));
                    }
                    return self.emit_stmt(
                        ctx,
                        &Stmt::MultiAssign {
                            names: names.clone(),
                            values: elems.clone(),
                            line: *line, anchor: *anchor,
                        },
                    );
                }
                // A static list variable unpacks by index, with a
                // runtime length check (panic 77077 on mismatch).
                if let Expr::Name(n, span) = value {
                    if let Some(t) = ctx.lookup(n) {
                        if t != Ty::Dyn && t.is_list() {
                            ctx.check_alive(n, *line, Some(span.start))?;
                            ctx.line(&format!(
                                "if (len({}) as i64) != {} {{ panic(77077); }};",
                                n,
                                names.len()
                            ));
                            for (j, name) in names.iter().enumerate() {
                                self.emit_stmt(
                                    ctx,
                                    &Stmt::Assign {
                                        name: name.clone(),
                                        value: Expr::Index(
                                            Box::new(Expr::Name(n.clone(), ExprSpan::synthetic())),
                                            Box::new(Expr::Int(j as i64)),
                                        ExprSpan::synthetic()),
                                        line: *line, anchor: *anchor,
                                    },
                                )?;
                            }
                            return Ok(());
                        }
                    }
                }
                // The general path: any dynamic sequence, checked
                // and indexed through the iteration protocol.
                let code = self.emit_dyn_value(ctx, value, *line)?;
                let code = self.dyn_clone_wrap(code, Ty::Dyn);
                ctx.tmp += 1;
                let tmp = format!("__psa_upk_{}", ctx.tmp);
                ctx.line(&format!("let {} = {};", tmp, code));
                ctx.line(&format!("__psa_unpack_check({}, {});", tmp, names.len()));
                for (j, name) in names.iter().enumerate() {
                    let el = format!("__psa_iter_get({}, {})", tmp, j);
                    self.emit_bind_dyn_code(ctx, name, &el, *line)?;
                }
                Ok(())
            }
            Stmt::AttrAssign { base, name, op, value, line, anchor: _} => {
                // Class-side attribute assignment writes the
                // class's own runtime-registry dict (CPython:
                // visible to the class, its subclasses and
                // every instance that does not shadow it).
                if let Expr::Name(n, _) = base {
                    if ctx.lookup(n).is_none() {
                        if let Some(&cid) = self.class_ids.get(n) {
                            self.used_dyn = true;
                            let ntmp = self.emit_str_temp(ctx, name, *line)?;
                            let (mut vcode, vty) = if let Expr::Str(s) = value {
                                (self.emit_str_temp(ctx, s, *line)?, Ty::Dyn)
                            } else if matches!(value, Expr::ListLit(_) | Expr::DictLit(_)) {
                                (self.emit_container_temp(ctx, value, *line)?, Ty::Dyn)
                            } else {
                                self.emit_expr(ctx, value, *line)?
                            };
                            if vty != Ty::Dyn {
                                vcode = self.box_code(vcode, vty, *line)?;
                            }
                            let d = format!("__psa_class_attrs_raw({})", cid);
                            match op {
                                None => ctx.line(&format!(
                                    "__psa_dict_set({}, {}, {});",
                                    d, ntmp, vcode
                                )),
                                Some(bop) => {
                                    let place =
                                        format!("__psa_class_{}_cgetattr({})", cid, ntmp);
                                    let (code, _) = self.emit_binop(
                                        place,
                                        Ty::Dyn,
                                        *bop,
                                        vcode,
                                        Ty::Dyn,
                                        *line,
                                    )?;
                                    ctx.line(&format!(
                                        "__psa_dict_set({}, {}, {});",
                                        d, ntmp, code
                                    ));
                                }
                            }
                            return Ok(());
                        }
                    }
                }
                let (bcode, bty) = self.emit_expr(ctx, base, *line)?;
                if bty != Ty::Dyn {
                    return Err(eerr(
                        *line,
                        &format!(
                            "attribute assignment on a {} value is not supported yet (objects are dynamic values in D4)",
                            bty.py_name()
                        ),
                    ));
                }
                self.used_dyn = true;
                let ntmp = self.emit_str_temp(ctx, name, *line)?;
                let (mut vcode, vty) = if let Expr::Str(s) = value {
                    (self.emit_str_temp(ctx, s, *line)?, Ty::Dyn)
                } else if matches!(value, Expr::ListLit(_) | Expr::DictLit(_)) {
                    (self.emit_container_temp(ctx, value, *line)?, Ty::Dyn)
                } else {
                    self.emit_expr(ctx, value, *line)?
                };
                if vty != Ty::Dyn {
                    vcode = self.box_code(vcode, vty, *line)?;
                }
                match op {
                    None => ctx.line(&format!("__psa_setattr({}, {}, {});", bcode, ntmp, vcode)),
                    Some(bop) => {
                        let place = format!("__psa_getattr({}, {})", bcode, ntmp);
                        let (code, _) =
                            self.emit_binop(place, Ty::Dyn, *bop, vcode, Ty::Dyn, *line)?;
                        ctx.line(&format!("__psa_setattr({}, {}, {});", bcode, ntmp, code));
                    }
                }
                Ok(())
            }
            Stmt::AttrDel { base, name, line, anchor: _} => {
                // Class-side deletion removes the name from
                // the class's OWN registry dict only (CPython);
                // a name that lives on an ancestor is not
                // deleted through the subclass.
                if let Expr::Name(n, _) = base {
                    if ctx.lookup(n).is_none() {
                        if let Some(&cid) = self.class_ids.get(n) {
                            self.used_dyn = true;
                            let ntmp = self.emit_str_temp(ctx, name, *line)?;
                            let d = format!("__dcl_{}", ctx.tmp + 1);
                            ctx.tmp += 1;
                            ctx.line(&format!(
                                "let {} = __psa_class_attrs_raw({});",
                                d, cid
                            ));
                            ctx.line(&format!(
                                "if __psa_dict_find({}, {}) < 0 {{ panic(77069); }};",
                                d, ntmp
                            ));
                            ctx.line(&format!("__psa_dict_del({}, {});", d, ntmp));
                            return Ok(());
                        }
                    }
                }
                let (bcode, bty) = self.emit_expr(ctx, base, *line)?;
                if bty != Ty::Dyn {
                    return Err(eerr(
                        *line,
                        "attribute deletion is only supported on dynamic (object) values in D4",
                    ));
                }
                self.used_dyn = true;
                let ntmp = self.emit_str_temp(ctx, name, *line)?;
                ctx.line(&format!("__psa_delattr({}, {});", bcode, ntmp));
                Ok(())
            }
            Stmt::IndexAssign { name, index, op, value, line, anchor } => {
                let ty = ctx
                    .lookup(name)
                    .ok_or_else(|| eerr(*line, &format!("index assignment to undefined variable `{}`", name)))?;
                ctx.check_alive(name, *line, Some(*anchor))?;
                if ty == Ty::Dyn {
                    self.used_dyn = true;
                    let (icode, ity) = if let Expr::Str(s) = index {
                        (self.emit_str_temp(ctx, s, *line)?, Ty::Dyn)
                    } else {
                        self.emit_expr(ctx, index, *line)?
                    };
                    let kcode = match ity {
                        Ty::Dyn => icode,
                        Ty::Int | Ty::Bool | Ty::Float => self.box_code(icode, ity, *line)?,
                        _ => {
                            return Err(eerr(
                                *line,
                                "a dynamic subscript key must be an int, float, bool, None, str or dynamic value",
                            ))
                        }
                    };
                    let (mut vcode, vty) = if let Expr::Str(s) = value {
                        (self.emit_str_temp(ctx, s, *line)?, Ty::Dyn)
                    } else if matches!(value, Expr::ListLit(_) | Expr::DictLit(_)) {
                        (self.emit_container_temp(ctx, value, *line)?, Ty::Dyn)
                    } else {
                        self.emit_expr(ctx, value, *line)?
                    };
                    if vty != Ty::Dyn {
                        vcode = self.box_code(vcode, vty, *line)?;
                    }
                    match op {
                        None => ctx.line(&format!("__psa_dyn_set({}, {}, {});", name, kcode, vcode)),
                        Some(bop) => {
                            let place = format!("__psa_dyn_get({}, {})", name, kcode);
                            let (code, _) = self.emit_binop(place, Ty::Dyn, *bop, vcode, Ty::Dyn, *line)?;
                            ctx.line(&format!("__psa_dyn_set({}, {}, {});", name, kcode, code));
                        }
                    }
                    return Ok(());
                }
                let (mut icode, ity) = self.emit_expr(ctx, index, *line)?;
                if ity == Ty::Dyn {
                    icode = self.unbox_code(icode, Ty::Int, *line)?;
                } else if ity != Ty::Int {
                    return Err(eerr(*line, "list index must be int"));
                }
                let elem = match ty.elem() {
                    Some(e) => e,
                    None => {
                        return Err(eerr(
                            *line,
                            &format!("index assignment is only supported on lists, not on {}", ty.py_name()),
                        ))
                    }
                };
                let (mut vcode, vty) = self.emit_expr(ctx, value, *line)?;
                if vty == Ty::Dyn {
                    vcode = self.unbox_code(vcode, elem, *line)?;
                } else if vty != elem {
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
            Stmt::AugAssign { name, op, value, line, anchor: _} => {
                let existing = ctx
                    .lookup(name)
                    .ok_or_else(|| eerr(*line, &format!("augmented assignment to undefined variable `{}`", name)))?;
                let fake = Expr::Bin(
                    Box::new(Expr::Name(name.clone(), ExprSpan::synthetic())),
                    *op,
                    Box::new(value.clone()),
                );
                let (code, ty) = self.emit_expr(ctx, &fake, *line)?;
                if existing == Ty::Dyn {
                    // The result is a runtime call that mentions the
                    // target; storing it goes through the stage
                    // pattern (a whole-value assignment would trap
                    // RegisterRedefinition, probes q5/q6).
                    self.emit_dyn_store(ctx, name, &code)
                        .map_err(|m| eerr(*line, &m))?;
                    return Ok(());
                }
                // A dynamic target always routes through a temp: the
                // result is a runtime call that mentions the target,
                // which is exactly the register-leak shape of defect
                // (xxvi) when assigned directly.
                let needs_temp = matches!(op, BinOp::FloorDiv | BinOp::Mod | BinOp::Pow)
                    || expr_has_call(value)
                    || existing == Ty::Dyn;
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
            Stmt::ExprStmt { value, line, anchor: _} => {
                // print(...) is a builtin statement-level call.
                if let Expr::Call(callee, args, kwargs, _) = value {
                    if let Expr::Name(n, _) = &**callee {
                        if n == "print" {
                            // D5: sep=/end= are accepted as string
                            // literals and route to the general
                            // byte-print path; any other keyword
                            // (file=, flush=) stays refused.
                            let mut sep = " ".to_string();
                            let mut end = "\n".to_string();
                            let mut custom = false;
                            for (k, v) in kwargs {
                                match (k.as_str(), v) {
                                    ("sep", Expr::Str(s)) => {
                                        sep = s.clone();
                                        custom = true;
                                    }
                                    ("end", Expr::Str(s)) => {
                                        end = s.clone();
                                        custom = true;
                                    }
                                    ("sep", _) | ("end", _) => {
                                        return Err(eerr(
                                            *line,
                                            &format!(
                                                "print() {}= must be a string literal in v1",
                                                k
                                            ),
                                        ));
                                    }
                                    _ => {
                                        return Err(eerr(
                                            *line,
                                            &format!(
                                                "print() keyword argument `{}` is not supported in v1 (only sep= and end=)",
                                                k
                                            ),
                                        ));
                                    }
                                }
                            }
                            if custom {
                                return self.emit_print_general(ctx, args, &sep, &end, *line);
                            }
                            return self.emit_print(ctx, args, *line);
                        }
                    }
                }
                let (code, _ty) = self.emit_expr(ctx, value, *line)?;
                if matches!(value, Expr::Walrus { .. }) {
                    // A statement-level walrus: emit_expr has
                    // already emitted the binding (the whole
                    // effect); the yielded value is discarded,
                    // as in CPython. Emitting `{name};` here
                    // would move the freshly bound variable —
                    // the host checker rejects the next read
                    // with UseAfterMove (w04).
                    return Ok(());
                }
                ctx.line(&format!("{};", code));
                Ok(())
            }
            Stmt::If { arms, els, line, anchor: _} => {
                for (i, (cond, body)) in arms.iter().enumerate() {
                    let (ccode, cty) = self.emit_expr(ctx, cond, *line)?;
                    let ccode = self.cond_code(ccode, cty, *line, "if condition must be bool")?;
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
            Stmt::While { cond, body, els, line, anchor: _} => {
                // A walrus as the WHOLE while condition is
                // restructured: the value must be re-evaluated
                // (and the name re-bound) on every iteration, so
                // the binding becomes the loop head's first
                // statement under `while true`, followed by a
                // negated-condition break. That break is a plain
                // text line, never emit_break_through, so it does
                // not set the loop-else break flag.
                if let Expr::Walrus { value, .. } = cond {
                    if expr_contains_walrus(value) {
                        return Err(eerr(
                            *line,
                            "a walrus nested inside a while-condition walrus value is not supported in v1; bind the inner value before the loop",
                        ));
                    }
                    let flag = self.loop_else_begin(ctx, els);
                    ctx.line("while true {");
                    ctx.indent += 1;
                    ctx.loop_depth += 1;
                    let (vcode, vty) = self.emit_expr(ctx, cond, *line)?;
                    let ccode =
                        self.cond_code(vcode, vty, *line, "while condition must be bool")?;
                    ctx.line(&format!("if !({}) {{ break; }};", ccode));
                    let r = self.emit_block(ctx, body);
                    ctx.loop_depth -= 1;
                    ctx.indent -= 1;
                    r?;
                    ctx.line("}");
                    self.loop_else_stack.pop();
                    if let (Some(els_body), Some(flag)) = (els, &flag) {
                        self.emit_loop_else_body(ctx, els_body, flag)?;
                    }
                    return Ok(());
                }
                if expr_contains_walrus(cond) {
                    return Err(eerr(
                        *line,
                        "a walrus inside a while condition is only supported as the whole condition in v1 (`while (name := value):`); a nested walrus would be bound once before the loop instead of on every iteration",
                    ));
                }
                let flag = self.loop_else_begin(ctx, els);
                let saved = ctx.in_while_cond;
                ctx.in_while_cond = true;
                let res = self.emit_expr(ctx, cond, *line);
                ctx.in_while_cond = saved;
                let (ccode, cty) = res?;
                let ccode = self.cond_code(ccode, cty, *line, "while condition must be bool")?;
                ctx.line(&format!("while {} {{", ccode));
                ctx.indent += 1;
                ctx.loop_depth += 1;
                let r = self.emit_block(ctx, body);
                ctx.loop_depth -= 1;
                ctx.indent -= 1;
                r?;
                ctx.line("}");
                self.loop_else_stack.pop();
                if let (Some(els_body), Some(flag)) = (els, &flag) {
                    self.emit_loop_else_body(ctx, els_body, flag)?;
                }
                Ok(())
            }
            Stmt::ForRange { var, start, stop, body, els, line, anchor: _} => {
                let (mut stop_code, stop_ty) = self.emit_expr(ctx, stop, *line)?;
                if stop_ty == Ty::Dyn {
                    stop_code = self.unbox_code(stop_code, Ty::Int, *line)?;
                } else if stop_ty != Ty::Int {
                    return Err(eerr(*line, "range() bounds must be int"));
                }
                let (mut start_code, start_ty) = match start {
                    Some(e) => self.emit_expr(ctx, e, *line)?,
                    None => ("0".to_string(), Ty::Int),
                };
                if start_ty == Ty::Dyn {
                    start_code = self.unbox_code(start_code, Ty::Int, *line)?;
                } else if start_ty != Ty::Int {
                    return Err(eerr(*line, "range() bounds must be int"));
                }
                ctx.line("{");
                ctx.indent += 1;
                ctx.scopes.push(ctx.vars.len());
                ctx.declare(var, Ty::Int).map_err(|m| eerr(*line, &m))?;
                // Materialise the stop value once (Python evaluates the
                // range bound once; a bare expression in the while
                // condition would re-evaluate every iteration).
                ctx.tmp += 1;
                let stop_var = format!("__psa_stop_{}_{}", line, ctx.tmp);
                ctx.line(&format!("let {}: i64 = {};", stop_var, stop_code));
                ctx.line(&format!("let {}: i64 = {};", var, start_code));
                // The range is driven by a separate counter and
                // the loop variable is (re)bound at the top of
                // each iteration, as CPython's range iterator
                // does: after the loop the variable holds the
                // last iterated value (le05's else clause reads
                // it), a body write to the variable cannot
                // perturb the iteration, and a break leaves the
                // current value in place. (continue is refused
                // in for-range loops, PSA-P023, so the counter
                // increment at the body end cannot be skipped.)
                ctx.tmp += 1;
                let ctr = format!("__psa_ctr_{}_{}", line, ctx.tmp);
                ctx.line(&format!("let {}: i64 = {};", ctr, var));
                let flag = self.loop_else_begin(ctx, els);
                ctx.line(&format!("while {} < {} {{", ctr, stop_var));
                ctx.indent += 1;
                ctx.line(&format!("{} = {};", var, ctr));
                // Body statements share the loop scope (the loop variable
                // stays visible); nested blocks scope themselves.
                ctx.loop_depth += 1;
                for s in body {
                    self.emit_stmt(ctx, s)?;
                }
                ctx.loop_depth -= 1;
                ctx.line(&format!("{} = {} + 1;", ctr, ctr));
                ctx.indent -= 1;
                ctx.line("}");
                self.loop_else_stack.pop();
                if let (Some(els_body), Some(flag)) = (els, &flag) {
                    self.emit_loop_else_body(ctx, els_body, flag)?;
                }
                let mark = ctx.scopes.pop().unwrap();
                ctx.vars.truncate(mark);
                ctx.indent -= 1;
                ctx.line("}");
                Ok(())
            }
            Stmt::ForEach { vars, iter, body, els, line, anchor: _} => {
                let (obj, ty) = match iter {
                    Expr::Name(n, span) => {
                        let t = ctx
                            .lookup(n)
                            .ok_or_else(|| eerr(*line, &format!("undefined variable `{}`", n)))?;
                        ctx.check_alive(n, *line, Some(span.start))?;
                        (n.clone(), t)
                    }
                    _ => {
                        // A general iterable expression (D5) is
                        // materialised into a dynamic temp first;
                        // static containers reach here as dynamic
                        // temps through emit_dyn_value.
                        let code = self.emit_dyn_value(ctx, iter, *line)?;
                        let code = self.dyn_clone_wrap(code, Ty::Dyn);
                        ctx.tmp += 1;
                        let tname = format!("__psa_fi_{}", ctx.tmp);
                        ctx.line(&format!("let {} = {};", tname, code));
                        (tname, Ty::Dyn)
                    }
                };
                // D7: a dynamic iterable may be a generator;
                // drain it into a list first (the loop below
                // iterates the list natively). Draining is full
                // even when the body breaks out early -- a
                // registered divergence from CPython's lazy
                // iteration (D7-NOTES).
                let (obj, ty) = if ty == Ty::Dyn && self.used_gen {
                    ctx.tmp += 1;
                    let tname = format!("__psa_fi_{}", ctx.tmp);
                    ctx.line(&format!("let {} = __psa_iter_source({});", tname, obj));
                    (tname, ty)
                } else {
                    (obj, ty)
                };
                let multi = vars.len() > 1;
                if multi && ty != Ty::Dyn {
                    return Err(eerr(
                        *line,
                        "unpacking loop targets (`for a, b in ...`) need a dynamic iterable in v1 (bind the sequence to a dynamic value first)",
                    ));
                }
                let (len_code, elem_ty, get_code) = if ty == Ty::Dyn {
                    self.used_dyn = true;
                    (
                        format!("__psa_dyn_len({})", obj),
                        Ty::Dyn,
                        format!("__psa_iter_get({}, {{}})", obj),
                    )
                } else if let Some(e) = ty.elem() {
                    (
                        format!("(len({}) as i64)", obj),
                        e,
                        format!("{}[{{}}]", obj),
                    )
                } else {
                    return Err(eerr(
                        *line,
                        &format!("for-each iteration is only supported over lists, not on {}", ty.py_name()),
                    ));
                };
                // The index temp must be unique per loop, not
                // per source line: comprehension/genexp
                // desugars put several synthetic loops on the
                // host line, and same-named nested indices
                // collide in the flattener (genexp1's
                // UseAfterMove on the outer increment).
                ctx.tmp += 1;
                let idx = format!("__psa_i_{}_{}", line, ctx.tmp);
                ctx.line("{");
                ctx.indent += 1;
                ctx.scopes.push(ctx.vars.len());
                if !multi {
                    ctx.declare(&vars[0], elem_ty).map_err(|m| eerr(*line, &m))?;
                }
                ctx.line(&format!("let {}: i64 = 0;", idx));
                if !multi {
                    // The loop variable is declared HERE, in
                    // the wrapping block — not by the per-
                    // iteration `let` inside the while body,
                    // whose SLA binding would die at the
                    // while's closing brace and leave the loop-
                    // else clause (and any post-loop read in
                    // this block) facing an undefined name
                    // (le23). The loop body assigns it.
                    ctx.line(&format!(
                        "let {}: {} = {};",
                        vars[0],
                        elem_ty.sla(),
                        elem_ty.default_lit()
                    ));
                }
                let flag = self.loop_else_begin(ctx, els);
                // The length is re-read every iteration: CPython's
                // list iterator observes appends made by the body.
                ctx.line(&format!("while {} < {} {{", idx, len_code));
                ctx.indent += 1;
                if multi {
                    // Each element is unpacked through the
                    // iteration protocol; the targets bind as
                    // dynamic values (hoisted declarations, so
                    // they stay visible after the loop, as in
                    // Python).
                    ctx.tmp += 1;
                    let el = format!("__psa_el_{}_{}", line, ctx.tmp);
                    ctx.line(&format!("let {} = {};", el, get_code.replace("{}", &idx)));
                    ctx.line(&format!("__psa_unpack_check({}, {});", el, vars.len()));
                    for (j, v) in vars.iter().enumerate() {
                        let ec = format!("__psa_iter_get({}, {})", el, j);
                        self.emit_bind_dyn_code(ctx, v, &ec, *line)?;
                    }
                } else {
                    ctx.line(&format!("{} = {};", vars[0], get_code.replace("{}", &idx)));
                }
                // The index advances immediately after the
                // bind, before the user body: a body ending in
                // `break` (or return) would otherwise leave the
                // increment as dead text after the terminator,
                // which the host rejects with
                // FallthroughForbidden (le10). The counter is
                // synthetic (invisible to the body), and
                // continue is refused in for-each loops
                // (PSA-P019), so no path can skip or observe
                // the difference.
                ctx.line(&format!("{} = {} + 1;", idx, idx));
                ctx.loop_depth += 1;
                for s in body {
                    self.emit_stmt(ctx, s)?;
                }
                ctx.loop_depth -= 1;
                ctx.indent -= 1;
                ctx.line("}");
                self.loop_else_stack.pop();
                if let (Some(els_body), Some(flag)) = (els, &flag) {
                    self.emit_loop_else_body(ctx, els_body, flag)?;
                }
                let mark = ctx.scopes.pop().unwrap();
                ctx.vars.truncate(mark);
                ctx.indent -= 1;
                ctx.line("}");
                Ok(())
            }
            Stmt::Try { body, handlers, els, finally, line, anchor: _} => {
                validate_try(handlers, finally, *line)?;
                self.begin_try(ctx);
                // The body runs inside a single-iteration guard
                // wrapper; the per-statement guards (emit_stmt)
                // break out of it the moment an exception is
                // pending or a control flag is set.
                ctx.line("while true {");
                ctx.indent += 1;
                ctx.guard_depth += 1;
                let r = self.emit_block(ctx, body);
                ctx.guard_depth -= 1;
                r?;
                // A body whose last top-level statement is a
                // return / break / continue already terminates
                // the guard wrapper with its own `break;` (the
                // deferred-control emission); a second trailing
                // `break;` is dead text after a terminator and
                // the host rejects it with
                // FallthroughForbidden (le12; pre-existing —
                // any try body ending in return hit it).
                let body_terminates = matches!(
                    body.last(),
                    Some(
                        Stmt::Return { .. }
                            | Stmt::Break { .. }
                            | Stmt::Continue { .. }
                    )
                );
                if !body_terminates {
                    ctx.line("break;");
                }
                ctx.indent -= 1;
                ctx.line("}");
                // Handler dispatch (this try's frame is still on
                // the stack, so control flow in a handler is
                // deferred until after the finally below).
                let mut handled_name: Option<String> = None;
                if !handlers.is_empty() {
                    ctx.tmp += 1;
                    let handled = format!("__psa_handled_{}", ctx.tmp);
                    handled_name = Some(handled.clone());
                    ctx.line(&format!("let {}: bool = false;", handled));
                    for h in handlers {
                        let cond = self.handler_cond(h)?;
                        // Nested ifs, never one && chain: SLA
                        // evaluates both sides of && eagerly,
                        // and the class condition dereferences
                        // the pending value — it may only run
                        // when an exception is actually
                        // pending (exc1 batch 8 segfault).
                        ctx.line("if __psa_exc_tag() != 3 {");
                        ctx.indent += 1;
                        ctx.line(&format!("if !{} {{", handled));
                        ctx.indent += 1;
                        ctx.line(&format!("if {} {{", cond));
                        ctx.indent += 1;
                        ctx.line(&format!("{} = true;", handled));
                        ctx.tmp += 1;
                        let cur = format!("__psa_excur_{}", ctx.tmp);
                        ctx.line(&format!("let {} = __psa_exc_get();", cur));
                        if let Some(bind) = &h.bind {
                            // The handler variable gets a fresh
                            // textual let inside this handler
                            // block. Routing through
                            // emit_bind_dyn_code breaks on the
                            // second handler binding the same
                            // name: Ctx still carries the first
                            // handler's registration (function
                            // level), so it takes the store path
                            // and writes to a variable whose
                            // textual let sits in a sibling
                            // block (exc1's UndefinedVariable
                            // `e`). CPython deletes the name at
                            // handler exit anyway, so a
                            // block-local binding is faithful.
                            match ctx.lookup(bind) {
                                None => {
                                    ctx.declare(bind, Ty::Dyn)
                                        .map_err(|m| eerr(h.line, &m))?;
                                }
                                Some(Ty::Dyn) => {}
                                Some(existing) => {
                                    return Err(eerr(
                                        h.line,
                                        &format!(
                                            "variable `{}` was inferred as {} from its first binding and cannot receive the exception value (psa pins one type per variable)",
                                            bind,
                                            existing.py_name()
                                        ),
                                    ));
                                }
                            }
                            ctx.clear_deleted(bind);
                            ctx.line(&format!(
                                "let {}: PsaVal = {};",
                                bind,
                                Ty::Dyn.default_lit()
                            ));
                            self.emit_dyn_store(ctx, bind, &cur)
                                .map_err(|m| eerr(h.line, &m))?;
                            ctx.revive(bind);
                        }
                        ctx.line(
                            "__psa_exc_store(PsaVal { tag: 3, ipay: 0, fpay: 0.0 });",
                        );
                        let saved_bare =
                            std::mem::replace(&mut ctx.exc_bare, Some(cur.clone()));
                        ctx.line("while true {");
                        ctx.indent += 1;
                        ctx.guard_depth += 1;
                        let r = self.emit_block(ctx, &h.body);
                        ctx.guard_depth -= 1;
                        ctx.exc_bare = saved_bare;
                        r?;
                        ctx.line("break;");
                        ctx.indent -= 1;
                        ctx.line("}");
                        ctx.indent -= 1;
                        ctx.line("};");
                        ctx.indent -= 1;
                        ctx.line("};");
                        ctx.indent -= 1;
                        ctx.line("};");
                    }
                }
                // The else clause runs only on a clean body exit
                // (no pending exception, no control flag, and no
                // handler ran — a handled exception clears the
                // slot, so the handled flag is part of the test).
                if let Some(els) = els {
                    let (rf, bf, cf) = {
                        let f = ctx.try_stack.last().expect("try frame");
                        (f.ret_flag.clone(), f.brk_flag.clone(), f.cont_flag.clone())
                    };
                    match &handled_name {
                        Some(h) => ctx.line(&format!(
                            "if __psa_exc_tag() == 3 && !{} && !{} && !{} && !{} {{",
                            h, rf, bf, cf
                        )),
                        None => ctx.line(&format!(
                            "if __psa_exc_tag() == 3 && !{} && !{} && !{} {{",
                            rf, bf, cf
                        )),
                    }
                    ctx.indent += 1;
                    ctx.line("while true {");
                    ctx.indent += 1;
                    ctx.guard_depth += 1;
                    let r = self.emit_block(ctx, els);
                    ctx.guard_depth -= 1;
                    r?;
                    ctx.line("break;");
                    ctx.indent -= 1;
                    ctx.line("}");
                    ctx.indent -= 1;
                    ctx.line("};");
                }
                let frame = ctx.try_stack.pop().expect("try frame");
                if let Some(fin) = finally {
                    ctx.in_finally += 1;
                    let r = self.emit_block(ctx, fin);
                    ctx.in_finally -= 1;
                    r?;
                }
                self.emit_try_epilogue(ctx, frame, *line)?;
                Ok(())
            }
            Stmt::Raise { value, line, anchor: _} => {
                self.used_dyn = true;
                self.ensure_exc_slot(ctx);
                match value {
                    None => {
                        let cur = match &ctx.exc_bare {
                            Some(c) => c.clone(),
                            None => {
                                return Err(eerr(
                                    *line,
                                    "a bare `raise` is only allowed inside an except handler (D6)",
                                ));
                            }
                        };
                        ctx.line(&format!("__psa_exc_store({});", cur));
                    }
                    Some(e) => {
                        // `raise SomeClass` instantiates the
                        // class with no arguments (CPython).
                        let mut done = false;
                        if let Expr::Name(n, _) = e {
                            if ctx.lookup(n).is_none() {
                                if let Some(&cid) = self.class_ids.get(n) {
                                    if !self.classes[cid].ancestors.contains(&0) {
                                        return Err(eerr(
                                            *line,
                                            &format!(
                                                "cannot raise class `{}`: it does not derive from Exception",
                                                n
                                            ),
                                        ));
                                    }
                                    ctx.tmp += 1;
                                    let t = format!("__psa_raise_{}", ctx.tmp);
                                    ctx.line(&format!(
                                        "let {} = __psa_exc_new({}, {});",
                                        t,
                                        cid,
                                        Ty::Dyn.default_lit()
                                    ));
                                    ctx.line(&format!("__psa_exc_store({});", t));
                                    done = true;
                                }
                            }
                        }
                        if !done {
                            let (code, ty) = self.emit_expr(ctx, e, *line)?;
                            if ty != Ty::Dyn {
                                return Err(eerr(
                                    *line,
                                    "raise needs an exception instance (a dynamic value); statically-typed values cannot be raised in v1",
                                ));
                            }
                            ctx.tmp += 1;
                            let t = format!("__psa_raise_{}", ctx.tmp);
                            ctx.line(&format!("let {} = {};", t, code));
                            // __psa_exc_check returns void and
                            // panics 77101 itself when the value
                            // is not an exception instance; an
                            // earlier shape negated its result
                            // (`if !(__psa_exc_check(..))`), which
                            // the SLA checker rejects as a binary
                            // expr type mismatch (exc1 bisect).
                            ctx.line(&format!("__psa_exc_check({});", t));
                            ctx.line(&format!("__psa_exc_store({});", t));
                        }
                    }
                }
                // With no enclosing try in this function, the
                // exception propagates to the caller through
                // the global pending slot: unwind this frame
                // now (in main it is uncaught). Inside a try,
                // the trailing guard breaks the wrapper and
                // the handler dispatch takes it from there.
                if ctx.try_stack.is_empty() {
                    let act = self.exc_unwind_action(ctx);
                    ctx.line(&act);
                }
                Ok(())
            }
            Stmt::Yield { value, line, anchor: _} => {
                // D7: only reachable inside a generator producer
                // (Ctx::gen_chan set); the def-level pre-pass
                // guarantees every def containing a yield is
                // emitted as a producer, so a missing channel
                // here means a yield leaked into a context the
                // pre-pass does not own (loud).
                let ch = match &ctx.gen_chan {
                    Some(c) => c.clone(),
                    None => {
                        return Err(eerr(
                            *line,
                            "yield outside a generator function (a def becomes a generator only when yield appears directly in its own body)",
                        ))
                    }
                };
                self.used_dyn = true;
                let vcode = match value {
                    None => Ty::Dyn.default_lit().to_string(),
                    Some(e) => self.emit_dyn_value(ctx, e, *line)?,
                };
                ctx.tmp += 1;
                let yw = format!("__psa_yw_{}", ctx.tmp);
                ctx.declare(&yw, Ty::Dyn).map_err(|m| eerr(*line, &m))?;
                ctx.line(&format!(
                    "let {}: PsaVal = __psa_gen_wrap({}, 0);",
                    yw, vcode
                ));
                ctx.line(&format!("{}.send({}.ipay);", ch, yw));
                Ok(())
            }
            Stmt::Match { subject, cases, line, anchor: _} => {
                self.emit_match(ctx, subject, cases, *line)
            }
            Stmt::With { items, body, line, anchor: _} => {
                // Items evaluate left to right and bind before
                // the guarded body; the finally-equivalent closes
                // them in reverse order on every exit path.
                let mut temps: Vec<String> = Vec::new();
                for (e, bind) in items {
                    let (code, ty) = self.emit_expr(ctx, e, *line)?;
                    if ty != Ty::Dyn {
                        return Err(eerr(
                            *line,
                            "with items must be dynamic values in v1 (file objects returned by open())",
                        ));
                    }
                    self.used_dyn = true;
                    self.used_file = true;
                    ctx.tmp += 1;
                    let t = format!("__psa_with_{}", ctx.tmp);
                    ctx.line(&format!("let {} = {};", t, code));
                    if let Some(b) = bind {
                        self.emit_bind_dyn_code(ctx, b, &t, *line)?;
                    }
                    temps.push(t);
                }
                self.begin_try(ctx);
                ctx.line("while true {");
                ctx.indent += 1;
                ctx.guard_depth += 1;
                let r = self.emit_block(ctx, body);
                ctx.guard_depth -= 1;
                r?;
                // Same trailing-break guard as the Try arm:
                // a body ending in return/break/continue has
                // already terminated the wrapper.
                let body_terminates = matches!(
                    body.last(),
                    Some(
                        Stmt::Return { .. }
                            | Stmt::Break { .. }
                            | Stmt::Continue { .. }
                    )
                );
                if !body_terminates {
                    ctx.line("break;");
                }
                ctx.indent -= 1;
                ctx.line("}");
                let frame = ctx.try_stack.pop().expect("try frame");
                for t in temps.iter().rev() {
                    ctx.line(&format!("__psa_with_close({});", t));
                }
                self.emit_try_epilogue(ctx, frame, *line)?;
                Ok(())
            }
        }
    }

    fn emit_print(&mut self, ctx: &mut Ctx, args: &[Expr], line: u32) -> Result<(), Diagnostic> {
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
        }
        // Single bool argument: Python prints True/False.
        if args.len() == 1 {
            let (code, ty) = self.emit_expr(ctx, &args[0], line)?;
            if ty == Ty::Dyn {
                self.used_dyn = true;
                ctx.line(&format!("__psa_print_dyn({});", code));
                return Ok(());
            }
            if ty == Ty::Float {
                return Err(eerr(
                    line,
                    "printing a float is blocked in v0.1: the Rust sla build-exe route truncates f64 in println (3.5 prints as 3.0000000000) while the value itself computes correctly; psa refuses to print a wrong value (upstream defect, see README)",
                ));
            }
            if ty.is_list() {
                // Static list repr, byte-printed elementwise through
                // the same helpers the dynamic repr uses (D2 slice C).
                let name = match &args[0] {
                    Expr::Name(n, _) => n.clone(),
                    // A slice temp is also a plain name.
                    _ if !code.is_empty()
                        && code
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == '_')
                        && !code.chars().next().unwrap().is_ascii_digit() =>
                    {
                        code.clone()
                    }
                    _ => {
                        return Err(eerr(
                            line,
                            "printing a list is only supported for a plain list variable in this slice",
                        ))
                    }
                };
                let elem = ty.elem().unwrap();
                if elem != Ty::Int && elem != Ty::Bool {
                    return Err(eerr(line, "static list elements must be int or bool to print"));
                }
                self.used_dyn = true;
                ctx.tmp += 1;
                let iv = format!("__psa_pi_{}", ctx.tmp);
                ctx.line("sla_print_bytes(STR_PTR(\"[\"), STR_LEN(\"[\"));");
                ctx.line(&format!("let {}: i64 = 0;", iv));
                ctx.line(&format!("while {} < (len({}) as i64) {{", iv, name));
                ctx.line(&format!(
                    "if {} > 0 {{ sla_print_bytes(STR_PTR(\", \"), STR_LEN(\", \")); }};",
                    iv
                ));
                if elem == Ty::Int {
                    ctx.line(&format!("__psa_print_i64({}[{}]);", name, iv));
                } else {
                    ctx.line(&format!(
                        "if {}[{}] {{ sla_print_bytes(STR_PTR(\"True\"), STR_LEN(\"True\")); }} else {{ sla_print_bytes(STR_PTR(\"False\"), STR_LEN(\"False\")); }};",
                        name, iv
                    ));
                }
                ctx.line(&format!("{} = {} + 1;", iv, iv));
                ctx.line("}");
                ctx.line("sla_print_bytes(STR_PTR(\"]\"), STR_LEN(\"]\"));");
                ctx.line("sla_print_bytes(STR_PTR(\"\\n\"), STR_LEN(\"\\n\"));");
                return Ok(());
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
        // Multiple arguments (D5): collect every argument once,
        // then either keep the historical single-println form
        // (only ints and string literals) or render through the
        // byte printers, which also covers dynamic values,
        // bools and f-strings.
        let mut items: Vec<(Option<String>, String, Ty)> = Vec::new();
        for a in args {
            if let Expr::Str(s) = a {
                items.push((Some(s.clone()), String::new(), Ty::Str));
            } else {
                let (code, ty) = self.emit_expr(ctx, a, line)?;
                if ty.is_list() {
                    return Err(eerr(
                        line,
                        "printing a whole static list in a multi-argument print is not supported (print it on its own line)",
                    ));
                }
                items.push((None, code, ty));
            }
        }
        let simple = items.iter().all(|(lit, _, ty)| lit.is_some() || *ty == Ty::Int);
        if simple {
            let mut fmt = String::new();
            let mut codes = Vec::new();
            for (i, (lit, code, _)) in items.iter().enumerate() {
                if i > 0 {
                    fmt.push(' ');
                }
                match lit {
                    Some(s) => {
                        if s.contains('{') || s.contains('}') {
                            return Err(eerr(
                                line,
                                "string literals containing braces are not supported in print in v0.1",
                            ));
                        }
                        fmt.push_str(&escape_sla_str(s));
                    }
                    None => {
                        fmt.push_str("{}");
                        codes.push(code.clone());
                    }
                }
            }
            if codes.is_empty() {
                ctx.line(&format!("println(\"{}\");", fmt));
            } else {
                ctx.line(&format!("println(\"{}\", {});", fmt, codes.join(", ")));
            }
            return Ok(());
        }
        self.emit_print_render(ctx, &items, " ", "\n", line)
    }

    /// Render collected print arguments through the byte
    /// printers with the given separator and terminator (D5).
    /// Items are (string-literal text, emitted code, type); a
    /// literal renders from its text, everything else from its
    /// code.
    fn emit_print_render(
        &mut self,
        ctx: &mut Ctx,
        items: &[(Option<String>, String, Ty)],
        sep: &str,
        end: &str,
        line: u32,
    ) -> Result<(), Diagnostic> {
        self.used_dyn = true;
        let sep_tmp = if items.len() > 1 && !sep.is_empty() {
            Some(self.emit_str_temp(ctx, sep, line)?)
        } else {
            None
        };
        for (i, (lit, code, ty)) in items.iter().enumerate() {
            if i > 0 {
                if let Some(st) = &sep_tmp {
                    ctx.line(&format!("__psa_print_str({});", st));
                }
            }
            if let Some(s) = lit {
                let t = self.emit_str_temp(ctx, s, line)?;
                ctx.line(&format!("__psa_print_str({});", t));
                continue;
            }
            match ty {
                Ty::Dyn => ctx.line(&format!("__psa_print_dyn_nl({});", code)),
                Ty::Int => ctx.line(&format!("__psa_print_i64({});", code)),
                Ty::Bool => ctx.line(&format!(
                    "if {} {{ sla_print_bytes(STR_PTR(\"True\"), STR_LEN(\"True\")); }} else {{ sla_print_bytes(STR_PTR(\"False\"), STR_LEN(\"False\")); }};",
                    code
                )),
                Ty::Float => {
                    return Err(eerr(
                        line,
                        "printing a float is blocked: f64 text is unavailable until the shortest-roundtrip formatter lands (upstream defect (xxiii) family)",
                    ))
                }
                _ => {
                    return Err(eerr(
                        line,
                        &format!(
                            "printing a {} value in this print form is not supported in v1",
                            ty.py_name()
                        ),
                    ))
                }
            }
        }
        if !end.is_empty() {
            let t = self.emit_str_temp(ctx, end, line)?;
            ctx.line(&format!("__psa_print_str({});", t));
        }
        Ok(())
    }

    /// print with sep=/end= (D5): collect the arguments once
    /// and render them through the byte printers.
    fn emit_print_general(
        &mut self,
        ctx: &mut Ctx,
        args: &[Expr],
        sep: &str,
        end: &str,
        line: u32,
    ) -> Result<(), Diagnostic> {
        let mut items: Vec<(Option<String>, String, Ty)> = Vec::new();
        for a in args {
            if let Expr::Str(s) = a {
                items.push((Some(s.clone()), String::new(), Ty::Str));
            } else {
                let (code, ty) = self.emit_expr(ctx, a, line)?;
                items.push((None, code, ty));
            }
        }
        self.emit_print_render(ctx, &items, sep, end, line)
    }

    /// print(f"...") — assemble println output from the f-string's parts.
    /// Emit an f-string as a dynamic string value; returns the
    /// temp holding the tag-6 PsaVal. Plain fields keep the
    /// historical behaviour exactly (including the float
    /// refusal, defect (xxiii) family). A field with a
    /// conversion, a debug marker or a format spec goes through
    /// the runtime formatter (`__psa_format_value`). Also used
    /// recursively to evaluate format specs (a spec is a mini
    /// f-string) and by the print route for f-strings that
    /// carry formatted fields.
    fn emit_fstr_value(
        &mut self,
        ctx: &mut Ctx,
        parts: &[FStrPart],
        line: u32,
    ) -> Result<String, Diagnostic> {
        self.used_dyn = true;
        ctx.tmp += 1;
        let acc = format!("__psa_ctmp_{}", ctx.tmp);
        ctx.line(&format!("let {} = __psa_str_new();", acc));
        for part in parts {
            match part {
                FStrPart::Lit(s) => {
                    if !s.is_empty() {
                        let t = self.emit_str_temp(ctx, s, line)?;
                        ctx.line(&format!(
                            "{} = __psa_str_append_str({}, {});",
                            acc, acc, t
                        ));
                    }
                }
                FStrPart::Field { value, conv, debug, spec } => {
                    if !part.is_formatted() {
                        if let Expr::Str(s) = value {
                            let t = self.emit_str_temp(ctx, s, line)?;
                            ctx.line(&format!(
                                "{} = __psa_str_append_str({}, {});",
                                acc, acc, t
                            ));
                            continue;
                        }
                        let (code, ty) = self.emit_expr(ctx, value, line)?;
                        let pcode = match ty {
                            Ty::Dyn => format!("__psa_str_of({})", code),
                            Ty::Int => format!("__psa_str_from_i64({})", code),
                            Ty::Bool => format!("__psa_str_from_bool({})", code),
                            Ty::Float => {
                                return Err(eerr(
                                    line,
                                    "interpolating a float in an f-string is blocked: f64 text is unavailable until the shortest-roundtrip formatter lands (upstream defect (xxiii) family)",
                                ))
                            }
                            _ => {
                                return Err(eerr(
                                    line,
                                    &format!(
                                        "values of type {} cannot be interpolated in an f-string yet",
                                        ty.py_name()
                                    ),
                                ))
                            }
                        };
                        ctx.line(&format!(
                            "{} = __psa_str_append_str({}, {});",
                            acc, acc, pcode
                        ));
                        continue;
                    }
                    // Formatted field: the debug prefix (verbatim
                    // literal text) comes first, then the value
                    // run through the runtime formatter.
                    if let Some(dbg) = debug {
                        let t = self.emit_str_temp(ctx, dbg, line)?;
                        ctx.line(&format!(
                            "{} = __psa_str_append_str({}, {});",
                            acc, acc, t
                        ));
                    }
                    let eff_conv =
                        (*conv).or(if debug.is_some() { Some('r') } else { None });
                    let t = self.emit_formatted_field(
                        ctx,
                        value,
                        eff_conv,
                        spec.as_deref(),
                        line,
                    )?;
                    ctx.line(&format!(
                        "{} = __psa_str_append_str({}, {});",
                        acc, acc, t
                    ));
                }
            }
        }
        Ok(acc)
    }

    /// Emit one formatted f-string field (a conversion, a debug
    /// marker and/or a format spec is present): evaluate the
    /// value as a dynamic value, evaluate the spec to a dynamic
    /// string, and call the runtime formatter. Returns the temp
    /// holding the formatted tag-6 result.
    fn emit_formatted_field(
        &mut self,
        ctx: &mut Ctx,
        value: &Expr,
        conv: Option<char>,
        spec: Option<&[FStrPart]>,
        line: u32,
    ) -> Result<String, Diagnostic> {
        self.used_dyn = true;
        let vcode = if let Expr::Str(s) = value {
            self.emit_str_temp(ctx, s, line)?
        } else {
            let (code, ty) = self.emit_expr(ctx, value, line)?;
            match ty {
                Ty::Dyn => code,
                Ty::Int | Ty::Float | Ty::Bool => self.box_code(code, ty, line)?,
                Ty::Str => {
                    return Err(eerr(
                        line,
                        "a dynamically-built string cannot appear in a formatted f-string field (only string literals become dynamic strings)",
                    ))
                }
                _ => {
                    return Err(eerr(
                        line,
                        &format!(
                            "values of type {} cannot be formatted in an f-string field",
                            ty.py_name()
                        ),
                    ))
                }
            }
        };
        let scode = match spec {
            Some(parts) => self.emit_fstr_value(ctx, parts, line)?,
            None => self.emit_str_temp(ctx, "", line)?,
        };
        let ccode = match conv {
            Some('r') => 1,
            Some('s') => 2,
            Some('a') => 3,
            _ => 0,
        };
        ctx.tmp += 1;
        let t = format!("__psa_fmt_{}", ctx.tmp);
        ctx.declare(&t, Ty::Dyn).map_err(|m| eerr(line, &m))?;
        ctx.line(&format!(
            "let {} = __psa_format_value({}, {}, {});",
            t, vcode, ccode, scode
        ));
        Ok(t)
    }

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
    ) -> Result<(), Diagnostic> {
        // A field with a conversion, a debug marker or a format
        // spec goes through the runtime formatter: build the
        // whole f-string as a dynamic string and print that.
        // (This also lifts the plain-path limits for such
        // f-strings: float fields and literal braces work,
        // because the text never reaches SLA's println format.)
        if parts.iter().any(|p| p.is_formatted()) {
            let t = self.emit_fstr_value(ctx, parts, line)?;
            ctx.line(&format!("__psa_print_dyn({});", t));
            return Ok(());
        }
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
                FStrPart::Field { value: e, .. } => {
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

    fn emit_expr(&mut self, ctx: &mut Ctx, e: &Expr, line: u32) -> Result<(String, Ty), Diagnostic> {
        // P1 source map: lines materialised while evaluating this
        // expression (temps) anchor to the expression's psa line
        // (kind Expr). `line == 0` means a synthetic expression
        // (closure conversion): keep the enclosing anchor.
        let saved = ctx.save_anchor();
        if line != 0 {
            let (anchor, precision) = match expr_span_start(e) {
                Some(a) if !a.is_synthetic() => {
                    (a, if a.col >= 1 { Precision::Exact } else { Precision::LineOnly })
                }
                _ => (SourceAnchor::at(line, 0, None), Precision::LineOnly),
            };
            ctx.set_anchor(anchor, SegKind::Expr, precision, None);
        }
        let r = self.emit_expr_inner(ctx, e, line);
        ctx.restore_anchor(saved);
        r
    }

    fn emit_expr_inner(&mut self, ctx: &mut Ctx, e: &Expr, line: u32) -> Result<(String, Ty), Diagnostic> {
        match e {
            Expr::Int(v) => Ok((v.to_string(), Ty::Int)),
            Expr::BigInt(d) => {
                // Merge note (V1): dyn contexts fall through to this
                // arm (emit_dyn_value and binop operands route via
                // emit_expr), so the literal is built as a dynamic
                // bigint here, exactly as literal_dyn_code does.
                // Static boundaries then refuse loudly: a static
                // call argument gets the PSA-E6010 compile refusal
                // (see the boundary loop in the call path), other
                // static slots unbox through __psa_unbox_int and
                // panic 77080 at runtime for a tag-17 value.
                self.used_dyn = true;
                let t = self.emit_str_temp(ctx, d, line)?;
                Ok((format!("__psa_big_from_psastr({})", t), Ty::Dyn))
            }
            Expr::Float(v) => Ok((fmt_f64(*v), Ty::Float)),
            Expr::Str(s) => Ok((format!("\"{}\"", escape_sla_str(s)), Ty::Str)),
            Expr::Bytes(bs) => {
                // Bytes have no static form: a literal is built
                // directly as a dynamic bytes value (tag 15).
                let t = self.emit_bytes_temp(ctx, bs, line)?;
                Ok((t, Ty::Dyn))
            }
            Expr::Walrus { name, value } => {
                // Evaluate, bind, yield: the binding reuses the
                // plain assignment machinery wholesale (type
                // pinning, dynamic stage stores, move marking,
                // the string-literal special case), via a
                // synthetic Assign statement; the expression's
                // value is then the target variable itself, with
                // whatever type the binding pinned.
                let stmt = Stmt::Assign {
                    name: name.clone(),
                    value: (**value).clone(),
                    line, anchor: SourceAnchor::at(line, 0, None),
                };
                self.emit_stmt(ctx, &stmt)?;
                let ty = ctx.lookup(name).ok_or_else(|| {
                    eerr(line, "internal error: walrus target unbound after binding")
                })?;
                Ok((name.clone(), ty))
            }
            Expr::Bool(b) => Ok((b.to_string(), Ty::Bool)),
            Expr::NoneLit => {
                self.used_dyn = true;
                Ok((Ty::Dyn.default_lit().to_string(), Ty::Dyn))
            }
            Expr::Name(n, span) => {
                ctx.check_not_deleted(n, line)?;
                match ctx.lookup(n) {
                    Some(t) => {
                        ctx.check_alive(n, line, Some(span.start))?;
                        Ok((n.clone(), t))
                    }
                    None => {
                        // A def name in value position is a
                        // function value (D3), not a variable.
                        if self.defs.contains_key(n) {
                            self.fn_val_code(n, line)
                        } else if let Some(lit) = self.const_names.get(n).cloned() {
                            // D6: a literal name bound by a
                            // from-import evaluates at the read.
                            match self.literal_dyn_code(ctx, &lit, line) {
                                Some(r) => r,
                                None => Err(eerr(
                                    line,
                                    &format!("imported name `{}` is not a literal value", n),
                                )),
                            }
                        } else {
                            Err(eerr(line, &format!("undefined variable `{}`", n)))
                        }
                    }
                }
            }
            Expr::FnVal(n) => self.fn_val_code(n, line),
            // D8 closures: the cell and environment nodes
            // produced by the closure-conversion pass. A
            // cell's contents are always dynamic: the stored
            // value goes through emit_dyn_value (scalars box,
            // containers materialise in temps).
            Expr::CellNew(v) => {
                self.used_dyn = true;
                self.used_closure = true;
                let c = self.emit_dyn_value(ctx, v, line)?;
                Ok((format!("__psa_cell_new({})", c), Ty::Dyn))
            }
            Expr::CellGet(c) => {
                self.used_dyn = true;
                self.used_closure = true;
                let (cc, _) = self.emit_expr(ctx, c, line)?;
                Ok((format!("__psa_cell_get({})", cc), Ty::Dyn))
            }
            Expr::CellSet(c, v) => {
                self.used_dyn = true;
                self.used_closure = true;
                let (cc, _) = self.emit_expr(ctx, c, line)?;
                let vc = self.emit_dyn_value(ctx, v, line)?;
                Ok((format!("__psa_cell_set({}, {})", cc, vc), Ty::Dyn))
            }
            Expr::ClosureVal(n, env) => {
                // The env is a ListLit of cell names built by
                // the converter; materialise it as a
                // dynamic-list temp. A raw emit_expr refuses
                // bare list literals outside assignment/return
                // positions (v0.3 gate), which is exactly the
                // position a closure value is built in.
                let env_code = self.emit_dyn_value(ctx, env, line)?;
                self.closure_val_code(n, &env_code, line)
            }
            Expr::ModEnv => {
                self.used_dyn = true;
                self.used_closure = true;
                Ok(("__psa_modenv()".to_string(), Ty::Dyn))
            }
            Expr::Attr(base, name, _) => {
                // D6: a module attribute read through a chain
                // (`foo.CONST`, `foo.bar.CONST`) whose root is
                // an imported module, not a variable: the
                // (literal) value evaluates at the read site.
                if let Some(segs) = Self::attr_chain(e) {
                    if segs.len() >= 2 && ctx.lookup(&segs[0]).is_none() {
                        let key = segs.join(".");
                        if let Some(lit) = self.mod_attrs.get(&key).cloned() {
                            match self.literal_dyn_code(ctx, &lit, line) {
                                Some(r) => return r,
                                None => {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "module attribute `{}` is not a literal value",
                                            key
                                        ),
                                    ))
                                }
                            }
                        }
                    }
                }
                // Class-side attribute read: `ClassName.attr`
                // where the name is a class, not a variable.
                // The value lives in the class's runtime
                // registry dict (seeded at program start,
                // mutable at runtime), so the read walks the
                // ancestor dicts; a name created by assignment
                // at runtime is found the same way. Method
                // names stay an emit error (a method is not a
                // value on the class side in v1).
                if let Expr::Name(n, _) = &**base {
                    if ctx.lookup(n).is_none() {
                        if let Some(&cid) = self.class_ids.get(n) {
                            if self.classes[cid]
                                .flat_methods
                                .iter()
                                .any(|(mn, _)| mn == name)
                            {
                                return Err(eerr(
                                    line,
                                    &format!(
                                        "class `{}` has no attribute `{}` readable as a value in D4 v1 (methods are reached through instances or explicit base calls)",
                                        n, name
                                    ),
                                ));
                            }
                            self.used_dyn = true;
                            let ntmp = self.emit_str_temp(ctx, name, line)?;
                            return Ok((
                                format!("__psa_class_{}_cgetattr({})", cid, ntmp),
                                Ty::Dyn,
                            ));
                        }
                    }
                }
                let (bcode, bty) = self.emit_expr(ctx, base, line)?;
                if bty != Ty::Dyn {
                    return Err(eerr(
                        line,
                        &format!(
                            "attribute access on a {} value is not supported yet (objects are dynamic values in D4; container/string attribute surfaces arrive in D5)",
                            bty.py_name()
                        ),
                    ));
                }
                self.used_dyn = true;
                let ntmp = self.emit_str_temp(ctx, name, line)?;
                Ok((format!("__psa_getattr({}, {})", bcode, ntmp), Ty::Dyn))
            }
            Expr::Lambda { .. } => Err(eerr(
                line,
                "internal error: a lambda reached emission without desugaring",
            )),
            Expr::ListLit(_) => Err(eerr(
                line,
                "list literals are only supported as an assignment value or a return value in v0.3",
            )),
            Expr::DictLit(_) => Err(eerr(
                line,
                "dict literals are only supported as an assignment value or a return value in this slice",
            )),
            Expr::TupleLit(elems) => {
                let t = self.emit_tuple_temp(ctx, elems, line)?;
                Ok((t, Ty::Dyn))
            }
            Expr::SetLit(elems) => {
                let t = self.emit_set_temp(ctx, elems, line)?;
                Ok((t, Ty::Dyn))
            }
            Expr::Comp { .. } => {
                let t = self.emit_comp(ctx, e, line)?;
                Ok((t, Ty::Dyn))
            }
            Expr::FStr(parts) => {
                // D5: an f-string as a general value concatenates
                // its parts into a dynamic string.
                let t = self.emit_fstr_value(ctx, parts, line)?;
                Ok((t, Ty::Dyn))
            }
            Expr::Index(base, idx, _) => {
                // Chained indexing on a dynamic sub-expression
                // (m[1][0]): the inner value is the container.
                if !matches!(&**base, Expr::Name(_, _)) {
                    let (bcode, bty) = self.emit_expr(ctx, base, line)?;
                    if bty == Ty::Dyn {
                        self.used_dyn = true;
                        let (icode, ity) = if let Expr::Str(s) = &**idx {
                            (self.emit_str_temp(ctx, s, line)?, Ty::Dyn)
                        } else {
                            self.emit_expr(ctx, idx, line)?
                        };
                        let kcode = match ity {
                            Ty::Dyn => icode,
                            Ty::Int | Ty::Bool | Ty::Float => self.box_code(icode, ity, line)?,
                            _ => {
                                return Err(eerr(
                                    line,
                                    "a dynamic subscript key must be an int, float, bool, None, str or dynamic value",
                                ))
                            }
                        };
                        return Ok((format!("__psa_dyn_get({}, {})", bcode, kcode), Ty::Dyn));
                    }
                    return Err(eerr(
                        line,
                        "indexing is only supported on plain variables in v0.3",
                    ));
                }
                let (obj, obj_anchor) = match &**base {
                    Expr::Name(n, span) => (n.clone(), span.start),
                    _ => unreachable!(),
                };
                let ty = ctx
                    .lookup(&obj)
                    .ok_or_else(|| eerr(line, &format!("undefined variable `{}`", obj)))?;
                ctx.check_alive(&obj, line, Some(obj_anchor))?;
                if ty == Ty::Dyn {
                    // Dynamic subscript: the key stays a PsaVal and
                    // the runtime dispatches on the container kind
                    // (list: int key, negative-index normalisation,
                    // bounds panic 77050; dict: key normalisation,
                    // missing-key panic 77054; other kinds 77052).
                    self.used_dyn = true;
                    let (icode, ity) = if let Expr::Str(s) = &**idx {
                        (self.emit_str_temp(ctx, s, line)?, Ty::Dyn)
                    } else {
                        self.emit_expr(ctx, idx, line)?
                    };
                    let kcode = match ity {
                        Ty::Dyn => icode,
                        Ty::Int | Ty::Bool | Ty::Float => self.box_code(icode, ity, line)?,
                        _ => {
                            return Err(eerr(
                                line,
                                "a dynamic subscript key must be an int, float, bool, None, str or dynamic value",
                            ))
                        }
                    };
                    return Ok((format!("__psa_dyn_get({}, {})", obj, kcode), Ty::Dyn));
                }
                let (mut icode, ity) = self.emit_expr(ctx, idx, line)?;
                if ity == Ty::Dyn {
                    icode = self.unbox_code(icode, Ty::Int, line)?;
                } else if ity != Ty::Int {
                    return Err(eerr(line, "list index must be int"));
                }
                let elem = match ty.elem() {
                    Some(e) => e,
                    None => {
                        return Err(eerr(
                            line,
                            &format!("indexing is only supported on lists, not on {}", ty.py_name()),
                        ))
                    }
                };
                // Scalar element: copied out of the borrow (spec s5).
                Ok((format!("{}[{}]", obj, icode), elem))
            }
            Expr::Slice(base, lo, hi, step) => {
                // A string literal base becomes a dynamic string first.
                if let Expr::Str(s) = &**base {
                    let tmp = self.emit_str_temp(ctx, s, line)?;
                    let lo_c = self.slice_bound_code(ctx, lo, line)?;
                    let hi_c = self.slice_bound_code(ctx, hi, line)?;
                    let st_c = self.slice_bound_code(ctx, step, line)?;
                    return Ok((
                        format!("__psa_dyn_slice({}, {}, {}, {})", tmp, lo_c, hi_c, st_c),
                        Ty::Dyn,
                    ));
                }
                // Chained slicing on a dynamic sub-expression.
                if !matches!(&**base, Expr::Name(_, _)) {
                    let (bcode, bty) = self.emit_expr(ctx, base, line)?;
                    if bty == Ty::Dyn {
                        self.used_dyn = true;
                        let lo_c = self.slice_bound_code(ctx, lo, line)?;
                        let hi_c = self.slice_bound_code(ctx, hi, line)?;
                        let st_c = self.slice_bound_code(ctx, step, line)?;
                        return Ok((
                            format!("__psa_dyn_slice({}, {}, {}, {})", bcode, lo_c, hi_c, st_c),
                            Ty::Dyn,
                        ));
                    }
                    return Err(eerr(
                        line,
                        "slicing is only supported on plain variables in this slice",
                    ));
                }
                let (obj, obj_anchor) = match &**base {
                    Expr::Name(n, span) => (n.clone(), span.start),
                    _ => unreachable!(),
                };
                let ty = ctx
                    .lookup(&obj)
                    .ok_or_else(|| eerr(line, &format!("undefined variable `{}`", obj)))?;
                ctx.check_alive(&obj, line, Some(obj_anchor))?;
                if ty == Ty::Dyn {
                    self.used_dyn = true;
                    let lo_c = self.slice_bound_code(ctx, lo, line)?;
                    let hi_c = self.slice_bound_code(ctx, hi, line)?;
                    let st_c = self.slice_bound_code(ctx, step, line)?;
                    return Ok((
                        format!("__psa_dyn_slice({}, {}, {}, {})", obj, lo_c, hi_c, st_c),
                        Ty::Dyn,
                    ));
                }
                let elem = match ty.elem() {
                    Some(e) => e,
                    None => {
                        return Err(eerr(
                            line,
                            &format!("slicing is only supported on lists, not on {}", ty.py_name()),
                        ))
                    }
                };
                // Static list slice: stage the bounds once, ask the
                // runtime for the normalised (start, stop, step),
                // then copy the selected elements into a fresh Vec.
                self.used_dyn = true;
                let mut staged: Vec<(String, &'static str)> = Vec::new();
                for (i, b) in [lo, hi, step].iter().enumerate() {
                    match b {
                        Some(e) => {
                            let (c, t) = self.emit_expr(ctx, e, line)?;
                            let c = match t {
                                Ty::Int => c,
                                Ty::Dyn => format!("__psa_unbox_int({})", c),
                                _ => {
                                    return Err(eerr(
                                        line,
                                        "slice indices must be integers",
                                    ))
                                }
                            };
                            ctx.tmp += 1;
                            let v = format!("__psa_sb_{}", ctx.tmp);
                            ctx.line(&format!("let {}: i64 = {};", v, c));
                            staged.push((v, "true"));
                            let _ = i;
                        }
                        None => staged.push(("0".to_string(), "false")),
                    }
                }
                ctx.tmp += 1;
                let tag = ctx.tmp;
                let (sv, se, st) = (
                    format!("__psa_ss_{}", tag),
                    format!("__psa_se_{}", tag),
                    format!("__psa_st_{}", tag),
                );
                ctx.line(&format!(
                    "let ({}, {}, {}) = __psa_slice_bounds((len({}) as i64), {}, {}, {}, {}, {}, {});",
                    sv, se, st, obj,
                    staged[0].0, staged[0].1, staged[1].0, staged[1].1, staged[2].0, staged[2].1
                ));
                ctx.tmp += 1;
                let out = format!("__psa_slc_{}", ctx.tmp);
                ctx.declare(&out, ty).map_err(|m| eerr(line, &m))?;
                ctx.line(&format!("let {}: {} = Vec::new();", out, ty.sla()));
                ctx.tmp += 1;
                let iv = format!("__psa_si_{}", ctx.tmp);
                ctx.line(&format!("let {}: i64 = {};", iv, sv));
                ctx.line(&format!(
                    "while ({} > 0 && {} < {}) || ({} < 0 && {} > {}) {{",
                    st, iv, se, st, iv, se
                ));
                ctx.line(&format!("{}.push({}[{}]);", out, obj, iv));
                ctx.line(&format!("{} = {} + {};", iv, iv, st));
                ctx.line("}");
                Ok((out, ty))
            }
            Expr::MethodCall { obj, method, args } => {
                // D4: a method call on a class name is an explicit
                // base call -- `Base.method(self, ...)` resolves
                // through Base's flattened method table and goes
                // down the ordinary def-call path (binding,
                // defaults and all).
                if ctx.lookup(obj).is_none() {
                    if let Some(&cid) = self.class_ids.get(obj) {
                        let key = self.classes[cid]
                            .flat_methods
                            .iter()
                            .find(|(n, _)| n == method)
                            .map(|(_, k)| k.clone());
                        return match key {
                            Some(k) => {
                                let call =
                                    Expr::Call(Box::new(Expr::Name(k, ExprSpan::synthetic())), args.clone(), Vec::new(), ExprSpan::synthetic());
                                self.emit_expr(ctx, &call, line)
                            }
                            None => Err(eerr(
                                line,
                                &format!("class `{}` has no method `{}`", obj, method),
                            )),
                        };
                    }
                }
                // psa_std: `json.dumps(v)` / `json.loads(s)`
                // with `import json` bound to the standard
                // library call the spliced module's SLA entry
                // points directly (one dynamic argument each).
                // The parser makes these MethodCall nodes, not
                // Call(Attr), so the intercept lives here.
                if self.std_json && obj == "json" && ctx.lookup(obj).is_none() {
                    let sla_fn = match method.as_str() {
                        "dumps" => "psa_json_dumps",
                        "loads" => "psa_json_parse",
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `json` has no function `{}` in v1 (dumps / loads)",
                                    method
                                ),
                            ))
                        }
                    };
                    if args.len() != 1 {
                        return Err(eerr(
                            line,
                            &format!(
                                "json.{} takes exactly one positional argument in v1",
                                method
                            ),
                        ));
                    }
                    let acode = self.emit_dyn_value(ctx, &args[0], line)?;
                    self.used_dyn = true;
                    self.used_psa_json = true;
                    return Ok((format!("{}({})", sla_fn, acode), Ty::Dyn));
                }
                // psa_std: `pickle.dumps(v)` / `pickle.dump(v, f)` /
                // `pickle.loads(data)` / `pickle.load(f)` with
                // `import pickle` bound to the standard library call
                // the spliced module's SLA entry points directly.
                // dump/load ride the file runtime (__psa_file_write /
                // __psa_file_read_all), so they set used_file.
                if self.std_pickle && obj == "pickle" && ctx.lookup(obj).is_none() {
                    let (sla_fn, arity): (&str, usize) = match method.as_str() {
                        "dumps" => ("psa_pickle_dumps", 1),
                        "dump" => ("psa_pickle_dump", 2),
                        "loads" => ("psa_pickle_loads", 1),
                        "load" => ("psa_pickle_load", 1),
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `pickle` has no function `{}` in v1 (dumps / dump / loads / load)",
                                    method
                                ),
                            ))
                        }
                    };
                    if args.len() != arity {
                        return Err(eerr(
                            line,
                            &format!(
                                "pickle.{} takes exactly {} positional argument(s) in v1",
                                method, arity
                            ),
                        ));
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    self.used_dyn = true;
                    self.used_psa_pickle = true;
                    // The pickle splice is monolithic: it
                    // always carries dump/load, which call
                    // __psa_file_write / __psa_file_read_all —
                    // so the file runtime must be emitted
                    // whenever ANY pickle entry point is used,
                    // or a loads-only program fails the host
                    // type check with Undefined call.
                    self.used_file = true;
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std L1 batch 2: `math.<fn>(...)` with
                // `import math` bound to the standard library
                // calls the spliced module's SLA entry points
                // directly, same shape as the json intercept
                // above. Arities are per-function (the CPython
                // signatures restricted to positional args).
                if self.std_math && obj == "math" && ctx.lookup(obj).is_none() {
                    let (sla_fn, arity): (&str, usize) = match method.as_str() {
                        "floor" => ("psa_math_floor", 1),
                        "ceil" => ("psa_math_ceil", 1),
                        "trunc" => ("psa_math_trunc", 1),
                        "fabs" => ("psa_math_fabs", 1),
                        "sqrt" => ("psa_math_sqrt", 1),
                        "sin" => ("psa_math_sin", 1),
                        "cos" => ("psa_math_cos", 1),
                        "degrees" => ("psa_math_degrees", 1),
                        "radians" => ("psa_math_radians", 1),
                        "factorial" => ("psa_math_factorial", 1),
                        "isqrt" => ("psa_math_isqrt", 1),
                        "gcd" => ("psa_math_gcd", 2),
                        "lcm" => ("psa_math_lcm", 2),
                        "comb" => ("psa_math_comb", 2),
                        "perm" => ("psa_math_perm", 2),
                        "pow" => ("psa_math_pow", 2),
                        "isclose" => ("psa_math_isclose", 2),
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `math` has no function `{}` in v1 (floor / ceil / trunc / fabs / sqrt / sin / cos / degrees / radians / factorial / isqrt / gcd / lcm / comb / perm / pow / isclose)",
                                    method
                                ),
                            ))
                        }
                    };
                    if args.len() != arity {
                        return Err(eerr(
                            line,
                            &format!(
                                "math.{} takes exactly {} positional argument(s) in v1",
                                method, arity
                            ),
                        ));
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    self.used_dyn = true;
                    self.used_psa_math = true;
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std L1 batch 2: `string.capwords(...)`
                // is the string module's one wired function
                // (its constants are values, reached through
                // the module's zero-argument entry points in
                // the SLA text, not through calls -- see
                // psa-std/README.md).
                if self.std_string && obj == "string" && ctx.lookup(obj).is_none() {
                    if method != "capwords" {
                        return Err(eerr(
                            line,
                            &format!(
                                "psa_std module `string` has no function `{}` in v1 (capwords; the constants are values, not functions)",
                                method
                            ),
                        ));
                    }
                    if args.len() != 1 && args.len() != 2 {
                        return Err(eerr(
                            line,
                            "string.capwords takes one or two positional arguments in v1",
                        ));
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    self.used_dyn = true;
                    self.used_psa_string = true;
                    let sla_fn = if args.len() == 2 {
                        "psa_string_capwords_sep"
                    } else {
                        "psa_string_capwords"
                    };
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std batch 3: `csv.<fn>(...)` with
                // `import csv` bound to the standard library,
                // same shape as the math intercept above.
                // write_rows additionally sets used_file so
                // the D6 file runtime (which provides
                // __psa_file_write) is emitted.
                if self.std_csv && obj == "csv" && ctx.lookup(obj).is_none() {
                    let (sla_fn, arity): (&str, usize) = match method.as_str() {
                        "reads" => ("psa_csv_reads", 1),
                        "reads_d" => ("psa_csv_reads_d", 2),
                        "reader" => ("psa_csv_reader", 1),
                        "reader_d" => ("psa_csv_reader_d", 2),
                        "writes" => ("psa_csv_writes", 1),
                        "writes_d" => ("psa_csv_writes_d", 2),
                        "write_rows" => ("psa_csv_write_rows", 3),
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `csv` has no function `{}` in v1 (reads / reads_d / reader / reader_d / writes / writes_d / write_rows)",
                                    method
                                ),
                            ))
                        }
                    };
                    if args.len() != arity {
                        return Err(eerr(
                            line,
                            &format!(
                                "csv.{} takes exactly {} positional argument(s) in v1",
                                method, arity
                            ),
                        ));
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    self.used_dyn = true;
                    self.used_psa_csv = true;
                    if method == "write_rows" {
                        self.used_file = true;
                    }
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std batch 3: `logging.<fn>(...)`. The
                // root logging calls (debug / info / warning
                // / error / critical, and log) are CPython
                // *args forms: the intercept passes the
                // message and collects any trailing
                // positional arguments into a tag-4 list
                // (None when there are none) for the root_*
                // SLA entry points. Logger method syntax is
                // not wired (it needs std object tracking;
                // see WIRING-logging.md).
                if self.std_logging && obj == "logging" && ctx.lookup(obj).is_none() {
                    let root_fn: Option<&str> = match method.as_str() {
                        "debug" => Some("psa_logging_root_debug"),
                        "info" => Some("psa_logging_root_info"),
                        "warning" => Some("psa_logging_root_warning"),
                        "error" => Some("psa_logging_root_error"),
                        "critical" => Some("psa_logging_root_critical"),
                        _ => None,
                    };
                    if let Some(sla_fn) = root_fn {
                        if args.is_empty() {
                            return Err(eerr(
                                line,
                                &format!(
                                    "logging.{} takes at least one positional argument (the message) in v1",
                                    method
                                ),
                            ));
                        }
                        let msgcode = self.emit_dyn_value(ctx, &args[0], line)?;
                        let argscode = if args.len() == 1 {
                            "__psa_m_none()".to_string()
                        } else {
                            self.emit_dyn_value(
                                ctx,
                                &Expr::ListLit(args[1..].to_vec()),
                                line,
                            )?
                        };
                        self.used_dyn = true;
                        self.used_psa_logging = true;
                        return Ok((
                            format!("{}({}, {})", sla_fn, msgcode, argscode),
                            Ty::Dyn,
                        ));
                    }
                    if method == "log" {
                        if args.len() < 2 {
                            return Err(eerr(
                                line,
                                "logging.log takes at least two positional arguments (level, message) in v1",
                            ));
                        }
                        let levelcode = self.emit_dyn_value(ctx, &args[0], line)?;
                        let msgcode = self.emit_dyn_value(ctx, &args[1], line)?;
                        let argscode = if args.len() == 2 {
                            "__psa_m_none()".to_string()
                        } else {
                            self.emit_dyn_value(
                                ctx,
                                &Expr::ListLit(args[2..].to_vec()),
                                line,
                            )?
                        };
                        self.used_dyn = true;
                        self.used_psa_logging = true;
                        return Ok((
                            format!(
                                "psa_logging_root_log({}, {}, {})",
                                levelcode, msgcode, argscode
                            ),
                            Ty::Dyn,
                        ));
                    }
                    let (sla_fn, arity): (&str, usize) = match method.as_str() {
                        "getLogger" => ("psa_logging_get_logger", 1),
                        "set_level" => ("psa_logging_set_level", 2),
                        "get_effective_level" => ("psa_logging_get_effective_level", 1),
                        "basicConfig" => ("psa_logging_basic_config", 2),
                        "basicConfig_force" => ("psa_logging_basic_config_force", 2),
                        "level_debug" => ("psa_logging_level_debug", 0),
                        "level_info" => ("psa_logging_level_info", 0),
                        "level_warning" => ("psa_logging_level_warning", 0),
                        "level_error" => ("psa_logging_level_error", 0),
                        "level_critical" => ("psa_logging_level_critical", 0),
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `logging` has no function `{}` in v1 (getLogger / set_level / get_effective_level / basicConfig / basicConfig_force / log / debug / info / warning / error / critical / level_debug / level_info / level_warning / level_error / level_critical)",
                                    method
                                ),
                            ))
                        }
                    };
                    if args.len() != arity {
                        return Err(eerr(
                            line,
                            &format!(
                                "logging.{} takes exactly {} positional argument(s) in v1",
                                method, arity
                            ),
                        ));
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    self.used_dyn = true;
                    self.used_psa_logging = true;
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std batch 3:
                // `argparse.ArgumentParser(prog, description)`
                // is the module's one wired entry point; a
                // missing description pads to None.
                // ArgumentParser method syntax (add_argument
                // / parse_args / ...) is not wired -- it
                // needs std object tracking (see
                // WIRING-argparse.md). used_psa_json was
                // primed at construction for the argv path.
                if self.std_argparse && obj == "argparse" && ctx.lookup(obj).is_none() {
                    if method != "ArgumentParser" {
                        return Err(eerr(
                            line,
                            &format!(
                                "psa_std module `argparse` has no function `{}` in v1 (ArgumentParser)",
                                method
                            ),
                        ));
                    }
                    if args.len() != 1 && args.len() != 2 {
                        return Err(eerr(
                            line,
                            "argparse.ArgumentParser takes one or two positional arguments (prog, description) in v1",
                        ));
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    if acodes.len() == 1 {
                        acodes.push("__psa_m_none()".to_string());
                    }
                    self.used_dyn = true;
                    self.used_psa_argparse = true;
                    return Ok((
                        format!("psa_argparse_new_parser({})", acodes.join(", ")),
                        Ty::Dyn,
                    ));
                }
                // psa_std batch 3: `collections.<fn>(...)`,
                // same shape as the math intercept above.
                // Only the module-level spellings are wired;
                // method-call syntax on the returned
                // containers is not (WIRING-collections.md).
                if self.std_collections && obj == "collections" && ctx.lookup(obj).is_none() {
                    let (sla_fn, arity): (&str, usize) = match method.as_str() {
                        "Counter" => ("psa_collections_counter_any", 1),
                        "defaultdict" => ("psa_collections_defaultdict_new", 1),
                        "deque" => ("psa_collections_deque_new", 1),
                        "ChainMap" => ("psa_collections_chainmap_new", 1),
                        "OrderedDict" => ("psa_collections_ordereddict_any", 1),
                        "namedtuple" => ("psa_collections_namedtuple", 2),
                        "counter_dict" => ("psa_collections_counter_dict", 1),
                        "counter_get" => ("psa_collections_counter_get", 2),
                        "counter_update" => ("psa_collections_counter_update", 2),
                        "counter_subtract" => ("psa_collections_counter_subtract", 2),
                        "counter_add" => ("psa_collections_counter_add", 2),
                        "counter_sub" => ("psa_collections_counter_sub", 2),
                        "counter_and" => ("psa_collections_counter_and", 2),
                        "counter_or" => ("psa_collections_counter_or", 2),
                        "counter_elements" => ("psa_collections_counter_elements", 1),
                        "counter_total" => ("psa_collections_counter_total", 1),
                        "counter_most_common" => ("psa_collections_counter_most_common", 1),
                        "counter_most_common_n" => ("psa_collections_counter_most_common_n", 2),
                        "defaultdict_get" => ("psa_collections_defaultdict_get", 3),
                        "deque_append" => ("psa_collections_deque_append", 2),
                        "deque_append_maxlen" => ("psa_collections_deque_append_maxlen", 3),
                        "deque_appendleft" => ("psa_collections_deque_appendleft", 2),
                        "deque_appendleft_maxlen" => {
                            ("psa_collections_deque_appendleft_maxlen", 3)
                        }
                        "deque_pop" => ("psa_collections_deque_pop", 1),
                        "deque_popleft" => ("psa_collections_deque_popleft", 1),
                        "deque_extend" => ("psa_collections_deque_extend", 2),
                        "deque_extendleft" => ("psa_collections_deque_extendleft", 2),
                        "deque_rotate" => ("psa_collections_deque_rotate", 2),
                        "deque_count" => ("psa_collections_deque_count", 2),
                        "deque_remove" => ("psa_collections_deque_remove", 2),
                        "deque_clear" => ("psa_collections_deque_clear", 1),
                        "chainmap_get" => ("psa_collections_chainmap_get", 2),
                        "chainmap_get_default" => ("psa_collections_chainmap_get_default", 3),
                        "chainmap_set" => ("psa_collections_chainmap_set", 3),
                        "chainmap_del" => ("psa_collections_chainmap_del", 2),
                        "chainmap_keys" => ("psa_collections_chainmap_keys", 1),
                        "chainmap_new_child" => ("psa_collections_chainmap_new_child", 2),
                        "chainmap_parents" => ("psa_collections_chainmap_parents", 1),
                        "ordereddict_move_to_end" => {
                            ("psa_collections_ordereddict_move_to_end", 2)
                        }
                        "ordereddict_move_to_end_first" => {
                            ("psa_collections_ordereddict_move_to_end_first", 2)
                        }
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `collections` has no function `{}` in v1 (Counter / defaultdict / deque / ChainMap / OrderedDict / namedtuple / counter_* / defaultdict_get / deque_* / chainmap_* / ordereddict_move_to_end*)",
                                    method
                                ),
                            ))
                        }
                    };
                    // `deque` alone among the constructors has
                    // a zero-argument form.
                    let (sla_fn, arity) = if method == "deque" && args.is_empty() {
                        ("psa_collections_deque_new_empty", 0)
                    } else {
                        (sla_fn, arity)
                    };
                    if args.len() != arity {
                        return Err(eerr(
                            line,
                            &format!(
                                "collections.{} takes exactly {} positional argument(s) in v1",
                                method, arity
                            ),
                        ));
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    self.used_dyn = true;
                    self.used_psa_collections = true;
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std batch 3: `itertools.<fn>(...)`,
                // same shape as the math intercept above.
                // Callable arguments arrive as tag-7/9 dyn
                // values through emit_dyn_value.
                if self.std_itertools && obj == "itertools" && ctx.lookup(obj).is_none() {
                    let (sla_fn, arity): (&str, usize) = match method.as_str() {
                        "chain" => ("psa_itertools_chain", 2),
                        "chain_from_iterable" => ("psa_itertools_chain_from_iterable", 1),
                        "islice" => ("psa_itertools_islice_stop", 2),
                        "repeat" => ("psa_itertools_repeat", 2),
                        "accumulate" => ("psa_itertools_accumulate", 1),
                        "accumulate_init" => ("psa_itertools_accumulate_init", 2),
                        "takewhile" => ("psa_itertools_takewhile", 2),
                        "dropwhile" => ("psa_itertools_dropwhile", 2),
                        "compress" => ("psa_itertools_compress", 2),
                        "starmap" => ("psa_itertools_starmap", 2),
                        "tee" => ("psa_itertools_tee2", 1),
                        "pairwise" => ("psa_itertools_pairwise", 1),
                        "zip_longest" => ("psa_itertools_zip_longest", 3),
                        "groupby" => ("psa_itertools_groupby", 1),
                        "product" => ("psa_itertools_product", 2),
                        "product_from" => ("psa_itertools_product_from", 1),
                        "permutations" => ("psa_itertools_permutations_full", 1),
                        "combinations" => ("psa_itertools_combinations", 2),
                        "combinations_with_replacement" => {
                            ("psa_itertools_combinations_with_replacement", 2)
                        }
                        "count" => ("psa_itertools_count", 2),
                        "cycle" => ("psa_itertools_cycle", 1),
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `itertools` has no function `{}` in v1 (chain / chain_from_iterable / islice / repeat / accumulate / accumulate_init / takewhile / dropwhile / compress / starmap / tee / pairwise / zip_longest / groupby / product / product_from / permutations / combinations / combinations_with_replacement / count / cycle)",
                                    method
                                ),
                            ))
                        }
                    };
                    // Arity-keyed entries, following the
                    // string.capwords precedent.
                    let sla_fn = match (method.as_str(), args.len()) {
                        ("islice", 3) => "psa_itertools_islice_start_stop",
                        ("islice", 4) => "psa_itertools_islice_start_stop_step",
                        ("accumulate", 2) => "psa_itertools_accumulate_fn",
                        ("groupby", 2) => "psa_itertools_groupby_key",
                        ("permutations", 2) => "psa_itertools_permutations",
                        _ => sla_fn,
                    };
                    let arity_ok = match method.as_str() {
                        "islice" => args.len() == 2 || args.len() == 3 || args.len() == 4,
                        "accumulate" | "groupby" | "permutations" => {
                            args.len() == 1 || args.len() == 2
                        }
                        _ => args.len() == arity,
                    };
                    if !arity_ok {
                        return Err(eerr(
                            line,
                            &format!(
                                "itertools.{} does not take {} positional argument(s) in v1",
                                method,
                                args.len()
                            ),
                        ));
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    self.used_dyn = true;
                    self.used_psa_itertools = true;
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std batch 3: `functools.<fn>(...)`,
                // same shape as the math intercept above.
                // Only reduce is a real function; the factory
                // names are wired deliberately and panic
                // 77148 at run time (registered gap).
                if self.std_functools && obj == "functools" && ctx.lookup(obj).is_none() {
                    let (sla_fn, arity): (&str, usize) = match method.as_str() {
                        "reduce" => ("psa_functools_reduce", 2),
                        "partial" => ("psa_functools_partial", 2),
                        "lru_cache" => ("psa_functools_lru_cache", 1),
                        "cache" => ("psa_functools_cache", 1),
                        "cmp_to_key" => ("psa_functools_cmp_to_key", 1),
                        "total_ordering" => ("psa_functools_total_ordering", 1),
                        "singledispatch" => ("psa_functools_singledispatch", 1),
                        "wraps" => ("psa_functools_wraps", 1),
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `functools` has no function `{}` in v1 (reduce / partial / lru_cache / cache / cmp_to_key / total_ordering / singledispatch / wraps)",
                                    method
                                ),
                            ))
                        }
                    };
                    let sla_fn = if method == "reduce" && args.len() == 3 {
                        "psa_functools_reduce_init"
                    } else {
                        sla_fn
                    };
                    if args.len() != arity && !(method == "reduce" && args.len() == 3) {
                        return Err(eerr(
                            line,
                            &format!(
                                "functools.{} takes exactly {} positional argument(s) in v1",
                                method, arity
                            ),
                        ));
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    self.used_dyn = true;
                    self.used_psa_functools = true;
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std batch 3: `bisect.<fn>(...)`. Every
                // entry point takes (a, x, lo, hi) at the SLA
                // level with CPython's defaults made
                // explicit, so the intercept accepts 2..4
                // positional arguments and pads the missing
                // trailing ones (lo -> __psa_box_int(0),
                // hi -> __psa_m_none()).
                if self.std_bisect && obj == "bisect" && ctx.lookup(obj).is_none() {
                    let sla_fn: &str = match method.as_str() {
                        "bisect_left" => "psa_bisect_bisect_left",
                        "bisect_right" => "psa_bisect_bisect_right",
                        "bisect" => "psa_bisect_bisect",
                        "insort_left" => "psa_bisect_insort_left",
                        "insort_right" => "psa_bisect_insort_right",
                        "insort" => "psa_bisect_insort",
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `bisect` has no function `{}` in v1 (bisect_left / bisect_right / bisect / insort_left / insort_right / insort)",
                                    method
                                ),
                            ))
                        }
                    };
                    if args.len() < 2 || args.len() > 4 {
                        return Err(eerr(
                            line,
                            &format!(
                                "bisect.{} takes two to four positional arguments in v1 (the key= keyword is not covered)",
                                method
                            ),
                        ));
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    if acodes.len() == 2 {
                        acodes.push("__psa_box_int(0)".to_string());
                    }
                    if acodes.len() == 3 {
                        acodes.push("__psa_m_none()".to_string());
                    }
                    self.used_dyn = true;
                    self.used_psa_bisect = true;
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std batch 3: `heapq.<fn>(...)`, same
                // shape as the math intercept above.
                if self.std_heapq && obj == "heapq" && ctx.lookup(obj).is_none() {
                    let (sla_fn, arity): (&str, usize) = match method.as_str() {
                        "heappush" => ("psa_heapq_heappush", 2),
                        "heappop" => ("psa_heapq_heappop", 1),
                        "heapify" => ("psa_heapq_heapify", 1),
                        "heapreplace" => ("psa_heapq_heapreplace", 2),
                        "heappushpop" => ("psa_heapq_heappushpop", 2),
                        "nlargest" => ("psa_heapq_nlargest", 2),
                        "nsmallest" => ("psa_heapq_nsmallest", 2),
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `heapq` has no function `{}` in v1 (heappush / heappop / heapify / heapreplace / heappushpop / nlargest / nsmallest)",
                                    method
                                ),
                            ))
                        }
                    };
                    if args.len() != arity {
                        return Err(eerr(
                            line,
                            &format!(
                                "heapq.{} takes exactly {} positional argument(s) in v1 (the key= keyword is not covered)",
                                method, arity
                            ),
                        ));
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    self.used_dyn = true;
                    self.used_psa_heapq = true;
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std batch 3: `base64.<fn>(...)`, same
                // shape as the math intercept above. The
                // two-argument casefold forms of b32decode /
                // b16decode fold at emit time for a literal
                // bool argument only (the intercept returns
                // one expression and cannot branch on a
                // runtime value -- BATCH-WIRING.md R7);
                // anything else is a loud error.
                if self.std_base64 && obj == "base64" && ctx.lookup(obj).is_none() {
                    let (sla_fn, arity): (&str, usize) = match method.as_str() {
                        "b64encode" => ("psa_base64_b64encode", 1),
                        "b64decode" => ("psa_base64_b64decode", 1),
                        "standard_b64encode" => ("psa_base64_standard_b64encode", 1),
                        "standard_b64decode" => ("psa_base64_standard_b64decode", 1),
                        "urlsafe_b64encode" => ("psa_base64_urlsafe_b64encode", 1),
                        "urlsafe_b64decode" => ("psa_base64_urlsafe_b64decode", 1),
                        "b32encode" => ("psa_base64_b32encode", 1),
                        "b32decode" => ("psa_base64_b32decode", 1),
                        "b16encode" => ("psa_base64_b16encode", 1),
                        "b16decode" => ("psa_base64_b16decode", 1),
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `base64` has no function `{}` in v1 (b64encode / b64decode / standard_b64encode / standard_b64decode / urlsafe_b64encode / urlsafe_b64decode / b32encode / b32decode / b16encode / b16decode)",
                                    method
                                ),
                            ))
                        }
                    };
                    let casefold_form =
                        (method == "b32decode" || method == "b16decode") && args.len() == 2;
                    let sla_fn = if casefold_form {
                        match &args[1] {
                            Expr::Bool(true) => {
                                if method == "b32decode" {
                                    "psa_base64_b32decode_casefold"
                                } else {
                                    "psa_base64_b16decode_casefold"
                                }
                            }
                            Expr::Bool(false) => sla_fn,
                            _ => {
                                return Err(eerr(
                                    line,
                                    &format!(
                                        "base64.{} casefold argument must be a literal bool in v1",
                                        method
                                    ),
                                ))
                            }
                        }
                    } else {
                        sla_fn
                    };
                    if args.len() != arity && !casefold_form {
                        return Err(eerr(
                            line,
                            &format!(
                                "base64.{} takes exactly {} positional argument(s) in v1",
                                method, arity
                            ),
                        ));
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    let emit_n = if casefold_form { 1 } else { args.len() };
                    for a in args.iter().take(emit_n) {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    self.used_dyn = true;
                    self.used_psa_base64 = true;
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std batch 3: `random.<fn>(...)`.
                // Default padding materialises missing
                // arguments as runtime box calls, as in the
                // bisect intercept. Only the positional forms
                // of `choices` are wired: the keyword forms
                // (k= / cum_weights=) are not implementable
                // on the current parser (MethodCall carries
                // no kwargs -- BATCH-WIRING.md R1) and fail
                // at parse time; a positional None weights
                // means unweighted, as in CPython.
                if self.std_random && obj == "random" && ctx.lookup(obj).is_none() {
                    match method.as_str() {
                        "seed" | "random" | "getrandbits" | "randbytes" | "randrange"
                        | "randint" | "choice" | "shuffle" | "sample" | "choices"
                        | "uniform" | "triangular" => {}
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `random` has no function `{}` in v1 (seed / random / getrandbits / randbytes / randrange / randint / choice / shuffle / sample / choices / uniform / triangular)",
                                    method
                                ),
                            ))
                        }
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    let sla_fn: &str = match method.as_str() {
                        "seed" => {
                            if args.len() > 1 {
                                return Err(eerr(
                                    line,
                                    "random.seed takes zero or one positional argument in v1",
                                ));
                            }
                            if acodes.is_empty() {
                                acodes.push("__psa_m_none()".to_string());
                            }
                            "psa_random_seed"
                        }
                        "random" => {
                            if !args.is_empty() {
                                return Err(eerr(
                                    line,
                                    "random.random takes no arguments in v1",
                                ));
                            }
                            "psa_random_random"
                        }
                        "getrandbits" | "randbytes" | "choice" | "shuffle" => {
                            if args.len() != 1 {
                                return Err(eerr(
                                    line,
                                    &format!(
                                        "random.{} takes exactly one positional argument in v1",
                                        method
                                    ),
                                ));
                            }
                            match method.as_str() {
                                "getrandbits" => "psa_random_getrandbits",
                                "randbytes" => "psa_random_randbytes",
                                "choice" => "psa_random_choice",
                                _ => "psa_random_shuffle",
                            }
                        }
                        "randint" | "sample" | "uniform" => {
                            if args.len() != 2 {
                                return Err(eerr(
                                    line,
                                    &format!(
                                        "random.{} takes exactly two positional arguments in v1",
                                        method
                                    ),
                                ));
                            }
                            match method.as_str() {
                                "randint" => "psa_random_randint",
                                "sample" => "psa_random_sample",
                                _ => "psa_random_uniform",
                            }
                        }
                        "randrange" => match args.len() {
                            1 => "psa_random_randrange1",
                            2 => "psa_random_randrange2",
                            3 => "psa_random_randrange3",
                            _ => {
                                return Err(eerr(
                                    line,
                                    "random.randrange takes one to three positional arguments in v1",
                                ))
                            }
                        },
                        "choices" => match args.len() {
                            1 => {
                                acodes.push("__psa_box_int(1)".to_string());
                                "psa_random_choices_k"
                            }
                            2 => {
                                if matches!(args[1], Expr::NoneLit) {
                                    acodes.truncate(1);
                                    acodes.push("__psa_box_int(1)".to_string());
                                    "psa_random_choices_k"
                                } else {
                                    acodes.push("__psa_box_int(1)".to_string());
                                    "psa_random_choices_w"
                                }
                            }
                            3 => {
                                if matches!(args[1], Expr::NoneLit) {
                                    let kcode = acodes.pop().unwrap();
                                    acodes.truncate(1);
                                    acodes.push(kcode);
                                    "psa_random_choices_k"
                                } else {
                                    "psa_random_choices_w"
                                }
                            }
                            _ => {
                                return Err(eerr(
                                    line,
                                    "random.choices takes one to three positional arguments in v1 (the k= / cum_weights= keyword forms are not covered)",
                                ))
                            }
                        },
                        "triangular" => {
                            if args.len() > 3 {
                                return Err(eerr(
                                    line,
                                    "random.triangular takes zero to three positional arguments in v1",
                                ));
                            }
                            let pads = [
                                "__psa_box_float(0.0)",
                                "__psa_box_float(1.0)",
                                "__psa_m_none()",
                            ];
                            while acodes.len() < 3 {
                                let p = pads[acodes.len()];
                                acodes.push(p.to_string());
                            }
                            "psa_random_triangular"
                        }
                        _ => unreachable!(),
                    };
                    self.used_dyn = true;
                    self.used_psa_random = true;
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std batch 3: `re.<fn>(...)`. CPython's
                // flags argument defaults to 0; the intercept
                // implements the defaulting by appending a
                // boxed int 0 to the shorter forms (the
                // decision WIRING-re.md leaves to the batch,
                // recorded in BATCH-WIRING.md). The flag
                // constants are zero-argument calls, as
                // std-module attribute reads are not wired.
                // There is deliberately no `compile` entry:
                // it hits the unknown-name error.
                if self.std_re && obj == "re" && ctx.lookup(obj).is_none() {
                    let (sla_fn, arity): (&str, usize) = match method.as_str() {
                        "match" => ("psa_re_match", 3),
                        "search" => ("psa_re_search", 3),
                        "fullmatch" => ("psa_re_fullmatch", 3),
                        "findall" => ("psa_re_findall", 3),
                        "split" => ("psa_re_split", 3),
                        "split_maxsplit" => ("psa_re_split_maxsplit", 4),
                        "sub" => ("psa_re_sub", 4),
                        "sub_count" => ("psa_re_sub_count", 5),
                        "subn" => ("psa_re_subn", 4),
                        "subn_count" => ("psa_re_subn_count", 5),
                        "escape" => ("psa_re_escape", 1),
                        "IGNORECASE" => ("psa_re_ignorecase", 0),
                        "MULTILINE" => ("psa_re_multiline", 0),
                        "DOTALL" => ("psa_re_dotall", 0),
                        "ASCII" => ("psa_re_ascii", 0),
                        "VERBOSE" => ("psa_re_verbose", 0),
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `re` has no function `{}` in v1 (match / search / fullmatch / findall / split / split_maxsplit / sub / sub_count / subn / subn_count / escape / IGNORECASE / MULTILINE / DOTALL / ASCII / VERBOSE; compile is not covered)",
                                    method
                                ),
                            ))
                        }
                    };
                    let defaultable = matches!(
                        method.as_str(),
                        "match" | "search" | "fullmatch" | "findall" | "split" | "sub"
                            | "subn"
                    );
                    if args.len() != arity && !(defaultable && args.len() == arity - 1) {
                        return Err(eerr(
                            line,
                            &format!(
                                "re.{} takes {} positional argument(s) in v1{}",
                                method,
                                arity,
                                if defaultable {
                                    " (flags may be omitted)"
                                } else {
                                    ""
                                }
                            ),
                        ));
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    if defaultable && acodes.len() == arity - 1 {
                        acodes.push("__psa_box_int(0)".to_string());
                    }
                    self.used_dyn = true;
                    self.used_psa_re = true;
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std net group: `socket.<fn>(...)`.
                // Module-level functions only; method calls
                // on socket objects held in variables await
                // std-object tracking (GAP-net.md N-1). The
                // constants are zero-argument calls, as
                // std-module attribute reads are not wired
                // (N-2). `socket.socket(family)` pads
                // SOCK_STREAM (boxed int 1) per
                // WIRING-socket.md.
                if self.std_socket && obj == "socket" && ctx.lookup(obj).is_none() {
                    match method.as_str() {
                        "socket" | "create_connection" | "gethostname" | "gethostbyname"
                        | "gethostbyname_ex" | "gethostbyaddr" | "getaddrinfo"
                        | "getnameinfo" | "socketpair" | "fromfd" | "inet_aton"
                        | "inet_ntoa" | "inet_pton" | "inet_ntop" | "AF_INET" | "AF_INET6"
                        | "SOCK_STREAM" | "SOCK_DGRAM" | "SOL_SOCKET" => {}
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `socket` has no function `{}` in v1 (socket / create_connection / gethostname / gethostbyname / gethostbyname_ex / gethostbyaddr / getaddrinfo / getnameinfo / socketpair / fromfd / inet_aton / inet_ntoa / inet_pton / inet_ntop / AF_INET / AF_INET6 / SOCK_STREAM / SOCK_DGRAM / SOL_SOCKET; object methods such as connect / send / recv are not covered)",
                                    method
                                ),
                            ))
                        }
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    let sla_fn: &str = match method.as_str() {
                        "socket" => match args.len() {
                            0 => "psa_socket_new_default",
                            1 => {
                                acodes.push("__psa_box_int(1)".to_string());
                                "psa_socket_new"
                            }
                            2 => "psa_socket_new",
                            _ => {
                                return Err(eerr(
                                    line,
                                    "socket.socket takes zero to two positional arguments in v1",
                                ))
                            }
                        },
                        "create_connection" => match args.len() {
                            1 => "psa_socket_create_connection_addr",
                            2 => "psa_socket_create_connection_addr_timeout",
                            _ => {
                                return Err(eerr(
                                    line,
                                    "socket.create_connection takes one or two positional arguments in v1",
                                ))
                            }
                        },
                        "gethostname" | "socketpair" | "AF_INET" | "AF_INET6" | "SOCK_STREAM"
                        | "SOCK_DGRAM" | "SOL_SOCKET" => {
                            if !args.is_empty() {
                                return Err(eerr(
                                    line,
                                    &format!("socket.{} takes no arguments in v1", method),
                                ));
                            }
                            match method.as_str() {
                                "gethostname" => "psa_socket_gethostname",
                                "socketpair" => "psa_socket_socketpair",
                                "AF_INET" => "psa_socket_af_inet",
                                "AF_INET6" => "psa_socket_af_inet6",
                                "SOCK_STREAM" => "psa_socket_sock_stream",
                                "SOCK_DGRAM" => "psa_socket_sock_dgram",
                                _ => "psa_socket_sol_socket",
                            }
                        }
                        "gethostbyname" | "gethostbyname_ex" | "gethostbyaddr" | "fromfd"
                        | "inet_aton" | "inet_ntoa" => {
                            if args.len() != 1 {
                                return Err(eerr(
                                    line,
                                    &format!(
                                        "socket.{} takes exactly one positional argument in v1",
                                        method
                                    ),
                                ));
                            }
                            match method.as_str() {
                                "gethostbyname" => "psa_socket_gethostbyname",
                                "gethostbyname_ex" => "psa_socket_gethostbyname_ex",
                                "gethostbyaddr" => "psa_socket_gethostbyaddr",
                                "fromfd" => "psa_socket_fromfd",
                                "inet_aton" => "psa_socket_inet_aton",
                                _ => "psa_socket_inet_ntoa",
                            }
                        }
                        "getaddrinfo" | "getnameinfo" | "inet_pton" | "inet_ntop" => {
                            if args.len() != 2 {
                                return Err(eerr(
                                    line,
                                    &format!(
                                        "socket.{} takes exactly two positional arguments in v1",
                                        method
                                    ),
                                ));
                            }
                            match method.as_str() {
                                "getaddrinfo" => "psa_socket_getaddrinfo",
                                "getnameinfo" => "psa_socket_getnameinfo",
                                "inet_pton" => "psa_socket_inet_pton",
                                _ => "psa_socket_inet_ntop",
                            }
                        }
                        _ => unreachable!(),
                    };
                    self.used_dyn = true;
                    self.used_psa_socket = true;
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // psa_std net group: `urllib.<fn>(...)`.
                // urllib.parse and urllib.request are
                // flattened into this one module
                // (WIRING-urllib.md); the submodule
                // spellings are not wired. hostname / port
                // are spelled hostname_of / port_of, as
                // std-module attribute reads are not wired.
                // Response and Request object methods
                // await std-object tracking (GAP-net.md
                // N-1); the function forms live in
                // urllib.sla and are harness-verified.
                if self.std_urllib && obj == "urllib" && ctx.lookup(obj).is_none() {
                    match method.as_str() {
                        "urlsplit" | "urlunsplit" | "urljoin" | "quote" | "quote_plus"
                        | "unquote" | "unquote_plus" | "urlencode" | "parse_qs" | "parse_qsl"
                        | "hostname_of" | "port_of" | "urlopen" | "Request" => {}
                        _ => {
                            return Err(eerr(
                                line,
                                &format!(
                                    "psa_std module `urllib` has no function `{}` in v1 (urlsplit / urlunsplit / urljoin / quote / quote_plus / unquote / unquote_plus / urlencode / parse_qs / parse_qsl / hostname_of / port_of / urlopen / Request; object methods such as read / getcode / getheader are not covered)",
                                    method
                                ),
                            ))
                        }
                    }
                    let mut acodes: Vec<String> = Vec::new();
                    for a in args {
                        acodes.push(self.emit_dyn_value(ctx, a, line)?);
                    }
                    let sla_fn: &str = match method.as_str() {
                        "urlsplit" | "urlunsplit" | "unquote" | "unquote_plus" | "hostname_of"
                        | "port_of" => {
                            if args.len() != 1 {
                                return Err(eerr(
                                    line,
                                    &format!(
                                        "urllib.{} takes exactly one positional argument in v1",
                                        method
                                    ),
                                ));
                            }
                            match method.as_str() {
                                "urlsplit" => "psa_urllib_parse_urlsplit",
                                "urlunsplit" => "psa_urllib_parse_urlunsplit",
                                "unquote" => "psa_urllib_parse_unquote",
                                "unquote_plus" => "psa_urllib_parse_unquote_plus",
                                "hostname_of" => "psa_urllib_parse_hostname",
                                _ => "psa_urllib_parse_port",
                            }
                        }
                        "urljoin" => {
                            if args.len() != 2 {
                                return Err(eerr(
                                    line,
                                    "urllib.urljoin takes exactly two positional arguments in v1",
                                ));
                            }
                            "psa_urllib_parse_urljoin"
                        }
                        "quote" => match args.len() {
                            1 => "psa_urllib_parse_quote",
                            2 => "psa_urllib_parse_quote_safe",
                            _ => {
                                return Err(eerr(
                                    line,
                                    "urllib.quote takes one or two positional arguments in v1",
                                ))
                            }
                        },
                        "quote_plus" => match args.len() {
                            1 => "psa_urllib_parse_quote_plus",
                            2 => "psa_urllib_parse_quote_plus_safe",
                            _ => {
                                return Err(eerr(
                                    line,
                                    "urllib.quote_plus takes one or two positional arguments in v1",
                                ))
                            }
                        },
                        "urlencode" => match args.len() {
                            1 => "psa_urllib_parse_urlencode",
                            2 => "psa_urllib_parse_urlencode_doseq",
                            _ => {
                                return Err(eerr(
                                    line,
                                    "urllib.urlencode takes one or two positional arguments in v1",
                                ))
                            }
                        },
                        "parse_qs" => match args.len() {
                            1 => "psa_urllib_parse_parse_qs",
                            2 => "psa_urllib_parse_parse_qs_kb",
                            _ => {
                                return Err(eerr(
                                    line,
                                    "urllib.parse_qs takes one or two positional arguments in v1",
                                ))
                            }
                        },
                        "parse_qsl" => match args.len() {
                            1 => "psa_urllib_parse_parse_qsl",
                            2 => "psa_urllib_parse_parse_qsl_kb",
                            _ => {
                                return Err(eerr(
                                    line,
                                    "urllib.parse_qsl takes one or two positional arguments in v1",
                                ))
                            }
                        },
                        "urlopen" => match args.len() {
                            1 => "psa_urllib_urlopen",
                            2 => "psa_urllib_urlopen_data",
                            3 => "psa_urllib_urlopen_full",
                            _ => {
                                return Err(eerr(
                                    line,
                                    "urllib.urlopen takes one to three positional arguments in v1",
                                ))
                            }
                        },
                        "Request" => match args.len() {
                            1 => "psa_urllib_request_new",
                            2 => "psa_urllib_request_new_data",
                            3 => {
                                acodes.push("__psa_m_none()".to_string());
                                "psa_urllib_request_new_full"
                            }
                            4 => "psa_urllib_request_new_full",
                            _ => {
                                return Err(eerr(
                                    line,
                                    "urllib.Request takes one to four positional arguments in v1",
                                ))
                            }
                        },
                        _ => unreachable!(),
                    };
                    self.used_dyn = true;
                    self.used_psa_urllib = true;
                    return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                }
                // D6: a method call on an imported module:
                // `foo.bar(...)` is a call of the module def
                // `foo.bar`; `foo.Klass(...)` constructs a
                // module class.
                if ctx.lookup(obj).is_none() && self.modules.iter().any(|m| m == obj) {
                    let key = format!("{}.{}", obj, method);
                    if self.defs.contains_key(&key) {
                        let call =
                            Expr::Call(Box::new(Expr::Name(key, ExprSpan::synthetic())), args.clone(), Vec::new(), ExprSpan::synthetic());
                        return self.emit_expr(ctx, &call, line);
                    }
                    if let Some(&cid) = self.class_ids.get(method) {
                        return self.emit_construct(ctx, method, cid, args, &[], line);
                    }
                }
                let ty = ctx
                    .lookup(obj)
                    .ok_or_else(|| eerr(line, &format!("undefined variable `{}`", obj)))?;
                // Method calls on a dynamic value (D4/D5).
                if ty == Ty::Dyn {
                    ctx.check_alive(obj, line, None)?;
                    self.used_dyn = true;
                    // append keeps a direct fast path (no
                    // argument packing); the runtime defers
                    // objects to their own append method.
                    if method == "append" {
                        if args.len() != 1 {
                            return Err(eerr(line, "append takes exactly one argument"));
                        }
                        let (mut acode, aty) = if let Expr::Str(s) = &args[0] {
                            (self.emit_str_temp(ctx, s, line)?, Ty::Dyn)
                        } else if matches!(&args[0], Expr::ListLit(_) | Expr::DictLit(_)) {
                            (self.emit_container_temp(ctx, &args[0], line)?, Ty::Dyn)
                        } else {
                            self.emit_expr(ctx, &args[0], line)?
                        };
                        if aty != Ty::Dyn {
                            acode = self.box_code(acode, aty, line)?;
                        }
                        let ntmp = self.emit_str_temp(ctx, method, line)?;
                        return Ok((
                            format!("__psa_append_fast({}, {}, {})", obj, ntmp, acode),
                            Ty::Dyn,
                        ));
                    }
                    // D5 builtin method set (the runtime defers
                    // objects to getattr, checks kinds and arity).
                    if let Some(code) =
                        self.emit_builtin_method(ctx, obj, method, args, line)?
                    {
                        return Ok((code, Ty::Dyn));
                    }
                    // D4: object methods go through getattr
                    // (which binds the method) and the callable
                    // dispatch.
                    let ntmp = self.emit_str_temp(ctx, method, line)?;
                    let codes = self.emit_dyn_args(ctx, args, line)?;
                    let tmp = self.pack_dyn_args(ctx, &codes);
                    return Ok((
                        format!("__psa_call_value(__psa_getattr({}, {}), {})", obj, ntmp, tmp),
                        Ty::Dyn,
                    ));
                }
                if method != "append" {
                    return Err(eerr(
                        line,
                        &format!(
                            "method `{}` is not supported on a {} value (the D5 method set lives on dynamic values; only list.append is static)",
                            method,
                            ty.py_name()
                        ),
                    ));
                }
                ctx.check_alive(obj, line, None)?;
                if args.len() != 1 {
                    return Err(eerr(line, "append takes exactly one argument"));
                }
                let elem = match ty.elem() {
                    Some(e) => e,
                    _ => {
                        return Err(eerr(
                            line,
                            &format!("append is only supported on lists, not on {}", ty.py_name()),
                        ))
                    }
                };
                let (mut acode, aty) = self.emit_expr(ctx, &args[0], line)?;
                if aty == Ty::Dyn {
                    acode = self.unbox_code(acode, elem, line)?;
                } else if aty != elem {
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
                if ty == Ty::Dyn {
                    self.used_dyn = true;
                    return match op {
                        UnOp::Neg => Ok((format!("__psa_dyn_neg({})", code), Ty::Dyn)),
                        UnOp::Not => {
                            Ok((format!("!(__psa_dyn_truthy({}))", code), Ty::Bool))
                        }
                        UnOp::BitNot => {
                            Ok((format!("__psa_dyn_bitnot({})", code), Ty::Dyn))
                        }
                    };
                }
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
            Expr::Call(callee, args, kwargs, _) => {
                // P1 source map: note the call boundary. The next
                // `Ctx::line` whose text contains the callee's call
                // becomes a `CallSite` segment (see `Ctx::line`).
                // First call wins when calls nest on one line; the
                // name is the callee's generated symbol when it is
                // a plain name / function value, and `None` for a
                // dynamic callee (method calls: the dispatch is
                // `__psa_call_value`, matched by the consumer).
                // psa_std module entry points (`json.dumps`, ...)
                // are deliberately NOT named here: their sla entry
                // names are decided inside the intercept blocks
                // below, and the call line is still covered by its
                // ordinary User segment — see RESULT.md.
                match &**callee {
                    Expr::Name(n, _) | Expr::FnVal(n) => ctx.note_call(Some(self.sla_name(n))),
                    Expr::Attr(..) => ctx.note_call(None),
                    _ => {}
                }
                // psa_std: `json.dumps(v)` / `json.loads(s)`
                // with `import json` bound to the standard
                // library call the spliced module's SLA entry
                // points directly (one dynamic argument each).
                if self.std_json {
                    if let Expr::Attr(base, mname, _) = &**callee {
                        if let Expr::Name(n, _) = &**base {
                            if n == "json" && ctx.lookup(n).is_none() {
                                let sla_fn = match mname.as_str() {
                                    "dumps" => "psa_json_dumps",
                                    "loads" => "psa_json_parse",
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `json` has no function `{}` in v1 (dumps / loads)",
                                                mname
                                            ),
                                        ))
                                    }
                                };
                                if args.len() != 1 || !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "json.{} takes exactly one positional argument in v1",
                                            mname
                                        ),
                                    ));
                                }
                                let acode = self.emit_dyn_value(ctx, &args[0], line)?;
                                self.used_dyn = true;
                                self.used_psa_json = true;
                                return Ok((format!("{}({})", sla_fn, acode), Ty::Dyn));
                            }
                        }
                    }
                }
                // psa_std: the Call(Attr) form of the pickle
                // intercept, mirroring the MethodCall block (same
                // entry points and arities, keywords refused).
                if self.std_pickle {
                    if let Expr::Attr(base, mname, _) = &**callee {
                        if let Expr::Name(n, _) = &**base {
                            if n == "pickle" && ctx.lookup(n).is_none() {
                                let (sla_fn, arity): (&str, usize) = match mname.as_str() {
                                    "dumps" => ("psa_pickle_dumps", 1),
                                    "dump" => ("psa_pickle_dump", 2),
                                    "loads" => ("psa_pickle_loads", 1),
                                    "load" => ("psa_pickle_load", 1),
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `pickle` has no function `{}` in v1 (dumps / dump / loads / load)",
                                                mname
                                            ),
                                        ))
                                    }
                                };
                                if args.len() != arity || !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "pickle.{} takes exactly {} positional argument(s) in v1",
                                            mname, arity
                                        ),
                                    ));
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                self.used_dyn = true;
                                self.used_psa_pickle = true;
                                // Monolithic splice: see the
                                // MethodCall intercept — the file
                                // runtime is always required.
                                self.used_file = true;
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                        }
                    }
                }
                // psa_std L1 batch 2: the Call(Attr) form of
                // the math / string intercepts, mirroring the
                // json block above (same entry points, same
                // arity rules, keywords refused).
                if self.std_math || self.std_string {
                    if let Expr::Attr(base, mname, _) = &**callee {
                        if let Expr::Name(n, _) = &**base {
                            if n == "math" && self.std_math && ctx.lookup(n).is_none() {
                                let (sla_fn, arity): (&str, usize) = match mname.as_str() {
                                    "floor" => ("psa_math_floor", 1),
                                    "ceil" => ("psa_math_ceil", 1),
                                    "trunc" => ("psa_math_trunc", 1),
                                    "fabs" => ("psa_math_fabs", 1),
                                    "sqrt" => ("psa_math_sqrt", 1),
                                    "sin" => ("psa_math_sin", 1),
                                    "cos" => ("psa_math_cos", 1),
                                    "degrees" => ("psa_math_degrees", 1),
                                    "radians" => ("psa_math_radians", 1),
                                    "factorial" => ("psa_math_factorial", 1),
                                    "isqrt" => ("psa_math_isqrt", 1),
                                    "gcd" => ("psa_math_gcd", 2),
                                    "lcm" => ("psa_math_lcm", 2),
                                    "comb" => ("psa_math_comb", 2),
                                    "perm" => ("psa_math_perm", 2),
                                    "pow" => ("psa_math_pow", 2),
                                    "isclose" => ("psa_math_isclose", 2),
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `math` has no function `{}` in v1 (floor / ceil / trunc / fabs / sqrt / sin / cos / degrees / radians / factorial / isqrt / gcd / lcm / comb / perm / pow / isclose)",
                                                mname
                                            ),
                                        ))
                                    }
                                };
                                if args.len() != arity || !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "math.{} takes exactly {} positional argument(s) in v1",
                                            mname, arity
                                        ),
                                    ));
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                self.used_dyn = true;
                                self.used_psa_math = true;
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                            if n == "string" && self.std_string && ctx.lookup(n).is_none() {
                                if mname != "capwords" {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "psa_std module `string` has no function `{}` in v1 (capwords; the constants are values, not functions)",
                                            mname
                                        ),
                                    ));
                                }
                                if (args.len() != 1 && args.len() != 2) || !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        "string.capwords takes one or two positional arguments in v1",
                                    ));
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                self.used_dyn = true;
                                self.used_psa_string = true;
                                let sla_fn = if args.len() == 2 {
                                    "psa_string_capwords_sep"
                                } else {
                                    "psa_string_capwords"
                                };
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                        }
                    }
                }
                // psa_std batch 3: the Call(Attr) forms of
                // the csv / logging / argparse intercepts,
                // mirroring the MethodCall blocks above (same
                // entry points, same arity rules, keywords
                // refused).
                if self.std_csv || self.std_logging || self.std_argparse {
                    if let Expr::Attr(base, mname, _) = &**callee {
                        if let Expr::Name(n, _) = &**base {
                            if n == "csv" && self.std_csv && ctx.lookup(n).is_none() {
                                let (sla_fn, arity): (&str, usize) = match mname.as_str() {
                                    "reads" => ("psa_csv_reads", 1),
                                    "reads_d" => ("psa_csv_reads_d", 2),
                                    "reader" => ("psa_csv_reader", 1),
                                    "reader_d" => ("psa_csv_reader_d", 2),
                                    "writes" => ("psa_csv_writes", 1),
                                    "writes_d" => ("psa_csv_writes_d", 2),
                                    "write_rows" => ("psa_csv_write_rows", 3),
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `csv` has no function `{}` in v1 (reads / reads_d / reader / reader_d / writes / writes_d / write_rows)",
                                                mname
                                            ),
                                        ))
                                    }
                                };
                                if args.len() != arity || !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "csv.{} takes exactly {} positional argument(s) in v1",
                                            mname, arity
                                        ),
                                    ));
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                self.used_dyn = true;
                                self.used_psa_csv = true;
                                if mname == "write_rows" {
                                    self.used_file = true;
                                }
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                            if n == "logging" && self.std_logging && ctx.lookup(n).is_none() {
                                if !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "logging.{} takes positional arguments only in v1 (the Call form refuses keywords)",
                                            mname
                                        ),
                                    ));
                                }
                                let root_fn: Option<&str> = match mname.as_str() {
                                    "debug" => Some("psa_logging_root_debug"),
                                    "info" => Some("psa_logging_root_info"),
                                    "warning" => Some("psa_logging_root_warning"),
                                    "error" => Some("psa_logging_root_error"),
                                    "critical" => Some("psa_logging_root_critical"),
                                    _ => None,
                                };
                                if let Some(sla_fn) = root_fn {
                                    if args.is_empty() {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "logging.{} takes at least one positional argument (the message) in v1",
                                                mname
                                            ),
                                        ));
                                    }
                                    let msgcode = self.emit_dyn_value(ctx, &args[0], line)?;
                                    let argscode = if args.len() == 1 {
                                        "__psa_m_none()".to_string()
                                    } else {
                                        self.emit_dyn_value(
                                            ctx,
                                            &Expr::ListLit(args[1..].to_vec()),
                                            line,
                                        )?
                                    };
                                    self.used_dyn = true;
                                    self.used_psa_logging = true;
                                    return Ok((
                                        format!("{}({}, {})", sla_fn, msgcode, argscode),
                                        Ty::Dyn,
                                    ));
                                }
                                if mname == "log" {
                                    if args.len() < 2 {
                                        return Err(eerr(
                                            line,
                                            "logging.log takes at least two positional arguments (level, message) in v1",
                                        ));
                                    }
                                    let levelcode = self.emit_dyn_value(ctx, &args[0], line)?;
                                    let msgcode = self.emit_dyn_value(ctx, &args[1], line)?;
                                    let argscode = if args.len() == 2 {
                                        "__psa_m_none()".to_string()
                                    } else {
                                        self.emit_dyn_value(
                                            ctx,
                                            &Expr::ListLit(args[2..].to_vec()),
                                            line,
                                        )?
                                    };
                                    self.used_dyn = true;
                                    self.used_psa_logging = true;
                                    return Ok((
                                        format!(
                                            "psa_logging_root_log({}, {}, {})",
                                            levelcode, msgcode, argscode
                                        ),
                                        Ty::Dyn,
                                    ));
                                }
                                let (sla_fn, arity): (&str, usize) = match mname.as_str() {
                                    "getLogger" => ("psa_logging_get_logger", 1),
                                    "set_level" => ("psa_logging_set_level", 2),
                                    "get_effective_level" => {
                                        ("psa_logging_get_effective_level", 1)
                                    }
                                    "basicConfig" => ("psa_logging_basic_config", 2),
                                    "basicConfig_force" => {
                                        ("psa_logging_basic_config_force", 2)
                                    }
                                    "level_debug" => ("psa_logging_level_debug", 0),
                                    "level_info" => ("psa_logging_level_info", 0),
                                    "level_warning" => ("psa_logging_level_warning", 0),
                                    "level_error" => ("psa_logging_level_error", 0),
                                    "level_critical" => ("psa_logging_level_critical", 0),
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `logging` has no function `{}` in v1 (getLogger / set_level / get_effective_level / basicConfig / basicConfig_force / log / debug / info / warning / error / critical / level_debug / level_info / level_warning / level_error / level_critical)",
                                                mname
                                            ),
                                        ))
                                    }
                                };
                                if args.len() != arity {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "logging.{} takes exactly {} positional argument(s) in v1",
                                            mname, arity
                                        ),
                                    ));
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                self.used_dyn = true;
                                self.used_psa_logging = true;
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                            if n == "argparse" && self.std_argparse && ctx.lookup(n).is_none() {
                                if mname != "ArgumentParser" {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "psa_std module `argparse` has no function `{}` in v1 (ArgumentParser)",
                                            mname
                                        ),
                                    ));
                                }
                                if (args.len() != 1 && args.len() != 2) || !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        "argparse.ArgumentParser takes one or two positional arguments (prog, description) in v1",
                                    ));
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                if acodes.len() == 1 {
                                    acodes.push("__psa_m_none()".to_string());
                                }
                                self.used_dyn = true;
                                self.used_psa_argparse = true;
                                return Ok((
                                    format!("psa_argparse_new_parser({})", acodes.join(", ")),
                                    Ty::Dyn,
                                ));
                            }
                        }
                    }
                }
                // psa_std batch 3: the Call(Attr) forms of
                // the collections / itertools / functools
                // intercepts, mirroring the MethodCall blocks
                // above (keywords refused).
                if self.std_collections || self.std_itertools || self.std_functools {
                    if let Expr::Attr(base, mname, _) = &**callee {
                        if let Expr::Name(n, _) = &**base {
                            if n == "collections" && self.std_collections && ctx.lookup(n).is_none() {
                                let (sla_fn, arity): (&str, usize) = match mname.as_str() {
                                    "Counter" => ("psa_collections_counter_any", 1),
                                    "defaultdict" => ("psa_collections_defaultdict_new", 1),
                                    "deque" => ("psa_collections_deque_new", 1),
                                    "ChainMap" => ("psa_collections_chainmap_new", 1),
                                    "OrderedDict" => ("psa_collections_ordereddict_any", 1),
                                    "namedtuple" => ("psa_collections_namedtuple", 2),
                                    "counter_dict" => ("psa_collections_counter_dict", 1),
                                    "counter_get" => ("psa_collections_counter_get", 2),
                                    "counter_update" => ("psa_collections_counter_update", 2),
                                    "counter_subtract" => {
                                        ("psa_collections_counter_subtract", 2)
                                    }
                                    "counter_add" => ("psa_collections_counter_add", 2),
                                    "counter_sub" => ("psa_collections_counter_sub", 2),
                                    "counter_and" => ("psa_collections_counter_and", 2),
                                    "counter_or" => ("psa_collections_counter_or", 2),
                                    "counter_elements" => {
                                        ("psa_collections_counter_elements", 1)
                                    }
                                    "counter_total" => ("psa_collections_counter_total", 1),
                                    "counter_most_common" => {
                                        ("psa_collections_counter_most_common", 1)
                                    }
                                    "counter_most_common_n" => {
                                        ("psa_collections_counter_most_common_n", 2)
                                    }
                                    "defaultdict_get" => {
                                        ("psa_collections_defaultdict_get", 3)
                                    }
                                    "deque_append" => ("psa_collections_deque_append", 2),
                                    "deque_append_maxlen" => {
                                        ("psa_collections_deque_append_maxlen", 3)
                                    }
                                    "deque_appendleft" => {
                                        ("psa_collections_deque_appendleft", 2)
                                    }
                                    "deque_appendleft_maxlen" => {
                                        ("psa_collections_deque_appendleft_maxlen", 3)
                                    }
                                    "deque_pop" => ("psa_collections_deque_pop", 1),
                                    "deque_popleft" => ("psa_collections_deque_popleft", 1),
                                    "deque_extend" => ("psa_collections_deque_extend", 2),
                                    "deque_extendleft" => {
                                        ("psa_collections_deque_extendleft", 2)
                                    }
                                    "deque_rotate" => ("psa_collections_deque_rotate", 2),
                                    "deque_count" => ("psa_collections_deque_count", 2),
                                    "deque_remove" => ("psa_collections_deque_remove", 2),
                                    "deque_clear" => ("psa_collections_deque_clear", 1),
                                    "chainmap_get" => ("psa_collections_chainmap_get", 2),
                                    "chainmap_get_default" => {
                                        ("psa_collections_chainmap_get_default", 3)
                                    }
                                    "chainmap_set" => ("psa_collections_chainmap_set", 3),
                                    "chainmap_del" => ("psa_collections_chainmap_del", 2),
                                    "chainmap_keys" => ("psa_collections_chainmap_keys", 1),
                                    "chainmap_new_child" => {
                                        ("psa_collections_chainmap_new_child", 2)
                                    }
                                    "chainmap_parents" => {
                                        ("psa_collections_chainmap_parents", 1)
                                    }
                                    "ordereddict_move_to_end" => {
                                        ("psa_collections_ordereddict_move_to_end", 2)
                                    }
                                    "ordereddict_move_to_end_first" => {
                                        ("psa_collections_ordereddict_move_to_end_first", 2)
                                    }
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `collections` has no function `{}` in v1 (Counter / defaultdict / deque / ChainMap / OrderedDict / namedtuple / counter_* / defaultdict_get / deque_* / chainmap_* / ordereddict_move_to_end*)",
                                                mname
                                            ),
                                        ))
                                    }
                                };
                                let (sla_fn, arity) = if mname == "deque" && args.is_empty() {
                                    ("psa_collections_deque_new_empty", 0)
                                } else {
                                    (sla_fn, arity)
                                };
                                if args.len() != arity || !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "collections.{} takes exactly {} positional argument(s) in v1",
                                            mname, arity
                                        ),
                                    ));
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                self.used_dyn = true;
                                self.used_psa_collections = true;
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                            if n == "itertools" && self.std_itertools && ctx.lookup(n).is_none() {
                                let (sla_fn, arity): (&str, usize) = match mname.as_str() {
                                    "chain" => ("psa_itertools_chain", 2),
                                    "chain_from_iterable" => {
                                        ("psa_itertools_chain_from_iterable", 1)
                                    }
                                    "islice" => ("psa_itertools_islice_stop", 2),
                                    "repeat" => ("psa_itertools_repeat", 2),
                                    "accumulate" => ("psa_itertools_accumulate", 1),
                                    "accumulate_init" => ("psa_itertools_accumulate_init", 2),
                                    "takewhile" => ("psa_itertools_takewhile", 2),
                                    "dropwhile" => ("psa_itertools_dropwhile", 2),
                                    "compress" => ("psa_itertools_compress", 2),
                                    "starmap" => ("psa_itertools_starmap", 2),
                                    "tee" => ("psa_itertools_tee2", 1),
                                    "pairwise" => ("psa_itertools_pairwise", 1),
                                    "zip_longest" => ("psa_itertools_zip_longest", 3),
                                    "groupby" => ("psa_itertools_groupby", 1),
                                    "product" => ("psa_itertools_product", 2),
                                    "product_from" => ("psa_itertools_product_from", 1),
                                    "permutations" => ("psa_itertools_permutations_full", 1),
                                    "combinations" => ("psa_itertools_combinations", 2),
                                    "combinations_with_replacement" => {
                                        ("psa_itertools_combinations_with_replacement", 2)
                                    }
                                    "count" => ("psa_itertools_count", 2),
                                    "cycle" => ("psa_itertools_cycle", 1),
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `itertools` has no function `{}` in v1 (chain / chain_from_iterable / islice / repeat / accumulate / accumulate_init / takewhile / dropwhile / compress / starmap / tee / pairwise / zip_longest / groupby / product / product_from / permutations / combinations / combinations_with_replacement / count / cycle)",
                                                mname
                                            ),
                                        ))
                                    }
                                };
                                let sla_fn = match (mname.as_str(), args.len()) {
                                    ("islice", 3) => "psa_itertools_islice_start_stop",
                                    ("islice", 4) => "psa_itertools_islice_start_stop_step",
                                    ("accumulate", 2) => "psa_itertools_accumulate_fn",
                                    ("groupby", 2) => "psa_itertools_groupby_key",
                                    ("permutations", 2) => "psa_itertools_permutations",
                                    _ => sla_fn,
                                };
                                let arity_ok = match mname.as_str() {
                                    "islice" => {
                                        args.len() == 2 || args.len() == 3 || args.len() == 4
                                    }
                                    "accumulate" | "groupby" | "permutations" => {
                                        args.len() == 1 || args.len() == 2
                                    }
                                    _ => args.len() == arity,
                                };
                                if !arity_ok || !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "itertools.{} does not take {} positional argument(s) in v1",
                                            mname,
                                            args.len()
                                        ),
                                    ));
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                self.used_dyn = true;
                                self.used_psa_itertools = true;
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                            if n == "functools" && self.std_functools && ctx.lookup(n).is_none() {
                                let (sla_fn, arity): (&str, usize) = match mname.as_str() {
                                    "reduce" => ("psa_functools_reduce", 2),
                                    "partial" => ("psa_functools_partial", 2),
                                    "lru_cache" => ("psa_functools_lru_cache", 1),
                                    "cache" => ("psa_functools_cache", 1),
                                    "cmp_to_key" => ("psa_functools_cmp_to_key", 1),
                                    "total_ordering" => ("psa_functools_total_ordering", 1),
                                    "singledispatch" => ("psa_functools_singledispatch", 1),
                                    "wraps" => ("psa_functools_wraps", 1),
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `functools` has no function `{}` in v1 (reduce / partial / lru_cache / cache / cmp_to_key / total_ordering / singledispatch / wraps)",
                                                mname
                                            ),
                                        ))
                                    }
                                };
                                let sla_fn = if mname == "reduce" && args.len() == 3 {
                                    "psa_functools_reduce_init"
                                } else {
                                    sla_fn
                                };
                                if (args.len() != arity
                                    && !(mname == "reduce" && args.len() == 3))
                                    || !kwargs.is_empty()
                                {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "functools.{} takes exactly {} positional argument(s) in v1",
                                            mname, arity
                                        ),
                                    ));
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                self.used_dyn = true;
                                self.used_psa_functools = true;
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                        }
                    }
                }
                // psa_std batch 3: the Call(Attr) forms of
                // the bisect / heapq / base64 / random / re
                // intercepts, mirroring the MethodCall blocks
                // above (keywords refused). The net group
                // (socket / urllib) mirrors are appended
                // after the re block below.
                if self.std_bisect || self.std_heapq || self.std_base64 || self.std_random
                    || self.std_re || self.std_socket || self.std_urllib
                {
                    if let Expr::Attr(base, mname, _) = &**callee {
                        if let Expr::Name(n, _) = &**base {
                            if n == "bisect" && self.std_bisect && ctx.lookup(n).is_none() {
                                let sla_fn: &str = match mname.as_str() {
                                    "bisect_left" => "psa_bisect_bisect_left",
                                    "bisect_right" => "psa_bisect_bisect_right",
                                    "bisect" => "psa_bisect_bisect",
                                    "insort_left" => "psa_bisect_insort_left",
                                    "insort_right" => "psa_bisect_insort_right",
                                    "insort" => "psa_bisect_insort",
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `bisect` has no function `{}` in v1 (bisect_left / bisect_right / bisect / insort_left / insort_right / insort)",
                                                mname
                                            ),
                                        ))
                                    }
                                };
                                if (args.len() < 2 || args.len() > 4) || !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "bisect.{} takes two to four positional arguments in v1 (the key= keyword is not covered)",
                                            mname
                                        ),
                                    ));
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                if acodes.len() == 2 {
                                    acodes.push("__psa_box_int(0)".to_string());
                                }
                                if acodes.len() == 3 {
                                    acodes.push("__psa_m_none()".to_string());
                                }
                                self.used_dyn = true;
                                self.used_psa_bisect = true;
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                            if n == "heapq" && self.std_heapq && ctx.lookup(n).is_none() {
                                let (sla_fn, arity): (&str, usize) = match mname.as_str() {
                                    "heappush" => ("psa_heapq_heappush", 2),
                                    "heappop" => ("psa_heapq_heappop", 1),
                                    "heapify" => ("psa_heapq_heapify", 1),
                                    "heapreplace" => ("psa_heapq_heapreplace", 2),
                                    "heappushpop" => ("psa_heapq_heappushpop", 2),
                                    "nlargest" => ("psa_heapq_nlargest", 2),
                                    "nsmallest" => ("psa_heapq_nsmallest", 2),
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `heapq` has no function `{}` in v1 (heappush / heappop / heapify / heapreplace / heappushpop / nlargest / nsmallest)",
                                                mname
                                            ),
                                        ))
                                    }
                                };
                                if args.len() != arity || !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "heapq.{} takes exactly {} positional argument(s) in v1 (the key= keyword is not covered)",
                                            mname, arity
                                        ),
                                    ));
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                self.used_dyn = true;
                                self.used_psa_heapq = true;
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                            if n == "base64" && self.std_base64 && ctx.lookup(n).is_none() {
                                let (sla_fn, arity): (&str, usize) = match mname.as_str() {
                                    "b64encode" => ("psa_base64_b64encode", 1),
                                    "b64decode" => ("psa_base64_b64decode", 1),
                                    "standard_b64encode" => {
                                        ("psa_base64_standard_b64encode", 1)
                                    }
                                    "standard_b64decode" => {
                                        ("psa_base64_standard_b64decode", 1)
                                    }
                                    "urlsafe_b64encode" => {
                                        ("psa_base64_urlsafe_b64encode", 1)
                                    }
                                    "urlsafe_b64decode" => {
                                        ("psa_base64_urlsafe_b64decode", 1)
                                    }
                                    "b32encode" => ("psa_base64_b32encode", 1),
                                    "b32decode" => ("psa_base64_b32decode", 1),
                                    "b16encode" => ("psa_base64_b16encode", 1),
                                    "b16decode" => ("psa_base64_b16decode", 1),
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `base64` has no function `{}` in v1 (b64encode / b64decode / standard_b64encode / standard_b64decode / urlsafe_b64encode / urlsafe_b64decode / b32encode / b32decode / b16encode / b16decode)",
                                                mname
                                            ),
                                        ))
                                    }
                                };
                                let casefold_form = (mname == "b32decode"
                                    || mname == "b16decode")
                                    && args.len() == 2;
                                let sla_fn = if casefold_form {
                                    match &args[1] {
                                        Expr::Bool(true) => {
                                            if mname == "b32decode" {
                                                "psa_base64_b32decode_casefold"
                                            } else {
                                                "psa_base64_b16decode_casefold"
                                            }
                                        }
                                        Expr::Bool(false) => sla_fn,
                                        _ => {
                                            return Err(eerr(
                                                line,
                                                &format!(
                                                    "base64.{} casefold argument must be a literal bool in v1",
                                                    mname
                                                ),
                                            ))
                                        }
                                    }
                                } else {
                                    sla_fn
                                };
                                if (args.len() != arity && !casefold_form)
                                    || !kwargs.is_empty()
                                {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "base64.{} takes exactly {} positional argument(s) in v1",
                                            mname, arity
                                        ),
                                    ));
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                let emit_n = if casefold_form { 1 } else { args.len() };
                                for a in args.iter().take(emit_n) {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                self.used_dyn = true;
                                self.used_psa_base64 = true;
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                            if n == "random" && self.std_random && ctx.lookup(n).is_none() {
                                if !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "random.{} takes positional arguments only in v1 (the Call form refuses keywords)",
                                            mname
                                        ),
                                    ));
                                }
                                match mname.as_str() {
                                    "seed" | "random" | "getrandbits" | "randbytes"
                                    | "randrange" | "randint" | "choice" | "shuffle"
                                    | "sample" | "choices" | "uniform" | "triangular" => {}
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `random` has no function `{}` in v1 (seed / random / getrandbits / randbytes / randrange / randint / choice / shuffle / sample / choices / uniform / triangular)",
                                                mname
                                            ),
                                        ))
                                    }
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                let sla_fn: &str = match mname.as_str() {
                                    "seed" => {
                                        if args.len() > 1 {
                                            return Err(eerr(
                                                line,
                                                "random.seed takes zero or one positional argument in v1",
                                            ));
                                        }
                                        if acodes.is_empty() {
                                            acodes.push("__psa_m_none()".to_string());
                                        }
                                        "psa_random_seed"
                                    }
                                    "random" => {
                                        if !args.is_empty() {
                                            return Err(eerr(
                                                line,
                                                "random.random takes no arguments in v1",
                                            ));
                                        }
                                        "psa_random_random"
                                    }
                                    "getrandbits" | "randbytes" | "choice" | "shuffle" => {
                                        if args.len() != 1 {
                                            return Err(eerr(
                                                line,
                                                &format!(
                                                    "random.{} takes exactly one positional argument in v1",
                                                    mname
                                                ),
                                            ));
                                        }
                                        match mname.as_str() {
                                            "getrandbits" => "psa_random_getrandbits",
                                            "randbytes" => "psa_random_randbytes",
                                            "choice" => "psa_random_choice",
                                            _ => "psa_random_shuffle",
                                        }
                                    }
                                    "randint" | "sample" | "uniform" => {
                                        if args.len() != 2 {
                                            return Err(eerr(
                                                line,
                                                &format!(
                                                    "random.{} takes exactly two positional arguments in v1",
                                                    mname
                                                ),
                                            ));
                                        }
                                        match mname.as_str() {
                                            "randint" => "psa_random_randint",
                                            "sample" => "psa_random_sample",
                                            _ => "psa_random_uniform",
                                        }
                                    }
                                    "randrange" => match args.len() {
                                        1 => "psa_random_randrange1",
                                        2 => "psa_random_randrange2",
                                        3 => "psa_random_randrange3",
                                        _ => {
                                            return Err(eerr(
                                                line,
                                                "random.randrange takes one to three positional arguments in v1",
                                            ))
                                        }
                                    },
                                    "choices" => match args.len() {
                                        1 => {
                                            acodes.push("__psa_box_int(1)".to_string());
                                            "psa_random_choices_k"
                                        }
                                        2 => {
                                            if matches!(args[1], Expr::NoneLit) {
                                                acodes.truncate(1);
                                                acodes.push("__psa_box_int(1)".to_string());
                                                "psa_random_choices_k"
                                            } else {
                                                acodes.push("__psa_box_int(1)".to_string());
                                                "psa_random_choices_w"
                                            }
                                        }
                                        3 => {
                                            if matches!(args[1], Expr::NoneLit) {
                                                let kcode = acodes.pop().unwrap();
                                                acodes.truncate(1);
                                                acodes.push(kcode);
                                                "psa_random_choices_k"
                                            } else {
                                                "psa_random_choices_w"
                                            }
                                        }
                                        _ => {
                                            return Err(eerr(
                                                line,
                                                "random.choices takes one to three positional arguments in v1 (the k= / cum_weights= keyword forms are not covered)",
                                            ))
                                        }
                                    },
                                    "triangular" => {
                                        if args.len() > 3 {
                                            return Err(eerr(
                                                line,
                                                "random.triangular takes zero to three positional arguments in v1",
                                            ));
                                        }
                                        let pads = [
                                            "__psa_box_float(0.0)",
                                            "__psa_box_float(1.0)",
                                            "__psa_m_none()",
                                        ];
                                        while acodes.len() < 3 {
                                            let p = pads[acodes.len()];
                                            acodes.push(p.to_string());
                                        }
                                        "psa_random_triangular"
                                    }
                                    _ => unreachable!(),
                                };
                                self.used_dyn = true;
                                self.used_psa_random = true;
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                            if n == "re" && self.std_re && ctx.lookup(n).is_none() {
                                let (sla_fn, arity): (&str, usize) = match mname.as_str() {
                                    "match" => ("psa_re_match", 3),
                                    "search" => ("psa_re_search", 3),
                                    "fullmatch" => ("psa_re_fullmatch", 3),
                                    "findall" => ("psa_re_findall", 3),
                                    "split" => ("psa_re_split", 3),
                                    "split_maxsplit" => ("psa_re_split_maxsplit", 4),
                                    "sub" => ("psa_re_sub", 4),
                                    "sub_count" => ("psa_re_sub_count", 5),
                                    "subn" => ("psa_re_subn", 4),
                                    "subn_count" => ("psa_re_subn_count", 5),
                                    "escape" => ("psa_re_escape", 1),
                                    "IGNORECASE" => ("psa_re_ignorecase", 0),
                                    "MULTILINE" => ("psa_re_multiline", 0),
                                    "DOTALL" => ("psa_re_dotall", 0),
                                    "ASCII" => ("psa_re_ascii", 0),
                                    "VERBOSE" => ("psa_re_verbose", 0),
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `re` has no function `{}` in v1 (match / search / fullmatch / findall / split / split_maxsplit / sub / sub_count / subn / subn_count / escape / IGNORECASE / MULTILINE / DOTALL / ASCII / VERBOSE; compile is not covered)",
                                                mname
                                            ),
                                        ))
                                    }
                                };
                                let defaultable = matches!(
                                    mname.as_str(),
                                    "match" | "search" | "fullmatch" | "findall" | "split"
                                        | "sub" | "subn"
                                );
                                if (args.len() != arity
                                    && !(defaultable && args.len() == arity - 1))
                                    || !kwargs.is_empty()
                                {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "re.{} takes {} positional argument(s) in v1{}",
                                            mname,
                                            arity,
                                            if defaultable {
                                                " (flags may be omitted)"
                                            } else {
                                                ""
                                            }
                                        ),
                                    ));
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                if defaultable && acodes.len() == arity - 1 {
                                    acodes.push("__psa_box_int(0)".to_string());
                                }
                                self.used_dyn = true;
                                self.used_psa_re = true;
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                            // psa_std net group: `socket.<fn>(...)`.
                            // Module-level functions only; mname calls
                            // on socket objects held in variables await
                            // std-object tracking (GAP-net.md N-1). The
                            // constants are zero-argument calls, as
                            // std-module attribute reads are not wired
                            // (N-2). `socket.socket(family)` pads
                            // SOCK_STREAM (boxed int 1) per
                            // WIRING-socket.md.
                            if n == "socket" && self.std_socket && ctx.lookup(n).is_none() {
                                if !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "socket.{} takes positional arguments only in v1 (the Call form refuses keywords)",
                                            mname
                                        ),
                                    ));
                                }
                                match mname.as_str() {
                                    "socket" | "create_connection" | "gethostname" | "gethostbyname"
                                    | "gethostbyname_ex" | "gethostbyaddr" | "getaddrinfo"
                                    | "getnameinfo" | "socketpair" | "fromfd" | "inet_aton"
                                    | "inet_ntoa" | "inet_pton" | "inet_ntop" | "AF_INET" | "AF_INET6"
                                    | "SOCK_STREAM" | "SOCK_DGRAM" | "SOL_SOCKET" => {}
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `socket` has no function `{}` in v1 (socket / create_connection / gethostname / gethostbyname / gethostbyname_ex / gethostbyaddr / getaddrinfo / getnameinfo / socketpair / fromfd / inet_aton / inet_ntoa / inet_pton / inet_ntop / AF_INET / AF_INET6 / SOCK_STREAM / SOCK_DGRAM / SOL_SOCKET; object methods such as connect / send / recv are not covered)",
                                                mname
                                            ),
                                        ))
                                    }
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                let sla_fn: &str = match mname.as_str() {
                                    "socket" => match args.len() {
                                        0 => "psa_socket_new_default",
                                        1 => {
                                            acodes.push("__psa_box_int(1)".to_string());
                                            "psa_socket_new"
                                        }
                                        2 => "psa_socket_new",
                                        _ => {
                                            return Err(eerr(
                                                line,
                                                "socket.socket takes zero to two positional arguments in v1",
                                            ))
                                        }
                                    },
                                    "create_connection" => match args.len() {
                                        1 => "psa_socket_create_connection_addr",
                                        2 => "psa_socket_create_connection_addr_timeout",
                                        _ => {
                                            return Err(eerr(
                                                line,
                                                "socket.create_connection takes one or two positional arguments in v1",
                                            ))
                                        }
                                    },
                                    "gethostname" | "socketpair" | "AF_INET" | "AF_INET6" | "SOCK_STREAM"
                                    | "SOCK_DGRAM" | "SOL_SOCKET" => {
                                        if !args.is_empty() {
                                            return Err(eerr(
                                                line,
                                                &format!("socket.{} takes no arguments in v1", mname),
                                            ));
                                        }
                                        match mname.as_str() {
                                            "gethostname" => "psa_socket_gethostname",
                                            "socketpair" => "psa_socket_socketpair",
                                            "AF_INET" => "psa_socket_af_inet",
                                            "AF_INET6" => "psa_socket_af_inet6",
                                            "SOCK_STREAM" => "psa_socket_sock_stream",
                                            "SOCK_DGRAM" => "psa_socket_sock_dgram",
                                            _ => "psa_socket_sol_socket",
                                        }
                                    }
                                    "gethostbyname" | "gethostbyname_ex" | "gethostbyaddr" | "fromfd"
                                    | "inet_aton" | "inet_ntoa" => {
                                        if args.len() != 1 {
                                            return Err(eerr(
                                                line,
                                                &format!(
                                                    "socket.{} takes exactly one positional argument in v1",
                                                    mname
                                                ),
                                            ));
                                        }
                                        match mname.as_str() {
                                            "gethostbyname" => "psa_socket_gethostbyname",
                                            "gethostbyname_ex" => "psa_socket_gethostbyname_ex",
                                            "gethostbyaddr" => "psa_socket_gethostbyaddr",
                                            "fromfd" => "psa_socket_fromfd",
                                            "inet_aton" => "psa_socket_inet_aton",
                                            _ => "psa_socket_inet_ntoa",
                                        }
                                    }
                                    "getaddrinfo" | "getnameinfo" | "inet_pton" | "inet_ntop" => {
                                        if args.len() != 2 {
                                            return Err(eerr(
                                                line,
                                                &format!(
                                                    "socket.{} takes exactly two positional arguments in v1",
                                                    mname
                                                ),
                                            ));
                                        }
                                        match mname.as_str() {
                                            "getaddrinfo" => "psa_socket_getaddrinfo",
                                            "getnameinfo" => "psa_socket_getnameinfo",
                                            "inet_pton" => "psa_socket_inet_pton",
                                            _ => "psa_socket_inet_ntop",
                                        }
                                    }
                                    _ => unreachable!(),
                                };
                                self.used_dyn = true;
                                self.used_psa_socket = true;
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                            // psa_std net group: `urllib.<fn>(...)`.
                            // urllib.parse and urllib.request are
                            // flattened into this one module
                            // (WIRING-urllib.md); the submodule
                            // spellings are not wired. hostname / port
                            // are spelled hostname_of / port_of, as
                            // std-module attribute reads are not wired.
                            // Response and Request object methods
                            // await std-object tracking (GAP-net.md
                            // N-1); the function forms live in
                            // urllib.sla and are harness-verified.
                            if n == "urllib" && self.std_urllib && ctx.lookup(n).is_none() {
                                if !kwargs.is_empty() {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "urllib.{} takes positional arguments only in v1 (the Call form refuses keywords)",
                                            mname
                                        ),
                                    ));
                                }
                                match mname.as_str() {
                                    "urlsplit" | "urlunsplit" | "urljoin" | "quote" | "quote_plus"
                                    | "unquote" | "unquote_plus" | "urlencode" | "parse_qs" | "parse_qsl"
                                    | "hostname_of" | "port_of" | "urlopen" | "Request" => {}
                                    _ => {
                                        return Err(eerr(
                                            line,
                                            &format!(
                                                "psa_std module `urllib` has no function `{}` in v1 (urlsplit / urlunsplit / urljoin / quote / quote_plus / unquote / unquote_plus / urlencode / parse_qs / parse_qsl / hostname_of / port_of / urlopen / Request; object methods such as read / getcode / getheader are not covered)",
                                                mname
                                            ),
                                        ))
                                    }
                                }
                                let mut acodes: Vec<String> = Vec::new();
                                for a in args {
                                    acodes.push(self.emit_dyn_value(ctx, a, line)?);
                                }
                                let sla_fn: &str = match mname.as_str() {
                                    "urlsplit" | "urlunsplit" | "unquote" | "unquote_plus" | "hostname_of"
                                    | "port_of" => {
                                        if args.len() != 1 {
                                            return Err(eerr(
                                                line,
                                                &format!(
                                                    "urllib.{} takes exactly one positional argument in v1",
                                                    mname
                                                ),
                                            ));
                                        }
                                        match mname.as_str() {
                                            "urlsplit" => "psa_urllib_parse_urlsplit",
                                            "urlunsplit" => "psa_urllib_parse_urlunsplit",
                                            "unquote" => "psa_urllib_parse_unquote",
                                            "unquote_plus" => "psa_urllib_parse_unquote_plus",
                                            "hostname_of" => "psa_urllib_parse_hostname",
                                            _ => "psa_urllib_parse_port",
                                        }
                                    }
                                    "urljoin" => {
                                        if args.len() != 2 {
                                            return Err(eerr(
                                                line,
                                                "urllib.urljoin takes exactly two positional arguments in v1",
                                            ));
                                        }
                                        "psa_urllib_parse_urljoin"
                                    }
                                    "quote" => match args.len() {
                                        1 => "psa_urllib_parse_quote",
                                        2 => "psa_urllib_parse_quote_safe",
                                        _ => {
                                            return Err(eerr(
                                                line,
                                                "urllib.quote takes one or two positional arguments in v1",
                                            ))
                                        }
                                    },
                                    "quote_plus" => match args.len() {
                                        1 => "psa_urllib_parse_quote_plus",
                                        2 => "psa_urllib_parse_quote_plus_safe",
                                        _ => {
                                            return Err(eerr(
                                                line,
                                                "urllib.quote_plus takes one or two positional arguments in v1",
                                            ))
                                        }
                                    },
                                    "urlencode" => match args.len() {
                                        1 => "psa_urllib_parse_urlencode",
                                        2 => "psa_urllib_parse_urlencode_doseq",
                                        _ => {
                                            return Err(eerr(
                                                line,
                                                "urllib.urlencode takes one or two positional arguments in v1",
                                            ))
                                        }
                                    },
                                    "parse_qs" => match args.len() {
                                        1 => "psa_urllib_parse_parse_qs",
                                        2 => "psa_urllib_parse_parse_qs_kb",
                                        _ => {
                                            return Err(eerr(
                                                line,
                                                "urllib.parse_qs takes one or two positional arguments in v1",
                                            ))
                                        }
                                    },
                                    "parse_qsl" => match args.len() {
                                        1 => "psa_urllib_parse_parse_qsl",
                                        2 => "psa_urllib_parse_parse_qsl_kb",
                                        _ => {
                                            return Err(eerr(
                                                line,
                                                "urllib.parse_qsl takes one or two positional arguments in v1",
                                            ))
                                        }
                                    },
                                    "urlopen" => match args.len() {
                                        1 => "psa_urllib_urlopen",
                                        2 => "psa_urllib_urlopen_data",
                                        3 => "psa_urllib_urlopen_full",
                                        _ => {
                                            return Err(eerr(
                                                line,
                                                "urllib.urlopen takes one to three positional arguments in v1",
                                            ))
                                        }
                                    },
                                    "Request" => match args.len() {
                                        1 => "psa_urllib_request_new",
                                        2 => "psa_urllib_request_new_data",
                                        3 => {
                                            acodes.push("__psa_m_none()".to_string());
                                            "psa_urllib_request_new_full"
                                        }
                                        4 => "psa_urllib_request_new_full",
                                        _ => {
                                            return Err(eerr(
                                                line,
                                                "urllib.Request takes one to four positional arguments in v1",
                                            ))
                                        }
                                    },
                                    _ => unreachable!(),
                                };
                                self.used_dyn = true;
                                self.used_psa_urllib = true;
                                return Ok((format!("{}({})", sla_fn, acodes.join(", ")), Ty::Dyn));
                            }
                        }
                    }
                }
                // D6: a call through an attribute chain rooted
                // at an imported module: `foo.bar.baz(...)` (a
                // def of a dotted module) or `foo.bar.Klass(...)`
                // (a class of a dotted module). A chain rooted
                // at a variable is never a module reference.
                if matches!(&**callee, Expr::Attr(..)) {
                    if let Some(key) = self.module_def_key(ctx, callee) {
                        let call = Expr::Call(
                            Box::new(Expr::Name(key, ExprSpan::synthetic())),
                            args.clone(),
                            kwargs.clone(),
                        ExprSpan::synthetic());
                        return self.emit_expr(ctx, &call, line);
                    }
                    if let Some((cname, cid)) = self.module_class(ctx, callee) {
                        return self.emit_construct(ctx, &cname, cid, args, kwargs, line);
                    }
                }
                // D4: calling an attribute of a dynamic value --
                // a chained `a.b.m(...)` whose callee is an Attr
                // node. getattr yields the value (a bound method
                // for methods) and the callable dispatch runs it.
                if let Expr::Attr(base, mname, _) = &**callee {
                    if !kwargs.is_empty() {
                        return Err(eerr(
                            line,
                            "keyword arguments are not supported when calling a method value yet (D4 v1 method slice)",
                        ));
                    }
                    let (bcode, bty) = if let Expr::Str(s) = &**base {
                        (self.emit_str_temp(ctx, s, line)?, Ty::Dyn)
                    } else {
                        self.emit_expr(ctx, base, line)?
                    };
                    if bty != Ty::Dyn {
                        return Err(eerr(
                            line,
                            "calling a method on a non-dynamic value is not supported yet (container/string methods live on dynamic values in D5)",
                        ));
                    }
                    self.used_dyn = true;
                    // D5 builtin method set first (the runtime
                    // defers objects to getattr).
                    if let Some(code) =
                        self.emit_builtin_method(ctx, &bcode, mname, args, line)?
                    {
                        return Ok((code, Ty::Dyn));
                    }
                    let ntmp = self.emit_str_temp(ctx, mname, line)?;
                    let codes = self.emit_dyn_args(ctx, args, line)?;
                    let tmp = self.pack_dyn_args(ctx, &codes);
                    return Ok((
                        format!("__psa_call_value(__psa_getattr({}, {}), {})", bcode, ntmp, tmp),
                        Ty::Dyn,
                    ));
                }
                // D8: a computed callee — anything that is
                // neither a plain name nor an attribute chain.
                // Closure conversion produces these when a
                // captured function value is called (the read
                // is a CellGet), and parenthesised / indexed
                // forms land here too. A dynamic callee goes
                // through the generated dispatch, which covers
                // tags 7, 9 and 16; a non-dynamic callee keeps
                // the historic loud refusal below.
                if !matches!(&**callee, Expr::Name(_, _) | Expr::Attr(..)) {
                    if !kwargs.is_empty() {
                        return Err(eerr(
                            line,
                            "keyword arguments are not supported when calling a computed function value yet (D8 v1 closure slice)",
                        ));
                    }
                    let (ccode, cty) = self.emit_expr(ctx, callee, line)?;
                    if cty != Ty::Dyn {
                        return Err(eerr(
                            line,
                            "v0.1 calls only plain function names (no methods, no first-class functions)",
                        ));
                    }
                    self.used_dyn = true;
                    let codes = self.emit_dyn_args(ctx, args, line)?;
                    let tmp = self.pack_dyn_args(ctx, &codes);
                    return Ok((format!("__psa_call_value({}, {})", ccode, tmp), Ty::Dyn));
                }
                let (fname, fname_anchor) = match &**callee {
                    Expr::Name(n, span) => (n.clone(), span.start),
                    _ => {
                        return Err(eerr(
                            line,
                            "v0.1 calls only plain function names (no methods, no first-class functions)",
                        ))
                    }
                };
                // Calling through a dynamic variable (D3 function
                // values): the variable holds a callable (tag 7)
                // and the generated dispatch checks the tag and
                // the arity. A variable shadows a def of the same
                // name, as in Python.
                if let Some(Ty::Dyn) = ctx.lookup(&fname) {
                    ctx.check_alive(&fname, line, Some(fname_anchor))?;
                    if !kwargs.is_empty() {
                        return Err(eerr(
                            line,
                            "keyword arguments are not supported when calling a function value yet (D3 v1 callable slice)",
                        ));
                    }
                    self.used_dyn = true;
                    let codes = self.emit_dyn_args(ctx, args, line)?;
                    let tmp = self.pack_dyn_args(ctx, &codes);
                    return Ok((format!("__psa_call_value({}, {})", fname, tmp), Ty::Dyn));
                }
                // D4 class construction: `Name(args)` where Name
                // is a class (and not a shadowing variable).
                if ctx.lookup(&fname).is_none() {
                    if let Some(&cid) = self.class_ids.get(&fname) {
                        return self.emit_construct(ctx, &fname, cid, args, kwargs, line);
                    }
                }
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
                    if !kwargs.is_empty() {
                        return Err(eerr(line, "len() takes no keyword arguments"));
                    }
                    if let Expr::Str(s) = &args[0] {
                        let tmp = self.emit_str_temp(ctx, s, line)?;
                        return Ok((format!("__psa_dyn_len({})", tmp), Ty::Int));
                    }
                    let (c, t) = self.emit_expr(ctx, &args[0], line)?;
                    if t == Ty::Dyn {
                        self.used_dyn = true;
                        return Ok((format!("__psa_dyn_len({})", c), Ty::Int));
                    }
                    if !t.is_list() {
                        return Err(eerr(line, "len() is only supported on lists in v0.3"));
                    }
                    return Ok((format!("(len({}) as i64)", c), Ty::Int));
                }
                // D5 conversion / constructor / reflection
                // builtins, unless the name is shadowed by a
                // user def or variable (then it is an ordinary
                // call, as in Python).
                if ctx.lookup(&fname).is_none() && !self.defs.contains_key(&fname) {
                    if let Some(r) = self.emit_builtin_call(ctx, &fname, args, kwargs, line)? {
                        return Ok(r);
                    }
                }
                // Argument binding (D3): positionals in order, then
                // keyword arguments by parameter name, then declared
                // defaults for the rest. The bound list always has
                // exactly the callee's parameter count, so ensure_fn
                // and the boundary/ownership machinery below see a
                // plain positional call. Defaults are restricted to
                // literals: psa evaluates a default at each call
                // site, which is indistinguishable from Python's
                // evaluate-once for immutable literals but NOT for
                // mutable or computed defaults -- those stay loud.
                let bound_args: Vec<Expr>;
                let mut vararg_slot: Option<usize> = None;
                let mut kwarg_slot: Option<usize> = None;
                let args: &[Expr] = if kwargs.is_empty()
                    && self
                        .defs
                        .get(&fname)
                        .map(|d| {
                            d.vararg.is_none()
                                && d.kwarg.is_none()
                                && d.params.iter().all(|(_, dflt)| dflt.is_none())
                        })
                        .unwrap_or(true)
                {
                    args
                } else if let Some(def) = self.defs.get(&fname) {
                    let fixed = def.params.len();
                    let mut bound: Vec<Option<Expr>> = vec![None; fixed];
                    let mut surplus_pos: Vec<Expr> = Vec::new();
                    for (i, a) in args.iter().enumerate() {
                        if i < fixed {
                            bound[i] = Some(a.clone());
                        } else if def.vararg.is_some() {
                            surplus_pos.push(a.clone());
                        } else {
                            return Err(eerr(
                                line,
                                &format!(
                                    "function `{}` takes {} arguments, called with {} positional",
                                    fname,
                                    fixed,
                                    args.len()
                                ),
                            ));
                        }
                    }
                    let mut surplus_kw: Vec<(String, Expr)> = Vec::new();
                    for (k, v) in kwargs.iter() {
                        match def.params.iter().position(|(pn, _)| pn == k) {
                            None => {
                                if def.kwarg.is_some() {
                                    surplus_kw.push((k.clone(), v.clone()));
                                } else {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "function `{}` has no parameter named `{}`",
                                            fname, k
                                        ),
                                    ));
                                }
                            }
                            Some(j) => {
                                if bound[j].is_some() {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "function `{}` got multiple values for argument `{}`",
                                            fname, k
                                        ),
                                    ));
                                }
                                bound[j] = Some(v.clone());
                            }
                        }
                    }
                    for (j, slot) in bound.iter_mut().enumerate() {
                        if slot.is_none() {
                            match &def.params[j].1 {
                                Some(dflt) => {
                                    match dflt {
                                        Expr::Int(_)
                                        | Expr::BigInt(_)
                                        | Expr::Float(_)
                                        | Expr::Str(_)
                                        | Expr::Bool(_)
                                        | Expr::NoneLit => {}
                                        _ => {
                                            return Err(eerr(
                                                line,
                                                &format!(
                                                    "default value for parameter `{}` of `{}` must be a literal (psa evaluates defaults per call; mutable or computed defaults are refused)",
                                                    def.params[j].0, fname
                                                ),
                                            ))
                                        }
                                    }
                                    *slot = Some(dflt.clone());
                                }
                                None => {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "function `{}` missing required argument `{}`",
                                            fname, def.params[j].0
                                        ),
                                    ))
                                }
                            }
                        }
                    }
                    let mut full: Vec<Expr> =
                        bound.into_iter().map(|s| s.unwrap()).collect();
                    if def.vararg.is_some() {
                        vararg_slot = Some(full.len());
                        full.push(Expr::ListLit(surplus_pos));
                    }
                    if def.kwarg.is_some() {
                        // Surplus keywords collect into a dynamic
                        // dict with str keys (D3 ordered dict keys
                        // are PsaVals under key equivalence).
                        kwarg_slot = Some(full.len());
                        let pairs: Vec<(Expr, Expr)> = surplus_kw
                            .into_iter()
                            .map(|(k, v)| (Expr::Str(k), v))
                            .collect();
                        full.push(Expr::DictLit(pairs));
                    }
                    bound_args = full;
                    &bound_args
                } else {
                    args
                };
                let mut arg_codes = Vec::new();
                let mut arg_tys = Vec::new();
                for (ai, a) in args.iter().enumerate() {
                    // *args/**kwargs collection slots are always
                    // dynamic containers, built as temps so the
                    // callee's collection parameters have the
                    // uniform type Dyn at every call site.
                    if Some(ai) == vararg_slot || Some(ai) == kwarg_slot {
                        let tmp = self.emit_container_temp(ctx, a, line)?;
                        arg_codes.push(tmp);
                        arg_tys.push(Ty::Dyn);
                        continue;
                    }
                    // A string literal argument becomes a dynamic
                    // string: the parameter is pinned Dyn, never Str.
                    if let Expr::Str(s) = a {
                        let tmp = self.emit_str_temp(ctx, s, line)?;
                        arg_codes.push(tmp);
                        arg_tys.push(Ty::Dyn);
                        continue;
                    }
                    let (c, t) = self.emit_expr(ctx, a, line)?;
                    arg_codes.push(c);
                    arg_tys.push(t);
                }
                let sig = self.ensure_fn(&fname, &arg_tys, line)?;
                let ret = sig.ret.unwrap_or(Ty::Void);
                // Static/dynamic boundary (design s3): a dynamic
                // parameter takes any D1 scalar, boxed here; a scalar
                // parameter takes a dynamic argument, unboxed here
                // with a runtime type assertion.
                for i in 0..args.len() {
                    let pt = sig.params[i];
                    let at = arg_tys[i];
                    if pt == Ty::Dyn && at != Ty::Dyn {
                        let c = std::mem::take(&mut arg_codes[i]);
                        arg_codes[i] = self.box_code(c, at, line)?;
                    } else if at == Ty::Dyn && pt != Ty::Dyn {
                        // A big integer literal consumed directly by
                        // a static parameter is a compile-time
                        // refusal (PSA-E6010), not a runtime unbox:
                        // the literal itself can never fit.
                        if matches!(args[i], Expr::BigInt(_)) {
                            return Err(eerr(
                                line,
                                "integer literal out of i64 range in a static position (dynamic positions accept arbitrary precision)",
                            ));
                        }
                        let c = std::mem::take(&mut arg_codes[i]);
                        arg_codes[i] = self.unbox_code(c, pt, line)?;
                    }
                }
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
                        Expr::Name(n, span) => {
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
                                ctx.mark_moved(n, span.start);
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
                Ok((format!("{}({})", self.sla_name(&fname), arg_codes.join(", ")), ret))
            }
            Expr::Bin(l, op, r) => {
                if *op == BinOp::Pow {
                    return self.emit_pow(ctx, l, r, line);
                }
                if *op == BinOp::And || *op == BinOp::Or {
                    // With a dynamic operand, and/or have Python's
                    // value semantics (the result is one of the two
                    // operands), which needs a materialised temp and
                    // short-circuit placement of the right operand.
                    // The right operand is therefore emitted into a
                    // scratch buffer first: if both sides turn out
                    // static, its lines are spliced back in order;
                    // if either side is dynamic, they land inside
                    // the branch that actually evaluates it.
                    let (lc, mut lt) = self.emit_expr(ctx, l, line)?;
                    let saved_out = std::mem::take(&mut ctx.out);
                    let r_res = self.emit_expr(ctx, r, line);
                    let mut r_lines = std::mem::replace(&mut ctx.out, saved_out);
                    let (mut rc, mut rt) = r_res?;
                    if lt != Ty::Dyn && rt != Ty::Dyn {
                        if lt == Ty::Bool && rt == Ty::Bool && !ctx.in_while_cond {
                            // Static and/or must short-circuit
                            // too: SLA's && / || evaluate both
                            // sides eagerly (probe P4; compiler
                            // defect (xviii)), so a bare
                            // (a && b) runs a bounds-guarded RHS
                            // read even when the guard failed.
                            // Mirror emit_dyn_logic's placement:
                            // the RHS lines land inside the
                            // branch that actually evaluates
                            // them. While conditions keep the
                            // bare form (one expression, like
                            // the dyn path's refusal there).
                            ctx.tmp += 1;
                            let name = format!("__psa_logic_{}", ctx.tmp);
                            if ctx.scopes.len() > 1 {
                                ctx.hoist_declare(&name, Ty::Bool);
                            } else {
                                ctx.declare(&name, Ty::Bool)
                                    .map_err(|m| eerr(line, &m))?;
                                ctx.line(&format!("let {}: bool = false;", name));
                            }
                            ctx.line(&format!("{} = {};", name, lc));
                            if *op == BinOp::And {
                                ctx.line(&format!("if {} {{", name));
                            } else {
                                ctx.line(&format!("if !({}) {{", name));
                            }
                            ctx.out.append_writer(r_lines);
                            ctx.indent += 1;
                            ctx.line(&format!("{} = {};", name, rc));
                            ctx.indent -= 1;
                            ctx.line("};");
                            return Ok((name, Ty::Bool));
                        }
                        ctx.out.append_writer(r_lines);
                        return self.emit_binop(lc, lt, *op, rc, rt, line);
                    }
                    // A string literal beside a dynamic operand
                    // becomes a dynamic string temp; the right one
                    // is built inside the deferred branch lines.
                    if lt == Ty::Str {
                        if let Expr::Str(s) = &**l {
                            let tmp = self.emit_str_temp(ctx, s, line)?;
                            return self.emit_dyn_logic(ctx, tmp, Ty::Dyn, *op, rc, rt, r_lines, line);
                        }
                    }
                    if rt == Ty::Str {
                        if let Expr::Str(s) = &**r {
                            std::mem::swap(&mut ctx.out, &mut r_lines);
                            let tmp_res = self.emit_str_temp(ctx, s, line);
                            std::mem::swap(&mut ctx.out, &mut r_lines);
                            rc = tmp_res?;
                            rt = Ty::Dyn;
                        }
                    }
                    return self.emit_dyn_logic(ctx, lc, lt, *op, rc, rt, r_lines, line);
                }
                let (mut lc, mut lt) = self.emit_expr(ctx, l, line)?;
                let (mut rc, mut rt) = self.emit_expr(ctx, r, line)?;
                // A string literal beside a dynamic operand becomes
                // a dynamic string temp (D2 slice C).
                if lt == Ty::Str && rt == Ty::Dyn {
                    if let Expr::Str(s) = &**l {
                        lc = self.emit_str_temp(ctx, s, line)?;
                        lt = Ty::Dyn;
                    }
                }
                if rt == Ty::Str && lt == Ty::Dyn {
                    if let Expr::Str(s) = &**r {
                        rc = self.emit_str_temp(ctx, s, line)?;
                        rt = Ty::Dyn;
                    }
                }
                {
                    // Cross-kind static identity (171-B
                    // NOTES: cross-kind identity is always
                    // False): when neither operand is a
                    // scalar of the other's kind and at
                    // least one side is a non-scalar (a
                    // static string/list/...), the two
                    // values can never be the same object,
                    // so the answer is a constant. Boxing
                    // is not an option — static containers
                    // cannot become dynamic values in D1
                    // (E5020) — and none is needed.
                    // Restricted to duplicable (pure,
                    // deterministic) operands, whose
                    // emitted codes may be dropped safely;
                    // anything else falls through to the
                    // loud refusal in emit_binop. Same-kind
                    // non-scalar pairs (Str/Str, List/List)
                    // are excluded: they stay refused.
                    let scalar =
                        |t: Ty| matches!(t, Ty::Int | Ty::Float | Ty::Bool);
                    if matches!(op, BinOp::Is | BinOp::IsNot)
                        && lt != rt
                        && !(scalar(lt) && scalar(rt))
                        && crate::parser::duplicable(l)
                        && crate::parser::duplicable(r)
                    {
                        return Ok((
                            if *op == BinOp::IsNot { "true" } else { "false" }
                                .to_string(),
                            Ty::Bool,
                        ));
                    }
                }
                self.emit_binop(lc, lt, *op, rc, rt, line)
            }
            Expr::Await(inner) => {
                // D7: awaiting a coroutine value runs its worker
                // to completion at the await point (sequential
                // semantics); awaiting anything else is a loud
                // error, statically for non-dynamic values and at
                // runtime (77082) for other dynamic kinds.
                let (c, t) = self.emit_expr(ctx, inner, line)?;
                if t != Ty::Dyn {
                    return Err(eerr(
                        line,
                        "await requires a coroutine value (the result of calling an async def); static values cannot be awaited in psa v1 (D7)",
                    ));
                }
                self.used_dyn = true;
                self.used_coro = true;
                Ok((format!("__psa_coro_run({})", c), Ty::Dyn))
            }
            Expr::IfExp { cond, then, els } => {
                if ctx.in_while_cond {
                    return Err(eerr(
                        line,
                        "if-expressions are not supported in while conditions in D0 (the condition is re-evaluated every iteration); compute the value before the loop",
                    ));
                }
                let (cc, ct) = self.emit_expr(ctx, cond, line)?;
                let cc = self.cond_code(cc, ct, line, "if-expression condition must be bool")?;
                // Emit each branch into a scratch buffer so any lines a
                // nested if-expression produces land *inside* the branch
                // block, not before the whole expression.
                let saved_out = std::mem::take(&mut ctx.out);
                ctx.indent += 1;
                let then_res = self.emit_expr(ctx, then, line);
                let mut then_lines = std::mem::replace(&mut ctx.out, saved_out);
                let (tc, tt) = then_res?;
                let saved_out = std::mem::take(&mut ctx.out);
                let els_res = self.emit_expr(ctx, els, line);
                let mut els_lines = std::mem::replace(&mut ctx.out, saved_out);
                ctx.indent -= 1;
                let (ec, et) = els_res?;
                if tt.is_list() || et.is_list() {
                    return Err(eerr(
                        line,
                        "if-expression branches cannot be lists in D0 (a branch value would move out of its scope)",
                    ));
                }
                // String literal branches become dynamic strings,
                // built inside their own branch lines (D2 slice C).
                let (tc, tt) = if tt == Ty::Str {
                    match &**then {
                        Expr::Str(s) => {
                            std::mem::swap(&mut ctx.out, &mut then_lines);
                            let tmp_res = self.emit_str_temp(ctx, s, line);
                            std::mem::swap(&mut ctx.out, &mut then_lines);
                            (tmp_res?, Ty::Dyn)
                        }
                        _ => {
                            return Err(eerr(
                                line,
                                "if-expression string branches must be plain string literals in this slice",
                            ))
                        }
                    }
                } else {
                    (tc, tt)
                };
                let (ec, et) = if et == Ty::Str {
                    match &**els {
                        Expr::Str(s) => {
                            std::mem::swap(&mut ctx.out, &mut els_lines);
                            let tmp_res = self.emit_str_temp(ctx, s, line);
                            std::mem::swap(&mut ctx.out, &mut els_lines);
                            (tmp_res?, Ty::Dyn)
                        }
                        _ => {
                            return Err(eerr(
                                line,
                                "if-expression string branches must be plain string literals in this slice",
                            ))
                        }
                    }
                } else {
                    (ec, et)
                };
                // A void branch (a call that returns nothing) is the
                // None value of the dynamic runtime.
                let then_void = tt == Ty::Void;
                let els_void = et == Ty::Void;
                let (tc, tt) = if tt == Ty::Void {
                    (Ty::Dyn.default_lit().to_string(), Ty::Dyn)
                } else {
                    (tc, tt)
                };
                let (ec, et) = if et == Ty::Void {
                    (Ty::Dyn.default_lit().to_string(), Ty::Dyn)
                } else {
                    (ec, et)
                };
                let (ty, tc, ec) = match (tt, et) {
                    (a, b) if a == b => (a, tc, ec),
                    (Ty::Float, Ty::Int) => (Ty::Float, tc, format!("{} as f64", ec)),
                    (Ty::Int, Ty::Float) => (Ty::Float, format!("{} as f64", tc), ec),
                    (Ty::Dyn, _) => {
                        let ec = self.box_code(ec, et, line)?;
                        (Ty::Dyn, tc, ec)
                    }
                    (_, Ty::Dyn) => {
                        let tc = self.box_code(tc, tt, line)?;
                        (Ty::Dyn, tc, ec)
                    }
                    (a, b) if a.is_scalar() && b.is_scalar() => {
                        // Mixed scalar branches (int/bool, float/bool):
                        // Python pins no single type, so the value is
                        // dynamic and both branches are boxed.
                        let tc = self.box_code(tc, tt, line)?;
                        let ec = self.box_code(ec, et, line)?;
                        (Ty::Dyn, tc, ec)
                    }
                    _ => {
                        return Err(eerr(
                            line,
                            &format!(
                                "if-expression branches have incompatible types ({} and {})",
                                tt.py_name(),
                                et.py_name()
                            ),
                        ))
                    }
                };
                ctx.tmp += 1;
                let name = format!("__psa_ifexp_{}", ctx.tmp);
                if ty == Ty::Dyn && ctx.scopes.len() > 1 {
                    // A block-scoped aggregate temp is never retired
                    // on this route (probe q29); the temp's `let`
                    // goes to the function prologue instead.
                    ctx.hoist_declare(&name, ty);
                } else {
                    ctx.declare(&name, ty).map_err(|m| eerr(line, &m))?;
                    ctx.line(&format!("let {}: {} = {};", name, ty.sla(), ty.default_lit()));
                }
                ctx.line(&format!("if {} {{", cc));
                ctx.out.append_writer(then_lines);
                ctx.indent += 1;
                if ty == Ty::Dyn {
                    if then_void {
                        ctx.line(&format!("{}.tag = 3;", name));
                        ctx.line(&format!("{}.ipay = 0;", name));
                        ctx.line(&format!("{}.fpay = 0.0;", name));
                    } else {
                        self.emit_dyn_store(ctx, &name, &tc)
                            .map_err(|m| eerr(line, &m))?;
                    }
                } else {
                    ctx.line(&format!("{} = {};", name, tc));
                }
                ctx.indent -= 1;
                ctx.line("} else {");
                ctx.out.append_writer(els_lines);
                ctx.indent += 1;
                if ty == Ty::Dyn {
                    if els_void {
                        ctx.line(&format!("{}.tag = 3;", name));
                        ctx.line(&format!("{}.ipay = 0;", name));
                        ctx.line(&format!("{}.fpay = 0.0;", name));
                    } else {
                        self.emit_dyn_store(ctx, &name, &ec)
                            .map_err(|m| eerr(line, &m))?;
                    }
                } else {
                    ctx.line(&format!("{} = {};", name, ec));
                }
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
    ) -> Result<(String, Ty), Diagnostic> {
        let (lc, lt) = self.emit_expr(ctx, l, line)?;
        let (rc, rt) = self.emit_expr(ctx, r, line)?;
        if lt == Ty::Dyn || rt == Ty::Dyn {
            // The runtime can branch on the exponent's actual value,
            // so the static path's literal-exponent restriction does
            // not apply to dynamic operands.
            let lc = self.box_code(lc, lt, line)?;
            let rc = self.box_code(rc, rt, line)?;
            self.used_dyn = true;
            return Ok((format!("__psa_dyn_pow({}, {})", lc, rc), Ty::Dyn));
        }
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

    /// `and`/`or` with a dynamic operand: Python returns one of the
    /// operands (not a bool), so the left value is materialised in a
    /// temp and the right operand's code is placed inside the branch
    /// that actually evaluates it (short-circuit preserved, exactly
    /// like the if-expression temp machinery).
    fn emit_dyn_logic(
        &mut self,
        ctx: &mut Ctx,
        lc: String,
        lt: Ty,
        op: BinOp,
        rc: String,
        rt: Ty,
        r_lines: SegmentWriter,
        line: u32,
    ) -> Result<(String, Ty), Diagnostic> {
        if ctx.in_while_cond {
            return Err(eerr(
                line,
                "and/or with a dynamic operand is not supported in while conditions in D1 (the condition is re-evaluated every iteration); compute the value before the loop",
            ));
        }
        let lc = self.box_code(lc, lt, line)?;
        let rc = self.box_code(rc, rt, line)?;
        self.used_dyn = true;
        ctx.tmp += 1;
        let name = format!("__psa_logic_{}", ctx.tmp);
        if ctx.scopes.len() > 1 {
            ctx.hoist_declare(&name, Ty::Dyn);
        } else {
            ctx.declare(&name, Ty::Dyn).map_err(|m| eerr(line, &m))?;
            ctx.line(&format!("let {}: PsaVal = {};", name, Ty::Dyn.default_lit()));
        }
        self.emit_dyn_store(ctx, &name, &lc)
            .map_err(|m| eerr(line, &m))?;
        if op == BinOp::And {
            ctx.line(&format!("if __psa_dyn_truthy({}) {{", name));
        } else {
            ctx.line(&format!("if !(__psa_dyn_truthy({})) {{", name));
        }
        ctx.out.append_writer(r_lines);
        ctx.indent += 1;
        self.emit_dyn_store(ctx, &name, &rc)
            .map_err(|m| eerr(line, &m))?;
        ctx.indent -= 1;
        ctx.line("};");
        Ok((name, Ty::Dyn))
    }

    /// A binary operation with at least one dynamic operand: the
    /// static side is boxed and the operation dispatches through the
    /// PsaVal runtime. Comparisons return a plain bool.
    fn emit_dyn_binop(
        &mut self,
        lc: String,
        lt: Ty,
        op: BinOp,
        rc: String,
        rt: Ty,
        line: u32,
    ) -> Result<(String, Ty), Diagnostic> {
        let lc = self.box_code(lc, lt, line)?;
        let rc = self.box_code(rc, rt, line)?;
        self.used_dyn = true;
        let dyn_fn = match op {
            BinOp::Add => "__psa_dyn_add",
            BinOp::Sub => "__psa_dyn_sub",
            BinOp::Mul => "__psa_dyn_mul",
            BinOp::Div => "__psa_dyn_div",
            BinOp::FloorDiv => "__psa_dyn_floordiv",
            BinOp::Mod => "__psa_dyn_mod",
            BinOp::Pow => "__psa_dyn_pow",
            BinOp::BitAnd => "__psa_dyn_bitand",
            BinOp::BitOr => "__psa_dyn_bitor",
            BinOp::BitXor => "__psa_dyn_bitxor",
            BinOp::Shl => "__psa_dyn_shl",
            BinOp::Shr => "__psa_dyn_shr",
            BinOp::In => return Ok((format!("__psa_dyn_contains({}, {})", rc, lc), Ty::Bool)),
            BinOp::NotIn => {
                return Ok((format!("!(__psa_dyn_contains({}, {}))", rc, lc), Ty::Bool))
            }
            BinOp::Eq => return Ok((format!("__psa_dyn_eq({}, {})", lc, rc), Ty::Bool)),
            BinOp::NotEq => return Ok((format!("__psa_dyn_ne({}, {})", lc, rc), Ty::Bool)),
            BinOp::Is => return Ok((format!("__psa_dyn_is({}, {})", lc, rc), Ty::Bool)),
            BinOp::IsNot => {
                return Ok((format!("!(__psa_dyn_is({}, {}))", lc, rc), Ty::Bool))
            }
            BinOp::Lt => return Ok((format!("__psa_dyn_lt({}, {})", lc, rc), Ty::Bool)),
            BinOp::LtEq => return Ok((format!("__psa_dyn_le({}, {})", lc, rc), Ty::Bool)),
            BinOp::Gt => return Ok((format!("__psa_dyn_gt({}, {})", lc, rc), Ty::Bool)),
            BinOp::GtEq => return Ok((format!("__psa_dyn_ge({}, {})", lc, rc), Ty::Bool)),
            BinOp::And | BinOp::Or => {
                return Err(eerr(
                    line,
                    "internal error: dynamic and/or is handled by emit_dyn_logic",
                ))
            }
        };
        Ok((format!("{}({}, {})", dyn_fn, lc, rc), Ty::Dyn))
    }

    fn emit_binop(
        &mut self,
        lc: String,
        lt: Ty,
        op: BinOp,
        rc: String,
        rt: Ty,
        line: u32,
    ) -> Result<(String, Ty), Diagnostic> {
        if lt == Ty::Dyn || rt == Ty::Dyn {
            return self.emit_dyn_binop(lc, lt, op, rc, rt, line);
        }
        match op {
            BinOp::In | BinOp::NotIn => Err(eerr(
                line,
                "`in` is currently supported on dynamic lists only (D2); a static list operand stays refused rather than miscompiled",
            )),
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
            BinOp::Is | BinOp::IsNot => {
                // Identity on static operands (171-B,
                // SEMANTICS.md §6): same-kind scalars compare
                // payloads (the documented immediate rule);
                // mixed scalar kinds box and go through the
                // dynamic test so the tag decides (`1 is 1.0`
                // and `True is 1` are False, as in CPython);
                // anything else has no heap identity to test
                // and stays refused rather than miscompiled.
                let neg = op == BinOp::IsNot;
                match (lt, rt) {
                    (Ty::Int, Ty::Int) | (Ty::Bool, Ty::Bool) => {
                        let sym = if neg { "!=" } else { "==" };
                        Ok((format!("({} {} {})", lc, sym, rc), Ty::Bool))
                    }
                    (Ty::Float, Ty::Float) => {
                        // Payload equality with the NaN clause:
                        // a NaN is identical to itself.
                        let eq = format!(
                            "(({} == {}) || ({} != {} && {} != {}))",
                            lc, rc, lc, lc, rc, rc
                        );
                        if neg {
                            Ok((format!("!{}", eq), Ty::Bool))
                        } else {
                            Ok((eq, Ty::Bool))
                        }
                    }
                    (Ty::Int | Ty::Float | Ty::Bool, Ty::Int | Ty::Float | Ty::Bool) => {
                        let lc = self.box_code(lc, lt, line)?;
                        let rc = self.box_code(rc, rt, line)?;
                        if neg {
                            Ok((format!("!(__psa_dyn_is({}, {}))", lc, rc), Ty::Bool))
                        } else {
                            Ok((format!("__psa_dyn_is({}, {})", lc, rc), Ty::Bool))
                        }
                    }
                    _ => Err(eerr(
                        line,
                        "`is` is currently supported on dynamic values and static scalars; a static string/list operand stays refused rather than miscompiled",
                    )),
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
            Expr::Call(callee, args, kwargs, _) => {
                let gname = match &**callee {
                    Expr::Name(n, _) => Some(n.clone()),
                    _ => None,
                };
                for (i, a) in args.iter().enumerate() {
                    if let Expr::Name(n, _) = a {
                        if n == param {
                            if let (Some(g), Some(gdef)) =
                                (gname.clone(), gname.as_ref().and_then(|g| em.defs.get(g)))
                            {
                                if let Some(gparam) = gdef.params.get(i) {
                                    let key = (g.clone(), gparam.0.clone());
                                    if visiting.contains(&key) {
                                        return true;
                                    }
                                    visiting.push(key);
                                    let used =
                                        param_usage(em, gdef, &gparam.0, visiting);
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
                for (k, v) in kwargs.iter() {
                    if let Expr::Name(n, _) = v {
                        if n == param {
                            if let (Some(g), Some(gdef)) =
                                (gname.clone(), gname.as_ref().and_then(|g| em.defs.get(g)))
                            {
                                if let Some(gparam) =
                                    gdef.params.iter().find(|(pn, _)| pn == k)
                                {
                                    let key = (g.clone(), gparam.0.clone());
                                    if visiting.contains(&key) {
                                        return true;
                                    }
                                    visiting.push(key);
                                    let used =
                                        param_usage(em, gdef, &gparam.0, visiting);
                                    visiting.pop();
                                    if used {
                                        return true;
                                    }
                                }
                            }
                            // Builtins borrow; unknown callees were
                            // already rejected by ensure_fn upstream.
                            continue;
                        }
                    }
                    if expr_consumes(em, v, param, visiting) {
                        return true;
                    }
                }
                false
            }
            Expr::Bin(l, _, r) => {
                expr_consumes(em, l, param, visiting) || expr_consumes(em, r, param, visiting)
            }
            Expr::Unary(_, i) => expr_consumes(em, i, param, visiting),
            Expr::Index(base, idx, _) => {
                // Indexing the parameter itself is a read (borrow).
                let _ = base;
                expr_consumes(em, idx, param, visiting)
            }
            Expr::Slice(base, lo, hi, st) => {
                let _ = base;
                [lo, hi, st]
                    .iter()
                    .any(|o| o.as_ref().map(|e| expr_consumes(em, e, param, visiting)).unwrap_or(false))
            }
            Expr::MethodCall { args, .. } => {
                args.iter().any(|a| expr_consumes(em, a, param, visiting))
            }
            Expr::Attr(base, _, _) => {
                // Reading an attribute of the parameter is a read (borrow).
                let _ = base;
                false
            }
            Expr::ListLit(elems) => elems.iter().any(|a| expr_consumes(em, a, param, visiting)),
            Expr::DictLit(pairs) => pairs.iter().any(|(k, v)| {
                expr_consumes(em, k, param, visiting) || expr_consumes(em, v, param, visiting)
            }),
            Expr::TupleLit(elems) | Expr::SetLit(elems) => {
                elems.iter().any(|a| expr_consumes(em, a, param, visiting))
            }
            Expr::Comp { key, value, clauses, .. } => {
                expr_consumes(em, key, param, visiting)
                    || value
                        .as_ref()
                        .map(|v| expr_consumes(em, v, param, visiting))
                        .unwrap_or(false)
                    || clauses.iter().any(|c| match c {
                        CompClause::For { iter, .. } => expr_consumes(em, iter, param, visiting),
                        CompClause::If(cond) => expr_consumes(em, cond, param, visiting),
                    })
            }
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
                    matches!(value, Expr::Name(n, _) if n == param)
                        || expr_consumes(em, value, param, visiting)
                }
                Stmt::AugAssign { value, .. } => expr_consumes(em, value, param, visiting),
                Stmt::ExprStmt { value, .. } => expr_consumes(em, value, param, visiting),
                Stmt::Return { value, .. } => match value {
                    Some(Expr::Name(n, _)) => n == param,
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
                Stmt::While { cond, body, els, .. } => {
                    expr_consumes(em, cond, param, visiting)
                        || stmts_consume(em, body, param, visiting)
                        || els
                            .as_ref()
                            .map(|b| stmts_consume(em, b, param, visiting))
                            .unwrap_or(false)
                }
                Stmt::ForRange { start, stop, body, els, .. } => {
                    start
                        .as_ref()
                        .map(|e| expr_consumes(em, e, param, visiting))
                        .unwrap_or(false)
                        || expr_consumes(em, stop, param, visiting)
                        || stmts_consume(em, body, param, visiting)
                        || els
                            .as_ref()
                            .map(|b| stmts_consume(em, b, param, visiting))
                            .unwrap_or(false)
                }
                Stmt::IndexAssign { value, index, .. } => {
                    expr_consumes(em, value, param, visiting)
                        || expr_consumes(em, index, param, visiting)
                }
                Stmt::AttrAssign { base, value, .. } => {
                    expr_consumes(em, base, param, visiting)
                        || expr_consumes(em, value, param, visiting)
                }
                Stmt::AttrDel { base, .. } => expr_consumes(em, base, param, visiting),
                Stmt::Unpack { value, .. } => expr_consumes(em, value, param, visiting),
                Stmt::ForEach { iter, body, els, .. } => {
                    expr_consumes(em, iter, param, visiting)
                        || stmts_consume(em, body, param, visiting)
                        || els
                            .as_ref()
                            .map(|b| stmts_consume(em, b, param, visiting))
                            .unwrap_or(false)
                }
                Stmt::Try { body, handlers, els, finally, .. } => {
                    stmts_consume(em, body, param, visiting)
                        || handlers.iter().any(|h| stmts_consume(em, &h.body, param, visiting))
                        || els.as_ref().map(|b| stmts_consume(em, b, param, visiting)).unwrap_or(false)
                        || finally
                            .as_ref()
                            .map(|b| stmts_consume(em, b, param, visiting))
                            .unwrap_or(false)
                }
                Stmt::With { items, body, .. } => {
                    items.iter().any(|(e, _)| expr_consumes(em, e, param, visiting))
                        || stmts_consume(em, body, param, visiting)
                }
                Stmt::Raise { value, .. } => value
                    .as_ref()
                    .map(|e| expr_consumes(em, e, param, visiting))
                    .unwrap_or(false),
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

/// May a call with an argument of type `arg` use a signature whose
/// parameter was inferred as `param`? Exact matches always may; a
/// dynamic parameter accepts any D1 scalar (boxed at the call site)
/// and a scalar parameter accepts a dynamic argument (unboxed with a
/// runtime type assertion). Everything else stays monomorphic.
fn sig_param_compatible(param: Ty, arg: Ty) -> bool {
    param == arg
        || (param == Ty::Dyn && arg.is_scalar())
        || (arg == Ty::Dyn && param.is_scalar())
}

/// D7: a def is a generator when a `yield` statement appears
/// in its own body (nested blocks included; there are no
/// nested defs in psa, so every yield found belongs to it).
fn func_is_generator(f: &Func) -> bool {
    fn stmts_have_yield(stmts: &[Stmt]) -> bool {
        stmts.iter().any(|s| match s {
            Stmt::Yield { .. } => true,
            Stmt::If { arms, els, .. } => {
                arms.iter().any(|(_, b)| stmts_have_yield(b))
                    || els.as_ref().map(|b| stmts_have_yield(b)).unwrap_or(false)
            }
            Stmt::While { body, els, .. }
            | Stmt::ForRange { body, els, .. }
            | Stmt::ForEach { body, els, .. } => {
                stmts_have_yield(body)
                    || els.as_ref().map(|b| stmts_have_yield(b)).unwrap_or(false)
            }
            Stmt::With { body, .. } => stmts_have_yield(body),
            Stmt::Try {
                body,
                handlers,
                els,
                finally,
                ..
            } => {
                stmts_have_yield(body)
                    || handlers.iter().any(|h| stmts_have_yield(&h.body))
                    || els.as_ref().map(|b| stmts_have_yield(b)).unwrap_or(false)
                    || finally
                        .as_ref()
                        .map(|b| stmts_have_yield(b))
                        .unwrap_or(false)
            }
            Stmt::Match { cases, .. } => {
                cases.iter().any(|c| stmts_have_yield(&c.body))
            }
            _ => false,
        })
    }
    stmts_have_yield(&f.body)
}

/// D7: a generator body may only return bare / None -- a
/// returned value would become StopIteration's payload in
/// CPython, which psa v1 does not model (loud).
fn gen_check_returns(stmts: &[Stmt], fname: &str, line: u32) -> Result<(), Diagnostic> {
    for s in stmts {
        match s {
            Stmt::Return { value, .. } => {
                if let Some(e) = value {
                    if !matches!(e, Expr::NoneLit) {
                        return Err(eerr(
                            line,
                            &format!(
                                "generator `{}` returns a value; psa v1 generators may only return bare / None (the value would be StopIteration's payload)",
                                fname
                            ),
                        ));
                    }
                }
            }
            Stmt::If { arms, els, .. } => {
                for (_, b) in arms {
                    gen_check_returns(b, fname, line)?;
                }
                if let Some(b) = els {
                    gen_check_returns(b, fname, line)?;
                }
            }
            Stmt::While { body, els, .. }
            | Stmt::ForRange { body, els, .. }
            | Stmt::ForEach { body, els, .. } => {
                gen_check_returns(body, fname, line)?;
                if let Some(b) = els {
                    gen_check_returns(b, fname, line)?;
                }
            }
            Stmt::With { body, .. } => gen_check_returns(body, fname, line)?,
            Stmt::Try {
                body,
                handlers,
                els,
                finally,
                ..
            } => {
                gen_check_returns(body, fname, line)?;
                for h in handlers {
                    gen_check_returns(&h.body, fname, line)?;
                }
                if let Some(b) = els {
                    gen_check_returns(b, fname, line)?;
                }
                if let Some(b) = finally {
                    gen_check_returns(b, fname, line)?;
                }
            }
            Stmt::Match { cases, .. } => {
                for c in cases {
                    gen_check_returns(&c.body, fname, line)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// D7: the capture names a pattern binds, in traversal order
/// (duplicates preserved; callers validate).
fn pattern_captures(p: &Pattern) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    fn walk(p: &Pattern, out: &mut Vec<String>) {
        match p {
            Pattern::Capture(n) => out.push(n.clone()),
            Pattern::Seq(pats) | Pattern::Or(pats) => {
                for x in pats {
                    walk(x, out);
                }
            }
            Pattern::Lit(_) | Pattern::Wildcard => {}
        }
    }
    walk(p, &mut out);
    out
}

fn unify_ret(ctx: &mut Ctx, ty: Ty, line: u32) -> Result<(), Diagnostic> {
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
            } else if (existing == Ty::Void && (ty.is_scalar() || ty == Ty::Dyn))
                || (ty == Ty::Void && (existing.is_scalar() || existing == Ty::Dyn))
            {
                // A bare `return` (None) mixed with value returns:
                // the function's result is dynamic, None on that path.
                ctx.ret = Some(Ty::Dyn);
                Ok(())
            } else if (existing.is_scalar() && ty.is_scalar())
                || (existing == Ty::Dyn && ty.is_scalar())
                || (ty == Ty::Dyn && existing.is_scalar())
            {
                // Mixed scalar returns (or a dynamic return among
                // scalar ones) make the function's return dynamic:
                // Python would not pin one type here either.
                ctx.ret = Some(Ty::Dyn);
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

/// D6: does this statement contain a `raise` anywhere the try
/// machinery of an ENCLOSING try would see it? Nested try
/// statements are opaque here: their own clauses are validated
/// when the nested try itself is emitted.
fn deep_has_raise(s: &Stmt) -> bool {
    match s {
        Stmt::Raise { .. } => true,
        Stmt::If { arms, els, .. } => {
            arms.iter().any(|(_, b)| b.iter().any(deep_has_raise))
                || els.as_ref().map(|b| b.iter().any(deep_has_raise)).unwrap_or(false)
        }
        Stmt::While { body, els, .. }
        | Stmt::ForRange { body, els, .. }
        | Stmt::ForEach { body, els, .. } => {
            body.iter().any(deep_has_raise)
                || els
                    .as_ref()
                    .map(|b| b.iter().any(deep_has_raise))
                    .unwrap_or(false)
        }
        Stmt::With { body, .. } => body.iter().any(deep_has_raise),
        Stmt::Try { .. } => false,
        _ => false,
    }
}

/// D6 static checks on a try statement's clauses: raises
/// inside a handler are restricted to a final bare `raise`
/// (re-raise), and raises inside a finally are refused --
/// both because the v1 machinery gives those positions no
/// sound propagation path (see D6-NOTES).
fn validate_try(
    handlers: &[ExcHandler],
    finally: &Option<Vec<Stmt>>,
    line: u32,
) -> Result<(), Diagnostic> {
    for h in handlers {
        let n = h.body.len();
        for (i, s) in h.body.iter().enumerate() {
            if i + 1 < n {
                if deep_has_raise(s) {
                    return Err(eerr(
                        h.line,
                        "`raise` inside an except handler is only allowed as the handler's final bare `raise` in v1 (D6)",
                    ));
                }
            } else {
                match s {
                    Stmt::Raise { value: None, .. } => {}
                    Stmt::Raise { .. } => {
                        return Err(eerr(
                            h.line,
                            "only a bare `raise` (re-raising the caught exception) is allowed inside an except handler in v1 (D6)",
                        ));
                    }
                    other => {
                        if deep_has_raise(other) {
                            return Err(eerr(
                                h.line,
                                "`raise` inside an except handler is only allowed as the handler's final bare `raise` in v1 (D6)",
                            ));
                        }
                    }
                }
            }
        }
    }
    if let Some(fin) = finally {
        for s in fin {
            if deep_has_raise(s) {
                return Err(eerr(
                    line,
                    "`raise` inside a finally block is not supported in v1 (D6)",
                ));
            }
        }
    }
    Ok(())
}

fn def_calls(stmts: &[Stmt], name: &str) -> bool {
    fn expr_calls(e: &Expr, name: &str) -> bool {
        match e {
            Expr::Call(callee, args, kwargs, _) => {
                if let Expr::Name(n, _) = &**callee {
                    if n == name {
                        return true;
                    }
                }
                args.iter().any(|a| expr_calls(a, name))
                    || kwargs.iter().any(|(_, v)| expr_calls(v, name))
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
            Stmt::While { cond, body, els, .. } => {
                expr_calls(cond, name)
                    || def_calls(body, name)
                    || els.as_ref().map(|b| def_calls(b, name)).unwrap_or(false)
            }
            Stmt::ForRange { start, stop, body, els, .. } => {
                start.as_ref().map(|e| expr_calls(e, name)).unwrap_or(false)
                    || expr_calls(stop, name)
                    || def_calls(body, name)
                    || els.as_ref().map(|b| def_calls(b, name)).unwrap_or(false)
            }
            Stmt::ForEach { iter, body, els, .. } => {
                expr_calls(iter, name)
                    || def_calls(body, name)
                    || els.as_ref().map(|b| def_calls(b, name)).unwrap_or(false)
            }
            Stmt::Try { body, handlers, els, finally, .. } => {
                def_calls(body, name)
                    || handlers.iter().any(|h| def_calls(&h.body, name))
                    || els.as_ref().map(|b| def_calls(b, name)).unwrap_or(false)
                    || finally.as_ref().map(|b| def_calls(b, name)).unwrap_or(false)
            }
            Stmt::With { items, body, .. } => {
                items.iter().any(|(e, _)| expr_calls(e, name)) || def_calls(body, name)
            }
            Stmt::Raise { value, .. } => {
                value.as_ref().map(|e| expr_calls(e, name)).unwrap_or(false)
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
        Expr::Name(n, _) => n == name,
        Expr::Bin(a, _, b) => expr_mentions(a, name) || expr_mentions(b, name),
        Expr::Unary(_, a) => expr_mentions(a, name),
        Expr::Call(f, args, kwargs, _) => {
            expr_mentions(f, name)
                || args.iter().any(|a| expr_mentions(a, name))
                || kwargs.iter().any(|(_, v)| expr_mentions(v, name))
        }
        Expr::Index(a, b, _) => expr_mentions(a, name) || expr_mentions(b, name),
        Expr::Slice(a, lo, hi, st) => {
            expr_mentions(a, name)
                || [lo, hi, st]
                    .iter()
                    .any(|o| o.as_ref().map(|e| expr_mentions(e, name)).unwrap_or(false))
        }
        Expr::MethodCall { obj, args, .. } => {
            obj == name || args.iter().any(|a| expr_mentions(a, name))
        }
        Expr::ListLit(elems) => elems.iter().any(|a| expr_mentions(a, name)),
        Expr::DictLit(pairs) => pairs
            .iter()
            .any(|(k, v)| expr_mentions(k, name) || expr_mentions(v, name)),
        Expr::TupleLit(elems) | Expr::SetLit(elems) => {
            elems.iter().any(|a| expr_mentions(a, name))
        }
        Expr::Comp { key, value, clauses, .. } => {
            expr_mentions(key, name)
                || value.as_ref().map(|v| expr_mentions(v, name)).unwrap_or(false)
                || clauses.iter().any(|c| match c {
                    CompClause::For { var, iter } => var == name || expr_mentions(iter, name),
                    CompClause::If(cond) => expr_mentions(cond, name),
                })
        }
        Expr::Attr(base, _, _) => expr_mentions(base, name),
        Expr::Walrus { name: target, value } => {
            target == name || expr_mentions(value, name)
        }
        Expr::IfExp { cond, then, els } => {
            expr_mentions(cond, name)
                || expr_mentions(then, name)
                || expr_mentions(els, name)
        }
        Expr::FStr(parts) => parts
            .iter()
            .any(|p| p.any_expr(&|x| expr_mentions(x, name))),
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
        Expr::Index(a, b, _) => expr_has_call(a) || expr_has_call(b),
        Expr::Slice(a, lo, hi, st) => {
            expr_has_call(a)
                || [lo, hi, st]
                    .iter()
                    .any(|o| o.as_ref().map(|e| expr_has_call(e)).unwrap_or(false))
        }
        Expr::ListLit(elems) => elems.iter().any(expr_has_call),
        Expr::DictLit(pairs) => pairs.iter().any(|(k, v)| expr_has_call(k) || expr_has_call(v)),
        Expr::TupleLit(elems) | Expr::SetLit(elems) => elems.iter().any(expr_has_call),
        Expr::Comp { key, value, clauses, .. } => {
            expr_has_call(key)
                || value.as_ref().map(|v| expr_has_call(v)).unwrap_or(false)
                || clauses.iter().any(|c| match c {
                    CompClause::For { iter, .. } => expr_has_call(iter),
                    CompClause::If(cond) => expr_has_call(cond),
                })
        }
        Expr::Attr(base, _, _) => expr_has_call(base),
        Expr::IfExp { cond, then, els } => {
            expr_has_call(cond) || expr_has_call(then) || expr_has_call(els)
        }
        Expr::FStr(parts) => parts
            .iter()
            .any(|p| p.any_expr(&|x| expr_has_call(x))),
        // An await runs the coroutine worker synchronously
        // through __psa_coro_run — a call in every sense that
        // matters here (it can leave a pending exception).
        Expr::Await(a) => expr_has_call(a),
        Expr::Walrus { value, .. } => expr_has_call(value),
        // D8: cell and closure nodes wrap ordinary
        // expressions (a cell read/write is a runtime call
        // in the emitted text, but it cannot raise a psa
        // exception; what matters here is the wrapped
        // expression).
        Expr::CellNew(a) | Expr::CellGet(a) => expr_has_call(a),
        Expr::CellSet(a, b) => expr_has_call(a) || expr_has_call(b),
        Expr::ClosureVal(_, env) => expr_has_call(env),
        Expr::ModEnv => false,
        _ => false,
    }
}

/// Does the expression contain a walrus (assignment expression)
/// anywhere? Used by the while-condition emission: only a
/// walrus that IS the whole condition gets the per-iteration
/// restructuring; one nested deeper is refused loudly.
fn expr_contains_walrus(e: &Expr) -> bool {
    match e {
        Expr::Walrus { .. } => true,
        Expr::Bin(a, _, b) => expr_contains_walrus(a) || expr_contains_walrus(b),
        Expr::Unary(_, a) => expr_contains_walrus(a),
        Expr::Call(f, args, kwargs, _) => {
            expr_contains_walrus(f)
                || args.iter().any(expr_contains_walrus)
                || kwargs.iter().any(|(_, v)| expr_contains_walrus(v))
        }
        Expr::Index(a, b, _) => expr_contains_walrus(a) || expr_contains_walrus(b),
        Expr::Slice(a, lo, hi, st) => {
            expr_contains_walrus(a)
                || [lo, hi, st]
                    .iter()
                    .any(|o| o.as_ref().map(|e| expr_contains_walrus(e)).unwrap_or(false))
        }
        Expr::MethodCall { args, .. } => args.iter().any(expr_contains_walrus),
        Expr::ListLit(elems) | Expr::TupleLit(elems) | Expr::SetLit(elems) => {
            elems.iter().any(expr_contains_walrus)
        }
        Expr::DictLit(pairs) => pairs
            .iter()
            .any(|(k, v)| expr_contains_walrus(k) || expr_contains_walrus(v)),
        Expr::Comp { key, value, clauses, .. } => {
            expr_contains_walrus(key)
                || value.as_ref().map(|e| expr_contains_walrus(e)).unwrap_or(false)
                || clauses.iter().any(|c| match c {
                    CompClause::For { iter, .. } => expr_contains_walrus(iter),
                    CompClause::If(cond) => expr_contains_walrus(cond),
                })
        }
        Expr::Attr(base, _, _) => expr_contains_walrus(base),
        Expr::IfExp { cond, then, els } => {
            expr_contains_walrus(cond)
                || expr_contains_walrus(then)
                || expr_contains_walrus(els)
        }
        Expr::FStr(parts) => parts
            .iter()
            .any(|p| p.any_expr(&|x| expr_contains_walrus(x))),
        Expr::Await(a) => expr_contains_walrus(a),
        _ => false,
    }
}

/// True when the expression contains a CellSet — the node a
/// walrus with a cell / environment / global target is
/// rewritten into by closure conversion (see rewrite_expr's
/// Walrus arm). Scope-sensitive walrus scans that run AFTER
/// rewriting (the generator-expression refusal in
/// emit_genexp) must treat CellSet as a walrus: it binds the
/// enclosing scope's cell, which the synthetic generator def
/// cannot do on the enclosing scope's behalf.
fn expr_contains_cellset(e: &Expr) -> bool {
    match e {
        Expr::CellSet(..) => true,
        Expr::Bin(a, _, b) => expr_contains_cellset(a) || expr_contains_cellset(b),
        Expr::Unary(_, a) => expr_contains_cellset(a),
        Expr::Call(f, args, kwargs, _) => {
            expr_contains_cellset(f)
                || args.iter().any(expr_contains_cellset)
                || kwargs.iter().any(|(_, v)| expr_contains_cellset(v))
        }
        Expr::Index(a, b, _) => expr_contains_cellset(a) || expr_contains_cellset(b),
        Expr::Slice(a, lo, hi, st) => {
            expr_contains_cellset(a)
                || [lo, hi, st]
                    .iter()
                    .any(|o| o.as_ref().map(|e| expr_contains_cellset(e)).unwrap_or(false))
        }
        Expr::MethodCall { args, .. } => args.iter().any(expr_contains_cellset),
        Expr::ListLit(elems) | Expr::TupleLit(elems) | Expr::SetLit(elems) => {
            elems.iter().any(expr_contains_cellset)
        }
        Expr::DictLit(pairs) => pairs
            .iter()
            .any(|(k, v)| expr_contains_cellset(k) || expr_contains_cellset(v)),
        Expr::Comp { key, value, clauses, .. } => {
            expr_contains_cellset(key)
                || value.as_ref().map(|e| expr_contains_cellset(e)).unwrap_or(false)
                || clauses.iter().any(|c| match c {
                    CompClause::For { iter, .. } => expr_contains_cellset(iter),
                    CompClause::If(cond) => expr_contains_cellset(cond),
                })
        }
        Expr::Attr(base, _, _) => expr_contains_cellset(base),
        Expr::IfExp { cond, then, els } => {
            expr_contains_cellset(cond)
                || expr_contains_cellset(then)
                || expr_contains_cellset(els)
        }
        Expr::FStr(parts) => parts
            .iter()
            .any(|p| p.any_expr(&|x| expr_contains_cellset(x))),
        Expr::Await(a) => expr_contains_cellset(a),
        Expr::CellNew(a) | Expr::CellGet(a) => expr_contains_cellset(a),
        Expr::Walrus { value, .. } => expr_contains_cellset(value),
        Expr::ClosureVal(_, env) => expr_contains_cellset(env),
        _ => false,
    }
}

/// Walrus targets inside f-string replacement fields, including
/// fields nested inside a format spec (fstring batch: FStrPart::Field
/// replaced the old FStrPart::Expr).
fn collect_fstr_walrus_targets(parts: &[FStrPart], out: &mut Vec<String>) {
    for p in parts {
        if let FStrPart::Field { value, spec, .. } = p {
            collect_walrus_targets(value, out);
            if let Some(spec) = spec {
                collect_fstr_walrus_targets(spec, out);
            }
        }
    }
}

/// Collect every walrus target an expression binds, in
/// traversal order (duplicates kept; callers deduplicate by
/// lookup). Nested comprehensions are entered — their walrus
/// targets bind in the containing function scope, like this
/// expression's own (CPython's comprehension rule). Lambda
/// bodies are NOT entered: a walrus there binds in the
/// lambda's own scope.
fn collect_walrus_targets(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Walrus { name, value } => {
            out.push(name.clone());
            collect_walrus_targets(value, out);
        }
        Expr::Bin(a, _, b) => {
            collect_walrus_targets(a, out);
            collect_walrus_targets(b, out);
        }
        Expr::Unary(_, a) => collect_walrus_targets(a, out),
        Expr::Call(f, args, kwargs, _) => {
            collect_walrus_targets(f, out);
            for a in args {
                collect_walrus_targets(a, out);
            }
            for (_, v) in kwargs {
                collect_walrus_targets(v, out);
            }
        }
        Expr::Index(a, b, _) => {
            collect_walrus_targets(a, out);
            collect_walrus_targets(b, out);
        }
        Expr::Slice(a, lo, hi, st) => {
            collect_walrus_targets(a, out);
            for o in [lo, hi, st] {
                if let Some(x) = o {
                    collect_walrus_targets(x, out);
                }
            }
        }
        Expr::MethodCall { args, .. } => {
            for a in args {
                collect_walrus_targets(a, out);
            }
        }
        Expr::ListLit(elems) | Expr::TupleLit(elems) | Expr::SetLit(elems) => {
            for a in elems {
                collect_walrus_targets(a, out);
            }
        }
        Expr::DictLit(pairs) => {
            for (k, v) in pairs {
                collect_walrus_targets(k, out);
                collect_walrus_targets(v, out);
            }
        }
        Expr::Comp { key, value, clauses, .. } => {
            collect_walrus_targets(key, out);
            if let Some(v) = value {
                collect_walrus_targets(v, out);
            }
            for c in clauses {
                match c {
                    CompClause::For { iter, .. } => collect_walrus_targets(iter, out),
                    CompClause::If(cond) => collect_walrus_targets(cond, out),
                }
            }
        }
        Expr::Attr(base, _, _) => collect_walrus_targets(base, out),
        Expr::IfExp { cond, then, els } => {
            collect_walrus_targets(cond, out);
            collect_walrus_targets(then, out);
            collect_walrus_targets(els, out);
        }
        Expr::FStr(parts) => collect_fstr_walrus_targets(parts, out),
        Expr::Await(a) => collect_walrus_targets(a, out),
        Expr::CellNew(a) | Expr::CellGet(a) => collect_walrus_targets(a, out),
        Expr::CellSet(a, b) => {
            collect_walrus_targets(a, out);
            collect_walrus_targets(b, out);
        }
        Expr::ClosureVal(_, env) => collect_walrus_targets(env, out),
        _ => {}
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
        let s = format!("{}", v);
        // Rust's Display gives the shortest round-trip decimal,
        // which for large integral floats is a bare integer string
        // (2^63 prints as 9223372036854776000). SLA lexes a bare
        // integer as an i64 first and rejects it past i64::MAX, so
        // keep the literal visibly floating: appending `.0` parses
        // back to the identical f64.
        if s.bytes().all(|b| b.is_ascii_digit() || b == b'-') {
            format!("{}.0", s)
        } else {
            s
        }
    }
}

/// The D1 dynamic runtime, emitted once when any dynamic value is
/// used. PsaVal tags: 0 = int, 1 = float, 2 = bool, 3 = None,
/// 4 = list, 5 = dict, 6 = str, 7 = fn, 8 = object, 9 = bound
/// method, 10 = tuple, 11 = set, 12 = file, 13 = generator,
/// 14 = coroutine, 15 = bytes (same Box<Vec<u8>> payload as
/// str, byte-wise semantics), 16 = closure (D8; payload mirrors
/// the bound-method layout, tag 9).
/// The payload is split into an integer slot and a float slot (the
/// design's single u64 payload would need an f64 bitcast primitive;
/// the split representation has identical semantics). D1 values are
/// inline scalars with no identity, so no reference counting yet:
/// the Perceus rc discipline arrives with the heap kinds in D2.
/// Panic codes are psa's runtime TypeError surface: 7701x operand
/// kind errors, 7702x unboxing failures, 77030 float print (blocked
/// by upstream defect (xxiii): the value is correct, only its text
/// form is unavailable until the f64 println truncation is fixed).
/// PsaStage/__psa_stage_of/__psa_drop_stage implement the store
/// pattern of emit_dyn_store: the SAB lowering cannot reassign an
/// aggregate variable (RegisterRedefinition) nor retire a
/// block-scoped aggregate local (MemoryLeak) — upstream defect
/// candidate (xxvii), evidence probes q1-q29 in the sapy workdir.
/// psa_std json module text (psa-std/json.sla in the repo),
/// spliced into programs that call json.dumps / json.loads.
const PSA_JSON_SLA: &str = include_str!("../../../psa-std/json.sla");
const PSA_PICKLE_SLA: &str = include_str!("../../../psa-std/pickle.sla");

/// psa_std math module text (psa-std/math.sla in the repo),
/// spliced into programs that call math.<fn>.
const PSA_MATH_SLA: &str = include_str!("../../../psa-std/math.sla");

/// psa_std string module text (psa-std/string.sla in the
/// repo), spliced into programs that call string.capwords.
const PSA_STRING_SLA: &str = include_str!("../../../psa-std/string.sla");

/// psa_std csv module text (psa-std/csv.sla in the repo),
/// spliced into programs that call csv.<fn>.
const PSA_CSV_SLA: &str = include_str!("../../../psa-std/csv.sla");

/// psa_std logging module text (psa-std/logging.sla in
/// the repo), spliced into programs that call
/// logging.<fn>. The text carries its own @import lines.
const PSA_LOGGING_SLA: &str = include_str!("../../../psa-std/logging.sla");

/// psa_std argparse module text (psa-std/argparse.sla in
/// the repo), spliced into programs that call
/// argparse.ArgumentParser. The text carries its own
/// @import lines.
const PSA_ARGPARSE_SLA: &str = include_str!("../../../psa-std/argparse.sla");

/// psa_std collections module text
/// (psa-std/collections.sla in the repo), spliced into
/// programs that call collections.<fn>.
const PSA_COLLECTIONS_SLA: &str = include_str!("../../../psa-std/collections.sla");

/// psa_std itertools module text (psa-std/itertools.sla
/// in the repo), spliced into programs that call
/// itertools.<fn>.
const PSA_ITERTOOLS_SLA: &str = include_str!("../../../psa-std/itertools.sla");

/// psa_std functools module text (psa-std/functools.sla
/// in the repo), spliced into programs that call
/// functools.<fn>.
const PSA_FUNCTOOLS_SLA: &str = include_str!("../../../psa-std/functools.sla");

/// psa_std bisect module text (psa-std/bisect.sla in the
/// repo), spliced into programs that call bisect.<fn>.
const PSA_BISECT_SLA: &str = include_str!("../../../psa-std/bisect.sla");

/// psa_std heapq module text (psa-std/heapq.sla in the
/// repo), spliced into programs that call heapq.<fn>.
const PSA_HEAPQ_SLA: &str = include_str!("../../../psa-std/heapq.sla");

/// psa_std base64 module text (psa-std/base64.sla in the
/// repo), spliced into programs that call base64.<fn>.
const PSA_BASE64_SLA: &str = include_str!("../../../psa-std/base64.sla");

/// psa_std random module text (psa-std/random.sla in the
/// repo), spliced into programs that call random.<fn>.
const PSA_RANDOM_SLA: &str = include_str!("../../../psa-std/random.sla");

/// psa_std re module text (psa-std/re.sla in the repo),
/// spliced into programs that call re.<fn>.
const PSA_RE_SLA: &str = include_str!("../../../psa-std/re.sla");

/// psa_std socket module text (psa-std/socket.sla in
/// the repo), spliced into programs that call
/// socket.<fn>.
const PSA_SOCKET_SLA: &str = include_str!("../../../psa-std/socket.sla");

/// psa_std urllib module text (psa-std/urllib.sla in
/// the repo), spliced into programs that call
/// urllib.<fn>.
const PSA_URLLIB_SLA: &str = include_str!("../../../psa-std/urllib.sla");

/// D8 closures: the runtime text spliced (after
/// DYN_RUNTIME) only into programs that use a closure, a
/// cell or the module environment, so capture-free programs
/// emit byte-identical text. Tag 16 is a closure value: its
/// payload mirrors the bound-method layout (tag 9) — a heap
/// list [box_int(callable id), env] whose pointer is the
/// value's ipay; env is itself a dynamic list of cells. A
/// cell is a one-element dynamic list (tag 4). The module
/// environment — the module scope's cell list, in slot
/// order — is published by main into the sa_std globals
/// cell "psa.modenv" (the class-registry precedent) and
/// fetched back by __psa_modenv; 77083 is the loud panic
/// for a fetch before publication, which the emission
/// order makes unreachable (PublishModEnv precedes every
/// user statement in main).
const CLOSURE_RUNTIME: &str = r#"fn __psa_box_closure(id: i64, env: PsaVal) -> PsaVal {
    let xs: Vec<PsaVal> = Vec::new();
    let p = Box::into_raw(Box::new(xs)) as i64;
    __psa_vec_push_h(p as ptr, __psa_box_int(id));
    __psa_vec_push_h(p as ptr, env);
    return PsaVal { tag: 16, ipay: p, fpay: 0.0 };
}

fn __psa_closure_id(f: PsaVal) -> i64 {
    return __psa_vec_get_h(f.ipay as ptr, 0).ipay;
}

fn __psa_closure_env(f: PsaVal) -> PsaVal {
    return __psa_vec_get_h(f.ipay as ptr, 1);
}

fn __psa_cell_new(v: PsaVal) -> PsaVal {
    let c = __psa_list_new();
    __psa_list_push(c, v);
    return c;
}

fn __psa_cell_get(c: PsaVal) -> PsaVal {
    return __psa_list_get(c, 0);
}

fn __psa_cell_set(c: PsaVal, v: PsaVal) -> PsaVal {
    __psa_list_set(c, 0, v);
    return v;
}

fn __psa_modenv() -> PsaVal {
    let out = [0u64; 1];
    let rc = sa_global_u64_get(STR_PTR("psa.modenv"), STR_LEN("psa.modenv"), &out[0]);
    if rc != 0 { panic(77083); };
    return PsaVal { tag: 4, ipay: out[0] as i64, fpay: 0.0 };
}

fn __psa_modenv_set(env: PsaVal) {
    let rc = sa_global_u64_set(STR_PTR("psa.modenv"), STR_LEN("psa.modenv"), env.ipay as u64);
}

"#;

const DYN_RUNTIME: &str = r#"fn __psa_box_int(x: i64) -> PsaVal {
    return PsaVal { tag: 0, ipay: x, fpay: 0.0 };
}

fn __psa_box_float(x: f64) -> PsaVal {
    return PsaVal { tag: 1, ipay: 0, fpay: x };
}

fn __psa_box_bool(x: bool) -> PsaVal {
    if x { return PsaVal { tag: 2, ipay: 1, fpay: 0.0 }; };
    return PsaVal { tag: 2, ipay: 0, fpay: 0.0 };
}

fn __psa_clone(v: PsaVal) -> PsaVal {
    return PsaVal { tag: v.tag, ipay: v.ipay, fpay: v.fpay };
}

fn __psa_stage_of(v: PsaVal) -> PsaStage {
    return PsaStage { tag: v.tag, ipay: v.ipay, fpay: v.fpay };
}

fn __psa_drop_stage(s: PsaStage) -> void {
    return;
}


fn __psa_dyn_floor_f(x: f64) -> f64 {
    if x >= 9223372036854775808.0 { return x; };
    if x <= -9223372036854775808.0 { return x; };
    let q = x as i64;
    let qf = q as f64;
    if qf > x { return qf - 1.0; };
    return qf;
}




fn __psa_dyn_div(a: PsaVal, b: PsaVal) -> PsaVal {
    return __psa_box_float(__psa_as_float(a) / __psa_as_float(b));
}





fn __psa_dyn_ne(a: PsaVal, b: PsaVal) -> bool {
    return !(__psa_dyn_eq(a, b));
}

// Identity (171-B, SEMANTICS.md): tag+payload equality. For the
// boxed kinds the ipay payload IS the heap pointer, so this is
// pointer identity; for singletons (tag 3, tag 2) it is exact;
// for immediates it is the documented deterministic rule. The
// last line is the NaN clause: two NaN payloads are the same
// payload, so a NaN is identical to itself as in CPython.
fn __psa_dyn_is(a: PsaVal, b: PsaVal) -> bool {
    if a.tag != b.tag { return false; };
    if a.ipay != b.ipay { return false; };
    if a.fpay == b.fpay { return true; };
    return a.fpay != a.fpay && b.fpay != b.fpay;
}














fn __psa_unbox_float(v: PsaVal) -> f64 {
    if v.tag == 1 { return v.fpay; };
    if v.tag == 0 || v.tag == 2 { return v.ipay as f64; };
    panic(77021);
}

fn __psa_unbox_bool(v: PsaVal) -> bool {
    if v.tag == 2 { return v.ipay != 0; };
    panic(77022);
}

fn __psa_print_i64(x: i64) {
    let buf = [0u8; 32];
    let olen = [0u64; 4];
    let rc = sla_fmt_i64_into(x, 10, PTR_BYTE_ADD(&buf, 0), 32, &olen[0]);
    sla_print_bytes(PTR_BYTE_ADD(&buf, 0), olen[0]);
}


fn __psa_print_dict_repr(v: PsaVal) {
    sla_print_bytes(STR_PTR("{"), STR_LEN("{"));
    let n = __psa_dict_len(v);
    let i: i64 = 0;
    while i < n {
        if i > 0 { sla_print_bytes(STR_PTR(", "), STR_LEN(", ")); };
        __psa_print_repr(__psa_dict_key_at(v, i));
        sla_print_bytes(STR_PTR(": "), STR_LEN(": "));
        __psa_print_repr(__psa_dict_val_at(v, i));
        i = i + 1;
    }
    sla_print_bytes(STR_PTR("}"), STR_LEN("}"));
}

fn __psa_print_list_repr(v: PsaVal) {
    sla_print_bytes(STR_PTR("["), STR_LEN("["));
    let n = __psa_list_len(v);
    let i: i64 = 0;
    while i < n {
        if i > 0 { sla_print_bytes(STR_PTR(", "), STR_LEN(", ")); };
        __psa_print_repr(__psa_list_get(v, i));
        i = i + 1;
    }
    sla_print_bytes(STR_PTR("]"), STR_LEN("]"));
}



fn __psa_vec_push_h(p: ptr, x: PsaVal) {
    let b = unsafe { Box::from_raw(p as *Vec<PsaVal>) };
    (*b).push(x);
    let rp: i64 = Box::into_raw(b) as i64;
}

fn __psa_vec_get_h(p: ptr, i: i64) -> PsaVal {
    let b = unsafe { Box::from_raw(p as *Vec<PsaVal>) };
    let v = (*b)[i];
    let rp: i64 = Box::into_raw(b) as i64;
    return v;
}

fn __psa_vec_set_h(p: ptr, i: i64, x: PsaVal) {
    let b = unsafe { Box::from_raw(p as *Vec<PsaVal>) };
    (*b)[i] = x;
    let rp: i64 = Box::into_raw(b) as i64;
}

fn __psa_vec_len_h(p: ptr) -> i64 {
    let b = unsafe { Box::from_raw(p as *Vec<PsaVal>) };
    let n = (*b).len() as i64;
    let rp: i64 = Box::into_raw(b) as i64;
    return n;
}

fn __psa_list_new() -> PsaVal {
    let xs: Vec<PsaVal> = Vec::new();
    return PsaVal { tag: 4, ipay: Box::into_raw(Box::new(xs)) as i64, fpay: 0.0 };
}

fn __psa_list_len(v: PsaVal) -> i64 {
    if v.tag != 4 { panic(77052); };
    return __psa_vec_len_h(v.ipay as ptr);
}

fn __psa_list_norm(v: PsaVal, i: i64) -> i64 {
    let n = __psa_list_len(v);
    let j = i;
    if j < 0 { j = j + n; };
    if j < 0 || j >= n { panic(77050); };
    return j;
}

fn __psa_list_get(v: PsaVal, i: i64) -> PsaVal {
    if v.tag != 4 { panic(77052); };
    let j = __psa_list_norm(v, i);
    return __psa_vec_get_h(v.ipay as ptr, j);
}

fn __psa_list_set(v: PsaVal, i: i64, x: PsaVal) {
    if v.tag != 4 { panic(77052); };
    let j = __psa_list_norm(v, i);
    __psa_vec_set_h(v.ipay as ptr, j, x);
}

fn __psa_list_push(v: PsaVal, x: PsaVal) {
    if v.tag != 4 { panic(77052); };
    __psa_vec_push_h(v.ipay as ptr, x);
}

fn __psa_list_eq(a: PsaVal, b: PsaVal) -> bool {
    let n = __psa_list_len(a);
    if n != __psa_list_len(b) { return false; };
    let i: i64 = 0;
    while i < n {
        if !(__psa_dyn_eq(__psa_list_get(a, i), __psa_list_get(b, i))) { return false; };
        i = i + 1;
    }
    return true;
}

fn __psa_dyn_len(v: PsaVal) -> i64 {
    if v.tag == 4 { return __psa_list_len(v); };
    if v.tag == 5 { return __psa_dict_len(v); };
    if v.tag == 6 { return __psa_str_len(v); };
    if v.tag == 15 { return __psa_str_bytes_len(v); };
    if v.tag == 10 { return __psa_vec_len_h(v.ipay as ptr); };
    if v.tag == 11 { return __psa_vec_len_h(v.ipay as ptr); };
    panic(77052);
}

fn __psa_list_contains(v: PsaVal, x: PsaVal) -> bool {
    if v.tag != 4 { panic(77052); };
    let n = __psa_list_len(v);
    let i: i64 = 0;
    while i < n {
        if __psa_dyn_eq(__psa_list_get(v, i), x) { return true; };
        i = i + 1;
    }
    return false;
}

fn __psa_box_fn(id: i64) -> PsaVal {
    return PsaVal { tag: 7, ipay: id, fpay: 0.0 };
}

// Object storage (D4): the payload mirrors the dict layout
// with a one-element header -- element 0 is box_int(class_id),
// elements 1.. are alternating key/value entries in insertion
// order, tombstoned on deletion exactly like dict entries.
// Tag 9 is a bound method: a payload of len 2 holding
// [box_int(callable_id), self].
fn __psa_obj_new(cid: i64) -> PsaVal {
    let xs: Vec<PsaVal> = Vec::new();
    let p = Box::into_raw(Box::new(xs)) as i64;
    __psa_vec_push_h(p as ptr, __psa_box_int(cid));
    return PsaVal { tag: 8, ipay: p, fpay: 0.0 };
}

fn __psa_obj_class(v: PsaVal) -> i64 {
    // Total: a non-object has no class (-1 matches nothing).
    // The handler dispatch condition used to call this on
    // the empty (None) pending slot under SLA's eager &&,
    // dereferencing ipay 0 — a segfault on every try whose
    // body did not raise (exc1 batch 8).
    if v.tag != 8 { return -1; };
    return __psa_vec_get_h(v.ipay as ptr, 0).ipay;
}

// Entries-vec offset of the live entry whose key equals k, or
// -1 (the header occupies offset 0, so entries start at 1).
fn __psa_obj_find(v: PsaVal, k: PsaVal) -> i64 {
    let n = __psa_vec_len_h(v.ipay as ptr);
    let i: i64 = 1;
    while i < n {
        let kk = __psa_vec_get_h(v.ipay as ptr, i);
        if __psa_key_eq(kk, k) { return i; };
        i = i + 2;
    }
    return -1;
}

fn __psa_obj_set(v: PsaVal, k: PsaVal, x: PsaVal) {
    let j = __psa_obj_find(v, k);
    if j >= 0 { __psa_vec_set_h(v.ipay as ptr, j + 1, x); return; };
    __psa_vec_push_h(v.ipay as ptr, k);
    __psa_vec_push_h(v.ipay as ptr, x);
}

fn __psa_setattr(v: PsaVal, k: PsaVal, x: PsaVal) {
    if v.tag != 8 { panic(77072); };
    __psa_obj_set(v, k, x);
}

fn __psa_delattr(v: PsaVal, k: PsaVal) {
    if v.tag != 8 { panic(77069); };
    let j = __psa_obj_find(v, k);
    if j < 0 { panic(77069); };
    let tomb = PsaVal { tag: 4, ipay: 0, fpay: 0.0 };
    __psa_vec_set_h(v.ipay as ptr, j, tomb);
}

fn __psa_obj_check_noargs(n: i64) {
    if n != 0 { panic(77073); };
}

fn __psa_box_method(id: i64, selfv: PsaVal) -> PsaVal {
    let xs: Vec<PsaVal> = Vec::new();
    let p = Box::into_raw(Box::new(xs)) as i64;
    __psa_vec_push_h(p as ptr, __psa_box_int(id));
    __psa_vec_push_h(p as ptr, selfv);
    return PsaVal { tag: 9, ipay: p, fpay: 0.0 };
}

fn __psa_method_id(f: PsaVal) -> i64 {
    return __psa_vec_get_h(f.ipay as ptr, 0).ipay;
}

fn __psa_method_self(f: PsaVal) -> PsaVal {
    return __psa_vec_get_h(f.ipay as ptr, 1);
}

// Dict storage (D3 ordered dict): the payload is a Box<Vec<PsaVal>>
// of alternating key/value entries in insertion order, manipulated
// only through the proven __psa_vec_*_h helpers. Lookup is a linear
// scan under Python key equivalence; deletion writes a tombstone
// (a tag-4 key slot, which can never be a live key because list
// keys are refused as unhashable) so the remaining order is kept.
// Semantics are full Python: insertion order, overwrite keeps the
// original position, re-insertion after deletion goes to the end,
// str/float/bool/int/None keys. Lookup cost is O(n) per operation
// -- a hash index is a performance follow-up if the bench demands
// it, not a semantic gap.
fn __psa_dict_new() -> PsaVal {
    let xs: Vec<PsaVal> = Vec::new();
    return PsaVal { tag: 5, ipay: Box::into_raw(Box::new(xs)) as i64, fpay: 0.0 };
}


// Entries-vec offset of the live entry whose key equals k, or -1.
// Tombstone slots never match: their key tag is 4 and key_eq
// against tag 4 is always false.
fn __psa_dict_find(v: PsaVal, k: PsaVal) -> i64 {
    let n = __psa_vec_len_h(v.ipay as ptr);
    let i: i64 = 0;
    while i < n {
        let kk = __psa_vec_get_h(v.ipay as ptr, i);
        if __psa_key_eq(kk, k) { return i; };
        i = i + 2;
    }
    return -1;
}

fn __psa_dict_set(v: PsaVal, k: PsaVal, x: PsaVal) {
    if v.tag != 5 { panic(77052); };
    if k.tag == 4 || k.tag == 5 || k.tag == 11 { panic(77066); };
    let j = __psa_dict_find(v, k);
    if j >= 0 { __psa_vec_set_h(v.ipay as ptr, j + 1, x); return; };
    __psa_vec_push_h(v.ipay as ptr, k);
    __psa_vec_push_h(v.ipay as ptr, x);
}

fn __psa_dict_get(v: PsaVal, k: PsaVal) -> PsaVal {
    if v.tag != 5 { panic(77052); };
    let j = __psa_dict_find(v, k);
    if j < 0 { panic(77054); };
    return __psa_vec_get_h(v.ipay as ptr, j + 1);
}

fn __psa_dict_contains(v: PsaVal, k: PsaVal) -> bool {
    if v.tag != 5 { panic(77052); };
    return __psa_dict_find(v, k) >= 0;
}

fn __psa_dict_del(v: PsaVal, k: PsaVal) {
    if v.tag != 5 { panic(77052); };
    let j = __psa_dict_find(v, k);
    if j < 0 { panic(77054); };
    let tomb = PsaVal { tag: 4, ipay: 0, fpay: 0.0 };
    __psa_vec_set_h(v.ipay as ptr, j, tomb);
}

fn __psa_dict_len(v: PsaVal) -> i64 {
    if v.tag != 5 { panic(77052); };
    let n = __psa_vec_len_h(v.ipay as ptr);
    let c: i64 = 0;
    let i: i64 = 0;
    while i < n {
        let kk = __psa_vec_get_h(v.ipay as ptr, i);
        if kk.tag != 4 { c = c + 1; };
        i = i + 2;
    }
    return c;
}

// Entries-vec offset of the i-th live entry (insertion order).
fn __psa_dict_pos(v: PsaVal, i: i64) -> i64 {
    let n = __psa_vec_len_h(v.ipay as ptr);
    let c: i64 = 0;
    let j: i64 = 0;
    while j < n {
        let kk = __psa_vec_get_h(v.ipay as ptr, j);
        if kk.tag != 4 {
            if c == i { return j; };
            c = c + 1;
        };
        j = j + 2;
    }
    panic(77050);
}

fn __psa_dict_key_at(v: PsaVal, i: i64) -> PsaVal {
    return __psa_vec_get_h(v.ipay as ptr, __psa_dict_pos(v, i));
}

fn __psa_dict_val_at(v: PsaVal, i: i64) -> PsaVal {
    return __psa_vec_get_h(v.ipay as ptr, __psa_dict_pos(v, i) + 1);
}

fn __psa_dict_eq(a: PsaVal, b: PsaVal) -> bool {
    let n = __psa_dict_len(a);
    if n != __psa_dict_len(b) { return false; };
    let i: i64 = 0;
    while i < n {
        let k = __psa_dict_key_at(a, i);
        let j = __psa_dict_find(b, k);
        if j < 0 { return false; };
        let bv = __psa_vec_get_h(b.ipay as ptr, j + 1);
        if !(__psa_dyn_eq(__psa_dict_val_at(a, i), bv)) { return false; };
        i = i + 1;
    }
    return true;
}

fn __psa_key_int(k: PsaVal) -> i64 {
    if k.tag == 0 { return k.ipay; };
    if k.tag == 2 { return k.ipay; };
    panic(77050);
}

fn __psa_dyn_get(v: PsaVal, k: PsaVal) -> PsaVal {
    if v.tag == 4 { return __psa_list_get(v, __psa_key_int(k)); };
    if v.tag == 5 { return __psa_dict_get(v, k); };
    if v.tag == 6 { return __psa_str_get_cp(v, __psa_key_int(k)); };
    if v.tag == 15 { return __psa_bytes_get(v, __psa_key_int(k)); };
    if v.tag == 10 { return __psa_tuple_get(v, __psa_key_int(k)); };
    panic(77052);
}

fn __psa_dyn_set(v: PsaVal, k: PsaVal, x: PsaVal) {
    if v.tag == 4 { __psa_list_set(v, __psa_key_int(k), x); return; };
    if v.tag == 5 { __psa_dict_set(v, k, x); return; };
    if v.tag == 6 { panic(77063); };
    if v.tag == 15 { panic(77063); };
    panic(77052);
}

fn __psa_dyn_contains(v: PsaVal, x: PsaVal) -> bool {
    if v.tag == 4 { return __psa_list_contains(v, x); };
    if v.tag == 5 { return __psa_dict_contains(v, x); };
    if v.tag == 6 {
        if x.tag != 6 { panic(77062); };
        return __psa_str_contains(v, x);
    };
    if v.tag == 15 {
        if x.tag == 15 { return __psa_str_contains(v, x); };
        if x.tag == 0 || x.tag == 2 {
            // CPython also allows an int needle: `97 in b"abc"`.
            if x.ipay < 0 || x.ipay > 255 { panic(77062); };
            let n = __psa_str_bytes_len(v);
            let i: i64 = 0;
            while i < n {
                if __psa_bytes_get_h(v.ipay as ptr, i) as i64 == x.ipay { return true; };
                i = i + 1;
            }
            return false;
        };
        panic(77062);
    };
    if v.tag == 10 { return __psa_tuple_contains(v, x); };
    if v.tag == 11 { return __psa_set_contains(v, x); };
    panic(77052);
}

fn __psa_iter_get(v: PsaVal, i: i64) -> PsaVal {
    if v.tag == 4 { return __psa_list_get(v, i); };
    if v.tag == 6 { return __psa_str_get_cp(v, i); };
    if v.tag == 15 { return __psa_bytes_get(v, i); };
    if v.tag == 5 { return __psa_dict_key_at(v, i); };
    if v.tag == 10 { return __psa_tuple_get(v, i); };
    if v.tag == 11 { return __psa_vec_get_h(v.ipay as ptr, i); };
    panic(77052);
}

fn __psa_bytes_push_h(p: ptr, x: u8) {
    let b = unsafe { Box::from_raw(p as *Vec<u8>) };
    (*b).push(x);
    let rp: i64 = Box::into_raw(b) as i64;
}

fn __psa_bytes_get_h(p: ptr, i: i64) -> u8 {
    let b = unsafe { Box::from_raw(p as *Vec<u8>) };
    let v = (*b)[i];
    let rp: i64 = Box::into_raw(b) as i64;
    return v;
}

fn __psa_bytes_len_h(p: ptr) -> i64 {
    let b = unsafe { Box::from_raw(p as *Vec<u8>) };
    let n = (*b).len() as i64;
    let rp: i64 = Box::into_raw(b) as i64;
    return n;
}

fn __psa_str_new() -> PsaVal {
    let xs: Vec<u8> = Vec::new();
    return PsaVal { tag: 6, ipay: Box::into_raw(Box::new(xs)) as i64, fpay: 0.0 };
}

fn __psa_str_push(s: PsaVal, b: u8) -> PsaVal {
    __psa_bytes_push_h(s.ipay as ptr, b);
    return s;
}

fn __psa_str_bytes_len(v: PsaVal) -> i64 {
    return __psa_bytes_len_h(v.ipay as ptr);
}

// Bytes values (tag 15): the same Box<Vec<u8>> payload as a
// dynamic string, but every operation is byte-wise (CPython
// bytes semantics): indexing yields an int, len() counts
// bytes, equality/ordering compare raw bytes, and printing
// uses the b'...' repr form.

fn __psa_bytes_new() -> PsaVal {
    let xs: Vec<u8> = Vec::new();
    return PsaVal { tag: 15, ipay: Box::into_raw(Box::new(xs)) as i64, fpay: 0.0 };
}

fn __psa_bytes_convert(v: PsaVal) -> PsaVal {
    // bytes(x) for a dynamic x: an exact bytes value passes
    // through unchanged (CPython returns the same object,
    // and psa bytes are immutable, so aliasing is exact).
    // Every other dynamic kind -- a str with no encoding
    // argument, an int count, an iterable of ints -- is a
    // registered gap (bytes constructor forms) and refuses
    // loudly with the shared gap code instead of guessing
    // a conversion.
    if v.tag == 15 { return v; };
    panic(77148);
}

fn __psa_bytes_get(v: PsaVal, idx: i64) -> PsaVal {
    let n = __psa_str_bytes_len(v);
    let j = idx;
    if j < 0 { j = j + n; };
    if j < 0 || j >= n { panic(77050); };
    return __psa_box_int(__psa_bytes_get_h(v.ipay as ptr, j) as i64);
}

fn __psa_bytes_concat(a: PsaVal, b: PsaVal) -> PsaVal {
    let out = __psa_bytes_new();
    let na = __psa_str_bytes_len(a);
    let i: i64 = 0;
    while i < na { out = __psa_str_push(out, __psa_bytes_get_h(a.ipay as ptr, i)); i = i + 1; }
    let nb = __psa_str_bytes_len(b);
    let k: i64 = 0;
    while k < nb { out = __psa_str_push(out, __psa_bytes_get_h(b.ipay as ptr, k)); k = k + 1; }
    return out;
}

fn __psa_bytes_repeat(a: PsaVal, n: i64) -> PsaVal {
    let out = __psa_bytes_new();
    let r: i64 = 0;
    while r < n {
        let na = __psa_str_bytes_len(a);
        let i: i64 = 0;
        while i < na { out = __psa_str_push(out, __psa_bytes_get_h(a.ipay as ptr, i)); i = i + 1; }
        r = r + 1;
    }
    return out;
}

fn __psa_bytes_slice(v: PsaVal, lo: PsaVal, hi: PsaVal, step: PsaVal) -> PsaVal {
    let n = __psa_str_bytes_len(v);
    let (start, stop, stp) = __psa_slice_bounds(n, __psa_slice_idx(lo), lo.tag != 3, __psa_slice_idx(hi), hi.tag != 3, __psa_slice_idx(step), step.tag != 3);
    let out = __psa_bytes_new();
    let i = start;
    if stp > 0 {
        while i < stop { out = __psa_str_push(out, __psa_bytes_get_h(v.ipay as ptr, i)); i = i + stp; }
    } else {
        while i > stop { out = __psa_str_push(out, __psa_bytes_get_h(v.ipay as ptr, i)); i = i + stp; }
    };
    return out;
}

fn __psa_bytes_repr(v: PsaVal) -> PsaVal {
    // CPython bytes repr: a b prefix, then the str repr quote
    // choice (single quotes unless the payload has a single
    // quote and no double quote); printable ASCII passes
    // through, backslash / the quote char / \t \n \r are
    // escaped, every other byte becomes \xNN.
    let n = __psa_str_bytes_len(v);
    let has_sq = false;
    let has_dq = false;
    let i: i64 = 0;
    while i < n {
        let b = __psa_bytes_get_h(v.ipay as ptr, i);
        if b == 39 { has_sq = true; };
        if b == 34 { has_dq = true; };
        i = i + 1;
    }
    let q: u8 = 39;
    if has_sq && !has_dq { q = 34; };
    let out = __psa_str_new();
    out = __psa_str_push(out, 98);
    out = __psa_str_push(out, q);
    let k: i64 = 0;
    while k < n {
        let b = __psa_bytes_get_h(v.ipay as ptr, k);
        if b == 92 { out = __psa_str_push(out, 92); out = __psa_str_push(out, 92); }
        else if b == q { out = __psa_str_push(out, 92); out = __psa_str_push(out, q); }
        else if b == 10 { out = __psa_str_push(out, 92); out = __psa_str_push(out, 110); }
        else if b == 9 { out = __psa_str_push(out, 92); out = __psa_str_push(out, 116); }
        else if b == 13 { out = __psa_str_push(out, 92); out = __psa_str_push(out, 114); }
        else if b < 32 || b >= 127 {
            out = __psa_str_push(out, 92);
            out = __psa_str_push(out, 120);
            out = __psa_str_push(out, __psa_hex_digit((b >> 4) as i64));
            out = __psa_str_push(out, __psa_hex_digit((b & 15) as i64));
        }
        else { out = __psa_str_push(out, b); };
        k = k + 1;
    }
    out = __psa_str_push(out, q);
    return out;
}

fn __psa_str_len(v: PsaVal) -> i64 {
    // Python len() counts code points; in UTF-8 those are exactly
    // the bytes that are not continuation bytes (10xxxxxx).
    let n = __psa_str_bytes_len(v);
    let c: i64 = 0;
    let i: i64 = 0;
    while i < n {
        let b = __psa_bytes_get_h(v.ipay as ptr, i);
        if ((b as i64) & 192) != 128 { c = c + 1; };
        i = i + 1;
    }
    return c;
}

fn __psa_str_cp_start(v: PsaVal, cp: i64) -> i64 {
    // Byte offset of code point cp, or -1 when out of range.
    let n = __psa_str_bytes_len(v);
    let c: i64 = 0;
    let i: i64 = 0;
    while i < n {
        let b = __psa_bytes_get_h(v.ipay as ptr, i);
        if ((b as i64) & 192) != 128 {
            if c == cp { return i; };
            c = c + 1;
        };
        i = i + 1;
    }
    return -1;
}

fn __psa_str_get_cp(v: PsaVal, idx: i64) -> PsaVal {
    let n = __psa_str_len(v);
    let j = idx;
    if j < 0 { j = j + n; };
    if j < 0 || j >= n { panic(77050); };
    let start = __psa_str_cp_start(v, j);
    let blen = __psa_str_bytes_len(v);
    let out = __psa_str_new();
    let i = start;
    while i < blen {
        let b = __psa_bytes_get_h(v.ipay as ptr, i);
        if i > start && ((b as i64) & 192) != 128 { return out; };
        out = __psa_str_push(out, b);
        i = i + 1;
    }
    return out;
}

fn __psa_str_concat(a: PsaVal, b: PsaVal) -> PsaVal {
    let out = __psa_str_new();
    let na = __psa_str_bytes_len(a);
    let i: i64 = 0;
    while i < na { out = __psa_str_push(out, __psa_bytes_get_h(a.ipay as ptr, i)); i = i + 1; }
    let nb = __psa_str_bytes_len(b);
    let k: i64 = 0;
    while k < nb { out = __psa_str_push(out, __psa_bytes_get_h(b.ipay as ptr, k)); k = k + 1; }
    return out;
}

fn __psa_str_repeat(a: PsaVal, n: i64) -> PsaVal {
    let out = __psa_str_new();
    let r: i64 = 0;
    while r < n {
        let na = __psa_str_bytes_len(a);
        let i: i64 = 0;
        while i < na { out = __psa_str_push(out, __psa_bytes_get_h(a.ipay as ptr, i)); i = i + 1; }
        r = r + 1;
    }
    return out;
}

fn __psa_str_eq(a: PsaVal, b: PsaVal) -> bool {
    let na = __psa_str_bytes_len(a);
    let nb = __psa_str_bytes_len(b);
    if na != nb { return false; };
    let i: i64 = 0;
    while i < na {
        if __psa_bytes_get_h(a.ipay as ptr, i) != __psa_bytes_get_h(b.ipay as ptr, i) { return false; };
        i = i + 1;
    }
    return true;
}

fn __psa_str_cmp(a: PsaVal, b: PsaVal) -> i64 {
    // Byte-wise lexicographic order equals code-point order in UTF-8.
    let na = __psa_str_bytes_len(a);
    let nb = __psa_str_bytes_len(b);
    let n = na;
    if nb < n { n = nb; };
    let i: i64 = 0;
    while i < n {
        let x = __psa_bytes_get_h(a.ipay as ptr, i) as i64;
        let y = __psa_bytes_get_h(b.ipay as ptr, i) as i64;
        if x != y { return x - y; };
        i = i + 1;
    }
    return na - nb;
}

fn __psa_str_match_at(hay: PsaVal, needle: PsaVal, s: i64) -> bool {
    let nn = __psa_str_bytes_len(needle);
    let k: i64 = 0;
    while k < nn {
        if __psa_bytes_get_h(hay.ipay as ptr, s + k) != __psa_bytes_get_h(needle.ipay as ptr, k) { return false; };
        k = k + 1;
    }
    return true;
}

fn __psa_str_contains(hay: PsaVal, needle: PsaVal) -> bool {
    let nh = __psa_str_bytes_len(hay);
    let nn = __psa_str_bytes_len(needle);
    if nn == 0 { return true; };
    if nn > nh { return false; };
    let s: i64 = 0;
    while s + nn <= nh {
        if __psa_str_match_at(hay, needle, s) { return true; };
        s = s + 1;
    }
    return false;
}

fn __psa_print_str(v: PsaVal) {
    let n = __psa_str_bytes_len(v);
    let buf = [0u8; 256];
    let base: i64 = 0;
    while base < n {
        let chunk = n - base;
        if chunk > 256 { chunk = 256; };
        let k: i64 = 0;
        while k < chunk { buf[k] = __psa_bytes_get_h(v.ipay as ptr, base + k); k = k + 1; }
        sla_print_bytes(PTR_BYTE_ADD(&buf, 0), chunk as u64);
        base = base + chunk;
    }
}

fn __psa_hex_digit(d: i64) -> u8 {
    if d < 10 { return (48 + d) as u8; };
    return (87 + d) as u8;
}

fn __psa_str_repr(v: PsaVal) -> PsaVal {
    // Python repr: single quotes unless the string contains a
    // single quote and no double quote. Backslash, the quote char,
    // \n \t \r and other control bytes are escaped; bytes >= 0x80
    // pass through as UTF-8 (Python additionally escapes
    // non-printable code points -- a documented corner).
    let n = __psa_str_bytes_len(v);
    let has_sq = false;
    let has_dq = false;
    let i: i64 = 0;
    while i < n {
        let b = __psa_bytes_get_h(v.ipay as ptr, i);
        if b == 39 { has_sq = true; };
        if b == 34 { has_dq = true; };
        i = i + 1;
    }
    let q: u8 = 39;
    if has_sq && !has_dq { q = 34; };
    let out = __psa_str_new();
    out = __psa_str_push(out, q);
    let k: i64 = 0;
    while k < n {
        let b = __psa_bytes_get_h(v.ipay as ptr, k);
        if b == 92 { out = __psa_str_push(out, 92); out = __psa_str_push(out, 92); }
        else if b == q { out = __psa_str_push(out, 92); out = __psa_str_push(out, q); }
        else if b == 10 { out = __psa_str_push(out, 92); out = __psa_str_push(out, 110); }
        else if b == 9 { out = __psa_str_push(out, 92); out = __psa_str_push(out, 116); }
        else if b == 13 { out = __psa_str_push(out, 92); out = __psa_str_push(out, 114); }
        else if b < 32 || b == 127 {
            out = __psa_str_push(out, 92);
            out = __psa_str_push(out, 120);
            out = __psa_str_push(out, __psa_hex_digit((b >> 4) as i64));
            out = __psa_str_push(out, __psa_hex_digit((b & 15) as i64));
        }
        else { out = __psa_str_push(out, b); };
        k = k + 1;
    }
    out = __psa_str_push(out, q);
    return out;
}

fn __psa_slice_idx(x: PsaVal) -> i64 {
    if x.tag == 3 { return 0; };
    if x.tag == 0 || x.tag == 2 { return x.ipay; };
    panic(77065);
}

fn __psa_slice_bounds(n: i64, lo: i64, has_lo: bool, hi: i64, has_hi: bool, step: i64, has_step: bool) -> (i64, i64, i64) {
    // CPython slice normalisation: wrap negatives, then clamp to
    // [0, n] for a positive step and to [-1, n-1] for a negative one.
    let stp = 1;
    if has_step { stp = step; };
    if stp == 0 { panic(77064); };
    let start = 0;
    let stop = 0;
    if stp > 0 {
        start = 0;
        if has_lo {
            start = lo;
            if start < 0 { start = start + n; };
            if start < 0 { start = 0; };
            if start > n { start = n; };
        };
        stop = n;
        if has_hi {
            stop = hi;
            if stop < 0 { stop = stop + n; };
            if stop < 0 { stop = 0; };
            if stop > n { stop = n; };
        };
    } else {
        start = n - 1;
        if has_lo {
            start = lo;
            if start < 0 { start = start + n; };
            if start < -1 { start = -1; };
            if start > n - 1 { start = n - 1; };
        };
        stop = -1;
        if has_hi {
            stop = hi;
            if stop < 0 { stop = stop + n; };
            if stop < -1 { stop = -1; };
            if stop > n - 1 { stop = n - 1; };
        };
    };
    return (start, stop, stp);
}

fn __psa_list_slice(v: PsaVal, lo: PsaVal, hi: PsaVal, step: PsaVal) -> PsaVal {
    let n = __psa_list_len(v);
    let (start, stop, stp) = __psa_slice_bounds(n, __psa_slice_idx(lo), lo.tag != 3, __psa_slice_idx(hi), hi.tag != 3, __psa_slice_idx(step), step.tag != 3);
    let out = __psa_list_new();
    let i = start;
    if stp > 0 {
        while i < stop { __psa_list_push(out, __psa_list_get(v, i)); i = i + stp; }
    } else {
        while i > stop { __psa_list_push(out, __psa_list_get(v, i)); i = i + stp; }
    };
    return out;
}

fn __psa_str_append_str(dst: PsaVal, src: PsaVal) -> PsaVal {
    let n = __psa_str_bytes_len(src);
    let i: i64 = 0;
    while i < n { dst = __psa_str_push(dst, __psa_bytes_get_h(src.ipay as ptr, i)); i = i + 1; }
    return dst;
}

fn __psa_str_slice(v: PsaVal, lo: PsaVal, hi: PsaVal, step: PsaVal) -> PsaVal {
    let n = __psa_str_len(v);
    let (start, stop, stp) = __psa_slice_bounds(n, __psa_slice_idx(lo), lo.tag != 3, __psa_slice_idx(hi), hi.tag != 3, __psa_slice_idx(step), step.tag != 3);
    let out = __psa_str_new();
    let i = start;
    if stp > 0 {
        while i < stop { out = __psa_str_append_str(out, __psa_str_get_cp(v, i)); i = i + stp; }
    } else {
        while i > stop { out = __psa_str_append_str(out, __psa_str_get_cp(v, i)); i = i + stp; }
    };
    return out;
}

fn __psa_dyn_slice(v: PsaVal, lo: PsaVal, hi: PsaVal, step: PsaVal) -> PsaVal {
    if v.tag == 4 { return __psa_list_slice(v, lo, hi, step); };
    if v.tag == 6 { return __psa_str_slice(v, lo, hi, step); };
    if v.tag == 15 { return __psa_bytes_slice(v, lo, hi, step); };
    if v.tag == 10 { return __psa_tuple_slice(v, lo, hi, step); };
    panic(77052);
}

// ------------------------------------------------------------------
// D5: tuples (tag 10), sets (tag 11), unpacking, string helpers,
// builtin method set, conversions, type names.
//
// Vec surgery helpers: the PsaVal payloads are heap Vec boxes;
// remove/truncate go through the same from_raw/into_raw pattern
// as the byte helpers above.

fn __psa_vec_remove_h(p: ptr, i: i64) -> PsaVal {
    // SLA has no Vec::remove method on a deref'd box (the SA
    // surface exposes it as a macro, not a method), and a
    // whole-value deref assignment (*b = xs) dies in codegen
    // (error.CodegenError, probe P1). Proven shape (probe
    // P3): shift left through index writes, then pop().
    let b = unsafe { Box::from_raw(p as *Vec<PsaVal>) };
    let v = (*b)[i];
    let n = (*b).len() as i64;
    let j = i;
    while j < n - 1 {
        (*b)[j] = (*b)[j + 1];
        j = j + 1;
    }
    (*b).pop();
    let rp: i64 = Box::into_raw(b) as i64;
    return v;
}

fn __psa_vec_truncate_h(p: ptr, n: i64) {
    let b = unsafe { Box::from_raw(p as *Vec<PsaVal>) };
    while ((*b).len() as i64) > n {
        (*b).pop();
    }
    let rp: i64 = Box::into_raw(b) as i64;
}

// Tuple storage (tag 10): a Box<Vec<PsaVal>>, immutable -- no
// mutating entry points exist beyond construction-time pushes.

fn __psa_tuple_new() -> PsaVal {
    let xs: Vec<PsaVal> = Vec::new();
    return PsaVal { tag: 10, ipay: Box::into_raw(Box::new(xs)) as i64, fpay: 0.0 };
}

fn __psa_tuple_push(t: PsaVal, x: PsaVal) {
    __psa_vec_push_h(t.ipay as ptr, x);
}

fn __psa_tuple_get(v: PsaVal, i: i64) -> PsaVal {
    let n = __psa_vec_len_h(v.ipay as ptr);
    let j = i;
    if j < 0 { j = j + n; };
    if j < 0 || j >= n { panic(77050); };
    return __psa_vec_get_h(v.ipay as ptr, j);
}

fn __psa_tuple_eq(a: PsaVal, b: PsaVal) -> bool {
    let n = __psa_vec_len_h(a.ipay as ptr);
    if n != __psa_vec_len_h(b.ipay as ptr) { return false; };
    let i: i64 = 0;
    while i < n {
        if !(__psa_dyn_eq(__psa_vec_get_h(a.ipay as ptr, i), __psa_vec_get_h(b.ipay as ptr, i))) { return false; };
        i = i + 1;
    }
    return true;
}

fn __psa_tuple_key_eq(a: PsaVal, b: PsaVal) -> bool {
    let n = __psa_vec_len_h(a.ipay as ptr);
    if n != __psa_vec_len_h(b.ipay as ptr) { return false; };
    let i: i64 = 0;
    while i < n {
        if !(__psa_key_eq(__psa_vec_get_h(a.ipay as ptr, i), __psa_vec_get_h(b.ipay as ptr, i))) { return false; };
        i = i + 1;
    }
    return true;
}

fn __psa_tuple_contains(v: PsaVal, x: PsaVal) -> bool {
    let n = __psa_vec_len_h(v.ipay as ptr);
    let i: i64 = 0;
    while i < n {
        if __psa_dyn_eq(__psa_vec_get_h(v.ipay as ptr, i), x) { return true; };
        i = i + 1;
    }
    return false;
}

fn __psa_tuple_slice(v: PsaVal, lo: PsaVal, hi: PsaVal, step: PsaVal) -> PsaVal {
    // __psa_slice_bounds returns (start, STOP, step); an
    // earlier shape of this fn misread the stop as a count
    // and walked off the end of the vec (tuple1's t[1:]
    // panic 32, debug batch).
    let (start, stop, stp) = __psa_slice_bounds(__psa_vec_len_h(v.ipay as ptr), __psa_slice_idx(lo), lo.tag != 3, __psa_slice_idx(hi), hi.tag != 3, __psa_slice_idx(step), step.tag != 3);
    let out = __psa_tuple_new();
    let i = start;
    if stp > 0 {
        while i < stop { __psa_tuple_push(out, __psa_vec_get_h(v.ipay as ptr, i)); i = i + stp; }
    } else {
        while i > stop { __psa_tuple_push(out, __psa_vec_get_h(v.ipay as ptr, i)); i = i + stp; }
    };
    return out;
}

fn __psa_print_tuple_repr(v: PsaVal) {
    let n = __psa_vec_len_h(v.ipay as ptr);
    sla_print_bytes(STR_PTR("("), STR_LEN("("));
    let i: i64 = 0;
    while i < n {
        if i > 0 { sla_print_bytes(STR_PTR(", "), STR_LEN(", ")); };
        __psa_print_repr(__psa_vec_get_h(v.ipay as ptr, i));
        i = i + 1;
    }
    if n == 1 { sla_print_bytes(STR_PTR(","), STR_LEN(",")); };
    sla_print_bytes(STR_PTR(")"), STR_LEN(")"));
}

// Set storage (tag 11): a Box<Vec<PsaVal>> in insertion order,
// membership under key equality (the dict-key rule).

fn __psa_set_new() -> PsaVal {
    let xs: Vec<PsaVal> = Vec::new();
    return PsaVal { tag: 11, ipay: Box::into_raw(Box::new(xs)) as i64, fpay: 0.0 };
}

fn __psa_set_add(v: PsaVal, x: PsaVal) {
    if x.tag == 4 || x.tag == 5 || x.tag == 11 { panic(77066); };
    if __psa_set_contains(v, x) { return; };
    __psa_vec_push_h(v.ipay as ptr, x);
}

fn __psa_set_contains(v: PsaVal, x: PsaVal) -> bool {
    if x.tag == 4 || x.tag == 5 || x.tag == 11 { panic(77066); };
    let n = __psa_vec_len_h(v.ipay as ptr);
    let i: i64 = 0;
    while i < n {
        if __psa_key_eq(__psa_vec_get_h(v.ipay as ptr, i), x) { return true; };
        i = i + 1;
    }
    return false;
}

fn __psa_set_eq(a: PsaVal, b: PsaVal) -> bool {
    let n = __psa_vec_len_h(a.ipay as ptr);
    if n != __psa_vec_len_h(b.ipay as ptr) { return false; };
    let i: i64 = 0;
    while i < n {
        if !(__psa_set_contains(b, __psa_vec_get_h(a.ipay as ptr, i))) { return false; };
        i = i + 1;
    }
    return true;
}

fn __psa_print_set_repr(v: PsaVal) {
    let n = __psa_vec_len_h(v.ipay as ptr);
    if n == 0 { sla_print_bytes(STR_PTR("set()"), STR_LEN("set()")); return; };
    sla_print_bytes(STR_PTR("{"), STR_LEN("{"));
    let i: i64 = 0;
    while i < n {
        if i > 0 { sla_print_bytes(STR_PTR(", "), STR_LEN(", ")); };
        __psa_print_repr(__psa_vec_get_h(v.ipay as ptr, i));
        i = i + 1;
    }
    sla_print_bytes(STR_PTR("}"), STR_LEN("}"));
}

// Unpacking (D5): the value must be an iterable whose live
// length equals the target count (CPython raises ValueError
// on a mismatch; psa panics 77077).

fn __psa_unpack_check(v: PsaVal, n: i64) {
    if v.tag == 4 || v.tag == 5 || v.tag == 6 || v.tag == 10 || v.tag == 11 || v.tag == 15 {
        if __psa_dyn_len(v) != n { panic(77077); };
        return;
    };
    panic(77077);
}

// String conversion helpers (D5).

fn __psa_str_from_i64(x: i64) -> PsaVal {
    let digs: Vec<u8> = Vec::new();
    let dp = Box::into_raw(Box::new(digs)) as i64;
    let neg = x < 0;
    let m = x;
    if neg { m = 0 - x; };
    if m == 0 { __psa_bytes_push_h(dp as ptr, 48u8); };
    while m > 0 {
        __psa_bytes_push_h(dp as ptr, ((m % 10) + 48) as u8);
        m = m / 10;
    }
    let out = __psa_str_new();
    if neg { __psa_str_push(out, 45u8); };
    let i = __psa_bytes_len_h(dp as ptr);
    while i > 0 {
        i = i - 1;
        out = __psa_str_push(out, __psa_bytes_get_h(dp as ptr, i));
    }
    return out;
}

fn __psa_str_from_bool(b: bool) -> PsaVal {
    let out = __psa_str_new();
    if b {
        out = __psa_str_push(out, 84u8);
        out = __psa_str_push(out, 114u8);
        out = __psa_str_push(out, 117u8);
        out = __psa_str_push(out, 101u8);
    } else {
        out = __psa_str_push(out, 70u8);
        out = __psa_str_push(out, 97u8);
        out = __psa_str_push(out, 108u8);
        out = __psa_str_push(out, 115u8);
        out = __psa_str_push(out, 101u8);
    };
    return out;
}

fn __psa_str_byte(v: PsaVal, i: i64) -> i64 {
    return __psa_bytes_get_h(v.ipay as ptr, i) as i64;
}

fn __psa_str_is_space(b: i64) -> bool {
    return b == 32 || b == 9 || b == 10 || b == 13 || b == 11 || b == 12;
}

fn __psa_str_to_int(v: PsaVal) -> i64 {
    // Bounds-guarded byte reads never ride on &&: SLA's &&
    // evaluates both sides eagerly (probe P4, defect
    // (xviii)), so `i < n && byte(v, i)` reads at i == n.
    // Every guarded read below sits inside a plain bounds
    // branch instead.
    let n = __psa_bytes_len_h(v.ipay as ptr);
    let i: i64 = 0;
    while i < n {
        if !(__psa_str_is_space(__psa_str_byte(v, i))) { break; };
        i = i + 1;
    }
    let neg = false;
    if i < n {
        if __psa_str_byte(v, i) == 43 || __psa_str_byte(v, i) == 45 {
            neg = __psa_str_byte(v, i) == 45;
            i = i + 1;
        };
    };
    let any = false;
    let acc: i64 = 0;
    while i < n {
        let b = __psa_str_byte(v, i);
        if b < 48 || b > 57 { break; };
        acc = acc * 10 + (b - 48);
        any = true;
        i = i + 1;
    }
    while i < n {
        if !(__psa_str_is_space(__psa_str_byte(v, i))) { break; };
        i = i + 1;
    }
    if !(any) || i != n { panic(77079); };
    if neg { return 0 - acc; };
    return acc;
}

fn __psa_str_to_float(v: PsaVal) -> f64 {
    // Same eager-&& discipline as __psa_str_to_int: guarded
    // byte reads sit inside plain bounds branches.
    let n = __psa_bytes_len_h(v.ipay as ptr);
    let i: i64 = 0;
    while i < n {
        if !(__psa_str_is_space(__psa_str_byte(v, i))) { break; };
        i = i + 1;
    }
    let neg = false;
    if i < n {
        if __psa_str_byte(v, i) == 43 || __psa_str_byte(v, i) == 45 {
            neg = __psa_str_byte(v, i) == 45;
            i = i + 1;
        };
    };
    let any = false;
    let acc: f64 = 0.0;
    while i < n {
        let b = __psa_str_byte(v, i);
        if b < 48 || b > 57 { break; };
        acc = acc * 10.0 + ((b - 48) as f64);
        any = true;
        i = i + 1;
    }
    if i < n {
        if __psa_str_byte(v, i) == 46 {
        i = i + 1;
        let scale: f64 = 0.1;
        while i < n {
            let b = __psa_str_byte(v, i);
            if b < 48 || b > 57 { break; };
            acc = acc + ((b - 48) as f64) * scale;
            scale = scale * 0.1;
            any = true;
            i = i + 1;
        }
        };
    };
    if i < n {
        if __psa_str_byte(v, i) == 101 || __psa_str_byte(v, i) == 69 {
        i = i + 1;
        let eneg = false;
        if i < n {
            if __psa_str_byte(v, i) == 43 || __psa_str_byte(v, i) == 45 {
                eneg = __psa_str_byte(v, i) == 45;
                i = i + 1;
            };
        };
        let eany = false;
        let e: i64 = 0;
        while i < n {
            let b = __psa_str_byte(v, i);
            if b < 48 || b > 57 { break; };
            e = e * 10 + (b - 48);
            eany = true;
            i = i + 1;
        }
        if !(eany) { panic(77079); };
        let k: i64 = 0;
        if eneg {
            while k < e { acc = acc * 0.1; k = k + 1; }
        } else {
            while k < e { acc = acc * 10.0; k = k + 1; }
        };
    };
    };
    while i < n {
        if !(__psa_str_is_space(__psa_str_byte(v, i))) { break; };
        i = i + 1;
    }
    if !(any) || i != n { panic(77079); };
    if neg { return 0.0 - acc; };
    return acc;
}

// Type names (D5): type(x) yields the name string; class
// instances defer to the generated __psa_class_name.

fn __psa_type_base_name(v: PsaVal) -> PsaVal {
    // Sequential ifs with early returns, not an else-if
    // chain: the SLA checker types a long else-if chain as
    // an expression and rejects it with "if expression
    // branch type mismatch" (debug batch bisect, run 3).
    if v.tag == 8 { return __psa_class_name(v); };
    let out = __psa_str_new();
    if v.tag == 0 {
        out = __psa_str_push(out, 105u8); __psa_str_push(out, 110u8); __psa_str_push(out, 116u8);
        return out;
    };
    if v.tag == 1 {
        out = __psa_str_push(out, 102u8); __psa_str_push(out, 108u8); __psa_str_push(out, 111u8); __psa_str_push(out, 97u8); __psa_str_push(out, 116u8);
        return out;
    };
    if v.tag == 2 {
        out = __psa_str_push(out, 98u8); __psa_str_push(out, 111u8); __psa_str_push(out, 111u8); __psa_str_push(out, 108u8);
        return out;
    };
    if v.tag == 3 {
        out = __psa_str_push(out, 78u8); __psa_str_push(out, 111u8); __psa_str_push(out, 110u8); __psa_str_push(out, 101u8); __psa_str_push(out, 84u8); __psa_str_push(out, 121u8); __psa_str_push(out, 112u8); __psa_str_push(out, 101u8);
        return out;
    };
    if v.tag == 4 {
        out = __psa_str_push(out, 108u8); __psa_str_push(out, 105u8); __psa_str_push(out, 115u8); __psa_str_push(out, 116u8);
        return out;
    };
    if v.tag == 5 {
        out = __psa_str_push(out, 100u8); __psa_str_push(out, 105u8); __psa_str_push(out, 99u8); __psa_str_push(out, 116u8);
        return out;
    };
    if v.tag == 6 {
        out = __psa_str_push(out, 115u8); __psa_str_push(out, 116u8); __psa_str_push(out, 114u8);
        return out;
    };
    if v.tag == 15 {
        out = __psa_str_push(out, 98u8); __psa_str_push(out, 121u8); __psa_str_push(out, 116u8); __psa_str_push(out, 101u8); __psa_str_push(out, 115u8);
        return out;
    };
    if v.tag == 7 {
        out = __psa_str_push(out, 102u8); __psa_str_push(out, 117u8); __psa_str_push(out, 110u8); __psa_str_push(out, 99u8); __psa_str_push(out, 116u8); __psa_str_push(out, 105u8); __psa_str_push(out, 111u8); __psa_str_push(out, 110u8);
        return out;
    };
    if v.tag == 9 {
        out = __psa_str_push(out, 109u8); __psa_str_push(out, 101u8); __psa_str_push(out, 116u8); __psa_str_push(out, 104u8); __psa_str_push(out, 111u8); __psa_str_push(out, 100u8);
        return out;
    };
    if v.tag == 10 {
        out = __psa_str_push(out, 116u8); __psa_str_push(out, 117u8); __psa_str_push(out, 112u8); __psa_str_push(out, 108u8); __psa_str_push(out, 101u8);
        return out;
    };
    if v.tag == 11 {
        out = __psa_str_push(out, 115u8); __psa_str_push(out, 101u8); __psa_str_push(out, 116u8);
        return out;
    };
    if v.tag == 12 {
        out = __psa_str_push(out, 84u8); __psa_str_push(out, 101u8); __psa_str_push(out, 120u8); __psa_str_push(out, 116u8); __psa_str_push(out, 73u8); __psa_str_push(out, 79u8); __psa_str_push(out, 87u8); __psa_str_push(out, 114u8); __psa_str_push(out, 97u8); __psa_str_push(out, 112u8); __psa_str_push(out, 112u8); __psa_str_push(out, 101u8); __psa_str_push(out, 114u8);
        return out;
    };
    if v.tag == 13 {
        out = __psa_str_push(out, 103u8); __psa_str_push(out, 101u8); __psa_str_push(out, 110u8); __psa_str_push(out, 101u8); __psa_str_push(out, 114u8); __psa_str_push(out, 97u8); __psa_str_push(out, 116u8); __psa_str_push(out, 111u8); __psa_str_push(out, 114u8);
        return out;
    };
    if v.tag == 14 {
        out = __psa_str_push(out, 99u8); __psa_str_push(out, 111u8); __psa_str_push(out, 114u8); __psa_str_push(out, 111u8); __psa_str_push(out, 117u8); __psa_str_push(out, 116u8); __psa_str_push(out, 105u8); __psa_str_push(out, 110u8); __psa_str_push(out, 101u8);
        return out;
    };
    return out;
}

// type(x) (D5): CPython renders a class object as
// `<class 'NAME'>`; psa has no class values, so type()
// returns that display string directly (a registered
// divergence: type() results cannot be compared against
// class names or used to construct).
fn __psa_type_name(v: PsaVal) -> PsaVal {
    let nm = __psa_type_base_name(v);
    let out = __psa_str_new();
    out = __psa_str_push(out, 60u8);
    out = __psa_str_push(out, 99u8);
    out = __psa_str_push(out, 108u8);
    out = __psa_str_push(out, 97u8);
    out = __psa_str_push(out, 115u8);
    out = __psa_str_push(out, 115u8);
    out = __psa_str_push(out, 32u8);
    out = __psa_str_push(out, 39u8);
    out = __psa_str_append_str(out, nm);
    out = __psa_str_push(out, 39u8);
    out = __psa_str_push(out, 62u8);
    return out;
}

// str(x) for a dynamic value (D5): scalars format exactly,
// objects ride their __str__ machinery; floats stay blocked
// by (xxiii) and containers by the missing repr-to-string
// builders -- both panic 77030 rather than print a wrong text.
fn __psa_str_of(v: PsaVal) -> PsaVal {
    if v.tag == 6 { return v; };
    if v.tag == 15 { return __psa_bytes_repr(v); };
    if v.tag == 0 { return __psa_str_from_i64(v.ipay); };
    if v.tag == 2 { return __psa_str_from_bool(v.ipay != 0); };
    if v.tag == 3 {
        let out = __psa_str_new();
        out = __psa_str_push(out, 78u8);
        out = __psa_str_push(out, 111u8);
        out = __psa_str_push(out, 110u8);
        out = __psa_str_push(out, 101u8);
        return out;
    };
    if v.tag == 8 { return __psa_obj_str(v); };
    if v.tag == 12 { return __psa_file_str(v); };
    if v.tag == 13 {
        let out = __psa_str_new();
        out = __psa_str_push(out, 60u8); out = __psa_str_push(out, 103u8); out = __psa_str_push(out, 101u8); out = __psa_str_push(out, 110u8); out = __psa_str_push(out, 101u8); out = __psa_str_push(out, 114u8); out = __psa_str_push(out, 97u8); out = __psa_str_push(out, 116u8); out = __psa_str_push(out, 111u8); out = __psa_str_push(out, 114u8); out = __psa_str_push(out, 32u8); out = __psa_str_push(out, 111u8); out = __psa_str_push(out, 98u8); out = __psa_str_push(out, 106u8); out = __psa_str_push(out, 101u8); out = __psa_str_push(out, 99u8); out = __psa_str_push(out, 116u8); out = __psa_str_push(out, 62u8);
        return out;
    };
    if v.tag == 14 {
        let out = __psa_str_new();
        out = __psa_str_push(out, 60u8); out = __psa_str_push(out, 99u8); out = __psa_str_push(out, 111u8); out = __psa_str_push(out, 114u8); out = __psa_str_push(out, 111u8); out = __psa_str_push(out, 117u8); out = __psa_str_push(out, 116u8); out = __psa_str_push(out, 105u8); out = __psa_str_push(out, 110u8); out = __psa_str_push(out, 101u8); out = __psa_str_push(out, 32u8); out = __psa_str_push(out, 111u8); out = __psa_str_push(out, 98u8); out = __psa_str_push(out, 106u8); out = __psa_str_push(out, 101u8); out = __psa_str_push(out, 99u8); out = __psa_str_push(out, 116u8); out = __psa_str_push(out, 62u8);
        return out;
    };
    if v.tag == 17 { return __psa_big_to_psastr(v); };
    panic(77030);
}

// int(x) / float(x) on a dynamic value (D5).
fn __psa_to_int(v: PsaVal) -> i64 {
    if v.tag == 0 { return v.ipay; };
    if v.tag == 2 { return v.ipay; };
    if v.tag == 1 { return v.fpay as i64; };
    if v.tag == 6 { return __psa_str_to_int(v); };
    if v.tag == 17 { panic(77080); };
    panic(77075);
}

fn __psa_to_float(v: PsaVal) -> f64 {
    if v.tag == 1 { return v.fpay; };
    if v.tag == 0 { return v.ipay as f64; };
    if v.tag == 2 { return v.ipay as f64; };
    if v.tag == 6 { return __psa_str_to_float(v); };
    if v.tag == 17 { return __psa_big_as_float(v); };
    panic(77075);
}

// list(x) / tuple(x) / set(x) from any iterable dynamic value
// (D5): built through the iteration protocol; dicts yield
// their keys, as in CPython.
fn __psa_list_from(v: PsaVal) -> PsaVal {
    let out = __psa_list_new();
    let n = __psa_dyn_len(v);
    let i: i64 = 0;
    while i < n {
        __psa_list_push(out, __psa_iter_get(v, i));
        i = i + 1;
    }
    return out;
}

fn __psa_tuple_from(v: PsaVal) -> PsaVal {
    if v.tag == 10 { return v; };
    let out = __psa_tuple_new();
    let n = __psa_dyn_len(v);
    let i: i64 = 0;
    while i < n {
        __psa_tuple_push(out, __psa_iter_get(v, i));
        i = i + 1;
    }
    return out;
}

fn __psa_set_from(v: PsaVal) -> PsaVal {
    let out = __psa_set_new();
    let n = __psa_dyn_len(v);
    let i: i64 = 0;
    while i < n {
        __psa_set_add(out, __psa_iter_get(v, i));
        i = i + 1;
    }
    return out;
}

// isinstance against a builtin type code (D5): 0 int (bool
// counts, as in CPython), 1 float, 2 bool, 3 str, 4 list,
// 5 dict, 6 tuple, 7 set, 8 object, 9 NoneType, 10 bytes.

fn __psa_is_type(v: PsaVal, code: i64) -> bool {
    if code == 0 { return v.tag == 0 || v.tag == 2; };
    if code == 1 { return v.tag == 1; };
    if code == 2 { return v.tag == 2; };
    if code == 3 { return v.tag == 6; };
    if code == 10 { return v.tag == 15; };
    if code == 4 { return v.tag == 4; };
    if code == 5 { return v.tag == 5; };
    if code == 6 { return v.tag == 10; };
    if code == 7 { return v.tag == 11; };
    if code == 8 { return true; };
    if code == 9 { return v.tag == 3; };
    return false;
}

fn __psa_isinst_class(v: PsaVal, cid: i64) -> bool {
    return v.tag == 8 && __psa_obj_class(v) == cid;
}

// ------------------------------------------------------------------
// D5 builtin methods. Shape: __psa_m_<name>(v, name, args) where
// args is a dynamic list of the call arguments. Every fn first
// defers tag 8 to getattr (an object's own method wins over a
// container builtin of the same name), then checks its kind
// (wrong kind -> 77069 AttributeError) and arity (77068).

fn __psa_m_argc(args: PsaVal, lo: i64, hi: i64) {
    let n = __psa_list_len(args);
    if n < lo || n > hi { panic(77068); };
}

fn __psa_m_arg(args: PsaVal, i: i64) -> PsaVal {
    return __psa_list_get(args, i);
}

fn __psa_m_none() -> PsaVal {
    return PsaVal { tag: 3, ipay: 0, fpay: 0.0 };
}

// append fast path (D5): one argument, no packing; objects
// still win via getattr, other kinds are refused as usual.
fn __psa_append_fast(v: PsaVal, name: PsaVal, x: PsaVal) -> PsaVal {
    if v.tag == 8 {
        let a = __psa_list_new();
        __psa_list_push(a, x);
        return __psa_call_value(__psa_getattr(v, name), a);
    };
    if v.tag != 4 { panic(77069); };
    __psa_list_push(v, x);
    return __psa_m_none();
}

fn __psa_m_append(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 4 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    __psa_list_push(v, __psa_m_arg(args, 0));
    return __psa_m_none();
}

fn __psa_m_extend(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 4 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    let src = __psa_m_arg(args, 0);
    if src.tag != 4 && src.tag != 5 && src.tag != 6 && src.tag != 10 && src.tag != 11 { panic(77075); };
    let n = __psa_dyn_len(src);
    let i: i64 = 0;
    while i < n {
        __psa_list_push(v, __psa_iter_get(src, i));
        i = i + 1;
    }
    return __psa_m_none();
}

fn __psa_m_insert(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 4 { panic(77069); };
    __psa_m_argc(args, 2, 2);
    let n = __psa_list_len(v);
    let idx = __psa_key_int(__psa_m_arg(args, 0));
    if idx < 0 { idx = idx + n; };
    if idx < 0 { idx = 0; };
    if idx > n { idx = n; };
    __psa_list_push(v, __psa_m_arg(args, 1));
    let i = n;
    while i > idx {
        __psa_list_set(v, i, __psa_list_get(v, i - 1));
        i = i - 1;
    }
    __psa_list_set(v, idx, __psa_m_arg(args, 1));
    return __psa_m_none();
}

fn __psa_m_pop(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag == 4 {
        __psa_m_argc(args, 0, 1);
        let n = __psa_list_len(v);
        if n == 0 { panic(77050); };
        let idx = n - 1;
        if __psa_list_len(args) == 1 { idx = __psa_key_int(__psa_m_arg(args, 0)); };
        if idx < 0 { idx = idx + n; };
        if idx < 0 || idx >= n { panic(77050); };
        let out = __psa_list_get(v, idx);
        __psa_vec_remove_h(v.ipay as ptr, idx);
        return out;
    };
    if v.tag == 5 {
        __psa_m_argc(args, 1, 2);
        let k = __psa_m_arg(args, 0);
        let j = __psa_dict_find(v, k);
        if j < 0 {
            if __psa_list_len(args) == 2 { return __psa_m_arg(args, 1); };
            panic(77054);
        };
        let out = __psa_vec_get_h(v.ipay as ptr, j + 1);
        let tomb = PsaVal { tag: 4, ipay: 0, fpay: 0.0 };
        __psa_vec_set_h(v.ipay as ptr, j, tomb);
        return out;
    };
    panic(77069);
}

fn __psa_m_remove(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag == 11 {
        __psa_m_argc(args, 1, 1);
        let x = __psa_m_arg(args, 0);
        let n = __psa_vec_len_h(v.ipay as ptr);
        let i: i64 = 0;
        while i < n {
            if __psa_key_eq(__psa_vec_get_h(v.ipay as ptr, i), x) {
                __psa_vec_remove_h(v.ipay as ptr, i);
                return __psa_m_none();
            };
            i = i + 1;
        }
        panic(77076);
    };
    if v.tag != 4 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    let x = __psa_m_arg(args, 0);
    let n = __psa_list_len(v);
    let i: i64 = 0;
    while i < n {
        if __psa_dyn_eq(__psa_list_get(v, i), x) {
            __psa_vec_remove_h(v.ipay as ptr, i);
            return __psa_m_none();
        };
        i = i + 1;
    }
    panic(77076);
}

fn __psa_m_clear(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    __psa_m_argc(args, 0, 0);
    if v.tag == 4 || v.tag == 11 {
        __psa_vec_truncate_h(v.ipay as ptr, 0);
        return __psa_m_none();
    };
    if v.tag == 5 {
        let n = __psa_vec_len_h(v.ipay as ptr);
        let tomb = PsaVal { tag: 4, ipay: 0, fpay: 0.0 };
        let i: i64 = 0;
        while i < n {
            __psa_vec_set_h(v.ipay as ptr, i, tomb);
            i = i + 2;
        }
        return __psa_m_none();
    };
    panic(77069);
}

fn __psa_m_copy(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    __psa_m_argc(args, 0, 0);
    if v.tag == 4 {
        let out = __psa_list_new();
        let n = __psa_list_len(v);
        let i: i64 = 0;
        while i < n { __psa_list_push(out, __psa_list_get(v, i)); i = i + 1; }
        return out;
    };
    if v.tag == 5 {
        let out = __psa_dict_new();
        let n = __psa_dict_len(v);
        let i: i64 = 0;
        while i < n {
            __psa_dict_set(out, __psa_dict_key_at(v, i), __psa_dict_val_at(v, i));
            i = i + 1;
        }
        return out;
    };
    if v.tag == 11 {
        let out = __psa_set_new();
        let n = __psa_vec_len_h(v.ipay as ptr);
        let i: i64 = 0;
        while i < n { __psa_vec_push_h(out.ipay as ptr, __psa_vec_get_h(v.ipay as ptr, i)); i = i + 1; }
        return out;
    };
    panic(77069);
}

fn __psa_m_index(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 4 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    let x = __psa_m_arg(args, 0);
    let n = __psa_list_len(v);
    let i: i64 = 0;
    while i < n {
        if __psa_dyn_eq(__psa_list_get(v, i), x) { return __psa_box_int(i); };
        i = i + 1;
    }
    panic(77076);
}

fn __psa_m_count(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    __psa_m_argc(args, 1, 1);
    let x = __psa_m_arg(args, 0);
    if v.tag == 4 {
        let n = __psa_list_len(v);
        let c: i64 = 0;
        let i: i64 = 0;
        while i < n {
            if __psa_dyn_eq(__psa_list_get(v, i), x) { c = c + 1; };
            i = i + 1;
        }
        return __psa_box_int(c);
    };
    if v.tag == 6 {
        if x.tag != 6 { panic(77075); };
        return __psa_box_int(__psa_str_count_sub(v, x));
    };
    panic(77069);
}

fn __psa_m_reverse(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 4 { panic(77069); };
    __psa_m_argc(args, 0, 0);
    let n = __psa_list_len(v);
    let i: i64 = 0;
    while i < n / 2 {
        let a = __psa_list_get(v, i);
        __psa_list_set(v, i, __psa_list_get(v, n - 1 - i));
        __psa_list_set(v, n - 1 - i, a);
        i = i + 1;
    }
    return __psa_m_none();
}

fn __psa_m_sort(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 4 { panic(77069); };
    __psa_m_argc(args, 0, 0);
    let n = __psa_list_len(v);
    let i: i64 = 0;
    while i < n {
        let j = i + 1;
        while j < n {
            if __psa_dyn_lt(__psa_list_get(v, j), __psa_list_get(v, i)) {
                let a = __psa_list_get(v, i);
                __psa_list_set(v, i, __psa_list_get(v, j));
                __psa_list_set(v, j, a);
            };
            j = j + 1;
        }
        i = i + 1;
    }
    return __psa_m_none();
}

fn __psa_m_get(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 5 { panic(77069); };
    __psa_m_argc(args, 1, 2);
    let j = __psa_dict_find(v, __psa_m_arg(args, 0));
    if j >= 0 { return __psa_vec_get_h(v.ipay as ptr, j + 1); };
    if __psa_list_len(args) == 2 { return __psa_m_arg(args, 1); };
    return __psa_m_none();
}

fn __psa_m_keys(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 5 { panic(77069); };
    __psa_m_argc(args, 0, 0);
    let out = __psa_list_new();
    let n = __psa_dict_len(v);
    let i: i64 = 0;
    while i < n { __psa_list_push(out, __psa_dict_key_at(v, i)); i = i + 1; }
    return out;
}

fn __psa_m_values(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 5 { panic(77069); };
    __psa_m_argc(args, 0, 0);
    let out = __psa_list_new();
    let n = __psa_dict_len(v);
    let i: i64 = 0;
    while i < n { __psa_list_push(out, __psa_dict_val_at(v, i)); i = i + 1; }
    return out;
}

fn __psa_m_items(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 5 { panic(77069); };
    __psa_m_argc(args, 0, 0);
    let out = __psa_list_new();
    let n = __psa_dict_len(v);
    let i: i64 = 0;
    while i < n {
        let t = __psa_tuple_new();
        __psa_tuple_push(t, __psa_dict_key_at(v, i));
        __psa_tuple_push(t, __psa_dict_val_at(v, i));
        __psa_list_push(out, t);
        i = i + 1;
    }
    return out;
}

fn __psa_m_setdefault(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 5 { panic(77069); };
    __psa_m_argc(args, 1, 2);
    let k = __psa_m_arg(args, 0);
    let j = __psa_dict_find(v, k);
    if j >= 0 { return __psa_vec_get_h(v.ipay as ptr, j + 1); };
    let d = __psa_m_none();
    if __psa_list_len(args) == 2 { d = __psa_m_arg(args, 1); };
    __psa_dict_set(v, k, d);
    return d;
}

fn __psa_m_update(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 5 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    let src = __psa_m_arg(args, 0);
    if src.tag != 5 { panic(77075); };
    let n = __psa_dict_len(src);
    let i: i64 = 0;
    while i < n {
        __psa_dict_set(v, __psa_dict_key_at(src, i), __psa_dict_val_at(src, i));
        i = i + 1;
    }
    return __psa_m_none();
}

fn __psa_m_add(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 11 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    __psa_set_add(v, __psa_m_arg(args, 0));
    return __psa_m_none();
}

fn __psa_m_discard(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 11 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    let x = __psa_m_arg(args, 0);
    let n = __psa_vec_len_h(v.ipay as ptr);
    let i: i64 = 0;
    while i < n {
        if __psa_key_eq(__psa_vec_get_h(v.ipay as ptr, i), x) {
            __psa_vec_remove_h(v.ipay as ptr, i);
            return __psa_m_none();
        };
        i = i + 1;
    }
    return __psa_m_none();
}

fn __psa_m_union(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 11 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    let src = __psa_m_arg(args, 0);
    if src.tag != 4 && src.tag != 5 && src.tag != 6 && src.tag != 10 && src.tag != 11 { panic(77075); };
    let out = __psa_set_new();
    let n = __psa_vec_len_h(v.ipay as ptr);
    let i: i64 = 0;
    while i < n { __psa_set_add(out, __psa_vec_get_h(v.ipay as ptr, i)); i = i + 1; }
    let m = __psa_dyn_len(src);
    let j: i64 = 0;
    while j < m { __psa_set_add(out, __psa_iter_get(src, j)); j = j + 1; }
    return out;
}

fn __psa_m_intersection(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 11 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    let src = __psa_m_arg(args, 0);
    if src.tag != 4 && src.tag != 5 && src.tag != 6 && src.tag != 10 && src.tag != 11 { panic(77075); };
    let out = __psa_set_new();
    let n = __psa_vec_len_h(v.ipay as ptr);
    let i: i64 = 0;
    while i < n {
        let x = __psa_vec_get_h(v.ipay as ptr, i);
        if __psa_dyn_contains(src, x) { __psa_set_add(out, x); };
        i = i + 1;
    }
    return out;
}

fn __psa_m_difference(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 11 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    let src = __psa_m_arg(args, 0);
    if src.tag != 4 && src.tag != 5 && src.tag != 6 && src.tag != 10 && src.tag != 11 { panic(77075); };
    let out = __psa_set_new();
    let n = __psa_vec_len_h(v.ipay as ptr);
    let i: i64 = 0;
    while i < n {
        let x = __psa_vec_get_h(v.ipay as ptr, i);
        if !(__psa_dyn_contains(src, x)) { __psa_set_add(out, x); };
        i = i + 1;
    }
    return out;
}

// String methods (D5): byte-level work on the Vec<u8> payload.
// Case mapping and classification are ASCII-only in v1; bytes
// >= 0x80 pass through unchanged (registered partial).

fn __psa_str_sub_bytes(v: PsaVal, lo: i64, hi: i64) -> PsaVal {
    let out = __psa_str_new();
    let i = lo;
    while i < hi {
        out = __psa_str_push(out, __psa_bytes_get_h(v.ipay as ptr, i));
        i = i + 1;
    }
    return out;
}

fn __psa_str_count_sub(hay: PsaVal, needle: PsaVal) -> i64 {
    let hn = __psa_bytes_len_h(hay.ipay as ptr);
    let nn = __psa_bytes_len_h(needle.ipay as ptr);
    if nn == 0 { return hn + 1; };
    let c: i64 = 0;
    let i: i64 = 0;
    while i + nn <= hn {
        if __psa_str_match_at(hay, needle, i) { c = c + 1; i = i + nn; } else { i = i + 1; };
    }
    return c;
}

fn __psa_str_find_sub(hay: PsaVal, needle: PsaVal) -> i64 {
    let hn = __psa_bytes_len_h(hay.ipay as ptr);
    let nn = __psa_bytes_len_h(needle.ipay as ptr);
    if nn == 0 { return 0; };
    let i: i64 = 0;
    while i + nn <= hn {
        if __psa_str_match_at(hay, needle, i) { return i; };
        i = i + 1;
    }
    return -1;
}

// Byte offset -> code point index (UTF-8 lead bytes < 0x80 or
// >= 0xC0 start a code point).
fn __psa_str_cp_of_byte(v: PsaVal, boff: i64) -> i64 {
    let c: i64 = 0;
    let i: i64 = 0;
    while i < boff {
        let b = __psa_str_byte(v, i);
        if b < 128 || b >= 192 { c = c + 1; };
        i = i + 1;
    }
    return c;
}

fn __psa_m_upper(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 6 { panic(77069); };
    __psa_m_argc(args, 0, 0);
    let out = __psa_str_new();
    let n = __psa_bytes_len_h(v.ipay as ptr);
    let i: i64 = 0;
    while i < n {
        let b = __psa_str_byte(v, i);
        if b >= 97 && b <= 122 { out = __psa_str_push(out, (b - 32) as u8); } else { out = __psa_str_push(out, b as u8); };
        i = i + 1;
    }
    return out;
}

fn __psa_m_lower(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 6 { panic(77069); };
    __psa_m_argc(args, 0, 0);
    let out = __psa_str_new();
    let n = __psa_bytes_len_h(v.ipay as ptr);
    let i: i64 = 0;
    while i < n {
        let b = __psa_str_byte(v, i);
        if b >= 65 && b <= 90 { out = __psa_str_push(out, (b + 32) as u8); } else { out = __psa_str_push(out, b as u8); };
        i = i + 1;
    }
    return out;
}

fn __psa_str_in_set(b: i64, set: PsaVal, has_set: bool) -> bool {
    if !(has_set) { return __psa_str_is_space(b); };
    let n = __psa_bytes_len_h(set.ipay as ptr);
    let i: i64 = 0;
    while i < n {
        if __psa_str_byte(set, i) == b { return true; };
        i = i + 1;
    }
    return false;
}

fn __psa_str_strip(v: PsaVal, args: PsaVal, left: bool, right: bool) -> PsaVal {
    __psa_m_argc(args, 0, 1);
    let has_set = __psa_list_len(args) == 1;
    let set = __psa_m_none();
    if has_set {
        set = __psa_m_arg(args, 0);
        if set.tag != 6 { panic(77075); };
    };
    let n = __psa_bytes_len_h(v.ipay as ptr);
    let lo: i64 = 0;
    let hi = n;
    if left {
        while lo < hi && __psa_str_in_set(__psa_str_byte(v, lo), set, has_set) { lo = lo + 1; }
    };
    if right {
        while hi > lo && __psa_str_in_set(__psa_str_byte(v, hi - 1), set, has_set) { hi = hi - 1; }
    };
    return __psa_str_sub_bytes(v, lo, hi);
}

fn __psa_m_strip(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 6 { panic(77069); };
    return __psa_str_strip(v, args, true, true);
}

fn __psa_m_lstrip(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 6 { panic(77069); };
    return __psa_str_strip(v, args, true, false);
}

fn __psa_m_rstrip(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 6 { panic(77069); };
    return __psa_str_strip(v, args, false, true);
}

fn __psa_m_split(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 6 { panic(77069); };
    __psa_m_argc(args, 0, 1);
    let out = __psa_list_new();
    let n = __psa_bytes_len_h(v.ipay as ptr);
    if __psa_list_len(args) == 0 {
        // Whitespace split; the skip/scan loops use plain
        // bounds branches, never && (eager, probe P4).
        let i: i64 = 0;
        while i < n {
            while i < n {
                if !(__psa_str_is_space(__psa_str_byte(v, i))) { break; };
                i = i + 1;
            }
            if i >= n { break; };
            let start = i;
            while i < n {
                if __psa_str_is_space(__psa_str_byte(v, i)) { break; };
                i = i + 1;
            }
            __psa_list_push(out, __psa_str_sub_bytes(v, start, i));
        }
        return out;
    };
    let sep = __psa_m_arg(args, 0);
    if sep.tag != 6 { panic(77075); };
    let sn = __psa_bytes_len_h(sep.ipay as ptr);
    if sn == 0 { panic(77079); };
    let start: i64 = 0;
    let i: i64 = 0;
    while i + sn <= n {
        if __psa_str_match_at(v, sep, i) {
            __psa_list_push(out, __psa_str_sub_bytes(v, start, i));
            i = i + sn;
            start = i;
        } else {
            i = i + 1;
        };
    }
    __psa_list_push(out, __psa_str_sub_bytes(v, start, n));
    return out;
}

fn __psa_m_join(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 6 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    let src = __psa_m_arg(args, 0);
    if src.tag != 4 && src.tag != 10 && src.tag != 11 { panic(77075); };
    let out = __psa_str_new();
    let n = __psa_dyn_len(src);
    let i: i64 = 0;
    while i < n {
        let el = __psa_iter_get(src, i);
        if el.tag != 6 { panic(77075); };
        if i > 0 { out = __psa_str_append_str(out, v); };
        out = __psa_str_append_str(out, el);
        i = i + 1;
    }
    return out;
}

fn __psa_m_replace(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 6 { panic(77069); };
    __psa_m_argc(args, 2, 3);
    let old = __psa_m_arg(args, 0);
    let new = __psa_m_arg(args, 1);
    if old.tag != 6 || new.tag != 6 { panic(77075); };
    let limit: i64 = -1;
    if __psa_list_len(args) == 3 {
        let c = __psa_m_arg(args, 2);
        if c.tag != 0 && c.tag != 2 { panic(77075); };
        limit = c.ipay;
    };
    let n = __psa_bytes_len_h(v.ipay as ptr);
    let on = __psa_bytes_len_h(old.ipay as ptr);
    let out = __psa_str_new();
    if on == 0 {
        // Insert `new` at both ends and between every byte
        // (CPython semantics for an empty old string), bounded
        // by the count argument.
        let done: i64 = 0;
        if limit != 0 { out = __psa_str_append_str(out, new); done = done + 1; };
        let i: i64 = 0;
        while i < n {
            out = __psa_str_push(out, __psa_bytes_get_h(v.ipay as ptr, i));
            if limit < 0 || done < limit {
                out = __psa_str_append_str(out, new);
                done = done + 1;
            };
            i = i + 1;
        }
        return out;
    };
    let i: i64 = 0;
    let done: i64 = 0;
    while i < n {
        // match_at reads needle-length bytes at i with no
        // internal bounds check, so it may only run once
        // i + on <= n is established -- nested branches,
        // not && (eager, probe P4).
        let hit = false;
        if i + on <= n {
            if limit < 0 || done < limit {
                hit = __psa_str_match_at(v, old, i);
            };
        };
        if hit {
            out = __psa_str_append_str(out, new);
            done = done + 1;
            i = i + on;
        } else {
            out = __psa_str_push(out, __psa_bytes_get_h(v.ipay as ptr, i));
            i = i + 1;
        };
    }
    return out;
}

fn __psa_m_startswith(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 6 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    let p = __psa_m_arg(args, 0);
    if p.tag != 6 { panic(77075); };
    return __psa_box_bool(__psa_str_match_at(v, p, 0));
}

fn __psa_m_endswith(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 6 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    let p = __psa_m_arg(args, 0);
    if p.tag != 6 { panic(77075); };
    let start = __psa_bytes_len_h(v.ipay as ptr) - __psa_bytes_len_h(p.ipay as ptr);
    if start < 0 { return __psa_box_bool(false); };
    return __psa_box_bool(__psa_str_match_at(v, p, start));
}

fn __psa_m_find(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 6 { panic(77069); };
    __psa_m_argc(args, 1, 1);
    let p = __psa_m_arg(args, 0);
    if p.tag != 6 { panic(77075); };
    let at = __psa_str_find_sub(v, p);
    if at < 0 { return __psa_box_int(-1); };
    return __psa_box_int(__psa_str_cp_of_byte(v, at));
}

fn __psa_m_isdigit(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 6 { panic(77069); };
    __psa_m_argc(args, 0, 0);
    let n = __psa_bytes_len_h(v.ipay as ptr);
    if n == 0 { return __psa_box_bool(false); };
    let i: i64 = 0;
    while i < n {
        let b = __psa_str_byte(v, i);
        if b < 48 || b > 57 { return __psa_box_bool(false); };
        i = i + 1;
    }
    return __psa_box_bool(true);
}

fn __psa_m_isalpha(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 6 { panic(77069); };
    __psa_m_argc(args, 0, 0);
    let n = __psa_bytes_len_h(v.ipay as ptr);
    if n == 0 { return __psa_box_bool(false); };
    let i: i64 = 0;
    while i < n {
        let b = __psa_str_byte(v, i);
        if !((b >= 65 && b <= 90) || (b >= 97 && b <= 122)) { return __psa_box_bool(false); };
        i = i + 1;
    }
    return __psa_box_bool(true);
}

"#;

/// Arbitrary-precision int runtime (bigint batch): PART 1 glue
/// plus PART 2 replacement bodies for the DYN_RUNTIME operators
/// (the master copies were removed from DYN_RUNTIME at merge).
/// Emitted under the same used_dyn gate as DYN_RUNTIME, plus
/// the merge-added __psa_dyn_int glue (M9) at the end.
const BIG_RUNTIME: &str = r#"// =====================================================================
// BIG_RUNTIME — psa dyn arbitrary-precision int (PsaVal tag 17).
//
// Merge shape (see ../NOTES.md for the full checklist):
//   * This text becomes a new const BIG_RUNTIME in emit.rs, emitted
//     immediately AFTER DYN_RUNTIME under the same gate
//     (emit.rs:844 `out.push_str(DYN_RUNTIME);`).
//   * Header imports (emit.rs:792-815): add, under the same gate,
//         @import "sla_std/bigint.sla"
//     sla_std/bigint.sla is present, byte-identical, in both trees
//     (sa_plugin_sla/sla_std/bigint.sla and sla/sla_std/bigint.sla,
//     md5 1bb52dd26bedccf2f469b37160cf402b).
//   * PART 1 below is new glue. PART 2 is FULL REPLACEMENT bodies for
//     existing DYN_RUNTIME functions (copy-replace at merge); each is
//     the master body plus its tag-17 branches, nothing else changed.
//
// Canonical form invariant: tag 17 carries ONLY values outside the
// i64 range; every int-producing path returns through __psa_big_norm,
// which demotes in-range results to tag 0. See ../SEMANTICS.md.
// =====================================================================

// ---------------------------------------------------------------------
// PART 1 — glue
// ---------------------------------------------------------------------

// Read the sign of a boxed bigint without cloning the magnitude.
fn __psa_big_sign(v: PsaVal) -> i64 {
    let b = unsafe { Box::from_raw(v.ipay as *BigInt) };
    let s = (*b).sign;
    let rp: i64 = Box::into_raw(b) as i64;
    return s;
}

// Clone the BigInt out of a tag-17 value (box round-trip idiom, same
// as __psa_vec_get_h).
fn __psa_big_unbox(v: PsaVal) -> BigInt {
    let b = unsafe { Box::from_raw(v.ipay as *BigInt) };
    let out = BigInt { sign: (*b).sign, mag: clone_mag(&(*b).mag) };
    let rp: i64 = Box::into_raw(b) as i64;
    return out;
}

// Any int-flavoured PsaVal (tag 0, 2 or 17) as a BigInt.
fn __psa_big_val(v: PsaVal) -> BigInt {
    if v.tag == 17 { return __psa_big_unbox(v); };
    return big_from_i64(v.ipay);
}

fn __psa_big_box(b: BigInt) -> PsaVal {
    return PsaVal { tag: 17, ipay: Box::into_raw(Box::new(b)) as i64, fpay: 0.0 };
}

// Does b fit the i64 range? Bound limbs (base 10^9, little-endian):
//   i64::MAX  = 9223372036854775807 -> [854775807, 223372036, 9]
//   |i64::MIN| = 9223372036854775808 -> [854775808, 223372036, 9]
fn __psa_big_fits(b: BigInt) -> bool {
    let n = len(b.mag);
    if n <= 2 { return true; };
    if n > 3 { return false; };
    if b.mag[2] != 9 { return b.mag[2] < 9; };
    if b.mag[1] != 223372036 { return b.mag[1] < 223372036; };
    if b.sign < 0 { return b.mag[0] <= 854775808; };
    return b.mag[0] <= 854775807;
}

// Precondition: __psa_big_fits(b). Accumulates negatively so the
// i64::MIN magnitude (2^63) never overflows a positive intermediate.
fn __psa_big_extract(b: BigInt) -> i64 {
    let v: i64 = 0;
    let i = len(b.mag);
    while i > 0 {
        i = i - 1;
        v = v * 1000000000 - b.mag[i];
    }
    if b.sign > 0 { return -v; };
    return v;
}

// The norm gate: EVERY int-producing dyn path returns through here.
fn __psa_big_norm(b: BigInt) -> PsaVal {
    if b.sign == 0 { return __psa_box_int(0); };
    if __psa_big_fits(b) { return __psa_box_int(__psa_big_extract(b)); };
    return __psa_big_box(b);
}

// Decimal text of a tag-17 value as a psa str (tag 6).
fn __psa_big_to_psastr(v: PsaVal) -> PsaVal {
    let b = __psa_big_unbox(v);
    let s = big_to_decimal_str(&b);
    let bs = s.as_bytes();
    let out = __psa_str_new();
    let i = 0;
    while i < len(bs) {
        out = __psa_str_push(out, bs[i]);
        i = i + 1;
    }
    return out;
}

// Parse a psa str (tag 6, optional leading - or +) into a normed int.
// Used for big literals (emit boxes the digit text as a tag-6 value)
// and for int(str) results that overflow i64.
fn __psa_big_from_psastr(v: PsaVal) -> PsaVal {
    let n = __psa_str_bytes_len(v);
    let digits: Vec<i64> = Vec::new();
    let sign = 1;
    let i = 0;
    if n > 0 {
        let c0 = __psa_str_byte(v, 0);
        if c0 == 45 { sign = -1; i = 1; } else if c0 == 43 { i = 1; };
    };
    while i < n {
        digits.push(__psa_str_byte(v, i) - 48);
        i = i + 1;
    }
    return __psa_big_norm(big_from_digits(&digits, sign));
}

fn __psa_big_add(a: PsaVal, b: PsaVal) -> PsaVal {
    let x = __psa_big_val(a);
    let y = __psa_big_val(b);
    return __psa_big_norm(big_add(&x, &y));
}

fn __psa_big_sub(a: PsaVal, b: PsaVal) -> PsaVal {
    let x = __psa_big_val(a);
    let y = __psa_big_val(b);
    return __psa_big_norm(big_sub(&x, &y));
}

fn __psa_big_mul(a: PsaVal, b: PsaVal) -> PsaVal {
    let x = __psa_big_val(a);
    let y = __psa_big_val(b);
    return __psa_big_norm(big_mul(&x, &y));
}

// sla_std big_divmod is already floor division (quotient rounded
// toward -inf, remainder takes the divisor's sign — see the sign
// adjustment at the tail of big_divmod in sla_std/bigint.sla), i.e.
// exactly CPython // and %. Division by zero panics inside
// big_divmod (sla_std panic 9001); see NOTES open point O2.
fn __psa_big_floordiv(a: PsaVal, b: PsaVal) -> PsaVal {
    let x = __psa_big_val(a);
    let y = __psa_big_val(b);
    let d = big_divmod(&x, &y);
    return __psa_big_norm(d.q);
}

fn __psa_big_floormod(a: PsaVal, b: PsaVal) -> PsaVal {
    let x = __psa_big_val(a);
    let y = __psa_big_val(b);
    let d = big_divmod(&x, &y);
    return __psa_big_norm(d.r);
}

fn __psa_big_cmp(a: PsaVal, b: PsaVal) -> i64 {
    let x = __psa_big_val(a);
    let y = __psa_big_val(b);
    return big_cmp(&x, &y);
}

// 2^k as a BigInt, k >= 0. Resource bound (SEMANTICS §7): results
// past ~2^20 limbs (~31.4M bits) are refused with the overflow code.
fn __psa_big_pow2(k: i64) -> BigInt {
    if k > 31457280 { panic(77080); };
    let r = big_one();
    let base = big_from_i64(2);
    let n = k;
    while n > 0 {
        if (n & 1) == 1 { r = big_mul(&r, &base); };
        n = n >> 1;
        if n > 0 {
            let sq = BigInt { sign: base.sign, mag: clone_mag(&base.mag) };
            base = big_mul(&base, &sq);
        };
    }
    return r;
}

// Bigint power. Caller guarantees the exponent is a tag-0/2 value
// >= 0; a tag-17 exponent is refused (SEMANTICS §7).
fn __psa_big_pow(a: PsaVal, e: PsaVal) -> PsaVal {
    if e.tag == 17 { panic(77080); };
    let base = __psa_big_val(a);
    let r = big_one();
    let n = e.ipay;
    while n > 0 {
        if (n & 1) == 1 {
            r = big_mul(&r, &base);
            if len(r.mag) > 1048576 { panic(77080); };
        };
        n = n >> 1;
        if n > 0 {
            let sq = BigInt { sign: base.sign, mag: clone_mag(&base.mag) };
            base = big_mul(&base, &sq);
            if len(base.mag) > 1048576 { panic(77080); };
        };
    }
    return __psa_big_norm(r);
}

fn __psa_big_shl(a: PsaVal, k: i64) -> PsaVal {
    if k < 0 { panic(77014); };
    let x = __psa_big_val(a);
    let m = __psa_big_pow2(k);
    return __psa_big_norm(big_mul(&x, &m));
}

// Arithmetic shift right == floor division by 2^k (CPython).
fn __psa_big_shr(a: PsaVal, k: i64) -> PsaVal {
    if k < 0 { panic(77014); };
    let x = __psa_big_val(a);
    let m = __psa_big_pow2(k);
    let d = big_divmod(&x, &m);
    return __psa_big_norm(d.q);
}

// Checked i64 power returning a PsaVal: the fast path stays in i64
// (per-step checks identical in shape to __psa_dyn_mul's), overflow
// promotes to the bigint path. Replaces the dyn use of __psa_ipow,
// which wraps silently on overflow (pre-existing dyn defect, fixed
// here for the dyn path only; the static path keeps __psa_ipow).
fn __psa_ipow2(a: i64, e: i64) -> PsaVal {
    let r: i64 = 1;
    let b = a;
    let n = e;
    while n > 0 {
        if (n & 1) == 1 {
            if (r == -1 && b == -9223372036854775807 - 1) || (b == -1 && r == -9223372036854775807 - 1) {
                return __psa_big_pow(__psa_box_int(a), __psa_box_int(e));
            };
            let t = r * b;
            if r != 0 && t / r != b {
                return __psa_big_pow(__psa_box_int(a), __psa_box_int(e));
            };
            r = t;
        };
        n = n >> 1;
        if n > 0 {
            let t = b * b;
            if b != 0 && t / b != b {
                return __psa_big_pow(__psa_box_int(a), __psa_box_int(e));
            };
            b = t;
        };
    }
    return __psa_box_int(r);
}

// tag-17 -> f64 with CPython OverflowError semantics: a magnitude
// past f64::MAX panics 77080 (its registered cp_surface IS
// OverflowError). The bound is exact: f64::MAX as an integer is
// 2^1024 - 2^971, built with the glue's own pow2 and compared in
// the bigint domain — no decimal-string constants needed.
fn __psa_big_as_float(v: PsaVal) -> f64 {
    let b = __psa_big_unbox(v);
    let hi = __psa_big_pow2(1024);
    let lo = __psa_big_pow2(971);
    let maxf = big_sub(&hi, &lo);
    let ab = big_abs(&b);
    if big_cmp(&ab, &maxf) > 0 { panic(77080); };
    if b.sign == 0 { return 0.0; };
    // Exact conversion (compile-batch fix for the staging O3 verify
    // point: the decimal-string route through the naive
    // __psa_str_to_float drifted an ulp on float(2^63)). Extract
    // the top 53 bits with big_divmod, round to nearest (ties to
    // even) in the integer domain, then scale by a power of two,
    // which is exact in f64.
    let top = ab.mag[len(ab.mag) - 1];
    let td = 0;
    let t = top;
    while t > 0 {
        td = td + 1;
        t = t / 10;
    }
    let d = (len(ab.mag) - 1) * 9 + td;
    let e = (d - 1) * 3321928 / 1000000;
    let p = __psa_big_pow2(e);
    while big_cmp(&ab, &p) < 0 {
        e = e - 1;
        p = __psa_big_pow2(e);
    }
    let p2 = __psa_big_pow2(e + 1);
    while big_cmp(&ab, &p2) >= 0 {
        e = e + 1;
        p2 = __psa_big_pow2(e + 1);
    }
    let s = e - 52;
    let q: i64 = 0;
    if s >= 0 {
        let ps = __psa_big_pow2(s);
        let dm = big_divmod(&ab, &ps);
        q = __psa_big_extract(dm.q);
        if s > 0 {
            let half = __psa_big_pow2(s - 1);
            let c = big_cmp(&dm.r, &half);
            if c > 0 || (c == 0 && (q & 1) == 1) { q = q + 1; };
        };
        if q == 9007199254740992 {
            q = q / 2;
            e = e + 1;
        };
    } else {
        q = __psa_big_extract(ab);
    };
    let f = q as f64;    let sc = e - 52;
    let i = 0;
    if sc >= 0 {
        while i < sc {
            f = f * 2.0;
            i = i + 1;
        }
    } else {
        while i < (0 - sc) {
            f = f / 2.0;
            i = i + 1;
        }
    };
    if b.sign < 0 { return 0.0 - f; };
    return f;
}

// ---------------------------------------------------------------------
// PART 2 — replacement bodies for DYN_RUNTIME functions.
// Each is the master body with tag-17 branches added; all other
// branches are byte-identical to master.
// ---------------------------------------------------------------------

// REPLACES __psa_as_float (emit.rs DYN_RUNTIME): tag 17 added.
fn __psa_as_float(v: PsaVal) -> f64 {
    if v.tag == 1 { return v.fpay; };
    if v.tag == 0 || v.tag == 2 { return v.ipay as f64; };
    if v.tag == 17 { return __psa_big_as_float(v); };
    panic(77010);
}

// REPLACES __psa_dyn_add: i64 overflow promotes instead of
// panic(77080); tag 17 goes to the bigint path.
fn __psa_dyn_add(a: PsaVal, b: PsaVal) -> PsaVal {
    if a.tag == 1 && (b.tag == 0 || b.tag == 2 || b.tag == 17) { return __psa_box_float(__psa_as_float(a) + __psa_as_float(b)); };
    if b.tag == 1 && (a.tag == 0 || a.tag == 2 || a.tag == 17) { return __psa_box_float(__psa_as_float(a) + __psa_as_float(b)); };
    if a.tag == 17 || b.tag == 17 { return __psa_big_add(a, b); };
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) {
        if b.ipay > 0 && a.ipay > 9223372036854775807 - b.ipay { return __psa_big_add(a, b); };
        if b.ipay < 0 && a.ipay < -9223372036854775807 - 1 - b.ipay { return __psa_big_add(a, b); };
        return __psa_box_int(a.ipay + b.ipay);
    };
    if a.tag <= 2 && b.tag <= 2 { return __psa_box_float(__psa_as_float(a) + __psa_as_float(b)); };
    if a.tag == 6 && b.tag == 6 { return __psa_str_concat(a, b); };
    if a.tag == 15 && b.tag == 15 { return __psa_bytes_concat(a, b); };
    panic(77011);
}

// REPLACES __psa_dyn_sub: same promotion rule.
fn __psa_dyn_sub(a: PsaVal, b: PsaVal) -> PsaVal {
    if a.tag == 1 && (b.tag == 0 || b.tag == 2 || b.tag == 17) { return __psa_box_float(__psa_as_float(a) - __psa_as_float(b)); };
    if b.tag == 1 && (a.tag == 0 || a.tag == 2 || a.tag == 17) { return __psa_box_float(__psa_as_float(a) - __psa_as_float(b)); };
    if a.tag == 17 || b.tag == 17 { return __psa_big_sub(a, b); };
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) {
        if b.ipay < 0 && a.ipay > 9223372036854775807 + b.ipay { return __psa_big_sub(a, b); };
        if b.ipay > 0 && a.ipay < -9223372036854775807 - 1 + b.ipay { return __psa_big_sub(a, b); };
        return __psa_box_int(a.ipay - b.ipay);
    };
    if a.tag <= 2 && b.tag <= 2 { return __psa_box_float(__psa_as_float(a) - __psa_as_float(b)); };
    panic(77011);
}

// REPLACES __psa_dyn_mul: same promotion rule; str/bytes repeat
// counts may now be tag 17 — a repeat count past i64 is refused
// (CPython would attempt an astronomically large allocation).
fn __psa_dyn_mul(a: PsaVal, b: PsaVal) -> PsaVal {
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) {
        let x = a.ipay;
        let y = b.ipay;
        if (x == -1 && y == -9223372036854775807 - 1) || (y == -1 && x == -9223372036854775807 - 1) { return __psa_big_mul(a, b); };
        let r = x * y;
        if x != 0 && r / x != y { return __psa_big_mul(a, b); };
        return __psa_box_int(r);
    };
    if a.tag == 17 || b.tag == 17 {
        if a.tag == 1 && (b.tag == 0 || b.tag == 2 || b.tag == 17) { return __psa_box_float(__psa_as_float(a) * __psa_as_float(b)); };
        if b.tag == 1 && (a.tag == 0 || a.tag == 2 || a.tag == 17) { return __psa_box_float(__psa_as_float(a) * __psa_as_float(b)); };
        if a.tag == 6 || b.tag == 6 || a.tag == 15 || b.tag == 15 { panic(77080); };
        return __psa_big_mul(a, b);
    };
    if a.tag <= 2 && b.tag <= 2 { return __psa_box_float(__psa_as_float(a) * __psa_as_float(b)); };
    if a.tag == 6 && (b.tag == 0 || b.tag == 2) { return __psa_str_repeat(a, b.ipay); };
    if b.tag == 6 && (a.tag == 0 || a.tag == 2) { return __psa_str_repeat(b, a.ipay); };
    if a.tag == 15 && (b.tag == 0 || b.tag == 2) { return __psa_bytes_repeat(a, b.ipay); };
    if b.tag == 15 && (a.tag == 0 || a.tag == 2) { return __psa_bytes_repeat(b, a.ipay); };
    panic(77011);
}

// REPLACES __psa_dyn_floordiv: int path exact via big_divmod when
// either side is tag 17 or the i64 corner (MIN // -1) promotes.
fn __psa_dyn_floordiv(a: PsaVal, b: PsaVal) -> PsaVal {
    if a.tag == 1 && (b.tag == 0 || b.tag == 2 || b.tag == 17) { return __psa_box_float(__psa_dyn_floor_f(__psa_as_float(a) / __psa_as_float(b))); };
    if b.tag == 1 && (a.tag == 0 || a.tag == 2 || a.tag == 17) { return __psa_box_float(__psa_dyn_floor_f(__psa_as_float(a) / __psa_as_float(b))); };
    if a.tag == 17 || b.tag == 17 { return __psa_big_floordiv(a, b); };
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) {
        if a.ipay == -9223372036854775807 - 1 && b.ipay == -1 { return __psa_big_floordiv(a, b); };
        return __psa_box_int(__psa_floordiv(a.ipay, b.ipay));
    };
    if a.tag <= 2 && b.tag <= 2 { return __psa_box_float(__psa_dyn_floor_f(__psa_as_float(a) / __psa_as_float(b))); };
    panic(77011);
}

// REPLACES __psa_dyn_mod: same routing (MIN % -1 == 0 already holds
// on the i64 path and on the bigint path alike).
fn __psa_dyn_mod(a: PsaVal, b: PsaVal) -> PsaVal {
    if (a.tag == 1 && (b.tag == 0 || b.tag == 2 || b.tag == 17)) || (b.tag == 1 && (a.tag == 0 || a.tag == 2 || a.tag == 17)) {
        let fa = __psa_as_float(a);
        let fb = __psa_as_float(b);
        return __psa_box_float(fa - __psa_dyn_floor_f(fa / fb) * fb);
    };
    if a.tag == 17 || b.tag == 17 { return __psa_big_floormod(a, b); };
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) { return __psa_box_int(__psa_floormod(a.ipay, b.ipay)); };
    if a.tag <= 2 && b.tag <= 2 {
        let fa = __psa_as_float(a);
        let fb = __psa_as_float(b);
        return __psa_box_float(fa - __psa_dyn_floor_f(fa / fb) * fb);
    };
    panic(77011);
}

// REPLACES __psa_dyn_pow: checked i64 path (__psa_ipow2) with
// promotion; negative exponent keeps the float route; a tag-17
// exponent on an int base is refused per SEMANTICS §7 (a float
// base still takes the float route, where as_float refuses past
// the f64 range).
fn __psa_dyn_pow(a: PsaVal, b: PsaVal) -> PsaVal {
    if b.tag == 17 {
        if a.tag == 1 { return __psa_box_float(sa_math_pow(__psa_as_float(a), __psa_big_as_float(b))); };
        panic(77080);
    };
    if a.tag == 17 {
        if b.tag == 1 { return __psa_box_float(sa_math_pow(__psa_big_as_float(a), b.fpay)); };
        if (b.tag == 0 || b.tag == 2) && b.ipay >= 0 { return __psa_big_pow(a, b); };
        if (b.tag == 0 || b.tag == 2) && b.ipay < 0 { return __psa_box_float(sa_math_pow(__psa_big_as_float(a), __psa_as_float(b))); };
        panic(77011);
    };
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) && b.ipay >= 0 { return __psa_ipow2(a.ipay, b.ipay); };
    if a.tag <= 2 && b.tag <= 2 { return __psa_box_float(sa_math_pow(__psa_as_float(a), __psa_as_float(b))); };
    panic(77011);
}

// REPLACES __psa_dyn_eq: tag-17 pairs compare by value; a tag-17
// against tag 0/2 is unequal BY THE CANONICAL FORM (disjoint
// ranges); against float it goes through as_float (precision
// caveat registered in NOTES O3 — exact float<->bigint comparison
// is a follow-up, not silently claimed here).
fn __psa_dyn_eq(a: PsaVal, b: PsaVal) -> bool {
    if a.tag == 17 && b.tag == 17 {
        let x = __psa_big_unbox(a);
        let y = __psa_big_unbox(b);
        return big_eq(&x, &y);
    };
    if a.tag == 17 || b.tag == 17 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) == __psa_as_float(b); };
        return false;
    };
    if a.tag == 5 && b.tag == 5 { return __psa_dict_eq(a, b); };
    if a.tag == 5 || b.tag == 5 { return false; };
    if a.tag == 4 && b.tag == 4 { return __psa_list_eq(a, b); };
    if a.tag == 6 && b.tag == 6 { return __psa_str_eq(a, b); };
    if a.tag == 15 && b.tag == 15 { return __psa_str_eq(a, b); };
    if a.tag == 10 && b.tag == 10 { return __psa_tuple_eq(a, b); };
    if a.tag == 11 && b.tag == 11 { return __psa_set_eq(a, b); };
    if a.tag == 8 || a.tag == 9 || a.tag == 16 || b.tag == 8 || b.tag == 9 || b.tag == 16 { return a.tag == b.tag && a.ipay == b.ipay; };
    if a.tag == 3 && b.tag == 3 { return true; };
    if a.tag <= 2 && b.tag <= 2 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) == __psa_as_float(b); };
        return a.ipay == b.ipay;
    };
    return false;
}

// REPLACES __psa_dyn_lt / __psa_dyn_le / __psa_dyn_gt / __psa_dyn_ge:
// int-vs-int comparisons are exact via big_cmp whenever a tag 17
// participates; int-vs-float uses as_float (NOTES O3 caveat).
fn __psa_dyn_lt(a: PsaVal, b: PsaVal) -> bool {
    if a.tag == 17 || b.tag == 17 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) < __psa_as_float(b); };
        return __psa_big_cmp(a, b) < 0;
    };
    if a.tag <= 2 && b.tag <= 2 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) < __psa_as_float(b); };
        return a.ipay < b.ipay;
    };
    if a.tag == 6 && b.tag == 6 { return __psa_str_cmp(a, b) < 0; };
    if a.tag == 15 && b.tag == 15 { return __psa_str_cmp(a, b) < 0; };
    panic(77012);
}

fn __psa_dyn_le(a: PsaVal, b: PsaVal) -> bool {
    if a.tag == 17 || b.tag == 17 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) <= __psa_as_float(b); };
        return __psa_big_cmp(a, b) <= 0;
    };
    if a.tag <= 2 && b.tag <= 2 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) <= __psa_as_float(b); };
        return a.ipay <= b.ipay;
    };
    if a.tag == 6 && b.tag == 6 { return __psa_str_cmp(a, b) <= 0; };
    if a.tag == 15 && b.tag == 15 { return __psa_str_cmp(a, b) <= 0; };
    panic(77012);
}

fn __psa_dyn_gt(a: PsaVal, b: PsaVal) -> bool {
    if a.tag == 17 || b.tag == 17 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) > __psa_as_float(b); };
        return __psa_big_cmp(a, b) > 0;
    };
    if a.tag <= 2 && b.tag <= 2 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) > __psa_as_float(b); };
        return a.ipay > b.ipay;
    };
    if a.tag == 6 && b.tag == 6 { return __psa_str_cmp(a, b) > 0; };
    if a.tag == 15 && b.tag == 15 { return __psa_str_cmp(a, b) > 0; };
    panic(77012);
}

fn __psa_dyn_ge(a: PsaVal, b: PsaVal) -> bool {
    if a.tag == 17 || b.tag == 17 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) >= __psa_as_float(b); };
        return __psa_big_cmp(a, b) >= 0;
    };
    if a.tag <= 2 && b.tag <= 2 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) >= __psa_as_float(b); };
        return a.ipay >= b.ipay;
    };
    if a.tag == 6 && b.tag == 6 { return __psa_str_cmp(a, b) >= 0; };
    if a.tag == 15 && b.tag == 15 { return __psa_str_cmp(a, b) >= 0; };
    panic(77012);
}

// REPLACES __psa_dyn_truthy: tag 17 added (nonzero sign == truthy).
fn __psa_dyn_truthy(v: PsaVal) -> bool {
    if v.tag == 0 { return v.ipay != 0; };
    if v.tag == 1 { return v.fpay != 0.0; };
    if v.tag == 2 { return v.ipay != 0; };
    if v.tag == 3 { return false; };
    if v.tag == 4 { return __psa_list_len(v) != 0; };
    if v.tag == 5 { return __psa_dict_len(v) != 0; };
    if v.tag == 6 { return __psa_str_bytes_len(v) != 0; };
    if v.tag == 15 { return __psa_str_bytes_len(v) != 0; };
    if v.tag == 8 { return __psa_obj_truthy(v); };
    if v.tag == 9 { return true; };
    if v.tag == 10 { return __psa_vec_len_h(v.ipay as ptr) != 0; };
    if v.tag == 11 { return __psa_vec_len_h(v.ipay as ptr) != 0; };
    if v.tag == 12 { return true; };
    if v.tag == 13 { return true; };
    if v.tag == 14 { return true; };
    if v.tag == 16 { return true; };
    if v.tag == 17 { return __psa_big_sign(v) != 0; };
    panic(77010);
}

// REPLACES __psa_dyn_neg: negating i64::MIN promotes (it is the
// demotion gateway for the -9223372036854775808 literal as well).
fn __psa_dyn_neg(v: PsaVal) -> PsaVal {
    if v.tag == 17 {
        let x = __psa_big_unbox(v);
        return __psa_big_norm(big_neg(&x));
    };
    if v.tag == 0 || v.tag == 2 {
        if v.ipay == -9223372036854775807 - 1 { return __psa_big_neg_of(v); };
        return __psa_box_int(-(v.ipay));
    };
    if v.tag == 1 { return __psa_box_float(-(v.fpay)); };
    panic(77015);
}

// Helper for __psa_dyn_neg's promotion edge.
fn __psa_big_neg_of(v: PsaVal) -> PsaVal {
    let x = big_from_i64(v.ipay);
    return __psa_big_norm(big_neg(&x));
}

// REPLACES __psa_dyn_bitnot: ~x == -x-1 in both domains.
fn __psa_dyn_bitnot(v: PsaVal) -> PsaVal {
    if v.tag == 17 {
        let x = __psa_big_unbox(v);
        let nx = big_neg(&x);
        let one = big_one();
        return __psa_big_norm(big_sub(&nx, &one));
    };
    if v.tag == 0 || v.tag == 2 { return __psa_box_int((-(v.ipay) - 1)); };
    panic(77013);
}

// REPLACES __psa_dyn_bitand / __psa_dyn_bitor / __psa_dyn_bitxor:
// bitwise ops on tag 17 are LOUDLY REFUSED in this batch — the
// sla_std base has no bitwise surface and infinite-two's-complement
// semantics on a sign-magnitude base must not be improvised
// (SEMANTICS §3, NOTES follow-up F1).
fn __psa_dyn_bitand(a: PsaVal, b: PsaVal) -> PsaVal {
    if a.tag == 17 || b.tag == 17 { panic(77013); };
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) { return __psa_box_int(a.ipay & b.ipay); };
    panic(77013);
}

fn __psa_dyn_bitor(a: PsaVal, b: PsaVal) -> PsaVal {
    if a.tag == 17 || b.tag == 17 { panic(77013); };
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) { return __psa_box_int(a.ipay | b.ipay); };
    panic(77013);
}

fn __psa_dyn_bitxor(a: PsaVal, b: PsaVal) -> PsaVal {
    if a.tag == 17 || b.tag == 17 { panic(77013); };
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) { return __psa_box_int(a.ipay ^ b.ipay); };
    panic(77013);
}

// REPLACES __psa_dyn_shl: CPython shifts are unbounded — counts
// >= 64 and overflowing results promote to the bigint path
// instead of the current panic(77014) (which stays for negative
// counts only).
fn __psa_dyn_shl(a: PsaVal, b: PsaVal) -> PsaVal {
    if b.tag == 17 {
        if a.tag != 0 && a.tag != 2 && a.tag != 17 { panic(77013); };
        if __psa_big_sign(b) < 0 { panic(77014); };
        panic(77080);
    };
    if a.tag == 17 {
        if b.tag == 0 || b.tag == 2 { return __psa_big_shl(a, b.ipay); };
        panic(77013);
    };
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) {
        if b.ipay < 0 { panic(77014); };
        if b.ipay >= 64 { return __psa_big_shl(a, b.ipay); };
        let r = a.ipay << b.ipay;
        if (r >> b.ipay) != a.ipay { return __psa_big_shl(a, b.ipay); };
        return __psa_box_int(r);
    };
    panic(77013);
}

// REPLACES __psa_dyn_shr: arithmetic shift; a tag-17 left operand
// goes through floor division by 2^k (exact CPython semantics).
fn __psa_dyn_shr(a: PsaVal, b: PsaVal) -> PsaVal {
    if b.tag == 17 {
        if __psa_big_sign(b) < 0 { panic(77014); };
        if a.tag == 0 || a.tag == 2 {
            if a.ipay >= 0 { return __psa_box_int(0); };
            return __psa_box_int(-1);
        };
        if a.tag == 17 {
            if __psa_big_sign(a) >= 0 { return __psa_box_int(0); };
            return __psa_box_int(-1);
        };
        panic(77013);
    };
    if a.tag == 17 {
        if b.tag == 0 || b.tag == 2 { return __psa_big_shr(a, b.ipay); };
        panic(77013);
    };
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) {
        if b.ipay < 0 { panic(77014); };
        if b.ipay >= 64 {
            if a.ipay >= 0 { return __psa_box_int(0); };
            return __psa_box_int(-1);
        };
        return __psa_box_int(a.ipay >> b.ipay);
    };
    panic(77013);
}

// REPLACES __psa_unbox_int: a tag-17 value can never satisfy a
// static i64 slot (canonical form) — overflow refusal, not a
// type error.
fn __psa_unbox_int(v: PsaVal) -> i64 {
    if v.tag == 0 { return v.ipay; };
    if v.tag == 2 { return v.ipay; };
    if v.tag == 17 { panic(77080); };
    panic(77020);
}

// REPLACES __psa_key_eq: dict/set key equality. By the canonical
// form a tag-17 key only ever equals another tag-17 key with the
// same value; float keys keep the existing float route.
fn __psa_key_eq(a: PsaVal, b: PsaVal) -> bool {
    if a.tag == 17 && b.tag == 17 {
        let x = __psa_big_unbox(a);
        let y = __psa_big_unbox(b);
        return big_eq(&x, &y);
    };
    if a.tag == 17 || b.tag == 17 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) == __psa_as_float(b); };
        return false;
    };
    if a.tag <= 2 && b.tag <= 2 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) == __psa_as_float(b); };
        return a.ipay == b.ipay;
    };
    if a.tag == 3 && b.tag == 3 { return true; };
    if a.tag == 6 && b.tag == 6 { return __psa_str_eq(a, b); };
    if a.tag == 15 && b.tag == 15 { return __psa_str_eq(a, b); };
    if a.tag == 10 && b.tag == 10 { return __psa_tuple_key_eq(a, b); };
    if a.tag == 8 && b.tag == 8 { return a.ipay == b.ipay; };
    return false;
}

// REPLACES __psa_print_repr: tag-17 branch (decimal text).
fn __psa_print_repr(v: PsaVal) {
    if v.tag == 0 { __psa_print_i64(v.ipay); return; };
    if v.tag == 17 { __psa_print_str(__psa_big_to_psastr(v)); return; };
    if v.tag == 2 {
        if v.ipay != 0 { sla_print_bytes(STR_PTR("True"), STR_LEN("True")); } else { sla_print_bytes(STR_PTR("False"), STR_LEN("False")); };
        return;
    };
    if v.tag == 3 { sla_print_bytes(STR_PTR("None"), STR_LEN("None")); return; };
    if v.tag == 4 { __psa_print_list_repr(v); return; };
    if v.tag == 6 { __psa_print_str(__psa_str_repr(v)); return; };
    if v.tag == 15 { __psa_print_str(__psa_bytes_repr(v)); return; };
    if v.tag == 5 { __psa_print_dict_repr(v); return; };
    if v.tag == 8 { __psa_print_str(__psa_obj_repr(v)); return; };
    if v.tag == 10 { __psa_print_tuple_repr(v); return; };
    if v.tag == 11 { __psa_print_set_repr(v); return; };
    panic(77058);
}

// REPLACES __psa_print_dyn_nl: tag-17 branch.
fn __psa_print_dyn_nl(v: PsaVal) {
    if v.tag == 4 { __psa_print_list_repr(v); return; };
    if v.tag == 6 { __psa_print_str(v); return; };
    if v.tag == 15 { __psa_print_str(__psa_bytes_repr(v)); return; };
    if v.tag == 5 { __psa_print_dict_repr(v); return; };
    if v.tag == 8 { __psa_print_str(__psa_obj_str(v)); return; };
    if v.tag == 10 { __psa_print_tuple_repr(v); return; };
    if v.tag == 11 { __psa_print_set_repr(v); return; };
    if v.tag == 12 { __psa_print_str(__psa_file_str(v)); return; };
    if v.tag == 13 { __psa_print_str(__psa_str_of(v)); return; };
    if v.tag == 14 { __psa_print_str(__psa_str_of(v)); return; };
    if v.tag == 0 { __psa_print_i64(v.ipay); return; };
    if v.tag == 17 { __psa_print_str(__psa_big_to_psastr(v)); return; };
    if v.tag == 2 {
        if v.ipay != 0 { sla_print_bytes(STR_PTR("True"), STR_LEN("True")); } else { sla_print_bytes(STR_PTR("False"), STR_LEN("False")); };
        return;
    };
    if v.tag == 3 { sla_print_bytes(STR_PTR("None"), STR_LEN("None")); return; };
    panic(77030);
}

// REPLACES __psa_print_dyn: tag-17 branch.
fn __psa_print_dyn(v: PsaVal) -> i32 {
    if v.tag == 4 { __psa_print_list_repr(v); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 6 { __psa_print_str(v); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 15 { __psa_print_str(__psa_bytes_repr(v)); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 5 { __psa_print_dict_repr(v); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 8 { __psa_print_str(__psa_obj_str(v)); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 10 { __psa_print_tuple_repr(v); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 11 { __psa_print_set_repr(v); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 12 { __psa_print_str(__psa_file_str(v)); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 13 { __psa_print_str(__psa_str_of(v)); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 14 { __psa_print_str(__psa_str_of(v)); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 0 { __psa_print_i64(v.ipay); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 17 { __psa_print_str(__psa_big_to_psastr(v)); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 2 {
        if v.ipay != 0 { println("True"); } else { println("False"); };
        return 0;
    };
    if v.tag == 3 { println("None"); return 0; };
    panic(77030);
}

// PATCH to __psa_str_of (emit.rs, the D5 string-conversion block):
// insert before its final panic(77030):
//     if v.tag == 17 { return __psa_big_to_psastr(v); };

// PATCH to __psa_to_float: insert before panic(77075):
//     if v.tag == 17 { return __psa_big_as_float(v); };

// PATCH to __psa_to_int: tag 17 can never yield an i64 — insert
// before panic(77075):
//     if v.tag == 17 { panic(77080); };
// (The DYNAMIC int() builtin gains its bigint behaviour through
// __psa_big_from_psastr for str inputs and through norm-preserving
// passthrough for tag-17 inputs — see NOTES merge point M9.)

// ---------------------------------------------------------------------
// Merge-added glue (compile batch, M9): the dynamic int() builtin.
// Tag 17 passes through untouched; a tag-6 string is parsed with
// __psa_str_to_int's surface (surrounding spaces, optional sign,
// digits; malformed input panics 77079) but overflow promotes to a
// bigint instead of wrapping; every other tag goes through the
// static conversion and re-boxes.
// ---------------------------------------------------------------------
fn __psa_dyn_int(v: PsaVal) -> PsaVal {
    if v.tag == 17 { return v; };
    if v.tag == 6 {
        let n = __psa_str_bytes_len(v);
        let i = 0;
        while i < n {
            if !(__psa_str_is_space(__psa_str_byte(v, i))) { break; };
            i = i + 1;
        }
        let neg = false;
        if i < n {
            if __psa_str_byte(v, i) == 43 || __psa_str_byte(v, i) == 45 {
                neg = __psa_str_byte(v, i) == 45;
                i = i + 1;
            };
        };
        let dstart = i;
        let any = false;
        let acc: i64 = 0;
        let big = false;
        while i < n {
            let b = __psa_str_byte(v, i);
            if b < 48 || b > 57 { break; };
            if !(big) {
                if acc > (9223372036854775807 - (b - 48)) / 10 {
                    big = true;
                } else {
                    acc = acc * 10 + (b - 48);
                };
            };
            any = true;
            i = i + 1;
        }
        let dend = i;
        while i < n {
            if !(__psa_str_is_space(__psa_str_byte(v, i))) { break; };
            i = i + 1;
        }
        if !(any) || i != n { panic(77079); };
        if big {
            let digits = __psa_str_new();
            if neg { digits = __psa_str_push(digits, 45u8); };
            let j = dstart;
            while j < dend {
                digits = __psa_str_push(digits, __psa_str_byte(v, j));
                j = j + 1;
            }
            return __psa_big_from_psastr(digits);
        };
        if neg { return __psa_box_int(0 - acc); };
        return __psa_box_int(acc);
    };
    return __psa_box_int(__psa_to_int(v));
}
"#;

/// D6 runtime, part 1 (emitted with every dynamic program):
/// exception values and the file-value payload accessors. An
/// exception instance is an ordinary tag-8 object; construction
/// through __psa_exc_new stores the message (a dynamic value,
/// None when absent) under the key "msg". __str__ is the
/// message's str ("" when None); __repr__ is Name('msg') or
/// Name(). The file accessors live here, not in the file part,
/// because the dynamic printers reference __psa_file_str
/// whenever dynamic values are in play; only the fs-calling
/// functions are gated on used_file.
const D6_EXC_RUNTIME: &str = r#"
// The pending-exception slot is process-global (sa_std
// globals cells), so an exception raised in a callee is
// visible to the caller's handler dispatch. Two u64 cells
// hold (tag, ipay); exception values are always tag 8 with
// fpay 0.0, so the pair is lossless. An unset cell reads
// back as tag 3 (None) = no exception pending.
fn __psa_exc_tag() -> i64 {
    let out = [0u64; 1];
    let rc = sa_global_u64_get(STR_PTR("psa.exc.tag"), STR_LEN("psa.exc.tag"), &out[0]);
    if rc != 0 { return 3; };
    return out[0] as i64;
}

fn __psa_exc_get() -> PsaVal {
    let out = [0u64; 1];
    let rc = sa_global_u64_get(STR_PTR("psa.exc.tag"), STR_LEN("psa.exc.tag"), &out[0]);
    if rc != 0 { return PsaVal { tag: 3, ipay: 0, fpay: 0.0 }; };
    let out2 = [0u64; 1];
    let rc2 = sa_global_u64_get(STR_PTR("psa.exc.ipay"), STR_LEN("psa.exc.ipay"), &out2[0]);
    if rc2 != 0 { return PsaVal { tag: 3, ipay: 0, fpay: 0.0 }; };
    return PsaVal { tag: out[0] as i64, ipay: out2[0] as i64, fpay: 0.0 };
}

fn __psa_exc_store(v: PsaVal) {
    let rc = sa_global_u64_set(STR_PTR("psa.exc.tag"), STR_LEN("psa.exc.tag"), v.tag as u64);
    let rc2 = sa_global_u64_set(STR_PTR("psa.exc.ipay"), STR_LEN("psa.exc.ipay"), v.ipay as u64);
}

// repr() of a dynamic value: strings quote, objects ride
// their __repr__ machinery, scalars format exactly;
// containers and floats stay blocked (77030) with the
// missing repr-to-string builders, exactly like str().
fn __psa_repr_of(v: PsaVal) -> PsaVal {
    if v.tag == 6 { return __psa_str_repr(v); };
    if v.tag == 15 { return __psa_bytes_repr(v); };
    if v.tag == 8 { return __psa_obj_repr(v); };
    if v.tag == 0 { return __psa_str_from_i64(v.ipay); };
    if v.tag == 2 { return __psa_str_from_bool(v.ipay != 0); };
    if v.tag == 3 {
        let out = __psa_str_new();
        out = __psa_str_push(out, 78u8);
        out = __psa_str_push(out, 111u8);
        out = __psa_str_push(out, 110u8);
        out = __psa_str_push(out, 101u8);
        return out;
    };
    if v.tag == 12 { return __psa_file_str(v); };
    if v.tag == 13 { return __psa_str_of(v); };
    if v.tag == 14 { return __psa_str_of(v); };
    panic(77030);
}

fn __psa_exc_msg_key() -> PsaVal {
    let k = __psa_str_new();
    k = __psa_str_push(k, 109u8);
    k = __psa_str_push(k, 115u8);
    k = __psa_str_push(k, 103u8);
    return k;
}

fn __psa_exc_msg(v: PsaVal) -> PsaVal {
    let j = __psa_obj_find(v, __psa_exc_msg_key());
    if j < 0 { return PsaVal { tag: 3, ipay: 0, fpay: 0.0 }; };
    return __psa_vec_get_h(v.ipay as ptr, j + 1);
}

fn __psa_exc_new(cid: i64, msg: PsaVal) -> PsaVal {
    let v = __psa_obj_new(cid);
    __psa_obj_set(v, __psa_exc_msg_key(), msg);
    return v;
}

fn __psa_exc_str(v: PsaVal) -> PsaVal {
    let m = __psa_exc_msg(v);
    if m.tag == 3 { return __psa_str_new(); };
    return __psa_str_of(m);
}

fn __psa_exc_repr(v: PsaVal) -> PsaVal {
    let out = __psa_str_new();
    out = __psa_str_append_str(out, __psa_class_name(v));
    let m = __psa_exc_msg(v);
    out = __psa_str_push(out, 40u8);
    if m.tag != 3 {
        out = __psa_str_append_str(out, __psa_str_repr(__psa_str_of(m)));
    };
    out = __psa_str_push(out, 41u8);
    return out;
}

// A raise of a value that is not an exception instance is a
// TypeError in CPython; v1 panics 77101.
fn __psa_exc_check(v: PsaVal) {
    if v.tag != 8 { panic(77101); };
    if !(__psa_is_exc_class(__psa_obj_class(v))) { panic(77101); };
}

// File values (tag 12): the payload is a Box<Vec<PsaVal>> of
// [handle (int), mode (int: 0 r, 1 w, 2 a), closed (bool),
// path (str)].
fn __psa_file_handle(v: PsaVal) -> i64 {
    return __psa_vec_get_h(v.ipay as ptr, 0).ipay;
}

fn __psa_file_mode(v: PsaVal) -> i64 {
    return __psa_vec_get_h(v.ipay as ptr, 1).ipay;
}

fn __psa_file_closed(v: PsaVal) -> bool {
    return __psa_vec_get_h(v.ipay as ptr, 2).ipay != 0;
}

fn __psa_file_path(v: PsaVal) -> PsaVal {
    return __psa_vec_get_h(v.ipay as ptr, 3);
}

// CPython renders a text file as
// <_io.TextIOWrapper name='PATH' mode='r'> (the encoding
// segment is omitted -- a registered divergence).
fn __psa_file_str(v: PsaVal) -> PsaVal {
    let out = __psa_str_new();
    out = __psa_str_push(out, 60u8);
    out = __psa_str_push(out, 95u8);
    out = __psa_str_push(out, 105u8);
    out = __psa_str_push(out, 111u8);
    out = __psa_str_push(out, 46u8);
    out = __psa_str_push(out, 84u8);
    out = __psa_str_push(out, 101u8);
    out = __psa_str_push(out, 120u8);
    out = __psa_str_push(out, 116u8);
    out = __psa_str_push(out, 73u8);
    out = __psa_str_push(out, 79u8);
    out = __psa_str_push(out, 87u8);
    out = __psa_str_push(out, 114u8);
    out = __psa_str_push(out, 97u8);
    out = __psa_str_push(out, 112u8);
    out = __psa_str_push(out, 112u8);
    out = __psa_str_push(out, 101u8);
    out = __psa_str_push(out, 114u8);
    out = __psa_str_push(out, 32u8);
    out = __psa_str_push(out, 110u8);
    out = __psa_str_push(out, 97u8);
    out = __psa_str_push(out, 109u8);
    out = __psa_str_push(out, 101u8);
    out = __psa_str_push(out, 61u8);
    out = __psa_str_push(out, 39u8);
    out = __psa_str_append_str(out, __psa_file_path(v));
    out = __psa_str_push(out, 39u8);
    out = __psa_str_push(out, 32u8);
    out = __psa_str_push(out, 109u8);
    out = __psa_str_push(out, 111u8);
    out = __psa_str_push(out, 100u8);
    out = __psa_str_push(out, 101u8);
    out = __psa_str_push(out, 61u8);
    out = __psa_str_push(out, 39u8);
    let m = __psa_file_mode(v);
    if m == 0 { out = __psa_str_push(out, 114u8); };
    if m == 1 { out = __psa_str_push(out, 119u8); };
    if m == 2 { out = __psa_str_push(out, 97u8); };
    out = __psa_str_push(out, 39u8);
    out = __psa_str_push(out, 62u8);
    return out;
}

// ---- f-string format specs (task 171-A) ----
// __psa_format_value(v, conv, spec) implements CPython 3.12's
// format-spec mini-language for the value kinds psa supports:
// [[fill]align][sign][#][0][width][grouping][.precision][type]
// for str / int / float / bool / None and dynamic values.
// conv: 0 = none, 1 = !r, 2 = !s, 3 = !a; spec is the
// evaluated spec text (tag 6, possibly empty). A malformed
// spec, or one invalid for the value's kind, panics 77085
// (CPython ValueError); a non-empty spec on a value kind that
// takes no spec (None, containers, objects, ...) panics 77086
// (CPython TypeError). Codes 77085/77086 are new in this batch
// and must be registered in PANIC-CODES.json at integration.

struct PsaFmt {
    fill: PsaVal,
    has_fill: bool,
    align: i64,
    sign: i64,
    alt: bool,
    zero: bool,
    width: i64,
    group: i64,
    prec: i64,
    typ: i64
}

fn __psa_fmt_space() -> PsaVal {
    let s = __psa_str_new();
    s = __psa_str_push(s, 32u8);
    return s;
}

fn __psa_fmt_default() -> PsaFmt {
    return PsaFmt { fill: __psa_fmt_space(), has_fill: false, align: 0, sign: 0, alt: false, zero: false, width: 0, group: 0, prec: -1, typ: 0 };
}

fn __psa_fmt_defaulted(f: PsaFmt) -> bool {
    return !f.has_fill && f.align == 0 && f.sign == 0 && !f.alt && !f.zero && f.width == 0 && f.group == 0 && f.prec == -1 && f.typ == 0;
}

fn __psa_fmt_is_align(b: i64) -> bool {
    return b == 60 || b == 62 || b == 94 || b == 61;
}

fn __psa_fmt_cp_len(b: i64) -> i64 {
    // Byte length of the UTF-8 code point whose lead byte
    // is b (the spec text is well-formed UTF-8: it was a
    // dynamic string built by psa code).
    if b < 128 { return 1; };
    if b / 32 == 6 { return 2; };
    if b / 16 == 14 { return 3; };
    return 4;
}

fn __psa_fmt_sub(s: PsaVal, a: i64, b: i64) -> PsaVal {
    // Bytes [a, b) of a dynamic string, as a new string.
    let out = __psa_str_new();
    let i = a;
    while i < b {
        out = __psa_str_push(out, __psa_str_byte(s, i) as u8);
        i = i + 1;
    }
    return out;
}

fn __psa_fmt_cat(a: PsaVal, b: PsaVal) -> PsaVal {
    let out = __psa_str_new();
    out = __psa_str_append_str(out, a);
    out = __psa_str_append_str(out, b);
    return out;
}

fn __psa_fmt_rev(s: PsaVal) -> PsaVal {
    let out = __psa_str_new();
    let i = __psa_str_bytes_len(s);
    while i > 0 {
        i = i - 1;
        out = __psa_str_push(out, __psa_str_byte(s, i) as u8);
    }
    return out;
}

fn __psa_fmt_parse(spec: PsaVal) -> PsaFmt {
    // Parse the format-spec mini-language over the spec's
    // bytes. Malformed input panics 77085.
    let fill_v = __psa_fmt_space();
    let has_fill = false;
    let align_v: i64 = 0;
    let sign_v: i64 = 0;
    let alt_v = false;
    let zero_v = false;
    let width_v: i64 = 0;
    let group_v: i64 = 0;
    let prec_v: i64 = -1;
    let typ_v: i64 = 0;
    let n = __psa_str_bytes_len(spec);
    let i: i64 = 0;
    // [[fill]align]: fill is one code point, recognised only
    // when an align character follows it.
    if n >= 1 {
        let l = __psa_fmt_cp_len(__psa_str_byte(spec, 0));
        if l < n && __psa_fmt_is_align(__psa_str_byte(spec, l)) {
            fill_v = __psa_fmt_sub(spec, 0, l);
            has_fill = true;
            align_v = __psa_str_byte(spec, l);
            i = l + 1;
        } else if __psa_fmt_is_align(__psa_str_byte(spec, 0)) {
            align_v = __psa_str_byte(spec, 0);
            i = 1;
        };
    };
    if has_fill {
        let fb = __psa_str_byte(fill_v, 0);
        if fb == 123 || fb == 125 { panic(77085); };
    };
    // [sign]
    if i < n {
        let b = __psa_str_byte(spec, i);
        if b == 43 || b == 45 || b == 32 {
            sign_v = b;
            i = i + 1;
        };
    };
    // [#]
    if i < n && __psa_str_byte(spec, i) == 35 {
        alt_v = true;
        i = i + 1;
    };
    // [0]: the zero flag. It makes '0' the fill (unless a
    // fill was given explicitly) and, for numbers with no
    // explicit align, selects '=' alignment.
    if i < n && __psa_str_byte(spec, i) == 48 {
        zero_v = true;
        i = i + 1;
    };
    // [width]
    let wdone = false;
    while i < n && !wdone {
        let b = __psa_str_byte(spec, i);
        if b >= 48 && b <= 57 {
            width_v = width_v * 10 + (b - 48);
            i = i + 1;
        } else {
            wdone = true;
        };
    }
    // [grouping option]
    if i < n {
        let b = __psa_str_byte(spec, i);
        if b == 44 || b == 95 {
            group_v = b;
            i = i + 1;
        };
    };
    // [.precision]
    if i < n && __psa_str_byte(spec, i) == 46 {
        i = i + 1;
        if i >= n { panic(77085); };
        let b0 = __psa_str_byte(spec, i);
        if b0 < 48 || b0 > 57 { panic(77085); };
        prec_v = 0;
        let pdone = false;
        while i < n && !pdone {
            let b = __psa_str_byte(spec, i);
            if b >= 48 && b <= 57 {
                prec_v = prec_v * 10 + (b - 48);
                i = i + 1;
            } else {
                pdone = true;
            };
        }
    };
    // [type]: a single byte; anything after it is an error.
    if i < n {
        typ_v = __psa_str_byte(spec, i);
        i = i + 1;
    };
    if i != n { panic(77085); };
    return PsaFmt { fill: fill_v, has_fill: has_fill, align: align_v, sign: sign_v, alt: alt_v, zero: zero_v, width: width_v, group: group_v, prec: prec_v, typ: typ_v };
}

fn __psa_fmt_fill(f: PsaFmt) -> PsaVal {
    // The effective fill character: the explicit fill, else
    // '0' when the zero flag is set, else the parsed default
    // (a space).
    if f.has_fill { return f.fill; };
    if f.zero {
        let s = __psa_str_new();
        s = __psa_str_push(s, 48u8);
        return s;
    };
    return f.fill;
}

fn __psa_fmt_is_zero_fill(f: PsaFmt) -> bool {
    if f.has_fill {
        return __psa_str_bytes_len(f.fill) == 1 && __psa_str_byte(f.fill, 0) == 48;
    };
    return f.zero;
}

fn __psa_fmt_repeat(out: PsaVal, fill: PsaVal, count: i64) -> PsaVal {
    let i: i64 = 0;
    while i < count {
        out = __psa_str_append_str(out, fill);
        i = i + 1;
    }
    return out;
}

fn __psa_fmt_pad(body: PsaVal, fill: PsaVal, align: i64, width: i64) -> PsaVal {
    // Whole-body padding; width counts code points. '=' is
    // treated like '>' here: the numeric paths that honour
    // '=' split the sign/prefix off first (pad_split).
    let bl = __psa_str_len(body);
    if bl >= width { return body; };
    let pad = width - bl;
    if align == 60 {
        let out = __psa_str_new();
        out = __psa_str_append_str(out, body);
        return __psa_fmt_repeat(out, fill, pad);
    };
    if align == 94 {
        // Centred: the left side gets pad / 2 (CPython puts
        // the odd extra character on the right).
        let left = pad / 2;
        let out = __psa_str_new();
        out = __psa_fmt_repeat(out, fill, left);
        out = __psa_str_append_str(out, body);
        return __psa_fmt_repeat(out, fill, pad - left);
    };
    let out = __psa_str_new();
    out = __psa_fmt_repeat(out, fill, pad);
    out = __psa_str_append_str(out, body);
    return out;
}

fn __psa_fmt_pad_split(pfx: PsaVal, rest: PsaVal, fill: PsaVal, width: i64) -> PsaVal {
    // '=' alignment: padding goes between the sign/prefix
    // and the digits.
    let total = __psa_str_len(pfx) + __psa_str_len(rest);
    let out = __psa_str_new();
    out = __psa_str_append_str(out, pfx);
    if total < width {
        out = __psa_fmt_repeat(out, fill, width - total);
    };
    out = __psa_str_append_str(out, rest);
    return out;
}

fn __psa_fmt_group(digs: PsaVal, sep: i64, size: i64) -> PsaVal {
    // Insert separator `sep` every `size` digits from the
    // right of an ASCII digit string.
    let n = __psa_str_bytes_len(digs);
    let out = __psa_str_new();
    let i: i64 = 0;
    while i < n {
        if i > 0 && (n - i) % size == 0 {
            out = __psa_str_push(out, sep as u8);
        };
        out = __psa_str_push(out, __psa_str_byte(digs, i) as u8);
        i = i + 1;
    }
    return out;
}

fn __psa_fmt_prepend_zero(digs: PsaVal) -> PsaVal {
    let out = __psa_str_new();
    out = __psa_str_push(out, 48u8);
    out = __psa_str_append_str(out, digs);
    return out;
}

fn __psa_fmt_num_assemble(pfx: PsaVal, intpart: PsaVal, rest: PsaVal, f: PsaFmt, grp: i64, gsize: i64) -> PsaVal {
    // Shared tail of the numeric formatters: pfx is the sign
    // (plus the base prefix for ints, folded in by the
    // caller), intpart the ungrouped integer digits, rest
    // the fraction/exponent/suffix text. grp is the grouping
    // separator byte to apply to intpart (0 = none), gsize
    // the group size (3 for decimal, 4 for binary bases).
    let fill = __psa_fmt_fill(f);
    let align = f.align;
    if align == 0 {
        if f.zero {
            align = 61;
        } else {
            align = 62;
        };
    };
    if align == 61 && grp != 0 && __psa_fmt_is_zero_fill(f) {
        // Zero padding with grouping: the zeros join the
        // digit run and the separators are recomputed, so
        // the result may overshoot the width (CPython:
        // format(1234, '08,') == '0,001,234').
        let ip = intpart;
        let done = false;
        while !done {
            let g = __psa_fmt_group(ip, grp, gsize);
            let total = __psa_str_len(pfx) + __psa_str_len(g) + __psa_str_len(rest);
            if total >= f.width {
                ip = g;
                done = true;
            } else {
                ip = __psa_fmt_prepend_zero(ip);
            };
        }
        let out = __psa_str_new();
        out = __psa_str_append_str(out, pfx);
        out = __psa_str_append_str(out, ip);
        out = __psa_str_append_str(out, rest);
        return out;
    };
    let ip2 = intpart;
    if grp != 0 {
        ip2 = __psa_fmt_group(intpart, grp, gsize);
    };
    if align == 61 {
        let joined = __psa_fmt_cat(ip2, rest);
        return __psa_fmt_pad_split(pfx, joined, fill, f.width);
    };
    let whole = __psa_fmt_cat(__psa_fmt_cat(pfx, ip2), rest);
    return __psa_fmt_pad(whole, fill, align, f.width);
}

fn __psa_fmt_digit_char(d: i64, upper: bool) -> i64 {
    if d < 10 { return 48 + d; };
    if upper { return 55 + d; };
    return 87 + d;
}

fn __psa_fmt_int_digits(x: i64, base: i64, upper: bool) -> PsaVal {
    // Digits of |x| in `base`. Extracted from the signed
    // value itself (x % base takes the sign of x, x / base
    // truncates toward zero), so i64::MIN is exact; a
    // magnitude-first version would overflow on -i64::MIN.
    let rev = __psa_str_new();
    if x == 0 {
        rev = __psa_str_push(rev, 48u8);
        return rev;
    };
    let v = x;
    while v != 0 {
        let d = v % base;
        if d < 0 { d = 0 - d; };
        rev = __psa_str_push(rev, __psa_fmt_digit_char(d, upper) as u8);
        v = v / base;
    }
    return __psa_fmt_rev(rev);
}

fn __psa_fmt_push_cp(out: PsaVal, cp: i64) -> PsaVal {
    // Append the UTF-8 encoding of code point cp.
    if cp < 128 {
        out = __psa_str_push(out, cp as u8);
        return out;
    };
    if cp < 2048 {
        out = __psa_str_push(out, (192 + cp / 64) as u8);
        out = __psa_str_push(out, (128 + cp % 64) as u8);
        return out;
    };
    if cp < 65536 {
        out = __psa_str_push(out, (224 + cp / 4096) as u8);
        out = __psa_str_push(out, (128 + (cp / 64) % 64) as u8);
        out = __psa_str_push(out, (128 + cp % 64) as u8);
        return out;
    };
    out = __psa_str_push(out, (240 + cp / 262144) as u8);
    out = __psa_str_push(out, (128 + (cp / 4096) % 64) as u8);
    out = __psa_str_push(out, (128 + (cp / 64) % 64) as u8);
    out = __psa_str_push(out, (128 + cp % 64) as u8);
    return out;
}

fn __psa_fmt_dec_mul(digs: PsaVal, m: i64) -> PsaVal {
    // Multiply an LSD-first ASCII digit string by a small
    // positive int; the result is LSD-first too.
    let out = __psa_str_new();
    let n = __psa_str_bytes_len(digs);
    let carry: i64 = 0;
    let i: i64 = 0;
    while i < n {
        let cur = (__psa_str_byte(digs, i) - 48) * m + carry;
        out = __psa_str_push(out, (48 + cur % 10) as u8);
        carry = cur / 10;
        i = i + 1;
    }
    while carry > 0 {
        out = __psa_str_push(out, (48 + carry % 10) as u8);
        carry = carry / 10;
    }
    return out;
}

fn __psa_fmt_f64_digits(x: f64) -> (PsaVal, i64) {
    // Exact decimal digits of a finite, non-negative f64:
    // returns (digits, exp10) with x = digits * 10^exp10,
    // digits an MSD-first ASCII string with no leading zero
    // (except the single "0" for zero itself).
    if x == 0.0 {
        let z = __psa_str_new();
        z = __psa_str_push(z, 48u8);
        return (z, 0);
    };
    // Normalise x = m * 2^e with m in [1, 2) by exact binary
    // scaling (scaling by powers of two is exact, subnormal
    // inputs included).
    let m = x;
    let e: i64 = 0;
    while m >= 4294967296.0 {
        m = m / 4294967296.0;
        e = e + 32;
    }
    while m >= 2.0 {
        m = m / 2.0;
        e = e + 1;
    }
    while m < 1.0 {
        if m < 1.0 / 4294967296.0 {
            m = m * 4294967296.0;
            e = e - 32;
        } else {
            m = m * 2.0;
            e = e - 1;
        };
    }
    // Extract the 53-bit mantissa: x = mant * 2^(e-52),
    // using only exact operations (doubling a value in
    // [0, 2) and subtracting 1.0 never round).
    let mant: i64 = 1;
    let frac = m - 1.0;
    let i: i64 = 0;
    while i < 52 {
        frac = frac * 2.0;
        mant = mant * 2;
        if frac >= 1.0 {
            mant = mant + 1;
            frac = frac - 1.0;
        };
        i = i + 1;
    }
    // Decimal digits of mant * 2^(e-52) via a decimal
    // bignum: for e-52 >= 0 multiply by 2 that many times;
    // else multiply by 5 and track the power of ten.
    let e2 = e - 52;
    let digs = __psa_fmt_rev(__psa_str_from_i64(mant));
    let exp10: i64 = 0;
    if e2 >= 0 {
        let k: i64 = 0;
        while k < e2 {
            digs = __psa_fmt_dec_mul(digs, 2);
            k = k + 1;
        }
    };
    if e2 < 0 {
        let k: i64 = 0;
        while k < 0 - e2 {
            digs = __psa_fmt_dec_mul(digs, 5);
            k = k + 1;
        }
        exp10 = e2;
    };
    return (__psa_fmt_rev(digs), exp10);
}

fn __psa_fmt_digit_at(digs: PsaVal, len: i64, exp10: i64, place: i64) -> i64 {
    // The digit at the 10^place position of the value whose
    // MSD-first digits are `digs` (0 outside the digit run).
    let idx = len - 1 - (place - exp10);
    if idx < 0 || idx >= len { return 0; };
    return __psa_str_byte(digs, idx) - 48;
}

fn __psa_fmt_inc(digs: PsaVal) -> PsaVal {
    // Add 1 to an MSD-first ASCII digit string.
    let rev = __psa_fmt_rev(digs);
    let n = __psa_str_bytes_len(rev);
    let out = __psa_str_new();
    let carry: i64 = 1;
    let i: i64 = 0;
    while i < n {
        let cur = (__psa_str_byte(rev, i) - 48) + carry;
        if cur >= 10 {
            out = __psa_str_push(out, 48u8);
            carry = 1;
        } else {
            out = __psa_str_push(out, (48 + cur) as u8);
            carry = 0;
        };
        i = i + 1;
    }
    if carry == 1 {
        out = __psa_str_push(out, 49u8);
    };
    return __psa_fmt_rev(out);
}

fn __psa_fmt_round_sig(digs: PsaVal, exp10: i64, keep: i64) -> (PsaVal, i64) {
    // Round an exact MSD-first digit string to `keep`
    // significant digits, half-even, exactly. Returns the
    // rounded digits (exactly `keep` of them, unless the
    // input was shorter) and the adjusted exp10.
    let len = __psa_str_bytes_len(digs);
    if keep >= len { return (digs, exp10); };
    let rd = __psa_str_byte(digs, keep) - 48;
    let tail = false;
    let i = keep + 1;
    while i < len {
        if __psa_str_byte(digs, i) != 48 { tail = true; };
        i = i + 1;
    }
    let up = false;
    if rd > 5 { up = true; };
    if rd == 5 {
        if tail {
            up = true;
        } else {
            let last = __psa_str_byte(digs, keep - 1) - 48;
            if last % 2 == 1 { up = true; };
        };
    };
    let out = __psa_fmt_sub(digs, 0, keep);
    if !up { return (out, exp10 + (len - keep)); };
    let inc = __psa_fmt_inc(out);
    if __psa_str_bytes_len(inc) > keep {
        // 999.. rounded up: renormalise to a 1 followed by
        // zeros, one power of ten higher.
        let rn = __psa_str_new();
        rn = __psa_str_push(rn, 49u8);
        let z: i64 = 1;
        while z < keep {
            rn = __psa_str_push(rn, 48u8);
            z = z + 1;
        }
        return (rn, exp10 + (len - keep) + 1);
    };
    return (inc, exp10 + (len - keep));
}

fn __psa_fmt_strip(digs: PsaVal, exp10: i64) -> (PsaVal, i64) {
    // Drop trailing zeros from a digit string, adjusting
    // exp10 so the value is unchanged.
    let len = __psa_str_bytes_len(digs);
    let nl = len;
    while nl > 1 && __psa_str_byte(digs, nl - 1) == 48 {
        nl = nl - 1;
    }
    return (__psa_fmt_sub(digs, 0, nl), exp10 + (len - nl));
}

fn __psa_fmt_strip_frac(fp: PsaVal) -> PsaVal {
    // Drop trailing zeros from a fraction digit string; an
    // all-zero fraction becomes empty.
    let nl = __psa_str_bytes_len(fp);
    while nl > 0 && __psa_str_byte(fp, nl - 1) == 48 {
        nl = nl - 1;
    }
    return __psa_fmt_sub(fp, 0, nl);
}

fn __psa_fmt_parts_plain(digs: PsaVal, exp10: i64) -> (PsaVal, PsaVal) {
    // Split exact digits into (integer, fraction) digit
    // strings with no rounding: every digit of the value
    // appears exactly once across the two parts.
    let len = __psa_str_bytes_len(digs);
    let point = len + exp10;
    let ip = __psa_str_new();
    let fp = __psa_str_new();
    if point <= 0 {
        ip = __psa_str_push(ip, 48u8);
        let z: i64 = 0;
        while z < 0 - point {
            fp = __psa_str_push(fp, 48u8);
            z = z + 1;
        }
        fp = __psa_str_append_str(fp, digs);
        return (ip, fp);
    };
    if point >= len {
        ip = __psa_str_append_str(ip, digs);
        let z: i64 = 0;
        while z < point - len {
            ip = __psa_str_push(ip, 48u8);
            z = z + 1;
        }
        return (ip, fp);
    };
    ip = __psa_str_append_str(ip, __psa_fmt_sub(digs, 0, point));
    fp = __psa_str_append_str(fp, __psa_fmt_sub(digs, point, len));
    return (ip, fp);
}

fn __psa_fmt_fixed_parts(digs: PsaVal, exp10: i64, prec: i64) -> (PsaVal, PsaVal) {
    // (integer_digits, fraction_digits) for f-style
    // formatting with exactly `prec` fraction digits,
    // rounded half-even at the cut, exactly.
    let len = __psa_str_bytes_len(digs);
    let cut = 0 - prec - 1;
    let rd = __psa_fmt_digit_at(digs, len, exp10, cut);
    let tail = false;
    let pl = cut - 1;
    while pl >= exp10 {
        if __psa_fmt_digit_at(digs, len, exp10, pl) != 0 { tail = true; };
        pl = pl - 1;
    }
    let up = false;
    if rd > 5 { up = true; };
    if rd == 5 {
        if tail {
            up = true;
        } else {
            let last = __psa_fmt_digit_at(digs, len, exp10, cut + 1);
            if last % 2 == 1 { up = true; };
        };
    };
    let top = len - 1 + exp10;
    if top < 0 { top = 0; };
    let ip = __psa_str_new();
    let p = top;
    while p >= 0 {
        ip = __psa_str_push(ip, (48 + __psa_fmt_digit_at(digs, len, exp10, p)) as u8);
        p = p - 1;
    }
    let fp = __psa_str_new();
    let q: i64 = 1;
    while q <= prec {
        fp = __psa_str_push(fp, (48 + __psa_fmt_digit_at(digs, len, exp10, 0 - q)) as u8);
        q = q + 1;
    }
    if !up { return (ip, fp); };
    let inc = __psa_fmt_inc(__psa_fmt_cat(ip, fp));
    let il = __psa_str_bytes_len(inc) - prec;
    let nip = __psa_fmt_sub(inc, 0, il);
    let nfp = __psa_fmt_sub(inc, il, __psa_str_bytes_len(inc));
    return (nip, nfp);
}

fn __psa_fmt_sci_parts(digs: PsaVal, exp10: i64, prec: i64) -> (PsaVal, i64) {
    // (mantissa digits of length prec+1, decimal exponent
    // of the leading digit) for e-style formatting.
    let (r, e10) = __psa_fmt_round_sig(digs, exp10, prec + 1);
    let exp = __psa_str_bytes_len(r) - 1 + e10;
    let out = __psa_str_new();
    let rl = __psa_str_bytes_len(r);
    let i: i64 = 0;
    while i < prec + 1 {
        if i < rl {
            out = __psa_str_push(out, __psa_str_byte(r, i) as u8);
        } else {
            out = __psa_str_push(out, 48u8);
        };
        i = i + 1;
    }
    return (out, exp);
}

fn __psa_fmt_exp_str(e: i64) -> PsaVal {
    // Exponent text: a sign and at least two digits.
    let out = __psa_str_new();
    let a = e;
    if e < 0 {
        out = __psa_str_push(out, 45u8);
        a = 0 - e;
    } else {
        out = __psa_str_push(out, 43u8);
    };
    let ds = __psa_str_from_i64(a);
    if __psa_str_bytes_len(ds) == 1 {
        out = __psa_str_push(out, 48u8);
    };
    out = __psa_str_append_str(out, ds);
    return out;
}

fn __psa_fmt_shortest(x: f64) -> (PsaVal, i64) {
    // Shortest digits that round back to x (x >= 0, finite):
    // round the exact digits to p significant digits for
    // p = 1..17 and parse each candidate back with the
    // runtime's own decimal parser. Returns (digits, exp10).
    let (digs, exp10) = __psa_fmt_f64_digits(x);
    if x == 0.0 { return (digs, exp10); };
    // Fallback if no candidate parses back exactly (the
    // runtime parser is not correctly rounded for every
    // 17-digit input): the 17-digit rounding, never the raw
    // digit string.
    let (rd, re10) = __psa_fmt_round_sig(digs, exp10, 17);
    let found = false;
    let p: i64 = 1;
    while p <= 17 && !found {
        let (r, e10) = __psa_fmt_round_sig(digs, exp10, p);
        let cand = __psa_str_new();
        cand = __psa_str_push(cand, __psa_str_byte(r, 0) as u8);
        if __psa_str_bytes_len(r) > 1 {
            cand = __psa_str_push(cand, 46u8);
            cand = __psa_str_append_str(cand, __psa_fmt_sub(r, 1, __psa_str_bytes_len(r)));
        };
        cand = __psa_str_push(cand, 101u8);
        let ex = __psa_str_bytes_len(r) - 1 + e10;
        cand = __psa_str_append_str(cand, __psa_fmt_exp_str(ex));
        if __psa_str_to_float(cand) == x {
            rd = r;
            re10 = e10;
            found = true;
        };
        p = p + 1;
    }
    return (rd, re10);
}

fn __psa_fmt_float(x: f64, f: PsaFmt) -> PsaVal {
    let t = f.typ;
    if t == 115 { panic(77085); };
    let is_pct = t == 37;
    let kind = t;
    if kind == 110 { kind = 103; };
    if !(kind == 0 || kind == 101 || kind == 69 || kind == 102 || kind == 70 || kind == 103 || kind == 71 || is_pct) {
        panic(77085);
    };
    let neg = false;
    if x < 0.0 { neg = true; };
    if x == 0.0 && 1.0 / x < 0.0 { neg = true; };
    let ax = x;
    if neg { ax = 0.0 - x; };
    let vx = ax;
    if is_pct { vx = ax * 100.0; };
    let pfx = __psa_str_new();
    if neg { pfx = __psa_str_push(pfx, 45u8); };
    if !neg && f.sign == 43 { pfx = __psa_str_push(pfx, 43u8); };
    if !neg && f.sign == 32 { pfx = __psa_str_push(pfx, 32u8); };
    // inf / nan: no digits, but the sign, the width and
    // padding still apply (and zero padding pads with '0',
    // as CPython does: format(inf, '010f') == '0000000inf').
    if vx != vx || (vx != 0.0 && vx * 2.0 == vx) {
        let word = __psa_str_new();
        let upper = kind == 70 || kind == 71 || kind == 69;
        if vx != vx {
            if upper {
                word = __psa_str_push(word, 78u8);
                word = __psa_str_push(word, 65u8);
                word = __psa_str_push(word, 78u8);
            } else {
                word = __psa_str_push(word, 110u8);
                word = __psa_str_push(word, 97u8);
                word = __psa_str_push(word, 110u8);
            };
        } else {
            if upper {
                word = __psa_str_push(word, 73u8);
                word = __psa_str_push(word, 78u8);
                word = __psa_str_push(word, 70u8);
            } else {
                word = __psa_str_push(word, 105u8);
                word = __psa_str_push(word, 110u8);
                word = __psa_str_push(word, 102u8);
            };
        };
        if is_pct { word = __psa_str_push(word, 37u8); };
        let fill = __psa_fmt_fill(f);
        let al = f.align;
        if al == 0 {
            if f.zero {
                al = 61;
            } else {
                al = 62;
            };
        };
        if al == 61 {
            return __psa_fmt_pad_split(pfx, word, fill, f.width);
        };
        return __psa_fmt_pad(__psa_fmt_cat(pfx, word), fill, al, f.width);
    };
    let (digs, exp10) = __psa_fmt_f64_digits(vx);
    if kind == 102 || kind == 70 || is_pct {
        let prec = f.prec;
        if prec == -1 { prec = 6; };
        let (ip, fp) = __psa_fmt_fixed_parts(digs, exp10, prec);
        let rest = __psa_str_new();
        if prec > 0 {
            rest = __psa_str_push(rest, 46u8);
            rest = __psa_str_append_str(rest, fp);
        };
        if prec == 0 && f.alt { rest = __psa_str_push(rest, 46u8); };
        if is_pct { rest = __psa_str_push(rest, 37u8); };
        return __psa_fmt_num_assemble(pfx, ip, rest, f, f.group, 3);
    };
    if kind == 101 || kind == 69 {
        let prec = f.prec;
        if prec == -1 { prec = 6; };
        let (md, ex) = __psa_fmt_sci_parts(digs, exp10, prec);
        let rest = __psa_str_new();
        rest = __psa_str_push(rest, __psa_str_byte(md, 0) as u8);
        if prec > 0 {
            rest = __psa_str_push(rest, 46u8);
            rest = __psa_str_append_str(rest, __psa_fmt_sub(md, 1, __psa_str_bytes_len(md)));
        };
        if prec == 0 && f.alt { rest = __psa_str_push(rest, 46u8); };
        if kind == 69 {
            rest = __psa_str_push(rest, 69u8);
        } else {
            rest = __psa_str_push(rest, 101u8);
        };
        rest = __psa_str_append_str(rest, __psa_fmt_exp_str(ex));
        // Grouping is accepted but never shows in scientific
        // form (a single digit before the point), as in
        // CPython.
        return __psa_fmt_num_assemble(pfx, __psa_str_new(), rest, f, 0, 3);
    };
    if kind == 103 || kind == 71 {
        // 'g' / 'G' (and 'n', mapped to 'g' above): round to
        // `prec` significant digits; scientific form iff the
        // decimal exponent is < -4 or >= prec.
        let prec = f.prec;
        if prec == -1 { prec = 6; };
        if prec == 0 { prec = 1; };
        let (r, e10) = __psa_fmt_round_sig(digs, exp10, prec);
        let xx = __psa_str_bytes_len(r) - 1 + e10;
        if xx < -4 || xx >= prec {
            let md = r;
            let i = __psa_str_bytes_len(md);
            while i < prec {
                md = __psa_str_push(md, 48u8);
                i = i + 1;
            }
            let frac = __psa_fmt_sub(md, 1, __psa_str_bytes_len(md));
            if !f.alt { frac = __psa_fmt_strip_frac(frac); };
            let rest = __psa_str_new();
            rest = __psa_str_push(rest, __psa_str_byte(md, 0) as u8);
            if __psa_str_bytes_len(frac) > 0 || f.alt {
                rest = __psa_str_push(rest, 46u8);
                rest = __psa_str_append_str(rest, frac);
            };
            if kind == 71 {
                rest = __psa_str_push(rest, 69u8);
            } else {
                rest = __psa_str_push(rest, 101u8);
            };
            rest = __psa_str_append_str(rest, __psa_fmt_exp_str(xx));
            return __psa_fmt_num_assemble(pfx, __psa_str_new(), rest, f, 0, 3);
        };
        let (ip, fp) = __psa_fmt_fixed_parts(digs, exp10, prec - 1 - xx);
        let fpd = fp;
        if !f.alt { fpd = __psa_fmt_strip_frac(fp); };
        let rest = __psa_str_new();
        if __psa_str_bytes_len(fpd) > 0 {
            rest = __psa_str_push(rest, 46u8);
            rest = __psa_str_append_str(rest, fpd);
        };
        if f.alt && __psa_str_bytes_len(fpd) == 0 {
            rest = __psa_str_push(rest, 46u8);
        };
        return __psa_fmt_num_assemble(pfx, ip, rest, f, f.group, 3);
    };
    // No type code (''): with a precision, digits are rounded
    // to max(prec, 1) significant digits and scientific form
    // is chosen iff the exponent is < -4 or >= prec - 1
    // (CPython's '' rule, one lower than 'g'); without a
    // precision, the shortest round-trip digits are used and
    // scientific form starts at exponent 16, as in repr.
    // Either way trailing zeros are stripped (unless '#')
    // and a bare integer result gains a '.0'.
    let pp = f.prec;
    let r = digs;
    let e10 = exp10;
    if pp == -1 {
        let (sr, se) = __psa_fmt_shortest(vx);
        r = sr;
        e10 = se;
    } else {
        if pp == 0 { pp = 1; };
        let (sr, se) = __psa_fmt_round_sig(digs, exp10, pp);
        r = sr;
        e10 = se;
    };
    if !f.alt {
        let (sr, se) = __psa_fmt_strip(r, e10);
        r = sr;
        e10 = se;
    };
    let xx = __psa_str_bytes_len(r) - 1 + e10;
    let sci = false;
    if xx < -4 { sci = true; };
    if pp == -1 && xx >= 16 { sci = true; };
    if pp != -1 && xx >= pp - 1 { sci = true; };
    if sci {
        let rest = __psa_str_new();
        rest = __psa_str_push(rest, __psa_str_byte(r, 0) as u8);
        let frac = __psa_fmt_sub(r, 1, __psa_str_bytes_len(r));
        if __psa_str_bytes_len(frac) > 0 || f.alt {
            rest = __psa_str_push(rest, 46u8);
            rest = __psa_str_append_str(rest, frac);
        };
        rest = __psa_str_push(rest, 101u8);
        rest = __psa_str_append_str(rest, __psa_fmt_exp_str(xx));
        return __psa_fmt_num_assemble(pfx, __psa_str_new(), rest, f, 0, 3);
    };
    let (ip, fp) = __psa_fmt_parts_plain(r, e10);
    let rest = __psa_str_new();
    if __psa_str_bytes_len(fp) > 0 {
        rest = __psa_str_push(rest, 46u8);
        rest = __psa_str_append_str(rest, fp);
    } else {
        if f.alt {
            rest = __psa_str_push(rest, 46u8);
        } else {
            rest = __psa_str_push(rest, 46u8);
            rest = __psa_str_push(rest, 48u8);
        };
    };
    return __psa_fmt_num_assemble(pfx, ip, rest, f, f.group, 3);
}

fn __psa_fmt_float_text(x: f64) -> PsaVal {
    // str(float) / repr(float): the '' format with no spec.
    return __psa_fmt_float(x, __psa_fmt_default());
}

fn __psa_fmt_int_fixed(x: i64, f: PsaFmt) -> PsaVal {
    // 'f' / 'F' on an int: exact fixed-point (the digits are
    // the integer's own; the fraction is all zeros).
    let prec = f.prec;
    if prec == -1 { prec = 6; };
    let pfx = __psa_str_new();
    if x < 0 { pfx = __psa_str_push(pfx, 45u8); };
    if x >= 0 && f.sign == 43 { pfx = __psa_str_push(pfx, 43u8); };
    if x >= 0 && f.sign == 32 { pfx = __psa_str_push(pfx, 32u8); };
    let digs = __psa_fmt_int_digits(x, 10, false);
    let rest = __psa_str_new();
    if prec > 0 {
        rest = __psa_str_push(rest, 46u8);
        let i: i64 = 0;
        while i < prec {
            rest = __psa_str_push(rest, 48u8);
            i = i + 1;
        }
    };
    if prec == 0 && f.alt { rest = __psa_str_push(rest, 46u8); };
    return __psa_fmt_num_assemble(pfx, digs, rest, f, f.group, 3);
}

fn __psa_fmt_int(x: i64, is_bool: bool, f: PsaFmt) -> PsaVal {
    if is_bool && __psa_fmt_defaulted(f) {
        return __psa_str_from_bool(x != 0);
    };
    let t = f.typ;
    if t == 102 || t == 70 {
        return __psa_fmt_int_fixed(x, f);
    };
    if t == 101 || t == 69 || t == 103 || t == 71 || t == 37 {
        // Float presentation types on an int go through the
        // float formatter, as in CPython (exact while
        // |x| < 2^53; 'f'/'F' above are exact at any size).
        return __psa_fmt_float(x as f64, f);
    };
    if t == 115 { panic(77085); };
    if f.prec != -1 { panic(77085); };
    if t == 99 {
        // 'c': the code point itself. Sign, '#' and grouping
        // are errors; '=' alignment pads before the
        // character, like '>'.
        if f.sign != 0 || f.alt || f.group != 0 { panic(77085); };
        if x < 0 || x > 1114111 { panic(77085); };
        let body = __psa_str_new();
        body = __psa_fmt_push_cp(body, x);
        let al = f.align;
        if al == 0 { al = 62; };
        return __psa_fmt_pad(body, __psa_fmt_fill(f), al, f.width);
    };
    if !(t == 0 || t == 100 || t == 110 || t == 98 || t == 111 || t == 120 || t == 88) {
        panic(77085);
    };
    let base: i64 = 10;
    let upper = false;
    if t == 98 { base = 2; };
    if t == 111 { base = 8; };
    if t == 120 { base = 16; };
    if t == 88 {
        base = 16;
        upper = true;
    };
    if f.group == 44 && base != 10 { panic(77085); };
    let neg = x < 0;
    let pfx = __psa_str_new();
    if neg { pfx = __psa_str_push(pfx, 45u8); };
    if !neg && f.sign == 43 { pfx = __psa_str_push(pfx, 43u8); };
    if !neg && f.sign == 32 { pfx = __psa_str_push(pfx, 32u8); };
    if f.alt {
        if base == 2 {
            pfx = __psa_str_push(pfx, 48u8);
            pfx = __psa_str_push(pfx, 98u8);
        };
        if base == 8 {
            pfx = __psa_str_push(pfx, 48u8);
            pfx = __psa_str_push(pfx, 111u8);
        };
        if base == 16 {
            pfx = __psa_str_push(pfx, 48u8);
            if upper {
                pfx = __psa_str_push(pfx, 88u8);
            } else {
                pfx = __psa_str_push(pfx, 120u8);
            };
        };
    };
    let digs = __psa_fmt_int_digits(x, base, upper);
    let gsize: i64 = 3;
    if base != 10 { gsize = 4; };
    return __psa_fmt_num_assemble(pfx, digs, __psa_str_new(), f, f.group, gsize);
}

fn __psa_fmt_str(s: PsaVal, f: PsaFmt) -> PsaVal {
    if f.typ != 0 && f.typ != 115 { panic(77085); };
    if f.sign != 0 { panic(77085); };
    if f.alt { panic(77085); };
    if f.group != 0 { panic(77085); };
    if f.align == 61 { panic(77085); };
    let body = s;
    if f.prec >= 0 {
        // Precision truncates to that many code points.
        let endb = __psa_str_cp_start(s, f.prec);
        if endb >= 0 {
            body = __psa_fmt_sub(s, 0, endb);
        };
    };
    let al = f.align;
    if al == 0 { al = 60; };
    return __psa_fmt_pad(body, __psa_fmt_fill(f), al, f.width);
}

fn __psa_fmt_hex(out: PsaVal, val: i64, width: i64) -> PsaVal {
    // Append exactly `width` lowercase hex digits of val.
    let div: i64 = 1;
    let k: i64 = 1;
    while k < width {
        div = div * 16;
        k = k + 1;
    }
    while div > 0 {
        let d = (val / div) % 16;
        out = __psa_str_push(out, __psa_fmt_digit_char(d, false) as u8);
        div = div / 16;
    }
    return out;
}

fn __psa_fmt_ascii_escape(s: PsaVal) -> PsaVal {
    // ascii(): the repr text with every non-ASCII code
    // point escaped as \xNN, \uNNNN or \UNNNNNNNN.
    let out = __psa_str_new();
    let n = __psa_str_bytes_len(s);
    let i: i64 = 0;
    while i < n {
        let b = __psa_str_byte(s, i);
        if b < 128 {
            out = __psa_str_push(out, b as u8);
            i = i + 1;
        } else {
            let len = __psa_fmt_cp_len(b);
            let cp = b % 64;
            if len == 3 { cp = b % 16; };
            if len == 4 { cp = b % 8; };
            let j: i64 = 1;
            while j < len {
                cp = cp * 64 + (__psa_str_byte(s, i + j) % 64);
                j = j + 1;
            }
            out = __psa_str_push(out, 92u8);
            if cp < 256 {
                out = __psa_str_push(out, 120u8);
                out = __psa_fmt_hex(out, cp, 2);
            } else if cp < 65536 {
                out = __psa_str_push(out, 117u8);
                out = __psa_fmt_hex(out, cp, 4);
            } else {
                out = __psa_str_push(out, 85u8);
                out = __psa_fmt_hex(out, cp, 8);
            };
            i = i + len;
        };
    }
    return out;
}

fn __psa_fmt_conv(v: PsaVal, conv: i64) -> PsaVal {
    // The conversion step of a formatted field: !s -> str,
    // !r -> repr, !a -> repr then ASCII-escape (the escape
    // itself happens in __psa_format_value). Floats use the
    // shortest-round-trip text for both str and repr
    // (CPython: str(float) == repr(float) since 3.1).
    if v.tag == 1 {
        return __psa_fmt_float_text(v.fpay);
    };
    if conv == 2 {
        return __psa_str_of(v);
    };
    return __psa_repr_of(v);
}

fn __psa_format_spec(v: PsaVal, spec: PsaVal) -> PsaVal {
    if v.tag != 6 && v.tag != 0 && v.tag != 2 && v.tag != 1 {
        // None, containers, bytes, objects and the rest:
        // CPython raises TypeError for ANY non-empty spec on
        // these (object.__format__), before the spec text is
        // even parsed. An empty spec is plain str().
        if __psa_str_bytes_len(spec) > 0 { panic(77086); };
        return __psa_str_of(v);
    };
    let f = __psa_fmt_parse(spec);
    if v.tag == 6 { return __psa_fmt_str(v, f); };
    if v.tag == 0 { return __psa_fmt_int(v.ipay, false, f); };
    if v.tag == 2 { return __psa_fmt_int(v.ipay, true, f); };
    return __psa_fmt_float(v.fpay, f);
}

fn __psa_format_value(v: PsaVal, conv: i64, spec: PsaVal) -> PsaVal {
    // Entry point called by psa's emit for every f-string
    // field that carries a conversion, a debug marker or a
    // format spec. With a conversion, the converted text is
    // formatted under string rules (CPython).
    if conv != 0 {
        let s = __psa_fmt_conv(v, conv);
        if conv == 3 {
            s = __psa_fmt_ascii_escape(s);
        };
        return __psa_fmt_str(s, __psa_fmt_parse(spec));
    };
    return __psa_format_spec(v, spec);
}

"#;

/// D6 runtime, part 2 (emitted only when a file operation is
/// in play; sa_std/fs.sai is imported then): open/read/write/
/// close over the fs externs. IO failures panic 77110;
/// operations on a closed file or in the wrong mode panic
/// 77111.
const D6_FILE_RUNTIME: &str = r#"
fn __psa_file_new(handle: i64, mode: i64, path: PsaVal) -> PsaVal {
    let xs: Vec<PsaVal> = Vec::new();
    let p = Box::into_raw(Box::new(xs)) as i64;
    __psa_vec_push_h(p as ptr, __psa_box_int(handle));
    __psa_vec_push_h(p as ptr, __psa_box_int(mode));
    __psa_vec_push_h(p as ptr, __psa_box_bool(false));
    __psa_vec_push_h(p as ptr, path);
    return PsaVal { tag: 12, ipay: p, fpay: 0.0 };
}

fn __psa_file_set_closed(v: PsaVal) {
    __psa_vec_set_h(v.ipay as ptr, 2, __psa_box_bool(true));
}

fn __psa_open(path: PsaVal, mode: i64) -> PsaVal {
    if path.tag != 6 { panic(77110); };
    let n = __psa_str_bytes_len(path);
    if n > 1000 { panic(77110); };
    let buf = [0u8; 1000];
    let i: i64 = 0;
    while i < n { buf[i] = __psa_bytes_get_h(path.ipay as ptr, i); i = i + 1; }
    let h: u64 = 0;
    if mode == 1 {
        let stw: i32 = sa_std_fs_open_write(PTR_BYTE_ADD(&buf, 0), n as u64, 1, &h);
        if stw != 0 { panic(77110); };
    };
    if mode == 2 {
        let sta: i32 = sa_std_fs_open_write(PTR_BYTE_ADD(&buf, 0), n as u64, 0, &h);
        if sta != 0 { panic(77110); };
        let pos: u64 = 0;
        let stseek: i32 = sa_std_fs_file_seek(h, 2, 0, &pos);
        if stseek != 0 { panic(77110); };
    };
    if mode == 0 {
        let strd: i32 = sa_std_fs_open_read(PTR_BYTE_ADD(&buf, 0), n as u64, &h);
        if strd != 0 { panic(77110); };
    };
    return __psa_file_new(h as i64, mode, path);
}

fn __psa_file_read_all(v: PsaVal) -> PsaVal {
    let out = __psa_str_new();
    let buf = [0u8; 4096];
    let nr: u64 = 4096;
    while nr == 4096 {
        let st: i32 = sa_std_fs_file_read(__psa_file_handle(v) as u64, PTR_BYTE_ADD(&buf, 0), 4096, &nr);
        if st != 0 { panic(77110); };
        let k: i64 = 0;
        while k < (nr as i64) { out = __psa_str_push(out, buf[k]); k = k + 1; }
    }
    return out;
}

// read(n) counts code points, as text mode does in CPython.
fn __psa_file_read_n(v: PsaVal, n: i64) -> PsaVal {
    let out = __psa_str_new();
    let buf = [0u8; 1];
    let cps: i64 = 0;
    let eof: bool = false;
    while cps < n && !(eof) {
        let nr: u64 = 0;
        let st: i32 = sa_std_fs_file_read(__psa_file_handle(v) as u64, PTR_BYTE_ADD(&buf, 0), 1, &nr);
        if st != 0 { panic(77110); };
        if nr == 0 { eof = true; } else {
            let b = buf[0];
            out = __psa_str_push(out, b);
            if ((b as i64) & 192) != 128 { cps = cps + 1; };
        };
    }
    return out;
}

fn __psa_file_readline(v: PsaVal) -> PsaVal {
    let out = __psa_str_new();
    let buf = [0u8; 1];
    let stop: bool = false;
    while !(stop) {
        let nr: u64 = 0;
        let st: i32 = sa_std_fs_file_read(__psa_file_handle(v) as u64, PTR_BYTE_ADD(&buf, 0), 1, &nr);
        if st != 0 { panic(77110); };
        if nr == 0 { stop = true; } else {
            let b = buf[0];
            out = __psa_str_push(out, b);
            if (b as i64) == 10 { stop = true; };
        };
    }
    return out;
}

fn __psa_file_write(v: PsaVal, s: PsaVal) -> i64 {
    let n = __psa_str_bytes_len(s);
    let buf = [0u8; 4096];
    let base: i64 = 0;
    while base < n {
        let chunk = n - base;
        if chunk > 4096 { chunk = 4096; };
        let k: i64 = 0;
        while k < chunk { buf[k] = __psa_bytes_get_h(s.ipay as ptr, base + k); k = k + 1; }
        let nw: u64 = 0;
        let st: i32 = sa_std_fs_file_write(__psa_file_handle(v) as u64, PTR_BYTE_ADD(&buf, 0), chunk as u64, &nw);
        if st != 0 { panic(77110); };
        if nw != (chunk as u64) { panic(77110); };
        base = base + chunk;
    }
    return __psa_str_len(s);
}

fn __psa_m_read(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 12 { panic(77069); };
    if __psa_file_closed(v) { panic(77111); };
    if __psa_file_mode(v) != 0 { panic(77111); };
    __psa_m_argc(args, 0, 1);
    if __psa_list_len(args) == 0 { return __psa_file_read_all(v); };
    let n = __psa_m_arg(args, 0);
    if n.tag != 0 { panic(77075); };
    if n.ipay < 0 { return __psa_file_read_all(v); };
    return __psa_file_read_n(v, n.ipay);
}

fn __psa_m_readline(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 12 { panic(77069); };
    if __psa_file_closed(v) { panic(77111); };
    if __psa_file_mode(v) != 0 { panic(77111); };
    __psa_m_argc(args, 0, 0);
    return __psa_file_readline(v);
}

fn __psa_m_write(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 12 { panic(77069); };
    if __psa_file_closed(v) { panic(77111); };
    if __psa_file_mode(v) == 0 { panic(77111); };
    __psa_m_argc(args, 1, 1);
    let s = __psa_m_arg(args, 0);
    if s.tag != 6 { panic(77075); };
    return __psa_box_int(__psa_file_write(v, s));
}

fn __psa_m_close(v: PsaVal, name: PsaVal, args: PsaVal) -> PsaVal {
    if v.tag == 8 { return __psa_call_value(__psa_getattr(v, name), args); };
    if v.tag != 12 { panic(77069); };
    __psa_m_argc(args, 0, 0);
    if !(__psa_file_closed(v)) {
        let st: i32 = sa_std_close(__psa_file_handle(v) as u64);
        if st != 0 { panic(77110); };
        __psa_file_set_closed(v);
    };
    return __psa_m_none();
}

// with-statement exit (D6): close the value if it is a file;
// other values have no context protocol in v1 and pass.
fn __psa_with_close(v: PsaVal) {
    if v.tag == 12 {
        if !(__psa_file_closed(v)) {
            let st: i32 = sa_std_close(__psa_file_handle(v) as u64);
            if st != 0 { panic(77110); };
            __psa_file_set_closed(v);
        };
    };
}

"#;

/// D7 generator runtime (emitted only when the program defines
/// or consumes a generator): the producer body runs on a
/// spawned thread and hands each yielded value over a
/// rendezvous channel as a 2-tuple wrap (item, end-flag), so no
/// user value can forge the end marker. The state box holds the
/// channel, the join handle and the exhausted flag.
const D7_GEN_RUNTIME: &str = r#"
struct PsaGenState { ch: Chan<i64>, h: JoinHandle<PsaVal>, done: i64, started: i64 }

fn __psa_gen_new(ch: Chan<i64>, h: JoinHandle<PsaVal>) -> PsaVal {
    let st = PsaGenState { ch: ch, h: h, done: 0, started: 0 };
    let p = Box::into_raw(Box::new(st)) as i64;
    return PsaVal { tag: 13, ipay: p, fpay: 0.0 };
}

fn __psa_gen_wrap(x: PsaVal, end: i64) -> PsaVal {
    let t = __psa_tuple_new();
    __psa_tuple_push(t, x);
    __psa_tuple_push(t, __psa_box_bool(end != 0));
    return t;
}

fn __psa_gen_wrap_end() -> PsaVal {
    return __psa_gen_wrap(PsaVal { tag: 3, ipay: 0, fpay: 0.0 }, 1);
}

// Pull one wrap off the generator: an end wrap when the
// producer has finished (or the channel closed under it).
// The first pull sends the start token that releases the
// drive thread into the producer body (call-time laziness).
fn __psa_gen_pull(g: PsaVal) -> PsaVal {
    let b = unsafe { Box::from_raw(g.ipay as *PsaGenState) };
    if (*b).done != 0 {
        let rp: i64 = Box::into_raw(b) as i64;
        return __psa_gen_wrap_end();
    };
    if (*b).started == 0 {
        (*b).started = 1;
        // Start token: the drive waits for any message
        // before entering the producer body; the value is
        // ignored, so a plain 0 travels (channel slots hold
        // at most 16 bytes and a PsaVal occupies 24 -- real
        // wraps cross as their tuple heap pointer, below).
        (*b).ch.send(0);
    };
    let (wp, closed) = (*b).ch.recv();
    if closed {
        (*b).done = 1;
        let rp: i64 = Box::into_raw(b) as i64;
        return __psa_gen_wrap_end();
    };
    let w = PsaVal { tag: 10, ipay: wp, fpay: 0.0 };
    if __psa_tuple_get(w, 1).ipay != 0 {
        (*b).done = 1;
        (*b).ch.destroy();
        let jr = (*b).h.join();
        let rp: i64 = Box::into_raw(b) as i64;
        return __psa_gen_wrap_end();
    };
    let rp: i64 = Box::into_raw(b) as i64;
    return w;
}

fn __psa_gen_next(g: PsaVal) -> PsaVal {
    if g.tag != 13 { panic(77052); };
    let w = __psa_gen_pull(g);
    if __psa_tuple_get(w, 1).ipay != 0 { panic(77081); };
    return __psa_tuple_get(w, 0);
}

// Iteration over a dynamic value (D7): a generator is drained
// into a list (the for/comprehension paths iterate lists
// natively); every other value passes through unchanged.
fn __psa_iter_source(v: PsaVal) -> PsaVal {
    if v.tag != 13 { return v; };
    let out = __psa_list_new();
    let done: bool = false;
    while !(done) {
        let w = __psa_gen_pull(v);
        if __psa_tuple_get(w, 1).ipay != 0 {
            done = true;
        } else {
            __psa_list_push(out, __psa_tuple_get(w, 0));
        };
    }
    return out;
}

"#;

/// D7 coroutine runtime (emitted only when the program defines
/// an async def): a coroutine value (tag 14) packs the worker
/// callable id and the argument list; `await` runs the worker
/// to completion at the await point (sequential semantics).
const D7_CORO_RUNTIME: &str = r#"
fn __psa_coro_new(id: i64, args: PsaVal) -> PsaVal {
    let xs: Vec<PsaVal> = Vec::new();
    let p = Box::into_raw(Box::new(xs)) as i64;
    __psa_vec_push_h(p as ptr, __psa_box_int(id));
    __psa_vec_push_h(p as ptr, args);
    return PsaVal { tag: 14, ipay: p, fpay: 0.0 };
}

fn __psa_coro_run(v: PsaVal) -> PsaVal {
    if v.tag != 14 { panic(77082); };
    let id = __psa_vec_get_h(v.ipay as ptr, 0).ipay;
    let args = __psa_vec_get_h(v.ipay as ptr, 1);
    return __psa_call_value(__psa_box_fn(id), args);
}

"#;

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
