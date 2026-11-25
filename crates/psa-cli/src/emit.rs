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
    /// This class's id plus every ancestor's id (D5, backing
    /// isinstance: an instance of a subclass carries all of
    /// them in its ancestry).
    ancestors: Vec<i64>,
}

pub struct Emitter {
    defs: HashMap<String, Func>,
    sigs: HashMap<String, Sig>,
    in_progress: Vec<String>,
    emitted: Vec<String>,
    emitted_set: Vec<String>,
    /// D3 function values: source name -> callable id, plus the
    /// id-ordered callable names. The dispatch (`__psa_call_value`
    /// and its per-function adapters) is generated at assembly
    /// time from `callables`.
    fn_ids: HashMap<String, i64>,
    callables: Vec<String>,
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
    /// D6: the program uses try/raise/with anywhere (pre-scan
    /// in emit_program). Gates the trailing pending-exception
    /// check after call-containing statements (emit_stmt),
    /// which is what unwinds a callee's exception through
    /// frames that have no try of their own.
    program_exc: bool,
}

struct Ctx {
    vars: Vec<(String, Ty, Option<u32>)>,
    scopes: Vec<usize>,
    ret: Option<Ty>,
    fname: String,
    out: String,
    /// Function-prologue lines (hoisted declarations). On the Rust
    /// sla build-exe route a block-scoped aggregate local is never
    /// retired (hand-written probes q16/q29: MemoryLeak at function
    /// exit), so a dynamic variable or temp first materialised inside
    /// a block has its `let` hoisted here, to function scope, and is
    /// filled in place via the stage pattern (see emit_dyn_store).
    prologue: String,
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
    fn check_not_deleted(&self, name: &str, line: u32) -> Result<(), String> {
        if let Some((_, at)) = self.deleted.iter().rev().find(|(n, _)| n == name) {
            return Err(eerr(
                line,
                &format!(
                    "use of `{}` after `del` at line {} (Python raises NameError; re-bind the name first)",
                    name, at
                ),
            ));
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
                "psa emit error: variable `{}` assigned twice in the same scope (v0.1 treats re-binding as an error; rename or restructure)",
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
        self.prologue.push_str(&format!(
            "    let {}: {} = {};\n",
            name,
            ty.sla(),
            ty.default_lit()
        ));
    }
}

fn eerr(line: u32, msg: &str) -> String {
    format!("psa emit error at line {}: {}", line, msg)
}

pub fn emit_program(prog: &Program) -> Result<String, String> {
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
        classes: Vec::new(),
        class_ids: HashMap::new(),
        sla_names: HashMap::new(),
        mod_attrs: prog.mod_attrs.iter().cloned().collect(),
        const_names: prog.const_names.iter().cloned().collect(),
        modules: prog.modules.clone(),
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
        program_exc: false,
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
            line: 0,
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
                line: 0,
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
        scopes: vec![0],
        ret: Some(Ty::Void),
        fname: "main".to_string(),
        out: String::new(),
        prologue: String::new(),
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
    };
    for s in &prog.top {
        em.emit_stmt(&mut main_ctx, s)?;
    }
    main_ctx.line("return 0;");
    let mut main_txt = String::from("fn main() -> i32 {\n");
    main_txt.push_str(&main_ctx.prologue);
    main_txt.push_str(&main_ctx.out);
    main_txt.push_str("}\n");

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
    if !em.callables.is_empty() || !em.classes.is_empty() {
        em.used_dyn = true;
    }

    // D3 function values + D4 bound methods: generate the call
    // dispatch and one adapter per callable function, from the
    // registry built during emission. The dispatch splits into
    // __psa_call_value (tag check: 7 = plain callable, 9 = bound
    // method) and __psa_call_id (per-id adapter dispatch), so a
    // bound method can prepend its receiver and reuse it.
    if !em.callables.is_empty() || !em.classes.is_empty() {
        let mut disp = String::from(
            "fn __psa_call_value(f: PsaVal, args: PsaVal) -> PsaVal {\n    if f.tag == 9 { return __psa_call_bound(f, args); };\n    if f.tag != 7 { panic(77067); };\n    return __psa_call_id(f.ipay, args);\n}\n",
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
        em.emitted.push(disp);
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
            em.emitted.push(ad);
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
        em.emitted.push(machinery);
    }

    let mut out = String::new();
    out.push_str("@import \"sa_std/io/print.sai\"\n@import \"sa_std/fmt.sai\"\n");
    if em.used_vec || em.used_dyn {
        out.push_str("@import \"sa_std/vec.sa\"\n");
    }
    if em.used_dyn {
        out.push_str("@import \"sa_std/globals.sai\"\n");
        out.push_str("@import \"sa_std/core/box.sa\"\n");
        out.push_str("@import \"sa_std/hashmap.sa\"\n");
        out.push_str("@import \"sa_std/core/option.sa\"\n");
        out.push_str("@import \"sa_std/string.sa\"\n");
        out.push_str("@import \"sa_std/ptr.sa\"\n");
        out.push_str("@import \"sla_std/wrap/io.sla\"\n");
        out.push_str("@import \"sla_std/wrap/fmt.sla\"\n");
    }
    if em.used_math || em.used_dyn {
        out.push_str("@import \"sa_std/math.sai\"\n");
    }
    if em.used_file {
        out.push_str("@import \"sa_std/fs.sai\"\n");
    }
    if em.used_gen {
        out.push_str("@import \"sa_std/task.sai\"\n");
        out.push_str("@import \"sa_std/thread.sa\"\n");
        out.push_str("@import \"sa_std/core/result.sa\"\n");
    }
    out.push('\n');
    if em.used_dyn {
        out.push_str("@derive(copy)\nstruct PsaVal { tag: i64, ipay: i64, fpay: f64 }\n\nstruct PsaStage { tag: i64, ipay: i64, fpay: f64 }\n\n");
    }
    if em.used_ipow || em.used_dyn {
        out.push_str(
            "fn __psa_ipow(a: i64, e: i64) -> i64 {\n    let r = 1;\n    let b = a;\n    let n = e;\n    while n > 0 {\n        if (n & 1) == 1 { r = r * b; };\n        b = b * b;\n        n = n >> 1;\n    }\n    return r;\n}\n\n",
        );
    }
    if em.used_floordiv || em.used_dyn {
        out.push_str(
            "fn __psa_floordiv(a: i64, b: i64) -> i64 {\n    if a == -9223372036854775807 - 1 && b == -1 { panic(77080); };\n    let q = a / b;\n    let r = a % b;\n    if r != 0 && ((r < 0) != (b < 0)) { return q - 1; };\n    return q;\n}\n\n",
        );
    }
    if em.used_floormod || em.used_dyn {
        out.push_str(
            "fn __psa_floormod(a: i64, b: i64) -> i64 {\n    if a == -9223372036854775807 - 1 && b == -1 { return 0; };\n    let r = a % b;\n    if r != 0 && ((r < 0) != (b < 0)) { return r + b; };\n    return r;\n}\n\n",
        );
    }
    if em.used_dyn {
        out.push_str(DYN_RUNTIME);
        out.push_str(D6_EXC_RUNTIME);
    }
    if em.used_file {
        out.push_str(D6_FILE_RUNTIME);
    }
    if em.used_gen {
        out.push_str(D7_GEN_RUNTIME);
    }
    if em.used_coro {
        out.push_str(D7_CORO_RUNTIME);
    }
    for f in &em.emitted {
        out.push_str(f);
        out.push('\n');
    }
    out.push_str(&main_txt);
    Ok(out)
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
/// (literals only in v1) evaluate at the read; anything else is
/// an AttributeError (panic 77069).
fn gen_class_getattr(em: &Emitter, ci: &ClassInfo) -> String {
    let mut t = format!(
        "fn __psa_class_{}_getattr(inst: PsaVal, name: PsaVal) -> PsaVal {{\n",
        ci.id
    );
    for (i, (mname, _)) in ci.flat_methods.iter().enumerate() {
        t.push_str(&sla_str_build(&format!("nm_{}", i), mname));
    }
    for (i, (aname, ae)) in ci.flat_attrs.iter().enumerate() {
        t.push_str(&sla_str_build(&format!("an_{}", i), aname));
        if let Expr::Str(s) = ae {
            t.push_str(&sla_str_build(&format!("av_{}", i), s));
        }
    }
    for (i, (_, key)) in ci.flat_methods.iter().enumerate() {
        let id = em.fn_ids.get(key).copied().unwrap_or(-1);
        t.push_str(&format!(
            "    if __psa_str_eq(name, nm_{}) {{ return __psa_box_method({}, inst); }};\n",
            i, id
        ));
    }
    for (i, (_, ae)) in ci.flat_attrs.iter().enumerate() {
        let rc = match ae {
            Expr::Int(v) => format!("__psa_box_int({})", v),
            Expr::Float(v) => format!("__psa_box_float({})", fmt_f64(*v)),
            Expr::Bool(b) => format!("__psa_box_bool({})", b),
            Expr::Str(_) => format!("av_{}", i),
            _ => Ty::Dyn.default_lit().to_string(),
        };
        t.push_str(&format!(
            "    if __psa_str_eq(name, an_{}) {{ return {}; }};\n",
            i, rc
        ));
    }
    t.push_str("    panic(77069);\n}\n");
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
const LAMBDA_BUILTINS: [&str; 15] = [
    "print", "range", "len", "int", "float", "str", "bool", "list", "tuple", "set", "type",
    "isinstance", "open", "next", "repr",
];

/// Collect every name an expression mentions: variable names,
/// call callee names and method-call object names.
fn collect_names(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Name(n) => out.push(n.clone()),
        Expr::Bin(a, _, b) => {
            collect_names(a, out);
            collect_names(b, out);
        }
        Expr::Unary(_, a) => collect_names(a, out),
        Expr::Call(f, args, kwargs) => {
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
        Expr::Index(a, b) => {
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
        Expr::Attr(base, _) => {
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
        Expr::FStr(parts) => {
            for p in parts {
                if let FStrPart::Expr(x) = p {
                    collect_names(x, out);
                }
            }
        }
        _ => {}
    }
}

fn desugar_expr(
    e: &mut Expr,
    defnames: &std::collections::HashSet<String>,
    counter: &mut u32,
    new_funcs: &mut Vec<Func>,
) -> Result<(), String> {
    if matches!(e, Expr::Lambda { .. }) {
        // Rewrite the body first: nested lambdas desugar before
        // this one's capture analysis runs.
        if let Expr::Lambda { body, .. } = e {
            desugar_expr(body, defnames, counter, new_funcs)?;
        }
        let old = std::mem::replace(e, Expr::NoneLit);
        if let Expr::Lambda { params, body, line } = old {
            let mut free: Vec<String> = Vec::new();
            collect_names(&body, &mut free);
            for n in &free {
                if params.iter().any(|p| p == n) {
                    continue;
                }
                if defnames.contains(n) || LAMBDA_BUILTINS.contains(&n.as_str()) {
                    continue;
                }
                return Err(eerr(
                    line,
                    &format!(
                        "lambda captures `{}`: closures over local variables are not supported yet (captured variables need heap cells with late binding); only function and builtin names may be referenced",
                        n
                    ),
                ));
            }
            let name = format!("__psa_lambda_{}", counter);
            *counter += 1;
            new_funcs.push(Func {
                name: name.clone(),
                params: params.iter().map(|p| (p.clone(), None)).collect(),
                vararg: None,
                kwarg: None,
                is_async: false,
                body: vec![Stmt::Return { value: Some(*body), line }],
                line,
            });
            *e = Expr::FnVal(name);
        }
        return Ok(());
    }
    match e {
        Expr::Bin(a, _, b) => {
            desugar_expr(a, defnames, counter, new_funcs)?;
            desugar_expr(b, defnames, counter, new_funcs)?;
        }
        Expr::Unary(_, a) => desugar_expr(a, defnames, counter, new_funcs)?,
        Expr::Call(f, args, kwargs) => {
            desugar_expr(f, defnames, counter, new_funcs)?;
            for a in args {
                desugar_expr(a, defnames, counter, new_funcs)?;
            }
            for (_, v) in kwargs {
                desugar_expr(v, defnames, counter, new_funcs)?;
            }
        }
        Expr::ListLit(elems) => {
            for a in elems {
                desugar_expr(a, defnames, counter, new_funcs)?;
            }
        }
        Expr::DictLit(pairs) => {
            for (k, v) in pairs {
                desugar_expr(k, defnames, counter, new_funcs)?;
                desugar_expr(v, defnames, counter, new_funcs)?;
            }
        }
        Expr::Index(a, b) => {
            desugar_expr(a, defnames, counter, new_funcs)?;
            desugar_expr(b, defnames, counter, new_funcs)?;
        }
        Expr::Slice(a, lo, hi, st) => {
            desugar_expr(a, defnames, counter, new_funcs)?;
            for o in [lo, hi, st] {
                if let Some(x) = o {
                    desugar_expr(x, defnames, counter, new_funcs)?;
                }
            }
        }
        Expr::MethodCall { args, .. } => {
            for a in args {
                desugar_expr(a, defnames, counter, new_funcs)?;
            }
        }
        Expr::TupleLit(elems) | Expr::SetLit(elems) => {
            for a in elems {
                desugar_expr(a, defnames, counter, new_funcs)?;
            }
        }
        Expr::Comp { key, value, clauses, .. } => {
            desugar_expr(key, defnames, counter, new_funcs)?;
            if let Some(v) = value {
                desugar_expr(v, defnames, counter, new_funcs)?;
            }
            for c in clauses {
                match c {
                    CompClause::For { iter, .. } => {
                        desugar_expr(iter, defnames, counter, new_funcs)?
                    }
                    CompClause::If(cond) => desugar_expr(cond, defnames, counter, new_funcs)?,
                }
            }
        }
        Expr::Attr(base, _) => {
            desugar_expr(base, defnames, counter, new_funcs)?;
        }
        Expr::IfExp { cond, then, els } => {
            desugar_expr(cond, defnames, counter, new_funcs)?;
            desugar_expr(then, defnames, counter, new_funcs)?;
            desugar_expr(els, defnames, counter, new_funcs)?;
        }
        Expr::FStr(parts) => {
            for p in parts {
                if let FStrPart::Expr(x) = p {
                    desugar_expr(x, defnames, counter, new_funcs)?;
                }
            }
        }
        Expr::Await(a) => desugar_expr(a, defnames, counter, new_funcs)?,
        _ => {}
    }
    Ok(())
}

fn desugar_stmt(
    s: &mut Stmt,
    defnames: &std::collections::HashSet<String>,
    counter: &mut u32,
    new_funcs: &mut Vec<Func>,
) -> Result<(), String> {
    match s {
        Stmt::Assign { value, .. }
        | Stmt::AugAssign { value, .. }
        | Stmt::ExprStmt { value, .. } => desugar_expr(value, defnames, counter, new_funcs),
        Stmt::IndexAssign { index, value, .. } => {
            desugar_expr(index, defnames, counter, new_funcs)?;
            desugar_expr(value, defnames, counter, new_funcs)
        }
        Stmt::AttrAssign { base, value, .. } => {
            desugar_expr(base, defnames, counter, new_funcs)?;
            desugar_expr(value, defnames, counter, new_funcs)
        }
        Stmt::AttrDel { base, .. } => desugar_expr(base, defnames, counter, new_funcs),
        Stmt::If { arms, els, .. } => {
            for (c, b) in arms {
                desugar_expr(c, defnames, counter, new_funcs)?;
                for x in b {
                    desugar_stmt(x, defnames, counter, new_funcs)?;
                }
            }
            if let Some(b) = els {
                for x in b {
                    desugar_stmt(x, defnames, counter, new_funcs)?;
                }
            }
            Ok(())
        }
        Stmt::While { cond, body, .. } => {
            desugar_expr(cond, defnames, counter, new_funcs)?;
            for x in body {
                desugar_stmt(x, defnames, counter, new_funcs)?;
            }
            Ok(())
        }
        Stmt::ForRange { start, stop, body, .. } => {
            if let Some(x) = start {
                desugar_expr(x, defnames, counter, new_funcs)?;
            }
            desugar_expr(stop, defnames, counter, new_funcs)?;
            for x in body {
                desugar_stmt(x, defnames, counter, new_funcs)?;
            }
            Ok(())
        }
        Stmt::ForEach { iter, body, .. } => {
            desugar_expr(iter, defnames, counter, new_funcs)?;
            for x in body {
                desugar_stmt(x, defnames, counter, new_funcs)?;
            }
            Ok(())
        }
        Stmt::Return { value, .. } => {
            if let Some(x) = value {
                desugar_expr(x, defnames, counter, new_funcs)?;
            }
            Ok(())
        }
        Stmt::Assert { cond, .. } => desugar_expr(cond, defnames, counter, new_funcs),
        Stmt::Del { targets, .. } => {
            for (_, idx) in targets {
                if let Some(x) = idx {
                    desugar_expr(x, defnames, counter, new_funcs)?;
                }
            }
            Ok(())
        }
        Stmt::MultiAssign { values, .. } => {
            for x in values {
                desugar_expr(x, defnames, counter, new_funcs)?;
            }
            Ok(())
        }
        Stmt::Unpack { value, .. } => desugar_expr(value, defnames, counter, new_funcs),
        Stmt::Try { body, handlers, els, finally, .. } => {
            for x in body {
                desugar_stmt(x, defnames, counter, new_funcs)?;
            }
            for h in handlers {
                for x in &mut h.body {
                    desugar_stmt(x, defnames, counter, new_funcs)?;
                }
            }
            if let Some(b) = els {
                for x in b {
                    desugar_stmt(x, defnames, counter, new_funcs)?;
                }
            }
            if let Some(b) = finally {
                for x in b {
                    desugar_stmt(x, defnames, counter, new_funcs)?;
                }
            }
            Ok(())
        }
        Stmt::Raise { value, .. } => {
            if let Some(x) = value {
                desugar_expr(x, defnames, counter, new_funcs)?;
            }
            Ok(())
        }
        Stmt::With { items, body, .. } => {
            for (e, _) in items {
                desugar_expr(e, defnames, counter, new_funcs)?;
            }
            for x in body {
                desugar_stmt(x, defnames, counter, new_funcs)?;
            }
            Ok(())
        }
        Stmt::Yield { value, .. } => {
            if let Some(x) = value {
                desugar_expr(x, defnames, counter, new_funcs)?;
            }
            Ok(())
        }
        Stmt::Match { subject, cases, .. } => {
            desugar_expr(subject, defnames, counter, new_funcs)?;
            for c in cases {
                desugar_pattern(&mut c.pattern, defnames, counter, new_funcs)?;
                if let Some(g) = &mut c.guard {
                    desugar_expr(g, defnames, counter, new_funcs)?;
                }
                for x in &mut c.body {
                    desugar_stmt(x, defnames, counter, new_funcs)?;
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// D7: desugar lambdas inside match patterns (literal
/// patterns carry plain expressions).
fn desugar_pattern(
    p: &mut Pattern,
    defnames: &std::collections::HashSet<String>,
    counter: &mut u32,
    new_funcs: &mut Vec<Func>,
) -> Result<(), String> {
    match p {
        Pattern::Lit(e) => desugar_expr(e, defnames, counter, new_funcs),
        Pattern::Seq(pats) | Pattern::Or(pats) => {
            for x in pats {
                desugar_pattern(x, defnames, counter, new_funcs)?;
            }
            Ok(())
        }
        Pattern::Capture(_) | Pattern::Wildcard => Ok(()),
    }
}

/// D3 lambda desugaring: replace every lambda in the program
/// with a synthetic def (`__psa_lambda_<n>`, body = return of
/// the lambda body) plus an FnVal node at the lambda's site.
/// A lambda may reference its own parameters, user functions
/// and builtins; referencing any other name is a capture and is
/// refused loudly (see desugar_expr).
fn desugar_program(prog: &mut Program) -> Result<(), String> {
    let defnames: std::collections::HashSet<String> =
        prog.funcs.iter().map(|f| f.name.clone()).collect();
    let mut counter = 0u32;
    let mut new_funcs: Vec<Func> = Vec::new();
    for s in &mut prog.top {
        desugar_stmt(s, &defnames, &mut counter, &mut new_funcs)?;
    }
    for f in &mut prog.funcs {
        for s in &mut f.body {
            desugar_stmt(s, &defnames, &mut counter, &mut new_funcs)?;
        }
    }
    // D4: lambdas inside method bodies desugar the same way.
    for c in &mut prog.classes {
        for m in &mut c.methods {
            for s in &mut m.body {
                desugar_stmt(s, &defnames, &mut counter, &mut new_funcs)?;
            }
        }
    }
    prog.funcs.extend(new_funcs);
    Ok(())
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
fn build_class_registry(em: &mut Emitter, prog: &Program) -> Result<(), String> {
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
                Expr::Int(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bool(_) | Expr::NoneLit => {}
                _ => {
                    return Err(eerr(
                        c.line,
                        &format!(
                            "class attribute `{}` of class `{}` must be a literal in D4 v1 (computed class attributes wait for the runtime-registry shape; see D4-NOTES)",
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
        em.classes.push(ClassInfo { id, name: c.name.clone(), flat_methods, flat_attrs, ancestors });
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
        Stmt::While { body, .. }
        | Stmt::ForRange { body, .. }
        | Stmt::ForEach { body, .. } => stmts_use_exc(body),
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
    fn fn_val_code(&mut self, name: &str, line: u32) -> Result<(String, Ty), String> {
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
        if sig.params.iter().any(|t| *t != Ty::Dyn) || sig.ret != Some(Ty::Dyn) {
            return Err(eerr(
                line,
                &format!(
                    "function `{}` cannot be used as a value: callable functions are pinned to all-dynamic signatures, but it was already emitted with a different one",
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

    /// D4: pin a method def as a callable and return its id.
    /// Like fn_val_code, the method is pinned to an all-Dyn
    /// signature and defaults/*args/**kwargs are refused (an
    /// adapter cannot bind them; explicit base calls still can,
    /// through the ordinary def-call path). Unlike fn_val_code,
    /// the return may be Void or a static scalar -- the adapter
    /// normalises it to a dynamic value -- because methods are
    /// routinely written for effect (`__init__`, mutators).
    fn method_callable_id(&mut self, key: &str, line: u32) -> Result<i64, String> {
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
    ) -> Result<Vec<String>, String> {
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
    fn emit_generator_def(&mut self, def: &Func) -> Result<(), String> {
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
        self.emitted.push(w);
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
        self.emitted.push(d);
        let prod_txt = self.emit_gen_producer(def, &prod_name)?;
        self.emitted.push(prod_txt);
        Ok(())
    }

    /// D7: emit a generator def's body as its producer function.
    /// Parameters are dynamic values; `yield` statements send
    /// their wrap on `__psa_gench` (Ctx::gen_chan). Returns may
    /// only be bare / None (checked by gen_check_returns).
    fn emit_gen_producer(&mut self, def: &Func, prod_sla: &str) -> Result<String, String> {
        let param_vars: Vec<(String, Ty, Option<u32>)> = def
            .param_names()
            .into_iter()
            .map(|n| (n, Ty::Dyn, None))
            .collect();
        let mut ctx = Ctx {
            vars: param_vars,
            scopes: vec![0],
            ret: None,
            fname: def.name.clone(),
            out: String::new(),
            prologue: String::new(),
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
        };
        for s in &def.body {
            self.emit_stmt(&mut ctx, s)?;
        }
        let mut txt = String::new();
        txt.push_str(&format!("fn {}(__psa_gench: Chan<i64>", prod_sla));
        for p in def.param_names() {
            txt.push_str(&format!(", {}: PsaVal", p));
        }
        txt.push_str(") -> PsaVal {\n");
        txt.push_str(&ctx.prologue);
        txt.push_str(&ctx.out);
        let last_is_return = ctx
            .out
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .map(|l| l.trim_start().starts_with("return "))
            .unwrap_or(false);
        if !last_is_return {
            txt.push_str(&format!("    return {};\n", Ty::Dyn.default_lit()));
        }
        txt.push_str("}\n");
        Ok(txt)
    }

    /// D7: install an async def's callable face. The body is
    /// registered as a worker def under a `::coro` key and
    /// pinned all-Dyn as a callable; the def's own SLA name
    /// becomes a wrapper that packs the arguments into a list
    /// and returns a lazy coroutine value (tag 14). Calling an
    /// async def never runs the body; `await` does
    /// (__psa_coro_run). Defaults and *args/**kwargs are
    /// refused: the callable adapter cannot bind them.
    fn emit_async_def(&mut self, def: &Func) -> Result<(), String> {
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
        self.emitted.push(w);
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
    ) -> Result<(), String> {
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
    ) -> Result<(), String> {
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
                        self.emit_dyn_store(ctx, &target, atemp)?;
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
    ) -> Result<(String, Ty), String> {
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
    /// type, isinstance. Returns None when `fname` is none of
    /// them; the caller has already established that no user
    /// def or variable shadows the name (a shadowed builtin is
    /// an ordinary call, as in Python).
    fn emit_builtin_call(
        &mut self,
        ctx: &mut Ctx,
        fname: &str,
        args: &[Expr],
        kwargs: &[(String, Expr)],
        line: u32,
    ) -> Result<Option<(String, Ty)>, String> {
        if !matches!(
            fname,
            "int" | "float" | "str" | "bool" | "list" | "tuple" | "set" | "type" | "isinstance"
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
                        Err(_) => Err(eerr(
                            line,
                            &format!(
                                "int() cannot parse the literal {:?} (CPython raises ValueError)",
                                s
                            ),
                        )),
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
                        self.used_dyn = true;
                        Ok(Some((format!("__psa_to_int({})", c), Ty::Int)))
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
                if let Expr::Call(callee, _, _) = &args[0] {
                    if matches!(&**callee, Expr::Name(n) if n == "range") {
                        let comp = Expr::Comp {
                            kind: CompKind::List,
                            key: Box::new(Expr::Name("__psa_rx".to_string())),
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
    ) -> Result<String, String> {
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
            Expr::Name(n) => {
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
    ) -> Result<Option<String>, String> {
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

    fn ensure_fn(&mut self, name: &str, arg_tys: &[Ty], line: u32) -> Result<Sig, String> {
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
        let param_vars: Vec<(String, Ty, Option<u32>)> = def
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
                scopes: vec![0],
                ret: None,
                fname: name.to_string(),
                out: String::new(),
                prologue: String::new(),
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
            };
            let r = (|| {
                for s in &def.body {
                    self.emit_stmt(&mut sctx, s)?;
                }
                Ok::<(), String>(())
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
            scopes: vec![0],
            ret: None,
            fname: name.to_string(),
            out: String::new(),
            prologue: String::new(),
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
        txt.push_str(&format!("fn {}(", self.sla_name(name)));
        for (i, (p, t)) in def.param_names().iter().zip(arg_tys.iter()).enumerate() {
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
        txt.push_str(&ctx.prologue);
        txt.push_str(&ctx.out);
        // Default fallthrough return, appended only when the body's
        // last top-level statement is not already a return. A second,
        // dead return after a live one is accepted by the SLA checker
        // but the SA-text flattener rejects its expansion: the dead
        // block materialises the return temp and then falls through
        // with no terminator (error[FallthroughForbidden], fib/list/
        // dyn1 on the text route, 2026-11-25).
        let last_is_return = ctx
            .out
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .map(|l| l.starts_with("    return "))
            .unwrap_or(false);
        if !last_is_return {
            txt.push_str(&format!("    return {};\n", ret_ty.default_lit()));
        }
        txt.push_str("}\n");
        if !self.emitted_set.contains(&name.to_string()) {
            self.emitted_set.push(name.to_string());
            self.emitted.push(txt);
        }
        Ok(self.sigs.get(name).unwrap().clone())
    }

    /// Wrap a statically-typed value's code as a dynamic value
    /// (boxing at a static/dynamic boundary). Dynamic values pass
    /// through unchanged; kinds that cannot be dynamic in D1 fail
    /// loudly with the phase that will carry them.
    fn box_code(&mut self, code: String, ty: Ty, line: u32) -> Result<String, String> {
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
    fn emit_str_temp(&mut self, ctx: &mut Ctx, s: &str, line: u32) -> Result<String, String> {
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

    /// One slice bound as a dynamic value: omitted is None (tag 3),
    /// ints and bools box, dynamic values pass through.
    fn slice_bound_code(
        &mut self,
        ctx: &mut Ctx,
        b: &Option<Box<Expr>>,
        line: u32,
    ) -> Result<String, String> {
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
    fn emit_container_temp(&mut self, ctx: &mut Ctx, e: &Expr, line: u32) -> Result<String, String> {
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
    ) -> Result<String, String> {
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
    ) -> Result<String, String> {
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
    ) -> Result<String, String> {
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
    /// variables, defs, classes and builtins: anything else
    /// would need a closure capture, which psa does not support
    /// (the loud refusal stands for those). Iterable
    /// expressions themselves are evaluated in the enclosing
    /// scope, so they are not subject to the check.
    fn emit_genexp(
        &mut self,
        ctx: &mut Ctx,
        key: &Expr,
        value: &Option<Box<Expr>>,
        clauses: &[CompClause],
        line: u32,
    ) -> Result<String, String> {
        if value.is_some() {
            return Err(eerr(
                line,
                "a parenthesised comprehension yields single values; there is no dict generator expression (D7)",
            ));
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
        for n in &names {
            if clause_vars.contains(n)
                || self.defs.contains_key(n)
                || self.class_ids.contains_key(n)
                || LAMBDA_BUILTINS.contains(&n.as_str())
            {
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
            line,
        }];
        let mut fors_from_end = 0usize;
        for c in clauses.iter().rev() {
            match c {
                CompClause::If(cond) => {
                    body = vec![Stmt::If {
                        arms: vec![(cond.clone(), body)],
                        els: None,
                        line,
                    }];
                }
                CompClause::For { var, .. } => {
                    let pi = total_fors - 1 - fors_from_end;
                    fors_from_end += 1;
                    body = vec![Stmt::ForEach {
                        vars: vec![var.to_string()],
                        iter: Expr::Name(format!("__psa_gx_{}", pi)),
                        body,
                        line,
                    }];
                }
            }
        }
        let gfunc = Func {
            name: gname.clone(),
            params: (0..total_fors)
                .map(|i| (format!("__psa_gx_{}", i), None))
                .collect(),
            vararg: None,
            kwarg: None,
            is_async: false,
            body,
            line,
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
                        key: Box::new(Expr::Name("__psa_gxe".to_string())),
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
        Ok(format!("{}({})", self.sla_name(&gname), codes.join(", ")))
    }

    /// A comprehension materialised in a temp (D5): the result
    /// container is built by the clause nest emitted as
    /// statements. Loop variables live in a pushed scope
    /// (CPython 3 comprehension scoping: they do not leak).
    fn emit_comp(&mut self, ctx: &mut Ctx, e: &Expr, line: u32) -> Result<String, String> {
        let (kind, key, value, clauses) = match e {
            Expr::Comp { kind, key, value, clauses } => (*kind, key, value, clauses),
            _ => unreachable!(),
        };
        if kind == CompKind::GenExp {
            return self.emit_genexp(ctx, key, value, clauses, line);
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
    ) -> Result<(), String> {
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
    ) -> Result<(), String> {
        ctx.tmp += 1;
        let n = ctx.tmp;
        let idx = format!("__psa_ci_{}", n);
        // range(...) iterates numerically.
        if let Expr::Call(callee, args, kwargs) = iter {
            if let Expr::Name(f) = &**callee {
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
        if let Expr::Name(name) = iter {
            if let Some(t) = ctx.lookup(name) {
                if let Some(elem) = t.elem() {
                    ctx.check_alive(name, line)?;
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
    ) -> Result<String, String> {
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
    ) -> Result<(Vec<String>, Vec<Ty>), String> {
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
    ) -> Result<Vec<String>, String> {
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
    fn unbox_code(&mut self, code: String, want: Ty, line: u32) -> Result<String, String> {
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
    ) -> Result<(), String> {
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
    fn cond_code(&mut self, code: String, ty: Ty, line: u32, what: &str) -> Result<String, String> {
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
    ) -> Result<ListLit, String> {
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
    ) -> Result<(), String> {
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
                    self.emit_dyn_store(ctx, name, &tmp)?;
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
    ) -> Result<Vec<(String, String)>, String> {
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
    ) -> Result<(), String> {
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
                self.emit_dyn_store(ctx, name, &tmp)?;
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
    ) -> Result<(String, Ty), String> {
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
                if let Expr::Name(n) = e {
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
    ) -> Result<(), String> {
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
            ctx.prologue.push_str(&format!(
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
            ctx.prologue.push_str(&format!("    let {}: bool = false;\n", f));
        }
        ctx.try_stack.push(frame);
    }

    /// D6: emit a `break` for the current emission point. When
    /// the innermost try frame was entered at the current loop
    /// depth, the break targets a loop outside that frame: set
    /// its flag and break the frame's guard wrapper instead
    /// (the epilogue re-emits the real break after the finally).
    fn emit_break_through(&mut self, ctx: &mut Ctx) {
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
    ) -> Result<(), String> {
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
                    ctx.prologue.push_str(&format!(
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
    fn handler_cond(&mut self, h: &ExcHandler) -> Result<String, String> {
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
                Expr::Attr(base, name) => {
                    segs.push(name.clone());
                    cur = base;
                }
                Expr::Name(n) => {
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
    ) -> Option<Result<(String, Ty), String>> {
        match e {
            Expr::Int(v) => {
                self.used_dyn = true;
                Some(Ok((format!("__psa_box_int({})", v), Ty::Dyn)))
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
            Expr::NoneLit => {
                self.used_dyn = true;
                Some(Ok((Ty::Dyn.default_lit().to_string(), Ty::Dyn)))
            }
            _ => None,
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
    fn emit_stmt(&mut self, ctx: &mut Ctx, stmt: &Stmt) -> Result<(), String> {
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
        r
    }

    fn emit_stmt_inner(&mut self, ctx: &mut Ctx, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Pass { .. } => Ok(()),
            Stmt::Break { .. } => {
                self.emit_break_through(ctx);
                Ok(())
            }
            Stmt::Continue { .. } => {
                self.emit_continue_through(ctx);
                Ok(())
            }
            Stmt::Return { value, line } => {
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
            Stmt::Assign { name, value, line } => {
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
                            if matches!(value, Expr::Name(_))
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
                if let Some(src) = src_name {
                    if src != *name {
                        ctx.mark_moved(&src, *line);
                    }
                }
                Ok(())
            }
            Stmt::Assert { cond, line } => {
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
                ctx.line(&format!("if !({}) {{ panic(77040); }};", code));
                Ok(())
            }
            Stmt::Del { targets, line } => {
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
                        ctx.check_alive(name, *line)?;
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
                    ctx.check_alive(name, *line)?;
                    if !ctx.deleted.iter().any(|(n, _)| n == name) {
                        ctx.deleted.push((name.clone(), *line));
                    }
                }
                Ok(())
            }
            Stmt::MultiAssign { names, values, line } => {
                // Evaluate every RHS value into its own temp first
                // (Python evaluates the whole RHS before binding, so
                // `a, b = b, a` swaps correctly).
                let mut temps: Vec<String> = Vec::new();
                for v in values {
                    ctx.tmp += 1;
                    let tname = format!("__psa_ma_{}", ctx.tmp);
                    self.emit_stmt(
                        ctx,
                        &Stmt::Assign { name: tname.clone(), value: v.clone(), line: *line },
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
                                value: Expr::Name(temps[0].clone()),
                                line: *line,
                            },
                        )?;
                    }
                } else if values.len() == names.len() {
                    for (n, t) in names.iter().zip(temps.iter()) {
                        self.emit_stmt(
                            ctx,
                            &Stmt::Assign {
                                name: n.clone(),
                                value: Expr::Name(t.clone()),
                                line: *line,
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
            Stmt::Unpack { names, value, line } => {
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
                            line: *line,
                        },
                    );
                }
                // A static list variable unpacks by index, with a
                // runtime length check (panic 77077 on mismatch).
                if let Expr::Name(n) = value {
                    if let Some(t) = ctx.lookup(n) {
                        if t != Ty::Dyn && t.is_list() {
                            ctx.check_alive(n, *line)?;
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
                                            Box::new(Expr::Name(n.clone())),
                                            Box::new(Expr::Int(j as i64)),
                                        ),
                                        line: *line,
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
            Stmt::AttrAssign { base, name, op, value, line } => {
                // Class-side attribute assignment is refused in
                // D4 v1: the fallback class-attribute shape is
                // read-only literals (the mutable registry waits
                // for the global probe; see D4-NOTES).
                if let Expr::Name(n) = base {
                    if ctx.lookup(n).is_none() && self.class_ids.contains_key(n) {
                        return Err(eerr(
                            *line,
                            &format!(
                                "assigning class attribute `{}.{}` is not supported yet (class attributes are read-only literals in D4 v1; the mutable registry shape is pending a probe)",
                                n, name
                            ),
                        ));
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
            Stmt::AttrDel { base, name, line } => {
                if let Expr::Name(n) = base {
                    if ctx.lookup(n).is_none() && self.class_ids.contains_key(n) {
                        return Err(eerr(
                            *line,
                            &format!(
                                "deleting class attribute `{}.{}` is not supported (class attributes are read-only in D4 v1)",
                                n, name
                            ),
                        ));
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
            Stmt::IndexAssign { name, index, op, value, line } => {
                let ty = ctx
                    .lookup(name)
                    .ok_or_else(|| eerr(*line, &format!("index assignment to undefined variable `{}`", name)))?;
                ctx.check_alive(name, *line)?;
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
            Stmt::ExprStmt { value, line } => {
                // print(...) is a builtin statement-level call.
                if let Expr::Call(callee, args, kwargs) = value {
                    if let Expr::Name(n) = &**callee {
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
                ctx.line(&format!("{};", code));
                Ok(())
            }
            Stmt::If { arms, els, line } => {
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
            Stmt::While { cond, body, line } => {
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
                Ok(())
            }
            Stmt::ForRange { var, start, stop, body, line } => {
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
                ctx.line(&format!("while {} < {} {{", var, stop_var));
                ctx.indent += 1;
                // Body statements share the loop scope (the loop variable
                // stays visible); nested blocks scope themselves.
                ctx.loop_depth += 1;
                for s in body {
                    self.emit_stmt(ctx, s)?;
                }
                ctx.loop_depth -= 1;
                ctx.line(&format!("{} = {} + 1;", var, var));
                ctx.indent -= 1;
                ctx.line("}");
                let mark = ctx.scopes.pop().unwrap();
                ctx.vars.truncate(mark);
                ctx.indent -= 1;
                ctx.line("}");
                Ok(())
            }
            Stmt::ForEach { vars, iter, body, line } => {
                let (obj, ty) = match iter {
                    Expr::Name(n) => {
                        let t = ctx
                            .lookup(n)
                            .ok_or_else(|| eerr(*line, &format!("undefined variable `{}`", n)))?;
                        ctx.check_alive(n, *line)?;
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
                    ctx.line(&format!("let {} = {};", vars[0], get_code.replace("{}", &idx)));
                }
                ctx.loop_depth += 1;
                for s in body {
                    self.emit_stmt(ctx, s)?;
                }
                ctx.loop_depth -= 1;
                ctx.line(&format!("{} = {} + 1;", idx, idx));
                ctx.indent -= 1;
                ctx.line("}");
                let mark = ctx.scopes.pop().unwrap();
                ctx.vars.truncate(mark);
                ctx.indent -= 1;
                ctx.line("}");
                Ok(())
            }
            Stmt::Try { body, handlers, els, finally, line } => {
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
                ctx.line("break;");
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
            Stmt::Raise { value, line } => {
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
                        if let Expr::Name(n) = e {
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
            Stmt::Yield { value, line } => {
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
            Stmt::Match { subject, cases, line } => {
                self.emit_match(ctx, subject, cases, *line)
            }
            Stmt::With { items, body, line } => {
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
                ctx.line("break;");
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
                    Expr::Name(n) => n.clone(),
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
    ) -> Result<(), String> {
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
    ) -> Result<(), String> {
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
            Expr::NoneLit => {
                self.used_dyn = true;
                Ok((Ty::Dyn.default_lit().to_string(), Ty::Dyn))
            }
            Expr::Name(n) => {
                ctx.check_not_deleted(n, line)?;
                match ctx.lookup(n) {
                    Some(t) => {
                        ctx.check_alive(n, line)?;
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
            Expr::Attr(base, name) => {
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
                // where the name is a class, not a variable. In
                // the D4 v1 fallback shape the (literal) value is
                // evaluated at the read site.
                if let Expr::Name(n) = &**base {
                    if ctx.lookup(n).is_none() {
                        if let Some(&cid) = self.class_ids.get(n) {
                            let found: Option<Expr> = self.classes[cid]
                                .flat_attrs
                                .iter()
                                .find(|(an, _)| an == name)
                                .map(|(_, ae)| ae.clone());
                            match found {
                                Some(Expr::Int(v)) => {
                                    self.used_dyn = true;
                                    return Ok((format!("__psa_box_int({})", v), Ty::Dyn));
                                }
                                Some(Expr::Float(v)) => {
                                    self.used_dyn = true;
                                    return Ok((format!("__psa_box_float({})", fmt_f64(v)), Ty::Dyn));
                                }
                                Some(Expr::Bool(b)) => {
                                    self.used_dyn = true;
                                    return Ok((format!("__psa_box_bool({})", b), Ty::Dyn));
                                }
                                Some(Expr::Str(s)) => {
                                    let t = self.emit_str_temp(ctx, &s, line)?;
                                    return Ok((t, Ty::Dyn));
                                }
                                Some(_) => {
                                    return Ok((Ty::Dyn.default_lit().to_string(), Ty::Dyn));
                                }
                                None => {
                                    return Err(eerr(
                                        line,
                                        &format!(
                                            "class `{}` has no attribute `{}` readable as a value in D4 v1 (methods are reached through instances or explicit base calls)",
                                            n, name
                                        ),
                                    ))
                                }
                            }
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
                self.used_dyn = true;
                ctx.tmp += 1;
                let acc = format!("__psa_ctmp_{}", ctx.tmp);
                ctx.line(&format!("let {} = __psa_str_new();", acc));
                for p in parts {
                    match p {
                        FStrPart::Lit(s) => {
                            if !s.is_empty() {
                                let t = self.emit_str_temp(ctx, s, line)?;
                                ctx.line(&format!(
                                    "{} = __psa_str_append_str({}, {});",
                                    acc, acc, t
                                ));
                            }
                        }
                        FStrPart::Expr(x) => {
                            if let Expr::Str(s) = x {
                                let t = self.emit_str_temp(ctx, s, line)?;
                                ctx.line(&format!(
                                    "{} = __psa_str_append_str({}, {});",
                                    acc, acc, t
                                ));
                                continue;
                            }
                            let (code, ty) = self.emit_expr(ctx, x, line)?;
                            let part = match ty {
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
                                acc, acc, part
                            ));
                        }
                    }
                }
                Ok((acc, Ty::Dyn))
            }
            Expr::Index(base, idx) => {
                // Chained indexing on a dynamic sub-expression
                // (m[1][0]): the inner value is the container.
                if !matches!(&**base, Expr::Name(_)) {
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
                let obj = match &**base {
                    Expr::Name(n) => n.clone(),
                    _ => unreachable!(),
                };
                let ty = ctx
                    .lookup(&obj)
                    .ok_or_else(|| eerr(line, &format!("undefined variable `{}`", obj)))?;
                ctx.check_alive(&obj, line)?;
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
                if !matches!(&**base, Expr::Name(_)) {
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
                let obj = match &**base {
                    Expr::Name(n) => n.clone(),
                    _ => unreachable!(),
                };
                let ty = ctx
                    .lookup(&obj)
                    .ok_or_else(|| eerr(line, &format!("undefined variable `{}`", obj)))?;
                ctx.check_alive(&obj, line)?;
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
                                    Expr::Call(Box::new(Expr::Name(k)), args.clone(), Vec::new());
                                self.emit_expr(ctx, &call, line)
                            }
                            None => Err(eerr(
                                line,
                                &format!("class `{}` has no method `{}`", obj, method),
                            )),
                        };
                    }
                }
                // D6: a method call on an imported module:
                // `foo.bar(...)` is a call of the module def
                // `foo.bar`; `foo.Klass(...)` constructs a
                // module class.
                if ctx.lookup(obj).is_none() && self.modules.iter().any(|m| m == obj) {
                    let key = format!("{}.{}", obj, method);
                    if self.defs.contains_key(&key) {
                        let call =
                            Expr::Call(Box::new(Expr::Name(key)), args.clone(), Vec::new());
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
                    ctx.check_alive(obj, line)?;
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
                ctx.check_alive(obj, line)?;
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
            Expr::Call(callee, args, kwargs) => {
                // D6: a call through an attribute chain rooted
                // at an imported module: `foo.bar.baz(...)` (a
                // def of a dotted module) or `foo.bar.Klass(...)`
                // (a class of a dotted module). A chain rooted
                // at a variable is never a module reference.
                if matches!(&**callee, Expr::Attr(..)) {
                    if let Some(key) = self.module_def_key(ctx, callee) {
                        let call = Expr::Call(
                            Box::new(Expr::Name(key)),
                            args.clone(),
                            kwargs.clone(),
                        );
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
                if let Expr::Attr(base, mname) = &**callee {
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
                let fname = match &**callee {
                    Expr::Name(n) => n.clone(),
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
                    ctx.check_alive(&fname, line)?;
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
                            ctx.out.push_str(&r_lines);
                            ctx.indent += 1;
                            ctx.line(&format!("{} = {};", name, rc));
                            ctx.indent -= 1;
                            ctx.line("};");
                            return Ok((name, Ty::Bool));
                        }
                        ctx.out.push_str(&r_lines);
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
                ctx.out.push_str(&then_lines);
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
                ctx.out.push_str(&els_lines);
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
    ) -> Result<(String, Ty), String> {
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
        r_lines: String,
        line: u32,
    ) -> Result<(String, Ty), String> {
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
        ctx.out.push_str(&r_lines);
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
    ) -> Result<(String, Ty), String> {
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
    ) -> Result<(String, Ty), String> {
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
            Expr::Call(callee, args, kwargs) => {
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
                    if let Expr::Name(n) = v {
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
            Expr::Index(base, idx) => {
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
            Expr::Attr(base, _) => {
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
                Stmt::AttrAssign { base, value, .. } => {
                    expr_consumes(em, base, param, visiting)
                        || expr_consumes(em, value, param, visiting)
                }
                Stmt::AttrDel { base, .. } => expr_consumes(em, base, param, visiting),
                Stmt::Unpack { value, .. } => expr_consumes(em, value, param, visiting),
                Stmt::ForEach { iter, body, .. } => {
                    expr_consumes(em, iter, param, visiting)
                        || stmts_consume(em, body, param, visiting)
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
            Stmt::While { body, .. }
            | Stmt::ForRange { body, .. }
            | Stmt::ForEach { body, .. }
            | Stmt::With { body, .. } => stmts_have_yield(body),
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
fn gen_check_returns(stmts: &[Stmt], fname: &str, line: u32) -> Result<(), String> {
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
            Stmt::While { body, .. }
            | Stmt::ForRange { body, .. }
            | Stmt::ForEach { body, .. }
            | Stmt::With { body, .. } => gen_check_returns(body, fname, line)?,
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
        Stmt::While { body, .. } => body.iter().any(deep_has_raise),
        Stmt::ForRange { body, .. } => body.iter().any(deep_has_raise),
        Stmt::ForEach { body, .. } => body.iter().any(deep_has_raise),
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
) -> Result<(), String> {
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
            Expr::Call(callee, args, kwargs) => {
                if let Expr::Name(n) = &**callee {
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
            Stmt::While { cond, body, .. } => expr_calls(cond, name) || def_calls(body, name),
            Stmt::ForRange { start, stop, body, .. } => {
                start.as_ref().map(|e| expr_calls(e, name)).unwrap_or(false)
                    || expr_calls(stop, name)
                    || def_calls(body, name)
            }
            Stmt::ForEach { iter, body, .. } => expr_calls(iter, name) || def_calls(body, name),
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
        Expr::Name(n) => n == name,
        Expr::Bin(a, _, b) => expr_mentions(a, name) || expr_mentions(b, name),
        Expr::Unary(_, a) => expr_mentions(a, name),
        Expr::Call(f, args, kwargs) => {
            expr_mentions(f, name)
                || args.iter().any(|a| expr_mentions(a, name))
                || kwargs.iter().any(|(_, v)| expr_mentions(v, name))
        }
        Expr::Index(a, b) => expr_mentions(a, name) || expr_mentions(b, name),
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
        Expr::Attr(base, _) => expr_mentions(base, name),
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
        Expr::Attr(base, _) => expr_has_call(base),
        Expr::IfExp { cond, then, els } => {
            expr_has_call(cond) || expr_has_call(then) || expr_has_call(els)
        }
        Expr::FStr(parts) => parts
            .iter()
            .any(|p| matches!(p, FStrPart::Expr(x) if expr_has_call(x))),
        // An await runs the coroutine worker synchronously
        // through __psa_coro_run — a call in every sense that
        // matters here (it can leave a pending exception).
        Expr::Await(a) => expr_has_call(a),
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

/// The D1 dynamic runtime, emitted once when any dynamic value is
/// used. PsaVal tags: 0 = int, 1 = float, 2 = bool, 3 = None.
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

fn __psa_as_float(v: PsaVal) -> f64 {
    if v.tag == 1 { return v.fpay; };
    if v.tag == 0 || v.tag == 2 { return v.ipay as f64; };
    panic(77010);
}

fn __psa_dyn_floor_f(x: f64) -> f64 {
    if x >= 9223372036854775808.0 { return x; };
    if x <= -9223372036854775808.0 { return x; };
    let q = x as i64;
    let qf = q as f64;
    if qf > x { return qf - 1.0; };
    return qf;
}

fn __psa_dyn_add(a: PsaVal, b: PsaVal) -> PsaVal {
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) {
        if b.ipay > 0 && a.ipay > 9223372036854775807 - b.ipay { panic(77080); };
        if b.ipay < 0 && a.ipay < -9223372036854775807 - 1 - b.ipay { panic(77080); };
        return __psa_box_int(a.ipay + b.ipay);
    };
    if a.tag <= 2 && b.tag <= 2 { return __psa_box_float(__psa_as_float(a) + __psa_as_float(b)); };
    if a.tag == 6 && b.tag == 6 { return __psa_str_concat(a, b); };
    panic(77011);
}

fn __psa_dyn_sub(a: PsaVal, b: PsaVal) -> PsaVal {
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) {
        if b.ipay < 0 && a.ipay > 9223372036854775807 + b.ipay { panic(77080); };
        if b.ipay > 0 && a.ipay < -9223372036854775807 - 1 + b.ipay { panic(77080); };
        return __psa_box_int(a.ipay - b.ipay);
    };
    if a.tag <= 2 && b.tag <= 2 { return __psa_box_float(__psa_as_float(a) - __psa_as_float(b)); };
    panic(77011);
}

fn __psa_dyn_mul(a: PsaVal, b: PsaVal) -> PsaVal {
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) {
        let x = a.ipay;
        let y = b.ipay;
        if (x == -1 && y == -9223372036854775807 - 1) || (y == -1 && x == -9223372036854775807 - 1) { panic(77080); };
        let r = x * y;
        if x != 0 && r / x != y { panic(77080); };
        return __psa_box_int(r);
    };
    if a.tag <= 2 && b.tag <= 2 { return __psa_box_float(__psa_as_float(a) * __psa_as_float(b)); };
    if a.tag == 6 && (b.tag == 0 || b.tag == 2) { return __psa_str_repeat(a, b.ipay); };
    if b.tag == 6 && (a.tag == 0 || a.tag == 2) { return __psa_str_repeat(b, a.ipay); };
    panic(77011);
}

fn __psa_dyn_div(a: PsaVal, b: PsaVal) -> PsaVal {
    return __psa_box_float(__psa_as_float(a) / __psa_as_float(b));
}

fn __psa_dyn_floordiv(a: PsaVal, b: PsaVal) -> PsaVal {
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) { return __psa_box_int(__psa_floordiv(a.ipay, b.ipay)); };
    if a.tag <= 2 && b.tag <= 2 { return __psa_box_float(__psa_dyn_floor_f(__psa_as_float(a) / __psa_as_float(b))); };
    panic(77011);
}

fn __psa_dyn_mod(a: PsaVal, b: PsaVal) -> PsaVal {
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) { return __psa_box_int(__psa_floormod(a.ipay, b.ipay)); };
    if a.tag <= 2 && b.tag <= 2 {
        let fa = __psa_as_float(a);
        let fb = __psa_as_float(b);
        return __psa_box_float(fa - __psa_dyn_floor_f(fa / fb) * fb);
    };
    panic(77011);
}

fn __psa_dyn_pow(a: PsaVal, b: PsaVal) -> PsaVal {
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) && b.ipay >= 0 { return __psa_box_int(__psa_ipow(a.ipay, b.ipay)); };
    if a.tag <= 2 && b.tag <= 2 { return __psa_box_float(sa_math_pow(__psa_as_float(a), __psa_as_float(b))); };
    panic(77011);
}

fn __psa_dyn_eq(a: PsaVal, b: PsaVal) -> bool {
    if a.tag == 5 && b.tag == 5 { return __psa_dict_eq(a, b); };
    if a.tag == 5 || b.tag == 5 { return false; };
    if a.tag == 4 && b.tag == 4 { return __psa_list_eq(a, b); };
    if a.tag == 6 && b.tag == 6 { return __psa_str_eq(a, b); };
    if a.tag == 10 && b.tag == 10 { return __psa_tuple_eq(a, b); };
    if a.tag == 11 && b.tag == 11 { return __psa_set_eq(a, b); };
    if a.tag == 8 || a.tag == 9 || b.tag == 8 || b.tag == 9 { return a.tag == b.tag && a.ipay == b.ipay; };
    if a.tag == 3 && b.tag == 3 { return true; };
    if a.tag <= 2 && b.tag <= 2 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) == __psa_as_float(b); };
        return a.ipay == b.ipay;
    };
    return false;
}

fn __psa_dyn_ne(a: PsaVal, b: PsaVal) -> bool {
    return !(__psa_dyn_eq(a, b));
}

fn __psa_dyn_lt(a: PsaVal, b: PsaVal) -> bool {
    if a.tag <= 2 && b.tag <= 2 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) < __psa_as_float(b); };
        return a.ipay < b.ipay;
    };
    if a.tag == 6 && b.tag == 6 { return __psa_str_cmp(a, b) < 0; };
    panic(77012);
}

fn __psa_dyn_le(a: PsaVal, b: PsaVal) -> bool {
    if a.tag <= 2 && b.tag <= 2 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) <= __psa_as_float(b); };
        return a.ipay <= b.ipay;
    };
    if a.tag == 6 && b.tag == 6 { return __psa_str_cmp(a, b) <= 0; };
    panic(77012);
}

fn __psa_dyn_gt(a: PsaVal, b: PsaVal) -> bool {
    if a.tag <= 2 && b.tag <= 2 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) > __psa_as_float(b); };
        return a.ipay > b.ipay;
    };
    if a.tag == 6 && b.tag == 6 { return __psa_str_cmp(a, b) > 0; };
    panic(77012);
}

fn __psa_dyn_ge(a: PsaVal, b: PsaVal) -> bool {
    if a.tag <= 2 && b.tag <= 2 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) >= __psa_as_float(b); };
        return a.ipay >= b.ipay;
    };
    if a.tag == 6 && b.tag == 6 { return __psa_str_cmp(a, b) >= 0; };
    panic(77012);
}

fn __psa_dyn_truthy(v: PsaVal) -> bool {
    if v.tag == 0 { return v.ipay != 0; };
    if v.tag == 1 { return v.fpay != 0.0; };
    if v.tag == 2 { return v.ipay != 0; };
    if v.tag == 3 { return false; };
    if v.tag == 4 { return __psa_list_len(v) != 0; };
    if v.tag == 5 { return __psa_dict_len(v) != 0; };
    if v.tag == 6 { return __psa_str_bytes_len(v) != 0; };
    if v.tag == 8 { return __psa_obj_truthy(v); };
    if v.tag == 9 { return true; };
    if v.tag == 10 { return __psa_vec_len_h(v.ipay as ptr) != 0; };
    if v.tag == 11 { return __psa_vec_len_h(v.ipay as ptr) != 0; };
    if v.tag == 12 { return true; };
    if v.tag == 13 { return true; };
    if v.tag == 14 { return true; };
    panic(77010);
}

fn __psa_dyn_neg(v: PsaVal) -> PsaVal {
    if v.tag == 0 || v.tag == 2 {
        if v.ipay == -9223372036854775807 - 1 { panic(77080); };
        return __psa_box_int(-(v.ipay));
    };
    if v.tag == 1 { return __psa_box_float(-(v.fpay)); };
    panic(77015);
}

fn __psa_dyn_bitnot(v: PsaVal) -> PsaVal {
    if v.tag == 0 || v.tag == 2 { return __psa_box_int((-(v.ipay) - 1)); };
    panic(77013);
}

fn __psa_dyn_bitand(a: PsaVal, b: PsaVal) -> PsaVal {
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) { return __psa_box_int(a.ipay & b.ipay); };
    panic(77013);
}

fn __psa_dyn_bitor(a: PsaVal, b: PsaVal) -> PsaVal {
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) { return __psa_box_int(a.ipay | b.ipay); };
    panic(77013);
}

fn __psa_dyn_bitxor(a: PsaVal, b: PsaVal) -> PsaVal {
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) { return __psa_box_int(a.ipay ^ b.ipay); };
    panic(77013);
}

fn __psa_dyn_shl(a: PsaVal, b: PsaVal) -> PsaVal {
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) {
        if b.ipay < 0 || b.ipay >= 64 { panic(77014); };
        return __psa_box_int(a.ipay << b.ipay);
    };
    panic(77013);
}

fn __psa_dyn_shr(a: PsaVal, b: PsaVal) -> PsaVal {
    if (a.tag == 0 || a.tag == 2) && (b.tag == 0 || b.tag == 2) {
        if b.ipay < 0 || b.ipay >= 64 { panic(77014); };
        return __psa_box_int(a.ipay >> b.ipay);
    };
    panic(77013);
}

fn __psa_unbox_int(v: PsaVal) -> i64 {
    if v.tag == 0 { return v.ipay; };
    if v.tag == 2 { return v.ipay; };
    panic(77020);
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

fn __psa_print_repr(v: PsaVal) {
    if v.tag == 0 { __psa_print_i64(v.ipay); return; };
    if v.tag == 2 {
        if v.ipay != 0 { sla_print_bytes(STR_PTR("True"), STR_LEN("True")); } else { sla_print_bytes(STR_PTR("False"), STR_LEN("False")); };
        return;
    };
    if v.tag == 3 { sla_print_bytes(STR_PTR("None"), STR_LEN("None")); return; };
    if v.tag == 4 { __psa_print_list_repr(v); return; };
    if v.tag == 6 { __psa_print_str(__psa_str_repr(v)); return; };
    if v.tag == 5 { __psa_print_dict_repr(v); return; };
    if v.tag == 8 { __psa_print_str(__psa_obj_repr(v)); return; };
    if v.tag == 10 { __psa_print_tuple_repr(v); return; };
    if v.tag == 11 { __psa_print_set_repr(v); return; };
    panic(77058);
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

// The no-newline half of the dynamic printer (D5), for
// multi-argument print and sep=/end= forms.
fn __psa_print_dyn_nl(v: PsaVal) {
    if v.tag == 4 { __psa_print_list_repr(v); return; };
    if v.tag == 6 { __psa_print_str(v); return; };
    if v.tag == 5 { __psa_print_dict_repr(v); return; };
    if v.tag == 8 { __psa_print_str(__psa_obj_str(v)); return; };
    if v.tag == 10 { __psa_print_tuple_repr(v); return; };
    if v.tag == 11 { __psa_print_set_repr(v); return; };
    if v.tag == 12 { __psa_print_str(__psa_file_str(v)); return; };
    if v.tag == 13 { __psa_print_str(__psa_str_of(v)); return; };
    if v.tag == 14 { __psa_print_str(__psa_str_of(v)); return; };
    if v.tag == 0 { __psa_print_i64(v.ipay); return; };
    if v.tag == 2 {
        if v.ipay != 0 { sla_print_bytes(STR_PTR("True"), STR_LEN("True")); } else { sla_print_bytes(STR_PTR("False"), STR_LEN("False")); };
        return;
    };
    if v.tag == 3 { sla_print_bytes(STR_PTR("None"), STR_LEN("None")); return; };
    panic(77030);
}

fn __psa_print_dyn(v: PsaVal) -> i32 {
    if v.tag == 4 { __psa_print_list_repr(v); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 6 { __psa_print_str(v); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 5 { __psa_print_dict_repr(v); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 8 { __psa_print_str(__psa_obj_str(v)); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 10 { __psa_print_tuple_repr(v); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 11 { __psa_print_set_repr(v); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 12 { __psa_print_str(__psa_file_str(v)); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 13 { __psa_print_str(__psa_str_of(v)); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 14 { __psa_print_str(__psa_str_of(v)); sla_print_bytes(STR_PTR("\n"), STR_LEN("\n")); return 0; };
    if v.tag == 0 { println("{}", v.ipay); return 0; };
    if v.tag == 2 {
        if v.ipay != 0 { println("True"); } else { println("False"); };
        return 0;
    };
    if v.tag == 3 { println("None"); return 0; };
    panic(77030);
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

// Python key equivalence: the numeric family (int/float/bool)
// compares by value across tags (1 == True == 1.0 are one key),
// None equals only None, str compares by content. Container keys
// never match (they are refused at set time, panic 77066).
fn __psa_key_eq(a: PsaVal, b: PsaVal) -> bool {
    if a.tag <= 2 && b.tag <= 2 {
        if a.tag == 1 || b.tag == 1 { return __psa_as_float(a) == __psa_as_float(b); };
        return a.ipay == b.ipay;
    };
    if a.tag == 3 && b.tag == 3 { return true; };
    if a.tag == 6 && b.tag == 6 { return __psa_str_eq(a, b); };
    if a.tag == 10 && b.tag == 10 { return __psa_tuple_key_eq(a, b); };
    if a.tag == 8 && b.tag == 8 { return a.ipay == b.ipay; };
    return false;
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
    if v.tag == 10 { return __psa_tuple_get(v, __psa_key_int(k)); };
    panic(77052);
}

fn __psa_dyn_set(v: PsaVal, k: PsaVal, x: PsaVal) {
    if v.tag == 4 { __psa_list_set(v, __psa_key_int(k), x); return; };
    if v.tag == 5 { __psa_dict_set(v, k, x); return; };
    if v.tag == 6 { panic(77063); };
    panic(77052);
}

fn __psa_dyn_contains(v: PsaVal, x: PsaVal) -> bool {
    if v.tag == 4 { return __psa_list_contains(v, x); };
    if v.tag == 5 { return __psa_dict_contains(v, x); };
    if v.tag == 6 {
        if x.tag != 6 { panic(77062); };
        return __psa_str_contains(v, x);
    };
    if v.tag == 10 { return __psa_tuple_contains(v, x); };
    if v.tag == 11 { return __psa_set_contains(v, x); };
    panic(77052);
}

fn __psa_iter_get(v: PsaVal, i: i64) -> PsaVal {
    if v.tag == 4 { return __psa_list_get(v, i); };
    if v.tag == 6 { return __psa_str_get_cp(v, i); };
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
    if v.tag == 4 || v.tag == 5 || v.tag == 6 || v.tag == 10 || v.tag == 11 {
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
    panic(77030);
}

// int(x) / float(x) on a dynamic value (D5).
fn __psa_to_int(v: PsaVal) -> i64 {
    if v.tag == 0 { return v.ipay; };
    if v.tag == 2 { return v.ipay; };
    if v.tag == 1 { return v.fpay as i64; };
    if v.tag == 6 { return __psa_str_to_int(v); };
    panic(77075);
}

fn __psa_to_float(v: PsaVal) -> f64 {
    if v.tag == 1 { return v.fpay; };
    if v.tag == 0 { return v.ipay as f64; };
    if v.tag == 2 { return v.ipay as f64; };
    if v.tag == 6 { return __psa_str_to_float(v); };
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
// 5 dict, 6 tuple, 7 set, 8 object, 9 NoneType.

fn __psa_is_type(v: PsaVal, code: i64) -> bool {
    if code == 0 { return v.tag == 0 || v.tag == 2; };
    if code == 1 { return v.tag == 1; };
    if code == 2 { return v.tag == 2; };
    if code == 3 { return v.tag == 6; };
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
