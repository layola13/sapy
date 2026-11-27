//! Indentation-aware lexer for the .psa subset.
//!
//! Produces Newline / Indent / Dedent tokens Python-style. Anything outside
//! the supported subset (tabs in indentation, ...) is a loud error,
//! never a silent skip.

use crate::diag::Diagnostic;

/// One part of an f-string token: literal text, or a replacement
/// field `{...}` (the parser re-parses the expression source).
#[derive(Debug, Clone, PartialEq)]
pub enum FPart {
    Lit(String),
    /// A plain field `{expr}`: just the expression source.
    Expr(String),
    /// A field carrying a conversion (`!r`/`!s`/`!a`), a debug
    /// `=` marker, and/or a format spec after `:`.
    Fmt {
        /// Source of the value expression (trimmed).
        src: String,
        /// Conversion character: 'r', 's' or 'a'.
        conv: Option<char>,
        /// Debug `=` marker: the verbatim literal prefix (the
        /// expression text as written, the `=` and any spaces
        /// after it) that CPython prints in front of the value.
        /// None for ordinary fields.
        debug: Option<String>,
        /// The raw format-spec source after the first top-level
        /// `:`. It is itself a mini f-string (literal text plus
        /// nested `{...}` fields) and is resolved by the parser,
        /// recursively.
        spec: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(i64),
    /// Integer literal that does not fit i64, kept as decimal text
    /// (sign included when written by the parser for patterns). The
    /// dynamic layer materialises it as an arbitrary-precision int
    /// (PsaVal tag 17); static positions refuse it loudly (PSA-E6010).
    BigInt(String),
    Float(f64),
    Str(String),
    /// `b"..."` bytes literal: the decoded byte content.
    Bytes(Vec<u8>),
    FStr(Vec<FPart>),
    // keywords
    Def,
    Class,
    Return,
    If,
    Elif,
    Else,
    While,
    For,
    In,
    Is,
    Break,
    Continue,
    Pass,
    And,
    Or,
    Not,
    True,
    False,
    NoneLit,
    Lambda,
    Try,
    Except,
    Finally,
    Raise,
    With,
    Import,
    From,
    Yield,
    Async,
    Await,
    // symbols
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Dot,
    Plus,
    Minus,
    Star,
    Slash,
    FloorDiv,
    Percent,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    Tilde,
    Shl,
    Shr,
    EqEq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Assign,
    /// `:=` assignment-expression operator.
    Walrus,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    FloorDivEq,
    PercentEq,
    PowEq,
    BitAndEq,
    BitOrEq,
    BitXorEq,
    ShlEq,
    ShrEq,
    Newline,
    Indent,
    Dedent,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub tok: Tok,
    pub line: u32,
    /// 1-based CHARACTER column (Unicode scalar count from the line
    /// start) of the token's first character (P0b, design §3.2).
    pub col: u32,
    /// 1-based BYTE column of the token's first character (P0b).
    /// Kept beside `col` so the design's byte/char question (§6.1)
    /// never needs a second pass over the source to answer.
    pub col_byte: u32,
}

/// 1-based (character column, byte column) of byte offset `pos` in
/// `src` (P0b). Found by scanning back to the previous newline, so
/// it stays exact on continuation lines and after multi-line string
/// literals, where a running line-start cursor would go stale.
/// `pos` must be a UTF-8 boundary (the lexer only ever advances by
/// whole characters).
/// (Ruling #3: this is the ONE column function. P0a's isomorphic
/// `cols` and its `token_at` helper were retired in the merge;
/// all call sites use `col_at`.)
fn col_at(src: &str, pos: usize) -> (u32, u32) {
    let line_start = src[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let col_byte = (pos - line_start + 1) as u32;
    let col = src[line_start..pos].chars().count() as u32 + 1;
    (col, col_byte)
}

/// Arbitrary-precision radix -> decimal conversion for integer
/// literals past i64 (compiler-side only, no external crates).
/// Little-endian base-10^9 limbs, one multiply-add per source digit.
fn radix_digits_to_decimal(digits: &str, radix: u32) -> String {
    let mut mag: Vec<u32> = vec![0];
    for ch in digits.chars() {
        let mut carry = ch.to_digit(radix).unwrap_or(0) as u64;
        for limb in mag.iter_mut() {
            let cur = (*limb as u64) * (radix as u64) + carry;
            *limb = (cur % 1_000_000_000) as u32;
            carry = cur / 1_000_000_000;
        }
        while carry > 0 {
            mag.push((carry % 1_000_000_000) as u32);
            carry /= 1_000_000_000;
        }
    }
    let mut s = String::new();
    let mut it = mag.iter().rev();
    if let Some(top) = it.next() {
        s.push_str(&top.to_string());
    }
    for limb in it {
        s.push_str(&format!("{:09}", limb));
    }
    s
}

fn err(line: u32, msg: &str) -> Diagnostic {
    Diagnostic::lex(line, msg)
}

/// Lex error with a known column (P0a sourcemap). The source
/// line text is attached as the site's snippet, so the text
/// rendering prints the excerpt + caret for lex errors without
/// any other stage having to know the source. `line` is
/// 1-based; a line 0 (never produced by the lexer) gets no
/// snippet rather than a wrong one.
fn err_at(line: u32, col: u32, src: &str, msg: &str) -> Diagnostic {
    let d = Diagnostic::lex_at(line, col, msg);
    if line == 0 {
        return d;
    }
    match src.lines().nth(line as usize - 1) {
        Some(text) => d.with_snippet(text.to_string()),
        None => d,
    }
}

pub fn lex(src: &str) -> Result<Vec<Token>, Diagnostic> {
    let mut out: Vec<Token> = Vec::new();
    let mut indents: Vec<usize> = vec![0];
    let bytes = src.as_bytes();
    let mut pos = 0usize;
    let mut line: u32 = 1;
    let n = bytes.len();

    while pos < n {
        // Start of a logical line: measure indentation.
        let line_start = pos;
        let mut indent = 0usize;
        while pos < n && bytes[pos] == b' ' {
            indent += 1;
            pos += 1;
        }
        if pos < n && bytes[pos] == b'\t' {
            // The tab sits right after `indent` spaces, so its
            // character column is indent + 1.
            return Err(err_at(
                line,
                indent as u32 + 1,
                src,
                "tab indentation is not supported in v0.1; use spaces",
            ));
        }
        // Skip blank and comment-only lines entirely (no indent tokens).
        let mut probe = pos;
        while probe < n && (bytes[probe] == b' ' || bytes[probe] == b'\r') {
            probe += 1;
        }
        if probe >= n || bytes[probe] == b'\n' || bytes[probe] == b'#' {
            while pos < n && bytes[pos] != b'\n' {
                pos += 1;
            }
            if pos < n {
                pos += 1;
                line += 1;
            }
            continue;
        }
        let _ = line_start;
        // Column of the first content character of this line; the
        // Indent/Dedent tokens emitted below start here (P0b).
        let (tok_col, tok_col_byte) = col_at(src, pos);
        // Indentation stack.
        let cur = *indents.last().unwrap();
        if indent > cur {
            indents.push(indent);
            out.push(Token { tok: Tok::Indent, line, col: tok_col, col_byte: tok_col_byte });
        } else if indent < cur {
            while *indents.last().unwrap() > indent {
                indents.pop();
                out.push(Token { tok: Tok::Dedent, line, col: tok_col, col_byte: tok_col_byte });
            }
            if *indents.last().unwrap() != indent {
                return Err(err_at(
                    line,
                    indent as u32 + 1,
                    src,
                    "inconsistent dedent: indentation matches no outer level",
                ));
            }
        }
        // Tokenize the rest of the line. Implicit line joining:
        // a newline inside ANY bracket pair ((), [] or {}) is
        // whitespace, never a statement terminator (CPython's
        // rule) -- this is what makes multi-line literals, call
        // argument lists and parenthesised expressions work.
        // The inner loop consumes the continuation lines itself,
        // so they never reach the indentation logic above.
        let mut depth_bracket = 0i32;
        loop {
            if pos >= n {
                break;
            }
            let c = bytes[pos];
            if c == b'\n' && depth_bracket == 0 {
                let (nl_col, nl_col_byte) = col_at(src, pos);
                pos += 1;
                line += 1;
                out.push(Token {
                    tok: Tok::Newline,
                    line: line - 1,
                    col: nl_col,
                    col_byte: nl_col_byte,
                });
                break;
            }
            if c == b'#' {
                while pos < n && bytes[pos] != b'\n' {
                    pos += 1;
                }
                continue;
            }
            if c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
                if c == b'\n' {
                    line += 1;
                }
                pos += 1;
                continue;
            }
            let tok_line = line;
            // Column of this token's first character (P0b).
            let (tok_col, tok_col_byte) = col_at(src, pos);
            match c {
                b'0'..=b'9' => {
                    let start = pos;
                    // Radix literals: 0x../0o../0b.. (underscores allowed).
                    if bytes[pos] == b'0'
                        && pos + 1 < n
                        && matches!(bytes[pos + 1], b'x' | b'X' | b'o' | b'O' | b'b' | b'B')
                    {
                        let radix = match bytes[pos + 1] {
                            b'x' | b'X' => 16,
                            b'o' | b'O' => 8,
                            _ => 2,
                        };
                        pos += 2;
                        let dstart = pos;
                        while pos < n
                            && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_')
                        {
                            pos += 1;
                        }
                        let digits: String =
                            src[dstart..pos].chars().filter(|&ch| ch != '_').collect();
                        if digits.is_empty() {
                            return Err(err_at(tok_line, tok_col, src, "radix literal with no digits"));
                        }
                        let v = match i64::from_str_radix(&digits, radix) {
                            Ok(v) => v,
                            Err(_) => {
                                if digits.chars().any(|c| c.to_digit(radix).is_none()) {
                                    return Err(err_at(tok_line, tok_col, src, "bad radix literal"));
                                }
                                // Valid digits past i64: arbitrary
                                // precision literal, decimal-normalised.
                                out.push(Token { tok: Tok::BigInt(radix_digits_to_decimal(&digits, radix)), line: tok_line, col: tok_col, col_byte: tok_col_byte });
                                continue
                            }
                        };
                        out.push(Token { tok: Tok::Int(v), line: tok_line, col: tok_col, col_byte: tok_col_byte });
                        continue;
                    }
                    let mut is_float = false;
                    while pos < n && (bytes[pos].is_ascii_digit() || bytes[pos] == b'_') {
                        pos += 1;
                    }
                    if pos < n && bytes[pos] == b'.' && pos + 1 < n && bytes[pos + 1].is_ascii_digit() {
                        is_float = true;
                        pos += 1;
                        while pos < n && bytes[pos].is_ascii_digit() {
                            pos += 1;
                        }
                    }
                    let text: String = src[start..pos].chars().filter(|&ch| ch != '_').collect();
                    if is_float {
                        let v: f64 = text.parse().map_err(|_| err(tok_line, "bad float literal"))?;
                        out.push(Token { tok: Tok::Float(v), line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        match text.parse::<i64>() {
                            Ok(v) => out.push(Token { tok: Tok::Int(v), line: tok_line, col: tok_col, col_byte: tok_col_byte }),
                            // Past i64: arbitrary-precision literal,
                            // decimal text preserved.
                            Err(_) => out.push(Token { tok: Tok::BigInt(text), line: tok_line, col: tok_col, col_byte: tok_col_byte }),
                        }
                    }
                }
                b'"' | b'\'' => {
                    let s = scan_str_literal(src, bytes, &mut pos, &mut line, false)?;
                    out.push(Token { tok: Tok::Str(s), line: tok_line, col: tok_col, col_byte: tok_col_byte });
                }
                b'A'..=b'Z' | b'a'..=b'z' | b'_' => {
                    let start = pos;
                    while pos < n && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
                        pos += 1;
                    }
                    let word = &src[start..pos];
                    // String prefixes: a prefix word immediately
                    // followed by a quote is a literal, exactly as
                    // in CPython (f, r, b, rb, br, rf, fr; any
                    // case). A prefix word NOT followed by a quote
                    // is an ordinary identifier.
                    let lower = word.to_ascii_lowercase();
                    if matches!(lower.as_str(), "f" | "r" | "b" | "rb" | "br" | "rf" | "fr")
                        && pos < n
                        && (bytes[pos] == b'"' || bytes[pos] == b'\'')
                    {
                        match lower.as_str() {
                            "f" => {
                                let parts = scan_fstring(src, bytes, &mut pos, tok_line, false)?;
                                out.push(Token { tok: Tok::FStr(parts), line: tok_line, col: tok_col, col_byte: tok_col_byte });
                            }
                            "rf" | "fr" => {
                                let parts = scan_fstring(src, bytes, &mut pos, tok_line, true)?;
                                out.push(Token { tok: Tok::FStr(parts), line: tok_line, col: tok_col, col_byte: tok_col_byte });
                            }
                            "r" => {
                                let s = scan_str_literal(src, bytes, &mut pos, &mut line, true)?;
                                out.push(Token { tok: Tok::Str(s), line: tok_line, col: tok_col, col_byte: tok_col_byte });
                            }
                            "b" => {
                                let bs =
                                    scan_bytes_literal(src, bytes, &mut pos, &mut line, false)?;
                                out.push(Token { tok: Tok::Bytes(bs), line: tok_line, col: tok_col, col_byte: tok_col_byte });
                            }
                            _ => {
                                // rb / br
                                let bs =
                                    scan_bytes_literal(src, bytes, &mut pos, &mut line, true)?;
                                out.push(Token { tok: Tok::Bytes(bs), line: tok_line, col: tok_col, col_byte: tok_col_byte });
                            }
                        }
                        continue;
                    }
                    let tok = match word {
                        "def" => Tok::Def,
                        "return" => Tok::Return,
                        "if" => Tok::If,
                        "elif" => Tok::Elif,
                        "else" => Tok::Else,
                        "while" => Tok::While,
                        "for" => Tok::For,
                        "in" => Tok::In,
                        "is" => Tok::Is,
                        "break" => Tok::Break,
                        "continue" => Tok::Continue,
                        "pass" => Tok::Pass,
                        "and" => Tok::And,
                        "or" => Tok::Or,
                        "not" => Tok::Not,
                        "True" => Tok::True,
                        "False" => Tok::False,
                        "None" => Tok::NoneLit,
                        "lambda" => Tok::Lambda,
                        "class" => Tok::Class,
                        "try" => Tok::Try,
                        "except" => Tok::Except,
                        "finally" => Tok::Finally,
                        "raise" => Tok::Raise,
                        "with" => Tok::With,
                        "import" => Tok::Import,
                        "from" => Tok::From,
                        "yield" => Tok::Yield,
                        "async" => Tok::Async,
                        "await" => Tok::Await,
                        _ => Tok::Ident(word.to_string()),
                    };
                    out.push(Token { tok, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                }
                b'(' => {
                    depth_bracket += 1;
                    pos += 1;
                    out.push(Token { tok: Tok::LParen, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                }
                b')' => {
                    depth_bracket -= 1;
                    pos += 1;
                    out.push(Token { tok: Tok::RParen, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                }
                b'[' => {
                    depth_bracket += 1;
                    pos += 1;
                    out.push(Token { tok: Tok::LBracket, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                }
                b']' => {
                    depth_bracket -= 1;
                    pos += 1;
                    out.push(Token { tok: Tok::RBracket, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                }
                b'{' => {
                    depth_bracket += 1;
                    pos += 1;
                    out.push(Token { tok: Tok::LBrace, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                }
                b'}' => {
                    depth_bracket -= 1;
                    pos += 1;
                    out.push(Token { tok: Tok::RBrace, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                }
                b',' => {
                    pos += 1;
                    out.push(Token { tok: Tok::Comma, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                }
                b':' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::Walrus, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Colon, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    }
                }
                b'.' => {
                    pos += 1;
                    out.push(Token { tok: Tok::Dot, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                }
                b'+' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::PlusEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Plus, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    }
                }
                b'-' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::MinusEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Minus, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    }
                }
                b'*' => {
                    if peek(bytes, pos, b'*') {
                        if pos + 2 < n && bytes[pos + 2] == b'=' {
                            pos += 3;
                            out.push(Token { tok: Tok::PowEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                        } else {
                            pos += 2;
                            out.push(Token { tok: Tok::Pow, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                        }
                    } else if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::StarEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Star, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    }
                }
                b'/' => {
                    if peek(bytes, pos, b'/') {
                        if pos + 2 < n && bytes[pos + 2] == b'=' {
                            pos += 3;
                            out.push(Token { tok: Tok::FloorDivEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                        } else {
                            pos += 2;
                            out.push(Token { tok: Tok::FloorDiv, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                        }
                    } else if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::SlashEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Slash, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    }
                }
                b'%' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::PercentEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Percent, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    }
                }
                b'&' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::BitAndEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::BitAnd, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    }
                }
                b'|' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::BitOrEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::BitOr, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    }
                }
                b'^' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::BitXorEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::BitXor, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    }
                }
                b'~' => {
                    pos += 1;
                    out.push(Token { tok: Tok::Tilde, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                }
                b'=' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::EqEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Assign, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    }
                }
                b'!' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::NotEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        return Err(err_at(tok_line, tok_col, src, "bare ! is not valid; use not"));
                    }
                }
                b'<' => {
                    if peek(bytes, pos, b'<') {
                        if pos + 2 < n && bytes[pos + 2] == b'=' {
                            pos += 3;
                            out.push(Token { tok: Tok::ShlEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                        } else {
                            pos += 2;
                            out.push(Token { tok: Tok::Shl, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                        }
                    } else if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::LtEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Lt, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    }
                }
                b'>' => {
                    if peek(bytes, pos, b'>') {
                        if pos + 2 < n && bytes[pos + 2] == b'=' {
                            pos += 3;
                            out.push(Token { tok: Tok::ShrEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                        } else {
                            pos += 2;
                            out.push(Token { tok: Tok::Shr, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                        }
                    } else if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::GtEq, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Gt, line: tok_line, col: tok_col, col_byte: tok_col_byte });
                    }
                }
                _ => {
                    return Err(err_at(tok_line, tok_col, src,
                        &format!("unexpected character {:?} in v0.1 subset", c as char),
                    ))
                }
            }
        }
        if pos >= n {
            let (tok_col, tok_col_byte) = col_at(src, pos);
            out.push(Token { tok: Tok::Newline, line, col: tok_col, col_byte: tok_col_byte });
            break;
        }
    }
    let (tok_col, tok_col_byte) = col_at(src, pos);
    while indents.len() > 1 {
        indents.pop();
        out.push(Token { tok: Tok::Dedent, line, col: tok_col, col_byte: tok_col_byte });
    }
    out.push(Token { tok: Tok::Eof, line, col: tok_col, col_byte: tok_col_byte });
    Ok(out)
}

/// Scan a string literal body; `*pos` is at the opening quote on
/// entry and just past the closing quote on return. Handles the
/// triple-quoted form (which may span lines; `*line` tracks the
/// running line counter). With `raw` set (an r"" literal) no
/// escape processing happens at all: a backslash is an ordinary
/// character, except that a backslash immediately before the
/// quote still keeps the quote from terminating the literal
/// (CPython's raw-string tokenising rule -- the backslash stays
/// in the value).
fn scan_str_literal(
    src: &str,
    bytes: &[u8],
    pos: &mut usize,
    line: &mut u32,
    raw: bool,
) -> Result<String, Diagnostic> {
    let n = bytes.len();
    let start_line = *line;
    // Column of the opening quote, for diagnostics.
    let start_col = col_at(src, *pos).0;
    let quote = bytes[*pos];
    let triple = *pos + 2 < n && bytes[*pos + 1] == quote && bytes[*pos + 2] == quote;
    *pos += if triple { 3 } else { 1 };
    let mut s = String::new();
    loop {
        if *pos >= n {
            return Err(err_at(start_line, start_col, src,
                if triple {
                    "unterminated triple-quoted string"
                } else {
                    "unterminated string literal"
                },
            ));
        }
        let ch = bytes[*pos];
        if triple
            && ch == quote
            && *pos + 2 < n
            && bytes[*pos + 1] == quote
            && bytes[*pos + 2] == quote
        {
            *pos += 3;
            break;
        }
        if !triple && ch == quote {
            *pos += 1;
            break;
        }
        if !triple && ch == b'\n' {
            return Err(err_at(start_line, start_col, src, "unterminated string literal"));
        }
        if ch == b'\\' {
            if raw {
                // Keep the backslash; a following quote is part of
                // the value and does not terminate the literal.
                s.push('\\');
                *pos += 1;
                if *pos >= n {
                    return Err(err_at(start_line, start_col, src, "unterminated string literal"));
                }
                let ch_len = utf8_len(bytes[*pos]);
                s.push_str(&src[*pos..*pos + ch_len]);
                *pos += ch_len;
                continue;
            }
            *pos += 1;
            if *pos >= n {
                return Err(err_at(start_line, start_col, src, "unterminated escape"));
            }
            match bytes[*pos] {
                b'n' => s.push('\n'),
                b'r' => s.push('\r'),
                b't' => s.push('\t'),
                b'\\' => s.push('\\'),
                b'"' => s.push('"'),
                b'\'' => s.push('\''),
                other => {
                    return Err(err_at(start_line, start_col, src,
                        &format!("unsupported escape \\{} in string", other as char),
                    ))
                }
            }
            *pos += 1;
            continue;
        }
        if ch == b'\n' {
            *line += 1;
        }
        let ch_len = utf8_len(ch);
        s.push_str(&src[*pos..*pos + ch_len]);
        *pos += ch_len;
    }
    Ok(s)
}

/// Scan a bytes literal body; same entry/exit contract as
/// scan_str_literal. Bytes literals are ASCII-only (CPython's
/// rule): a non-ASCII character must be written as a \xNN
/// escape. Supported escapes are CPython's full bytes set:
/// the string set (\n \r \t \\ \" \') plus \a \b \f \v,
/// octal escapes (\0 through \377, one to three digits) and
/// \xNN; anything else is a loud error. With `raw` set (rb"")
/// backslashes are ordinary bytes, with the same
/// backslash-before-quote tokenising rule as raw strings.
fn scan_bytes_literal(
    src: &str,
    bytes: &[u8],
    pos: &mut usize,
    line: &mut u32,
    raw: bool,
) -> Result<Vec<u8>, Diagnostic> {
    let n = bytes.len();
    let start_line = *line;
    // Column of the opening quote, for diagnostics.
    let start_col = col_at(src, *pos).0;
    let quote = bytes[*pos];
    let triple = *pos + 2 < n && bytes[*pos + 1] == quote && bytes[*pos + 2] == quote;
    *pos += if triple { 3 } else { 1 };
    let mut out: Vec<u8> = Vec::new();
    loop {
        if *pos >= n {
            return Err(err_at(start_line, start_col, src,
                if triple {
                    "unterminated triple-quoted bytes literal"
                } else {
                    "unterminated bytes literal"
                },
            ));
        }
        let ch = bytes[*pos];
        if triple
            && ch == quote
            && *pos + 2 < n
            && bytes[*pos + 1] == quote
            && bytes[*pos + 2] == quote
        {
            *pos += 3;
            break;
        }
        if !triple && ch == quote {
            *pos += 1;
            break;
        }
        if !triple && ch == b'\n' {
            return Err(err_at(start_line, start_col, src, "unterminated bytes literal"));
        }
        if ch == b'\\' {
            if raw {
                out.push(b'\\');
                *pos += 1;
                if *pos >= n {
                    return Err(err_at(start_line, start_col, src, "unterminated bytes literal"));
                }
                let b = bytes[*pos];
                if b >= 0x80 {
                    return Err(err_at(start_line, start_col, src,
                        "bytes literals may only contain ASCII characters; use \\xNN escapes",
                    ));
                }
                out.push(b);
                *pos += 1;
                continue;
            }
            *pos += 1;
            if *pos >= n {
                return Err(err_at(start_line, start_col, src, "unterminated escape in bytes literal"));
            }
            match bytes[*pos] {
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'\\' => out.push(b'\\'),
                b'"' => out.push(b'"'),
                b'\'' => out.push(b'\''),
                b'a' => out.push(7),
                b'b' => out.push(8),
                b'f' => out.push(12),
                b'v' => out.push(11),
                b'0'..=b'7' => {
                    // Octal escape: one to three digits, and
                    // the value must fit in one byte (CPython's
                    // rule for bytes literals; \0 is the
                    // one-digit case).
                    let mut v: u32 = 0;
                    let mut ndig = 0usize;
                    while ndig < 3
                        && *pos < n
                        && (b'0'..=b'7').contains(&bytes[*pos])
                    {
                        v = v * 8 + u32::from(bytes[*pos] - b'0');
                        *pos += 1;
                        ndig += 1;
                    }
                    if v > 255 {
                        return Err(err(
                            start_line,
                            "octal escape in bytes literal is larger than \\377 (one byte)",
                        ));
                    }
                    out.push(v as u8);
                    // Leave *pos on the last digit consumed;
                    // the common step after the match moves
                    // past it (the same convention as \xNN).
                    *pos -= 1;
                }
                b'x' => {
                    if *pos + 2 >= n {
                        return Err(err_at(start_line, start_col, src, "truncated \\xNN escape in bytes literal"));
                    }
                    let h = (bytes[*pos + 1] as char).to_digit(16);
                    let l = (bytes[*pos + 2] as char).to_digit(16);
                    match (h, l) {
                        (Some(h), Some(l)) => out.push((h * 16 + l) as u8),
                        _ => {
                            return Err(err_at(start_line, start_col, src,
                                "bad \\xNN escape in bytes literal (two hex digits required)",
                            ))
                        }
                    }
                    *pos += 2;
                }
                other => {
                    return Err(err_at(start_line, start_col, src,
                        &format!("unsupported escape \\{} in bytes literal", other as char),
                    ))
                }
            }
            *pos += 1;
            continue;
        }
        if ch >= 0x80 {
            return Err(err_at(start_line, start_col, src,
                "bytes literals may only contain ASCII characters; use \\xNN escapes",
            ));
        }
        if ch == b'\n' {
            *line += 1;
        }
        out.push(ch);
        *pos += 1;
    }
    Ok(out)
}

/// Scan an f-string body; `*pos` is at the opening quote on entry and just
/// past the closing quote on return. A replacement field may carry a
/// conversion (`{x!r}`), a debug `=` marker (`{x=}`) and/or a format
/// spec (`{x:.2f}`); the field is split by `split_fstr_field`. With
/// `raw` set (rf""), backslashes in literal text are ordinary
/// characters.
fn scan_fstring(
    src: &str,
    bytes: &[u8],
    pos: &mut usize,
    line: u32,
    raw: bool,
) -> Result<Vec<FPart>, Diagnostic> {
    let quote = bytes[*pos];
    // Column of the opening quote, for diagnostics.
    let start_col = col_at(src, *pos).0;
    *pos += 1;
    scan_fstring_body(src, bytes, pos, line, raw, quote, false, start_col)
}

/// Scan an f-string *fragment*: text in f-string syntax without
/// surrounding quotes, ending at the end of the input. The parser
/// uses this to resolve a format spec, which is itself a mini
/// f-string (literal text plus nested replacement fields).
pub fn scan_fstring_fragment(src: &str, line: u32) -> Result<Vec<FPart>, Diagnostic> {
    let bytes = src.as_bytes();
    let mut pos = 0usize;
    scan_fstring_body(src, bytes, &mut pos, line, false, 0, true, col_at(src, 0).0)
}

/// Shared scanner behind `scan_fstring` and
/// `scan_fstring_fragment`. In fragment mode there is no quote
/// and no opening-quote skip: the scan ends at end of input.
fn scan_fstring_body(
    src: &str,
    bytes: &[u8],
    pos: &mut usize,
    line: u32,
    raw: bool,
    quote: u8,
    fragment: bool,
    start_col: u32,
) -> Result<Vec<FPart>, Diagnostic> {
    let n = bytes.len();
    let mut parts: Vec<FPart> = Vec::new();
    let mut lit = String::new();
    loop {
        if *pos >= n {
            if fragment {
                break;
            }
            return Err(err_at(line, start_col, src, "unterminated f-string literal"));
        }
        if !fragment && bytes[*pos] == b'\n' {
            return Err(err_at(line, start_col, src, "unterminated f-string literal"));
        }
        let c = bytes[*pos];
        if !fragment && c == quote {
            *pos += 1;
            break;
        }
        if c == b'\\' {
            *pos += 1;
            if *pos >= n {
                return Err(err_at(line, start_col, src, "unterminated escape in f-string"));
            }
            if raw {
                // Raw f-string: the backslash and the following
                // character are both literal text. Braces are the
                // exception: a backslash does not escape them in
                // CPython either, so leave a following brace to
                // the interpolation logic below.
                lit.push('\\');
                if bytes[*pos] == b'{' || bytes[*pos] == b'}' {
                    continue;
                }
                let ch_len = utf8_len(bytes[*pos]);
                lit.push_str(&src[*pos..*pos + ch_len]);
                *pos += ch_len;
                continue;
            }
            match bytes[*pos] {
                b'n' => lit.push('\n'),
                b'r' => lit.push('\r'),
                b't' => lit.push('\t'),
                b'\\' => lit.push('\\'),
                b'"' => lit.push('"'),
                b'\'' => lit.push('\''),
                other => {
                    return Err(err_at(line, start_col, src,
                        &format!("unsupported escape \\{} in f-string", other as char),
                    ))
                }
            }
            *pos += 1;
            continue;
        }
        if c == b'{' {
            if *pos + 1 < n && bytes[*pos + 1] == b'{' {
                lit.push('{');
                *pos += 2;
                continue;
            }
            // Capture the interpolated expression source.
            let start = *pos + 1;
            let mut i = start;
            let mut depth = 1usize; // the f-string brace itself
            let mut paren = 0usize;
            let mut bracket = 0usize;
            // The first top-level field marker: the byte offset
            // (relative to `start`) and kind of a ':' opening the
            // format spec, a '!' opening a conversion, or the '='
            // of a debug field. Once a marker is seen, the rest
            // of the field is conversion/spec text, in which ':'
            // and '!' are ordinary characters (a fill character
            // may be '!', and a nested field's own markers sit at
            // brace depth 2 or more).
            let mut marker: Option<(usize, u8)> = None;
            while i < n && depth > 0 {
                let ch = bytes[i];
                if ch == b'\n' {
                    return Err(err_at(line, start_col, src, "unterminated {...} in f-string"));
                }
                if ch == b'"' || ch == b'\'' {
                    // Skip a string literal inside the expression.
                    let q = ch;
                    i += 1;
                    while i < n && bytes[i] != q {
                        if bytes[i] == b'\\' {
                            i += 1;
                        }
                        i += 1;
                    }
                    if i >= n {
                        return Err(err_at(line, start_col, src, "unterminated string inside f-string expression"));
                    }
                    i += 1;
                    continue;
                }
                // Format spec / conversion / debug markers only
                // count at the expression's top level (not inside
                // (), [], or {}), and only the first one counts.
                if depth == 1 && paren == 0 && bracket == 0 && marker.is_none() {
                    // A ':' at the field's top level always
                    // opens the format spec — including when
                    // the spec itself starts with '=' (the
                    // '=' alignment, c24/n05: `{-42:=8}`,
                    // `{'a':=5}`). A walrus can only appear
                    // inside parentheses at this level in
                    // CPython too, where paren > 0 already
                    // excludes the marker check.
                    if ch == b':' {
                        marker = Some((i - start, b':'));
                    } else if ch == b'!'
                        && i + 1 < n
                        && bytes[i + 1].is_ascii_alphabetic()
                    {
                        marker = Some((i - start, b'!'));
                    } else if ch == b'=' {
                        // Debug marker: a bare '=' whose next
                        // non-space byte ends the field or opens a
                        // conversion/spec. Comparisons are
                        // excluded: '==' (the next byte is '='), and
                        // '<=', '>=', '!=' and the walrus ':=' (the
                        // previous byte is an operator or ':').
                        let prev_ok = i > start
                            && !matches!(bytes[i - 1], b'=' | b'!' | b'<' | b'>' | b':');
                        let mut j = i + 1;
                        while j < n && bytes[j] == b' ' {
                            j += 1;
                        }
                        let next_ok =
                            j < n && matches!(bytes[j], b'}' | b'!' | b':');
                        if prev_ok && next_ok {
                            marker = Some((i - start, b'='));
                        }
                    }
                }
                match ch {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    b'(' => paren += 1,
                    b')' => paren = paren.saturating_sub(1),
                    b'[' => bracket += 1,
                    b']' => bracket = bracket.saturating_sub(1),
                    _ => {}
                }
                i += 1;
            }
            if depth != 0 {
                return Err(err_at(line, start_col, src, "unterminated {...} in f-string"));
            }
            if !lit.is_empty() {
                parts.push(FPart::Lit(std::mem::take(&mut lit)));
            }
            parts.push(split_fstr_field(&src[start..i - 1], marker, line)?);
            *pos = i;
            continue;
        }
        if c == b'}' {
            if *pos + 1 < n && bytes[*pos + 1] == b'}' {
                lit.push('}');
                *pos += 2;
                continue;
            }
            return Err(err_at(line, start_col, src, "single '}' in f-string; use '}}' for a literal brace"));
        }
        let ch_len = utf8_len(c);
        lit.push_str(&src[*pos..*pos + ch_len]);
        *pos += ch_len;
    }
    if !lit.is_empty() {
        parts.push(FPart::Lit(lit));
    }
    Ok(parts)
}

/// Split one scanned replacement field into its expression
/// source, conversion, debug prefix and format spec, given the
/// first top-level marker found while scanning (`:` spec, `!`
/// conversion, `=` debug). Malformed shapes are loud lexer
/// errors, mirroring CPython's SyntaxErrors.
fn split_fstr_field(
    field: &str,
    marker: Option<(usize, u8)>,
    line: u32,
) -> Result<FPart, Diagnostic> {
    let (at, kind) = match marker {
        Some(m) => m,
        None => {
            let expr_src = field.trim().to_string();
            if expr_src.is_empty() {
                return Err(err(line, "empty {} in f-string"));
            }
            return Ok(FPart::Expr(expr_src));
        }
    };
    // `at` is a byte index into `field`: markers are ASCII
    // bytes recorded outside strings, and multibyte characters
    // before the marker count identically on both sides, so
    // every slice here lands on a char boundary.
    let expr_src = field[..at].trim().to_string();
    if expr_src.is_empty() {
        return Err(err(line, "empty {} in f-string"));
    }
    // Parse a conversion tail (the text just after a '!'):
    // exactly one of r/s/a, then either the end of the field
    // or ':' + spec.
    fn parse_conv_tail(
        tail: &str,
        line: u32,
    ) -> Result<(char, Option<String>), Diagnostic> {
        let tb = tail.as_bytes();
        if tb.is_empty() || !matches!(tb[0], b'r' | b's' | b'a') {
            return Err(err(
                line,
                "invalid f-string conversion (only !r, !s and !a are supported)",
            ));
        }
        let conv = tb[0] as char;
        let rest = &tail[1..];
        if rest.is_empty() {
            Ok((conv, None))
        } else if let Some(spec) = rest.strip_prefix(':') {
            Ok((conv, Some(spec.to_string())))
        } else {
            Err(err(line, "expected ':' or '}' after f-string conversion"))
        }
    }
    match kind {
        b':' => Ok(FPart::Fmt {
            src: expr_src,
            conv: None,
            debug: None,
            spec: Some(field[at + 1..].to_string()),
        }),
        b'!' => {
            let (conv, spec) = parse_conv_tail(&field[at + 1..], line)?;
            Ok(FPart::Fmt {
                src: expr_src,
                conv: Some(conv),
                debug: None,
                spec,
            })
        }
        _ => {
            // Debug '=' marker. The literal prefix is the field
            // text up to the start of the conversion/spec tail,
            // verbatim: CPython keeps the expression text, the
            // '=' and any spaces that follow it.
            let after = &field[at + 1..];
            let tail = after.trim_start_matches(' ');
            let debug = field[..field.len() - tail.len()].to_string();
            if tail.is_empty() {
                Ok(FPart::Fmt {
                    src: expr_src,
                    conv: None,
                    debug: Some(debug),
                    spec: None,
                })
            } else if let Some(ct) = tail.strip_prefix('!') {
                let (conv, spec) = parse_conv_tail(ct, line)?;
                Ok(FPart::Fmt {
                    src: expr_src,
                    conv: Some(conv),
                    debug: Some(debug),
                    spec,
                })
            } else if let Some(spec) = tail.strip_prefix(':') {
                Ok(FPart::Fmt {
                    src: expr_src,
                    conv: None,
                    debug: Some(debug),
                    spec: Some(spec.to_string()),
                })
            } else {
                Err(err(
                    line,
                    "expected '!', ':' or '}' after '=' in f-string debug field",
                ))
            }
        }
    }
}

fn peek(bytes: &[u8], pos: usize, want: u8) -> bool {
    pos + 1 < bytes.len() && bytes[pos + 1] == want
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else {
        4
    }
}
