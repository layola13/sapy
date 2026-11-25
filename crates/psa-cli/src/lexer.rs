//! Indentation-aware lexer for the .psa subset.
//!
//! Produces Newline / Indent / Dedent tokens Python-style. Anything outside
//! the supported subset (tabs in indentation, f-string format specs, ...)
//! is a loud error, never a silent skip.

/// One part of an f-string token: literal text, or the source text of an
/// interpolated expression (the parser re-parses it).
#[derive(Debug, Clone, PartialEq)]
pub enum FPart {
    Lit(String),
    Expr(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),
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
}

fn err(line: u32, msg: &str) -> String {
    format!("psa lexer error at line {}: {}", line, msg)
}

pub fn lex(src: &str) -> Result<Vec<Token>, String> {
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
            return Err(err(line, "tab indentation is not supported in v0.1; use spaces"));
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
        // Indentation stack.
        let cur = *indents.last().unwrap();
        if indent > cur {
            indents.push(indent);
            out.push(Token { tok: Tok::Indent, line });
        } else if indent < cur {
            while *indents.last().unwrap() > indent {
                indents.pop();
                out.push(Token { tok: Tok::Dedent, line });
            }
            if *indents.last().unwrap() != indent {
                return Err(err(line, "inconsistent dedent: indentation matches no outer level"));
            }
        }
        // Tokenize the rest of the line.
        let mut depth_paren = 0i32;
        loop {
            if pos >= n {
                break;
            }
            let c = bytes[pos];
            if c == b'\n' && depth_paren == 0 {
                pos += 1;
                line += 1;
                out.push(Token { tok: Tok::Newline, line: line - 1 });
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
                            return Err(err(tok_line, "radix literal with no digits"));
                        }
                        let v = i64::from_str_radix(&digits, radix)
                            .map_err(|_| err(tok_line, "bad radix literal or out of i64 range"))?;
                        out.push(Token { tok: Tok::Int(v), line: tok_line });
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
                        out.push(Token { tok: Tok::Float(v), line: tok_line });
                    } else {
                        let v: i64 = text.parse().map_err(|_| err(tok_line, "integer literal out of i64 range"))?;
                        out.push(Token { tok: Tok::Int(v), line: tok_line });
                    }
                }
                b'"' | b'\'' => {
                    let quote = c;
                    // Triple-quoted string: may span lines, ends at the
                    // matching triple quote.
                    if pos + 2 < n && bytes[pos + 1] == quote && bytes[pos + 2] == quote {
                        pos += 3;
                        let mut s = String::new();
                        loop {
                            if pos >= n {
                                return Err(err(tok_line, "unterminated triple-quoted string"));
                            }
                            if bytes[pos] == quote
                                && pos + 2 < n
                                && bytes[pos + 1] == quote
                                && bytes[pos + 2] == quote
                            {
                                pos += 3;
                                break;
                            }
                            let ch = bytes[pos];
                            if ch == b'\\' {
                                pos += 1;
                                if pos >= n {
                                    return Err(err(tok_line, "unterminated escape"));
                                }
                                match bytes[pos] {
                                    b'n' => s.push('\n'),
                                    b't' => s.push('\t'),
                                    b'\\' => s.push('\\'),
                                    b'"' => s.push('"'),
                                    b'\'' => s.push('\''),
                                    other => {
                                        return Err(err(
                                            tok_line,
                                            &format!(
                                                "unsupported escape \\{} in string",
                                                other as char
                                            ),
                                        ))
                                    }
                                }
                                pos += 1;
                                continue;
                            }
                            if ch == b'\n' {
                                line += 1;
                            }
                            let ch_len = utf8_len(bytes[pos]);
                            s.push_str(&src[pos..pos + ch_len]);
                            pos += ch_len;
                        }
                        out.push(Token { tok: Tok::Str(s), line: tok_line });
                        continue;
                    }
                    pos += 1;
                    let mut s = String::new();
                    loop {
                        if pos >= n || bytes[pos] == b'\n' {
                            return Err(err(tok_line, "unterminated string literal"));
                        }
                        let ch = bytes[pos];
                        if ch == quote {
                            pos += 1;
                            break;
                        }
                        if ch == b'\\' {
                            pos += 1;
                            if pos >= n {
                                return Err(err(tok_line, "unterminated escape"));
                            }
                            match bytes[pos] {
                                b'n' => s.push('\n'),
                                b't' => s.push('\t'),
                                b'\\' => s.push('\\'),
                                b'"' => s.push('"'),
                                b'\'' => s.push('\''),
                                other => {
                                    return Err(err(
                                        tok_line,
                                        &format!("unsupported escape \\{} in v0.1", other as char),
                                    ))
                                }
                            }
                            pos += 1;
                            continue;
                        }
                        // Copy one UTF-8 char.
                        let ch_len = utf8_len(bytes[pos]);
                        s.push_str(&src[pos..pos + ch_len]);
                        pos += ch_len;
                    }
                    out.push(Token { tok: Tok::Str(s), line: tok_line });
                }
                b'A'..=b'Z' | b'a'..=b'z' | b'_' => {
                    let start = pos;
                    while pos < n && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
                        pos += 1;
                    }
                    let word = &src[start..pos];
                    // f-string: `f` immediately followed by a quote.
                    if (word == "f" || word == "F")
                        && pos < n
                        && (bytes[pos] == b'"' || bytes[pos] == b'\'')
                    {
                        let parts = scan_fstring(src, bytes, &mut pos, tok_line)?;
                        out.push(Token { tok: Tok::FStr(parts), line: tok_line });
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
                    out.push(Token { tok, line: tok_line });
                }
                b'(' => {
                    depth_paren += 1;
                    pos += 1;
                    out.push(Token { tok: Tok::LParen, line: tok_line });
                }
                b')' => {
                    depth_paren -= 1;
                    pos += 1;
                    out.push(Token { tok: Tok::RParen, line: tok_line });
                }
                b'[' => {
                    pos += 1;
                    out.push(Token { tok: Tok::LBracket, line: tok_line });
                }
                b']' => {
                    pos += 1;
                    out.push(Token { tok: Tok::RBracket, line: tok_line });
                }
                b'{' => {
                    pos += 1;
                    out.push(Token { tok: Tok::LBrace, line: tok_line });
                }
                b'}' => {
                    pos += 1;
                    out.push(Token { tok: Tok::RBrace, line: tok_line });
                }
                b',' => {
                    pos += 1;
                    out.push(Token { tok: Tok::Comma, line: tok_line });
                }
                b':' => {
                    pos += 1;
                    out.push(Token { tok: Tok::Colon, line: tok_line });
                }
                b'.' => {
                    pos += 1;
                    out.push(Token { tok: Tok::Dot, line: tok_line });
                }
                b'+' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::PlusEq, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Plus, line: tok_line });
                    }
                }
                b'-' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::MinusEq, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Minus, line: tok_line });
                    }
                }
                b'*' => {
                    if peek(bytes, pos, b'*') {
                        if pos + 2 < n && bytes[pos + 2] == b'=' {
                            pos += 3;
                            out.push(Token { tok: Tok::PowEq, line: tok_line });
                        } else {
                            pos += 2;
                            out.push(Token { tok: Tok::Pow, line: tok_line });
                        }
                    } else if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::StarEq, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Star, line: tok_line });
                    }
                }
                b'/' => {
                    if peek(bytes, pos, b'/') {
                        if pos + 2 < n && bytes[pos + 2] == b'=' {
                            pos += 3;
                            out.push(Token { tok: Tok::FloorDivEq, line: tok_line });
                        } else {
                            pos += 2;
                            out.push(Token { tok: Tok::FloorDiv, line: tok_line });
                        }
                    } else if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::SlashEq, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Slash, line: tok_line });
                    }
                }
                b'%' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::PercentEq, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Percent, line: tok_line });
                    }
                }
                b'&' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::BitAndEq, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::BitAnd, line: tok_line });
                    }
                }
                b'|' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::BitOrEq, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::BitOr, line: tok_line });
                    }
                }
                b'^' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::BitXorEq, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::BitXor, line: tok_line });
                    }
                }
                b'~' => {
                    pos += 1;
                    out.push(Token { tok: Tok::Tilde, line: tok_line });
                }
                b'=' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::EqEq, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Assign, line: tok_line });
                    }
                }
                b'!' => {
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::NotEq, line: tok_line });
                    } else {
                        return Err(err(tok_line, "bare ! is not valid; use not"));
                    }
                }
                b'<' => {
                    if peek(bytes, pos, b'<') {
                        if pos + 2 < n && bytes[pos + 2] == b'=' {
                            pos += 3;
                            out.push(Token { tok: Tok::ShlEq, line: tok_line });
                        } else {
                            pos += 2;
                            out.push(Token { tok: Tok::Shl, line: tok_line });
                        }
                    } else if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::LtEq, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Lt, line: tok_line });
                    }
                }
                b'>' => {
                    if peek(bytes, pos, b'>') {
                        if pos + 2 < n && bytes[pos + 2] == b'=' {
                            pos += 3;
                            out.push(Token { tok: Tok::ShrEq, line: tok_line });
                        } else {
                            pos += 2;
                            out.push(Token { tok: Tok::Shr, line: tok_line });
                        }
                    } else if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::GtEq, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Gt, line: tok_line });
                    }
                }
                _ => {
                    return Err(err(
                        tok_line,
                        &format!("unexpected character {:?} in v0.1 subset", c as char),
                    ))
                }
            }
        }
        if pos >= n {
            out.push(Token { tok: Tok::Newline, line });
            break;
        }
    }
    while indents.len() > 1 {
        indents.pop();
        out.push(Token { tok: Tok::Dedent, line });
    }
    out.push(Token { tok: Tok::Eof, line });
    Ok(out)
}

/// Scan an f-string body; `*pos` is at the opening quote on entry and just
/// past the closing quote on return. Format specs (`{x:.2f}`) and
/// conversions (`{x!r}`) are loud errors in D0.
fn scan_fstring(
    src: &str,
    bytes: &[u8],
    pos: &mut usize,
    line: u32,
) -> Result<Vec<FPart>, String> {
    let n = bytes.len();
    let quote = bytes[*pos];
    *pos += 1;
    let mut parts: Vec<FPart> = Vec::new();
    let mut lit = String::new();
    loop {
        if *pos >= n || bytes[*pos] == b'\n' {
            return Err(err(line, "unterminated f-string literal"));
        }
        let c = bytes[*pos];
        if c == quote {
            *pos += 1;
            break;
        }
        if c == b'\\' {
            *pos += 1;
            if *pos >= n {
                return Err(err(line, "unterminated escape in f-string"));
            }
            match bytes[*pos] {
                b'n' => lit.push('\n'),
                b't' => lit.push('\t'),
                b'\\' => lit.push('\\'),
                b'"' => lit.push('"'),
                b'\'' => lit.push('\''),
                other => {
                    return Err(err(
                        line,
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
            while i < n && depth > 0 {
                let ch = bytes[i];
                if ch == b'\n' {
                    return Err(err(line, "unterminated {...} in f-string"));
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
                        return Err(err(line, "unterminated string inside f-string expression"));
                    }
                    i += 1;
                    continue;
                }
                // Format spec / conversion markers only count at the
                // expression's top level (not inside (), [], or {}).
                if depth == 1 && paren == 0 && bracket == 0 {
                    if ch == b':' && !(i + 1 < n && bytes[i + 1] == b'=') {
                        return Err(err(
                            line,
                            "f-string format specs ({x:...}) are not supported in D0",
                        ));
                    }
                    if ch == b'!'
                        && i + 1 < n
                        && bytes[i + 1].is_ascii_alphabetic()
                    {
                        return Err(err(
                            line,
                            "f-string conversions ({x!r}) are not supported in D0",
                        ));
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
                return Err(err(line, "unterminated {...} in f-string"));
            }
            let expr_src = src[start..i - 1].trim().to_string();
            if expr_src.is_empty() {
                return Err(err(line, "empty {} in f-string"));
            }
            if !lit.is_empty() {
                parts.push(FPart::Lit(std::mem::take(&mut lit)));
            }
            parts.push(FPart::Expr(expr_src));
            *pos = i;
            continue;
        }
        if c == b'}' {
            if *pos + 1 < n && bytes[*pos + 1] == b'}' {
                lit.push('}');
                *pos += 2;
                continue;
            }
            return Err(err(line, "single '}' in f-string; use '}}' for a literal brace"));
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
