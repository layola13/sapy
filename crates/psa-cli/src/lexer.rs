//! Indentation-aware lexer for the .psa subset.
//!
//! Produces Newline / Indent / Dedent tokens Python-style. Anything outside
//! the v0.1 subset (tabs in indentation, `**`, subscripts, f-strings, ...)
//! is a loud error, never a silent skip.

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),
    // keywords
    Def,
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
    // symbols
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Dot,
    Plus,
    Minus,
    Star,
    Slash,
    FloorDiv,
    Percent,
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
                        return Err(err(tok_line, "operator ** is not supported in v0.1"));
                    }
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::StarEq, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Star, line: tok_line });
                    }
                }
                b'/' => {
                    if peek(bytes, pos, b'/') {
                        pos += 2;
                        out.push(Token { tok: Tok::FloorDiv, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Slash, line: tok_line });
                    }
                }
                b'%' => {
                    pos += 1;
                    out.push(Token { tok: Tok::Percent, line: tok_line });
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
                    if peek(bytes, pos, b'=') {
                        pos += 2;
                        out.push(Token { tok: Tok::LtEq, line: tok_line });
                    } else {
                        pos += 1;
                        out.push(Token { tok: Tok::Lt, line: tok_line });
                    }
                }
                b'>' => {
                    if peek(bytes, pos, b'=') {
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
