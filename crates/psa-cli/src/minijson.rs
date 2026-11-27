//! Minimal JSON value parser (no serde dependency in psa).
//!
//! Used by exactly two P3 readers, both of which only need a small
//! subset of JSON: the SA trap-JSON host adapter (`hostdiag`) and
//! the `.psa-map.json` segment-table loader (`srcmap`). It parses
//! objects, arrays, strings (with `\"`, `\\`, `\n`, `\r`, `\t`,
//! `\uXXXX` escapes), numbers (kept as f64 plus the raw literal),
//! `true`/`false`/`null`. Anything else is a parse failure
//! (`None`); callers treat that as "unparsed", never as an error
//! to invent content for.

#[derive(Debug, Clone)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Integer value of a JSON number, if it is integral and fits.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::Num(n) if *n >= 0.0 && n.fract() == 0.0 && *n <= u32::MAX as f64 => {
                Some(*n as u32)
            }
            _ => None,
        }
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

pub fn parse(text: &str) -> Option<Value> {
    let mut p = Parser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    p.skip_ws();
    let v = p.value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return None;
    }
    Some(v)
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            match b {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn expect(&mut self, lit: &str) -> Option<()> {
        if self.bytes[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            Some(())
        } else {
            None
        }
    }

    fn value(&mut self) -> Option<Value> {
        self.skip_ws();
        match self.peek()? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => self.string().map(Value::Str),
            b't' => self.expect("true").map(|_| Value::Bool(true)),
            b'f' => self.expect("false").map(|_| Value::Bool(false)),
            b'n' => self.expect("null").map(|_| Value::Null),
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    fn object(&mut self) -> Option<Value> {
        self.pos += 1; // consume '{'
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Some(Value::Obj(pairs));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return None;
            }
            self.pos += 1;
            let val = self.value()?;
            pairs.push((key, val));
            self.skip_ws();
            match self.peek()? {
                b',' => {
                    self.pos += 1;
                }
                b'}' => {
                    self.pos += 1;
                    return Some(Value::Obj(pairs));
                }
                _ => return None,
            }
        }
    }

    fn array(&mut self) -> Option<Value> {
        self.pos += 1; // consume '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Some(Value::Arr(items));
        }
        loop {
            let val = self.value()?;
            items.push(val);
            self.skip_ws();
            match self.peek()? {
                b',' => {
                    self.pos += 1;
                }
                b']' => {
                    self.pos += 1;
                    return Some(Value::Arr(items));
                }
                _ => return None,
            }
        }
    }

    fn string(&mut self) -> Option<String> {
        if self.peek() != Some(b'"') {
            return None;
        }
        self.pos += 1;
        let mut out = String::new();
        loop {
            let b = self.peek()?;
            match b {
                b'"' => {
                    self.pos += 1;
                    return Some(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let esc = self.peek()?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'u' => {
                            let mut code: u32 = 0;
                            for _ in 0..4 {
                                let h = self.peek()?;
                                self.pos += 1;
                                let digit = (h as char).to_digit(16)?;
                                code = code * 16 + digit;
                            }
                            let ch = char::from_u32(code).unwrap_or('\u{FFFD}');
                            out.push(ch);
                        }
                        _ => return None,
                    }
                }
                0x00..=0x1F => return None,
                _ => {
                    // Copy one UTF-8 char verbatim.
                    let rest = &self.bytes[self.pos..];
                    let s = std::str::from_utf8(rest).ok()?;
                    let ch = s.chars().next()?;
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
    }

    fn number(&mut self) -> Option<Value> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-' => self.pos += 1,
                _ => break,
            }
        }
        let lit = std::str::from_utf8(&self.bytes[start..self.pos]).ok()?;
        lit.parse::<f64>().ok().map(Value::Num)
    }
}
