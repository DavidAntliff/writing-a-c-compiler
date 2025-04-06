use thiserror::Error;
use winnow::ascii::{alphanumeric0, digit1, multispace0};
use winnow::combinator::{alt, not, repeat, terminated};
use winnow::prelude::*;
use winnow::stream::AsChar;
use winnow::token::{one_of, take_while};
use winnow::LocatingSlice;

pub(crate) type Constant = usize;
pub(crate) type Identifier = String;

#[derive(Debug, PartialEq, Error)]
#[error("Lexer error: {message}")]
pub struct LexerError {
    message: String,
}

// TODO: add line, column data to each token, so that the parser
//       can report errors with line and column numbers.

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) struct Token {
    pub(crate) kind: TokenKind,
    pub(crate) span: std::ops::Range<usize>,
}

impl Token {
    pub(crate) fn is_binary_operator(&self) -> bool {
        self.kind.is_binary_operator()
    }

    pub(crate) fn precedence(&self) -> usize {
        self.kind.precedence()
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) enum TokenKind {
    Keyword(Keyword),
    Identifier(Identifier),
    Constant(Constant),
    OpenParen,          // (
    CloseParen,         // )
    OpenBrace,          // {
    CloseBrace,         // }
    Semicolon,          // ;
    BitwiseComplement,  // ~
    Negation,           // -
    Decrement,          // --
    Add,                // +
    Multiply,           // *
    Divide,             // /
    Remainder,          // %
    BitwiseAnd,         // &
    BitwiseOr,          // |
    BitwiseXor,         // ^
    BitwiseShiftLeft,   // <<
    BitwiseShiftRight,  // >>
    LogicalNot,         // !
    LogicalAnd,         // &&
    LogicalOr,          // ||
    Equal,              // ==
    NotEqual,           // !=
    LessThan,           // <
    GreaterThan,        // >
    LessThanOrEqual,    // <=
    GreaterThanOrEqual, // >=
}

impl TokenKind {
    fn is_binary_operator(&self) -> bool {
        matches!(
            self,
            TokenKind::Add
                | TokenKind::Negation
                | TokenKind::Multiply
                | TokenKind::Divide
                | TokenKind::Remainder
                | TokenKind::BitwiseAnd
                | TokenKind::BitwiseOr
                | TokenKind::BitwiseXor
                | TokenKind::BitwiseShiftLeft
                | TokenKind::BitwiseShiftRight
                | TokenKind::LogicalAnd
                | TokenKind::LogicalOr
                | TokenKind::Equal
                | TokenKind::NotEqual
                | TokenKind::LessThan
                | TokenKind::GreaterThan
                | TokenKind::LessThanOrEqual
                | TokenKind::GreaterThanOrEqual
        )
    }

    fn precedence(&self) -> usize {
        match self {
            TokenKind::Multiply => 50,
            TokenKind::Divide => 50,
            TokenKind::Remainder => 50,
            TokenKind::Add => 45,
            TokenKind::Negation => 45,
            TokenKind::BitwiseShiftLeft => 40,
            TokenKind::BitwiseShiftRight => 40,
            TokenKind::LessThan => 35,
            TokenKind::GreaterThan => 35,
            TokenKind::LessThanOrEqual => 35,
            TokenKind::GreaterThanOrEqual => 35,
            TokenKind::Equal => 30,
            TokenKind::NotEqual => 30,
            TokenKind::BitwiseAnd => 25,
            TokenKind::BitwiseXor => 20,
            TokenKind::BitwiseOr => 15,
            TokenKind::LogicalAnd => 10,
            TokenKind::LogicalOr => 5,

            _ => panic!("Unexpected token: {:?}", self),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) enum Keyword {
    Int,
    Void,
    Return,
}

pub(crate) fn lex(input: &str) -> Result<Vec<Token>, LexerError> {
    let input = LocatingInput::new(input);
    tokens.parse(input).map_err(|e| LexerError {
        message: e.to_string(),
    })
}

type LocatingInput<'a> = LocatingSlice<&'a str>;

fn tokens(input: &mut LocatingInput<'_>) -> winnow::Result<Vec<Token>> {
    let tokens = repeat(0.., token).parse_next(input);
    multispace0.parse_next(input)?;
    tokens
}

fn token(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    multispace0.parse_next(input)?;
    // TODO: optimise with dispatch! / peek
    alt((
        // longer tokens first
        alt((
            identifier,
            constant,
            decrement,
            bitwise_shift_left,
            bitwise_shift_right,
            logical_and,
            logical_or,
            equal,
            not_equal,
            less_than_or_equal,
            greater_than_or_equal,
        )),
        alt((
            open_paren,
            close_paren,
            open_brace,
            close_brace,
            semicolon,
            bitwise_complement,
            negation,
            add,
            multiply,
            divide,
            remainder,
            bitwise_and,
            bitwise_or,
            bitwise_xor,
            logical_not,
            less_than,
            greater_than,
        )),
    ))
    .parse_next(input)
}

// Parser for a single unsigned integer token:
fn constant(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    // Look ahead: the next character must not be a word character.
    terminated(
        digit1,
        not(one_of(|c: char| c.is_alphanum() || c == '_')), // \b
    )
    .parse_to::<Constant>()
    .with_span()
    .map(|(constant, span)| Token {
        kind: TokenKind::Constant(constant),
        span,
    })
    .parse_next(input)
}

fn open_paren(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    '('.with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::OpenParen,
            span,
        })
        .parse_next(input)
}

fn close_paren(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    ')'.with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::CloseParen,
            span,
        })
        .parse_next(input)
}
fn open_brace(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    '{'.with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::OpenBrace,
            span,
        })
        .parse_next(input)
}

fn close_brace(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    '}'.with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::CloseBrace,
            span,
        })
        .parse_next(input)
}

fn semicolon(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    ';'.with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::Semicolon,
            span,
        })
        .parse_next(input)
}

fn identifier(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    let (identifier, span) = terminated(
        (
            take_while(1, |c: char| c.is_alpha() || c == '_'),
            take_while(0.., |c: char| c.is_alphanum() || c == '_'),
            alphanumeric0,
        )
            .take(),
        not(one_of(|c: char| c.is_alphanum() || c == '_')), // \b
    )
    .with_span()
    .parse_next(input)?;

    match identifier {
        "int" => Ok(Token {
            kind: TokenKind::Keyword(Keyword::Int),
            span,
        }),
        "void" => Ok(Token {
            kind: TokenKind::Keyword(Keyword::Void),
            span,
        }),
        "return" => Ok(Token {
            kind: TokenKind::Keyword(Keyword::Return),
            span,
        }),
        _ => Ok(Token {
            kind: TokenKind::Identifier(identifier.to_string()),
            span,
        }),
    }
}

fn bitwise_complement(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    '~'.with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::BitwiseComplement,
            span,
        })
        .parse_next(input)
}

fn negation(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    '-'.with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::Negation,
            span,
        })
        .parse_next(input)
}

fn decrement(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "--".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::Decrement,
            span,
        })
        .parse_next(input)
}

fn add(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "+".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::Add,
            span,
        })
        .parse_next(input)
}

fn multiply(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "*".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::Multiply,
            span,
        })
        .parse_next(input)
}

fn divide(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "/".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::Divide,
            span,
        })
        .parse_next(input)
}

fn remainder(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "%".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::Remainder,
            span,
        })
        .parse_next(input)
}

fn bitwise_and(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "&".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::BitwiseAnd,
            span,
        })
        .parse_next(input)
}

fn bitwise_or(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "|".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::BitwiseOr,
            span,
        })
        .parse_next(input)
}

fn bitwise_xor(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "^".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::BitwiseXor,
            span,
        })
        .parse_next(input)
}

fn bitwise_shift_left(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "<<".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::BitwiseShiftLeft,
            span,
        })
        .parse_next(input)
}

fn bitwise_shift_right(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    ">>".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::BitwiseShiftRight,
            span,
        })
        .parse_next(input)
}

fn logical_not(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "!".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::LogicalNot,
            span,
        })
        .parse_next(input)
}

fn logical_and(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "&&".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::LogicalAnd,
            span,
        })
        .parse_next(input)
}

fn logical_or(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "||".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::LogicalOr,
            span,
        })
        .parse_next(input)
}

fn equal(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "==".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::Equal,
            span,
        })
        .parse_next(input)
}

fn not_equal(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "!=".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::NotEqual,
            span,
        })
        .parse_next(input)
}

fn less_than(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "<".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::LessThan,
            span,
        })
        .parse_next(input)
}

fn greater_than(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    ">".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::GreaterThan,
            span,
        })
        .parse_next(input)
}

fn less_than_or_equal(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    "<=".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::LessThanOrEqual,
            span,
        })
        .parse_next(input)
}

fn greater_than_or_equal(input: &mut LocatingInput<'_>) -> winnow::Result<Token> {
    ">=".with_span()
        .map(|(_, span)| Token {
            kind: TokenKind::GreaterThanOrEqual,
            span,
        })
        .parse_next(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use winnow::error::ContextError;

    #[test]
    fn test_lex() {
        assert_eq!(
            lex("123"),
            Ok(vec![Token {
                kind: TokenKind::Constant(123),
                span: 0..3
            },])
        );
        assert_eq!(
            lex("123()"),
            Ok(vec![
                Token {
                    kind: TokenKind::Constant(123),
                    span: 0..3
                },
                Token {
                    kind: TokenKind::OpenParen,
                    span: 3..4
                },
                Token {
                    kind: TokenKind::CloseParen,
                    span: 4..5
                },
            ])
        );
        assert_eq!(
            lex("  123 \n  456 "),
            Ok(vec![
                Token {
                    kind: TokenKind::Constant(123),
                    span: 2..5
                },
                Token {
                    kind: TokenKind::Constant(456),
                    span: 9..12
                }
            ])
        );
        assert_eq!(
            lex("(123)"),
            Ok(vec![
                Token {
                    kind: TokenKind::OpenParen,
                    span: 0..1
                },
                Token {
                    kind: TokenKind::Constant(123),
                    span: 1..4
                },
                Token {
                    kind: TokenKind::CloseParen,
                    span: 4..5
                }
            ])
        );
        assert_eq!(
            lex("{123}"),
            Ok(vec![
                Token {
                    kind: TokenKind::OpenBrace,
                    span: 0..1
                },
                Token {
                    kind: TokenKind::Constant(123),
                    span: 1..4
                },
                Token {
                    kind: TokenKind::CloseBrace,
                    span: 4..5
                },
            ])
        );
        assert_eq!(
            lex("\n;123  ;"),
            Ok(vec![
                Token {
                    kind: TokenKind::Semicolon,
                    span: 1..2
                },
                Token {
                    kind: TokenKind::Constant(123),
                    span: 2..5
                },
                Token {
                    kind: TokenKind::Semicolon,
                    span: 7..8
                },
            ])
        );
        assert_eq!(
            lex("main(void)"),
            Ok(vec![
                Token {
                    kind: TokenKind::Identifier("main".to_string()),
                    span: 0..4
                },
                Token {
                    kind: TokenKind::OpenParen,
                    span: 4..5
                },
                Token {
                    kind: TokenKind::Keyword(Keyword::Void),
                    span: 5..9
                },
                Token {
                    kind: TokenKind::CloseParen,
                    span: 9..10
                },
            ])
        );
    }

    #[test]
    fn test_lex_constant() {
        assert_eq!(
            constant.parse(LocatingInput::new("0")),
            Ok(Token {
                kind: TokenKind::Constant(0),
                span: 0..1
            })
        );
        assert_eq!(
            constant.parse(LocatingInput::new("123")),
            Ok(Token {
                kind: TokenKind::Constant(123),
                span: 0..3
            })
        );
    }

    #[test]
    fn test_lex_delimiter() {
        assert_eq!(
            token.parse(LocatingInput::new("(")),
            Ok(Token {
                kind: TokenKind::OpenParen,
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new(")")),
            Ok(Token {
                kind: TokenKind::CloseParen,
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("{")),
            Ok(Token {
                kind: TokenKind::OpenBrace,
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("}")),
            Ok(Token {
                kind: TokenKind::CloseBrace,
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new(";")),
            Ok(Token {
                kind: TokenKind::Semicolon,
                span: 0..1
            })
        );
    }

    #[test]
    fn test_lex_keyword() {
        assert_eq!(
            token.parse(LocatingInput::new("int")),
            Ok(Token {
                kind: TokenKind::Keyword(Keyword::Int),
                span: 0..3
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("void")),
            Ok(Token {
                kind: TokenKind::Keyword(Keyword::Void),
                span: 0..4
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("return")),
            Ok(Token {
                kind: TokenKind::Keyword(Keyword::Return),
                span: 0..6
            })
        );
    }

    #[test]
    fn test_lex_identifier() {
        assert_eq!(
            token.parse(LocatingInput::new("a")),
            Ok(Token {
                kind: TokenKind::Identifier("a".into()),
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("_")),
            Ok(Token {
                kind: TokenKind::Identifier("_".into()),
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("a1")),
            Ok(Token {
                kind: TokenKind::Identifier("a1".into()),
                span: 0..2
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("_1")),
            Ok(Token {
                kind: TokenKind::Identifier("_1".into()),
                span: 0..2
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("a_")),
            Ok(Token {
                kind: TokenKind::Identifier("a_".into()),
                span: 0..2
            })
        );
    }

    #[test]
    fn test_lex_unary_operator() {
        assert_eq!(
            token.parse(LocatingInput::new("~")),
            Ok(Token {
                kind: TokenKind::BitwiseComplement,
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("-")),
            Ok(Token {
                kind: TokenKind::Negation,
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("--")),
            Ok(Token {
                kind: TokenKind::Decrement,
                span: 0..2
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("!")),
            Ok(Token {
                kind: TokenKind::LogicalNot,
                span: 0..1
            })
        );
    }

    #[test]
    fn test_lex_binary_operator() {
        assert_eq!(
            token.parse(LocatingInput::new("+")),
            Ok(Token {
                kind: TokenKind::Add,
                span: 0..1
            })
        );

        // - is a special case, as it's also a unary operator

        assert_eq!(
            token.parse(LocatingInput::new("*")),
            Ok(Token {
                kind: TokenKind::Multiply,
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("/")),
            Ok(Token {
                kind: TokenKind::Divide,
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("%")),
            Ok(Token {
                kind: TokenKind::Remainder,
                span: 0..1
            })
        );
    }

    #[test]
    fn test_lex_bitwise_binary_operator() {
        assert_eq!(
            token.parse(LocatingInput::new("&")),
            Ok(Token {
                kind: TokenKind::BitwiseAnd,
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("|")),
            Ok(Token {
                kind: TokenKind::BitwiseOr,
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("^")),
            Ok(Token {
                kind: TokenKind::BitwiseXor,
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("<<")),
            Ok(Token {
                kind: TokenKind::BitwiseShiftLeft,
                span: 0..2
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new(">>")),
            Ok(Token {
                kind: TokenKind::BitwiseShiftRight,
                span: 0..2
            })
        );
    }

    #[test]
    fn test_lex_logical_binary_operator() {
        assert_eq!(
            token.parse(LocatingInput::new("&&")),
            Ok(Token {
                kind: TokenKind::LogicalAnd,
                span: 0..2
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("||")),
            Ok(Token {
                kind: TokenKind::LogicalOr,
                span: 0..2
            })
        );
    }

    #[test]
    fn test_lex_comparison_binary_operator() {
        assert_eq!(
            token.parse(LocatingInput::new("==")),
            Ok(Token {
                kind: TokenKind::Equal,
                span: 0..2
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("!=")),
            Ok(Token {
                kind: TokenKind::NotEqual,
                span: 0..2
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("<")),
            Ok(Token {
                kind: TokenKind::LessThan,
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new(">")),
            Ok(Token {
                kind: TokenKind::GreaterThan,
                span: 0..1
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new("<=")),
            Ok(Token {
                kind: TokenKind::LessThanOrEqual,
                span: 0..2
            })
        );
        assert_eq!(
            token.parse(LocatingInput::new(">=")),
            Ok(Token {
                kind: TokenKind::GreaterThanOrEqual,
                span: 0..2
            })
        );
    }

    #[test]
    fn test_lex_token_error() {
        assert_eq!(
            token.parse_peek(LocatingInput::new("12_34")),
            Err(ContextError::new())
        );
    }

    #[test]
    fn test_word_boundary_error() {
        // As per book, '123abc' should raise an error because 123 is not followed by a word boundary,
        // but '123;bar' should not raise an error because 123 is followed by a semicolon, which
        // is a word boundary.
        let (_, t) = constant.parse_peek(LocatingInput::new("123;abc")).unwrap();
        assert_eq!(t.kind, TokenKind::Constant(123));

        let (_, t) = constant.parse_peek(LocatingInput::new("123 bar")).unwrap();
        assert_eq!(t.kind, TokenKind::Constant(123));

        let (_, t) = constant.parse_peek(LocatingInput::new("123(")).unwrap();
        assert_eq!(t.kind, TokenKind::Constant(123));

        let e = constant
            .parse_peek(LocatingInput::new("123abc"))
            .unwrap_err();
        assert_eq!(e, ContextError::new());

        let e = constant
            .parse_peek(LocatingInput::new("123_bc"))
            .unwrap_err();
        assert_eq!(e, ContextError::new());
    }

    #[test]
    fn test_basic_program() {
        let input = r#"
int main() {
    return 0;
}
"#;

        let expected = vec![
            Token {
                kind: TokenKind::Keyword(Keyword::Int),
                span: 1..4,
            },
            Token {
                kind: TokenKind::Identifier("main".to_string()),
                span: 5..9,
            },
            Token {
                kind: TokenKind::OpenParen,
                span: 9..10,
            },
            Token {
                kind: TokenKind::CloseParen,
                span: 10..11,
            },
            Token {
                kind: TokenKind::OpenBrace,
                span: 12..13,
            },
            Token {
                kind: TokenKind::Keyword(Keyword::Return),
                span: 18..24,
            },
            Token {
                kind: TokenKind::Constant(0),
                span: 25..26,
            },
            Token {
                kind: TokenKind::Semicolon,
                span: 26..27,
            },
            Token {
                kind: TokenKind::CloseBrace,
                span: 28..29,
            },
        ];

        assert_eq!(lex(input), Ok(expected));
    }
}
