//! Parser for the C language
//!
//! Grammar:
//!   <program> ::= <function>
//!   <function> ::= "int" <identifier> "(" "void" ")" "{" <statement> "}"
//!   <statement> ::= "return" <exp> ";"
//!   <exp> ::= <int> | <unop> <exp> | "(" <exp> ")"
//!   <unop> ::= "-" | "~"
//!   <identifier> ::= ? An identifier token ?
//!   <int> ::= ? A constant token ?
//!

use crate::ast_c::{Expression, Function, Program, Statement, UnaryOperator};
use crate::lexer::{Keyword, Token, TokenKind};
use thiserror::Error;
use winnow::combinator::{fail, peek};
use winnow::error::{StrContext, StrContextValue};
use winnow::prelude::*;
use winnow::stream::TokenSlice;
use winnow::token::{any, literal, take};

#[derive(Debug, PartialEq, Error)]
#[error("{message}")]
pub struct ParserError {
    pub message: String, // TODO: remove?
    pub expected: String,
    pub found: String,
    pub offset: usize,
}

impl ParserError {
    // Avoiding `From` so winnow types don't become part of our public API
    fn from_parse(
        error: winnow::error::ParseError<TokenSlice<'_, Token>, winnow::error::ContextError>,
    ) -> Self {
        let context = error.inner().context();
        let expected = context
            .filter_map(|c| match c {
                StrContext::Expected(e) => Some(e.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>();

        let expected = expected.first().cloned().unwrap_or("unknown".to_string());

        let found = error
            .input()
            .get(error.offset())
            .map(|t| format!("{:?}", t.kind))
            .unwrap_or("EOF".into());

        let offset = if error.input().is_empty() {
            0
        } else {
            error
                .input()
                .get(error.offset())
                .map(|e| e.span.start)
                .unwrap_or_else(|| {
                    if error.offset() == 0 {
                        return 0;
                    }
                    // Unexpected EOF, so get the end of the last token's span
                    error
                        .input()
                        .get(error.offset() - 1)
                        .map(|e| e.span.end)
                        .unwrap_or(0)
                })
        };

        ParserError {
            message: format!("Expected {}, found {:?}", expected, found,),
            expected: expected.clone(),
            found: found.clone(),
            offset,
        }
    }
}

type Tokens<'i> = TokenSlice<'i, Token>;

impl PartialEq<TokenKind> for Token {
    fn eq(&self, other: &TokenKind) -> bool {
        self.kind == *other
    }
}

impl winnow::stream::ContainsToken<&'_ Token> for TokenKind {
    #[inline(always)]
    fn contains_token(&self, token: &'_ Token) -> bool {
        *self == token.kind
    }
}

impl winnow::stream::ContainsToken<&'_ Token> for &'_ [TokenKind] {
    #[inline]
    fn contains_token(&self, token: &'_ Token) -> bool {
        self.iter().any(|t| *t == token.kind)
    }
}

impl<const LEN: usize> winnow::stream::ContainsToken<&'_ Token> for &'_ [TokenKind; LEN] {
    #[inline]
    fn contains_token(&self, token: &'_ Token) -> bool {
        self.iter().any(|t| *t == token.kind)
    }
}

impl<const LEN: usize> winnow::stream::ContainsToken<&'_ Token> for [TokenKind; LEN] {
    #[inline]
    fn contains_token(&self, token: &'_ Token) -> bool {
        self.iter().any(|t| *t == token.kind)
    }
}

pub(crate) fn parse(input: &[Token]) -> Result<Program, ParserError> {
    let tokens = Tokens::new(input);
    let program = program.parse(tokens).map_err(ParserError::from_parse)?;
    Ok(program)
}

fn program(i: &mut Tokens<'_>) -> winnow::Result<Program> {
    let function = function
        .context(StrContext::Label("program"))
        .context(StrContext::Expected(StrContextValue::Description(
            "function",
        )))
        .parse_next(i)?;
    Ok(Program { function })
}

fn function(i: &mut Tokens<'_>) -> winnow::Result<Function> {
    literal(TokenKind::Keyword(Keyword::Int))
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;
    let name = identifier
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::Description(
            "identifier",
        )))
        .parse_next(i)?;
    literal(TokenKind::OpenParen)
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::StringLiteral("(")))
        .parse_next(i)?;
    literal(TokenKind::Keyword(Keyword::Void))
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;
    literal(TokenKind::CloseParen)
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::StringLiteral(")")))
        .parse_next(i)?;
    literal(TokenKind::OpenBrace)
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::StringLiteral("{")))
        .parse_next(i)?;
    let body = statement
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::Description(
            "statement",
        )))
        .parse_next(i)?;
    literal(TokenKind::CloseBrace)
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::StringLiteral("}")))
        .parse_next(i)?;
    Ok(Function { name, body })
}

fn statement(i: &mut Tokens<'_>) -> winnow::Result<Statement> {
    literal(TokenKind::Keyword(Keyword::Return))
        .context(StrContext::Label("statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;
    let exp = exp
        .context(StrContext::Label("statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "expression",
        )))
        .parse_next(i)?;
    literal(TokenKind::Semicolon)
        .context(StrContext::Label("statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "semicolon",
        )))
        .parse_next(i)?;
    Ok(Statement::Return(exp))
}

fn exp(i: &mut Tokens<'_>) -> winnow::Result<Expression> {
    let next_token = peek(any).parse_next(i)?;

    let exp = match next_token.kind {
        TokenKind::Constant(_) => Expression::Constant(int.parse_next(i)?),
        TokenKind::BitwiseComplement | TokenKind::Negation => {
            let op = unop.parse_next(i)?;
            let exp = exp.parse_next(i)?;
            Expression::Unary(op, Box::new(exp))
        }
        TokenKind::OpenParen => {
            take(1usize).parse_next(i)?;
            let exp = exp.parse_next(i)?;
            literal(TokenKind::CloseParen).parse_next(i)?;
            exp
        }
        _ => fail.parse_next(i)?,
    };

    Ok(exp)
}

fn identifier(i: &mut Tokens<'_>) -> winnow::Result<String> {
    let identifier = any
        .try_map(|t: &Token| {
            if let TokenKind::Identifier(ref id) = t.kind {
                Ok(id.clone())
            } else {
                Err(ParserError {
                    message: "Expected an identifier".to_string(),
                    expected: "identifier".to_string(),
                    found: format!("{:?}", t.kind),
                    offset: t.span.start,
                })
            }
        })
        .context(StrContext::Label("identifier"))
        .parse_next(i)?;
    Ok(identifier)
}

fn int(i: &mut Tokens<'_>) -> winnow::Result<usize> {
    let constant = any
        .try_map(|t: &Token| {
            if let TokenKind::Constant(c) = t.kind {
                Ok(c)
            } else {
                Err(ParserError {
                    message: "Expected a constant".to_string(),
                    expected: "constant".to_string(),
                    found: format!("{:?}", t.kind),
                    offset: t.span.start,
                })
            }
        })
        .context(StrContext::Label("int"))
        .parse_next(i)?;
    Ok(constant)
}

fn unop(i: &mut Tokens<'_>) -> winnow::Result<UnaryOperator> {
    let op = any
        .try_map(|t: &Token| match t.kind {
            TokenKind::BitwiseComplement => Ok(UnaryOperator::Complement),
            TokenKind::Negation => Ok(UnaryOperator::Negate),
            _ => Err(ParserError {
                message: "Expected a unary operator".to_string(),
                expected: "unary operator".to_string(),
                found: format!("{:?}", t.kind),
                offset: t.span.start,
            }),
        })
        .context(StrContext::Label("unop"))
        .parse_next(i)?;
    Ok(op)
}

#[cfg(test)]
mod tests {
    use crate::ast_c::{Expression, UnaryOperator};
    use crate::lexer::lex;
    use crate::parser::{exp, Tokens};
    use assert_matches::assert_matches;
    use winnow::error::ParseError;
    use winnow::Parser;

    #[test]
    fn test_bring_up() {
        let input = r#"
        int main(void) {
            return 2;
        }
        "#;
        let tokens = lex(input).unwrap();
        //dbg!(&tokens);
        let mut tokens = crate::parser::Tokens::new(&tokens);
        let _program = crate::parser::program.parse_next(&mut tokens);
        //dbg!(&program);
    }

    #[test]
    fn test_expression_constant() {
        let input = r#"2"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let expression = exp.parse(tokens).unwrap();
        assert_eq!(expression, Expression::Constant(2));
    }

    #[test]
    fn test_expression_bitwise_complement() {
        let input = r#"~2"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let expression = exp.parse(tokens).unwrap();
        assert_eq!(
            expression,
            Expression::Unary(UnaryOperator::Complement, Box::new(Expression::Constant(2)))
        );
    }

    #[test]
    fn test_expression_negation() {
        let input = r#"-3"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let expression = exp.parse(tokens).unwrap();
        assert_eq!(
            expression,
            Expression::Unary(UnaryOperator::Negate, Box::new(Expression::Constant(3)))
        );
    }

    #[test]
    fn test_expression_complement_of_negation() {
        let input = r#"~(-4)"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let expression = exp.parse(tokens).unwrap();
        assert_eq!(
            expression,
            Expression::Unary(
                UnaryOperator::Complement,
                Box::new(Expression::Unary(
                    UnaryOperator::Negate,
                    Box::new(Expression::Constant(4))
                ))
            )
        );
    }

    #[test]
    fn test_expression_error() {
        let input = r#";"#;
        let tokens = lex(input).unwrap();
        let tokens = Tokens::new(&tokens);
        let error = exp.parse(tokens).unwrap_err();
        assert_matches!(error, ParseError { .. });
    }
}
