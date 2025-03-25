use crate::ast::{Expression, Function, Program, Statement};
use crate::lexer::{lex, Keyword, LexerError, Token};
use thiserror::Error;
use winnow::prelude::*;
use winnow::stream::TokenSlice;
use winnow::token::{any, literal};

#[derive(Error, Debug, PartialEq)]
pub(crate) enum ParserError {
    // #[error(transparent)]
    // ParseError(
    //     #[from] winnow::error::ParseError<&'static mut &'a [Token], winnow::error::ContextError>,
    // ),
    #[error("Parse error: {0}")]
    Parse(String),

    #[error(transparent)]
    Lexer(#[from] LexerError),
}

type Tokens<'i> = TokenSlice<'i, Token>;

pub(crate) fn lex_and_parse(input: &str) -> Result<Program, ParserError> {
    parse(&lex(input)?)
}

pub(crate) fn parse(input: &[Token]) -> Result<Program, ParserError> {
    let tokens = Tokens::new(input);
    let program = program
        .parse(tokens)
        .inspect_err(|e| {
            eprintln!("Parse error: {e:?}");
        })
        .map_err(|e| ParserError::Parse(format!("{e:?}")))?;
    Ok(program)
}

fn program(i: &mut Tokens<'_>) -> winnow::Result<Program> {
    let function = function.parse_next(i)?;
    Ok(Program { function })
}

fn function(i: &mut Tokens<'_>) -> winnow::Result<Function> {
    literal(Token::Keyword(Keyword::Int)).parse_next(i)?;
    let name = identifier.parse_next(i)?;
    literal(Token::OpenParen).parse_next(i)?;
    literal(Token::Keyword(Keyword::Void)).parse_next(i)?;
    literal(Token::CloseParen).parse_next(i)?;
    literal(Token::OpenBrace).parse_next(i)?;
    let body = statement.parse_next(i)?;
    literal(Token::CloseBrace).parse_next(i)?;
    Ok(Function { name, body })
}

fn statement(i: &mut Tokens<'_>) -> winnow::Result<Statement> {
    literal(Token::Keyword(Keyword::Return)).parse_next(i)?;
    let exp = exp.parse_next(i)?;
    literal(Token::Semicolon).parse_next(i)?;
    Ok(Statement::Return(exp))
}

fn exp(i: &mut Tokens<'_>) -> winnow::Result<Expression> {
    let exp = int.map(Expression::Constant).parse_next(i)?;
    Ok(exp)
}

fn identifier(i: &mut Tokens<'_>) -> winnow::Result<String> {
    let identifier = any
        .try_map(|t: &Token| {
            if let Token::Identifier(ref id) = *t {
                Ok(id.clone())
            } else {
                Err(ParserError::Parse("Expected an identifier".to_string()))
            }
        })
        .parse_next(i)?;
    Ok(identifier)
}

fn int(i: &mut Tokens<'_>) -> winnow::Result<usize> {
    let constant = any
        .try_map(|t: &Token| {
            if let Token::Constant(c) = *t {
                Ok(c)
            } else {
                Err(ParserError::Parse("Expected a constant".to_string()))
            }
        })
        .parse_next(i)?;
    Ok(constant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;

    #[test]
    fn test_bring_up() {
        let input = r#"
        int main(void) {
            return 2;
        }
        "#;
        let tokens = lex(input).unwrap();
        dbg!(&tokens);
        let mut tokens = Tokens::new(&tokens);
        let program = program.parse_next(&mut tokens);
        dbg!(&program);
    }

    #[test]
    fn test_listing_1_1() {
        // Listing on page 4, AST on page 13
        let input = r#"
        int main(void) {
            return 2;
        }
        "#;
        assert_eq!(
            lex_and_parse(input),
            Ok(Program {
                function: Function {
                    name: "main".to_string(),
                    body: Statement::Return(Expression::Constant(2)),
                }
            })
        );
    }

    #[test]
    fn test_parse_incomplete() {
        let input = r#"
        int main(void) {
            return"#;
        assert_matches!(lex_and_parse(input), Err(_));
    }

    #[test]
    fn test_parse_extra() {
        let input = r#"
        int main(void) {
            return 2;
        }
        foo"#;
        assert_matches!(lex_and_parse(input), Err(_));
    }
}
