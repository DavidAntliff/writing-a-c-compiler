use crate::ast::{Expression, Function, Program, Statement};
use crate::lexer::{Keyword, Token};
use thiserror::Error;
use winnow::error::{StrContext, StrContextValue};
use winnow::prelude::*;
use winnow::stream::TokenSlice;
use winnow::token::{any, literal};

#[derive(Debug, PartialEq, Error)]
#[error("{message}")]
pub struct ParserError {
    message: String,
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

        ParserError {
            message: format!(
                "Expected {}, found {:?}",
                expected,
                error.input()[error.offset()]
            ),
        }
    }
}

type Tokens<'i> = TokenSlice<'i, Token>;

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
    literal(Token::Keyword(Keyword::Int))
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
    literal(Token::OpenParen)
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::StringLiteral("(")))
        .parse_next(i)?;
    literal(Token::Keyword(Keyword::Void))
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::Description(
            "keyword",
        )))
        .parse_next(i)?;
    literal(Token::CloseParen)
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::StringLiteral(")")))
        .parse_next(i)?;
    literal(Token::OpenBrace)
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::StringLiteral("{")))
        .parse_next(i)?;
    let body = statement
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::Description(
            "statement",
        )))
        .parse_next(i)?;
    literal(Token::CloseBrace)
        .context(StrContext::Label("function"))
        .context(StrContext::Expected(StrContextValue::StringLiteral("}")))
        .parse_next(i)?;
    Ok(Function { name, body })
}

fn statement(i: &mut Tokens<'_>) -> winnow::Result<Statement> {
    literal(Token::Keyword(Keyword::Return))
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
    literal(Token::Semicolon)
        .context(StrContext::Label("statement"))
        .context(StrContext::Expected(StrContextValue::Description(
            "semicolon",
        )))
        .parse_next(i)?;
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
                Err(ParserError {
                    message: "Expected an identifier".to_string(),
                })
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
                Err(ParserError {
                    message: "Expected a constant".to_string(),
                })
            }
        })
        .parse_next(i)?;
    Ok(constant)
}

#[cfg(test)]
mod tests {
    use crate::lexer::lex;
    use winnow::Parser;

    #[test]
    fn test_bring_up() {
        let input = r#"
        int main(void) {
            return 2;
        }
        "#;
        let tokens = lex(input).unwrap();
        dbg!(&tokens);
        let mut tokens = crate::parser::Tokens::new(&tokens);
        let program = crate::parser::program.parse_next(&mut tokens);
        dbg!(&program);
    }
}
