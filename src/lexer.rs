use winnow::ascii::{alphanumeric0, digit1, multispace0};
use winnow::combinator::{alt, not, repeat, terminated};
use winnow::prelude::*;
use winnow::stream::AsChar;
use winnow::token::{one_of, take_while};

#[derive(Debug, PartialEq)]
pub(crate) struct LexerError {
    message: String,
}

#[derive(Debug, PartialEq)]
pub(crate) enum Token {
    Keyword(Keyword),
    Identifier(String),
    Constant(usize),
    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,
    Semicolon,
}

#[derive(Debug, PartialEq)]
pub(crate) enum Keyword {
    Int,
    Void,
    Return,
}

pub(crate) fn lex(input: &str) -> Result<Vec<Token>, LexerError> {
    tokens.parse(input).map_err(|e| LexerError {
        message: format!("Lexer error: {}", e),
    })
}

fn tokens(input: &mut &str) -> winnow::Result<Vec<Token>> {
    let tokens = repeat(0.., token).parse_next(input);
    multispace0.parse_next(input)?;
    tokens
}

fn token(input: &mut &str) -> winnow::Result<Token> {
    multispace0.parse_next(input)?;
    alt((
        // keyword_parser,
        identifier,
        constant,
        open_paren,
        close_paren,
        open_brace,
        close_brace,
        semicolon,
    ))
    .parse_next(input)
}

// Parser for a single unsigned integer token:
fn constant(input: &mut &str) -> winnow::Result<Token> {
    // Look ahead: the next character must not be a word character.
    terminated(
        digit1,
        not(one_of(|c: char| c.is_alphanum() || c == '_')), // \b
    )
    .parse_to::<usize>()
    .map(Token::Constant)
    .parse_next(input)
}

fn open_paren(input: &mut &str) -> winnow::Result<Token> {
    '('.map(|_| Token::OpenParen).parse_next(input)
}

fn close_paren(input: &mut &str) -> winnow::Result<Token> {
    ')'.map(|_| Token::CloseParen).parse_next(input)
}
fn open_brace(input: &mut &str) -> winnow::Result<Token> {
    '{'.map(|_| Token::OpenBrace).parse_next(input)
}

fn close_brace(input: &mut &str) -> winnow::Result<Token> {
    '}'.map(|_| Token::CloseBrace).parse_next(input)
}

fn semicolon(input: &mut &str) -> winnow::Result<Token> {
    ';'.map(|_| Token::Semicolon).parse_next(input)
}

fn identifier(input: &mut &str) -> winnow::Result<Token> {
    let identifier = terminated(
        (
            take_while(1, |c: char| c.is_alpha() || c == '_'),
            take_while(0.., |c: char| c.is_alphanum() || c == '_'),
            alphanumeric0,
        )
            .take(),
        not(one_of(|c: char| c.is_alphanum() || c == '_')), // \b
    )
    .parse_next(input)?;

    match identifier {
        "int" => Ok(Token::Keyword(Keyword::Int)),
        "void" => Ok(Token::Keyword(Keyword::Void)),
        "return" => Ok(Token::Keyword(Keyword::Return)),
        _ => Ok(Token::Identifier(identifier.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winnow::error::ContextError;
    use winnow::Partial;

    #[test]
    fn test_lex() {
        assert_eq!(lex("123"), Ok(vec![Token::Constant(123)]));
        assert_eq!(
            lex("123()"),
            Ok(vec![
                Token::Constant(123),
                Token::OpenParen,
                Token::CloseParen
            ])
        );
        assert_eq!(
            lex("  123 \n  456 "),
            Ok(vec![Token::Constant(123), Token::Constant(456)])
        );
        assert_eq!(
            lex("(123)"),
            Ok(vec![
                Token::OpenParen,
                Token::Constant(123),
                Token::CloseParen
            ])
        );
        assert_eq!(
            lex("{123}"),
            Ok(vec![
                Token::OpenBrace,
                Token::Constant(123),
                Token::CloseBrace
            ])
        );
        assert_eq!(
            lex("\n;123  ;"),
            Ok(vec![
                Token::Semicolon,
                Token::Constant(123),
                Token::Semicolon
            ])
        );
        assert_eq!(
            lex("main(void)"),
            Ok(vec![
                Token::Identifier("main".to_string()),
                Token::OpenParen,
                Token::Keyword(Keyword::Void),
                Token::CloseParen
            ])
        );
    }

    #[test]
    fn test_lex_constant() {
        assert_eq!(constant.parse("0"), Ok(Token::Constant(0)));
        assert_eq!(constant.parse("123"), Ok(Token::Constant(123)));
    }

    #[test]
    fn test_lex_token() {
        assert_eq!(token.parse("0"), Ok(Token::Constant(0)));
        assert_eq!(token.parse("123"), Ok(Token::Constant(123)));
        assert_eq!(token.parse("("), Ok(Token::OpenParen));
        assert_eq!(token.parse(")"), Ok(Token::CloseParen));
        assert_eq!(token.parse("{"), Ok(Token::OpenBrace));
        assert_eq!(token.parse("}"), Ok(Token::CloseBrace));
        assert_eq!(token.parse(";"), Ok(Token::Semicolon));
        assert_eq!(token.parse("int"), Ok(Token::Keyword(Keyword::Int)));
        assert_eq!(token.parse("void"), Ok(Token::Keyword(Keyword::Void)));
        assert_eq!(token.parse("return"), Ok(Token::Keyword(Keyword::Return)));
        assert_eq!(token.parse("a"), Ok(Token::Identifier("a".into())));
        assert_eq!(token.parse("_"), Ok(Token::Identifier("_".into())));
        assert_eq!(token.parse("a1"), Ok(Token::Identifier("a1".into())));
        assert_eq!(token.parse("_1"), Ok(Token::Identifier("_1".into())));
        assert_eq!(token.parse("a_"), Ok(Token::Identifier("a_".into())));
    }

    #[test]
    fn test_lex_token_error() {
        assert_eq!(
            token.parse_peek(&Partial::new("12_34")),
            Err(ContextError::new())
        );
    }

    #[test]
    fn test_word_boundary_error() {
        // As per book, '123abc' should raise an error because 123 is not followed by a word boundary,
        // but '123;bar' should not raise an error because 123 is followed by a semicolon, which
        // is a word boundary.
        assert_eq!(
            constant.parse_peek(&Partial::new("123;bar")),
            Ok((";bar", Token::Constant(123)))
        );
        assert_eq!(
            constant.parse_peek(&Partial::new("123 bar")),
            Ok((" bar", Token::Constant(123)))
        );
        assert_eq!(
            constant.parse_peek(&Partial::new("123(")),
            Ok(("(", Token::Constant(123)))
        );
        assert_eq!(
            constant.parse_peek(&Partial::new("123abc")),
            Err(ContextError::new())
        );
        assert_eq!(
            constant.parse_peek(&Partial::new("123_bc")),
            Err(ContextError::new())
        );
    }

    #[test]
    fn test_basic_program() {
        let input = r#"
            int main() {
                return 0;
            }
        "#;

        let expected = vec![
            Token::Keyword(Keyword::Int),
            Token::Identifier("main".to_string()),
            Token::OpenParen,
            Token::CloseParen,
            Token::OpenBrace,
            Token::Keyword(Keyword::Return),
            Token::Constant(0),
            Token::Semicolon,
            Token::CloseBrace,
        ];

        assert_eq!(lex(input), Ok(expected));
    }
}
