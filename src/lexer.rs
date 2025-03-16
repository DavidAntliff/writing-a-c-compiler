use winnow::ascii::{alphanumeric1, digit1, space0, space1};
use winnow::combinator::{alt, eof, not, peek, separated, terminated};
use winnow::error::{StrContext, StrContextValue};
use winnow::prelude::*;
use winnow::token::{none_of, one_of};
use winnow::stream::AsChar;

#[derive(Debug)]
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
    tokens.parse(input)
        .map_err(|e| LexerError {
            message: format!("Lexer error: {}", e),
        })
}

fn tokens(input: &mut &str) -> winnow::Result<Vec<Token>> {
    let list_parser = separated(0.., token, space1);

    let mut full_parser = (space0, list_parser, space0, eof)
        .map(|(_, tokens, _, _)| tokens);

    full_parser.parse_next(input)
}

fn token(input: &mut &str) -> winnow::Result<Token> {
    alt((
        // keyword_parser,
        // identifier_parser,
        int,
        // open_paren_parser,
        // close_paren_parser,
        // open_brace_parser,
        // close_brace_parser,
        // semicolon_parser,
    ))
    .parse_next(input)
}

// Parser for a single unsigned integer token:
fn int(input: &mut &str) -> winnow::Result<Token> {
    // Look ahead: the next character must not be a word character.
    terminated(digit1, not(one_of(|c: char | c.is_alphanum() || c == '_')))
        .parse_to::<usize>()
        .map(Token::Constant)
        .parse_next(input)
    // terminated(digit1, none_of(|c: char | c.is_alphanum() || c == '_'))
    //     .parse_to::<usize>()
    //     .map(Token::Constant)
    //     .parse_next(input)
}


// fn parse_digits<'s>(input: &mut &'s str) -> winnow::Result<(&'s str, &'s str)> {
//     alt((
//         // ("0b", parse_bin_digits)
//         //     .context(StrContext::Label("digit"))
//         //     .context(StrContext::Expected(StrContextValue::Description("binary"))),
//         // ("0o", parse_oct_digits)
//         //     .context(StrContext::Label("digit"))
//         //     .context(StrContext::Expected(StrContextValue::Description("octal"))),
//         parse_digits2
//             .context(StrContext::Label("digit"))
//             .context(StrContext::Expected(StrContextValue::Description("decimal"))),
//         // ("0x", parse_hex_digits)
//         //     .context(StrContext::Label("digit"))
//         //     .context(StrContext::Expected(StrContextValue::Description("hexadecimal"))),
//     )).parse_next(input)
// }

// fn parse_input()
//
// fn parse_digits2<'s>(input: &mut &'s str) -> winnow::Result<(&'s str, &'s str)> {
//     digit1
//         .parse_to()
//         .parse_next(input)
// }
//

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use winnow::error::{ContextError, ErrMode, Needed, ParseError};
    use winnow::Partial;
    use super::*;

    #[test]
    fn test_lex() {
        // let input = "123";
        // let result = lex(input);
        // assert_eq!(result.unwrap(), 123);
    }

    #[test]
    fn test_lex_int() {
        assert_eq!(int.parse("0"), Ok(Token::Constant(0)));
        assert_eq!(int.parse("123"), Ok(Token::Constant(123)));
    }

    #[test]
    fn test_lex_token() {
        assert_eq!(int.parse("123"), Ok(Token::Constant(123)));
    }

    #[test]
    fn test_lex_tokens() {
        let input = "  123   456 ";
        let result = tokens.parse(input);
        assert_eq!(result, Ok(vec![Token::Constant(123), Token::Constant(456)]));
    }

    #[test]
    fn test_word_boundary_error() {
        // as per book, '123abc' should raise an error because 123 is not followed by a word boundary,
        // but '123;bar' should not raise an error because 123 is followed by a semicolon.
        //let a = int.parse("123abc");
        assert_eq!(int.parse_peek(&Partial::new("123;bar")), Ok((";bar", Token::Constant(123))));
        assert_eq!(int.parse_peek(&Partial::new("123 bar")), Ok((" bar", Token::Constant(123))));
        assert_eq!(int.parse_peek(&Partial::new("123abc")), Err(ContextError::new()));
        assert_eq!(int.parse_peek(&Partial::new("123_bc")), Err(ContextError::new()));
        // assert_matches!(int.parse("123abc").unwrap_err(), ParseError::<&str, ContextError>(
        //     ContextError::Expected(StrContextValue::Description("word boundary"))
        // ));
    }
}