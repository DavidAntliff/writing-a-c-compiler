mod ast;
mod lexer;
mod parser;

use std::fs;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O: {path}")]
    Io {
        source: std::io::Error,
        path: PathBuf,
    },

    #[error(transparent)]
    Lexer(#[from] lexer::LexerError),

    #[error(transparent)]
    Parser(#[from] parser::ParserError),

    #[error("Compiler error: {0}")]
    Custom(String),
}

pub fn do_the_thing(
    input_filename: PathBuf,
    output_filename: Option<PathBuf>,
    stop_after_lex: bool,
    stop_after_parse: bool,
    stop_after_codegen: bool,
) -> Result<(), Error> {
    let dummy = "\n\
    .text\n\
    .globl _main\n\
_main:\n\
    movl $0, %eax\n\
    ret\n\
";

    // read the input file as a string into memory
    log::info!("Reading input file: {}", input_filename.display());
    let input = fs::read_to_string(&input_filename).map_err(|e| Error::Io {
        source: e,
        path: input_filename.clone(),
    })?;

    log::info!("Lexing input file: {}", input_filename.display());
    let lexed = lexer::lex(&input)?;

    log::debug!("Lexed input: {lexed:#?}");

    if stop_after_lex {
        return Ok(());
    }

    log::info!("Parsing input file: {}", input_filename.display());
    let parsed = parser::parse(&lexed)?;

    log::debug!("AST: {parsed:#?}");

    if stop_after_parse {
        return Ok(());
    }

    // TODO: codegen

    if stop_after_codegen {
        return Ok(());
    }

    let output_filename = output_filename.unwrap_or_else(|| input_filename.with_extension("s"));
    log::info!("Emitting output file: {}", output_filename.display());

    fs::write(&output_filename, dummy).map_err(|e| Error::Io {
        source: e,
        path: output_filename.clone(),
    })?;

    Ok(())
}

#[cfg(test)]
fn lex_and_parse(input: &str) -> Result<ast::Program, Error> {
    Ok(parser::parse(&lexer::lex(input)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ast::{Expression, Function, Program, Statement};
    use assert_matches::assert_matches;

    #[test]
    fn test_listing_1_1() {
        // Listing on page 4, AST on page 13
        let input = r#"
        int main(void) {
            return 2;
        }
        "#;
        assert_eq!(
            lex_and_parse(input).unwrap(),
            Program {
                function: Function {
                    name: "main".to_string(),
                    body: Statement::Return(Expression::Constant(2)),
                }
            }
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
