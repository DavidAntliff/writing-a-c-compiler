mod ast;
mod lexer;
mod parser;

use std::fs;
use std::path::PathBuf;

pub fn do_the_thing(
    input_filename: PathBuf,
    output_filename: Option<PathBuf>,
    stop_after_lex: bool,
    stop_after_parse: bool,
    stop_after_codegen: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let dummy = "\n\
    .text\n\
    .globl _main\n\
_main:\n\
    movl $0, %eax\n\
    ret\n\
";

    // read the input file as a string into memory
    let input = fs::read_to_string(&input_filename).unwrap_or_else(|_| {
        eprintln!("Failed to read input file: {}", input_filename.display());
        std::process::exit(1);
    });

    let lexed = lexer::lex(&input).unwrap_or_else(|e| {
        eprintln!("Failed to lex input file: {}", input_filename.display());
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    });

    dbg!(&lexed);
    if stop_after_lex {
        return Ok(());
    }

    let parsed = parser::parse(&lexed).unwrap_or_else(|e| {
        eprintln!("Failed to parse input file: {}", input_filename.display());
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    });

    dbg!(&parsed);
    if stop_after_parse {
        return Ok(());
    }

    // TODO: codegen

    if stop_after_codegen {
        return Ok(());
    }

    let output_filename = output_filename.unwrap_or_else(|| input_filename.with_extension("s"));

    if let Err(e) = fs::write(&output_filename, dummy) {
        eprintln!("Failed to write to file: {}", e);
        std::process::exit(1);
    }

    Ok(())
}

#[cfg(test)]
fn lex_and_parse(input: &str) -> Result<ast::Program, parser::ParserError> {
    parser::parse(&lexer::lex(input)?)
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
