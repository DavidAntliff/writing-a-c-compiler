mod ast_asm;
mod ast_c;
mod codegen;
mod emitter;
mod id_gen;
mod lexer;
mod parser;
mod semantics;
mod tacky;

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

    #[error(transparent)]
    Semantics(#[from] semantics::Error),

    #[error(transparent)]
    Tacky(#[from] tacky::TackyError),

    #[error(transparent)]
    Codegen(#[from] codegen::CodegenError),

    #[error(transparent)]
    Emitter(#[from] emitter::EmitterError),

    #[error("Compiler error: {0}")]
    Custom(String),
}

pub fn read_input(input_filename: PathBuf) -> Result<String, Error> {
    // read the input file as a string into memory
    log::info!("Reading input file: {}", input_filename.display());
    let input = fs::read_to_string(&input_filename).map_err(|e| Error::Io {
        source: e,
        path: input_filename.clone(),
    })?;
    Ok(input)
}

pub struct StopAfter {
    pub lex: bool,
    pub parse: bool,
    pub semantics: bool,
    pub tacky: bool,
    pub codegen: bool,
}

pub fn do_the_thing(
    input: &str,
    input_filename: PathBuf,
    output_filename: Option<PathBuf>,
    stop_after: StopAfter,
) -> Result<(), Error> {
    log::info!("Lexing input file: {}", input_filename.display());
    let lexed = lexer::lex(input)?;

    log::debug!("Lexed input: {lexed:#?}");

    if stop_after.lex {
        return Ok(());
    }

    log::info!("Parsing input file: {}", input_filename.display());
    let mut ast = parser::parse(&lexed)?;

    log::debug!("AST: {ast:#?}");

    if stop_after.parse {
        return Ok(());
    }

    log::info!("Semantic analysis");
    semantics::analyse(&mut ast)?;

    log::debug!("Semantics AST: {ast:#?}");

    if stop_after.semantics {
        return Ok(());
    }

    let tacky = tacky::emit_program(&ast)?;

    log::debug!("TACKY: {tacky:#?}");

    if stop_after.tacky {
        return Ok(());
    }

    let assembly = codegen::parse(&tacky)?;

    if stop_after.codegen {
        return Ok(());
    }

    emitter::emit(
        assembly,
        output_filename.unwrap_or(input_filename.with_extension(".s")),
    )?;

    Ok(())
}

#[cfg(test)]
fn lex_and_parse(input: &str) -> Result<ast_c::Program, Error> {
    Ok(parser::parse(&lexer::lex(input)?)?)
}

#[cfg(test)]
fn lex_parse_and_analyse(input: &str) -> Result<ast_c::Program, Error> {
    let mut ast = parser::parse(&lexer::lex(input)?)?;
    semantics::analyse(&mut ast)?;
    Ok(ast)
}

#[cfg(test)]
fn lex_parse_analyse_and_codegen(input: &str) -> Result<ast_asm::Program, Error> {
    let lexed = lexer::lex(input)?;
    let mut ast = parser::parse(&lexed)?;
    semantics::analyse(&mut ast)?;
    let tacky = tacky::emit_program(&ast)?;
    let asm = codegen::parse(&tacky)?;
    Ok(asm)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ast_c::{Block, BlockItem, Expression, Function, Program, Statement};
    use crate::parser::ParserError;
    use assert_matches::assert_matches;

    use std::sync::Once;
    static INIT: Once = Once::new();

    #[allow(dead_code)]
    pub fn init_logger() {
        INIT.call_once(|| {
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
                .is_test(true)
                .try_init()
                .ok();
        });
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
            lex_and_parse(input).unwrap(),
            Program {
                function: Function {
                    name: "main".into(),
                    body: Block {
                        items: vec![BlockItem::S(Statement::Return(Expression::Constant(2)))]
                    },
                }
            }
        );
    }

    #[test]
    fn test_parse_error_incomplete_identifier() {
        let input = r#"int "#;
        assert_matches!(
            lex_and_parse(input).unwrap_err(),
            Error::Parser(ParserError {
                message: _,
                expected,
                found,
                offset
            }) if expected == "identifier" && found == "EOF" && offset == 3
        );
    }

    #[test]
    fn test_parse_error_incomplete_keyword() {
        let input = r#"int main("#;
        assert_matches!(
            lex_and_parse(input).unwrap_err(),
            Error::Parser(ParserError {
                message: _,
                expected,
                found,
                offset
            }) if expected == "keyword" && found == "EOF" && offset == 9
        );
    }

    #[test]
    fn test_parse_error_incomplete_integer() {
        let input = r#"int main(void) { return"#;
        assert_matches!(
            lex_and_parse(input).unwrap_err(),
            Error::Parser(ParserError {
                message: _,
                expected,
                found,
                offset
            }) if expected == "expression" && found == "EOF" && offset == 23
        );
    }

    #[test]
    fn test_parse_error_switched_parens() {
        let input = r#"int main )( { return 0; }"#;
        assert_matches!(
            lex_and_parse(input).unwrap_err(),
            Error::Parser(ParserError {
                message: _,
                expected,
                found,
                offset
            }) if expected == "`(`" && found == "CloseParen" && offset == 9
        );
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

    #[test]
    fn test_codegen() {
        use ast_asm::{Function, Instruction, Operand, Program};
        // Listing on page 4, AST on page 13
        let input = r#"
        int main(void) {
            return 2;
        }
        "#;
        assert_eq!(
            lex_parse_analyse_and_codegen(input).unwrap(),
            Program {
                function_definition: Function {
                    name: "main".into(),
                    instructions: vec![
                        Instruction::AllocateStack(0),
                        Instruction::Mov {
                            src: Operand::Imm(2),
                            dst: Operand::Reg(ast_asm::Reg::AX),
                        },
                        Instruction::Ret,
                        // Default Return(0) at end of every function
                        Instruction::Mov {
                            src: Operand::Imm(0),
                            dst: Operand::Reg(ast_asm::Reg::AX),
                        },
                        Instruction::Ret,
                    ]
                }
            }
        );
    }

    // Chapter 2
    #[test]
    fn test_invalid_decrement_constant() {
        // Listing 2-3, Page 32
        let input = r#"
        int main(void) {
            return --2;
        }"#;
        assert_matches!(
            lex_and_parse(input).unwrap_err(),
            Error::Parser(ParserError {
                message: _,
                expected,
                found,
                offset
            }) if expected == "expression" && found == "Decrement" && offset == 45
        );
    }

    // TODO: Add integration tests - full .c to .s compilation

    #[test]
    fn test_invalid_assign_to_constant() {
        // Page 104
        let input = r#"
        int main(void) {
            2 = a * 3;
        }"#;
        assert_matches!(
            lex_parse_and_analyse(input).unwrap_err(),
            Error::Semantics(semantics::Error::InvalidLValue)
        );
    }

    #[test]
    fn test_invalid_declare_variable_twice() {
        // Page 104
        let input = r#"
        int main(void) {
            int a = 3;
            int a;
        }"#;
        assert_matches!(
            lex_parse_and_analyse(input).unwrap_err(),
            Error::Semantics(semantics::Error::DuplicateVariableDeclaration(v))
            if v == "a"
        );
    }

    #[test]
    fn test_invalid_not_declared() {
        // Page 104
        let input = r#"
        int main(void) {
            a = 4;
            return a;
        }"#;
        assert_matches!(
            lex_parse_and_analyse(input).unwrap_err(),
            Error::Semantics(semantics::Error::UndeclaredVariable(v))
            if v == "a"
        );
    }
}
