mod ast_asm;
mod ast_c;
mod codegen;
mod emitter;
mod id_gen;
mod lexer;
mod parser;
mod semantics;
mod tacky;

use crate::lexer::Token;
use crate::semantics::SymbolTable;
use log::trace;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O: {path}")]
    Io {
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("Command error")]
    Command(String),

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

pub fn read_input(input_filename: &Path) -> Result<String, Error> {
    // read the input file as a string into memory
    log::info!("Reading input file: {}", input_filename.display());
    let input = fs::read_to_string(input_filename).map_err(|e| Error::Io {
        source: e,
        path: input_filename.into(),
    })?;
    Ok(input)
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum StopAfter {
    Lexing,
    Parsing,
    Semantics,
    Tacky,
    Codegen,
    NoStop,
}

impl StopAfter {
    pub fn will_stop(self) -> bool {
        self != StopAfter::NoStop
    }
}

pub fn do_the_thing(
    input: &str,
    input_filename: &Path,
    output_filename: Option<&Path>,
    stop_after: StopAfter,
) -> Result<(), Error> {
    let outputs = compile_assembly(input, input_filename, stop_after)?;

    trace!("{outputs:#?}");

    if stop_after == StopAfter::Codegen {
        return Ok(());
    }

    let (assembly, symbol_table) = if let Some(assembly) = outputs.assembly
        && let Some(symbol_table) = outputs.symbol_table
    {
        (assembly, symbol_table)
    } else {
        return Ok(());
    };

    // if output_filename is none, set it to input filename with .s extension
    let output_filename = match output_filename {
        Some(path) => path.to_path_buf(),
        None => {
            let mut path = input_filename.to_path_buf();
            path.set_extension("s");
            path
        }
    };

    emitter::emit(assembly, symbol_table, output_filename)?;

    Ok(())
}

#[allow(dead_code)]
#[derive(Debug)]
struct CompileOutputs {
    lexed: Option<Vec<Token>>,
    ast: Option<ast_c::Program>,
    symbol_table: Option<SymbolTable>,
    tacky: Option<tacky::Program>,
    assembly: Option<ast_asm::Program>,
}

fn compile_assembly(
    input: &str,
    input_filename: &Path,
    stop_after: StopAfter,
) -> Result<CompileOutputs, Error> {
    log::info!("Lexing input file: {}", input_filename.display());
    let lexed = lexer::lex(input)?;

    //log::debug!("Lexed input: {lexed:#?}");

    if stop_after == StopAfter::Lexing {
        return Ok(CompileOutputs {
            lexed: Some(lexed),
            ast: None,
            symbol_table: None,
            tacky: None,
            assembly: None,
        });
    }

    log::info!("Parsing input file: {}", input_filename.display());
    let mut ast = parser::parse(&lexed)?;

    //log::debug!("AST: {ast:#?}");

    if stop_after == StopAfter::Parsing {
        return Ok(CompileOutputs {
            lexed: Some(lexed),
            ast: Some(ast),
            symbol_table: None,
            tacky: None,
            assembly: None,
        });
    }

    log::info!("Semantic analysis");
    let symbol_table = semantics::analyse(&mut ast)?;

    //log::debug!("Semantics AST: {ast:#?}");

    if stop_after == StopAfter::Semantics {
        return Ok(CompileOutputs {
            lexed: Some(lexed),
            ast: Some(ast),
            symbol_table: Some(symbol_table),
            tacky: None,
            assembly: None,
        });
    }

    let tacky = tacky::emit_program(&ast, &symbol_table)?;

    //log::debug!("TACKY: {tacky:#?}");

    if stop_after == StopAfter::Tacky {
        return Ok(CompileOutputs {
            lexed: Some(lexed),
            ast: Some(ast),
            symbol_table: Some(symbol_table),
            tacky: Some(tacky),
            assembly: None,
        });
    }

    let assembly = codegen::parse(&tacky, &symbol_table)?;

    Ok(CompileOutputs {
        lexed: Some(lexed),
        ast: Some(ast),
        symbol_table: Some(symbol_table),
        tacky: Some(tacky),
        assembly: Some(assembly),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emitter::abi::ALIGN_DIRECTIVE;
    use crate::emitter::abi::INDIRECT_CALL_SUFFIX;
    use emitter::abi::{FOOTER, PUBLIC_PREFIX};
    use std::io::BufWriter;

    use crate::ast_c::{Block, BlockItem, Expression, FunDecl, Program, Statement};
    use crate::parser::ParserError;
    use crate::semantics::SymbolTable;
    use assert_matches::assert_matches;
    use assertables::{assert_eq_as_result, assert_ok};
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn lex_and_parse(input: &str) -> Result<Program, Error> {
        let outputs = compile_assembly(input, Path::new("input.c"), StopAfter::Parsing)?;

        Ok(outputs
            .ast
            .expect("AST should be present after lexing and parsing"))
    }

    fn lex_parse_and_analyse(input: &str) -> Result<(Program, SymbolTable), Error> {
        let outputs = compile_assembly(input, Path::new("input.c"), StopAfter::Semantics)?;

        Ok((
            outputs
                .ast
                .expect("AST should be present after lexing and parsing"),
            outputs
                .symbol_table
                .expect("Symbol table should be present after semantic analysis"),
        ))
    }

    fn lex_parse_analyse_and_codegen(input: &str) -> Result<ast_asm::Program, Error> {
        let outputs = compile_assembly(input, Path::new("input.c"), StopAfter::Codegen)?;
        Ok(outputs
            .assembly
            .expect("Assembly should be present after codegen"))
    }

    fn full_compile(input: &str) -> Result<String, Error> {
        let outputs = compile_assembly(input, Path::new("input.c"), StopAfter::Codegen)?;

        let buffer = Vec::new();
        let mut writer = BufWriter::new(buffer);

        assert!(
            emitter::write_out(
                outputs
                    .assembly
                    .expect("Assembly should be present after codegen"),
                outputs
                    .symbol_table
                    .expect("Symbol table should be present after semantic analysis"),
                &mut writer
            )
            .is_ok()
        );
        let result = String::from_utf8(writer.into_inner().unwrap()).unwrap();
        Ok(result)
    }

    pub(crate) fn listing_is_equivalent(listing: &str, expected: &str) -> Result<(), String> {
        let listing = listing
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(str::trim)
            .collect::<Vec<_>>(); //.join("\n");
        let expected = expected
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(str::trim)
            .collect::<Vec<_>>(); //.join("\n");

        assert_eq_as_result!(listing.len(), expected.len())?;

        for (actual, expected) in listing.iter().zip(expected) {
            asm_line_equivalent(actual, expected)?;
            // if actual != expected {
            //     log::debug!("Mismatch:\nActual: {actual}\nExpected: {expected}");
            //     return false;
            // }
        }
        Ok(())
    }

    fn asm_line_equivalent(line: &str, expected: &str) -> Result<(), String> {
        let line = line.trim();
        let expected = expected.trim();

        let line_parts = line.split_whitespace().collect::<Vec<_>>();
        let expected_parts = expected.split_whitespace().collect::<Vec<_>>();

        assert_eq_as_result!(line_parts, expected_parts)
    }

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
                declarations: vec![ast_c::Declaration::FunDecl(FunDecl {
                    name: "main".into(),
                    params: vec![],
                    body: Some(Block {
                        items: vec![BlockItem::S(Statement::Return(Expression::Constant(2)))]
                    }),
                    storage_class: None,
                })]
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
    fn test_parse_error_function_no_parameters() {
        // This is legal C99, but we don't support it yet, as it means an
        // unspecified number of parameters.
        let input = r#"int main() { }"#;
        assert_matches!(
            lex_and_parse(input).unwrap_err(),
            Error::Parser(ParserError {
                message: _,
                expected,
                found,
                offset
            }) if expected == "keyword" && found == "CloseParen" && offset == 9
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
        use ast_asm::{Function, Instruction, Operand, Program, TopLevel};
        // Listing on page 4, AST on page 13
        let input = r#"
        int main(void) {
            return 2;
        }
        "#;
        assert_eq!(
            lex_parse_analyse_and_codegen(input).unwrap(),
            Program {
                top_level: vec![TopLevel::Function(Function {
                    name: "main".into(),
                    global: true,
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
                    ],
                    stack_size: Some(0),
                })]
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
            Error::Semantics(semantics::Error::ConflictingVariableDeclaration(v))
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

    #[test]
    fn test_multiple_func_decls_same_scope() {
        // Listing 9-17: valid
        let input = r#"
        int main(void) {
            int foo(void);
            int foo(void);
            return foo();
        }"#;
        assert!(lex_parse_and_analyse(input).is_ok());
    }

    #[test]
    fn test_function_parameter_shadows_variable() {
        // Page 177
        let input = r#"
        int main(void) {
            int a;
            int foo(int a);
        }"#;
        assert!(lex_parse_and_analyse(input).is_ok());
    }

    #[test]
    fn test_invalid_function_declaration() {
        // Page 177
        let input = r#"
        int main(void) {
            int foo(int a) {
                return a;
            }
        }"#;
        assert_matches!(
            lex_parse_and_analyse(input).unwrap_err(),
            Error::Semantics(semantics::Error::InvalidFunctionDefinition(v))
            if v == "foo"
        );
    }

    #[test]
    fn test_invalid_function_undeclared() {
        // Page 177
        let input = r#"
        int main(void) {
            foo(42);
        }"#;
        assert_matches!(
            lex_parse_and_analyse(input).unwrap_err(),
            Error::Semantics(semantics::Error::UndeclaredFunction(v))
            if v == "foo"
        );
    }

    #[test]
    fn test_invalid_function_duplicate_parameters() {
        // Page 177
        let input = r#"
        int main(void) {
            int foo(int a, int a);
        }"#;
        assert_matches!(
            lex_parse_and_analyse(input).unwrap_err(),
            Error::Semantics(semantics::Error::DuplicateFunctionParameter(v))
            if v == "a"
        );
    }

    #[test]
    fn test_invalid_function_duplicate_declaration() {
        // Page 177
        let input = r#"
        int foo(int a) {
            int a = 3;
            return a;
        }"#;
        assert_matches!(
            lex_parse_and_analyse(input).unwrap_err(),
            Error::Semantics(semantics::Error::ConflictingVariableDeclaration(v))
            if v == "a"
        );
    }

    #[test]
    fn test_invalid_function_call() {
        // Page 179
        let input = r#"
        int main(void) {
            int x = 3;
            return x();
        }"#;
        assert_matches!(
            lex_parse_and_analyse(input).unwrap_err(),
            Error::Semantics(semantics::Error::InvalidFunctionCall(v))
            if v == "x.0"
        );
    }

    #[test]
    fn test_invalid_function_mismatched_declaration() {
        // Page 179
        let input = r#"
        int foo(int a, int b);
        int foo(int a);
        "#;
        assert_matches!(
            lex_parse_and_analyse(input).unwrap_err(),
            Error::Semantics(semantics::Error::IncompatibleFunctionDeclaration(v))
            if v == "foo"
        );
    }

    #[test]
    fn test_invalid_function_mismatched_nested_declaration() {
        // Page 181
        let input = r#"
        int main(void) {
            int foo(int a);
            return foo(1);
        }

        int foo(int a, int b);
        "#;
        assert_matches!(
            lex_parse_and_analyse(input).unwrap_err(),
            Error::Semantics(semantics::Error::IncompatibleFunctionDeclaration(v))
            if v == "foo"
        );
    }

    #[test]
    fn test_invalid_function_wrong_number_parameters() {
        // Page 179
        let input = r#"
        int foo(int a, int b);

        int main(void) {
            return foo(1);
        }
        "#;
        assert_matches!(
            lex_parse_and_analyse(input).unwrap_err(),
            Error::Semantics(semantics::Error::MismatchedFunctionArguments(v))
            if v == "foo"
        );
    }

    #[test]
    fn test_invalid_function_multiple_definitions() {
        // Page 179
        let input = r#"
        int foo(void) {
            return 1;
        }
        int foo(void) {
            return 2;
        }"#;
        assert_matches!(
            lex_parse_and_analyse(input).unwrap_err(),
            Error::Semantics(semantics::Error::MultipleDefinitions(v))
            if v == "foo"
        );
    }

    #[test]
    fn test_listing_9_29() {
        // Listing 9-29 - include Mov at start of function body
        let input = r#"
        int simple(int param) {
           return param;
        }
        "#;

        let listing = full_compile(input);
        assert_ok!(&listing);

        let simple = format!("{PUBLIC_PREFIX}simple");

        assert_ok!(listing_is_equivalent(
            &listing.unwrap(),
            &format!(
                r#"
                .globl {simple}
                .text
            {simple}:
                pushq %rbp
                movq %rsp, %rbp
                subq $16, %rsp
                movl %edi, -4(%rbp)
                movl -4(%rbp), %eax
                movq %rbp, %rsp
                popq %rbp
                ret
                movl $0, %eax
                movq %rbp, %rsp
                popq %rbp
                ret
                {FOOTER}
            "#
            )
        ));
    }

    #[test]
    fn test_file_scope_definitions() {
        let input = r#"
        int a;
        int b = 3;
        extern int c;
        static int d = 4;

        extern int external(void);

        static int simple(int param) {
           return param;
        }

        int main(void) {
            return simple(a + b + c + d + external());
        }
        "#;

        let listing = full_compile(input);
        assert_ok!(&listing);

        let simple = format!("{PUBLIC_PREFIX}simple");
        let main = format!("{PUBLIC_PREFIX}main");

        assert_ok!(listing_is_equivalent(
            &listing.unwrap(),
            &format!(
                r#"
                .text
            {simple}:
                pushq   %rbp
                movq    %rsp, %rbp
                subq    $16, %rsp
                movl    %edi, -4(%rbp)
                movl    -4(%rbp), %eax
                movq    %rbp, %rsp
                popq    %rbp
                ret
                movl    $0, %eax
                movq    %rbp, %rsp
                popq    %rbp
                ret
                .globl  {main}
                .text
            {main}:
                pushq   %rbp
                movq    %rsp, %rbp
                subq    $32, %rsp
                movl    _a(%rip), %r10d
                movl    %r10d, -4(%rbp)
                movl    _b(%rip), %r10d
                addl    %r10d, -4(%rbp)
                movl    -4(%rbp), %r10d
                movl    %r10d, -8(%rbp)
                movl    _c(%rip), %r10d
                addl    %r10d, -8(%rbp)
                movl    -8(%rbp), %r10d
                movl    %r10d, -12(%rbp)
                movl    _d(%rip), %r10d
                addl    %r10d, -12(%rbp)
                call    _external{INDIRECT_CALL_SUFFIX}
                movl    %eax, -16(%rbp)
                movl    -12(%rbp), %r10d
                movl    %r10d, -20(%rbp)
                movl    -16(%rbp), %r10d
                addl    %r10d, -20(%rbp)
                movl    -20(%rbp), %edi
                call    _simple{INDIRECT_CALL_SUFFIX}
                movl    %eax, -24(%rbp)
                movl    -24(%rbp), %eax
                movq    %rbp, %rsp
                popq    %rbp
                ret
                movl    $0, %eax
                movq    %rbp, %rsp
                popq    %rbp
                ret
                .globl  _a
                .bss
                {ALIGN_DIRECTIVE} 4
            _a:
                .zero   4
                .globl  _b
                .data
                {ALIGN_DIRECTIVE} 4
            _b:
                .long   3
                .data
                {ALIGN_DIRECTIVE} 4
            _d:
                .long   4
                {FOOTER}
            "#
            )
        ));
    }
}
