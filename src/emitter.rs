use crate::ast_asm;
use crate::ast_asm::{Identifier, Instruction, TopLevel};
use crate::emitter::abi::{ALIGN_DIRECTIVE, FOOTER};
use crate::semantics::SymbolTable;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use thiserror::Error;

// NOTE:
//  • “local” (STB_LOCAL) symbols are only visible inside the current object file (translation unit).
//  • “global” (STB_GLOBAL) symbols are visible to the linker across object files, but
//    don’t necessarily end up in the dynamic symbol table.
//  • “public” or “exported” symbols are those placed in the *dynamic* symbol table
//    (so they can be looked up or interposed at runtime by other binaries/DSOs).

#[cfg(target_os = "linux")]
pub(crate) mod abi {
    pub const PUBLIC_PREFIX: &str = "";
    pub const PRIVATE_PREFIX: &str = ".L";
    pub const INDIRECT_CALL_SUFFIX: &str = "@PLT";
    pub const ALIGN_DIRECTIVE: &str = ".align";
    pub const FOOTER: &str = ".section\t.note.GNU-stack,\"\",@progbits";
}

#[cfg(target_os = "macos")]
pub(crate) mod abi {
    pub const PUBLIC_PREFIX: &str = "_";
    pub const PRIVATE_PREFIX: &str = "L";
    pub const INDIRECT_CALL_SUFFIX: &str = "";
    pub const ALIGN_DIRECTIVE: &str = ".balign";
    pub const FOOTER: &str = "";
}

// #[cfg(target_os = "linux")]
// pub(crate) const EXPORT_SYMBOL_PREFIX: &str = "";
// #[cfg(target_os = "macos")]
// pub(crate) const EXPORT_SYMBOL_PREFIX: &str = "_";
//
// #[cfg(target_os = "linux")]
// pub(crate) const LOCAL_SYMBOL_PREFIX: &str = ".L";
// #[cfg(target_os = "macos")]
// pub(crate) const LOCAL_SYMBOL_PREFIX: &str = "L";

// #[cfg(target_os = "linux")]
// pub(crate) const ALIGN_DIRECTIVE: &str = ".align";
// #[cfg(target_os = "macos")]
// pub(crate) const ALIGN_DIRECTIVE: &str = ".balign";

// #[cfg(target_os = "linux")]
// pub(crate) const EXPORT_FUNCTION_SUFFIX: &str = "@PLT";
// #[cfg(target_os = "macos")]
// pub(crate) const EXPORT_FUNCTION_SUFFIX: &str = "";

const INDENT: &str = "\t";

#[derive(Debug, PartialEq, Error)]
#[error("{message}")]
pub struct EmitterError {
    pub message: String,
}

pub(crate) fn emit(
    assembly: ast_asm::Program,
    symbol_table: SymbolTable,
    output_filename: PathBuf,
) -> Result<(), EmitterError> {
    log::info!("Emitting output file: {}", output_filename.display());

    let file = File::create(&output_filename).map_err(|e| EmitterError {
        message: format!("{e} while writing to {}", output_filename.display()),
    })?;
    let mut writer = BufWriter::new(file);

    write_out(assembly, symbol_table, &mut writer).map_err(|e| EmitterError {
        message: format!("{e} while writing to {}", output_filename.display()),
    })?;

    Ok(())
}

pub(crate) fn write_out<W: Write>(
    assembly: ast_asm::Program,
    symbol_table: SymbolTable,
    writer: &mut BufWriter<W>,
) -> std::io::Result<()> {
    // TODO: explicitly emit functions first?
    for item in assembly.top_level {
        match item {
            TopLevel::Function(function) => {
                write_out_function(function, &symbol_table, writer)?;
            }
            TopLevel::StaticVariable { name, global, init } => {
                write_out_static_variable(name, global, init, writer)?;
            }
        }
    }

    #[allow(clippy::const_is_empty)]
    if !FOOTER.is_empty() {
        writeln!(writer, "{INDENT}{FOOTER}")?;
    }

    writer.flush()?;

    Ok(())
}

fn write_out_function<W: Write>(
    function: ast_asm::Function,
    symbol_table: &SymbolTable,
    writer: &mut BufWriter<W>,
) -> std::io::Result<()> {
    let symbol = &function.name.emit(symbol_table);

    if function.global {
        writeln!(writer, "{INDENT}.globl\t{symbol}")?;
    };
    writeln!(writer, "{INDENT}.text")?;
    writeln!(writer, "{symbol}:")?;

    // Allocate stack
    writeln!(writer, "{INDENT}pushq\t%rbp")?;
    writeln!(writer, "{INDENT}movq\t%rsp, %rbp")?;

    for instruction in function.instructions {
        if instruction == Instruction::Ret {
            writeln!(writer, "{INDENT}movq\t%rbp, %rsp")?;
            writeln!(writer, "{INDENT}popq\t%rbp")?;
        }
        if !matches!(instruction, Instruction::Label(_)) {
            write!(writer, "{INDENT}")?;
        }
        writeln!(writer, "{}", instruction.emit(symbol_table))?;
    }

    Ok(())
}

fn write_out_static_variable<W: Write>(
    name: Identifier,
    global: bool,
    init: i64,
    writer: &mut BufWriter<W>,
) -> std::io::Result<()> {
    let symbol = &name.as_public_symbol();

    if global {
        writeln!(writer, "{INDENT}.globl\t{symbol}")?;
    };

    if init == 0 {
        writeln!(writer, "{INDENT}.bss")?;
        writeln!(writer, "{ALIGN_DIRECTIVE} 4")?;
        writeln!(writer, "{symbol}:")?;
        writeln!(writer, "{INDENT}.zero\t4")?;
    } else {
        writeln!(writer, "{INDENT}.data")?;
        writeln!(writer, "{ALIGN_DIRECTIVE} 4")?;
        writeln!(writer, "{symbol}:")?;
        writeln!(writer, "{INDENT}.long\t{init}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_asm::{
        BinaryOperator, Function, Offset, Program, Reg, TopLevel, ABI_STACK_ALIGNMENT,
    };
    use crate::ast_asm::{Operand, UnaryOperator};
    use crate::emitter::abi::FOOTER;
    use crate::emitter::abi::PRIVATE_PREFIX;
    use crate::emitter::abi::PUBLIC_PREFIX;
    use crate::semantics::{IdentifierAttrs, Type};
    use crate::tests::listing_is_equivalent;
    use assertables::assert_ok;

    #[test]
    fn test_emit_instructions() {
        let program = Program {
            top_level: vec![TopLevel::Function(Function {
                name: "main".into(),
                global: true,
                instructions: vec![
                    Instruction::Mov {
                        src: Operand::Reg(Reg::AX),
                        dst: Operand::Imm(2),
                    },
                    Instruction::Unary {
                        op: UnaryOperator::Not,
                        dst: Operand::Stack(Offset(-4)),
                    },
                    Instruction::Binary {
                        op: BinaryOperator::Add,
                        src: Operand::Stack(Offset(-4)),
                        dst: Operand::Reg(Reg::R10),
                    },
                    Instruction::Binary {
                        op: BinaryOperator::Mult,
                        src: Operand::Imm(42),
                        dst: Operand::Reg(Reg::R10),
                    },
                    Instruction::Idiv(Operand::Reg(Reg::R10)),
                    Instruction::Cdq,
                    Instruction::Binary {
                        op: BinaryOperator::BitShiftLeft,
                        src: Operand::Imm(4),
                        dst: Operand::Stack(Offset(-4)),
                    },
                    Instruction::Mov {
                        src: Operand::Stack(Offset(-4)),
                        dst: Operand::Reg(Reg::CX),
                    },
                    Instruction::Binary {
                        op: BinaryOperator::BitShiftRight,
                        src: Operand::Reg(Reg::CX),
                        dst: Operand::Imm(42),
                    },
                    Instruction::Cmp {
                        src1: Operand::Imm(2),
                        src2: Operand::Reg(Reg::R11),
                    },
                    Instruction::Jmp {
                        target: "label1".into(),
                    },
                    Instruction::JmpCC {
                        cc: ast_asm::ConditionCode::GreaterOrEqual,
                        target: "label2".into(),
                    },
                    Instruction::Label("label1".into()),
                    Instruction::SetCC {
                        cc: ast_asm::ConditionCode::NotEqual,
                        dst: Operand::Reg(Reg::R11),
                    },
                    Instruction::Label("label2".into()),
                    Instruction::AllocateStack(4),
                    Instruction::Ret,
                ],
                stack_size: Some(ABI_STACK_ALIGNMENT),
            })],
        };

        // Create buffer to write to
        let buffer = Vec::new();
        let mut writer = BufWriter::new(buffer);

        let mut symbol_table = SymbolTable::new();
        symbol_table.add(
            "main".into(),
            Type::Function { param_count: 0 },
            IdentifierAttrs::Fun {
                defined: true,
                global: true,
            },
        );

        assert!(write_out(program, symbol_table, &mut writer).is_ok());
        let listing = String::from_utf8(writer.into_inner().unwrap()).unwrap();

        let main = format!("{PUBLIC_PREFIX}main");

        assert_ok!(listing_is_equivalent(
            &listing,
            &format!(
                r#"	.globl  {main}
                    .text
                {main}:
                    pushq   %rbp
                    movq    %rsp, %rbp
                    movl    %eax, $2
                    notl    -4(%rbp)
                    addl    -4(%rbp), %r10d
                    imull   $42, %r10d
                    idivl   %r10d
                    cdq
                    shll    $4, -4(%rbp)
                    movl    -4(%rbp), %ecx
                    sarl    %cl, $42
                    cmpl    $2, %r11d
                    jmp     {PRIVATE_PREFIX}label1
                    jge     {PRIVATE_PREFIX}label2
                {PRIVATE_PREFIX}label1:
                    setne   %r11b
                {PRIVATE_PREFIX}label2:
                    subq    $4, %rsp
                    movq    %rbp, %rsp
                    popq    %rbp
                    ret
                    {FOOTER}"#
            )
        ))
    }
}
