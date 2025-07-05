use crate::ast_asm;
use crate::ast_asm::Instruction;
use crate::semantics::SymbolTable;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use thiserror::Error;

#[cfg(target_os = "linux")]
pub(crate) const LABEL_PREFIX: &str = ".L";
#[cfg(target_os = "macos")]
pub(crate) const LABEL_PREFIX: &str = "L";

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
    for function in assembly.function_definitions {
        write_out_function(function, &symbol_table, writer)?;
    }

    let indent = "\t";

    if cfg!(target_os = "linux") {
        writeln!(writer, "{indent}.section\t.note.GNU-stack,\"\",@progbits")?;
    }

    writer.flush()?;

    Ok(())
}

fn write_out_function<W: Write>(
    function: ast_asm::Function,
    symbol_table: &SymbolTable,
    writer: &mut BufWriter<W>,
) -> std::io::Result<()> {
    let indent = "\t";

    let symbol = &function.name.emit(symbol_table);

    //writeln!(writer, "    .text")?;
    writeln!(writer, "{indent}.globl\t{symbol}")?;
    writeln!(writer, "{symbol}:")?;

    // Allocate stack
    writeln!(writer, "{indent}pushq\t%rbp")?;
    writeln!(writer, "{indent}movq\t%rsp, %rbp")?;

    for instruction in function.instructions {
        if instruction == Instruction::Ret {
            writeln!(writer, "{indent}movq\t%rbp, %rsp")?;
            writeln!(writer, "{indent}popq\t%rbp")?;
        }
        if !matches!(instruction, Instruction::Label(_)) {
            write!(writer, "{indent}")?;
        }
        writeln!(writer, "{}", instruction.emit(symbol_table))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_asm::{BinaryOperator, Function, Offset, Program, Reg, ABI_STACK_ALIGNMENT};
    use crate::ast_asm::{Operand, UnaryOperator};
    use crate::emitter::LABEL_PREFIX;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_emit_instructions() {
        let program = Program {
            function_definitions: vec![Function {
                name: "main".into(),
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
            }],
        };

        // Create buffer to write to
        let buffer = Vec::new();
        let mut writer = BufWriter::new(buffer);

        let symbol_table = SymbolTable::new();
        assert!(write_out(program, symbol_table, &mut writer).is_ok());
        let result = String::from_utf8(writer.into_inner().unwrap()).unwrap();

        let main_prefix = if cfg!(target_os = "macos") { "_" } else { "" };

        let suffix = if cfg!(target_os = "linux") {
            "\t.section\t.note.GNU-stack,\"\",@progbits\n"
        } else {
            r#""#
        };

        assert_eq!(
            result,
            format!(
                r#"	.globl	{main_prefix}main
{main_prefix}main:
	pushq	%rbp
	movq	%rsp, %rbp
	movl	%eax, $2
	notl	-4(%rbp)
	addl	-4(%rbp), %r10d
	imull	$42, %r10d
	idivl	%r10d
	cdq
	shll	$4, -4(%rbp)
	movl	-4(%rbp), %ecx
	sarl	%cl, $42
	cmpl	$2, %r11d
	jmp	{LABEL_PREFIX}label1
	jge	{LABEL_PREFIX}label2
{LABEL_PREFIX}label1:
	setne	%r11b
{LABEL_PREFIX}label2:
	subq	$4, %rsp
	movq	%rbp, %rsp
	popq	%rbp
	ret
{suffix}"#
            )
        )
    }
}
