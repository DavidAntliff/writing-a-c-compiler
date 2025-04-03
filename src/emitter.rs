use crate::ast_asm;
use crate::ast_asm::Instruction;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, PartialEq, Error)]
#[error("{message}")]
pub struct EmitterError {
    pub message: String,
}

pub(crate) fn emit(
    assembly: ast_asm::Program,
    output_filename: PathBuf,
) -> Result<(), EmitterError> {
    log::info!("Emitting output file: {}", output_filename.display());

    let file = File::create(&output_filename).map_err(|e| EmitterError {
        message: format!("{e} while writing to {}", output_filename.display()),
    })?;
    let mut writer = BufWriter::new(file);

    write_out(assembly, &mut writer).map_err(|e| EmitterError {
        message: format!("{e} while writing to {}", output_filename.display()),
    })?;

    Ok(())
}

fn write_out<W: Write>(
    assembly: ast_asm::Program,
    writer: &mut BufWriter<W>,
) -> std::io::Result<()> {
    let main_prefix = if cfg!(target_os = "macos") { "_" } else { "" };
    let main_symbol = format!(
        "{main_prefix}{symbol}",
        symbol = assembly.function_definition.name.0
    );

    let indent = "\t";

    //writeln!(writer, "    .text")?;
    writeln!(writer, "{indent}.globl\t{}", main_symbol)?;
    writeln!(writer, "{}:", main_symbol)?;

    // Allocate stack
    writeln!(writer, "{indent}pushq\t%rbp")?;
    writeln!(writer, "{indent}movq\t%rsp, %rbp")?;

    for instruction in assembly.function_definition.instructions {
        if instruction == Instruction::Ret {
            writeln!(writer, "{indent}movq\t%rbp, %rsp")?;
            writeln!(writer, "{indent}popq\t%rbp")?;
        }
        writeln!(writer, "{indent}{instruction}")?;
    }

    if cfg!(target_os = "linux") {
        writeln!(writer, "{indent}.section\t.note.GNU-stack,\"\",@progbits")?;
    }

    writer.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_asm::{BinaryOperator, Function, Identifier, Offset, Program, Reg};
    use crate::ast_asm::{Operand, UnaryOperator};
    use pretty_assertions::assert_eq;

    #[test]
    fn test_emit_instructions() {
        let program = Program {
            function_definition: Function {
                name: Identifier("main".to_string()),
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
                    Instruction::AllocateStack(4),
                    Instruction::Ret,
                ],
            },
        };

        // Create buffer to write to
        let buffer = Vec::new();
        let mut writer = BufWriter::new(buffer);

        assert!(write_out(program, &mut writer).is_ok());
        let result = String::from_utf8(writer.into_inner().unwrap()).unwrap();

        assert_eq!(
            result,
            r#"	.globl	main
main:
	pushq	%rbp
	movq	%rsp, %rbp
	movl	%eax, $2
	notl	-4(%rbp)
	addl	-4(%rbp), %r10d
	imull	$42, %r10d
	idivl	%r10d
	cdq
	subq	$4, %rsp
	movq	%rbp, %rsp
	popq	%rbp
	ret
	.section	.note.GNU-stack,"",@progbits
"#
        )
    }
}
