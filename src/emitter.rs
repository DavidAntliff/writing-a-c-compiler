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

fn write_out(assembly: ast_asm::Program, writer: &mut BufWriter<File>) -> std::io::Result<()> {
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
        write!(writer, "")?;
        match instruction {
            Instruction::Mov { src, dst } => {
                writeln!(writer, "{indent}movl\t{src}, {dst}")?;
            }
            Instruction::Ret => {
                writeln!(writer, "{indent}movq\t%rbp, %rsp")?;
                writeln!(writer, "{indent}popq\t%rbp")?;
                writeln!(writer, "{indent}ret")?;
            }
            Instruction::Unary { op, dst } => {
                writeln!(writer, "{indent}{op}\t{dst}")?;
            }
            Instruction::AllocateStack(n) => {
                writeln!(writer, "{indent}subq\t${n}, %rsp")?;
            }
        }
    }

    if cfg!(target_os = "linux") {
        writeln!(writer, "{indent}.section\t.note.GNU-stack,\"\",@progbits")?;
    }

    Ok(())
}
