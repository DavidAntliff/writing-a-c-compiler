use crate::ast_assembly;
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
    assembly: ast_assembly::Program,
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

fn write_out(assembly: ast_assembly::Program, writer: &mut BufWriter<File>) -> std::io::Result<()> {
    let main_prefix = if cfg!(target_os = "macos") { "_" } else { "" };
    let main_symbol = format!(
        "{main_prefix}{symbol}",
        symbol = assembly.function_definition.name.0
    );

    writeln!(writer, ".text")?;
    writeln!(writer, ".globl {}", main_symbol)?;
    writeln!(writer, "{}:", main_symbol)?;

    for instruction in assembly.function_definition.instructions {
        write!(writer, "    ")?;
        match instruction {
            ast_assembly::Instruction::Mov { src, dst } => {
                writeln!(writer, "movl {src}, {dst}")?;
            }
            ast_assembly::Instruction::Ret => {
                writeln!(writer, "ret")?;
            }
        }
    }

    if cfg!(target_os = "linux") {
        writeln!(writer, ".section .note.GNU-stack,\"\",@progbits")?;
    }

    Ok(())
}
