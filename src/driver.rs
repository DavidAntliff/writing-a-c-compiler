use anyhow::anyhow;
use clap::Parser;
use env_logger::Env;
use line_numbers::LinePositions;
use log::{debug, info};
use pcc::{do_the_thing, Error, StopAfter};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(short = 'd', long = "debug", action)]
    debug: bool,

    #[arg(short = 'v', long = "verbose", action)]
    verbose: bool,

    #[arg(short = 'q', long = "quiet", action)]
    quiet: bool,

    /// Path to the file(s) to be compiled
    #[arg(value_name = "FILE", required = true, num_args = 1..)]
    input: Vec<PathBuf>,

    /// Path to the compiled binary (optional)
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Stop after lexing
    #[arg(long)]
    lex: bool,

    /// Stop after parsing
    #[arg(long)]
    parse: bool,

    /// Stop after semantic analysis (validation)
    #[arg(long)]
    validate: bool,

    /// Stop after tacky generation
    #[arg(long)]
    tacky: bool,

    /// Stop after codegen
    #[arg(long)]
    codegen: bool,

    /// Generate assembly code
    #[arg(short = 'S', long = "asm")]
    asm: bool,

    /// Generate object code, skip linker
    #[arg(short = 'c')]
    skip_link: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let default_level = match (cli.debug, cli.verbose, cli.quiet) {
        (_, _, true) => "error",
        (true, _, _) => "debug",
        (_, true, _) => "info",
        (_, _, _) => "warn",
    };
    env_logger::Builder::from_env(Env::default().default_filter_or(default_level)).init();

    let stop_after = StopAfter {
        lex: cli.lex,
        parse: cli.parse,
        semantics: cli.validate,
        tacky: cli.tacky,
        codegen: cli.codegen,
    };

    let mut assembly_files = Vec::new();

    for input_filename in &cli.input {
        let assembly_file = process_file(input_filename, &stop_after)?;
        assembly_files.push(assembly_file);
    }

    if !stop_after.will_stop() {
        let result = assemble_and_link(&assembly_files, &cli.output, cli.skip_link);

        if !cli.asm {
            for assembly_file in &assembly_files {
                let _ = std::fs::remove_file(assembly_file);
            }
        }

        result?;
    }

    Ok(())
}

// {
//             Ok(x) => x,
//             Err(Error::Parser(e)) => {
//                 let line_positions = LinePositions::from(input_filename.into());
//                 let (line_num, column) = line_positions.from_offset(e.offset);
//                 log::error!(
//                     "Parser error at line {line_num}, column {column}: {e}",
//                     line_num = line_num.display(),
//                     column = column + 1
//                 );
//                 std::process::exit(1);
//
//             }
//             Err(e) => {
//                 log::error!("Error: {e}");
//                 std::process::exit(1);
//                 ()
//             }
//         };

fn process_file(input_filename: &Path, stop_after: &StopAfter) -> anyhow::Result<PathBuf> {
    let preprocessed_input = preprocess(input_filename)?;
    let input = pcc::read_input(&preprocessed_input)?;
    let assembly_file = compile(&input, &preprocessed_input, stop_after).map_err(|e| match e {
        Error::Parser(e) => {
            let line_positions = LinePositions::from(input.as_str());
            let (line_num, column) = line_positions.from_offset(e.offset);
            anyhow!(
                "Parser error at line {line_num}, column {column}: {e}",
                line_num = line_num.display(),
                column = column + 1
            )
        }
        e => {
            anyhow!("Error: {e}")
        }
    });
    let _ = std::fs::remove_file(preprocessed_input);
    assembly_file
}

/// Generate .i files from the input files.
fn preprocess(input_filename: &Path) -> Result<PathBuf, Error> {
    let mut preprocessed_filename = input_filename.to_path_buf();
    preprocessed_filename.set_extension("i");

    info!(
        "Preprocessing input file: {} -> {}",
        input_filename.display(),
        preprocessed_filename.display()
    );

    // TODO: remove the -P option to keep line numbers in the output
    let cmd = format!(
        "gcc -E -P {} -o {}",
        input_filename.display(),
        preprocessed_filename.display()
    );
    debug!("Preprocess command: {cmd}");

    std::process::Command::new("sh")
        .arg("-c")
        .arg(&cmd)
        .status()
        .map_err(|_| Error::Command(cmd))?;

    Ok(preprocessed_filename)
}

fn compile(input: &str, input_filename: &Path, stop_after: &StopAfter) -> Result<PathBuf, Error> {
    info!("Compiling input file: {}", input_filename.display());

    // .map_err(|e| {
    //     log::error!("Error reading input file: {e}");
    //     std::process::exit(1);
    // })?;

    // TODO: Due to the preprocessor not emitting #line directives, the reported
    //       line numbers in errors may be incorrect. This is a known issue due to
    //       the preprocessor using the -P option.

    let mut output_filename = input_filename.to_path_buf();
    output_filename.set_extension("s");

    do_the_thing(input, input_filename, Some(&output_filename), stop_after)?;

    Ok(output_filename)
}

fn assemble_and_link(
    assembly_files: &[PathBuf],
    output_filename: &Option<PathBuf>,
    skip_link: bool,
) -> Result<(), Error> {
    let output_filename = match output_filename {
        Some(filename) => filename.clone(),
        None => {
            if assembly_files.len() == 1 {
                assembly_files[0].with_extension(if skip_link { "o" } else { "" })
            } else {
                "a.out".into()
            }
        }
    };

    info!(
        "Assemble and link {} -> {}",
        assembly_files
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", "),
        if skip_link {
            "object file".to_string()
        } else {
            output_filename.display().to_string()
        }
    );

    let filenames = assembly_files
        .iter()
        .map(|f| f.to_string_lossy())
        .collect::<Vec<_>>()
        .join(" ");

    let cmd = format!(
        "gcc {c} {filenames} {out}",
        c = if skip_link { "-c" } else { "" },
        out = if skip_link {
            "".into()
        } else {
            format!("-o {}", output_filename.display())
        },
    );

    debug!("Assemble and link command: {cmd}");

    std::process::Command::new("sh")
        .arg("-c")
        .arg(&cmd)
        .status()
        .map_err(|_| Error::Command(cmd))?;

    Ok(())
}
