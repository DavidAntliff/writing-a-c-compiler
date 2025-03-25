mod ast;
mod lexer;
mod parser;

use clap::Parser;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Input filename
    input: PathBuf,

    /// Output filename
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Stop after lexing
    #[arg(long)]
    lex: bool,

    /// Stop after parsing
    #[arg(long)]
    parse: bool,

    /// Stop after codegen
    #[arg(long)]
    codegen: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let dummy = "\n\
    .text\n\
    .globl _main\n\
_main:\n\
    movl $0, %eax\n\
    ret\n\
";

    // read the input file as a string into memory
    let input = fs::read_to_string(&cli.input).unwrap_or_else(|_| {
        eprintln!("Failed to read input file: {}", cli.input.display());
        std::process::exit(1);
    });

    let lexed = lexer::lex(&input).unwrap_or_else(|e| {
        eprintln!("Failed to lex input file: {}", cli.input.display());
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    });

    dbg!(&lexed);

    let parsed = parser::parse(&lexed).unwrap_or_else(|e| {
        eprintln!("Failed to parse input file: {}", cli.input.display());
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    });

    dbg!(&parsed);

    let output_filename = cli.output.unwrap_or_else(|| cli.input.with_extension("s"));

    if let Err(e) = fs::write(&output_filename, dummy) {
        eprintln!("Failed to write to file: {}", e);
        std::process::exit(1);
    }

    Ok(())
}
