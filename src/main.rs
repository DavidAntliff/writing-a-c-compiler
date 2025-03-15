use std::fs;
use std::path::PathBuf;
use clap::Parser;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Input filename
    input: PathBuf,

    /// Output filename
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,
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

    let output_filename = cli.output.unwrap_or_else(|| {
        cli.input.with_extension("s")
    });

    if let Err(e) = fs::write(&output_filename, dummy) {
        eprintln!("Failed to write to file: {}", e);
        std::process::exit(1);
    }

    Ok(())
}
