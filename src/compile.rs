use clap::Parser;
use env_logger::Env;
use line_numbers::LinePositions;
use pcc::{do_the_thing, Error, StopAfter};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(short = 'd', long = "debug", action)]
    debug: bool,

    #[arg(short = 'v', long = "verbose", action)]
    verbose: bool,

    #[arg(short = 'q', long = "quiet", action)]
    quiet: bool,

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

    /// Stop after semantic analysis (validation)
    #[arg(long)]
    validate: bool,

    /// Stop after tacky generation
    #[arg(long)]
    tacky: bool,

    /// Stop after codegen
    #[arg(long)]
    codegen: bool,
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

    let input = pcc::read_input(&cli.input).map_err(|e| {
        log::error!("Error reading input file: {e}");
        std::process::exit(1);
    })?;

    let stop_after = StopAfter {
        lex: cli.lex,
        parse: cli.parse,
        semantics: cli.validate,
        tacky: cli.tacky,
        codegen: cli.codegen,
    };

    // TODO: Due to the preprocessor not emitting #line directives, the reported
    //       line numbers in errors may be incorrect. This is a known issue due to
    //       the preprocessor using the -P option.

    match do_the_thing(&input, &cli.input, cli.output.as_deref(), &stop_after) {
        Ok(_) => Ok(()),
        Err(Error::Parser(e)) => {
            let line_positions = LinePositions::from(input.as_str());
            let (line_num, column) = line_positions.from_offset(e.offset);
            log::error!(
                "Parser error at line {line_num}, column {column}: {e}",
                line_num = line_num.display(),
                column = column + 1
            );
            std::process::exit(1);
        }
        Err(e) => {
            log::error!("Error: {e}");
            std::process::exit(1);
        }
    }
}
