use clap::Parser;
use env_logger::Env;
use pcc::do_the_thing;
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

    Ok(do_the_thing(
        cli.input,
        cli.output,
        cli.lex,
        cli.parse,
        cli.codegen,
    )?)
}
