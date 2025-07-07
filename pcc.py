#!/usr/bin/env python3
"""
Compiler Driver script
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--debug", action="store_const", dest="loglevel", const=logging.DEBUG,
                        default=logging.WARNING, help="Show debug output")
    parser.add_argument("-v", "--verbose", action="store_const", dest="loglevel", const=logging.INFO,
                        help="Show more output")

    parser.add_argument("filenames", nargs="+", help="Path to the file(s) to be compiled")
    parser.add_argument("--lex", action="store_true", help="Run up to the lexer output only")
    parser.add_argument("--parse", action="store_true", help="Run up to the parser output only")
    parser.add_argument("--validate", action="store_true", help="Run up to the semantic analysis output only")
    parser.add_argument("--tacky", action="store_true", help="Run up to the tacky generation output only")
    parser.add_argument("--codegen", action="store_true", help="Run up to codegen output only")
    parser.add_argument("-o", "--output", help="Path to the compiled binary (optional)", default=None)
    parser.add_argument("-S", "--asm", action="store_true", help="Generate assembly code")
    parser.add_argument("-c", action="store_true", dest="skip_link", help="Generate object code, skip linker")

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    preprocessed_file = None
    assembly_files = []

    for filename in args.filenames:
        # If an error occurs, delete all intermediate files
        try:
            preprocessed_file = preprocess(Path(filename))
            assembly_files.append(compile(preprocessed_file, args.lex, args.parse, args.validate, args.tacky, args.codegen))
        finally:
            if preprocessed_file is not None:
                preprocessed_file.unlink(missing_ok=True)

    try:
        if not (args.lex or args.parse or args.validate or args.tacky or args.codegen):
            assemble_and_link(assembly_files, args.output, args.skip_link)
    finally:
        if not args.asm:
            for assembly_file in assembly_files:
                assembly_file.unlink(missing_ok=True)


def preprocess(filename: Path) -> Path:
    """
    Generate .i file from the source code.
    """
    target = filename.with_suffix('.i')
    # TODO: remove the -P option to keep line numbers in the output
    cmd = f"gcc -E -P {filename} -o {target}"
    logger.debug(f"Preprocessing command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True)
    if result.returncode != 0:
        logger.error(f"Preprocessing failed: {result.stderr.decode()}")
        raise RuntimeError("Preprocessing failed")
    return target


def compile(filename: Path,
            stop_after_lex: bool,
            stop_after_parse: bool,
            stop_after_validate: bool,
            stop_after_tacky: bool,
            stop_after_codegen: bool) -> Path:
    target = filename.with_suffix('.s')
    logger.debug(f"Compiling {target}")

    cmd = f"target/debug/compile {filename} -o {target}"

    if stop_after_lex:
        cmd += " --lex"
    if stop_after_parse:
        cmd += " --parse"
    if stop_after_validate:
        cmd += " --validate"
    if stop_after_tacky:
        cmd += " --tacky"
    if stop_after_codegen:
        cmd += " --codegen"

    match logger.getEffectiveLevel():
        case logging.DEBUG:
            cmd += " --debug"
        case logging.INFO:
            cmd += " --verbose"
        case _:
            pass

    logger.debug(f"Compiler command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True)
    if result.returncode != 0:
        logger.error(f"Compile failed: {result.stderr.decode()}")
        raise RuntimeError("Compile failed")
    else:
        if logger.getEffectiveLevel() <= logging.INFO:
            print(result.stderr.decode(), file=sys.stderr)

    return target


def assemble_and_link(filenames: list[Path], output_file: Path | None, skip_link: bool = False):
    if output_file is None:
        if len(filenames) == 1:
            output_file = filenames[0].with_suffix(".o" if skip_link else "")
        else:
            output_file = "a.out"
    #cmd = f"gcc -arch x86_64 {filename} -o {output_file}"
    filenames_s = " ".join((str(f) for f in filenames))
    cmd = f"gcc {'-c' if skip_link else ''} {filenames_s} -o {output_file}"
    logger.debug(f"Assemble and link command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True)
    if result.returncode != 0:
        logger.error(f"Assemble and link failed: {result.stderr.decode()}")
        raise RuntimeError("Assemble and link failed")


if __name__ == '__main__':
    main()
