#!/usr/bin/env python3
"""
Compiler Driver script
"""
import argparse
from pathlib import Path
import subprocess
import sys

import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--debug", action="store_const", dest="loglevel", const=logging.DEBUG,
                        default=logging.WARNING, help="Show debug output")
    parser.add_argument("-v", "--verbose", action="store_const", dest="loglevel", const=logging.INFO,
                        help="Show more output")

    parser.add_argument("filename", help="Path to the file to be compiled")
    parser.add_argument("--lex", action="store_true", help="Run up to the lexer only")
    parser.add_argument("--parse", action="store_true", help="Run up to the parser only")
    parser.add_argument("--codegen", action="store_true", help="Run up to codegen only")
    parser.add_argument("-S", "--asm", action="store_true", help="Generate assembly code")

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    preprocessed_file = None
    assembly_file = None

    # If an error occurs, delete all intermediate files
    try:
        preprocessed_file = preprocess(Path(args.filename))
        assembly_file = compile(preprocessed_file, args.lex, args.parse, args.codegen)
    finally:
        if preprocessed_file is not None:
            preprocessed_file.unlink(missing_ok=True)

    try:
        if not (args.lex or args.parse or args.codegen):
            assemble_and_link(assembly_file)
    finally:
        if not args.asm and assembly_file is not None:
            assembly_file.unlink(missing_ok=True)


def preprocess(filename: Path) -> Path:
    """
    Generate .i file from the source code.
    """
    target = filename.with_suffix('.i')
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
            stop_after_codegen: bool) -> Path:
    target = filename.with_suffix('.s')
    logger.debug(f"Compiling {target}")

    cmd = f"target/debug/pcc {filename} -o {target}"
    if stop_after_lex:
        cmd += " --lex"
    if stop_after_parse:
        cmd += " --parse"
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


def assemble_and_link(filename: Path):
    output_file = filename.with_suffix("")
    cmd = f"gcc {filename} -o {output_file}"
    logger.debug(f"Assemble and link command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True)
    if result.returncode != 0:
        logger.error(f"Assemble and link failed: {result.stderr.decode()}")
        raise RuntimeError("Assemble and link failed")


if __name__ == '__main__':
    main()
