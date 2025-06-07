from unittest.loader import VALID_MODULE_NAME

import pytest

from pathlib import Path
from dataclasses import dataclass
import subprocess


@dataclass
class CProgram:
    path: Path
    expected_exit_code: int | None = None
    expected_stdout: str | None = None

# TODO: provide input (cmdline, stdin, etc.)

# compile with gcc and compare outputs
VALID_C_PROGRAMS = {
    "compound_statements": CProgram(Path("tests/c/valid/compound_statements.c"), 3),
    "conditional": CProgram(Path("tests/c/valid/conditional.c"),),# 3),
    "if_else": CProgram(Path("tests/c/valid/if_else.c"),),# 3),
    "goto": CProgram(Path("tests/c/valid/goto.c"), 9),
    "return_2": CProgram(Path("tests/c/valid/return_2.c"),),# 2),
    "return_1+2": CProgram(Path("tests/c/valid/return_1+2.c"),),# 3),
    "return_logical_ops": CProgram(Path("tests/c/valid/return_logical_ops.c"),),# 1),
}

@dataclass
class ParserError:
    line: int
    column: int
    expected: str
    found: str

    def __str__(self):
        return f"Parser error at line {self.line}, column {self.column}: Expected {self.expected}, found {self.found}"


@dataclass
class InvalidCProgram:
    path: Path
    expected: ParserError


INVALID_C_PROGRAMS = {
    "decrement_constant": InvalidCProgram(Path("tests/c/invalid/decrement_constant.c"),
                                          ParserError(2, 12, "expression", '"Decrement"')),    
}

def check_valid_output(c_program: CProgram):
    """
    Compile and run the C code, check the output.
    """

    # Compile the C code with gcc
    gcc_compile_cmd = f"gcc {c_program.path} -o gcc.{c_program.path.stem}"
    gcc_compile_result = subprocess.run(gcc_compile_cmd, shell=True, capture_output=True)
    assert gcc_compile_result.returncode == 0, f"gcc compilation failed: {gcc_compile_result.stderr.decode()}"

    # Compile the C code with pcc
    pcc_compile_cmd = f"python ./pcc.py {c_program.path} -o pcc.{c_program.path.stem}"
    pcc_compile_result = subprocess.run(pcc_compile_cmd, shell=True, capture_output=True)
    assert pcc_compile_result.returncode == 0, f"pcc compilation failed: {pcc_compile_result.stderr.decode()}"

    # Run the gcc compiled program
    gcc_cmd = f"./gcc.{c_program.path.stem}"
    gcc_result = subprocess.run(gcc_cmd, shell=True, capture_output=True)

    # Run the pcc compiled program
    pcc_cmd = f"./pcc.{c_program.path.stem}"
    pcc_result = subprocess.run(pcc_cmd, shell=True, capture_output=True)

    # Check the exit codes match
    assert gcc_result.returncode == pcc_result.returncode, (
        f"Exit codes do not match: gcc {gcc_result.returncode}, pcc {pcc_result.returncode}"
    )

    # If expected_exit_code is provided, check it
    if c_program.expected_exit_code is not None:
        assert pcc_result.returncode == c_program.expected_exit_code, (
            f"Expected exit code {c_program.expected_exit_code}, got {pcc_result.returncode}"
        )

    # Check the stdout outputs match
    assert gcc_result.stdout.decode() == pcc_result.stdout.decode(), (
        f"stdout does not match:\nGCC: {gcc_result.stdout.decode()}\nPCC: {pcc_result.stdout.decode()}"
    )

    # Check the stderr outputs match
    assert gcc_result.stderr.decode() == pcc_result.stderr.decode(), (
        f"stderr does not match:\nGCC: {gcc_result.stderr.decode()}\nPCC: {pcc_result.stderr.decode()}"
    )

    # Check the output
    if c_program.expected_stdout is not None:
        assert pcc_result.stdout.decode() == c_program.expected_stdout, (
            f"Expected output '{c_program.expected_stdout}', got '{pcc_result.stdout.decode()}'"
        )


def check_error(c_program: InvalidCProgram):
    """
    Compile and run the C code, check for errors.
    """
    # Compile the C code
    compile_cmd = f"python ./pcc.py {c_program.path} -o {c_program.path.stem}"
    result = subprocess.run(compile_cmd, shell=True, capture_output=True)
    
    # Check the exit code
    assert result.returncode != 0, (
        f"Expected compilation to fail, but it succeeded with exit code {result.returncode}"
    )

    # Check the error message
    assert str(c_program.expected) in result.stderr.decode(), (
        f"Expected error message, got: {result.stderr.decode()}"
    )
    

@pytest.mark.parametrize("c_program_id", VALID_C_PROGRAMS.keys())
def test_valid_c_program(c_program_id):
    c_program = VALID_C_PROGRAMS[c_program_id]
    try:
        check_valid_output(c_program)
    finally:
        Path(f"gcc.{c_program.path.stem}").unlink()
        Path(f"pcc.{c_program.path.stem}").unlink()


@pytest.mark.parametrize("c_program_id", INVALID_C_PROGRAMS.keys())
def test_invalid_c_program(c_program_id):
    c_program = INVALID_C_PROGRAMS[c_program_id]
    check_error(c_program)
    assert not Path(c_program.path.stem).exists()
