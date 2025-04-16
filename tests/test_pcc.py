from unittest.loader import VALID_MODULE_NAME

import pytest

from pathlib import Path
from dataclasses import dataclass
import subprocess


@dataclass
class CProgram:
    path: Path
    expected_exit_code: int
    expected_stdout: str | None = None


VALID_C_PROGRAMS = {
    "return_2": CProgram(Path("tests/c/valid/return_2.c"), 2),
    "return_1+2": CProgram(Path("tests/c/valid/return_1+2.c"), 3),
    "return_logical_ops": CProgram(Path("tests/c/valid/return_logical_ops.c"), 1),
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
    # Compile the C code
    compile_cmd = f"python ./pcc.py {c_program.path} -o {c_program.path.stem}"
    result = subprocess.run(compile_cmd, shell=True, capture_output=True)
    assert result.returncode == 0, f"Compilation failed: {result.stderr.decode()}"

    # Run the compiled program
    run_cmd = f"./{c_program.path.stem}"
    result = subprocess.run(run_cmd, shell=True, capture_output=True)

    # Check the exit code
    assert result.returncode == c_program.expected_exit_code, (
        f"Expected exit code {c_program.expected_exit_code}, got {result.returncode}"
    )

    # Check the output
    if c_program.expected_stdout is not None:
        assert result.stdout.decode() == c_program.expected_stdout, (
            f"Expected output '{c_program.expected_stdout}', got '{result.stdout.decode()}'"
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
        Path(c_program.path.stem).unlink()


@pytest.mark.parametrize("c_program_id", INVALID_C_PROGRAMS.keys())
def test_invalid_c_program(c_program_id):
    c_program = INVALID_C_PROGRAMS[c_program_id]
    check_error(c_program)
    assert not Path(c_program.path.stem).exists()
