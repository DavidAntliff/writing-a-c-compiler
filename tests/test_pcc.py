import pytest

from pathlib import Path
from dataclasses import dataclass
import subprocess


@dataclass
class CProgram:
    path: Path
    expected_exit_code: int
    expected_stdout: str | None = None


C_PROGRAMS = {
    "return_2": CProgram(Path("tests/c/return_2.c"), 2),
    "return_1+2": CProgram(Path("tests/c/return_1+2.c"), 3),
}


def check_output(c_program: CProgram):
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


@pytest.mark.parametrize("c_program_id", C_PROGRAMS.keys())
def test_c_program(c_program_id):
    check_output(C_PROGRAMS[c_program_id])
