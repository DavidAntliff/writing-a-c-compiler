
## Links

 * [C draft standard](https://www.open-std.org/JTC1/SC22/WG14/www/docs/n2310.pdf)
 * [Compiler Explorer](https://godbolt.org/)
 * [x86-64 ABI](https://gitlab.com/x86-psABIs/x86-64-ABI)
 * [Intel x86-64 Software Developer Manuals](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
 * [x86 & AMD64 instruction reference](https://www.felixcloutier.com/x86/

## x86-64 Assembly

Instructions have a suffix that indicates the size of the operands. The suffixes include:

 * "long word" / "l" means 32-bit word,
 * "quad word" / "q" means 64-bit word.

Return register is usually `eax` or `rax` for 32-bit and 64-bit respectively.

AT&T syntax puts the source before the destination, while Intel syntax puts the destination before the source.

 * AT&T syntax: `movl %eax, %ebx` means "move the value in `eax` to `ebx`", 
 * Intel syntax: `mov ebx, eax`. 

The AT&T syntax also uses a `$` prefix for immediate values and a `%` prefix for registers.
In Intel syntax, immediate values are written without a prefix.

On Mac OS, the `main` symbol must be called `_main`.

On Linux, the line `.section .note.GNU-stack,"",@progbits` is used to mark the stack as not requiring an
executable stack, thereby allowing an executable stack to be disabled.



