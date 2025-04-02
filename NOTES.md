
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

On macOS, the `main` symbol must be called `_main`.

On Linux, the line `.section .note.GNU-stack,"",@progbits` is used to mark the stack as not requiring an
executable stack, thereby allowing an executable stack to be disabled.

### The Stack

Register `RSP` is the stack pointer. It always holds the address of the top of the stack (last used value, not first free).
Push and pop values with `push` and `pop`.

The stack grows down, towards lower memory addresses. Thus, `push` decrements `RSP`.

So the "top" of the stack is the lowest address of valid data on the stack.

For a 64-bit CPU, the stack slot size is 8 bytes. For a 32-bit CPU, it is 4 bytes.

The `push X` instruction:

 * Writes X to the next empty spot on the stack, at `RSP - 8`,
 * Decrements `RSP` by 8 bytes, so that `RSP` now points at the location of value X.

The `pop %reg` instruction:

 * Copies the value at the top of the stack to the register,
 * Increments `RSP` by 8 bytes, so that `RSP` now points to the next value on the stack.

On a 64-bit CPU, you can't push 4 bytes, but you can copy a value to a stack slot you've just allocated.

`pushw` and `popw` works for 2 bytes.


Allocating a stack frame is done by decrementing the stack pointer. Deallocating is the inverse.

`RBP` is the stack base pointer, and points to the base of the current stack frame.

Function prologue:

 * `pushq %rbp` saves the current value of `RBP`, the caller's base pointer, which needs to be restored afterwards,
 * `movq %rsp, %rbp` makes the top of the stack the base of the new stack frame, STATE A,
 * `subq $n, %rsp` decrements the stack pointer by n bytes, for use by the function.

Function epilogue:

 * `movq %rbp, %rsp` deallocates the n bytes by restoring STATE A,
 * `popq %rbp` restores the caller's values for the `RSP` and `RBP` registers.

This leaves `RBP` with the value of the caller's `RBP` before the prologue, and `RSP` pointing to the top of the caller's stack frame.



