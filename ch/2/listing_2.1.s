	.globl	main
main:
	pushq	%rbp             # function prologue, setting up the current stack frame
	movq	%rsp, %rbp
	subq	$8, %rsp
	movl	$2, -4(%rbp)     # store 2_u32 in memory at address in rbp - 4
	negl	-4(%rbp)         # negate the 32-bit value at address rbp - 4
	movl	-4(%rbp), %r10d  # copy the result to scratch register r10d
	movl	%r10d, -8(%rbp)  # copy r10d contents to rbp - 8
	notl	-8(%rbp)         # bitwise complement of contents of rbp - 8
	movl	-8(%rbp), %eax   # prepare to return the result
	movq	%rbp, %rsp       # function epilogue, tear down the current stack frame
	popq	%rbp
	ret
