#S_SYNTAX := intel
S_SYNTAX := att

all: return_2

return_2.s: return_2.c
	gcc -S -masm=${S_SYNTAX} -O -fno-asynchronous-unwind-tables -fcf-protection=none $<

return_2: return_2.s
	gcc $< -o $@

.PHONY: clean
clean:
	rm -rf return_2 return_2.o return_2.s

