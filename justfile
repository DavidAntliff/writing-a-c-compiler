# test_compiler also tests previous chapters
default: build test ch11 pytest

test_compiler := 'book-tests/test_compiler'
#cc := './pcc.py'
profile := 'debug'
#profile := 'release'
cc := "target/" + profile + "/pcc"


check-i386:
    #!/usr/bin/env bash
    set -euo pipefail
    arch="$(uname -p)"
    if [[ "$arch" != "i386" && "$arch" != "x86_64" ]]; then
      echo "Error: Expected x86 architecture, got $arch." >&2;
      exit 1;
    fi


#driver-test:
#    rm -f driver/driver driver/driver.i driver/driver.s
#    python3 pcc.py driver/driver.c
#    ./driver ; echo $?


build: check-i386
    cargo build {{ if profile == "release" { "--release" } else { "" } }}


test:
    cargo test {{ if profile == "release" { "--release" } else { "" } }}


pytest: build
    pytest -sv -n 8 tests


ch1: build
    #{{test_compiler}} {{cc}} --chapter 1 --stage lex
    #{{test_compiler}} {{cc}} --chapter 1 --stage parse
    #{{test_compiler}} {{cc}} --chapter 1 --stage codegen
    {{test_compiler}} {{cc}} --chapter 1


listing_1-1: build
    {{cc}} ch/1/listing_1.1.c && ch/1/listing_1.1 ; echo $?


ch2: build
    #{{test_compiler}} {{cc}} --chapter 2 --stage lex
    #{{test_compiler}} {{cc}} --chapter 2 --stage parse
    #{{test_compiler}} {{cc}} --chapter 2 --stage tacky
    #{{test_compiler}} {{cc}} --chapter 2 --stage codegen
    {{test_compiler}} {{cc}} --chapter 2


listing_2-1: build
    {{cc}} ch/2/listing_2.1.c && ch/2/listing_2.1 ; echo $?


ch3: build
    #{{test_compiler}} {{cc}} --chapter 3 --stage lex
    #{{test_compiler}} {{cc}} --chapter 3 --stage parse
    #{{test_compiler}} {{cc}} --chapter 3 --stage tacky
    #{{test_compiler}} {{cc}} --chapter 3 --stage codegen
    {{test_compiler}} {{cc}} --chapter 3 --bitwise


ch4: build
    #{{test_compiler}} {{cc}} --chapter 4 --stage lex
    #{{test_compiler}} {{cc}} --chapter 4 --stage parse
    #{{test_compiler}} {{cc}} --chapter 4 --stage tacky
    #{{test_compiler}} {{cc}} --chapter 4 --stage codegen
    {{test_compiler}} {{cc}} --chapter 4 --bitwise


ch5: build
    #{{test_compiler}} {{cc}} --chapter 5 --stage lex
    #{{test_compiler}} {{cc}} --chapter 5 --stage parse
    #{{test_compiler}} {{cc}} --chapter 5 --stage validate
    {{test_compiler}} {{cc}} --chapter 5 --bitwise


ch6: build
    #{{test_compiler}} {{cc}} --chapter 6 --stage lex
    #{{test_compiler}} {{cc}} --chapter 6 --stage parse
    #{{test_compiler}} {{cc}} --chapter 6 --stage validate
    #{{test_compiler}} {{cc}} --chapter 6 --bitwise
    {{test_compiler}} {{cc}} --chapter 6 --bitwise --goto


ch7: build
    #{{test_compiler}} {{cc}} --chapter 7 --stage parse
    #{{test_compiler}} {{cc}} --chapter 7 --stage validate
    {{test_compiler}} {{cc}} --chapter 7 --bitwise --goto


ch8: build
    #{{test_compiler}} {{cc}} --chapter 8 --stage lex
    #{{test_compiler}} {{cc}} --chapter 8 --stage parse
    #{{test_compiler}} {{cc}} --chapter 8 --stage validate
    #{{test_compiler}} {{cc}} --chapter 8
    {{test_compiler}} {{cc}} --chapter 8 --bitwise --goto


ch9: build
    #{{test_compiler}} {{cc}} --chapter 9 --stage lex
    #{{test_compiler}} {{cc}} --chapter 9 --stage parse
    #{{test_compiler}} {{cc}} --chapter 9 --stage validate
    #{{test_compiler}} {{cc}} --chapter 9 --stage tacky
    #{{test_compiler}} {{cc}} --chapter 9 --stage codegen
    #{{test_compiler}} {{cc}} --chapter 9
    {{test_compiler}} {{cc}} --chapter 9 --bitwise --goto


ch10: build
    #{{test_compiler}} {{cc}} --chapter 10 --stage lex
    #{{test_compiler}} {{cc}} --chapter 10 --stage parse
    #{{test_compiler}} {{cc}} --chapter 10 --stage validate
    #{{test_compiler}} {{cc}} --chapter 10 --stage tacky
    #{{test_compiler}} {{cc}} --chapter 10 --stage codegen
    #{{test_compiler}} {{cc}} --chapter 10
    {{test_compiler}} {{cc}} --chapter 10 --bitwise --goto


ch11: build
    {{test_compiler}} {{cc}} --chapter 11 --stage lex
