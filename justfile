# test_compiler also tests previous chapters
default: test ch9 pytest


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


test:
    cargo test


pytest: check-i386
    pytest -sv -n 8 tests


ch1: check-i386
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 1 --stage lex
    #book-tests/test_compiler ./pcc.py --chapter 1 --stage parse
    #book-tests/test_compiler ./pcc.py --chapter 1 --stage codegen
    book-tests/test_compiler ./pcc.py --chapter 1


listing_1-1: check-i386
    cargo build && ./pcc.py ch/1/listing_1.1.c && ch/1/listing_1.1 ; echo $?


ch2: check-i386
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 2 --stage lex
    #book-tests/test_compiler ./pcc.py --chapter 2 --stage parse
    #book-tests/test_compiler ./pcc.py --chapter 2 --stage tacky
    #book-tests/test_compiler ./pcc.py --chapter 2 --stage codegen
    book-tests/test_compiler ./pcc.py --chapter 2


listing_2-1: check-i386
    cargo build && ./pcc.py ch/2/listing_2.1.c && ch/2/listing_2.1 ; echo $?


ch3: check-i386
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 3 --stage lex
    #book-tests/test_compiler ./pcc.py --chapter 3 --stage parse
    #book-tests/test_compiler ./pcc.py --chapter 3 --stage tacky
    #book-tests/test_compiler ./pcc.py --chapter 3 --stage codegen
    book-tests/test_compiler ./pcc.py --chapter 3 --bitwise


ch4: check-i386
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 4 --stage lex
    #book-tests/test_compiler ./pcc.py --chapter 4 --stage parse
    #book-tests/test_compiler ./pcc.py --chapter 4 --stage tacky
    #book-tests/test_compiler ./pcc.py --chapter 4 --stage codegen
    book-tests/test_compiler ./pcc.py --chapter 4 --bitwise


ch5: check-i386
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 5 --stage lex
    #book-tests/test_compiler ./pcc.py --chapter 5 --stage parse
    #book-tests/test_compiler ./pcc.py --chapter 5 --stage validate
    book-tests/test_compiler ./pcc.py --chapter 5 --bitwise


ch6: check-i386
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 6 --stage lex
    #book-tests/test_compiler ./pcc.py --chapter 6 --stage parse
    #book-tests/test_compiler ./pcc.py --chapter 6 --stage validate
    #book-tests/test_compiler ./pcc.py --chapter 6 --bitwise
    book-tests/test_compiler ./pcc.py --chapter 6 --bitwise --goto


ch7: check-i386
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 7 --stage parse
    #book-tests/test_compiler ./pcc.py --chapter 7 --stage validate
    book-tests/test_compiler ./pcc.py --chapter 7 --bitwise --goto


ch8: check-i386
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 8 --stage lex
    #book-tests/test_compiler ./pcc.py --chapter 8 --stage parse
    #book-tests/test_compiler ./pcc.py --chapter 8 --stage validate
    #book-tests/test_compiler ./pcc.py --chapter 8
    book-tests/test_compiler ./pcc.py --chapter 8 --bitwise --goto


ch9: check-i386
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 9 --stage lex
    #book-tests/test_compiler ./pcc.py --chapter 9 --stage parse
    book-tests/test_compiler ./pcc.py --chapter 9 --stage validate
