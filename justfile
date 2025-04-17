# test_compiler also tests previous chapters
default: test ch5 pytest


driver-test:
    rm -f driver driver.i driver.s
    python3 pcc.py driver/driver.c
    ./driver ; echo $?


test:
    cargo test


pytest:
    #!/usr/bin/env bash
    if [[ $OSTYPE == 'darwin'* && $(arch) != "i386" ]]
    then
        echo "Skipping pytest on non-i386 architecture"
        exit 0
    fi
    pytest -sv -n 8 tests


ch1:
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 1 --stage lex
    #book-tests/test_compiler ./pcc.py --chapter 1 --stage parse
    #book-tests/test_compiler ./pcc.py --chapter 1 --stage codegen
    book-tests/test_compiler ./pcc.py --chapter 1


listing_1-1:
    cargo build && ./pcc.py ch/1/listing_1.1.c && ch/1/listing_1.1 ; echo $?


ch2:
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 2 --stage lex
    #book-tests/test_compiler ./pcc.py --chapter 2 --stage parse
    #book-tests/test_compiler ./pcc.py --chapter 2 --stage tacky
    #book-tests/test_compiler ./pcc.py --chapter 2 --stage codegen
    book-tests/test_compiler ./pcc.py --chapter 2


listing_2-1:
    cargo build && ./pcc.py ch/2/listing_2.1.c && ch/2/listing_2.1 ; echo $?


ch3:
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 3 --stage lex
    #book-tests/test_compiler ./pcc.py --chapter 3 --stage parse
    #book-tests/test_compiler ./pcc.py --chapter 3 --stage tacky
    #book-tests/test_compiler ./pcc.py --chapter 3 --stage codegen
    book-tests/test_compiler ./pcc.py --chapter 3 --bitwise


ch4:
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 4 --stage lex
    #book-tests/test_compiler ./pcc.py --chapter 4 --stage parse
    #book-tests/test_compiler ./pcc.py --chapter 4 --stage tacky
    #book-tests/test_compiler ./pcc.py --chapter 4 --stage codegen
    book-tests/test_compiler ./pcc.py --chapter 4 --bitwise


ch5:
    cargo build
    #book-tests/test_compiler ./pcc.py --chapter 5 --stage lex
    #book-tests/test_compiler ./pcc.py --chapter 5 --stage parse
    book-tests/test_compiler ./pcc.py --chapter 5 --stage validate
