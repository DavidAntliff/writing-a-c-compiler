default: ch1-test


driver-test:
    rm -f driver driver.i driver.s
    python3 pcc.py driver.c
    ./driver ; echo $?


ch1-test:
    cargo build
    cargo test
    book-tests/test_compiler ./pcc.py --chapter 1 --stage lex
    book-tests/test_compiler ./pcc.py --chapter 1 --stage parse
    book-tests/test_compiler ./pcc.py --chapter 1 --stage codegen
    book-tests/test_compiler ./pcc.py --chapter 1


ch1-listing:
    cargo build && ./pcc.py listing_1.1.c && ./listing_1.1 ; echo $?
