default: test ch1 ch2 ch3 ch4


driver-test:
    rm -f driver driver.i driver.s
    python3 pcc.py driver/driver.c
    ./driver ; echo $?


# test_compiler also tests previous chapters

test:
    cargo test


ch1:
    cargo build
    book-tests/test_compiler ./pcc.py --chapter 1 --stage lex
    book-tests/test_compiler ./pcc.py --chapter 1 --stage parse
    book-tests/test_compiler ./pcc.py --chapter 1 --stage codegen
    book-tests/test_compiler ./pcc.py --chapter 1


listing_1-1:
    cargo build && ./pcc.py ch/1/listing_1.1.c && ch/1/listing_1.1 ; echo $?


ch2:
    cargo build
    book-tests/test_compiler ./pcc.py --chapter 2 --stage lex
    book-tests/test_compiler ./pcc.py --chapter 2 --stage parse
    book-tests/test_compiler ./pcc.py --chapter 2 --stage tacky
    book-tests/test_compiler ./pcc.py --chapter 2 --stage codegen
    book-tests/test_compiler ./pcc.py --chapter 2


listing_2-1:
    cargo build && ./pcc.py ch/2/listing_2.1.c && ch/2/listing_2.1 ; echo $?


ch3:
    cargo build
    book-tests/test_compiler ./pcc.py --chapter 3 --stage lex
    book-tests/test_compiler ./pcc.py --chapter 3 --stage parse
    book-tests/test_compiler ./pcc.py --chapter 3 --stage tacky
    book-tests/test_compiler ./pcc.py --chapter 3 --stage codegen
    book-tests/test_compiler ./pcc.py --chapter 3 --bitwise


ch4:
    cargo build
    book-tests/test_compiler ./pcc.py --chapter 4 --stage lex
