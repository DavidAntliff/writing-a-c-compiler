driver-test:
    rm -f driver driver.i driver.s
    python3 pcc.py driver.c
    ./driver ; echo $?
