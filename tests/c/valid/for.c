int main(void) {
    int s = 0;
    for (int x = 0; x < 100; x = x + 1) {
        if (x > 9)
            break;

        if (x % 2 == 0)
            continue;

        s = s + x;
    }

    int x = 0;
    for (x = 0; x < 5; x = x + 1) {
        s = s + x;
    }

    for ( ; x < 10; x = x + 1) {
        s = s + x;
    }

    for (int x = 0;    ; x = x + 1) {
        if (x > 5)
            break;
        s = s + x;
    }

    for (; ; x = x + 1) {
        if (x > 10)
            break;
        s = s + x;
    }

    for (;;) {
        x = x - 1;
        if (x == 0)
            break;
        s = s + x;
    }

    return s;
}