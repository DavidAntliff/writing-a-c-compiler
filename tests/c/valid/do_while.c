int main(void) {
    int x = 0;
    int s = 0;
    do {
        if (x > 9)
            break;

        x = x + 1;

        if (x % 2 == 0)
            continue;

        s = s + x;

    } while (x < 100);

    return s;
}