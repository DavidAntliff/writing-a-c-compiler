int main(void) {
    int s = 0;
    for (int x = 0; x < 100; x = x + 1) {
        if (x > 9)
            break;

        if (x % 2 == 0)
            continue;

        int y = 0;
        do {
            s = s + x + y;
            y = y + 1;

            while (s > 100) {
                if (s < 200)
                    break;

                s = s / 2;
            }
        } while (y < 10);
    }

    return s;
}
