int simple(int a) {
    return a + 1;
}

int eight(int a, int b, int c, int d, int e, int f, int g, int h) {
    return a + simple(b) + c + d + simple(e) + f + g + h;
}

int main(void) {
    return eight(1, 2, 3, 4, 5, 6, 7, 8);
}
