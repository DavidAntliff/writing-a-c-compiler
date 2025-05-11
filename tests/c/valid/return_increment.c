int main(void) {
    int a = 1;
    int b = 2;
    int c = a++ + ++b;
    c++;
    return ++a + ++b + ++c;
}
