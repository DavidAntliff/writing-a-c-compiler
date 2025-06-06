int main(void) {
    int a = 2;

    goto label2;

    label0:
        a = a + 1;
    goto label_end;

    label1:
        a = a * 2;
    goto label0;

    label2:
        a = a * a;
    goto label1;

    label_end:
        return a;
}
