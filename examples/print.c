/*
LampVM -- Example: Print Function

This program shows how to initialize the serial port and output a single char to serial every call.

This program may changes frequently, as this project is still in early access stage.
*/

int SERIAL_DATA = 0x01;
int SERIAL_CTRL = 0x02;
int SERIAL_CTRL_RX_INT_ENABLE = 0x01;

int serial_init() {
    // Enable serial feature.
    asm("movi r1, SERIAL_CTRL \nload32 r2, [r1] \nmovi r1, SERIAL_CTRL_RX_INT_ENABLE \nload32 r5, [r1] \nout  [r2], r5\n");
    return 0;
}

int print_char(int ch) {
    // Send 1 char every call.
    asm("movi r1, SERIAL_DATA\nload32 r1, [r1]\nout  [r1], r0\n");
    return 0;
}

int print_str(char *s) {
    int i = 0;
    while (s[i] != 0) {
        print_char(s[i]);
        i = i + 1;
    }
    return 0;
}

int main() {
    serial_init();
    print_char(79);
    print_char(75);
    print_char(10);

    char msg[3];
    msg[0] = 79;
    msg[1] = 75;
    msg[2] = 0;
    print_str(msg);
    print_char(10);
    return 0;
}
