/*
LampVM -- Example: Print Function

This program shows how to initialize the serial port and output a single char to serial every call.

This program may changes frequently, as this project is still in early access stage.
*/

int serial_init() {
    //Send SERIAL_CTRL_RX_INT_ENABLE(0x01) to SCREEN_ATTRIBUTE(0x02) to enable Serial feature.
    asm("movi r2, 0x02 \n movi r5, 0x01 \n out  [r2], r5\n");
    return 0;
}

int print(int ch) {
    //Send 1 char every time to serial
    asm("movi r1, 0x01\nout  [r1], r0\n");
    return 0;
}

int main() {
    serial_init();
    print(79);
    print(75);
    print(10);
    return 0;
}
