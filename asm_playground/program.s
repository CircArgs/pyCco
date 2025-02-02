.data
string: .asciz "579\n"
filename: .asciz "example.txt"

.text
.global _start

_start:
    bl open @ open the file
    mov r8, r0 @ store file descriptor in r8 for later use and free r0


    @ calculate the length of the string
    mov r0, #0
    ldr r1, =string
    bl length
    @ end calculate length of string

    mov r2, r0
    mov r0, r8
    ldr r1, =string
    bl write

    b exit_success
    
open:
    @ expects filename to exist in .data
    ldr r0, =filename @ file path (address in memory)
    ldr r1, =0x242 @ How to open the file (note: using ldr because mov only applies to 8 bit)
    ldr r2, =0x284 @ permissions: octal 644
    ldr r7, =0x5 @ system call number - 5 means open 0
    swi 0 @ SoftWare Interrupt - swi - return control to the operating system to perform the action prescribed by the values in the set registers
    cmp r0, #0
    blt exit_fail
    bx lr @ return control to caller (_start)

write:
    @ expects string address to already be in r1
    @ expects its length to be in r2
    @ expects the file descriptor to be in r0
    mov r7, #4 @ make syscall
    swi 0
    cmp r0, #0
    blt exit_fail
    bx lr


length:
    ldrb r2, [r1], #1
    cmp r2, #0
    beq _bxlr
    add r0, #1
    b length

_bxlr:
    bx lr

exit_fail:
    mov r0, #1
    mov r7, #1
    swi 0

exit_success:
    mov r0, #0
    mov r7, #1
    swi 0
    