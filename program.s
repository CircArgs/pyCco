.global _start

_start:
    mov x0, #5            // Load 5 into x0
    mov x1, #7            // Load 7 into x1
    mul x0, x0, x1        // Add x0 and x1, result stored in x0

    mov x8, #93           // Syscall number for 'exit' on ARM64 Linux
    svc 0                 // Trigger the syscall


