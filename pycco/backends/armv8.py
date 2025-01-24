from pycco.ast import *
from pycco.visitor import Visitor, CompilationContext, Compiler


armv8 = Visitor()


class ArmCompilationContext(CompilationContext):
    """
    Holds data needed during ARM code generation.
    E.g., label counters, symbol tables, CPU features, etc.
    """

    def __init__(self):
        self.labels_created = 0

    def new_label(self, prefix="L"):
        """
        Generate a fresh label name (e.g., L0, L1, etc.).
        """
        lbl = f"{prefix}{self.labels_created}"
        self.labels_created += 1
        return lbl


compiler: Compiler = lambda: (ArmCompilationContext(), armv8)


@armv8.register
def visit_code(ctx: CompilationContext, node: Code) -> str:
    """
    Generate top-level code section, plus an optional 'main' function
    if node.main is present.
    """
    lines = []
    lines.append(".section .text")
    lines.append(".global main")

    # Compile each statement
    for stmt in node.statements:
        lines.append(arm(ctx, stmt))

    if node.main:
        lines.append(arm(ctx, node.main))
    return "\n".join(lines)


@armv8.register
def visit_ident(ctx: CompilationContext, node: Ident) -> str:
    """
    Return minimal code that references the address or value of an identifier.
    """
    return f"    LDR R0, ={node.name}    @ Ident {node.name}"


@armv8.register
def visit_type(ctx: CompilationContext, node: Type) -> str:
    """
    Typically doesn't emit direct code in a real compiler,
    but we'll return a comment for demonstration.
    """
    pointer_str = "*" if node.pointer else ""
    return f"    @ type: {node.name}{pointer_str}"


@armv8.register
def visit_variabledecl(ctx: CompilationContext, node: VariableDecl) -> str:
    """
    Handles variable declarations. ARM assembly usually relies on stack management,
    so this generates comments for clarity.
    """
    return f"    @ var decl: {node.name.name}"


@armv8.register
def visit_assign(ctx: CompilationContext, node: Assign) -> str:
    """
    Evaluate RHS, then store into the variable's address.
    """
    lines = []
    lines.append("@ Assignment")
    rhs_code = arm(ctx, node.to)
    lines.append(rhs_code)

    if isinstance(node.var, Ident):
        lines.append(f"    STR R0, [R1]    @ store into {node.var.name}")
    elif isinstance(node.var, VariableDecl):
        lines.append(f"    STR R0, [R1]    @ store into {node.var.name.name}")
    return "\n".join(lines)


@armv8.register
def visit_arg(ctx: CompilationContext, node: Arg) -> str:
    """
    Function arguments are stored in registers R0-R3 or pushed to the stack.
    This generates comments to show the argument logic.
    """
    return f"    @ Arg: {node.name.name}"


@armv8.register
def visit_return(ctx: CompilationContext, node: Return) -> str:
    """
    Evaluate the return expression and use 'MOV PC, LR' to return from the function.
    """
    value_code = arm(ctx, node.value)
    lines = []
    lines.append(value_code)
    lines.append("    MOV PC, LR    @ Return from function")
    return "\n".join(lines)


@armv8.register
def visit_number(ctx: CompilationContext, node: Number) -> str:
    """
    Load an immediate value into R0.
    """
    return f"    MOV R0, #{node.value}    @ load immediate {node.value}"


@armv8.register
def visit_function(ctx: CompilationContext, node: Function) -> str:
    """
    Compile a function with prologue, body, and epilogue.
    """
    func_name = node.var.name.name
    lines = []
    lines.append(f"{func_name}:")
    lines.append("    PUSH {LR}    @ Save link register")

    for stmt in node.body:
        lines.append(arm(ctx, stmt))

    if node.ret:
        lines.append(arm(ctx, node.ret))
    else:
        lines.append("    POP {PC}    @ Return from function (no explicit return)")

    return "\n".join(lines)


@armv8.register
def visit_functioncall(ctx: CompilationContext, node: FunctionCall) -> str:
    """
    Call a function with arguments placed in registers or stack.
    """
    lines = []
    for i, arg in enumerate(node.args):
        lines.append(f"    @ Evaluate arg {i}")
        lines.append(arm(ctx, arg))
        lines.append(f"    MOV R{i}, R0    @ Move arg to R{i}")

    lines.append(f"    BL {node.name.name}    @ Call function {node.name.name}")
    return "\n".join(lines)


@armv8.register
def visit_ifstatement(ctx: CompilationContext, node: IfStatement) -> str:
    """
    Compile an if statement with optional else branch.
    """
    else_label = ctx.new_label("else")
    end_label = ctx.new_label("endif")
    lines = []

    cond_code = arm(ctx, node.condition)
    lines.append(cond_code)
    lines.append("    CMP R0, #0")
    lines.append(f"    BEQ {else_label}")

    # Then branch
    for stmt in node.then_branch:
        lines.append(arm(ctx, stmt))
    lines.append(f"    B {end_label}")

    # Else branch
    lines.append(f"{else_label}:")
    if node.else_branch:
        for stmt in node.else_branch:
            lines.append(arm(ctx, stmt))

    lines.append(f"{end_label}:")
    return "\n".join(lines)


@armv8.register
def visit_whileloop(ctx: CompilationContext, node: WhileLoop) -> str:
    """
    Compile a while loop with condition and body.
    """
    start_label = ctx.new_label("while")
    end_label = ctx.new_label("endwhile")
    lines = []

    lines.append(f"{start_label}:")
    cond_code = arm(ctx, node.condition)
    lines.append(cond_code)
    lines.append("    CMP R0, #0")
    lines.append(f"    BEQ {end_label}")

    for stmt in node.body:
        lines.append(arm(ctx, stmt))
    lines.append(f"    B {start_label}")
    lines.append(f"{end_label}:")
    return "\n".join(lines)


@armv8.register
def visit_binaryop(ctx: CompilationContext, node: BinaryOp) -> str:
    """
    Compile a binary operation (e.g., +, -, *, /).
    """
    lines = []
    left_code = arm(ctx, node.left)
    lines.append(left_code)
    lines.append("    PUSH {R0}")

    right_code = arm(ctx, node.right)
    lines.append(right_code)
    lines.append("    POP {R1}")

    if node.operator == "+":
        lines.append("    ADD R0, R1, R0    @ R0 = R1 + R0")
    elif node.operator == "-":
        lines.append("    SUB R0, R1, R0    @ R0 = R1 - R0")
    elif node.operator == "*":
        lines.append("    MUL R0, R1, R0    @ R0 = R1 * R0")
    elif node.operator == "/":
        lines.append("    SDIV R0, R1, R0   @ R0 = R1 / R0")
    else:
        lines.append(f"    @ Unhandled operator {node.operator}")

    return "\n".join(lines)


@armv8.register
def visit_unaryop(ctx: CompilationContext, node: UnaryOp) -> str:
    """
    Compile a unary operation (e.g., -x).
    """
    operand_code = arm(ctx, node.operand)
    lines = [operand_code]

    if node.operator == "-":
        lines.append("    RSBS R0, R0, #0    @ Negate R0")
    else:
        lines.append(f"    @ Unhandled unary operator {node.operator}")

    return "\n".join(lines)


@armv8.register
def visit_char(ctx: CompilationContext, node: Char) -> str:
    """
    Load a character literal.
    """
    char_val = ord(node.value)
    return f"    MOV R0, #{char_val}    @ char '{node.value}'"


@armv8.register
def visit_stringliteral(ctx: CompilationContext, node: StringLiteral) -> str:
    """
    Handle a string literal by loading its address (assuming .rodata section).
    """
    return f'    @ string "{node.value}" (would be in .rodata)'


@armv8.register
def visit_arrayindex(ctx: CompilationContext, node: ArrayIndex) -> str:
    """
    Compile an array access (array[index]).
    """
    lines = []
    lines.append(arm(ctx, node.array))
    lines.append("    PUSH {R0}    @ Push array base")

    index_code = arm(ctx, node.index)
    lines.append(index_code)
    lines.append("    POP {R1}    @ Pop array base into R1")
    lines.append("    LDR R0, [R1, R0, LSL #2]    @ Load array element")

    return "\n".join(lines)


@armv8.register
def visit_structaccess(ctx: CompilationContext, node: StructAccess) -> str:
    """
    Compile struct field access.
    """
    obj_code = arm(ctx, node.obj)
    lines = [obj_code]
    lines.append(f"    @ Access field {node.field.name} with operator {node.operator}")

    return "\n".join(lines)


@armv8.register
def visit_parens(ctx: CompilationContext, node: Parens) -> str:
    """
    Compile parenthesized expressions.
    """
    return arm(ctx, node.inner)
