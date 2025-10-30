"""
R-type instructions (for register)
opcode(basic operation of instruction eg. add,sub,ld,st): = [0:6]
rd(register destination riscv contains 32 registers):[7:11]
funct3(additional opcode field) = [12:14]
rs1(1st register source operand): [15:19]
rs2(2nd register source operand): [20:24]
funct7(additional opcode field): [25:31]
"""

"""
I-type (arithmetic operands with one constant)
opcode: [0:6]
rd: [7:11]
funct3: [12:14]
rs1: [15:19]
immediate: [20:31]
"""

"""
S-type 
opcode: [0:6]
immediate ([0:4]): [7:11]
funct3: [12:14]
rs1: [15:19]
rs2: [20:24]
immediate([5:11]): [25:31]
"""

"""
B-type
opcode: [0:6]
immediate: [7:11]
funct3: [12:14]
rs1: [15:19]
rs2: [20:24]
immediate: [25:31]
"""