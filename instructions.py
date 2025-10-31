#Instruction types
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


#-------------------------------------------------------------------------------
#MULTIPLEXORS
"""
opcode_multiplexor: analyzed first 7 bytes of code. determines operation(eg. add, subtract,lw, sw)
case: 001, 010, 011...  called every time

rd_multiplexor: determines register destination. bits [7:11]. called on R-type, I-type

immediate1_multiplexor: first immediate, bits [7:11]. called on S-type, and B-type opcode instructions

funct3_ multiplexor: additional opcode [12:14] bits. called on every type

rs1_multiplexor: register 1 [15:19] called on every type

rs2_multiplexor: register 2 [20:24] called on R-type, S-type, B-type

immediate_imultiplexor: registor [20:31] called on I-type

immediate2_multiplexor: [25:31] called on S-type, B-type

funct7_multiplexor: additional opcode [25:31] called on R-type

"""

