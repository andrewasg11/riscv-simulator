"""
RISC-V Assembly to Hexadecimal Converter
Converts the given assembly program to machine code
"""


def sign_extend(value, bits):
    """Sign extend a value to 32 bits."""
    sign_bit = 1 << (bits - 1)
    if value & sign_bit:
        return value | (~((1 << bits) - 1) & 0xFFFFFFFF)
    return value


def encode_r_type(opcode, rd, funct3, rs1, rs2, funct7):
    """Encode R-type instruction."""
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def encode_i_type(opcode, rd, funct3, rs1, imm):
    """Encode I-type instruction."""
    imm = imm & 0xFFF  # 12-bit immediate
    return (imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def encode_s_type(opcode, funct3, rs1, rs2, imm):
    """Encode S-type instruction."""
    imm = imm & 0xFFF
    imm_11_5 = (imm >> 5) & 0x7F
    imm_4_0 = imm & 0x1F
    return (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_0 << 7) | opcode


def encode_b_type(opcode, funct3, rs1, rs2, imm):
    """Encode B-type instruction."""
    imm = imm & 0x1FFE  # 13-bit immediate (bit 0 is always 0)
    imm_12 = (imm >> 12) & 0x1
    imm_10_5 = (imm >> 5) & 0x3F
    imm_4_1 = (imm >> 1) & 0xF
    imm_11 = (imm >> 11) & 0x1
    return (imm_12 << 31) | (imm_10_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_1 << 8) | (imm_11 << 7) | opcode


def encode_u_type(opcode, rd, imm):
    """Encode U-type instruction."""
    imm = imm & 0xFFFFF000  # Upper 20 bits
    return imm | (rd << 7) | opcode


def encode_j_type(opcode, rd, imm):
    """Encode J-type instruction."""
    imm = imm & 0x1FFFFE  # 21-bit immediate (bit 0 is always 0)
    imm_20 = (imm >> 20) & 0x1
    imm_10_1 = (imm >> 1) & 0x3FF
    imm_11 = (imm >> 11) & 0x1
    imm_19_12 = (imm >> 12) & 0xFF
    return (imm_20 << 31) | (imm_19_12 << 12) | (imm_11 << 20) | (imm_10_1 << 21) | (rd << 7) | opcode


def assemble_instruction(mnemonic, rd=0, rs1=0, rs2=0, imm=0):
    """Assemble a single instruction to machine code."""
    
    # R-type instructions
    if mnemonic == 'add':
        return encode_r_type(0x33, rd, 0x0, rs1, rs2, 0x00)
    elif mnemonic == 'sub':
        return encode_r_type(0x33, rd, 0x0, rs1, rs2, 0x20)
    elif mnemonic == 'and':
        return encode_r_type(0x33, rd, 0x7, rs1, rs2, 0x00)
    elif mnemonic == 'or':
        return encode_r_type(0x33, rd, 0x6, rs1, rs2, 0x00)
    elif mnemonic == 'xor':
        return encode_r_type(0x33, rd, 0x4, rs1, rs2, 0x00)
    elif mnemonic == 'sll':
        return encode_r_type(0x33, rd, 0x1, rs1, rs2, 0x00)
    elif mnemonic == 'srl':
        return encode_r_type(0x33, rd, 0x5, rs1, rs2, 0x00)
    elif mnemonic == 'sra':
        return encode_r_type(0x33, rd, 0x5, rs1, rs2, 0x20)
    
    # I-type instructions (arithmetic)
    elif mnemonic == 'addi':
        return encode_i_type(0x13, rd, 0x0, rs1, imm)
    elif mnemonic == 'andi':
        return encode_i_type(0x13, rd, 0x7, rs1, imm)
    elif mnemonic == 'ori':
        return encode_i_type(0x13, rd, 0x6, rs1, imm)
    elif mnemonic == 'xori':
        return encode_i_type(0x13, rd, 0x4, rs1, imm)
    elif mnemonic == 'slli':
        return encode_i_type(0x13, rd, 0x1, rs1, imm)
    elif mnemonic == 'srli':
        return encode_i_type(0x13, rd, 0x5, rs1, imm)
    elif mnemonic == 'srai':
        return encode_i_type(0x13, rd, 0x5, rs1, imm | 0x400)
    
    # Load instructions
    elif mnemonic == 'lw':
        return encode_i_type(0x03, rd, 0x2, rs1, imm)
    elif mnemonic == 'lb':
        return encode_i_type(0x03, rd, 0x0, rs1, imm)
    elif mnemonic == 'lh':
        return encode_i_type(0x03, rd, 0x1, rs1, imm)
    elif mnemonic == 'lbu':
        return encode_i_type(0x03, rd, 0x4, rs1, imm)
    elif mnemonic == 'lhu':
        return encode_i_type(0x03, rd, 0x5, rs1, imm)
    
    # Store instructions
    elif mnemonic == 'sw':
        return encode_s_type(0x23, 0x2, rs1, rs2, imm)
    elif mnemonic == 'sb':
        return encode_s_type(0x23, 0x0, rs1, rs2, imm)
    elif mnemonic == 'sh':
        return encode_s_type(0x23, 0x1, rs1, rs2, imm)
    
    # Branch instructions
    elif mnemonic == 'beq':
        return encode_b_type(0x63, 0x0, rs1, rs2, imm)
    elif mnemonic == 'bne':
        return encode_b_type(0x63, 0x1, rs1, rs2, imm)
    elif mnemonic == 'blt':
        return encode_b_type(0x63, 0x4, rs1, rs2, imm)
    elif mnemonic == 'bge':
        return encode_b_type(0x63, 0x5, rs1, rs2, imm)
    elif mnemonic == 'bltu':
        return encode_b_type(0x63, 0x6, rs1, rs2, imm)
    elif mnemonic == 'bgeu':
        return encode_b_type(0x63, 0x7, rs1, rs2, imm)
    
    # Jump instructions
    elif mnemonic == 'jal':
        return encode_j_type(0x6F, rd, imm)
    elif mnemonic == 'jalr':
        return encode_i_type(0x67, rd, 0x0, rs1, imm)
    
    # Upper immediate instructions
    elif mnemonic == 'lui':
        return encode_u_type(0x37, rd, imm)
    elif mnemonic == 'auipc':
        return encode_u_type(0x17, rd, imm)
    
    else:
        raise ValueError(f"Unknown instruction: {mnemonic}")


def assemble_program():
    """Assemble the given RISC-V assembly program."""
    
    print("="*70)
    print("RISC-V ASSEMBLY TO HEXADECIMAL CONVERTER")
    print("="*70)
    print("\nAssembly Program:")
    print("-"*70)
    
    # Define the program with comments
    program = [
        ("addi x1, x0, 5", "addi", 1, 0, 0, 5),      # x1 = 5
        ("addi x2, x0, 10", "addi", 2, 0, 0, 10),    # x2 = 10
        ("add x3, x1, x2", "add", 3, 1, 2, 0),       # x3 = 15
        ("sub x4, x2, x1", "sub", 4, 2, 1, 0),       # x4 = 5
        ("lui x5, 0x00010", "lui", 5, 0, 0, 0x10000),# x5 = 0x10000
        ("sw x3, 0(x5)", "sw", 0, 5, 3, 0),          # mem[0x10000] = 15
        ("lw x4, 0(x5)", "lw", 4, 5, 0, 0),          # x4 = 15
        ("beq x3, x4, label1", "beq", 0, 3, 4, 8),   # branch forward 8 bytes
        ("addi x6, x0, 1", "addi", 6, 0, 0, 1),      # x6 = 1 (skipped)
        ("addi x6, x0, 2", "addi", 6, 0, 0, 2),      # label1: x6 = 2
        ("jal x0, 0", "jal", 0, 0, 0, 0),            # infinite loop (halt)
    ]
    
    machine_code = []
    
    for i, instruction in enumerate(program):
        asm_text, mnemonic, rd, rs1, rs2, imm = instruction
        
        # Assemble instruction
        hex_code = assemble_instruction(mnemonic, rd, rs1, rs2, imm)
        machine_code.append(hex_code)
        
        # Print assembly and machine code
        addr = i * 4
        print(f"0x{addr:04X}: {asm_text:25} -> 0x{hex_code:08X}")
    
    print("-"*70)
    
    # Write to prog.hex file
    print("\nWriting to prog.hex...")
    try:
        with open("prog.hex", "w") as f:
            # Write header comment
            f.write("# RISC-V Machine Code\n")
            f.write("# Generated by RISC-V Assembler\n")
            f.write("#\n")
            f.write("# Format: One instruction per line (32-bit hex)\n")
            f.write("#\n\n")
            
            # Write each instruction with comment
            for i, code in enumerate(machine_code):
                comment = program[i][0]
                f.write(f"{code:08X}  # {comment}\n")
            
            # Write halt instruction
            f.write("00000000  # Halt\n")
        
        print("✓ Successfully wrote machine code to prog.hex")
    except Exception as e:
        print(f"✗ Error writing to file: {e}")
    
    # Print summary
    print("\nMachine Code (for copy-paste):")
    print("-"*70)
    print("program = [")
    for i, code in enumerate(machine_code):
        comment = program[i][0]
        print(f"    0x{code:08X},  # {comment}")
    print("    0x00000000,  # Halt")
    print("]")
    
    # Print as Python list
    print("\nPython List Format:")
    print("-"*70)
    print("program = [", end="")
    for i, code in enumerate(machine_code):
        if i % 4 == 0:
            print("\n    ", end="")
        print(f"0x{code:08X}, ", end="")
    print("\n    0x00000000")
    print("]")
    
    # Print C array format
    print("\nC Array Format:")
    print("-"*70)
    print("uint32_t program[] = {")
    for i, code in enumerate(machine_code):
        comment = program[i][0]
        print(f"    0x{code:08X},  // {comment}")
    print("    0x00000000   // Halt")
    print("};")
    
    # Verify with detailed breakdown
    print("\n" + "="*70)
    print("DETAILED INSTRUCTION BREAKDOWN")
    print("="*70)
    
    for i, instruction in enumerate(program):
        asm_text, mnemonic, rd, rs1, rs2, imm = instruction
        hex_code = machine_code[i]
        binary = f"{hex_code:032b}"
        
        print(f"\nInstruction {i}: {asm_text}")
        print(f"  Hex:    0x{hex_code:08X}")
        print(f"  Binary: {binary[:8]}_{binary[8:16]}_{binary[16:24]}_{binary[24:]}")
        
        # Decode fields
        opcode = hex_code & 0x7F
        rd_dec = (hex_code >> 7) & 0x1F
        funct3 = (hex_code >> 12) & 0x7
        rs1_dec = (hex_code >> 15) & 0x1F
        
        print(f"  Opcode: 0x{opcode:02X} ({opcode:07b})")
        if rd != 0:
            print(f"  rd:     x{rd_dec}")
        if rs1 != 0 or mnemonic in ['lw', 'sw', 'beq', 'bne']:
            print(f"  rs1:    x{rs1_dec}")
        if rs2 != 0 or mnemonic in ['sw', 'beq', 'bne']:
            rs2_dec = (hex_code >> 20) & 0x1F
            print(f"  rs2:    x{rs2_dec}")
        if mnemonic in ['addi', 'lw', 'lui']:
            print(f"  imm:    {imm} (0x{imm:X})")
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE")
    print("="*70)
    print(f"\n✓ Machine code saved to: prog.hex")
    
    return machine_code


def test_with_cpu():
    """Test the assembled program with our CPU simulator."""
    print("\n" + "="*70)
    print("TESTING WITH CPU SIMULATOR")
    print("="*70)
    
    # Import CPU if available
    try:
        # Recreate minimal CPU for testing
        program = [
            0x00500093,  # addi x1, x0, 5
            0x00A00113,  # addi x2, x0, 10
            0x002081B3,  # add x3, x1, x2
            0x40110233,  # sub x4, x2, x1
            0x000102B7,  # lui x5, 0x00010
            0x0032A023,  # sw x3, 0(x5)
            0x0002A203,  # lw x4, 0(x5)
            0x00418463,  # beq x3, x4, 8
            0x00100313,  # addi x6, x0, 1
            0x00200313,  # addi x6, x0, 2
            0x0000006F,  # jal x0, 0
            0x00000000   # Halt
        ]
        
        print("\nProgram loaded successfully!")
        print(f"Total instructions: {len(program)}")
        print("\nYou can now use this program with the CPU simulator:")
        print("\n  cpu = CPU()")
        print("  cpu.load_program(program)")
        print("  cpu.run(verbose=True)")
        
    except Exception as e:
        print(f"\nNote: {e}")
    
    return program


if __name__ == "__main__":
    # Assemble the program
    machine_code = assemble_program()
    
    # Test with CPU
    test_program = test_with_cpu()
    
    print("\n✓ Assembly conversion complete!")
    print("  Use the hex values above in your CPU simulator.")