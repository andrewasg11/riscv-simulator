"""
Single-Cycle RISC-V CPU Implementation
Implements RV32I subset with full instruction set
"""


class ALU:
    """Arithmetic Logic Unit - performs all computational operations."""
    
    # ALU operation codes
    OP_ADD = 0
    OP_SUB = 1
    OP_AND = 2
    OP_OR = 3
    OP_XOR = 4
    OP_SLL = 5  # Shift left logical
    OP_SRL = 6  # Shift right logical
    OP_SRA = 7  # Shift right arithmetic
    OP_SLT = 8  # Set less than
    OP_SLTU = 9 # Set less than unsigned
    
    def __init__(self):
        self.zero_flag = False
        self.N = False
        self.C = False
        self.V = False
    
    def execute(self, op, a, b):
        """
        Execute ALU operation.
        
        Args:
            op: Operation code
            a, b: 32-bit operands (as integers)
        
        Returns:
            32-bit result
        """
        # Mask to 32 bits
        a = a & 0xFFFFFFFF
        b = b & 0xFFFFFFFF
        
        if op == self.OP_ADD:
            result = self.add(a,b)
        
        elif op == self.OP_SUB:
            result = self.sub(a,b)
        
        elif op == self.OP_AND:
            result = a & b
        
        elif op == self.OP_OR:
            result = a | b
        
        elif op == self.OP_XOR:
            result = a ^ b
        
        elif op == self.OP_SLL:
            # Shift left logical (shift amount is lower 5 bits of b)
            shift_amt = b & 0x1F
            result = (a << shift_amt) & 0xFFFFFFFF
        
        elif op == self.OP_SRL:
            # Shift right logical
            shift_amt = b & 0x1F
            result = a >> shift_amt
        
        elif op == self.OP_SRA:
            # Shift right arithmetic (sign extend)
            shift_amt = b & 0x1F
            if a & 0x80000000:  # Negative
                result = (a >> shift_amt) | (0xFFFFFFFF << (32 - shift_amt))
            else:
                result = a >> shift_amt
            result &= 0xFFFFFFFF
        
        elif op == self.OP_SLT:
            # Set less than (signed)
            a_signed = self._to_signed(a)
            b_signed = self._to_signed(b)
            result = 1 if a_signed < b_signed else 0
        
        elif op == self.OP_SLTU:
            # Set less than unsigned
            result = 1 if a < b else 0
        
        else:
            result = 0
        
        # Set zero flag
        self.zero_flag = (result == 0)
        
        return result & 0xFFFFFFFF
    
    def _to_signed(self, value):
        """Convert 32-bit unsigned to signed."""
        if value & 0x80000000:
            return value - 0x100000000
        return value
    def _to_bitvec(self, value):
        """Convert integer to 32-bit vector (list of 0/1)."""
        if isinstance(value,list):
            return value
        elif isinstance(value, str):
            # Binary string input
            value = value.replace('0b', '')
            if len(value) > 32:
                raise ValueError("Binary string exceeds 32 bits")
            value = value.zfill(32)
            return [int(b) for b in value]
        else:
            # Integer input - handle two's complement
            value = value & 0xFFFFFFFF  # Mask to 32 bits
            return [(value >> i) & 1 for i in range(32)]
    
    def _to_64_bitvec(self,value):
        if isinstance(value,list):
            pass
        elif isinstance(value,str):
            value = value.replace('0b','')
            if len(value) > 64 & len(value) < 33:
                raise ValueError("Binary string exceeds 64 bits")
            value = value.zfill(64)
            return [int(b) for b in value]
        else:
            value = value & 0xFFFFFFFFFFFFFFFF
            return[(value >> i) & 1 for i in range(64)]
    def __from_64bitvec(self,bitvec):
        result = 0
        for i in range(64):
            if bitvec[i]:
                result |= (1<<i)
        return result

    def _from_bitvec(self, bitvec):
        """Convert 32-bit vector to unsigned integer."""
        result = 0
        for i in range(32):
            if bitvec[i]:
                result |= (1 << i)
        return result
    
    def _bitvec_to_signed(self, bitvec):
        """Convert 32-bit vector to signed integer (two's complement)."""
        if len(bitvec) != 32:
            unsigned = self.__from_64bitvec(bitvec)
            if bitvec[-1]:
                return unsigned - (1 << 64)
        unsigned = self._from_bitvec(bitvec)
        if bitvec[31]:  # MSB set = negative
            return unsigned - (1 << 32)
        return unsigned
    
    def _full_adder(self, a, b, carry_in):
        """
        Single-bit full adder.
        Returns: (sum, carry_out)
        """
        # XOR for sum without carry
        sum_without_carry = a ^ b
        sum_bit = sum_without_carry ^ carry_in
        
        # Carry out occurs if at least 2 inputs are 1
        carry_out = (a & b) | (carry_in & sum_without_carry)
        
        return sum_bit, carry_out
    
    def _add_bitvec(self, a, b, carry_in=0):
        """
        Add two 32-bit vectors bit by bit.
        Returns: (result_bitvec, final_carry)
        """
        if len(a) > 32 or len(b) > 32:
            result = [0] * 64
        else:
            result = [0] * 32
        carry = carry_in
        
        for i in range(len(result)):
            result[i], carry = self._full_adder(a[i], b[i], carry)
        
        return result, carry
    
    def _twos_complement(self, bitvec):
        """Compute two's complement: invert all bits and add 1."""
        # Invert all bits
        inverted = [1 - bit for bit in bitvec]
        # Add 1
        if len(bitvec) == 64:
            one = [1] + [0] *63
        else:
            one = [1] + [0] * 31
        result, _ = self._add_bitvec(inverted, one)
        return result
    
    def _update_flags(self, result, carry_out, a_msb, b_msb):
        """Update ALU flags based on operation result."""
        # N: Negative (MSB is 1)
        self.N = bool(result[31])
        
        # Z: Zero (all bits are 0)
        self.Z = all(bit == 0 for bit in result)
        
        # C: Carry out of MSB
        self.C = carry_out
        
        # V: Signed overflow
        # Overflow occurs when:
        # - Adding two positive numbers yields negative result, OR
        # - Adding two negative numbers yields positive result
        result_msb = result[31]
        self.V = (a_msb == b_msb) and (a_msb != result_msb)
    def sll(self,bin):
        """
        shift left logical
        shifts binary array towards most significant bit
        """
        x = len(bin)-1
    
        while(x):
            bin[x] = bin[x-1]
            x = x - 1
        bin[0] = 0
        return bin
    def srl(self,bin):
        """
        shift right logical
        shifts binary array towards least significant bit
        """
        x = 0
        while(x<len(bin)-1):
            bin[x] = bin[x+1]
            x = x+1
        bin[-1] = 0
        return bin
    def sra(self,bin):
        """
        shift right arithmetic
        shifts binary array towards least significant bit
        keeps signed bit the same
        """
        x = 0
        while(x<len(bin)-1):
            bin[x] = bin[x+1]
            x = x+1
        return bin
    def add(self, a, b):
        """
        ADD operation: a + b
        Returns: 32-bit result as unsigned integer
        """
        a_vec = self._to_bitvec(a)
        b_vec = self._to_bitvec(b)
        
        result, carry_out = self._add_bitvec(a_vec, b_vec, carry_in=0)
        self._update_flags(result, carry_out, a_vec[31], b_vec[31])
        
        if isinstance(a,list) or isinstance(b,list):
            return result

        return self._from_bitvec(result)
    
    def sub(self, a, b):
        """
        SUB operation: a - b
        Implemented as a + (-b) using two's complement
        Returns: 32-bit result as unsigned integer
        """
        a_vec = self._to_bitvec(a)
        b_vec = self._to_bitvec(b)
        
        # Compute -b using two's complement
        neg_b = self._twos_complement(b_vec)
        
        # Add a + (-b)
        result, carry_out = self._add_bitvec(a_vec, neg_b, carry_in=0)
        
        # For subtraction, flags are based on a + (-b)
        self._update_flags(result, carry_out, a_vec[31], neg_b[31])

        if isinstance(a,list) or isinstance(b,list):
            return result
        if self.N:
            return self._bitvec_to_signed(result)
        if len(result) > 32:
            return self.__from_64bitvec(result)
        else:
            return self._from_bitvec(result)
    
    def addi(self, a, immediate):
        """
        ADDI operation: a + immediate (convenience for CLI)
        Returns: 32-bit result as unsigned integer
        """
        return self.add(a, immediate)


class RegisterFile:
    """32 general-purpose registers (x0-x31)."""
    
    def __init__(self):
        self.registers = [0] * 32
        self.registers[0] = 0  # x0 is hardwired to 0
    
    def read(self, reg_num):
        """Read from register."""
        if 0 <= reg_num < 32:
            return self.registers[reg_num] & 0xFFFFFFFF
        return 0
    
    def write(self, reg_num, value):
        """Write to register (x0 is always 0)."""
        if 0 < reg_num < 32:  # x0 cannot be written
            self.registers[reg_num] = value & 0xFFFFFFFF
    
    def dump(self):
        """Return register contents."""
        return {f"x{i}": self.registers[i] for i in range(32)}
    
    def print_registers(self, show_zero=False):
        """Print non-zero registers."""
        print("\n" + "="*60)
        print("REGISTER FILE")
        print("="*60)
        for i in range(32):
            if show_zero or self.registers[i] != 0:
                name = f"x{i}"
                if i == 2:
                    name += " (sp)"
                elif i == 1:
                    name += " (ra)"
                print(f"  {name:8} = 0x{self.registers[i]:08X}  ({self.registers[i]:10})")
        print("="*60)


class Memory:
    """Data and instruction memory."""
    
    def __init__(self, size=4096):
        """Initialize memory with given size in bytes."""
        self.size = size
        self.data = bytearray(size)
    
    def read_word(self, address):
        """Read 32-bit word (little endian)."""
        address = address & 0xFFFFFFFF
        if address + 3 < self.size:
            return (self.data[address] |
                    (self.data[address + 1] << 8) |
                    (self.data[address + 2] << 16) |
                    (self.data[address + 3] << 24))
        return 0
    
    def write_word(self, address, value):
        """Write 32-bit word (little endian)."""
        address = address & 0xFFFFFFFF
        value = value & 0xFFFFFFFF
        if address + 3 < self.size:
            self.data[address] = value & 0xFF
            self.data[address + 1] = (value >> 8) & 0xFF
            self.data[address + 2] = (value >> 16) & 0xFF
            self.data[address + 3] = (value >> 24) & 0xFF
    
    def load_program(self, instructions, start_addr=0):
        """Load instructions into memory."""
        for i, instr in enumerate(instructions):
            self.write_word(start_addr + (i * 4), instr)
    
    def dump_range(self, start, end):
        """Dump memory range."""
        print(f"\nMemory [{start:04X} - {end:04X}]:")
        for addr in range(start, min(end + 1, self.size), 4):
            value = self.read_word(addr)
            if value != 0:
                print(f"  0x{addr:04X}: 0x{value:08X}")


class ControlUnit:
    """Combinational control unit - generates control signals."""
    
    def __init__(self):
        pass
    
    def decode(self, instruction):
        """
        Decode instruction and generate control signals.
        
        Returns dict with control signals
        """
        opcode = instruction & 0x7F
        funct3 = (instruction >> 12) & 0x7
        funct7 = (instruction >> 25) & 0x7F
        
        rd = (instruction >> 7) & 0x1F
        rs1 = (instruction >> 15) & 0x1F
        rs2 = (instruction >> 20) & 0x1F
        
        signals = {
            'opcode': opcode,
            'funct3': funct3,
            'funct7': funct7,
            'rd': rd,
            'rs1': rs1,
            'rs2': rs2,
            'reg_write': False,
            'mem_read': False,
            'mem_write': False,
            'alu_src': 0,
            'alu_op': ALU.OP_ADD,
            'branch': False,
            'jump': False,
            'mem_to_reg': False,
            'pc_src': 0,
            'imm': 0
        }
        
        # R-type (0x33)
        if opcode == 0x33:
            signals['reg_write'] = True
            signals['alu_src'] = 0
            
            if funct3 == 0x0:
                signals['alu_op'] = ALU.OP_SUB if funct7 == 0x20 else ALU.OP_ADD
            elif funct3 == 0x7:
                signals['alu_op'] = ALU.OP_AND
            elif funct3 == 0x6:
                signals['alu_op'] = ALU.OP_OR
            elif funct3 == 0x4:
                signals['alu_op'] = ALU.OP_XOR
            elif funct3 == 0x1:
                signals['alu_op'] = ALU.OP_SLL
            elif funct3 == 0x5:
                signals['alu_op'] = ALU.OP_SRA if funct7 == 0x20 else ALU.OP_SRL
        
        # I-type arithmetic (0x13)
        elif opcode == 0x13:
            signals['reg_write'] = True
            signals['alu_src'] = 1
            
            imm = (instruction >> 20) & 0xFFF
            if imm & 0x800:
                imm |= 0xFFFFF000
            signals['imm'] = imm
            
            if funct3 == 0x0:
                signals['alu_op'] = ALU.OP_ADD
            elif funct3 == 0x7:
                signals['alu_op'] = ALU.OP_AND
            elif funct3 == 0x6:
                signals['alu_op'] = ALU.OP_OR
            elif funct3 == 0x4:
                signals['alu_op'] = ALU.OP_XOR
            elif funct3 == 0x1:
                signals['alu_op'] = ALU.OP_SLL
            elif funct3 == 0x5:
                signals['alu_op'] = ALU.OP_SRA if funct7 == 0x20 else ALU.OP_SRL
        
        # Load (0x03)
        elif opcode == 0x03:
            signals['reg_write'] = True
            signals['mem_read'] = True
            signals['mem_to_reg'] = True
            signals['alu_src'] = 1
            signals['alu_op'] = ALU.OP_ADD
            
            imm = (instruction >> 20) & 0xFFF
            if imm & 0x800:
                imm |= 0xFFFFF000
            signals['imm'] = imm
        
        # Store (0x23)
        elif opcode == 0x23:
            signals['mem_write'] = True
            signals['alu_src'] = 1
            signals['alu_op'] = ALU.OP_ADD
            
            imm = ((instruction >> 7) & 0x1F) | (((instruction >> 25) & 0x7F) << 5)
            if imm & 0x800:
                imm |= 0xFFFFF000
            signals['imm'] = imm
        
        # Branch (0x63)
        elif opcode == 0x63:
            signals['branch'] = True
            signals['alu_op'] = ALU.OP_SUB
            
            imm = (((instruction >> 8) & 0xF) << 1) | \
                  (((instruction >> 25) & 0x3F) << 5) | \
                  (((instruction >> 7) & 0x1) << 11) | \
                  (((instruction >> 31) & 0x1) << 12)
            if imm & 0x1000:
                imm |= 0xFFFFE000
            signals['imm'] = imm
        
        # JAL (0x6F)
        elif opcode == 0x6F:
            signals['jump'] = True
            signals['reg_write'] = True
            signals['pc_src'] = 2
            
            imm = (((instruction >> 21) & 0x3FF) << 1) | \
                  (((instruction >> 20) & 0x1) << 11) | \
                  (((instruction >> 12) & 0xFF) << 12) | \
                  (((instruction >> 31) & 0x1) << 20)
            if imm & 0x100000:
                imm |= 0xFFE00000
            signals['imm'] = imm
        
        # JALR (0x67)
        elif opcode == 0x67:
            signals['jump'] = True
            signals['reg_write'] = True
            signals['pc_src'] = 2
            signals['alu_src'] = 1
            
            imm = (instruction >> 20) & 0xFFF
            if imm & 0x800:
                imm |= 0xFFFFF000
            signals['imm'] = imm
        
        # LUI (0x37)
        elif opcode == 0x37:
            signals['reg_write'] = True
            imm = instruction & 0xFFFFF000
            signals['imm'] = imm
        
        # AUIPC (0x17)
        elif opcode == 0x17:
            signals['reg_write'] = True
            imm = instruction & 0xFFFFF000
            signals['imm'] = imm
        
        return signals


class CPU:
    """Single-cycle RISC-V CPU."""
    
    def __init__(self, mem_size=4096):
        """Initialize CPU components."""
        self.alu = ALU()
        self.registers = RegisterFile()
        self.memory = Memory(mem_size)
        self.control = ControlUnit()
        
        self.pc = 0
        self.cycle = 0
        self.halted = False
        self.trace = []
    
    def load_program(self, instructions):
        """Load program into instruction memory."""
        self.memory.load_program(instructions)
    
    def step(self, verbose=False):
        """Execute one clock cycle."""
        if self.halted:
            return False
        
        # 1. INSTRUCTION FETCH
        instruction = self.memory.read_word(self.pc)
        
        if instruction == 0:
            self.halted = True
            return False
        
        # 2. INSTRUCTION DECODE
        signals = self.control.decode(instruction)
        
        # 3. REGISTER READ
        rs1_data = self.registers.read(signals['rs1'])
        rs2_data = self.registers.read(signals['rs2'])
        
        # 4. EXECUTE
        alu_in_a = rs1_data
        alu_in_b = signals['imm'] if signals['alu_src'] == 1 else rs2_data
        
        if signals['opcode'] == 0x37:  # LUI
            alu_result = signals['imm']
        elif signals['opcode'] == 0x17:  # AUIPC
            alu_result = self.pc + signals['imm']
        else:
            alu_result = self.alu.execute(signals['alu_op'], alu_in_a, alu_in_b)
        
        # 5. MEMORY ACCESS
        mem_data = 0
        if signals['mem_read']:
            mem_data = self.memory.read_word(alu_result)
        
        if signals['mem_write']:
            self.memory.write_word(alu_result, rs2_data)
        
        # 6. WRITE BACK
        if signals['reg_write']:
            if signals['jump']:
                self.registers.write(signals['rd'], self.pc + 4)
            elif signals['mem_to_reg']:
                self.registers.write(signals['rd'], mem_data)
            else:
                self.registers.write(signals['rd'], alu_result)
        
        # 7. PC UPDATE
        next_pc = self.pc + 4
        
        if signals['branch']:
            take_branch = False
            if signals['funct3'] == 0x0:  # BEQ
                take_branch = self.alu.zero_flag
            elif signals['funct3'] == 0x1:  # BNE
                take_branch = not self.alu.zero_flag
            
            if take_branch:
                next_pc = self.pc + signals['imm']
        
        elif signals['jump']:
            if signals['opcode'] == 0x6F:  # JAL
                next_pc = self.pc + signals['imm']
            elif signals['opcode'] == 0x67:  # JALR
                next_pc = (rs1_data + signals['imm']) & 0xFFFFFFFE
        
        if verbose:
            self.print_cycle(instruction, signals, rs1_data, rs2_data, alu_result)
        
        self.pc = next_pc
        self.cycle += 1
        
        return True
    
    def run(self, max_cycles=1000, verbose=False):
        """Run program until halt or max cycles."""
        print(f"\n{'='*70}")
        print(f"STARTING CPU EXECUTION")
        print(f"{'='*70}\n")
        
        while self.cycle < max_cycles and not self.halted:
            if not self.step(verbose):
                break
        
        if self.cycle >= max_cycles:
            print(f"\n⚠ Reached maximum cycles ({max_cycles})")
        else:
            print(f"\n✓ Program halted after {self.cycle} cycles")
    
    def print_cycle(self, instruction, signals, rs1_data, rs2_data, alu_result):
        """Print detailed cycle information."""
        print(f"\n{'='*70}")
        print(f"CYCLE {self.cycle}")
        print(f"{'='*70}")
        print(f"PC: 0x{self.pc:08X}")
        print(f"Instruction: 0x{instruction:08X}")
        print(f"Opcode: 0x{signals['opcode']:02X}")
        
        instr_name = self.get_instruction_name(signals)
        print(f"Decoded: {instr_name}")
        
        print(f"RS1 (x{signals['rs1']}): 0x{rs1_data:08X}")
        print(f"RS2 (x{signals['rs2']}): 0x{rs2_data:08X}")
        
        if signals['alu_src'] == 1:
            print(f"Immediate: 0x{signals['imm'] & 0xFFFFFFFF:08X}")
        
        print(f"ALU Result: 0x{alu_result:08X}")
        
        if signals['reg_write']:
            print(f"Write RD (x{signals['rd']})")
    
    def get_instruction_name(self, signals):
        """Get human-readable instruction name."""
        opcode = signals['opcode']
        funct3 = signals['funct3']
        funct7 = signals['funct7']
        
        if opcode == 0x33:  # R-type
            names = {
                (0x0, 0x00): 'ADD', (0x0, 0x20): 'SUB',
                (0x7, 0x00): 'AND', (0x6, 0x00): 'OR',
                (0x4, 0x00): 'XOR', (0x1, 0x00): 'SLL',
                (0x5, 0x00): 'SRL', (0x5, 0x20): 'SRA',
            }
            return names.get((funct3, funct7), 'UNKNOWN')
        elif opcode == 0x13:  # I-type
            names = {
                0x0: 'ADDI', 0x7: 'ANDI', 0x6: 'ORI',
                0x4: 'XORI', 0x1: 'SLLI',
                0x5: 'SRLI' if funct7 == 0x00 else 'SRAI'
            }
            return names.get(funct3, 'UNKNOWN')
        elif opcode == 0x03:
            return 'LW'
        elif opcode == 0x23:
            return 'SW'
        elif opcode == 0x63:
            return 'BEQ' if funct3 == 0x0 else 'BNE'
        elif opcode == 0x6F:
            return 'JAL'
        elif opcode == 0x67:
            return 'JALR'
        elif opcode == 0x37:
            return 'LUI'
        elif opcode == 0x17:
            return 'AUIPC'
        return 'UNKNOWN'


def assemble_instruction(opcode, rd=0, rs1=0, rs2=0, funct3=0, funct7=0, imm=0):
    """Helper to manually assemble instructions."""
    if opcode == 0x33:  # R-type
        return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    elif opcode in [0x13, 0x03, 0x67]:  # I-type
        return ((imm & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    elif opcode == 0x23:  # S-type
        imm_low = imm & 0x1F
        imm_high = (imm >> 5) & 0x7F
        return (imm_high << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_low << 7) | opcode
    elif opcode in [0x37, 0x17]:  # U-type
        return (imm & 0xFFFFF000) | (rd << 7) | opcode
    return 0


def test_arithmetic():
    """Test arithmetic instructions."""
    print("\n" + "="*70)
    print("TEST 1: ARITHMETIC INSTRUCTIONS")
    print("="*70)
    
    cpu = CPU()
    
    

    program = [
        assemble_instruction(0x13, rd=1, rs1=0, imm=10, funct3=0),
        assemble_instruction(0x13, rd=2, rs1=0, imm=5, funct3=0),
        assemble_instruction(0x33, rd=3, rs1=1, rs2=2, funct3=0),
        assemble_instruction(0x33, rd=4, rs1=1, rs2=2, funct3=0, funct7=0x20),
        0
    ]
    
    cpu.load_program(program)
    cpu.run(verbose=True)
    cpu.registers.print_registers()
    
    assert cpu.registers.read(1) == 10
    assert cpu.registers.read(2) == 5
    assert cpu.registers.read(3) == 15
    assert cpu.registers.read(4) == 5
    assert cpu.registers.read(3) == 15
    print("\n✓ Arithmetic tests passed!")


def test_logical():
    """Test logical instructions."""
    print("\n" + "="*70)
    print("TEST 2: LOGICAL INSTRUCTIONS")
    print("="*70)
    
    cpu = CPU()
    
    program = [
        assemble_instruction(0x13, rd=1, rs1=0, imm=0xFF, funct3=0),
        assemble_instruction(0x13, rd=2, rs1=0, imm=0x0F, funct3=0),
        assemble_instruction(0x33, rd=3, rs1=1, rs2=2, funct3=0x7),
        assemble_instruction(0x33, rd=4, rs1=1, rs2=2, funct3=0x6),
        assemble_instruction(0x33, rd=5, rs1=1, rs2=2, funct3=0x4),
        0
    ]
    
    cpu.load_program(program)
    cpu.run(verbose=True)
    cpu.registers.print_registers()
    
    assert cpu.registers.read(3) == 0x0F
    assert cpu.registers.read(4) == 0xFF
    assert cpu.registers.read(5) == 0xF0
    print("\n✓ Logical tests passed!")


def test_memory():
    """Test load and store."""
    print("\n" + "="*70)
    print("TEST 3: MEMORY INSTRUCTIONS")
    print("="*70)
    
    cpu = CPU()
    
    program = [
        assemble_instruction(0x13, rd=1, rs1=0, imm=100, funct3=0),
        assemble_instruction(0x13, rd=2, rs1=0, imm=0x42, funct3=0),
        assemble_instruction(0x23, rs1=1, rs2=2, imm=0, funct3=0x2),
        assemble_instruction(0x03, rd=3, rs1=1, imm=0, funct3=0x2),
        0
    ]
    
    cpu.load_program(program)
    cpu.run(verbose=True)
    cpu.registers.print_registers()
    cpu.memory.dump_range(100, 104)
    
    assert cpu.registers.read(3) == 0x42
    print("\n✓ Memory tests passed!")

'''
def test_hexcode():
    """Test hex code"""
    print("\n" + "="*70)
    print("TEST 4: HEX INSTRUCTIONS")
    print("="*70)

    cpu = CPU()

    instructions = []
    program = []
    file_path = "test_base.hex"

    try:
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, 1): # enumerate adds line numbers, starting from 1
                print(f"Line {line_number}: {line.strip()}")
                hex_string = line.strip()
                int_value = int(hex_string,16)
                instructions.append(int_value)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    
    for i in instructions:
        signals = cpu.control.decode(i)
        program.append(assemble_instruction(signals['opcode'],signals['rd'],signals['rs1'],signals['rs2'],signals['funct3'],signals['funct7'],signals['imm']))

    cpu.load_program(program)
    cpu.run(verbose=True)
    cpu.registers.print_registers()
'''
    
def test_all():
    """Run all tests."""
    print("\n" + "="*70)
    print("SINGLE-CYCLE RISC-V CPU TEST SUITE")
    print("="*70)
    
    test_arithmetic()
    test_logical()
    test_memory()
    test_hexcode()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    test_all()
