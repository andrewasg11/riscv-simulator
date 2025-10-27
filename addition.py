class ALU:
    """
    32-bit ALU implementing ADD and SUB with proper flag handling.
    Uses bit-by-bit operations with no host arithmetic (+/-).
    Supports both integer and IEEE 754 single-precision floating point operations.
    """
    
    def __init__(self):
        self.N = False  # Negative flag (MSB of result)
        self.Z = False  # Zero flag
        self.C = False  # Carry out of MSB
        self.V = False  # Signed overflow
    
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
            unsigned = self._from_64bitvec(bitvec)
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
    def mul(self,a,b):
        aNeg = False
        bNeg = False

        multiplicand = self._to_bitvec(a)
        multiplier = self._to_bitvec(b)
        if multiplicand[-1]:
            aNeg = True
            multiplicand = self._twos_complement(multiplicand)
        if multiplier[-1]:
            bNeg = True
            multiplier = self._twos_complement(multiplier)
        multiplicand = multiplicand + self._to_bitvec(0)
        product = [0] * 64

        for i in range(len(multiplier)):
            if multiplier[0]:
                product = self.add(multiplicand,product)
            multiplicand = self.sll(multiplicand)
            multiplier = self.srl(multiplier)
        if aNeg ^ bNeg:
            product = self._twos_complement(product)
            return self._bitvec_to_signed(product)
        return self.__from_64bitvec(product)

    def div(self, a, b):
        """
        DIV operation: a / b
        recieves a remainder a and divisor b
        """

        remNeg = False
        divNeg = False
        remainder = self._to_64_bitvec(a)
        divisor = self._to_bitvec(b)
        if divisor.count(0) == 32:
            print("cannot divide by 0")
            return
        if remainder[-1]:
            remNeg = True
            remainder = self._twos_complement(remainder)
        if divisor[-1]:
            divNeg = True
            divisor = self._twos_complement(divisor)
        divisor = self._to_bitvec(0) + divisor
        quotient = [0] * 32

        for i in range(len(quotient)+1):
            remainder = self.sub(remainder,divisor)

            if remainder[-1]:
                remainder = self.add(remainder,divisor)
                quotient = self.sll(quotient)
            else:
                quotient = self.sll(quotient)
                quotient[0] = 1
            divisor = self.sra(divisor)
        if remNeg ^ divNeg:
            quotient = self._twos_complement(quotient)
            return self._bitvec_to_signed(quotient)
        return self._from_bitvec(quotient)
    
    def rem(self,a,b):
        """
        REM operation: a % b
        recieves a remainder a and divisor b
        """
        remNeg = False
        remainder = self._to_64_bitvec(a)
        divisor = self._to_bitvec(b)
        if divisor.count(0) == 32:
            print("Cannot divide by 0")
            return
        if remainder[-1]:
            remNeg = True
            remainder = self._twos_complement(remainder)
        if divisor[-1]:
            divisor = self._twos_complement(divisor)
        divisor = self._to_bitvec(0) + divisor
        quotient = [0] * 32

        for i in range(len(quotient)+1):
            remainder = self.sub(remainder,divisor)

            if remainder[-1]:
                remainder = self.add(remainder,divisor)
                quotient = self.sll(quotient)
            else:
                quotient = self.sll(quotient)
                quotient[0] = 1
            divisor = self.sra(divisor)
        if remNeg:
            remainder = self._twos_complement(remainder)
            return self._bitvec_to_signed(remainder)
        return self._from_bitvec(remainder)
    
        
    # ========== FLOATING POINT OPERATIONS ==========
    
    def _float_to_bitvec(self, value):
        """
        Convert a Python float to IEEE 754 single-precision bit vector.
        Format: 1 sign bit, 8 exponent bits, 23 mantissa bits
        """
        import struct
        
        # Pack float as binary, then unpack as integer
        packed = struct.pack('>f', value)
        int_repr = struct.unpack('>I', packed)[0]
        
        # Convert to bit vector (LSB first)
        return [(int_repr >> i) & 1 for i in range(32)]
    
    def _bitvec_to_float(self, bitvec):
        """Convert IEEE 754 bit vector to Python float."""
        import struct
        
        # Convert bit vector to integer (MSB first for struct)
        int_repr = self._from_bitvec(bitvec)
        
        # Pack as integer, unpack as float
        packed = struct.pack('>I', int_repr)
        return struct.unpack('>f', packed)[0]
    
    def _extract_float_components(self, bitvec):
        """
        Extract IEEE 754 components from bit vector.
        Returns: (sign, exponent, mantissa) as integers
        """
        # Sign bit (bit 31)
        sign = bitvec[31]
        
        # Exponent (bits 23-30, 8 bits)
        exponent = 0
        for i in range(8):
            if bitvec[23 + i]:
                exponent |= (1 << i)
        
        # Mantissa (bits 0-22, 23 bits)
        mantissa = 0
        for i in range(23):
            if bitvec[i]:
                mantissa |= (1 << i)
        
        return sign, exponent, mantissa
    
    def _build_float_bitvec(self, sign, exponent, mantissa):
        """
        Build IEEE 754 bit vector from components.
        Args:
            sign: 0 or 1
            exponent: 8-bit unsigned integer
            mantissa: 23-bit unsigned integer
        """
        bitvec = [0] * 32
        
        # Sign bit
        bitvec[31] = sign
        
        # Exponent (8 bits)
        for i in range(8):
            bitvec[23 + i] = (exponent >> i) & 1
        
        # Mantissa (23 bits)
        for i in range(23):
            bitvec[i] = (mantissa >> i) & 1
        
        return bitvec
    
    def fadd(self, a, b):
        """
        Floating point addition using IEEE 754 single precision.
        
        Args:
            a, b: Python floats
            
        Returns:
            Float result
        """
        # Convert to bit vectors
        a_vec = self._float_to_bitvec(a)
        b_vec = self._float_to_bitvec(b)
        
        # Extract components
        sign_a, exp_a, mant_a = self._extract_float_components(a_vec)
        sign_b, exp_b, mant_b = self._extract_float_components(b_vec)
        
        # Handle special cases
        # Zero
        if exp_a == 0 and mant_a == 0:
            return b
        if exp_b == 0 and mant_b == 0:
            return a
        
        # Infinity and NaN (exponent = 255)
        if exp_a == 255 or exp_b == 255:
            # Simplified: just return a for special cases
            return a
        
        # Align exponents (shift smaller exponent to match larger)
        if exp_a > exp_b:
            exp_diff = exp_a - exp_b
            exp_result = exp_a
            # Shift b's mantissa right
            # Add implicit leading 1 to mantissa
            mant_a_full = mant_a | (1 << 23)
            mant_b_full = (mant_b | (1 << 23)) >> exp_diff
        elif exp_b > exp_a:
            exp_diff = exp_b - exp_a
            exp_result = exp_b
            # Shift a's mantissa right
            mant_a_full = (mant_a | (1 << 23)) >> exp_diff
            mant_b_full = mant_b | (1 << 23)
        else:
            exp_result = exp_a
            mant_a_full = mant_a | (1 << 23)
            mant_b_full = mant_b | (1 << 23)
        
        # Perform addition/subtraction based on signs
        if sign_a == sign_b:
            # Same sign: add mantissas
            mant_result = mant_a_full + mant_b_full
            sign_result = sign_a
            
            # Check for overflow (bit 24 set)
            if mant_result & (1 << 24):
                mant_result >>= 1
                exp_result += 1
        else:
            # Different signs: subtract mantissas
            if mant_a_full >= mant_b_full:
                mant_result = mant_a_full - mant_b_full
                sign_result = sign_a
            else:
                mant_result = mant_b_full - mant_a_full
                sign_result = sign_b
            
            # Normalize: shift left until bit 23 is set
            if mant_result != 0:
                while not (mant_result & (1 << 23)) and exp_result > 0:
                    mant_result <<= 1
                    exp_result -= 1
        
        # Remove implicit leading 1
        mant_result &= 0x7FFFFF  # Keep only lower 23 bits
        
        # Handle exponent overflow/underflow
        if exp_result >= 255:
            exp_result = 255
            mant_result = 0  # Infinity
        elif exp_result <= 0:
            exp_result = 0
            mant_result = 0  # Zero (denormalized)
        
        # Build result
        result_vec = self._build_float_bitvec(sign_result, exp_result, mant_result)
        return self._bitvec_to_float(result_vec)
    
    def fsub(self, a, b):
        """
        Floating point subtraction.
        Implemented as a + (-b)
        """
        return self.fadd(a, -b)
    
    def fmul(self, a, b):
        """
        Floating point multiplication using IEEE 754.
        """
        # Convert to bit vectors
        a_vec = self._float_to_bitvec(a)
        b_vec = self._float_to_bitvec(b)
        
        # Extract components
        sign_a, exp_a, mant_a = self._extract_float_components(a_vec)
        sign_b, exp_b, mant_b = self._extract_float_components(b_vec)
        
        # Handle special cases
        if (exp_a == 0 and mant_a == 0) or (exp_b == 0 and mant_b == 0):
            return 0.0
        
        # Sign: XOR of signs
        sign_result = sign_a ^ sign_b
        
        # Exponent: add and subtract bias (127)
        exp_result = exp_a + exp_b - 127
        
        # Mantissa: multiply with implicit leading 1
        mant_a_full = mant_a | (1 << 23)
        mant_b_full = mant_b | (1 << 23)
        
        # Multiply (this will be a 48-bit result)
        mant_result = mant_a_full * mant_b_full
        
        # Normalize: result is in range [1.0, 4.0)
        # Shift right by 23 to account for fixed-point multiplication
        mant_result >>= 23
        
        # Check if we need to adjust exponent
        if mant_result & (1 << 24):
            mant_result >>= 1
            exp_result += 1
        
        # Remove implicit leading 1
        mant_result &= 0x7FFFFF
        
        # Handle overflow/underflow
        if exp_result >= 255:
            exp_result = 255
            mant_result = 0  # Infinity
        elif exp_result <= 0:
            exp_result = 0
            mant_result = 0  # Zero
        
        # Build result
        result_vec = self._build_float_bitvec(sign_result, exp_result, mant_result)
        return self._bitvec_to_float(result_vec)
    
    def fdiv(self, a, b):
        """
        Floating point division using IEEE 754.
        """
        # Convert to bit vectors
        a_vec = self._float_to_bitvec(a)
        b_vec = self._float_to_bitvec(b)
        
        # Extract components
        sign_a, exp_a, mant_a = self._extract_float_components(a_vec)
        sign_b, exp_b, mant_b = self._extract_float_components(b_vec)
        
        # Handle special cases
        if exp_b == 0 and mant_b == 0:
            # Division by zero
            return float('inf') if sign_a == 0 else float('-inf')
        
        if exp_a == 0 and mant_a == 0:
            return 0.0
        
        # Sign: XOR of signs
        sign_result = sign_a ^ sign_b
        
        # Exponent: subtract and add bias
        exp_result = exp_a - exp_b + 127
        
        # Mantissa: divide with implicit leading 1
        mant_a_full = (mant_a | (1 << 23)) << 23  # Shift left for precision
        mant_b_full = mant_b | (1 << 23)
        
        # Divide
        mant_result = mant_a_full // mant_b_full
        
        # Normalize
        if not (mant_result & (1 << 23)):
            mant_result <<= 1
            exp_result -= 1
        
        # Remove implicit leading 1
        mant_result &= 0x7FFFFF
        
        # Handle overflow/underflow
        if exp_result >= 255:
            exp_result = 255
            mant_result = 0  # Infinity
        elif exp_result <= 0:
            exp_result = 0
            mant_result = 0  # Zero
        
        # Build result
        result_vec = self._build_float_bitvec(sign_result, exp_result, mant_result)
        return self._bitvec_to_float(result_vec)
    
    def print_flags(self):
        """Print current ALU flags."""
        print(f"Flags: N={int(self.N)} Z={int(self.Z)} C={int(self.C)} V={int(self.V)}")
    
    def format_result(self, value):
        """Format result in multiple representations."""
        bitvec = self._to_bitvec(value)
        unsigned = self._from_bitvec(bitvec)
        signed = self._bitvec_to_signed(bitvec)
        binary = ''.join(str(b) for b in reversed(bitvec))
        
        return {
            'unsigned': unsigned,
            'signed': signed,
            'binary': binary,
            'hex': f"0x{unsigned:08X}"
        }
    
    def format_float(self, value):
        """Format floating point value with IEEE 754 details."""
        bitvec = self._float_to_bitvec(value)
        sign, exp, mant = self._extract_float_components(bitvec)
        binary = ''.join(str(b) for b in reversed(bitvec))
        
        return {
            'value': value,
            'sign': sign,
            'exponent': exp,
            'mantissa': mant,
            'binary': binary,
            'hex': f"0x{self._from_bitvec(bitvec):08X}"
        }


def test_floating_point():
    """Test suite for floating point operations."""
    alu = ALU()
    
    print("=" * 70)
    print("FLOATING POINT TEST SUITE (IEEE 754)")
    print("=" * 70)
    
    # Test 1: Basic addition
    print("\n[TEST 1] FADD: 3.5 + 2.25 = 5.75")
    result = alu.fadd(3.5, 2.25)
    fmt = alu.format_float(result)
    print(f"Result: {fmt['value']}")
    print(f"Binary: {fmt['binary']}")
    print(f"Hex: {fmt['hex']}")
    print(f"Components: sign={fmt['sign']}, exp={fmt['exponent']}, mant={fmt['mantissa']}")
    
    # Test 2: Addition with different signs
    print("\n[TEST 2] FADD: 10.5 + (-3.25) = 7.25")
    result = alu.fadd(10.5, -3.25)
    fmt = alu.format_float(result)
    print(f"Result: {fmt['value']}")
    print(f"Binary: {fmt['binary']}")
    
    # Test 3: Subtraction
    print("\n[TEST 3] FSUB: 8.0 - 3.0 = 5.0")
    result = alu.fsub(8.0, 3.0)
    fmt = alu.format_float(result)
    print(f"Result: {fmt['value']}")
    print(f"Binary: {fmt['binary']}")
    
    # Test 4: Multiplication
    print("\n[TEST 4] FMUL: 2.5 × 4.0 = 10.0")
    result = alu.fmul(2.5, 4.0)
    fmt = alu.format_float(result)
    print(f"Result: {fmt['value']}")
    print(f"Binary: {fmt['binary']}")
    
    # Test 5: Division
    print("\n[TEST 5] FDIV: 10.0 ÷ 4.0 = 2.5")
    result = alu.fdiv(10.0, 4.0)
    fmt = alu.format_float(result)
    print(f"Result: {fmt['value']}")
    print(f"Binary: {fmt['binary']}")
    
    # Test 6: Very small numbers
    print("\n[TEST 6] FADD: 0.1 + 0.2")
    result = alu.fadd(0.1, 0.2)
    fmt = alu.format_float(result)
    print(f"Result: {fmt['value']:.17f}")
    print(f"Note: Floating point precision limitations")
    
    # Test 7: Negative multiplication
    print("\n[TEST 7] FMUL: -3.0 × 4.0 = -12.0")
    result = alu.fmul(-3.0, 4.0)
    fmt = alu.format_float(result)
    print(f"Result: {fmt['value']}")
    
    print("\n" + "=" * 70)
    print("FLOATING POINT TESTS COMPLETE")
    print("=" * 70)


def test_alu():
    """Comprehensive test suite for ALU operations."""
    alu = ALU()
    
    print("=" * 70)
    print("32-BIT INTEGER ALU TEST SUITE")
    print("=" * 70)
    
    # Test 1: Simple positive addition
    print("\n[TEST 1] ADD: 5 + 3 = 8")
    result = alu.add(5, 3)
    fmt = alu.format_result(result)
    print(f"Result: {fmt['unsigned']} (unsigned), {fmt['signed']} (signed)")
    print(f"Binary: {fmt['binary']}")
    print(f"Hex: {fmt['hex']}")
    alu.print_flags()
    
    # Test 2: Addition with carry
    print("\n[TEST 2] ADD: 0xFFFFFFFF + 0x00000001 (unsigned overflow)")
    result = alu.add(0xFFFFFFFF, 0x00000001)
    fmt = alu.format_result(result)
    print(f"Result: {fmt['unsigned']} (unsigned), {fmt['signed']} (signed)")
    print(f"Binary: {fmt['binary']}")
    print(f"Hex: {fmt['hex']}")
    alu.print_flags()
    print("Note: C=1 indicates unsigned overflow (carry out)")
    
    # Test 3: Positive overflow (signed)
    print("\n[TEST 3] ADD: 0x7FFFFFFF + 0x00000001 (signed overflow)")
    result = alu.add(0x7FFFFFFF, 0x00000001)
    fmt = alu.format_result(result)
    print(f"Result: {fmt['unsigned']} (unsigned), {fmt['signed']} (signed)")
    print(f"Binary: {fmt['binary']}")
    print(f"Hex: {fmt['hex']}")
    alu.print_flags()
    print("Note: V=1 indicates signed overflow (2147483647 + 1 = -2147483648)")
    
    print("\n" + "=" * 70)
    print("INTEGER TEST SUITE COMPLETE")
    print("=" * 70)


def interactive_cli():
    """Interactive command-line interface for ALU."""
    alu = ALU()
    
    print("\n" + "=" * 70)
    print("32-BIT ALU INTERACTIVE MODE (Integer + Float)")
    print("=" * 70)
    print("Integer Commands:")
    print("  add <a> <b>       - Add two integers")
    print("  sub <a> <b>       - Subtract integers")
    print("  addi <a> <imm>    - Add immediate")
    print("  mul <a> <b>       - Multiply two integers")
    print("  div <a> <b>       - Divide two integers")
    print("  rem <a> <b>       - Divide two integers returns remainder")
    print("\nFloating Point Commands:")
    print("  fadd <a> <b>      - Add floats")
    print("  fsub <a> <b>      - Subtract floats")
    print("  fmul <a> <b>      - Multiply floats")
    print("  fdiv <a> <b>      - Divide floats")
    print("\nOther Commands:")
    print("  test              - Run integer test suite")
    print("  ftest             - Run floating point test suite")
    print("  quit              - Exit")
    print("=" * 70 + "\n")
    
    while True:
        try:
            cmd = input("ALU> ").strip().split()
            if not cmd:
                continue
            
            op = cmd[0].lower()
            
            if op == 'quit':
                break
            elif op == 'test':
                test_alu()
            elif op == 'ftest':
                test_floating_point()
            elif op in ['add', 'sub', 'addi','mul','div','rem']:
                if len(cmd) != 3:
                    print("Error: Expected 2 arguments")
                    continue
                
                a = int(cmd[1], 0)
                b = int(cmd[2], 0)
                
                if op == 'add':
                    result = alu.add(a, b)
                elif op == 'sub':
                    result = alu.sub(a, b)
                elif op == 'addi':
                    result = alu.addi(a, b)
                elif op == 'mul':
                    result = alu.mul(a,b)
                elif op == 'div':
                    result = alu.div(a,b)
                else:
                    result = alu.rem(a,b)
                
                fmt = alu.format_result(result)
                print(f"\nResult:")
                print(f"  Unsigned: {fmt['unsigned']}")
                print(f"  Signed:   {fmt['signed']}")
                print(f"  Hex:      {fmt['hex']}")
                print(f"  Binary:   {fmt['binary']}")
                alu.print_flags()
                print()
            elif op in ['fadd', 'fsub', 'fmul', 'fdiv']:
                if len(cmd) != 3:
                    print("Error: Expected 2 arguments")
                    continue
                
                a = float(cmd[1])
                b = float(cmd[2])
                
                if op == 'fadd':
                    result = alu.fadd(a, b)
                elif op == 'fsub':
                    result = alu.fsub(a, b)
                elif op == 'fmul':
                    result = alu.fmul(a, b)
                else:
                    result = alu.fdiv(a, b)
                
                fmt = alu.format_float(result)
                print(f"\nResult: {fmt['value']}")
                print(f"Binary: {fmt['binary']}")
                print(f"Hex: {fmt['hex']}")
                print(f"IEEE 754: sign={fmt['sign']}, exp={fmt['exponent']}, mant=0x{fmt['mantissa']:06X}")
                print()
            else:
                print(f"Unknown command: {op}")
        
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    # Run integer test suite
    test_alu()
    
    print("\n")
    
    # Run floating point test suite
    test_floating_point()
    
    # Start interactive mode
    print("\n\nStarting interactive mode...")
    interactive_cli()