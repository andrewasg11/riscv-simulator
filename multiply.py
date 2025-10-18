import struct


class BinaryMultiplier:
    """
    Binary multiplication with support for:
    - Signed/unsigned integers
    - Two's complement
    - Overflow detection
    - IEEE 754 floating point
    """
    
    def __init__(self, bit_width=32):
        """
        Initialize multiplier with specified bit width.
        
        Args:
            bit_width: Number of bits for integer operations (default: 32)
        """
        self.bit_width = bit_width
        self.overflow = False
        self.negative = False
    
    def binary_add(self, bin1, bin2):
        """Add two binary numbers bit by bit."""
        # Pad to same length
        max_len = max(len(bin1), len(bin2))
        bin1 = bin1.zfill(max_len)
        bin2 = bin2.zfill(max_len)
        
        result = []
        carry = 0
        
        # Add from right to left
        for i in range(max_len - 1, -1, -1):
            bit1 = int(bin1[i])
            bit2 = int(bin2[i])
            
            total = bit1 + bit2 + carry
            result.append(str(total % 2))
            carry = total // 2
        
        if carry:
            result.append('1')
        
        # Reverse and return
        return ''.join(reversed(result))
    
    def twos_complement(self, binary):
        """
        Compute two's complement: invert all bits and add 1.
        
        Args:
            binary: Binary string
            
        Returns:
            Two's complement as binary string
        """
        # Invert all bits
        inverted = ''.join('1' if b == '0' else '0' for b in binary)
        
        # Add 1
        return self.binary_add(inverted, '1')
    
    def binary_multiply_unsigned(self, bin1, bin2, verbose=True):
        """
        Multiply two unsigned binary numbers bit by bit.
        
        Args:
            bin1: First binary number as string (e.g., '1011')
            bin2: Second binary number as string (e.g., '101')
            verbose: Print step-by-step process
        
        Returns:
            Result as binary string
        """
        # Remove '0b' prefix if present
        bin1 = bin1.replace('0b', '')
        bin2 = bin2.replace('0b', '')
        
        # Result storage
        partial_products = []
        
        if verbose:
            print(f"Multiplying {bin1} × {bin2} (unsigned)")
            print("-" * 50)
        
        # Process each bit of the multiplier (right to left)
        for i in range(len(bin2) - 1, -1, -1):
            bit = bin2[i]
            shift = len(bin2) - 1 - i
            
            # Create partial product
            if bit == '1':
                # Copy multiplicand and shift left
                partial = bin1 + '0' * shift
            else:
                # Bit is 0, partial product is 0
                partial = '0'
            
            partial_products.append(partial)
            if verbose:
                print(f"Bit {len(bin2) - i}: {bin1} × {bit} = {partial}")
        
        if verbose:
            print("\nAdding partial products:")
            for pp in partial_products:
                print(f"  {pp:>25}")
        
        # Add all partial products
        result = '0'
        for partial in partial_products:
            result = self.binary_add(result, partial)
        
        if verbose:
            print("-" * 50)
            print(f"Result: {result}")
        
        return result
    
    def binary_multiply_signed(self, bin1, bin2, verbose=True):
        """
        Multiply two signed binary numbers using Booth's algorithm (simplified).
        Handles two's complement representation.
        
        Args:
            bin1: First binary number (two's complement)
            bin2: Second binary number (two's complement)
            verbose: Print step-by-step process
            
        Returns:
            Result as binary string (two's complement)
        """
        bin1 = bin1.replace('0b', '')
        bin2 = bin2.replace('0b', '')
        
        # Pad to bit width
        bin1 = bin1.zfill(self.bit_width)
        bin2 = bin2.zfill(self.bit_width)
        
        if verbose:
            print(f"Multiplying {bin1} × {bin2} (signed, two's complement)")
            print("-" * 50)
        
        # Determine signs
        sign1 = bin1[0] == '1'
        sign2 = bin2[0] == '1'
        result_negative = sign1 ^ sign2  # XOR for result sign
        
        if verbose:
            print(f"Operand 1 sign: {'negative' if sign1 else 'positive'}")
            print(f"Operand 2 sign: {'negative' if sign2 else 'positive'}")
            print(f"Expected result sign: {'negative' if result_negative else 'positive'}")
        
        # Convert to positive if negative
        abs_bin1 = self.twos_complement(bin1) if sign1 else bin1
        abs_bin2 = self.twos_complement(bin2) if sign2 else bin2
        
        # Multiply absolute values
        if verbose:
            print(f"\nMultiplying absolute values:")
        result = self.binary_multiply_unsigned(abs_bin1, abs_bin2, verbose=verbose)
        
        # Truncate to bit width and check overflow
        if len(result) > self.bit_width:
            self.overflow = True
            result = result[-self.bit_width:]  # Keep lower bits
            if verbose:
                print(f"\n⚠ WARNING: Overflow detected! Result truncated to {self.bit_width} bits")
        else:
            self.overflow = False
            result = result.zfill(self.bit_width)
        
        # Apply sign if needed
        if result_negative:
            result = self.twos_complement(result)
            self.negative = True
        else:
            self.negative = False
        
        if verbose:
            print(f"\nFinal signed result: {result}")
        
        return result
    
    def multiply_integers(self, a, b, signed=True, verbose=True):
        """
        Multiply two integers with overflow detection.
        
        Args:
            a, b: Integer values
            signed: Treat as signed integers (default: True)
            verbose: Print details
            
        Returns:
            Dictionary with result and flags
        """
        if signed:
            # Convert to two's complement binary
            bin1 = self._int_to_binary_signed(a)
            bin2 = self._int_to_binary_signed(b)
            result_bin = self.binary_multiply_signed(bin1, bin2, verbose=verbose)
            result_int = self._binary_to_int_signed(result_bin)
        else:
            # Unsigned multiplication
            bin1 = bin(a & ((1 << self.bit_width) - 1))[2:]
            bin2 = bin(b & ((1 << self.bit_width) - 1))[2:]
            result_bin = self.binary_multiply_unsigned(bin1, bin2, verbose=verbose)
            
            # Check overflow for unsigned
            max_unsigned = (1 << self.bit_width) - 1
            result_int = int(result_bin, 2)
            self.overflow = result_int > max_unsigned
            
            if self.overflow:
                result_int = result_int & max_unsigned
                if verbose:
                    print(f"\n⚠ Unsigned overflow detected!")
        
        if verbose:
            print(f"\nDecimal result: {result_int}")
            print(f"Overflow: {self.overflow}")
            print(f"Negative: {self.negative}")
        
        return {
            'result': result_int,
            'binary': result_bin,
            'overflow': self.overflow,
            'negative': self.negative
        }
    
    def _int_to_binary_signed(self, value):
        """Convert signed integer to two's complement binary string."""
        if value >= 0:
            binary = bin(value)[2:].zfill(self.bit_width)
        else:
            # Convert negative number to two's complement
            binary = bin((1 << self.bit_width) + value)[2:]
        
        return binary[-self.bit_width:]  # Ensure bit width
    
    def _binary_to_int_signed(self, binary):
        """Convert two's complement binary string to signed integer."""
        value = int(binary, 2)
        # Check if negative (MSB is 1)
        if binary[0] == '1':
            value -= (1 << len(binary))
        return value
    
    # ========== FLOATING POINT MULTIPLICATION ==========
    
    def _float_to_ieee754(self, value):
        """Convert Python float to IEEE 754 single-precision binary string."""
        packed = struct.pack('>f', value)
        int_repr = struct.unpack('>I', packed)[0]
        return bin(int_repr)[2:].zfill(32)
    
    def _ieee754_to_float(self, binary):
        """Convert IEEE 754 binary string to Python float."""
        int_repr = int(binary, 2)
        packed = struct.pack('>I', int_repr)
        return struct.unpack('>f', packed)[0]
    
    def _extract_ieee754_parts(self, binary):
        """
        Extract IEEE 754 components.
        Returns: (sign, exponent, mantissa)
        """
        sign = int(binary[0])
        exponent = int(binary[1:9], 2)
        mantissa = int(binary[9:], 2)
        return sign, exponent, mantissa
    
    def multiply_floats(self, a, b, verbose=True):
        """
        Multiply two floating point numbers using IEEE 754.
        
        Args:
            a, b: Float values
            verbose: Print details
            
        Returns:
            Dictionary with result and IEEE 754 details
        """
        # Convert to IEEE 754
        bin_a = self._float_to_ieee754(a)
        bin_b = self._float_to_ieee754(b)
        
        if verbose:
            print(f"\nMultiplying {a} × {b} (IEEE 754)")
            print("-" * 50)
            print(f"A = {a}")
            print(f"  Binary: {bin_a}")
            print(f"B = {b}")
            print(f"  Binary: {bin_b}")
        
        # Extract components
        sign_a, exp_a, mant_a = self._extract_ieee754_parts(bin_a)
        sign_b, exp_b, mant_b = self._extract_ieee754_parts(bin_b)
        
        if verbose:
            print(f"\nA components: sign={sign_a}, exp={exp_a}, mant=0x{mant_a:06X}")
            print(f"B components: sign={sign_b}, exp={exp_b}, mant=0x{mant_b:06X}")
        
        # Handle special cases
        if (exp_a == 0 and mant_a == 0) or (exp_b == 0 and mant_b == 0):
            result = 0.0
            if verbose:
                print("\nResult: 0.0 (one operand is zero)")
            return {'result': result, 'binary': self._float_to_ieee754(result)}
        
        # Sign: XOR of signs
        sign_result = sign_a ^ sign_b
        
        # Exponent: add and subtract bias (127)
        exp_result = exp_a + exp_b - 127
        
        # Mantissa: multiply with implicit leading 1
        mant_a_full = mant_a | (1 << 23)  # Add implicit 1
        mant_b_full = mant_b | (1 << 23)
        
        if verbose:
            print(f"\nMantissa with implicit 1:")
            print(f"  A: 0x{mant_a_full:06X}")
            print(f"  B: 0x{mant_b_full:06X}")
        
        # Multiply mantissas (48-bit result)
        mant_result = mant_a_full * mant_b_full
        
        if verbose:
            print(f"  Product: 0x{mant_result:012X}")
        
        # Normalize: shift right by 23 bits
        mant_result >>= 23
        
        # Check if normalization needed
        if mant_result & (1 << 24):
            mant_result >>= 1
            exp_result += 1
        
        # Remove implicit leading 1
        mant_result &= 0x7FFFFF
        
        # Handle overflow/underflow
        if exp_result >= 255:
            exp_result = 255
            mant_result = 0  # Infinity
            if verbose:
                print("\n⚠ Exponent overflow - result is infinity")
        elif exp_result <= 0:
            exp_result = 0
            mant_result = 0  # Zero (denormalized)
            if verbose:
                print("\n⚠ Exponent underflow - result is zero")
        
        # Build result binary
        result_bin = (
            str(sign_result) +
            bin(exp_result)[2:].zfill(8) +
            bin(mant_result)[2:].zfill(23)
        )
        
        result = self._ieee754_to_float(result_bin)
        
        if verbose:
            print(f"\nResult components: sign={sign_result}, exp={exp_result}, mant=0x{mant_result:06X}")
            print(f"Result binary: {result_bin}")
            print(f"Result value: {result}")
        
        return {
            'result': result,
            'binary': result_bin,
            'sign': sign_result,
            'exponent': exp_result,
            'mantissa': mant_result
        }


def test_unsigned_multiplication():
    """Test unsigned binary multiplication."""
    mult = BinaryMultiplier(32)
    
    print("=" * 60)
    print("UNSIGNED BINARY MULTIPLICATION TESTS")
    print("=" * 60)
    
    # Test 1: Simple multiplication
    print("\n[TEST 1] 13 × 5 = 65")
    mult.multiply_integers(13, 5, signed=False, verbose=True)
    
    # Test 2: Larger numbers
    print("\n" + "=" * 60)
    print("\n[TEST 2] 255 × 256")
    mult.multiply_integers(255, 256, signed=False, verbose=True)
    
    # Test 3: Overflow
    print("\n" + "=" * 60)
    print("\n[TEST 3] Large multiplication causing overflow")
    mult.multiply_integers(0xFFFF, 0xFFFF, signed=False, verbose=True)


def test_signed_multiplication():
    """Test signed binary multiplication with two's complement."""
    mult = BinaryMultiplier(32)
    
    print("\n" + "=" * 60)
    print("SIGNED BINARY MULTIPLICATION TESTS (Two's Complement)")
    print("=" * 60)
    
    # Test 1: Positive × Positive
    print("\n[TEST 1] 7 × 3 = 21")
    mult.multiply_integers(7, 3, signed=True, verbose=True)
    
    # Test 2: Positive × Negative
    print("\n" + "=" * 60)
    print("\n[TEST 2] 5 × (-3) = -15")
    mult.multiply_integers(5, -3, signed=True, verbose=True)
    
    # Test 3: Negative × Negative
    print("\n" + "=" * 60)
    print("\n[TEST 3] (-4) × (-6) = 24")
    mult.multiply_integers(-4, -6, signed=True, verbose=True)
    
    # Test 4: Overflow with signed
    print("\n" + "=" * 60)
    print("\n[TEST 4] Large signed multiplication")
    mult.multiply_integers(100000, 50000, signed=True, verbose=True)


def test_floating_point_multiplication():
    """Test IEEE 754 floating point multiplication."""
    mult = BinaryMultiplier()
    
    print("\n" + "=" * 60)
    print("FLOATING POINT MULTIPLICATION TESTS (IEEE 754)")
    print("=" * 60)
    
    # Test 1: Simple multiplication
    print("\n[TEST 1] 2.5 × 4.0 = 10.0")
    mult.multiply_floats(2.5, 4.0, verbose=True)
    
    # Test 2: Negative multiplication
    print("\n" + "=" * 60)
    print("\n[TEST 2] -3.0 × 2.0 = -6.0")
    mult.multiply_floats(-3.0, 2.0, verbose=True)
    
    # Test 3: Fractional multiplication
    print("\n" + "=" * 60)
    print("\n[TEST 3] 0.5 × 0.25 = 0.125")
    mult.multiply_floats(0.5, 0.25, verbose=True)
    
    # Test 4: Very small numbers
    print("\n" + "=" * 60)
    print("\n[TEST 4] 1e-10 × 1e-10")
    mult.multiply_floats(1e-10, 1e-10, verbose=True)


def interactive_mode():
    """Interactive multiplication calculator."""
    mult = BinaryMultiplier(32)
    
    print("\n" + "=" * 60)
    print("INTERACTIVE BINARY MULTIPLIER")
    print("=" * 60)
    print("Commands:")
    print("  umul <a> <b>     - Unsigned multiplication")
    print("  smul <a> <b>     - Signed multiplication")
    print("  fmul <a> <b>     - Floating point multiplication")
    print("  test             - Run all tests")
    print("  quit             - Exit")
    print("=" * 60 + "\n")
    
    while True:
        try:
            cmd = input("Multiplier> ").strip().split()
            if not cmd:
                continue
            
            op = cmd[0].lower()
            
            if op == 'quit':
                break
            elif op == 'test':
                test_unsigned_multiplication()
                test_signed_multiplication()
                test_floating_point_multiplication()
            elif op == 'umul':
                if len(cmd) != 3:
                    print("Error: Expected 2 arguments")
                    continue
                a = int(cmd[1], 0)
                b = int(cmd[2], 0)
                mult.multiply_integers(a, b, signed=False, verbose=True)
            elif op == 'smul':
                if len(cmd) != 3:
                    print("Error: Expected 2 arguments")
                    continue
                a = int(cmd[1], 0)
                b = int(cmd[2], 0)
                mult.multiply_integers(a, b, signed=True, verbose=True)
            elif op == 'fmul':
                if len(cmd) != 3:
                    print("Error: Expected 2 arguments")
                    continue
                a = float(cmd[1])
                b = float(cmd[2])
                mult.multiply_floats(a, b, verbose=True)
            else:
                print(f"Unknown command: {op}")
            
            print()
        
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    # Run test suites
    test_unsigned_multiplication()
    test_signed_multiplication()
    test_floating_point_multiplication()
    
    # Start interactive mode
    print("\n\nStarting interactive mode...")
    interactive_mode()