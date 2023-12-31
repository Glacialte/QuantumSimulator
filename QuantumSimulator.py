from itertools import product
import cupy as cp
import numpy as np
from collections import Counter


def qubit():
    """Create a qubit, initialized to |0〉.
    Returns a complex tensor of shape (2,).
    Equivalent to quantum_register(1)."""
    return cp.array([1, 0j])


def random_qubit():
    """Create a qubit with random complex amplitudes.
    Returns a complex tensor of shape (2,).
    Equivalent to random_register(1)."""
    qubit = cp.random.randn(2) + 1j * cp.random.randn(2)
    qubit /= cp.linalg.norm(qubit)
    return qubit


def quantum_register(number_of_qubits):
    """Creates a linear register of qubits, initialized to |000...0〉.
    Returns a complex tensor of shape (2, 2, ..., 2)."""
    shape = (2,) * number_of_qubits
    first = (0,) * number_of_qubits
    register = cp.zeros(shape, dtype=cp.complex128)
    register[first] = 1+0j
    return register


def random_register(number_of_qubits):
    """Creates a linear register of qubits with random complex amplitudes.
    Returns a complex tensor of shape (2, 2, ..., 2)."""
    shape = (2,) * number_of_qubits
    register = cp.random.randn(*shape) + cp.random.randn(*shape) * 1j
    register = register / cp.linalg.norm(register)
    return register


def display(tensor):
    """Displays the tensor entries in tabular form."""
    for multiindex in product(*map(range, tensor.shape)):
        ket = '|' + str(multiindex)[1:-1].replace(', ', '') + '⟩'
        value = tensor[multiindex]
        print('%s\t%.5f + %.5f i' % (ket, value.real, value.imag))


def get_transposition(n, indices):
    """Helper function that reorders a tensor after a quantum gate is applied."""
    transpose = [0] * n
    k = len(indices)
    ptr = 0
    for i in range(n):
        if i in indices:
            transpose[i] = n - k + indices.index(i)
        else:
            transpose[i] =  ptr
            ptr += 1
    return transpose


def apply_gate(gate, *indices):
    """Applies a gate to one or more indices of a quantum register.
    This is a higher-order function: it returns another function that
    can be applied to a register."""
    axes = (indices, range(len(indices)))
    def op(register):
        return cp.tensordot(register, gate, axes=axes).transpose(
               get_transposition(register.ndim, indices))
    return op


def circuit(ops):
    """Constructs a circuit as a sequence of quantum gates.
    This higher-order function returns another function that
    can be applied to a quantum register."""
    def circ(register):
        for op in ops:
            register = op(register)
        return register
    return circ


def measure_circuit(ops, index=None):
    """Constructs a circuit and performs a measurement."""
    circ = circuit(ops)
    def m(register):
        if index is None:
            return measure_all() (circ(register))
        return measure(index) (circ(register))
    return m

##### Unary gates

def X(index):
    """Generates a Pauli X gate (also called a NOT gate) acting on a given index.
    It returns a function that can be applied to a register."""
    gate = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
    return apply_gate(gate, index)


def Y(index):
    """Generates a Pauli Y gate acting on a given index.
    It returns a function that can be applied to a register."""
    gate = cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128)
    return apply_gate(gate, index)


def Z(index):
    """Generates a Pauli Z gate acting on a given index.
    This is the same as a rotation of the Bloch sphere by pi radians about the Z-axis.
    It returns a function that can be applied to a register."""
    gate = cp.array([[1, 0], [0, -1]], dtype=cp.complex128)
    return apply_gate(gate, index)


def R(index, angle):
    """Generates a rotation of the Block sphere about the Z-axis by a given
    It returns a function that can be applied to a register."""
    gate = cp.array([[1, 0], [0, np.exp(1j * angle)]], dtype=cp.complex128)
    return apply_gate(gate, index)


def S(index):
    """The S gate is a 90 degree rotation of the Bloch sphere about the Z-axis."""
    return R(index, cp.pi/2)


def T(index):
    """The T gate is a 45 degree rotation of the Bloch sphere about the Z-axis."""
    return R(index, cp.pi/4)


def H(index):
    """Generates a Hadamard gate. It returns a function that can be applied to a register."""
    gate = cp.array([[1, 1], [1, -1]]) / cp.sqrt(2)
    return apply_gate(gate, index)


def SNOT(index):
    """Generates a 'square root of NOT' gate. It returns a function that can be applied to a register."""
    gate = cp.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
    return apply_gate(gate, index)


### Binary gates


def SWAP(i, j):
    """Generates a SWAP gate."""
    gate = cp.array([1, 0, 0, 0,
                     0, 0, 1, 0,
                     0, 1, 0, 0,
                     0, 0, 0, 1]
                     ).reshape((2, 2, 2, 2))
    return apply_gate(gate, i, j)


def CNOT(i, j):
    """Generates a controlled NOT gate, also called a controlled X gate."""
    gate = cp.array([1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 0, 1,
                     0, 0, 1, 0]
                     ).reshape((2, 2, 2, 2))
    return apply_gate(gate,i, j)

CX = CNOT


def CY(i, j):
    """Generates a controlled Y gate."""
    gate = cp.array([1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 0, -1j,
                     0, 0, 1j, 0]
                     ).reshape((2, 2, 2, 2))
    return apply_gate(gate, i, j)


def CZ(i, j):
    """Generates a controlled Z gate."""
    gate = cp.array([1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, -1]
                     ).reshape((2, 2, 2, 2))
    return apply_gate(gate, i, j)


def C(i, j, unary_gate):
    """Generates a controlled (binary) version of a unary gate.
    When the value at the first index is 1, apply the unary gate to the value at the second index.
    When the value at the first index is 0, do nothing."""
    gate = cp.zeros((2, 2, 2, 2))
    gate[0, :, 0, :] = cp.eye(2)
    gate[1, :, 1, :] = unary_gate
    return apply_gate(gate, i, j)


## Ternary gates

def CCNOT(i, j, k):
    """Generates a Toffoli gate."""
    gate = cp.eye(8)
    gate[6:8, 6:8] = cp.array([[0, 1], [1, 0]])
    gate = gate.reshape((2, 2, 2, 2, 2, 2))
    return apply_gate(gate, i, j, k)


def CSWAP(i, j, k):
    """Generates a Fredkin gate, or a controlled SWAP gate."""
    gate = cp.eye(8)
    gate[5:7, 5:7] = cp.array([[0, 1], [1, 0]])
    gate = gate.reshape((2, 2, 2, 2, 2, 2))
    return apply_gate(gate, i, j, k)

## Quantum circuits

def bell_state(i, j):
    """Generates an entangled Bell state on two qubits."""
    def b(register):
        return CNOT(i, j) (H(i) (register))
    return b


## MEASUREMENT

def measure(index):
    """Performs a partial measurement on a particular index.
    Returns a function that, when applied to a register,
    partially collapses the quantum state, and returns 0 or 1."""
    def m(register):
        n = register.ndim
        axis = tuple(range(index)) + tuple(range(index + 1, n))
        probs = cp.sum(cp.abs(register) ** 2, axis=axis)
        p = probs[0] / cp.sum(probs)
        s = [slice(0, 2)] * n
        result = int(cp.random.rand() > p)
        s[index] = slice(0,1) if result else slice(1, 2)
        register[tuple(s)] = 0
        register *= 1 / cp.linalg.norm(register)
        return result
    return m


def measure_all(collapse=True):
    """Returns a function that performs a full measurement on a quantum register,
    fully collapsing the quantum state, and returns a binary vector of length equal
    to the number of qubits in the register."""
    def m(register):
        r = register.ravel()
        r = r / cp.linalg.norm(r)
        index = cp.random.choice(range(len(r)), size=1, p=cp.abs(r)**2)
        index = index[0]
        multiindex = cp.unravel_index(index, register.shape)
        # multiindexをintのtupleに変換
        multiindex = tuple([int(i) for i in multiindex])
        if collapse:
            register.fill(0)
            register[multiindex] = 1.0
        return multiindex
    return m


# See https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
def max_multiindex(register):
    """Locates the entry of a tensor having the largest magnitude.
    Returns a binary vector of length equal to the number of qubits in the register."""
    # intのtupleにして返す
    a = cp.unravel_index(cp.argmax(cp.abs(register)), register.shape)
    ret = tuple([int(i) for i in a])
    return ret


# TESTING

def test_unary_gates():
    """Run tests on all unary gates."""
    register = random_register(2)
    x = X(0) (register)
    y = Y(0) (register)
    z = Z(0) (register)
    s = S(0) (register)
    t = T(0) (register)
    x1 = X(1) (register)
    sn = SNOT(1) (register)
    sn2 = SNOT(1) (sn)
    hxh = H(0) (X(0) (H(0) (register)))
    hyh = H(0) (Y(0) (H(0) (register)))
    hzh = H(0) (Z(0) (H(0) (register)))
    hh = H(0) (H(0) (register))
    ss = S(0) (S(0) (register))
    tt = T(0) (T(0) (register))
    uu = R(0, cp.pi/8) (R(0, cp.pi/8) (register))
    assert cp.allclose(hxh, z)
    assert cp.allclose(hyh, -y)
    assert cp.allclose(hh, register)
    assert cp.allclose(ss, z)
    assert cp.allclose(tt, s)
    assert cp.allclose(hxh, z)
    assert cp.allclose(hyh, -y)
    assert cp.allclose(hzh, x)
    assert cp.allclose(sn2, x1)
    assert cp.allclose(uu, t)


def test_binary_gates():
    """Run tests on all binary gates."""
    inp = random_register(3)
    out1 = CNOT(0, 1) (inp)
    out2 = CNOT(1, 0) (out1)
    out3 = CNOT(0, 1) (out2)
    swap = SWAP(0, 1) (inp)
    assert cp.allclose(out3, swap)
    not_gate = cp.array([[0, 1], [1, 0]])
    out4 = C(0, 1, not_gate) (inp)
    assert cp.allclose(out1, out4)


def test_ternary_gates():
    """Run tests on all ternary gates."""
    inp = random_register(4)
    assert cp.allclose(inp, CCNOT(0,3,2) (CCNOT(0, 3, 2) (inp)))
    assert cp.allclose(inp, CSWAP(0,3,2) (CSWAP(0, 3, 2) (inp)))


def test_bell_state():
    """Test the Bell state generator."""
    counts = Counter(
        measure_all() (bell_state(0,1) (quantum_register(3)))
        for _ in range(10000))
    assert counts[0, 0, 0] + counts[1, 1, 0] == 10000
    assert abs(counts[0, 0, 0] - counts[1, 1, 0]) < 500


def test_partial_measurement():
    """Run tests of the partial measurement function."""
    c = Counter()
    for _ in range(14000):
        register = cp.array([[1, -2j], [3, 0]]) / 14 ** 0.5
        measure(0) (register)
        measure(1) (register)
        c[max_multiindex(register)] += 1
    assert abs(c[0, 0] - 1000) < 150
    assert abs(c[0, 1] - 4000) < 250
    assert abs(c[1, 0] - 9000) < 250
    assert c[1, 1] == 0
    m1 = m2 = 0
    for _ in range(14000):
        register = cp.array([[1, -2j], [3, 0]]) / 14 ** 0.5
        m1 += measure(0) (register)
        m2 += measure(1) (register)
    assert abs(m1 - 9000) < 250
    assert abs(m2 - 4000) < 150


def test_full_measurement():
    register = cp.array([0.3, 0.4j, -0.5, 0.5 ** 0.5]).reshape((2, 2))
    m = measure_all(collapse=False)
    N = 100000
    expected = [N * 0.09, N * 0.16, N * 0.25, N * 0.5]
    counts = Counter(m(register) for _ in range(N))
    observed = [counts[0,0], counts[0,1], counts[1,0], counts[1,1]]
    assert all(abs(x - y) < 400 for x, y in zip(expected, observed))


def test_circuits():
    register = random_register(2)
    bell1 = bell_state(0, 1)
    bell2 = circuit([H(0), CNOT(0, 1)])
    assert cp.allclose(bell1(register), bell2(register))


def run_tests():
    """Run all tests."""
    test_unary_gates()
    print('Unary gates passed.')
    test_binary_gates()
    print('Binary gates passed.')
    test_ternary_gates()
    print('Ternary gates passed.')
    test_bell_state()
    print('Bell state passed.')
    test_partial_measurement()
    print('Partial measurement passed.')
    test_full_measurement()
    print('Full measurement passed.')
    test_circuits()
    print('Circuits passed.')


if __name__ == '__main__':
    # print('Running tests.')
    # run_tests()
    # print('Tests complete. No errors found.')

    import random
    # 時間計測も行う
    import time
    # 20qubitの量子レジスタを作成
    register = random_register(25)
    # 適当なゲートを作用させることを1000回繰り返す
    repeat = 1000
    start = time.time()
    for _ in range(repeat):
        # 量子レジスタに作用させるゲートを選択
        gate = random.choice([H(0), X(0), Y(0), Z(0), S(0), T(0), CNOT(0, 1), SWAP(0, 1), CCNOT(0, 1, 2), CSWAP(0, 1, 2)])
        # 量子レジスタにゲートを作用させる
        gate(register)
    end = time.time()
    print('average time: {}s'.format((end - start)/repeat))
    # 量子レジスタを測定
    m = measure_all(collapse=False)
    # 量子レジスタを表示
    print(m(register))
