import cupy as cp
import time
import os
import sys
import json
from QuantumSimulator import *

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
    # test_binary_gates()
    # print('Binary gates passed.')
    # test_ternary_gates()
    # print('Ternary gates passed.')
    # test_bell_state()
    # print('Bell state passed.')
    # test_partial_measurement()
    # print('Partial measurement passed.')
    # test_full_measurement()
    # print('Full measurement passed.')
    # test_circuits()
    # print('Circuits passed.')

if __name__ == '__main__':
    os.makedirs('./outputs', exist_ok=True)
    repeat = 10
    start_qubit = 20
    end_qubit = 30
    x_list = []
    y_list = []
    for n in range(start_qubit, end_qubit+1):
        cp.cuda.Stream.null.synchronize()
        start_time = time.time()
        for _ in range(repeat):
            test_unary_gates()
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        print('qubit: {}, time: {}'.format(n, end_time - start_time))
        x_list.append(n)
        y_list.append((end_time - start_time) / repeat)
    with open('./outputs/{}.json'.format('test'), 'w') as f:
        json.dump({'x': x_list, 'y': y_list}, f)

