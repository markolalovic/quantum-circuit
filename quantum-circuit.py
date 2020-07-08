#!/usr/bin/python3
# quantum-circuit.py - An implementation of a quantum circuit.

'''
A quantum circuit consists of a network of interconnected gates. Each gate has a
quantum-mechanical behaviour of applying a unitary transformation to the state
vector of qubits. Behaviour of the entire circuit is likewise a unitary transformation
on the state vector.

The data structure for quantum circuits consists of four classes:
* Circuit class represents a list of slices which are objects of the Slice class.
* Slice class represents the conectivity information of how the input wires are
connected with the gates and output wires.
* State class represents qubits and coresponding complex values.
* Gate class represents each gate as a matrix.

We simulate the behaviour of quantum circuits computation via an iterative application
of the gate matrices to the input vector.

We can use it to simulate quantum algorithms. Here we simulate Grover's algorithm
for searching an unstructured database of unordered list. Grover's algorithm
runs quadratically faster than the best possible classical algorithm for the same task.
'''

from math import *
import itertools
import copy
import operator


## Implementation of a quantum circuit

class State:
    '''
    Represents qubits and coresponding complex values.
    '''
    objectName = "State object"
    def __init__(self, name, b):
        self.name = name
        self.n = len(b)       # number of qubits n is the length of b
        self.state_dict = {key : complex(0)  # initialize the state with all 0's
                           for key in list(itertools.product([0,1], repeat=self.n))}
        self.state_dict[b] = complex(1) # set to 1 where b is
    def __str__(self):
        for key in sorted(self.state_dict.keys()):
            print(key, self.state_dict[key])
        return self.name + "(" + self.objectName + ")" + "\n"
    def isCorrect(self):
        # check if the state is of the correct size
        if len(self.state_dict) != 2**self.n:
            print("The state must be of a size 2^n.")
            return False
        # and the keys are of the correct sizes
        elif sum([len(key) == self.n for key in self.state_dict]) != 2**self.n:
            print("Each key must be of size n.")
            return False
        # and if squares of values add up to 1
        elif (sum([squaremodulus(self.state_dict[key]) for key in self.state_dict]) > 1 + 0.0001) or \
                (sum([squaremodulus(self.state_dict[key]) for key in self.state_dict]) < 1 - 0.0001):
            print("The modulus of state values must add up to 1.")
            print(sum([squaremodulus(self.state_dict[key]) for key in self.state_dict]))
            return False
        else:
            return True

class Gate:
    '''
    Represents each gate as a matrix.
    '''
    objectName = "Gate object"
    def __init__(self, name, matrix):
        self.name = name
        self.matrix = matrix
        self.m = isqrt(len(self.matrix))  # number of inputs and outputs
    def __str__(self):
        print_matrix(self.matrix)
        return self.name + "(" + self.objectName + ")" + "\n \n"
    def isCorrect(self, eps):
        # check if the matrix is unitary to epsilon precision
        key_tuples_list = [key for key in list(itertools.product([0,1], repeat=len(bin(self.m-1))-2 ))]
        key_tuples_list = key_tuples_list[:self.m]

        # build the conjugate transpose
        conjugate_transpose = copy.deepcopy(self.matrix)
        for i in key_tuples_list:
            for j in key_tuples_list:
                conjugate_transpose[j,i] = self.matrix[i,j].conjugate()
        # this is inefficient, better to just read it differently
        print("The conjugate transpose: ")
        print_matrix(conjugate_transpose)

        # calculate the product of matrix and its conjugate transpose
        ct_times_m = multiply_matrices(conjugate_transpose, self.matrix)
        print("Conjugate transpose times matrix is: ")
        print_matrix(ct_times_m)

        # and the product of conjugate transpose and the matrix
        m_times_ct = multiply_matrices(self.matrix, conjugate_transpose)
        print("Conjugate transpose times matrix is: ")
        print_matrix(m_times_ct)

        # check if the products are identity matrices to the epsilon precision
        for i in key_tuples_list:
            for j in key_tuples_list:
                a = ct_times_m[i,j]
                b = m_times_ct[i,j]
                if i == j: # on the diagonal
                    if a.real > 1 + eps or a.real < 1 - eps or b.real > 1 + eps or b.real < 1 - eps:
                        print("Real part on the diagonal must be close to 1.")
                        return False
                    elif a.imag > eps or a.imag < -eps or b.imag > eps or b.imag < -eps:
                        print("Imaginary part on the diagonal must be close to 0.")
                        return False
                else: # off the diagonal
                    if a.real > eps or a.real < -eps or b.real > eps or b.real < -eps:
                        print("Real part off the diagonal must be close to 0.")
                        return False
                    elif a.imag > eps or a.imag < -eps or b.imag > eps or b.imag < -eps:
                        print("Imaginary part off the diagonal must be close to 0.")
                        return False
        print(self.name + " is a unitary matrix.")
        return True

class Slice:
    '''
    Represents the conectivity information of how the input wires are connected
    with the gates and output wires.
    '''
    objectName = "Slice object"
    def __init__ (self, name, n, m, slice_input, gate_output, gate):
        self.name = name                # slice name
        self.n = n 					    # number of slice wires
        self.m = m					    # number of gate wires
        self.slice_input = slice_input  # slice wires connection information
        self.gate_output = gate_output  # gate wires connection information
        self.gate = gate                # set the gate first to None
    def __str__(self):
        print(self.name)
        print(self.slice_input)
        print(self.gate_output)
        return self.name + "(" + self.objectName + ")" + "\n"
    def isCorrect(self):
        inputs = []
        outputs = []
        gate_inputs = []
        gate_outputs = []
        for i in self.slice_input:
            inputs.append(i)    # get all input wire info
            if self.slice_input[i][0] == 'to_output':
                outputs.append(self.slice_input[i][1])  # get all wire that don't go trough the gate
            elif self.slice_input[i][0] == 'to_gate':
                gate_inputs.append(self.slice_input[i][1])  # get all wire that go trough the gate
            else:
                print("Wire from the slice input can only connect to the output or to the gate.")
                return False
        for i in self.gate_output:
            gate_outputs.append(i)  # get all gate output wire info
            if self.gate_output[i][0] == 'to_output':
                outputs.append(self.gate_output[i][1]) # get all wire that comes out of the gate
            else:
                print("Wire from gate output can only connect to the output.")
                return False
        # first check if content of all lists is unique
        allputs = [inputs, outputs, gate_inputs, gate_outputs]
        for l in allputs:
            if len(l) > len(set(l)):
                print("There is a list in lists with repeated elements.")
                return False
        # then check if lists of in's and out's of slices and gates are equal sets and of the correct sizes
        if set(inputs) != set(outputs):
            print("There must be as many slice inputs as slice outputs.")
            return False
        elif set(inputs) != set(range(self.n)):
            print("Each slice wire must have a connection information.")
            return False
        elif set(gate_inputs) != set(gate_outputs):
            print("There must be as many gate inputs as gate outputs.")
            return False
        elif set(gate_inputs) != set(range(self.m)):
            print("Each gate wire must have a connection information.")
            return False
        else:
            return True

    def propagate_input(self, b):
        # returns the tuple which is the input to the gate
        b_in_list = [0]*self.m  # prepare list of zeros that we will transform into output tuple
        for i in self.slice_input:
            if self.slice_input[i][0] == 'to_gate':
                b_in_list[self.slice_input[i][1]] = b[i]
        return tuple(b_in_list)
    def propagate_output(self, b, c):
        # returns the tuple which is the output from the slice
        b_prime = [0]*self.n  # prepare list of zeros that we will transform into output tuple
        for i in self.slice_input:
            if self.slice_input[i][0] == 'to_output':
                b_prime[self.slice_input[i][1]] = b[i]
        for i in self.gate_output:
            b_prime[self.gate_output[i][1]] = c[i]
        return tuple(b_prime)


    def evaluate(self, fi):
        # assume input fi is a State object with vector
        # (dictionary state_dict) of complex values (amplitudes) of size 2^n
        # and set output fi_prime to be State object with vector (dictionary state_dict) of size 2^n
        fi_prime = State("fi_prime",(0,)*fi.n)
        fi_prime.state_dict[(0,)*fi.n] = 0 + 0j  # must set all to 0

        # there are 2^n basis vectors in Fi where n is the number of slice wires
        fi_vectors = [key for key in list(itertools.product([0,1], repeat=len(bin(2**self.n-1))-2 ))]
        # and 2^m basis vectors for the gate where m is the number of gate wires
        gate_vectors = [key for key in list(itertools.product([0,1], repeat=len(bin(2**self.m-1))-2 ))]

        for b in fi_vectors:
            b_in = self.propagate_input(b)  # b_in is the gate input vector
            for b_out in gate_vectors:
                b_prime = self.propagate_output(b,b_out)
                fi_prime.state_dict[b_prime] += fi.state_dict[b] * self.gate.matrix[b_out, b_in]
        return fi_prime

class Circuit:
    '''
    Represents a list of slices which are objects of the Slice class.
    '''
    objectName = "Circuit object"
    def __init__ (self, name, n, slice_list):
        self.name = name
        self.n = n 					  # number of qubits
        self.slice_list = slice_list  # list of Slice objects
    def __str__(self):
        return self.name + "(" + self.objectName + ")" + "\n"
    def isCorrect(self):
        # for each Slice object in slice_list we check if isCorrect and all are the same size
        sizes = []  #
        for s in self.slice_list:
            sizes.append(s.n)
            if not s.isCorrect():
                print(s + " is not correct.")
                return False
        if all(x == sizes[0] for x in sizes) and sizes[0] == self.n:
            return True
        else:
            print("The size n of slices must match.")
            return False
    def evaluate(self, input_fi):
        output_fi = input_fi
        for slice in self.slice_list:
            output_fi = slice.evaluate(output_fi)
        return output_fi


## Auxiliary functions

def modulus(z):
    return sqrt(z.real**2 + z.imag**2)

def squaremodulus(z):
    return float(z.real**2 + z.imag**2)

def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def print_matrix(matrix):
    m = isqrt(len(matrix))
    if m**2 != len(matrix):
        print("Matrix has to be square.")
        return False
    # generate a list of key tuples of the right size: length( binary (m - 1) )
    key_tuples_list = [key for key in list(itertools.product([0,1], repeat=len(bin(m-1))-2 ))]

    # cut it to the size of m because this is the number of rows we need to access
    key_tuples_list = key_tuples_list[:m]
    for i in key_tuples_list:
        for j in key_tuples_list:
            print(matrix[i, j])
            print("\n")

def multiply_matrices(a,b):
    # assume square matrices of equal size
    m = isqrt(len(a))
    if m**2 != len(a):
        print("Matrix has to be square.")
        return False
    key_tuples_list = [key for key in list(itertools.product([0,1], repeat=len(bin(m-1))-2 ))]
    key_tuples_list = key_tuples_list[:m]
    ab = copy.deepcopy(a)
    for i in key_tuples_list:
        for j in key_tuples_list:
            ab[i,j] = sum([a[i,k]*b[k,j] for k in key_tuples_list])
    return ab

def build_hadamard_matrix():
    return {((0,),(0,)): 1/sqrt(2), ((0,),(1,)): 1/sqrt(2),
                ((1,),(0,)): 1/sqrt(2), ((1,),(1,)): -1/sqrt(2)}

def build_zero_matrix(m):
    # m is the size of matrix
    ktl = [key for key in list(itertools.product([0,1], repeat=m))]
    return {key : 0 for key in list(itertools.product(ktl, repeat=2))}

def build_identity_matrix(m):
    # m is the size of matrix
    ktl = [key for key in list(itertools.product([0,1], repeat=m))]
    return {key : int(key[0] == key[1]) for key in list(itertools.product(ktl, repeat = 2))}

def build_flipx_matrix(x0):
    # assume x0 is represented as a list; example: x = [0,1]
    # x0 defines f(x): f(x) = 1 if x = x0 else f(x) = 0
    flipx_matrix = build_identity_matrix(len(x0) + 1)
    flipx_matrix[tuple(x0) + (1,), tuple(x0) + (0,)] = 1
    flipx_matrix[tuple(x0) + (0,), tuple(x0) + (1,)] = 1
    flipx_matrix[tuple(x0) + (0,), tuple(x0) + (0,)] = 0
    flipx_matrix[tuple(x0) + (1,), tuple(x0) + (1,)] = 0
    return flipx_matrix

def build_flip_non_zero_matrix(x0):
    # assume x0 is represented as a list; example: x = [0,1]
    # x0 defines f(x): f(x) = 1 if x = x0 else f(x) = 0
    m = len(x0) + 1
    flip_non_zero_matrix = build_zero_matrix(m)
    ktl = [key for key in list(itertools.product([0,1], repeat=m))]
    for i in range(0,len(ktl),2):
        flip_non_zero_matrix[ktl[i], ktl[i+1]] = 1
        flip_non_zero_matrix[ktl[i+1], ktl[i]] = 1

    # change the zero part of the matrix to identity
    # because we don't want to flip the zero part of the matrix
    z0 = [0]*len(x0)
    flip_non_zero_matrix[tuple(z0) + (0,), tuple(z0) + (0,)] = 1
    flip_non_zero_matrix[tuple(z0) + (0,), tuple(z0) + (1,)] = 0
    flip_non_zero_matrix[tuple(z0) + (1,), tuple(z0) + (0,)] = 0
    flip_non_zero_matrix[tuple(z0) + (1,), tuple(z0) + (1,)] = 1
    return flip_non_zero_matrix

def build_hadamard_slice(n,h):
    # n is the number of qubits of the circuit
    # h indicates on which quibit we apply the hadamard gate
    slice_input = {}
    gate_output = {}
    for i in range(n):
        if i == h:
            slice_input[i] = ('to_gate', 0)
        else:
            slice_input[i] = ('to_output', i)
    gate_output[0] = ('to_output', h)
    hadamard_gate = Gate("hadamard_gate", build_hadamard_matrix())
    return Slice("hadamard slice " + str(h), n, 1, slice_input, gate_output, hadamard_gate)

def build_flipx_slice(n,x0):
    # n is the number of qubits of the circuit
    # x0 defines f(x): f(x) = 1 if x = x0 else f(x) = 0
    slice_input = {}
    gate_output = {}
    for i in range(n):
        slice_input[i] = ('to_gate', i)
        gate_output[i] = ('to_output', i)
    flipx_gate = Gate("flipx_gate", build_flipx_matrix(x0))
    return Slice("flipx slice", n, n, slice_input, gate_output, flipx_gate)

def build_flip_non_zero_slice(n,x0):
    # n is the number of qubits of the circuit
    # x0 defines f(x): f(x) = 1 if x = x0 else f(x) = 0
    slice_input = {}
    gate_output = {}
    for i in range(n):
        slice_input[i] = ('to_gate', i)
        gate_output[i] = ('to_output', i)
    flip_non_zero_gate = Gate("flip_non_zero_gate", build_flip_non_zero_matrix(x0))
    return Slice("flip_non_zero slice", n, n, slice_input, gate_output, flip_non_zero_gate)


## Grover's algorithm

def grovers_algorithm(x):
    x = list(x)
    N = 2**len(x)    # the number of classical states is the size of the problem
    n = len(x) + 1   # n is the number of qubits, add 1 because we need to add result qubit to the input state

    input_fi = State("input_fi", (0,)*len(x) + (1,)) # (0...01) <-- 1 else 0

    print(input_fi)

    # first build n hadamard slices to put input_fi into a superposition state
    super_slice_list = []
    for i in range(n):
        super_slice_list.append(build_hadamard_slice(n, i))

    super_circuit = Circuit("hadamards", n, super_slice_list)
    super_fi = super_circuit.evaluate(input_fi)

    print(super_fi)

    # build the loop part of the circuit
    output_fi = super_fi
    k_max = int(floor(0.5 + (pi * sqrt(N))/4))
    print("Number of iterations k_max is: " + str((pi * sqrt(N))/4) + "\n")
    for k in range(k_max + 5):
        loop_slice_list = []
        loop_slice_list.append(build_flipx_slice(n,x))
        for i in range(n - 1):
            loop_slice_list.append(build_hadamard_slice(n, i))
        loop_slice_list.append(build_flip_non_zero_slice(n, x))
        for i in range(n - 1):
            loop_slice_list.append(build_hadamard_slice(n, i))

        loop_circuit = Circuit("loop part", n, loop_slice_list)
        output_fi = loop_circuit.evaluate(output_fi)

        # measure the result
        print(str(k+1) + " iterations:")
        result = State("output probabilities", (n-1)*(0,))
        keys = [key for key in list(itertools.product([0,1], repeat=n))]
        result_keys = [key for key in list(itertools.product([0,1], repeat=(n - 1)))]
        for i in range(len(keys)-1, -1, -2):
            result.state_dict[result_keys[i//2]] = squaremodulus(output_fi.state_dict[keys[i]]) \
                                                   + squaremodulus(output_fi.state_dict[keys[i - 1]])
        print(result)

    # output the classical state with maximum probability
    max_key = max(result.state_dict.items(), key=operator.itemgetter(1))[0]
    max_val = result.state_dict[max_key]

    output_list = []
    for key in result.state_dict:
        if fabs(result.state_dict[key] - max_val) < 0.01:
            output_list.append(key)
    return output_list  # return all the classical states with maximum probability


def main():
    x = (0,1,0)          # x defines the function
    grovers_algorithm(x)

main()
