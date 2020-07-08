# quantum-circuit
An implementation of a quantum circuit with an example simulation of Grover's search algorithm.

A quantum circuit consists of a network of interconnected gates. Each gate has a quantum-mechanical behaviour of applying a unitary transformation to the state
vector of qubits. Behaviour of the entire circuit is likewise a unitary transformation on the state vector. 

The data structure for quantum circuits consists of four classes:
* Circuit class represents a list of slices which are objects of the Slice class.
* Slice class represents the conectivity information of how the input wires are connected with the gates and output wires.
* State class represents qubits and coresponding complex values.
* Gate class represents each gate as a matrix.

We simulate the behaviour of quantum circuits computation via an iterative application of the gate matrices to the input vector.

We can use it to simulate quantum algorithms. Here we simulate Grover's algorithm for searching an unstructured database of unordered list. Grover's algorithm runs quadratically faster than the best possible classical algorithm for the same task. 

To run the simulation of the Grover's algorithm:
```bash
python3 quantum-circuit.py
```
