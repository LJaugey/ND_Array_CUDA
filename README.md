# N-D Array CUDA version

This is a work in progress to add a CUDA backend to the ND::Array class.

This project contains a header file that defines a N-dimensional array (variadic) template. This definition takes advantage of data locality which improves performance over e.g. nested `std::vectors`.
Mathematical expression are also optimized using lazy evaluation which is implemented with expression templates. Expression such as `(A+B)*C` is compiled to a single `for` loop containing `(A[i]+B[i])*C[i]`, which makes it as efficient as hand-written C code. ~~This also works when evaluating specific array element since only the specified element is effectively computed (e.g. `(A+B)(i,j,k)` simply compiles to `(A(i,j,k)+B(i,j,k))`).~~ This is not the case with the CUDA backend. Since data is on GPU, data should not be accessed directly.

The file `speed_test.cpp/cu` is a demonstration of the speed difference between this template and nested `std::valarray` (which also uses lazy evaluation) as well as other interesting performance metrics. `test.cpp/cu` contains tests but can also serve as a comprehensive list of features.

**Note:** Must be compiled with c++20

## Usage

### Definition
An array of `double` with shape `[Dim_1,Dim_2,Dim_3]` is defined with
```
Array<Backend::CUDA, double, Dim_1,Dim_2,Dim_3> A;
```
By default, the values are not initialized. The array can be set to a specific `value` with
```
A = value;
```
This can be combined:
```
Array<Backend::CUDA, double, Dim_1,Dim_2,Dim_3> B = value;
// OR
Array<Backend::CUDA, double, Dim_1,Dim_2,Dim_3> C(value);
```
Arrays can be declared with other sub-arrays if dimensions match. In the example below, `B` contains `Dim_1` copies of `A`.
```
Array<Backend::CUDA, double, Dim_2,Dim_3> A = 42;
Array<Backend::CUDA, double, Dim_1,Dim_2,Dim_3> B(A);
```
Finally, they can be created from expressions:
```
Array<Backend::CUDA, double, Dim_1,Dim_2,Dim_3> B = 2*A - 1;
```
### Operations
The usual mathematical, boolean and bitwise operations are implemented (in parallel if OpenMP is enabled) and use lazy evaluation. By default, all operations are performed element-wise, except in obvious situation such as for min/max or sum which all return a single value.
The type of the resulting array is automatically determined based on the operation and its operands.
For binary operation (e.g. `+` or `<=`), one of the operand can be a single value (e.g. a scalar) that is convertible to an element of the array.
```
C = 2*A + pow(A,B) - 2;     // Element-wise pow(A,B): a^b
D = A<2;                    // Would evaluate to a boolean array
E = floor(1.5*A);           // Would evaluate to an int array
```
**Note:** Using the `auto` keyword does not perform the computation and keeps an expression. To force the evaluation, use `.eval()`
```
auto C = A+B;           // Array expression
auto D = A+B.eval();    // Array
```
### Access
There are 3 ways to access array elements

~~`Array::operator()` with the right amount of indices. This returns a reference to the corresponding element.~~
```
A(4,3,2) = 5; // NOT valid with CUDA backend
std::cout<<A(4,3,2)<<std::endl;
```
```
5
```
`operator[]` with 1 index. This returns a slice of the original array at the specified index. This is still valid with CUDA as long as the array's dimension is 2 or more. If the dimension is 1, the data pointer is dereference and will crash.
```
A[2] = 5;
std::cout<<A[2]<<std::endl;
```
```
5 5 5
5 5 5
5 5 5
```
Mask indexing. This can be used to accessed every elements respecting a specific condition:
```
A[B<2] = 42;
```

### Other
The size of each dimension can be accessed with `Array::size(dim)`, where `dim` is the dimension (starts at 0). The default value for `dim` is `0` so `A.size()` returns the first dimension.

`.all()` can be used on boolean arrays to check if all elements are true.


### Compilation
As mentioned before, this requires c++20.

To get a useful speed-up from lazy evaluation, a reasonable optimization level is required (at least -O1 on gcc). The makefile uses more aggresive optimizations (`-Ofast -march=native`) which could cause some issues. Consider reducing `-Ofast` to `-O2` (or `-O3`) if you encounter issues. Some tests may fail with `-Ofast` but should run fine with `-O2` or even `-O3`. This comes from the flag `-ffast-math` and more precisely `-funsafe-math-optimizations` on gcc.

Operations are parallelised with OpenMP using multiple cores as well as simd operations.
Note that simd operations will probably be automatically performed even without OpenMP if you enable a high optimization level (at least `-O3` on gcc).
