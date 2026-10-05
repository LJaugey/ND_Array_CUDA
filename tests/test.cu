#include <cstddef>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../ND_Array/ND_Array.hpp"

using doctest::Approx;

const size_t dim_1 = 2;
const size_t dim_2 = 3;



TEST_CASE("Constructors") {

    const double a = 0.0;
    ND::Array<ComputeBackend::CUDA, double, dim_2> A_sub(a);


    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A1(a);
    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A2(0.0);
    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A3(0);
    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A4(A_sub);


    CHECK((A1==A2).all());
    CHECK((A1==A3).all());
    CHECK((A1==A4).all());
    
    CHECK((A2==A3).all());
    CHECK((A2==A4).all());
    
    CHECK((A3==A4).all());
    

    
    //copy constructor
    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> B(A1);
    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> B1;
    B1 = A1;

    CHECK((B==A1).all());
    CHECK((B1==A1).all());


    //from array expression
    const double b = 2.0;

    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> C(A1+b);
    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> C1;
    C1 = A1 + b;

    CHECK((C==(A1+b).eval()).all());
    CHECK((C1==(A1+b).eval()).all());
}


TEST_CASE("Comparison operations") {

    const double a = 1.0;

    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A(a);
    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A_2(2*a);

    CHECK((A==A).all());
    CHECK((A_2==A+A).all());
    CHECK((A+A==A_2).all());
    CHECK((A+A==A+A).all());

    CHECK((A!=A+A).all());
    CHECK((A+A!=A).all());
    CHECK_FALSE((A==A+A).all());

    CHECK((A<A+A).all());
    CHECK((2*A>A).all());
    
    CHECK((A<=A+A).all());
    CHECK((2*A>=A).all());
    
    CHECK((2*A<=A+A).all());
    CHECK((A+A>=2*A).all());
}

TEST_CASE("Access operators") {

    const double a = 2.0;
    const double b = 1.0;

    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A(a);
    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> B(b);
    ND::Array<ComputeBackend::CUDA, double,dim_2> A_s(a);
    ND::Array<ComputeBackend::CUDA, double,dim_2> B_s(b);

    CHECK((A==A).all());
    CHECK((A[0]==A_s).all());
    CHECK((A[dim_1-1]==A_s).all());
/*
    CHECK_EQ(A[0][0],a);

    CHECK_EQ(A(0,0),a);
    CHECK_EQ(A(dim_1-1,dim_2-1),a);

    A(0,0) = b;
    CHECK_EQ(A(0,0),b);
    CHECK_EQ(A(dim_1-1,dim_2-1),a);
    A(0,0) = a;

    CHECK_EQ((A+B)(0,0),a+b);
    CHECK_EQ((A+b)(0,0),a+b);
    CHECK_EQ((A+b)(dim_1-1,dim_2-1),a+b);
*/
    CHECK(((A+B)[0]==A_s+B_s).all());
/*
    CHECK_EQ((A+b)[0][0],a+b);
    CHECK_EQ((A+b)[dim_1-1][dim_2-1],a+b);
*/
    A[0] = b;
    CHECK((A[0]==B_s).all());
    CHECK_FALSE((A[dim_1-1]==B_s).all());
    CHECK((A[dim_1-1]==A_s).all());
/*
    CHECK_EQ(A[0][0],b);

    CHECK_EQ(A(0,0),b);
    CHECK_EQ(A(dim_1-1,dim_2-1),a);
*/

    // Mask indexing
    double c = (a+b)/2.0;
    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> C;
    C[0] = a;   C[1] = b;

    A[C<a] = c;
    CHECK((A[1]==c).all());

    A[A==c] = a;
    CHECK((A[1]==a).all());

    A = a;
    A[C<a] += c;
    CHECK((A[1]==a+c).all());
    A = a;
    A[C<a] -= c;
    CHECK((A[1]==a-c).all());
    A = a;
    A[C<a] *= c;
    CHECK((A[1]==a*c).all());
    A = a;
    A[C<a] /= c;
    CHECK((A[1]==a/c).all());
}


TEST_CASE("Arithmetic operations") {

    const double a = 1.0;
    const double b = 0.2;
    const double c = -0.7;

    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A(a);
    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> B(b);
    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> C(c);



    SUBCASE("Addition") {
        ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A_B(a+b);
        ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A_B_C(a+b+c);

        CHECK((A+B == A_B).all());
        CHECK((A+b == A_B).all());
        CHECK((a+B == A_B).all());

        CHECK((A+B+C == A_B_C).all());
        CHECK((A+B+c == A_B_C).all());
        CHECK((A+b+C == A_B_C).all());
        CHECK((a+B+C == A_B_C).all());

        CHECK(((A+=B+C) == (a+(B+C))).all());
        CHECK(((B+=c) == (b+C)).all());

    }
    SUBCASE("Subtraction") {
        ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A_B(a-b);
        ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A_B_C(a-b-c);

        CHECK((A-B == A_B).all());
        CHECK((A-b == A_B).all());
        CHECK((a-B == A_B).all());

        CHECK((A-B-C == A_B_C).all());
        CHECK((A-B-c == A_B_C).all());
        CHECK((A-b-C == A_B_C).all());
        CHECK((a-B-C == A_B_C).all());

        CHECK(((A-=B-C) == (a-(B-C))).all());
        CHECK(((B-=c) == (b-C)).all());
    }
    SUBCASE("Multiplication") {
        ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A_B(a*b);
        ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A_B_C(a*b*c);

        CHECK((A*B == A_B).all());
        CHECK((A*b == A_B).all());
        CHECK((a*B == A_B).all());

        CHECK((A*B*C == A_B_C).all());
        CHECK((A*B*c == A_B_C).all());
        CHECK((A*b*C == A_B_C).all());
        CHECK((a*B*C == A_B_C).all());

        CHECK(((A*=B*C) == (a*(B*C))).all());
        CHECK(((B*=c) == (b*C)).all());
    }
    SUBCASE("Division") {
        ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A_B(a/b);
        ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> A_B_C(a/b/c);

        CHECK((A/B == A_B).all());
        CHECK((A/b == A_B).all());
        CHECK((a/B == A_B).all());

        CHECK((A/B/C == A_B_C).all());
        CHECK((A/B/c == A_B_C).all());
        CHECK((A/b/C == A_B_C).all());
        CHECK((a/B/C == A_B_C).all());

        CHECK(((A/=B/C) == (a/(B/C))).all());
        CHECK(((B/=c) == (b/C)).all());
    }
}

TEST_CASE("Trigonometric/Hyperbolic operations") {

    const double a = 0.4;
    const double b = -0.7;
    const double c = 1.7;

    using testArray = ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2>;
    testArray A(a);
    testArray B(b);
    testArray C(c);

    double tol = 1e-12;
    
    SUBCASE("sin") {

        CHECK((abs(sin(A) - testArray(sin(a))) < tol).all());
        CHECK((abs(sin(B) - testArray(sin(b))) < tol).all());

        CHECK((abs(asin(A) - testArray(asin(a))) < tol).all());
        CHECK((abs(asin(B) - testArray(asin(b))) < tol).all());

        CHECK((abs(sinh(A) - testArray(sinh(a))) < tol).all());
        CHECK((abs(sinh(B) - testArray(sinh(b))) < tol).all());

        CHECK((abs(asinh(A) - testArray(asinh(a))) < tol).all());
        CHECK((abs(asinh(B) - testArray(asinh(b))) < tol).all());
    }
    
    SUBCASE("cos") {

        CHECK((abs(cos(A) - testArray(cos(a))) < tol).all());
        CHECK((abs(cos(B) - testArray(cos(b))) < tol).all());

        CHECK((abs(acos(A) - testArray(acos(a))) < tol).all());
        CHECK((abs(acos(B) - testArray(acos(b))) < tol).all());

        CHECK((abs(cosh(A) - testArray(cosh(a))) < tol).all());
        CHECK((abs(cosh(B) - testArray(cosh(b))) < tol).all());

        CHECK((abs(acosh(C) - testArray(acosh(c))) < tol).all());      // for acosh, arg>=1 is required
    }
    
    SUBCASE("tan") {

        CHECK((abs(tan(A) - testArray(tan(a))) < tol).all());
        CHECK((abs(tan(B) - testArray(tan(b))) < tol).all());

        CHECK((abs(atan(A) - testArray(atan(a))) < tol).all());
        CHECK((abs(atan(B) - testArray(atan(b))) < tol).all());

        CHECK((abs(tanh(A) - testArray(tanh(a))) < tol).all());
        CHECK((abs(tanh(B) - testArray(tanh(b))) < tol).all());

        CHECK((abs(atanh(A) - testArray(atanh(a))) < tol).all());
        CHECK((abs(atanh(B) - testArray(atanh(b))) < tol).all());
    }

    
    SUBCASE("tan2") {
        
        CHECK((abs(atan2(A,B) - testArray(atan2(a,b))) < tol).all());
        CHECK((abs(atan2(B,A) - testArray(atan2(b,a))) < tol).all());
        
        CHECK((abs(atan2(A,b) - testArray(atan2(a,b))) < tol).all());
        CHECK((abs(atan2(a,B) - testArray(atan2(a,b))) < tol).all());
    }
}



TEST_CASE("bitwise operations") {

    const int a = 5;
    const int b = 2;

    using testArray = ND::Array<ComputeBackend::CUDA, int,dim_1, dim_2>;
    testArray A(a);
    testArray B(b);


    CHECK(((A&B) == testArray(a&b)).all());
    CHECK(((A|B) == testArray(a|b)).all());
    CHECK(((A^B) == testArray(a^b)).all());
    CHECK(((A<<B) == testArray(a<<b)).all());
    CHECK(((A>>B) == testArray(a>>b)).all());
}


    

TEST_CASE("Other") {

    const double a = 1.7;
    const double a_ = 1.2;

    using testArray = ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2>;
    testArray A(a);
    testArray A_(a_);

    double tol = 1e-12;

    CHECK_EQ(A.size(0), dim_1);
    CHECK_EQ(A.size(1), dim_2);


    CHECK(((-A) ==  testArray(-a)).all());
    CHECK((abs(A) ==  testArray(std::abs(a))).all());
    CHECK((abs(exp(A) - testArray(std::exp(a))) < tol).all());
    CHECK((abs(log(A) - testArray(std::log(a))) < tol).all());
    CHECK((abs(log2(A) - testArray(std::log2(a))) < tol).all());
    CHECK((abs(log10(A) - testArray(std::log10(a))) < tol).all());
    CHECK((abs(sqrt(A) - testArray(std::sqrt(a))) < tol).all());

    CHECK((abs(pow(A,A_) - testArray(std::pow(a,a_))) < tol).all());
    CHECK((abs(pow(A,a_) - testArray(std::pow(a,a_))) < tol).all());
    CHECK((abs(pow(a_,A) - testArray(std::pow(a_,a))) < tol).all());
    
    CHECK((round(A) == testArray(round(a))).all());
    CHECK((ceil(A) == testArray(ceil(a))).all());
    CHECK((floor(A) == testArray(floor(a))).all());


    ND::Array<ComputeBackend::CUDA, double,dim_1, dim_2> B;
    const double x = 1.0;
    const double y = 2.0;

    B[0] = x; B[1] = y;

    CHECK_EQ(B.min(), std::min(x,y));
    CHECK_EQ(min(B), std::min(x,y));

    CHECK_EQ(B.max(), std::max(x,y));
    CHECK_EQ(max(B), std::max(x,y));

    CHECK_EQ(B.sum(), (x+y)*dim_2);
    CHECK_EQ(sum(B), (x+y)*dim_2);


    int c = 5; int d = 2;
    ND::Array<ComputeBackend::CUDA, int, dim_1,dim_2> C(c);
    ND::Array<ComputeBackend::CUDA, int, dim_1,dim_2> D(d);

    //CHECK_EQ((C%D)(0,0), c%d);
}