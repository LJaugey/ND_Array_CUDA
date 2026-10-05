#include <cstddef>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../ND_Array/ND_Array.hpp"

using doctest::Approx;

const size_t dim_1 = 2;
const size_t dim_2 = 3;



TEST_CASE("Constructors") {

    const double a = 0.0;
    ND::Array_CPU< double, dim_2> A_sub(a);


    ND::Array_CPU< double,dim_1, dim_2> A1(a);
    ND::Array_CPU< double,dim_1, dim_2> A2(0.0);
    ND::Array_CPU< double,dim_1, dim_2> A3(0);
    ND::Array_CPU< double,dim_1, dim_2> A4(A_sub);


    CHECK((A1==A2).all());
    CHECK((A1==A3).all());
    CHECK((A1==A4).all());
    
    CHECK((A2==A3).all());
    CHECK((A2==A4).all());
    
    CHECK((A3==A4).all());
    

    
    //copy constructor
    ND::Array_CPU< double,dim_1, dim_2> B(A1);
    ND::Array_CPU< double,dim_1, dim_2> B1;
    B1 = A1;

    CHECK((B==A1).all());
    CHECK((B1==A1).all());


    //from array expression
    const double b = 2.0;

    ND::Array_CPU< double,dim_1, dim_2> C(A1+b);
    ND::Array_CPU< double,dim_1, dim_2> C1;
    C1 = A1 + b;

    CHECK((C==(A1+b).eval()).all());
    CHECK((C1==(A1+b).eval()).all());
}


TEST_CASE("Comparison operations") {

    const double a = 1.0;

    ND::Array_CPU< double,dim_1, dim_2> A(a);
    ND::Array_CPU< double,dim_1, dim_2> A_2(2*a);

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

    ND::Array_CPU< double,dim_1, dim_2> A(a);
    ND::Array_CPU< double,dim_1, dim_2> B(b);
    ND::Array_CPU< double,dim_2> A_s(a);
    ND::Array_CPU< double,dim_2> B_s(b);

    CHECK((A==A).all());
    CHECK((A[0]==A_s).all());
    CHECK((A[dim_1-1]==A_s).all());
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
    CHECK(((A+B)[0]==A_s+B_s).all());
    CHECK_EQ((A+b)[0][0],a+b);
    CHECK_EQ((A+b)[dim_1-1][dim_2-1],a+b);
    
    A[0] = b;
    CHECK((A[0]==B_s).all());
    CHECK_FALSE((A[dim_1-1]==B_s).all());
    CHECK((A[dim_1-1]==A_s).all());
    CHECK_EQ(A[0][0],b);

    CHECK_EQ(A(0,0),b);
    CHECK_EQ(A(dim_1-1,dim_2-1),a);


    // Mask indexing
    double c = (a+b)/2.0;
    ND::Array_CPU< double,dim_1, dim_2> C;
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

    ND::Array_CPU< double,dim_1, dim_2> A(a);
    ND::Array_CPU< double,dim_1, dim_2> B(b);
    ND::Array_CPU< double,dim_1, dim_2> C(c);



    SUBCASE("Addition") {
        ND::Array_CPU< double,dim_1, dim_2> A_B(a+b);
        ND::Array_CPU< double,dim_1, dim_2> A_B_C(a+b+c);

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
        ND::Array_CPU< double,dim_1, dim_2> A_B(a-b);
        ND::Array_CPU< double,dim_1, dim_2> A_B_C(a-b-c);

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
        ND::Array_CPU< double,dim_1, dim_2> A_B(a*b);
        ND::Array_CPU< double,dim_1, dim_2> A_B_C(a*b*c);

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
        ND::Array_CPU< double,dim_1, dim_2> A_B(a/b);
        ND::Array_CPU< double,dim_1, dim_2> A_B_C(a/b/c);

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

    ND::Array_CPU< double,dim_1, dim_2> A(a);
    ND::Array_CPU< double,dim_1, dim_2> B(b);
    ND::Array_CPU< double,dim_1, dim_2> C(c);


    
    SUBCASE("sin") {

        CHECK_EQ(sin(A)(0,0), Approx(sin(a)));
        CHECK_EQ(sin(B)(0,0), Approx(sin(b)));

        CHECK_EQ(asin(A)(0,0), Approx(asin(a)));
        CHECK_EQ(asin(B)(0,0), Approx(asin(b)));

        CHECK_EQ(sinh(A)(0,0), Approx(sinh(a)));
        CHECK_EQ(sinh(B)(0,0), Approx(sinh(b)));

        CHECK_EQ(asinh(A)(0,0), Approx(asinh(a)));
        CHECK_EQ(asinh(B)(0,0), Approx(asinh(b)));
    }
    
    SUBCASE("cos") {

        CHECK_EQ(cos(A)(0,0), Approx(cos(a)));
        CHECK_EQ(cos(B)(0,0), Approx(cos(b)));

        CHECK_EQ(acos(A)(0,0), Approx(acos(a)));
        CHECK_EQ(acos(B)(0,0), Approx(acos(b)));

        CHECK_EQ(cosh(A)(0,0), Approx(cosh(a)));
        CHECK_EQ(cosh(B)(0,0), Approx(cosh(b)));

        CHECK_EQ(acosh(C)(0,0), Approx(acosh(c)));      // for acosh, arg>=1 is required
    }
    
    SUBCASE("tan") {

        CHECK_EQ(tan(A)(0,0), Approx(tan(a)));
        CHECK_EQ(tan(B)(0,0), Approx(tan(b)));

        CHECK_EQ(atan(A)(0,0), Approx(atan(a)));
        CHECK_EQ(atan(B)(0,0), Approx(atan(b)));

        CHECK_EQ(tanh(A)(0,0), Approx(tanh(a)));
        CHECK_EQ(tanh(B)(0,0), Approx(tanh(b)));

        CHECK_EQ(atanh(A)(0,0), Approx(atanh(a)));
        CHECK_EQ(atanh(B)(0,0), Approx(atanh(b)));
    }

    
    SUBCASE("tan2") {
        
        CHECK_EQ(atan2(A,B)(0,0), Approx(atan2(a,b)));
        CHECK_EQ(atan2(B,A)(0,0), Approx(atan2(b,a)));
        
        CHECK_EQ(atan2(A,b)(0,0), Approx(atan2(a,b)));
        CHECK_EQ(atan2(a,B)(0,0), Approx(atan2(a,b)));
    }
}



TEST_CASE("bitwise operations") {

    const int a = 5;
    const int b = 2;

    ND::Array_CPU< int,dim_1, dim_2> A(a);
    ND::Array_CPU< int,dim_1, dim_2> B(b);


    CHECK_EQ((A&B)(0,0), a&b);
    CHECK_EQ((A|B)(0,0), a|b);
    CHECK_EQ((A^B)(0,0), a^b);
    CHECK_EQ((A<<B)(0,0), a<<b);
    CHECK_EQ((A>>B)(0,0), a>>b);
}

    

TEST_CASE("Other") {

    const double a = 1.7;
    const double a_ = 1.2;

    ND::Array_CPU< double,dim_1, dim_2> A(a);
    ND::Array_CPU< double,dim_1, dim_2> A_(a_);

    CHECK_EQ(A.size(0), dim_1);
    CHECK_EQ(A.size(1), dim_2);


    CHECK_EQ((-A)(0,0), (-a));
    CHECK_EQ(abs(A)(0,0), Approx(std::abs(a)));
    CHECK_EQ(exp(A)(0,0), Approx(std::exp(a)));
    CHECK_EQ(log(A)(0,0), Approx(std::log(a)));
    CHECK_EQ(log2(A)(0,0), Approx(std::log2(a)));
    CHECK_EQ(log10(A)(0,0), Approx(std::log10(a)));
    CHECK_EQ(sqrt(A)(0,0), Approx(std::sqrt(a)));

    CHECK_EQ(pow(A,A_)(0,0), Approx(std::pow(a,a_)));
    CHECK_EQ(pow(A,a_)(0,0), Approx(std::pow(a,a_)));
    CHECK_EQ(pow(a_,A)(0,0), Approx(std::pow(a_,a)));
    
    CHECK_EQ(round(A)(0,0), round(a));
    CHECK_EQ(ceil(A)(0,0), ceil(a));
    CHECK_EQ(floor(A)(0,0), floor(a));


    ND::Array_CPU< double,dim_1, dim_2> B;
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
    ND::Array_CPU< int, dim_1,dim_2> C(c);
    ND::Array_CPU< int, dim_1,dim_2> D(d);

    CHECK_EQ((C%D)(0,0), c%d);
}