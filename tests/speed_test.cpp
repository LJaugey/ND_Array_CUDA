#include <valarray>
#include <chrono>
#include <iostream>

#include "../ND_Array/ND_Array.hpp"

typedef std::valarray<double> Mat_1;
typedef std::valarray<Mat_1> Mat_2;
typedef std::valarray<Mat_2> Mat_3;
typedef std::valarray<Mat_3> Mat_4;
typedef std::valarray<Mat_4> Mat_5;

using namespace std;


double t = 0;
const double dt = 0.01;
const double t_fin = 2;

const int N = 200;

const int dim = 3;

void func_par(ND::Array_CPU<double,dim,dim,dim> u)
{
    for(int l = 0; l<dim; l++)
    {
        for(int m = 0; m<dim; m++)
        {
            for(int s = 0; s<dim; s++)
            {
                u(l,m,s) += dt;
            }
        }
    }
}
void func_brac(ND::Array_CPU<double,dim,dim,dim> u)
{
    for(int l = 0; l<dim; l++)
    {
        for(int m = 0; m<dim; m++)
        {
            for(int s = 0; s<dim; s++)
            {
                u[l][m][s] += dt;
            }
        }
    }
}




void func_(Mat_3 & u)
{
    for(int l = 0; l<dim; l++)
    {
        for(int m = 0; m<dim; m++)
        {
            for(int s = 0; s<dim; s++)
            {
                u[l][m][s] += dt;
            }
        }
    }
}



int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    cout<<"================================"<<endl<<endl;
    cout<<"Testing lazy evaluation speed-up"<<endl<<endl;
    cout<<"================================"<<endl<<endl;

    ND::Array_CPU<double,N,N,dim,dim,dim> A(2.0);
    ND::Array_CPU<double,N,N,dim,dim,dim> B(0.5);
    ND::Array_CPU<double,N,N,dim,dim,dim> C(0.0);


    cout<<endl<<"Normal evaluation : ";
    #ifdef _OPENMP
    cout<<"Multi-core ("<<omp_get_max_threads()<<" threads)"<<endl;
    #else
    cout<<"Single-core"<<endl;
    #endif
    t = 0;
    start = std::chrono::high_resolution_clock::now();
    while(t<t_fin)
    {
        C = (((cos(A).eval() + sin(B).eval()).eval() - (A*dt).eval()).eval() + (((2*B).eval()*dt).eval() * tan(C).eval()).eval()).eval();

        t+=dt;
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout<<endl<<"test end. Code ran in "<<(double)duration.count()/1000.0<<" seconds"<<endl<<endl<<endl;
    
    double base_time = (double)duration.count()/1000.0;


    C = 0.0;

    cout<<endl<<"Lazy evaluation : ";
    #ifdef _OPENMP
    cout<<"Multi-core ("<<omp_get_max_threads()<<" threads)"<<endl;
    #else
    cout<<"Single-core"<<endl;
    #endif
    t = 0;
    start = std::chrono::high_resolution_clock::now();
    while(t<t_fin)
    {
        C = cos(A) + sin(B) - A*dt + 2*B*dt * tan(C);
        t+=dt;
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout<<endl<<"test end. Code ran in "<<(double)duration.count()/1000.0<<" seconds"<<endl<<"Speed-up :"<<base_time/((double)duration.count()/1000.0)<<"x"<<endl<<endl;



    std::valarray<double> A_(2.0, N*N*dim*dim*dim);
    std::valarray<double> B_(0.5, N*N*dim*dim*dim);
    std::valarray<double> C_(0.0, N*N*dim*dim*dim);
    
    cout<<endl<<"valarray (lazy evaluation) : Single-core"<<endl;
    t = 0;
    start = std::chrono::high_resolution_clock::now();
    while(t<t_fin)
    {
        C_ = cos(A_) + sin(B_) - A_*dt + 2*B_*dt * tan(C_);
        t+=dt;
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout<<endl<<"test end. Code ran in "<<(double)duration.count()/1000.0<<" seconds"<<endl<<"Speed-up :"<<base_time/((double)duration.count()/1000.0)<<"x"<<endl<<endl;



    double* _A_ = (double*)malloc(sizeof(double)*N*N*dim*dim*dim);
    double* _B_ = (double*)malloc(sizeof(double)*N*N*dim*dim*dim);
    double* _C_ = (double*)malloc(sizeof(double)*N*N*dim*dim*dim);
    
    for(int i = 0; i<N*N*dim*dim*dim; i++)
    {
        _A_[i] = 2.0;
        _B_[i] = 0.5;
        _C_[i] = 0.0;
    }
    
    cout<<endl<<"Hand-written loop: ";
    #ifdef _OPENMP
    cout<<"Multi-core ("<<omp_get_max_threads()<<" threads)"<<endl;
    #else
    cout<<"Single-core"<<endl;
    #endif
    t = 0;
    start = std::chrono::high_resolution_clock::now();
    while(t<t_fin)
    {
        PARALLEL_FOR(N*N*dim*dim*dim)
        for(int i = 0; i<N*N*dim*dim*dim; i++)
            _C_[i] = cos(_A_[i]) + sin(_B_[i]) - _A_[i]*dt + 2*_B_[i]*dt * tan(_C_[i]);
        t+=dt;
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout<<endl<<"test end. Code ran in "<<(double)duration.count()/1000.0<<" seconds"<<endl<<"Speed-up :"<<base_time/((double)duration.count()/1000.0)<<"x"<<endl<<endl;




    cout<<endl<<endl;


    ND::Array_CPU<double,N,N,dim,dim,dim> u;
    u.fill(1.0);

    Mat_5 u_(Mat_4(Mat_3(Mat_2(Mat_1(1.0, dim), dim), dim), N), N);


    cout<<"====================================="<<endl<<endl;
    cout<<"Testing operator() vs operator[] vs nested valarrays"<<endl<<endl;
    cout<<"====================================="<<endl<<endl;


    // Traverse the array once to reduce the difference with/without caching
    #pragma omp parallel for collapse(2)
    for(int i = 0; i<N; i++)
    {
        for(int j = 0; j<N; j++)
        {
            func_(u_[i][j]);
        }
    }

    cout<<endl<<"test start with valarray"<<endl;
    t = 0;
    start = std::chrono::high_resolution_clock::now();
    while(t<t_fin*10)
    {
        for(int i = 0; i<N; i++)
        {
            for(int j = 0; j<N; j++)
            {
                func_(u_[i][j]);
            }
        }
        t += dt;
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout<<endl<<"test end. Code ran in "<<(double)duration.count()/1000.0<<" seconds"<<endl<<endl;

    base_time = (double)duration.count()/1000.0;


    // Traverse the array once to reduce the difference with/without caching
    #pragma omp parallel for collapse(2)
    for(int i = 0; i<N; i++)
    {
        for(int j = 0; j<N; j++)
        {
            func_par(u[i][j]);
        }
    }

    cout<<endl<<"test start with operator[]"<<endl;
    t = 0;
    start = std::chrono::high_resolution_clock::now();
    while(t<t_fin*10)
    {
        for(int i = 0; i<N; i++)
        {
            for(int j = 0; j<N; j++)
            {
                func_brac(u[i][j]);
            }
        }
        t += dt;
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout<<endl<<"test end. Code ran in "<<(double)duration.count()/1000.0<<" seconds"<<endl<<"Speed-up :"<<base_time/((double)duration.count()/1000.0)<<"x"<<endl<<endl;




    cout<<endl<<"test start with operator()"<<endl;
    t = 0;
    start = std::chrono::high_resolution_clock::now();
    while(t<t_fin*10)
    {
        for(int i = 0; i<N; i++)
        {
            for(int j = 0; j<N; j++)
            {
                func_par(u[i][j]);
            }
        }
        t += dt;
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout<<endl<<"test end. Code ran in "<<(double)duration.count()/1000.0<<" seconds"<<endl<<"Speed-up :"<<base_time/((double)duration.count()/1000.0)<<"x"<<endl<<endl;

    
    

    return 0;
}