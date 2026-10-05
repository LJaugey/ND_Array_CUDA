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

constexpr ComputeBackend backend = ComputeBackend::CUDA;

double t = 0;
const double dt = 0.01;
const double t_fin = 2;

const int N = 200;

const int dim = 3;


int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    cout<<"================================"<<endl<<endl;
    cout<<"Testing lazy evaluation speed-up"<<endl<<endl;
    cout<<"================================"<<endl<<endl;

    ND::Array<backend, double,N,N,dim,dim,dim> A(2.0);
    ND::Array<backend, double,N,N,dim,dim,dim> B(0.5);
    ND::Array<backend, double,N,N,dim,dim,dim> C(0.0);

    
    cout<<endl<<"Normal evaluation : ";
    if(backend == ComputeBackend::CUDA)
        cout<<"CUDA"<<endl;
    else
    {
        #ifdef _OPENMP
        cout<<"Multi-core ("<<omp_get_max_threads()<<" threads)"<<endl;
        #else
        cout<<"Single-core"<<endl;
        #endif
    }
    t = 0;
    start = std::chrono::high_resolution_clock::now();
    while(t<t_fin)
    {
        C = (((cos(A).eval() + sin(B).eval()).eval() - (A*dt).eval()).eval() + (((2*B).eval()*dt).eval() * tan(C).eval()).eval()).eval();

        t+=dt;
    }
    if(backend == ComputeBackend::CUDA)
        cudaStreamSynchronize(0);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout<<endl<<"test end. Code ran in "<<(double)duration.count()/1000.0<<" seconds"<<endl<<endl<<endl;

    double base_time = (double)duration.count()/1000.0;


    C = 0.0;

    cout<<endl<<"Lazy evaluation : ";
    if(backend == ComputeBackend::CUDA)
        cout<<"CUDA"<<endl;
    else
    {
        #ifdef _OPENMP
        cout<<"Multi-core ("<<omp_get_max_threads()<<" threads)"<<endl;
        #else
        cout<<"Single-core"<<endl;
        #endif
    }
    
    t = 0;
    start = std::chrono::high_resolution_clock::now();
    while(t<t_fin)
    {
        C = cos(A) + sin(B) - A*dt + 2*B*dt * tan(C);
        t+=dt;
    }
    if(backend == ComputeBackend::CUDA)
        cudaStreamSynchronize(0);
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



    return 0;
}