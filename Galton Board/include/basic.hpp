
#include <list>

template<class Fn, std::size_t... N> 
void unroll_impl
(
  Fn fn, 
  std::integer_sequence<std::size_t, N...> iter) 
{
(
  void(fn(N)), 
  ...
);
}

template<int N, class Fn>
void unroll
(
  Fn fn
) 
{
  unroll_impl(fn, std::make_index_sequence<N>());
}


/////////////
//Does c++ support inductive type definitions?
//https://stackoverflow.com/questions/36504245/does-c-support-inductive-type-definitions
//https://en.wikipedia.org/wiki/Inductive_type:
template <int k>
struct Myclass_helper { using type = std::list<typename Myclass_helper<k-1>::type>; };

//if ommitted : template instantiation depth exceeds maximum of 900
template < >
struct Myclass_helper<1> { using type = double; };
template < >
struct Myclass_helper<0> { using type = int; };

template <int k>
using Myclass = typename Myclass_helper<k>::type;

//class My_class<int k>=
//  if k=0
//     int
//  else
//     list<Myclass<k-1>>

//Myclass<0> nn;
//std::cout << typeid(nn).name() << std::endl;

///////////////
//https://rosettacode.org/wiki/Mutual_recursion#C.2B.2B
//https://en.wikipedia.org/wiki/Mutual_recursion
//Two functions are said to be mutually recursive if the first calls the second, and in turn the second calls the first. 
//////////////

//Recursive data type:
//type for values that may contain other values of the same type

//Algebraic data type:
//type formed by combining other types

//Inductive data type: 
//Recursive data type or Algebraic data type
//Inductive family, a family of inductive data types indexed by another type or value
 
//Inductive type
//In type theory, a system has inductive types if it has facilities for creating a new type 
//along with constants and functions that create terms of that type. 

//////////////


/* elem
 
    /// addition
    const multicomplex operator +  ( 
    const multicomplex& b
    ) const 
    {
      elem x = real;
      elem y = imag;
      elem u = b.real;
      elem v = b.imag;
      //w+z = (x+u) + (y+v) in+1 
      elem c = (x+u);
      elem d = (y+v);
      return {c,d};
    }
    
    /// subtraction
    const multicomplex operator -  ( 
    const multicomplex& b
    ) const 
    {
      elem x = real;
      elem y = imag;
      elem u = b.real;
      elem v = b.imag;
      //w+z = (x+u) + (y+v) in+1 
      elem c = (x-u);
      elem d = (y-v);
      return {c,d};
    }
    
    /// multiplication
    const multicomplex operator *  ( 
    const multicomplex& b
    ) const 
    {
      elem x = real;
      elem y = imag;
      elem u = b.real;
      elem v = b.imag;
      //wz = (ux − vy) + (uy + vx) in+1 
      elem c = (u*x - v*y);
      elem d = (u*y + v*x);
      return {c,d};
    }
  
    /// division
    const multicomplex operator /  ( 
    const multicomplex& b
    ) const 
    {
      elem x = real;
      elem y = imag;
      elem u = b.real;
      elem v = b.imag;
      //w/z = (x*u+y*v) / (u*u+v*v) + (y*u-x*v) / (u*u+v*v) in+1 
      elem c = (x*u+y*v) / (u*u+v*v);
      elem d = (y*u-x*v) / (u*u+v*v);
      return {c,d};
    }
    
  */
  
  //Settings :
  
  //c++ compiler options: -target x86_64-pc-windows-gnu
  //linker options: -target x86_64;-Wl,--stack,104857600
  
  //-emit-llvm
  
  //rgb(171, 178, 191) old identifier setting
  //rgb(117, 148, 224) new identifier, in: settings, colors and fonts, customize,  c++ , styles, identifier
  
  //compiler
  //http://winlibs.com/
  //https://nuwen.net/mingw.html

template<typename T>    
bool isPrime2
(
  T n
) 
{ 
    // Corner case 
    if (n <= 1) 
        return false; 
  
    // Check from 2 to n-1 
    for (int i = 2; i < n; i++) 
        if (n % i == 0) 
            return false; 
  
    return true; 
} 

template<typename T>
bool isPrime_driver
(
  T n
)
{ 
  bool k = isPrime2(n);
  k ? std::cout << " true\n" : std::cout << " false\n";
  return k;
}

//#include <boost/xpressive/xpressive.hpp>
/*
 * 
#ifdef __MINGW32__
#include <windows.h>
#endif
#include <ginac/ginac.h>
 * 
using namespace boost::xpressive;
  std::string hello( "hello world!" );

    sregex rex = sregex::compile( "(\\w+) (\\w+)!" );
    smatch what;

    if( regex_match( hello, what, rex ) )
    {
        std::cout << what[0] << '\n'; // whole match
        std::cout << what[1] << '\n'; // first capture
        std::cout << what[2] << '\n'; // second capture
    }
  
  
  
void test_ginac()
{
  GiNaC::symbol a("a"), b("b"), x("x"), y("y"), z("z");
  
  GiNaC::lst vars = {x,y,z};
  GiNaC::lst eqns;
  //eqns.append(-6 * x + 3 * y == 6 * x );
  //eqns.append( 4 * x + 5 * y == 6 * y );
  
  eqns.append(2*x == -x );
  eqns.append(4*y + 5*z == -y );
  eqns.append(4*y + 3*z == -z );
  
  std::cout << GiNaC::lsolve(eqns, vars) << std::endl;
  
  std::cout << std::endl;
}
*/

/*
//hash test
std::vector<std::string> str  = {"∇","+","3","*","4","/","3","-2"};
  std::vector<std::size_t> strh (str.size());  
  
  for(std::size_t i=0; i < str.size(); i++) strh[i] = std::hash<std::string>{}(str[i]);
  
  for (auto &i : strh){
      if(i==strh[0]) std::cout << "\nbla bla\n"; 
  }
*/

// General case
template<unsigned int N>
struct fibonacci
{
    static constexpr int64_t value =
        fibonacci<N - 2>::value +
        fibonacci<N - 1>::value;
};

// Special case, value 0
template<>
struct fibonacci<0>
{
    static constexpr int64_t value = 0;
};

// Special case, value 1
template<>
struct fibonacci<1>
{
    static constexpr int64_t value = 1;
};

template <int64_t n>
int Fibonacci_driver() 
{
 constexpr auto fib6 = fibonacci<6>::value; // 8
    std::cout << fib6 << '\n';
  return 1;
}


