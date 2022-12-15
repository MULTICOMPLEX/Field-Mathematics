

#include <type_traits>

using std::conditional;
using std::integral_constant;

template<int... Ints>
struct Max;

template<int First, int... Rest>
struct Max<First, Rest...> : Max<First, Max<Rest...>::value> {};

template<int First, int Second>
struct Max<First, Second> :
    conditional<(First > Second),
        integral_constant<int, First>,
        integral_constant<int, Second>>::type {};
    
template<int Only>
struct Max<Only> : integral_constant<int, Only> {};


using std::integral_constant;

template<long N> struct Factorial : 
    integral_constant<unsigned, N * Factorial<N - 1>::value> {};

template<> 
struct Factorial<0> :
    integral_constant<long, 1> {};

///////////////
template <int N> 
struct FActorial {
 static const int result = N * FActorial<N-1>::result;
};
 
template <> 
struct FActorial<0> {
 static const int result = 1;
};

//////////////

template <int N, int D> struct Frak {
        static const long Num = N;
        static const long Den = D;
};
  
template <int N, typename F> struct ScalarMultiplication {
    typedef Frak<N*F::Num, N*F::Den> result;
};
  
template <int X, int Y>   struct MCD {
        static const long result = MCD<Y, X % Y>::result;
};
  
template <int X> struct MCD<X, 0> {
        static const long result = X;
};
  
template <class F> struct Simpl {
        static const long mcd = MCD<F::Num, F::Den>::result;
        typedef Frak< F::Num / mcd, F::Den / mcd > result;
};
  
template <typename X1, typename Y1> struct SameBase {
    typedef typename ScalarMultiplication< Y1::Den, X1>::result X;
    typedef typename ScalarMultiplication< X1::Den, Y1>::result Y;
};
  
template <typename X, typename Y> struct Sum {
    typedef SameBase<X, Y> B;
    static const long Num = B::X::Num + B::Y::Num;
    static const long Den = B::Y::Den; // == B::X::Den
    typedef typename Simpl< Frak<Num, Den> >::result result;
};


template <int N> struct Fact {
    static const long result = N * Fact<N-1>::result;
};
template <> struct Fact<0> {
    static const long result = 1;
};
 
template <int N> struct E {
    // e = S(1/n!) = 1/0! + 1/1! + 1/2! + ...
    static const long Den = Fact<N>::result;
    typedef Frak< 1, Den > term;
    typedef typename E<N-1>::result next_term;
    typedef typename Sum< term, next_term >::result result;
};
template <> struct E<0> {
    typedef Frak<1, 1> result;
};
///////////

static_assert(Factorial<0>::value == 1);
static_assert(Factorial<1>::value == 1);
static_assert(Factorial<2>::value == 2);
static_assert(Factorial<3>::value == 6);
static_assert(Factorial<4>::value == 24);
static_assert(Factorial<5>::value == 120);
  
static_assert(Max<1>::value == 1);
static_assert(Max<1, 2>::value == 2);
static_assert(Max<1, 2, 3>::value == 3);
static_assert(Max<3, 2, 1>::value == 3);


template<int n> 
struct funStruct 
{ 
    enum { val = 2*funStruct<n-1>::val }; 
}; 
  
template<> 
struct funStruct<0> 
{ 
    enum { val = 1 }; 
}; 
  

//static_assert(std::is_same<MX1, MX0>::value);