#ifndef __ELEMENTRY_FUNCTIONS_HPP__
#define __ELEMENTRY_FUNCTIONS_HPP__

/// primary functions

template <typename elem, int order>
  requires ((order >= 0) && (order < 25))
class multicomplex;

/// sqrt
template <typename elem, int order> 
multicomplex<elem,order> sqrt
(
  const multicomplex<elem,order>& x
) 
{
 // if(x<0 && x > -lambda)return 0;
 // if(x>0 && x <  lambda)return 0;
  
  //multicomplex<elem,order> x_0{half,half};//initial guess
  
  multicomplex<elem,order> x_0;//initial guess 
  
  elem a = 1;
  elem k = 1;
  
  //start in the right quadrant
  if(x.signr()<0){k = -k;}
  if(x.signi()<0){a = -a;}
  
  multicomplex<elem,0> ig{k,a};//initial guess
  x_0 += ig; 
  
  unroll<14>([&](int iter){
    
  x_0 *= (x_0*x_0 + 3*x)/(3*x_0*x_0+x); //Halley 
  //  x_0 = half*(x_0 + (x/x_0)); //Newton 
  });
    
  return x_0;
}

//---------------------------------------------------

template <typename elem, int order> 
inline multicomplex<elem,order> sqrto 
(
  const multicomplex<elem,order>& x
) //Newton method complex
{ 
  multicomplex<elem,order> x_0{half,half};//initial guess
  
  for(int t = 0; t < 8; t++){
    x_0 *= (x_0*x_0 + 3*x)/(3*x_0*x_0+x); //Halley
    //x_0 = half*(x_0 + (x/x_0)); //Newton 
  }
  return x_0;
}

//---------------------------------------------------

template <typename elem, int order> 
multicomplex<elem,order> sqrts 
(
  const multicomplex<elem,order>& x
) //Newton method complex
{ 
  multicomplex<elem,order> x_0{half,half};//initial guess
 
  for(int t = 0; t < 4; t++){
    x_0 *= (x_0*x_0 + 3*x)/(3*x_0*x_0+x); //Halley 
   // x_0 = half*(x_0 + (x/x_0)); //Newton 
  }
  return x_0;
}

//---------------------------------------------------

template <typename elem, int order>
inline multicomplex<elem, order> conj
(
  const multicomplex<elem, order>& x
)
{
  auto k = x;
  return k.conj();
}

//---------------------------------------------------

template <typename elem, int order> 
multicomplex<elem,order> sqrtpow 
(
  const multicomplex<elem,order>& x
) 
{ 
  return pow(x,{half,0});
}

//---------------------------------------------------

/// log
template <typename elem, int order>
multicomplex<elem,order> log 
(
  const multicomplex<elem,order> x
)// log Newton
{   
 // multicomplex<elem,order> x_0(sqrts(x));
 // multicomplex<elem,order> x_0{1,1};
  multicomplex<elem,order> x_0;//initial guess 
  
  elem a = 1;
  elem k = 1;
  
  //start in the right quadrant
  if(x.signr()<0){k = -k;}
  if(x.signi()<0){a = -a;}
  
  multicomplex<elem,0> ig{k,a};//initial guess
  x_0 += ig; 
   
  for(int t = 0; t < 17; t++)
    x_0 -= 2*((exp(x_0)-x)/(exp(x_0)+x));
    
  return x_0;
  //log derivative
  //MX1 ax; 
  //return {ax.Real,x.Imag/x.Real};
}

//---------------------------------------------------

///pow ^complex
template <typename elem, int order>
multicomplex<elem,order> pow 
(
  const multicomplex<elem,order>& b, 
  const multicomplex<elem,order>& e
)
{
	return exp(e * log(b));
}

//---------------------------------------------------

///pow ^REAL
template <typename elem, int order>
multicomplex<elem, order> pow
(
  const multicomplex<elem, order>& b,
  double exp
)
{
  return pow(b, static_cast<multicomplex<elem, order>>(exp));
}

//---------------------------------------------------

// pow ^int
template <typename elem, int order>
inline multicomplex<elem, order> pow
(
  const multicomplex<elem, order>& b, 
  size_t exp
) 
{
  if( exp == 0)
    return 1;
  
  multicomplex<elem,order> temp = pow(b, exp/2);       
  
  if (exp%2 == 0)
      return temp*temp;
  else 
  {
    if(exp > 0)
        return b*temp*temp;
    else
      return (temp*temp)/b; //negative exponent computation 
  }
}

template <typename elem, int order>
inline multicomplex<elem, order> pow
(
  const multicomplex<elem, order>& b,
  int exp
)
{
  if (exp == 0)
    return 1;

  multicomplex<elem, order> temp = pow(b, exp / 2);

  if (exp % 2 == 0)
    return temp * temp;
  else
  {
    if (exp > 0)
      return b * temp * temp;
    else
      return (temp * temp) / b; //negative exponent computation 
  }
}

template <typename elem, int order>
  requires std::floating_point<elem>
inline multicomplex<elem, order> floor
(
  const multicomplex<elem, order> n
) {
  return {floor(n.Real), floor(n.Imag)};
}


//---------------------------------------------------

template <typename elem, int order>
  requires std::floating_point<elem>
inline multicomplex<elem, order> fmod
(
  const multicomplex<elem, order>& numer, 
  const multicomplex<elem, order>& denom
)
{
  return numer - floor(numer / denom) * denom;
}


//---------------------------------------------------

/// trigonometric functions

// sine
template <typename elem, int order>
inline multicomplex<elem,order> sin 
(
  multicomplex<elem,order> const & z
) 
{
  return 
  {
    sin (z.Real) * cosh (z.Imag),
    cos (z.Real) * sinh (z.Imag)
  };
}

//---------------------------------------------------

template <typename elem>
inline multicomplex<elem, 0> sin 
(
  multicomplex<elem,0> const & z
) 
{
  return 
  {
    std::sin (z.Real) * std::cosh (z.Imag),
    std::cos (z.Real) * std::sinh (z.Imag)
  };
}

//---------------------------------------------------

// sine hyperbolic
template <typename elem, int order>
inline multicomplex<elem,order> sinh 
(
  multicomplex<elem,order> const & z) 
{
  return 
  {
    sinh (z.Real) * cos (z.Imag),
    cosh (z.Real) * sin (z.Imag)
  };
}

//---------------------------------------------------

template <typename elem>
inline multicomplex<elem, 0> sinh 
(
  multicomplex<elem,0> const & z
)
{
  return 
  {
    std::sinh (z.Real) * std::cos (z.Imag),
    std::cosh (z.Real) * std::sin (z.Imag)
  };
}

//---------------------------------------------------

// cosine
template <typename elem, int order>
inline multicomplex<elem,order> cos 
(
  multicomplex<elem,order> const & z
) 
{
 return 
  {
    + cos (z.Real) * cosh (z.Imag),
    - sin (z.Real) * sinh (z.Imag)
  };
}

//---------------------------------------------------

template <typename elem>
inline multicomplex<elem, 0> cos 
(
  multicomplex<elem,0> const & z) 
{
  return 
  {
    + std::cos (z.Real) * std::cosh (z.Imag),
    - std::sin (z.Real) * std::sinh (z.Imag)
  };
}

//---------------------------------------------------

// tan
template <typename elem, int order>
inline multicomplex<elem,order> tan 
(
  multicomplex<elem,order> const & z
) 
{
  return 
  {
    sin(z) / cos(z)
  };
}

//---------------------------------------------------

// cosine hyperbolic
template <typename elem, int order>
inline multicomplex<elem,order> cosh 
(
  multicomplex<elem,order> const & z
) 
{
  return 
  {
    cosh (z.Real) * cos (z.Imag),
    sinh (z.Real) * sin (z.Imag)
  };
}

//---------------------------------------------------

template <typename elem>
inline multicomplex<elem, 0> cosh 
(
  multicomplex<elem,0> const & z
) 
{
  return 
  {
    std::cosh (z.Real) * std::cos (z.Imag),
    std::sinh (z.Real) * std::sin (z.Imag)
  };
}

//---------------------------------------------------

// tanh
template <typename elem, int order>
inline multicomplex<elem,order> tanh 
(
  multicomplex<elem,order> const & z
) 
{
  return 
  {
    sinh(z) / cosh(z)
  };
}

//---------------------------------------------------

// exponential
template <typename elem, int order>
inline multicomplex<elem,order> exp 
(
  multicomplex<elem,order> const & z
) 
{
  multicomplex<elem,order-1> const r {exp (z.Real)};
  return  
  {
    r * cos (z.Imag),
    r * sin (z.Imag)
  };
}

//---------------------------------------------------

template <typename elem>
inline multicomplex<elem, 0> exp 
(
  multicomplex<elem,0> const & z
) 
{
  return 
  {
    std::exp (z.Real) * std::cos (z.Imag),
    std::exp (z.Real) * std::sin (z.Imag)
  };
}

//---------------------------------------------------

template <typename elem, int order>
inline multicomplex<elem, order> ldexp 
(
  multicomplex<elem,0> const & z,
  int const & e
) 
{
  return 
  {
    z * elem(std::pow(2,e))
  };
}

//---------------------------------------------------

template <typename elem>
inline multicomplex<elem, 0> ldexp 
(
  multicomplex<elem,0> const & z,
  int const & e
) 
{
  return 
  {
    z * elem(std::pow(2,e))
  };
}

//---------------------------------------------------

template <typename elem, int order>
inline multicomplex<elem,order> expl 
(
  multicomplex<elem,order> const & z
) 
{
  multicomplex<elem,order-1> const r {expl (z.Real)};
  return  
  {
    r * cos (z.Imag),
    r * sin (z.Imag)
  };
}

//---------------------------------------------------

template <typename elem>
inline multicomplex<elem, 0> expl 
(
  multicomplex<elem,0> const & z
) 
{
  return 
  {
    elem(expl (z.Real)) * std::cos (z.Imag),
    elem(expl (z.Real)) * std::sin (z.Imag)
  };
}

//---------------------------------------------------

template <typename T>
T factorial
(
  std::size_t number
) 
{ 
  T num = T(1); 
  for (size_t i = 1; i <= number; i++) 
      num *= i; 
  return num; 
}

//---------------------------------------------------


template <typename elem, int order> 
inline multicomplex<elem,order> Sin 
(
  const multicomplex<elem,order>& x
)
{ 
  multicomplex<elem,order> result;
  elem j;

  unroll<15>([&](size_t n){
 // for(int n = 0; n < N; n++)
 // { 
    j = (1.0/factorial<elem>(2*n+1)) * std::pow(-1, n);
    result += pow(x, 2*n+1) * j;
   // std::cout << result << std::endl;
 // }
  
    });
    
  return result;
}

//---------------------------------------------------

template <typename elem, int order> 
multicomplex<elem,order> Cos 
(
  const multicomplex<elem,order>& x
) 
{ 
  multicomplex<elem,order> result;
  elem j;
  
  for(int n = 0; n < 15; n++)
  {
    j = (1.0/factorial<elem>(2*n)) * pow(-1, n);
    result += pow(x, 2*n) * j;
  }

  return result;
}

//---------------------------------------------------

template <typename elem, int order> 
multicomplex<elem,order> Cosh 
(
  const multicomplex<elem,order>& x
)
{ 
  multicomplex<elem,order> result;
  elem j;
  
  for(int n = 0; n < 15; n++)
  {
    j = 1.0/factorial<elem>(2*n);
    result += pow(x, 2*n) * j;
  }

  return result;
}

//---------------------------------------------------

template <typename elem, int order> 
multicomplex<elem,order> Sinh 
(
  const multicomplex<elem,order>& x
)
{ 
  multicomplex<elem,order> result;
  elem j;
  
  for(int n = 0; n < 15; n++)
  {
    j = 1.0/factorial<elem>(2*n+1);
    result += pow(x, 2*n+1) * j;
  }

  return result;
}

//---------------------------------------------------

template <typename elem, int order> 
multicomplex<elem,order> Exp 
(
  const multicomplex<elem,order>& x
)
{ 
  multicomplex<elem,order> result;
  elem j;  
  elem k=1;
  for(int n = 0; n < 2*15; n++)
  {
    if(n>1) k *= n;
    j = 1.0/k;
    result += pow(x, n) * j;
  }

  return result;
}

//---------------------------------------------------

template <typename elem, int order>
inline const multicomplex<elem, order-1> abs 
(
  const multicomplex<elem, order>& a 
) 
{
  return 
  {
    sqrt(a.Real*a.Real + a.Imag*a.Imag)
  };
}

//---------------------------------------------------

template <typename elem>
inline const elem abs 
(
  const multicomplex<elem, 0>& a 
) 
{
  return 
  {
    std::sqrt(a.Real*a.Real + a.Imag*a.Imag)
  };
}

//---------------------------------------------------

template<typename elem, int order>
inline multicomplex<elem,order> arg 
(
  const multicomplex<elem,order> & z
) 
{
  std::complex a(z.Real.Real,z.Real.Imag);
  auto x = std::arg(a);
  
  multicomplex<elem,order> b{x};
  return 
  {
    b
  };
}

//---------------------------------------------------

template<typename elem>
inline multicomplex<elem,0> arg 
(
  const multicomplex<elem,0> & z
) 
{
  std::complex a(z.Real,z.Imag);
  auto x = std::arg(a);
  
  multicomplex<elem,0> b{x};
  return 
  {
    b
  };
}

//---------------------------------------------------

template <typename T>
T Fac
(
  std::size_t number
) 
{ 
  T num = T(1); 
  for (size_t i = 2; i <= number; i++) 
      num *= i; 
  return num; 
}

//---------------------------------------------------


inline size_t Pochhammer(size_t x, size_t n)
{
  size_t c = 1;
  for(size_t k = 0; k <= n-1; k++)
  { 
      c *= x + k;
  }
  return c;
  
  //The coefficients that appear in the expansions are Stirling numbers of the first kind. 
  //return std::tgamma(x+n) / std::tgamma(x);
}

//---------------------------------------------------


template<typename A, typename T>
A Binomial_Coefficient(const T& n, const T& k)
{
  return Fac<A>(n) / (Fac<A>(n-k) * Fac<A>(k));
}

//---------------------------------------------------

template<typename elem, int order>
multicomplex<elem,order> Riemann_Zeta
(
  const multicomplex<elem, order>& s
)
{
  multicomplex<elem, order> sum1 = 0, sum2 = 0;
  
  for(size_t n=0; n <= 24; n++){
  
    sum1 = 0;
    for(size_t k=0; k <= n; k++){
    
    sum1 += elem(std::pow(-1, k)) * 
    Binomial_Coefficient<size_t>(n,k) * pow(multicomplex<elem,order>(elem(k)+1), -s);
  }
  
  sum1 *= 1 / elem(pow(2,n+1));

  sum2 += sum1;
  }
  
  return (1 / (1-pow(multicomplex<elem,order>(2),1-s))) * sum2;
}

//---------------------------------------------------

//RiemannSiegelTheta()
template<typename elem, int order>
multicomplex<elem, order> Riemann_Siegel_theta
(
  const multicomplex<elem, order>& t
)
{
  multicomplex<elem,0> i = {0,1};
  return arg(gamma(quarter + half * i * t.Real)) - half * t * std::log(pi);
}

//---------------------------------------------------

template <typename elem>
multicomplex<elem, 0> Polygamma
(
  int n,
  const multicomplex<elem, 0>& z
) 
{   
  mcdv mcdv; 
  
  multicomplex<elem, 0> r;
  
  MX1 x1;
  MX2 x2;
  MX3 x3;
  MX4 x4;
  MX5 x5;
  
  n +=1; 

  if(n == 1){ mcdv.sh<0>(x1, z);  r = mcdv.dv<0>(log(gamma(x1)));}
  if(n == 2){ mcdv.sh<0>(x2, z);  r = mcdv.dv<0>(log(gamma(x2)));} 
  if(n == 3){ mcdv.sh<0>(x3, z);  r = mcdv.dv<0>(log(gamma(x3)));} 
  if(n == 4){ mcdv.sh<0>(x4, z);  r = mcdv.dv<0>(log(gamma(x4)));} 
  if(n == 5){ mcdv.sh<0>(x5, z);  r = mcdv.dv<0>(log(gamma(x5)));}  
  
  return r;
} 

//---------------------------------------------------

template <typename elem>
elem Kronecker_Delta
(
  const elem & k
) 
{   
  if(k==0)return 1;
  return 0;
}

//---------------------------------------------------

template<typename elem>
multicomplex<elem, 0> log_gamma
(
  const multicomplex<elem, 0>& t
)
{
  return log(gamma(t));
}

//---------------------------------------------------

template<typename elem>
multicomplex<elem, 0> Riemann_Siegel_theta
(
  const multicomplex<elem, 0>& t
)
{
  multicomplex<elem,0> i = {0,1};
  
  return -(i/2) * (log_gamma(0.25L+i*t/2) - log_gamma(0.25L-i*t/2)) - ((std::log(pi)*t)/2);
}

//---------------------------------------------------

//RiemannSiegelZ()
template<typename elem, int order>
multicomplex<elem, order> Riemann_Siegel_Z
(
  const multicomplex<elem,order>& t
)
{
  multicomplex<elem,0> i = {0,1};

  return exp(i * Riemann_Siegel_theta(t)) * Riemann_Zeta(half + i * t);
}

//---------------------------------------------------

template<typename elem, int order>
multicomplex<elem, order> Riemann_Siegel_Z2
(
  const multicomplex<elem,order>& t
)
{
  multicomplex<elem,0> i = {0,1};

  return  Riemann_Zeta(half + i * t);
}

//---------------------------------------------------

template <typename elem, int order> 
multicomplex<elem, order> erf
(
  multicomplex<elem, order> const & z
)
{
  multicomplex<elem, order> s = 0;
  for(int n=0; n < 40; n++)
    s += elem(std::pow(-1,n)) * pow(z,2*n+1) / (factorial<elem> (n) * (2*n+1));
  
  return (2./std::sqrt(pi)) * s;
}

//---------------------------------------------------

template <typename T>
T erf
(
  T const & z
)
{
  T s = 0;
  for(int n=0; n < 40; n++)
    s += T(std::pow(-1,n)) * pow(z,2*n+1) / (factorial<T> (n) * (2*n+1));
  
  return (2./std::sqrt(pi)) * s;
}


template <typename elem, int order> 
multicomplex<elem,order> asin 
(
	const multicomplex<elem,order> & z 
) 
{
	MX0 i;
	i.Real = 0;
	i.Imag = 1;
		
	return -i * log(sqrt(1 - z*z) + i * z);	//-i log(sqrt(1 - z^2) + i z)	
}

template <typename elem, int order> 
multicomplex<elem,order> acos 
(
	const multicomplex<elem,order> & z 
) 
{
	MX0 i;
	i.Real = 0;
	i.Imag = 1;
		
	return half_pi + i * log(sqrt(1 - z*z) + i * z);//π/2 + i log(sqrt(1 - z^2) + i z)		
}

template <typename elem, int order> 
multicomplex<elem,order> atan
(
  const multicomplex<elem,order>& z
) 
{
	MX0 i;
	i.Real = 0;
	i.Imag = 1;
	return half * i * log(1 - i * z) - half * i * log(1 + i * z);//1/2 i log(1 - i z) - 1/2 i log(1 + i z)		
}


template <typename elem, int order> 
multicomplex<elem,order> atan2
(
  const multicomplex<elem,order>& z
) 
{
	return atan2(z.Imag,z.Real);
}

template <typename elem> 
elem atan2
(
  const multicomplex<elem,0>& zi
) 
{
	//multicomplex<elem,0> z;
	//z.Real = zi.Imag;
	//z.Imag = zi.Real;
	//if(z.Real>0)return std::atan(z.Imag/z.Real);
	//else if((z.Real<0) && (z.Imag >= 0))return std::atan(z.Imag/z.Real) + pi;
	//else if((z.Real<0) && (z.Imag < 0))return std::atan(z.Imag/z.Real) - pi;
	//else if((z.Real==0) && (z.Imag > 0))return half_pi;
	//else if((z.Real==0) && (z.Imag < 0))return -half_pi;
	//else return 0; //if((z.Real==0) && (z.Imag == 0))
	return std::atan2(zi.Imag,zi.Real);
}

template <typename elem, int order> 
multicomplex<elem, order> polar_to_cartesian
(
	const multicomplex<elem,order>& z
)
{	
	//x=rcosθ,y=rsinθ
	return
	{
		abs(z) * cos(atan2(z)),
		abs(z) * sin(atan2(z))
	};
}

template <typename elem, int order> 
multicomplex<elem,order> asinh 
(
	const multicomplex<elem,order> & z 
) 
{		
	return log(sqrt(z*z + 1) + z);	//log(sqrt(z^2 + 1) + z)
}

template <typename elem, int order> 
multicomplex<elem,order> acosh 
(
	const multicomplex<elem,order> & z 
) 
{		
	return log(z + sqrt(z - 1) * sqrt(z + 1));//log(z + sqrt(z - 1) sqrt(z + 1))
}

template <typename elem, int order> 
multicomplex<elem,order> atanh
(
  const multicomplex<elem,order>& z
) 
{
	return half * log(z + 1) - half * log(1 - z);//1/2 log(z + 1) - 1/2 log(1 - z)		
}

template <typename elem, int order> 
multicomplex<elem,order> csc  //Cosecant
(
  const multicomplex<elem,order>& z
) 
{
	const multicomplex<elem,order> i(0,1);
	return (2 * i) / ((exp(i * z) - exp(-i * z)) + lambda); 
	
}

template <typename elem, int order> 
multicomplex<elem,order> atanh_dv
(
  const multicomplex<elem,order>& z
) 
{
	return 1 / (1 - z * z);	
}

template <typename elem, int order> 
multicomplex<elem,order> function
(
  const multicomplex<elem,order>& z0,
  const int formula
) 
{ 
			mcdv mcdv;
			
			MX0 i;
			i.Real = 0;
			i.Imag = 1;
			
			multicomplex<elem,order+1> z1;
			multicomplex<elem,order+2> z2;
			multicomplex<elem,order+3> z3;
			multicomplex<elem,order+4> z4;
			
			mcdv.sh<order>(z1, z0);
			mcdv.sh<order>(z2, z0);
			mcdv.sh<order>(z3, z0);
			mcdv.sh<order>(z4, z0);
			
			auto f = formula;
			if(formula == 1) return z0;
			
			else if(formula == 2) return log(z0);
			else if(formula == 3) return sqrt(z0);
			else if(formula == 4) return sin(z0);
			else if(formula == 5) return csc(z0);		
			else if(formula == 6) return cos(z0);
			else if(formula == 7) return tan(z0);
			else if(formula == 8) return sinh(z0);
			else if(formula == 9) return cosh(z0);
			else if(formula == 10) return tanh(z0);
			else if(formula == 11) return asin(z0);
			else if(formula == 12) return acos(z0);
			else if(formula == 13) return atan(z0);
			else if(formula == 14) return asinh(z0);
			else if(formula == 15) return acosh(z0);
			else if(formula == 16) return atanh(z0);
			else if(formula == 17) return atanh_dv(z0); //  1 / (1 - z^2)
			else if(formula == 18) return exp(z0);
			else if(formula == 19) return gamma(z0);
			else if(formula == 20) return LambertW(0,z0);
			else if(formula == 21) return erf(z0);
			else if(formula == 22) return Riemann_Zeta(z0);
			
			else if(formula == 23) return mcdv.dv<order>(z1);
			else if(formula == 24) return mcdv.dv<order>(log(z1));
			else if(formula == 25) return mcdv.dv<order>(sqrt(z1));
			else if(formula == 26) return mcdv.dv<order>(sin(z1));
			else if(formula == 27) return mcdv.dv<order>(csc(z1));
			else if(formula == 28) return mcdv.dv<order>(cos(z1));
			else if(formula == 29) return mcdv.dv<order>(tan(z1));
			else if(formula == 30) return mcdv.dv<order>(sinh(z1));
			else if(formula == 31) return mcdv.dv<order>(cosh(z1));
			else if(formula == 32) return mcdv.dv<order>(tanh(z1));
			else if(formula == 33) return mcdv.dv<order>(atanh(z1));
			else if(formula == 34) return mcdv.dv<order>(exp(z1));
			else if(formula == 35) return mcdv.dv<order>(gamma(z1));
			else if(formula == 36) return mcdv.dv<order>(LambertW(0,z1));
			else if(formula == 37) return mcdv.dv<order>(erf(z1));
			else if(formula == 38) return mcdv.dv<order>(Riemann_Zeta(z1));
			
			else if(formula == 39) return mcdv.dv<order>(z2);
			else if(formula == 40) return mcdv.dv<order>(log(z2));
			else if(formula == 41) return mcdv.dv<order>(sqrt(z2));
			else if(formula == 42) return mcdv.dv<order>(sin(z2));
			else if(formula == 43) return mcdv.dv<order>(csc(z2));
			else if(formula == 44) return mcdv.dv<order>(cos(z2));
			else if(formula == 45) return mcdv.dv<order>(tan(z2));
			else if(formula == 46) return mcdv.dv<order>(sinh(z2));
			else if(formula == 47) return mcdv.dv<order>(cosh(z2));
			else if(formula == 48) return mcdv.dv<order>(tanh(z2));
			else if(formula == 49) return mcdv.dv<order>(atanh(z2));
			else if(formula == 50) return mcdv.dv<order>(exp(z2));
			else if(formula == 51) return mcdv.dv<order>(gamma(z2));
			else if(formula == 52) return mcdv.dv<order>(LambertW(0,z2));
			else if(formula == 53) return mcdv.dv<order>(erf(z2));
			else if(formula == 54) return mcdv.dv<order>(Riemann_Zeta(z2));
			
			else if(formula == 55) return mcdv.dv<order>(log(z3));
			else if(formula == 56) return mcdv.dv<order>(sqrt(z3));
			else if(formula == 57) return mcdv.dv<order>(sin(z3));
			else if(formula == 58) return mcdv.dv<order>(csc(z3));
			else if(formula == 59) return mcdv.dv<order>(cos(z3));
			else if(formula == 60) return mcdv.dv<order>(tan(z3));
			else if(formula == 61) return mcdv.dv<order>(sinh(z3));
			else if(formula == 62) return mcdv.dv<order>(cosh(z3));
			else if(formula == 63) return mcdv.dv<order>(tanh(z3));
			else if(formula == 64) return mcdv.dv<order>(exp(z3));
			else if(formula == 65) return mcdv.dv<order>(gamma(z3));
			else if(formula == 66) return mcdv.dv<order>(LambertW(0,z3));
			else if(formula == 67) return mcdv.dv<order>(erf(z3));
			else if(formula == 68) return mcdv.dv<order>(Riemann_Zeta(z3));
			
			else if(formula == 69) return mcdv.dv<order>(log(z4));
			else if(formula == 70) return mcdv.dv<order>(sqrt(z4));
			else if(formula == 71) return mcdv.dv<order>(sin(z4));
			else if(formula == 72) return mcdv.dv<order>(csc(z4));
			else if(formula == 73) return mcdv.dv<order>(cos(z4));
			else if(formula == 74) return mcdv.dv<order>(tan(z4));
			else if(formula == 75) return mcdv.dv<order>(sinh(z4));
			else if(formula == 76) return mcdv.dv<order>(cosh(z4));
			else if(formula == 77) return mcdv.dv<order>(tanh(z4));
			else if(formula == 78) return mcdv.dv<order>(exp(z4));
			else if(formula == 79) return mcdv.dv<order>(gamma(z4));
			else if(formula == 80) return mcdv.dv<order>(LambertW(0,z4));
			else if(formula == 81) return mcdv.dv<order>(erf(z4));
			else if(formula == 82) return mcdv.dv<order>(Riemann_Zeta(z4));
			
			else if(formula == 83) return mcdv.dv<order>(gamma(log(z2))) * i;
			else if(formula == 84) return mcdv.dv<order>(gamma(sqrt(z2))) * i;
			else if(formula == 85) return mcdv.dv<order>(gamma(sin(z2))) * i;
			else if(formula == 86) return mcdv.dv<order>(gamma(cos(z2))) * i;
			else if(formula == 87) return mcdv.dv<order>(gamma(tan(z2))) * i;
			else if(formula == 88) return mcdv.dv<order>(gamma(sinh(z2))) * i;
			else if(formula == 89) return mcdv.dv<order>(gamma(cosh(z2))) * i;
			else if(formula == 90) return mcdv.dv<order>(gamma(tanh(z2))) * i;
			else if(formula == 91) return mcdv.dv<order>(gamma(exp(z2))) * i;
			else if(formula == 92) return mcdv.dv<order>(gamma(LambertW(0,z2))) * i;
			
			
			else return mcdv.dv<order>(gamma(log(z2)));
		};

#endif //__ELEMENTRY_FUNCTIONS_HPP__