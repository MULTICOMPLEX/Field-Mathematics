
template<typename value_type, typename function_type>
value_type integral(const value_type a,
                           const value_type b,
                           const value_type tol,
                           function_type func)
{
   unsigned n = 1U;

   value_type h = (b - a);
   value_type I = (func(a) + func(b)) * (h / 2);

   for(unsigned k = 0U; k < 8U; k++)
   {
      h /= 2;

      value_type sum(0);
      for(unsigned j = 1U; j <= n; j++)
      {
         sum += func(a + (value_type((j * 2) - 1) * h));
      }

      const value_type I0 = I;
      I = (I / 2) + (h * sum);

      const value_type ratio     = I0 / I;
      const value_type delta     = ratio - 1;
      const value_type delta_abs = ((delta < 0) ? -delta : delta);

      if((k > 1U) && (delta_abs < tol))
      {
         break;
      }

      n *= 2U;
   }

   return I;
}

//Bisection method
template<typename value_type, typename function_type>
   value_type derivative(const value_type x, const value_type dx, function_type func)
{
   // Compute d/dx[func(*first)] using a three-point
   // central difference rule of O(dx^6).

   const value_type dx1 = dx;
   const value_type dx2 = dx1 * 2;
   const value_type dx3 = dx1 * 3;

   const value_type m1 = (func(x + dx1) - func(x - dx1)) / 2;
   const value_type m2 = (func(x + dx2) - func(x - dx2)) / 4;
   const value_type m3 = (func(x + dx3) - func(x - dx3)) / 6;

   const value_type fifteen_m1 = 15 * m1;
   const value_type six_m2     =  6 * m2;
   const value_type ten_dx1    = 10 * dx1;

   return ((fifteen_m1 - six_m2) + m3) / ten_dx1;
}

template<typename value_type>
class cyl_bessel_j_integral_rep
{
public:
   cyl_bessel_j_integral_rep(const unsigned N,
      const value_type& X) : n(N), x(X) { }

   value_type operator()(const value_type& t) const
   {
      // pi * Jn(x) = Int_0^pi [cos(x * sin(t) - n*t) dt]
      return cos(x * sin(t) - (n * t));
   }

private:
   const unsigned n;
   const value_type x;
};

template <typename T> 
void driver_classicdifint()
{  
   double h = std::numeric_limits<double>::epsilon();
 
   //
   // In the double case, the sin function is multiply overloaded
   // to handle expression templates etc.  As a result it's hard to take its
   // address without knowing about its implementation details.  We'll use a 
   // C++11 lambda expression to capture the call.
   // We also need a typecast on the first argument so we don't accidentally pass
   // an expression template to a template function:
   //
 /*  
   const double d_mp = derivative(
      double(pi<double>() / 3),
      double(1e-6),
      [](const double& x) -> double
      {
         return sin(x);
      }
      );
   
   
    // 0.50000000000000
   std::cout
      << std::setprecision(std::numeric_limits<double>::digits10)
      << d_mp
      << std::endl;
 */   
    //////////
    
   const auto j2_mp =
      integral(-.1, 1., 
      h,
      
      [](const auto& x) -> auto
      {
         return log(x);
      });
      
      //cyl_bessel_j_integral_rep<double>(2U, double(123) / 100)
      
      //) / pi<double>();
   
   // 0.166369383786814
   std::cout
      << std::setprecision(std::numeric_limits<double>::digits10)
      << j2_mp
      << std::endl;
      
}


template <typename F1, typename elem>
elem funcr(F1 f1, elem& x) 
{   
  return f1(x);
  //return pow(x,2)-9*x+20;
  //Abel–Ruffini theorem
  //https://en.wikipedia.org/wiki/Abel%E2%80%93Ruffini_theorem
  //return pow(x,5)-x-1;//https://www.wolframalpha.com/input/?i=x%5E5-x-1%3D0 , Abel–Ruffini theorem
  //return x * (exp(x)-MX0(1));//black body radiation, roots 2 i π n, n element Z
  //return 1-exp(x); ////capacitor inductor, roots 2 i π n, n element Z
 // return exp(x) - x; ////semiconductor, roots -W_n(-1), n element Z
} 

template <typename F1, typename elem>
elem derivFuncr(F1 f1, elem& z) 
{   
  elem vv = 1e-6;
  return derivative(z,vv,f1);
  
  //return dv(pow(x,2)-9*x+20);
  //return  dv(pow(x,5)-x-1);
  //return  dv(x * (exp(x)-MX0(1)));
  //return dv(MX0(1)-exp(x));
 // return dv(exp(x) - x);
} 

template <typename F1, typename elem>
elem rootr(F1 f1, elem& x) 
{ 
  int tel=0;
  elem h; 
  while (tel < 15)
  { 
      h = funcr(f1,x)/(derivFuncr(f1,x)); 
   
      // x(i+1) = x(i) - f(x) / f'(x)   
      x -= h;

      printf("%02d ", tel);
      std::cout << x << std::endl;
      tel++;
  }

 // cout << ", The value of the root is : " << x << ",Nsteps : " << tel << endl;
  return x;
}