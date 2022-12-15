
template <typename elem, int order>
  requires ((order >= 0) && (order < 25))
class multicomplex;

template <typename F1, typename elem, int order>
multicomplex<elem, order> func(F1 f1, multicomplex<elem, order>& x) 
{   
  return f1(x);
  //return pow(x,2)-9*x+20;
  //Abel–Ruffini theorem
  //return pow(x,5)-x-1;
  //return x * (exp(x)-1);//black body radiation, roots 2 i π n, n element Z
  //return 1-exp(x); ////capacitor inductor, roots 2 i π n, n element Z
 // return exp(x) - x; ////semiconductor, roots -W_n(-1), n element Z
} 

template <typename F1, typename elem, int order>
multicomplex<elem, order> derivFunc(F1 f1, multicomplex<elem, order>& z) 
{   
  mcdv mcdv; 
  
  multicomplex<elem, order+1> x;
  //sh(x, z);
  mcdv.sh<order>(x, z);
  
  //return dv(f1(x));
  return mcdv.dv<order>(f1(x));
  
  //return dv(pow(x,2)-9*x+20);
  //return dv(pow(x,5)-x-1);
  //return dv(x * (exp(x)-1));
  //return dv(MX0(1)-exp(x));
 // return dv(exp(x) - x);
} 

template <typename F1, typename elem, int order>
multicomplex<elem, order> derivFunc2(F1 f1, multicomplex<elem, order>& z)
{
  multicomplex<elem, order + 2> x;
  mcdv mcdv;
  mcdv.sh<order>(x, z);
  return mcdv.dv<order>(f1(x));
}

template <typename F1, typename elem, int order>
multicomplex<elem, order> derivFunc3(F1 f1, multicomplex<elem, order>& z)
{
  multicomplex<elem, order + 3> x;
  sh(x, z);
  return dv(f1(x));
}

template <typename F1, typename elem, int order>
multicomplex<elem, order> root(F1 f1, multicomplex<elem, order>& x, int n) 
{ 
  int tel=0;
  multicomplex<elem, order> h; 

  while (tel < n)
  { 
      h = func(f1,x)/(derivFunc(f1,x)); 
   
      // x(i+1) = x(i) - f(x) / f'(x)   
      x -= h;
      
      printf("%02d ", tel);
      std::cout << x << std::endl;
      tel++;
  }

 // cout << ", The value of the root is : " << x << ",Nsteps : " << tel << endl;
  return x;
}

//root finding on the first derivative.
//https://blog.demofox.org/2020/03/17/basic-methods-for-finding-zeroes-and-mins-maxes-of-functions/
//https://en.wikipedia.org/wiki/Halley%27s_method
//https://en.wikipedia.org/wiki/Householder%27s_method
template <typename F1, typename elem, int order>
multicomplex<elem, order> Critical_point(F1 f1, multicomplex<elem, order>& x, int n)
{
  int tel = 0;
  multicomplex<elem, order> h;

  while (tel < n)
  {
    h = derivFunc(f1, x) / derivFunc2(f1, x);

    // x(i+1) = x(i) - f'(x) / f''(x)   
    x -= h;

    printf("%02d ", tel);
    std::cout << x << std::endl;
    tel++;
  }

  return x;
}