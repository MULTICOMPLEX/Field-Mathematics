// prime for derivative complex

//https://en.wikipedia.org/wiki/Numerical_differentiation

template <typename elem, int order>
  requires ((order >= 0) && (order < 25))
class multicomplex;

template <typename elem, int order>
multicomplex<elem, order+1> di
(
  const multicomplex<elem, order>& x
)
{
  multicomplex<elem, order+1> z;
  
  flat a = z;
  
  a.at(0) = x.real;
  a.at(1) = x.imag;
  
  for(int j=2; j < a.size(); j*=2)
    a.at(j) = lambda;// 2 4 8 16 32 64...
    
  z = a;
  
  return z;
}

template <typename elem, int order>
void sh
(
  multicomplex<elem, order>& z,
  const multicomplex<elem, 0>& x
)
{
  flat a = z;
  
  a.at(0) = x.real;
  a.at(1) = x.imag;
  
  for(int j=2; j < a.size(); j*=2)
    a.at(j) = lambda;// 2 4 8 16 32 64...
    
  z = a;
}

/// retrieve derivative complex

template <typename elem, int order>
multicomplex<elem, 0> dv
(
  const multicomplex<elem, order>& z
)
{ 
  flat a = z;
  auto s = a.size();

  multicomplex<elem, 0> k = {a.at(s-2), a.at(s-1)};
  return k / std::pow(lambda,order);
}

//multicomplex derivatives
class mcdv
{
public :
  
  mcdv()=default;
  virtual ~mcdv()=default;
  
  template <int dv_order, typename elem, int order>  
  void sh
  (
    multicomplex<elem, order>& z,
    const multicomplex<elem, dv_order>& x
  )
  {
    static_assert(order >= dv_order);
    static_assert(dv_order >= 0);
  
    flat a = z;
    flat b = x;
  
    int e = int(std::pow(2,dv_order+1));
  
    for(int i = 0; i < e; i++)
      a.at(i) = b.at(i);
  
    for(int j=e; j < a.size(); j*=2)
      a.at(j) = lambda;// 2 4 8 16 32 64...
    
    z = a;
  }
  
  template <int dv_order, typename elem, int order>  
  multicomplex<elem, dv_order> dv
  (
    const multicomplex<elem, order>& z
  )
  { 
    flat a = z;
    auto s = a.size();
  
    int e = int(std::pow(2,dv_order+1));
  
    multicomplex<elem, dv_order> r;
    
    flat k = r;
  
    for(int i = 0; i < e; i++)  
      k.at(i) = a.at(s-i-1);
  
    flat j = k;
    for(int i = 0; i < e; i++)  
      k.at(i) = j.at(e-i-1);
  
    r = k;
    
    return r / std::pow(lambda,order-dv_order);
  }
   
};

/// get first 

template <typename elem, int order>
elem gr
(
  const multicomplex<elem, order>& z
)
{
  flat a = z;
  return a.at(0);
}