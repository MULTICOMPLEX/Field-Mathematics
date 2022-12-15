
#pragma once

namespace mcp{

const REAL N = 0;
const REAL h = std::numeric_limits<REAL>::epsilon();
//const long double h = 1e-10;
//const long double h = 6.62607015e-34; //Planck constant ℏ

///////real output
inline mcd0 r(const mcd0& x)
{ 
  return x; 
}

template <typename elem, int order>
inline mcd0 r(const multicomplex<elem, order>& x)
{ 
  flat a = x;
  auto s = (sizeof(x)/sizeof(elem)) - 1;
  
  mcd0 k = {a.at(s)};
  return k / pow(h,order);
}

///////complex output
inline mcd1 c(const mcd0& x)
{ 
  return {x, N}; 
}

mcd1 c(const mcd1& x)
{ 
  return x; 
}

template <typename elem, int order>
multicomplex<elem, 1> c(const multicomplex<elem, order>& x)
{ 
  flat<elem,order> a = x;
  int s = (sizeof(x)/sizeof(elem)) - 1;
  multicomplex<elem, 1> k = {a.at(s-1), a.at(s)};
  return k / pow(h,order-1);
}

///////

template <typename elem, int order>
multicomplex<elem, 0> gtr(const multicomplex<elem, order>& x)
{ 
  flat<elem,order> a = x; 
  return  a.at(0);
}

////
inline mcd0 gti(const mcd0& x)
{ 
  return x; 
}

template <typename elem, int order>
inline mcd0 gti(const multicomplex<elem, order>& x)
{ 
  flat a = x;
  return  a.at(1);
}

////
inline mcd1 gtc(const mcd0& x)
{ 
  return {x, N}; 
}

template <typename elem, int order>
inline mcd1 gtc(const multicomplex<elem, order>& x)
{ 
  flat a = x;  
  mcd1 k = {a.at(0), a.at(1)};
  return k;
}
////


////// real input

void seth(mcd0& z, const mcd0& x)
{ 
  z =  {x}; 
}

void seth(mcd1& z, const mcd0& x)
{ 
  z = {x, h}; 
}

void seth(mcd7& z, const mcd0& x)
{ 
  z = {{{{{{{x, h}, {h, N}}, {{h, N}, {N, N}}},{{{h, N}, {N, N}}, {{N, N}, {N, N}}}},
         {{{{h, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}},
        {{{{{h, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}},
         {{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}}},
       {{{{{{h, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}},
         {{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}},
        {{{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}},
         {{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}}}}; 
}

template <typename elem, int order>
inline void seth(multicomplex<elem, order>& z,const mcd0& x)//Tinit
{
  flat a = z;
  
  a.at(0) = x.sole;
  a.at(1) = h;
  
  for(int j=2; j < std::pow(2,order); j*=2)
  {
    a.at(j) = h;// 2 4 8 16 32 64 128 256 512 1024 2048 4096
  }
  
  z = multicomplex<elem, order>(a);
}

////// complex input

void seth(mcd0& z, const mcd1& x)
{ 
  z = x.real; 
}

void seth(mcd1& z, const mcd1& x)
{ 
  z = x; 
}

inline void seth(mcd7& z, const mcd1& x)
{ 
  z = {{{{{{x,     {h, N}}, {{h, N}, {N, N}}},{{{h, N}, {N, N}}, {{N, N}, {N, N}}}},
        {{{{h, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}},
       {{{{{h, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}},
        {{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}}},
      {{{{{{h, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}},
        {{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}},
       {{{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}},
        {{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}}}}; 
}

template <typename elem, int order>
void seth(multicomplex<elem, order>& z,const mcd1& x)//Tinit
{
  flat a = z;
  
  a.at(0) = x.real.sole;
  a.at(1) = x.imag.sole;
  
  for(int j=2; j < std::pow(2,order); j*=2)
  {
    a.at(j) = h;// 2 4 8 16 32 64
  }
  
  z = multicomplex<elem, order>(a);
}


///////

void sethb(mcd2& z, const mcd2& x)
{ 
  z = x; 
}

void sethb(mcd7& z, const mcd2& x)
{ 
  z =   {{{{{x,              {{h, N}, {N, N}}},{{{h, N}, {N, N}}, {{N, N}, {N, N}}}},
         {{{{h, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}},
        {{{{{h, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}},
         {{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}}},
       {{{{{{h, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}},
         {{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}},
        {{{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}},
         {{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}}}}; 
}

template <typename elem, int order>
inline void sethb(multicomplex<elem, order>& z,const mcd2& x)//Tinit
{
  flat a = z;
  flat b = x;
  
  for(int j=0; j < 4; j++)
    a.at(j) = b.at(j);
  
  for(int j=4; j < std::pow(2,order); j*=2)
  {
    a.at(j) = h;// 4 8 16 32 64
  }
  
  z = multicomplex<elem, order>(a);
}

mcd2 cb(const mcd3& x)
{ 
  return x.imag / (h); 
}

mcd2 cb(const mcd4& x)
{ 
  return x.imag.imag / (h*h); 
}

mcd2 cb(const mcd5& x)
{ 
  return x.imag.imag.imag / (h*h*h); 
}

mcd2 cb(const mcd6& x)
{ 
  return x.imag.imag.imag.imag / (h*h*h*h); 
}

mcd2 cb(const mcd7& x)
{ 
  return x.imag.imag.imag.imag.imag / (h*h*h*h*h); 
}

/////////////

void setht(mcd3& z, const mcd3& x)
{ 
  z = x; 
}

void setht(mcd7& z, const mcd3& x)
{ 
z =      {{{{x, {{{h, N}, {N, N}}, {{N, N}, {N, N}}}},
         {{{{h, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}},
        {{{{{h, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}},
         {{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}}},
       {{{{{{h, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}},
         {{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}},
        {{{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}},
         {{{{N, N}, {N, N}}, {{N, N}, {N, N}}},{{{N, N}, {N, N}}, {{N, N}, {N, N}}}}}}}; 
}

template <typename elem, int order>
void setht(multicomplex<elem, order>& z,const mcd3& x)//Tinit
{
  flat a = z;
  flat b = x;
  
  for(int j=0; j < 8; j++)
    a.at(j) = b.at(j);
  
  for(int j=8; j < std::pow(2,order); j*=2)
  {
    a.at(j) = h;// 8 16 32 64
  }
  
  z = multicomplex<elem, order>(a);
}

mcd3 ct(const mcd4& x)
{ 
  return x.imag / (h); 
}

mcd3 ct(const mcd5& x)
{ 
  return x.imag.imag / (h*h); 
}

mcd3 ct(const mcd6& x)
{ 
  return x.imag.imag.imag / (h*h*h); 
}

mcd3 ct(const mcd7& x)
{ 
  return x.imag.imag.imag.imag / (h*h*h*h); 
}

/////////

inline void str(mcd0& x, const mcd0& c)
{
  x.at_var<0>(0) = c.sole;
}

template <typename elem, int order>
inline void str(multicomplex<elem, order>& x, const mcd0& c)
{ 
  flat a = x;  
  a.at(0) = c.sole;
  x = multicomplex<elem, order>(a);
}

/////////

inline void stc(mcd0& x, const mcd1& c)
{
  x.at_var<0>(0) = c.real.sole;
}

template <typename elem, int order>
inline void stc(multicomplex<elem, order>& x, const mcd1& c)
{ 
  flat a = x;  
  a.at(0) = c.real.sole;
  a.at(1) = c.imag.sole;
  x = multicomplex<elem, order>(a);
}

}

