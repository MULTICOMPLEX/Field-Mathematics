/// flatten class

#include <vector>

template <typename elem, int order>
  requires ((order >= 0) && (order < 25))
class multicomplex;

template <typename elem, int order>
class flat 
{
  
public:

  std::vector<elem> vec;

  const int raw_size {2 << order};

  template <int x_order>
  void help_lower (multicomplex<elem,x_order> const & mc) 
  {
    help_lower<x_order-1> (mc.real);//recursion
    help_lower<x_order-1> (mc.imag);//recursion
  }

  template <int x_order>
  void help_lower (multicomplex<elem,0> const & mc) 
  {
    vec.push_back (mc.real);
    vec.push_back (mc.imag);
  }

  template <int x_order> 
  typename std::enable_if <(x_order > 0), multicomplex<elem,x_order>>::type
  help_raise (size_t const lo, size_t const hi)
  {
    size_t const md {lo + (hi - lo) / 2};

    return 
    {
      help_raise<x_order-1> (lo, md),
      help_raise<x_order-1> (md, hi)
    };
  }

  template <int x_order> 
  typename std::enable_if <(x_order == 0), multicomplex<elem,x_order>>::type
  help_raise (size_t const lo, size_t const hi)
  {
    return {vec.at(lo), vec.at(lo+1)};
  }

  public:
  std::vector<elem> v;
  flat () = delete;
  flat (std::initializer_list<elem> l) : v(l){};
  flat (flat const &) = default;
  flat (flat &&) = default;
  flat & operator= (flat const &) = default;
  flat & operator= (flat &&) = default;
  ~flat () = default;

  flat (multicomplex<elem,order> const & mc) 
  {
    vec.reserve (raw_size);
    help_lower<order> (mc);
  }

  elem & at (int const j)
  { return vec.at(j); }

  int size () const
  { return raw_size; }

  operator multicomplex<elem,order> () 
  { return help_raise<order> (0, raw_size); }

};