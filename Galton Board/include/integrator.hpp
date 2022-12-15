///integrator 

#include <functional>
#include <hep/mc.hpp>

template <typename elem, int order>
	requires ((order >= 0) && (order < 25))
class multicomplex;

//https://en.wikipedia.org/wiki/Numerical_integration

template <typename elem, int order>
class integrator
{
public:
	std::vector<elem> weights;
	std::vector<elem> points;
	
	integrator()
	{
		max_level = 11;
		initial_width = half;
		final_width = ldexp(initial_width, -max_level + 1);
		table_size = static_cast<int>(2.0 * 7.0 / final_width);
		weights.resize(table_size);
		points.resize(table_size);
		init_table();
	}
	
	virtual ~integrator(){}

private:
	int max_level;
	elem initial_width, final_width;
	size_t table_size;
	
	/* Pre-computed quadrature points. */

	/* Scales and translates the given function f from the
		 interval  [a, b] to [-1, 1] so it can be evaluated using
		 the tanh-sinh substitution.                             */

	class UnitFunction
	{
	private:
		std::function<multicomplex<elem, order>(const multicomplex<elem, order>&)> f;
		multicomplex<elem, order> offset, h;
	public:

		UnitFunction
		(
			std::function<multicomplex<elem, order>(const multicomplex<elem, order>&)> f,
			const multicomplex<elem, order>& a,
			const multicomplex<elem, order>& b
		) : f(f) {
			offset = half * (a + b);
			h = (b - a) * half;
		}

		multicomplex<elem, order> operator()(multicomplex<elem, order> x) const {
			return f(offset + h * x) * h;
		}
	};

	/* Initializes the weight and abcissa table. */
	void init_table()
	{
		elem h = initial_width * 2;
		elem dt;
		elem t;
		int i = 0;
		elem sinh_t, cosh_t, cosh_s;
		elem x, w;
		for (int level = 1; level <= max_level; level++, h *= half) {
			t = h * half;
			dt = (level == 1) ? t : h;
			for (;; t += dt) {

				cosh_t = std::cosh(t);
				sinh_t = std::sinh(t);

				cosh_s = std::cosh(sinh_t);

				x = std::tanh(sinh_t);

				w = (cosh_t) / (cosh_s * cosh_s);

				if (x == 1.0 || w < 0) {
					weights[i++] = 0;
					break;
				}

				points[i] = x;
				weights[i] = w;
				i++;
			}
		}
	}

	/* Computes the integral of the function f from a to b. */

	template<typename function_type>
	int integrate
	(
		function_type f,
		const multicomplex<elem, order>& a,
		const multicomplex<elem, order>& b,
		const elem& tol,
		multicomplex<elem, order>& result,
		elem& err
	)
	{
		if (a.real == -1.0 && b.real == 1.0)
			return integrate_u(f, tol, result, err);
		else {
			UnitFunction unit_f(f, a, b);
			return integrate_u(unit_f, tol, result, err);
		}
	}

	/* Computes the integral of the function f from -1 to 1. */
	int integrate_u
	(
		std::function<multicomplex<elem, order>(const multicomplex<elem, order>&)> f,
		elem tol,
		multicomplex<elem, order>& result, elem& err
	)
	{
		multicomplex<elem, order> r1, r2, r3, s = f(0);
		elem x, w;
		int level;
		elem h = initial_width;
		bool conv = false;
		int i = 0;

		for (level = 1; level <= max_level; level++, h *= half) {

			/* Compute the integral */
			for (;;) {
				x = points[i];
				w = weights[i];
				i++;

				if (w == 0) break;

				s += w * (f(x) + f(-x));
			}

			r1 = s * h;

			/* Check for convergence. */
			if (level > 2) {
				elem e1, e2, d1, d2;

				e1 = std::abs(r1.norm() - r2.norm());
				if (e1 == 0)
					err = 0;
				else {
					e2 = std::abs(r1.norm() - r3.norm());
					d1 = std::log(e1);
					d2 = std::log(e2);

					err = std::exp(d1 * d1 / d2);
				}

				// std::cout << " level = " << level << std::endl;
				// std::cout << "     r = " << r1 << std::endl;
				// std::cout << "   err = " << err << std::endl;
				// std::cout << "     i = " << i << std::endl;

				if (err < std::sqrt(r1.norm()) * tol) { // sqrt(r1.norm())
					conv = true;
					break;
				}
			}

			r2 = r1;
			r3 = r2;
		}

		if (level > max_level)
			puts("Level exhausted.");

		result = r1;
		if (!conv) {
			/* No convergence. */
			return -1;
		}

		return 0;
	}

public:

	template<typename function_type>
	const multicomplex<elem, order> ix
	(
		function_type func,
		const multicomplex<elem, order>& a,
		const multicomplex<elem, order>& b
	)
	{
		elem tol = 1e-8;
		multicomplex<elem, order> result;
		elem err;

		integrate(func, a, b, tol, result, err);

		return result;
	}

};

namespace ps
{

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

	template <typename elem, int order>
	const multicomplex<elem, order> Ei
	(
		const multicomplex<elem, order>& z
	)//ExpIntegralEi[x]
	{
		multicomplex<elem, order> c{};

		for (int k = 1; k < 20; k++) {
			c = c + (pow(z, k) / (factorial<elem>(k) * elem(k)));
		}

		return euler + log(z) + c;
	}

	template <typename elem, int order>
	const multicomplex<elem, order> li
	(
		const multicomplex<elem, order>& z
	)// LogIntegral[z] 
	{
		return  Ei(log(z));
	}

	template <typename elem, int order>
	const multicomplex<elem, order> E1
	(
		const multicomplex<elem, order>& z
	)//ExpIntegralE[1, z]
	{
		multicomplex<elem, order> c{};

		for (int k = 1; k < 20; k++) {
			c = c + (pow(-1, k) * pow(z, k)) / (factorial<elem>(k) * k);
		}

		return -euler - log(z) - c;
	}

	template <typename elem, int order>
	const multicomplex<elem, order> Shi
	( //The hyperbolic sine integral
		const multicomplex<elem, order>& z
	)//SinhIntegral[z] 
	{
		multicomplex<elem, order> c{};

		for (size_t k = 0; k < 20; k++) {
			c = c + pow(z, 2 * k + 1) / (pow(2 * k + 1, 2) * (factorial<elem>(2 * k)));
		}

		return c;
	}

	template <typename elem, int order>
	const multicomplex<elem, order> Chi
	( //The hyperbolic sine integral
		const multicomplex<elem, order>& z
	)//CoshIntegral[z] 
	{
		return -half * (E1(-z) + E1(z) + log(-z) - log(z));
	}

	template <typename elem, int order>
	const multicomplex<elem, order> e1
	(
		const multicomplex<elem, order>& z
	)
	{
		return -Ei(-z);
	}

}

template<typename function_type, typename elem, int order>
const multicomplex<elem, order> midpoint //Generalized midpoint rule formula
(
	function_type func,
	const multicomplex<elem, order>& a,
	const multicomplex<elem, order>& b
)
{
	int64_t M = 40, tel = 0;

	 mcdv mcdv;

	 multicomplex<elem,order+2> d2;
	 multicomplex<elem,order+4> d4;
	 multicomplex<elem,order+6> d6;

	multicomplex<elem, order> r, c;

	for (int64_t m = 1; m <= M; m++)
	{
		multicomplex<elem, order> mr = (m - half) / M;

		mr = (b - a) * mr + a;

		for (int64_t n = 0; n <= tel; n += 2)
		{
			if (n == 0) { r = func(mr); }
			if(n == 2){mcdv.sh<order>(d2, mr); r = mcdv.dv<order>(func(d2));}
			if(n == 4){mcdv.sh<order>(d4, mr); r = mcdv.dv<order>(func(d4));}
			if(n == 6){mcdv.sh<order>(d6, mr); r = mcdv.dv<order>(func(d6));}

			c += (elem(std::pow(-1, n) + 1) / elem(std::pow(2 * M, n + 1) * ps::factorial<elem>(n + 1))) * r;
		}
	}

	return (b - a) * c;
}

template<typename function_type, typename elem, int order>
const multicomplex<elem, order> newtonCotes
(
	function_type function,
	multicomplex<elem, order> a,
	multicomplex<elem, order> b,
	int n
)
{
	multicomplex<elem, order> c0 = (elem)2 / 45;
	int w[5] = { 7,32,12,32,7 };
	multicomplex<elem, order> h = (b - a) / n;
	multicomplex<elem, order> answ = 0;
	multicomplex<elem, order>* x = new multicomplex<elem, order>[n + 1];

	for (int i = 0; i < n + 1; i++)
		x[i] = a + h * i;

	for (int j = 0; j < n; j += 4)
		for (int i = 0; i < 5; i++)
			answ += w[i] * function(x[j] + 4 * h * i);
	return c0 * h * answ;
}

///////////////////

template <typename T>
T denormalize(T normalized, T min, T max) {
	auto denormalized = (normalized * (max - min) + min);
	return denormalized;
}

///////////////////

// the function that shall be integrated
template<typename T>
T square(hep::mc_point<T> const& x)
{
	const extern T x_max;
	const extern T x_min;
	auto p = (x_max - x_min) * x.point()[0] + x_min;
	return exp(1 - p * p);
}

template<typename T>
void hep_driver(T min, T max)
{
	// print reference result
	std::cout << ">> computing integral of exp(1-x*x) from " + std::to_string(min) + " to " + std::to_string(max) + " \n\n";

	// perform 5 iteration with 1000 calls each; this function will also call
	// vegas_verbose_callback after each iteration which in turn prints the
	// individual iterations

	auto results = hep::vegas(
		hep::make_integrand<T>(square<T>, 1),
		std::vector<std::size_t>(5, 1000000)
	).results();

	// results contains the estimations for each iteration. We could take the
	// result from last iteration, but here we instead choose to combine the
	// results of all iterations but the first one in a cumulative result
	auto result = hep::accumulate<hep::weighted_with_variance>(
		results.begin() + 1, results.end());
	auto chi_square_dof = hep::chi_square_dof<hep::weighted_with_variance>(
		results.begin() + 1, results.end());

	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	std::cout.precision(12);

	// print the cumulative result
	std::cout << ">> cumulative result (excluding first iteration):\n>> N="
		<< result.calls() << " I=" << (max - min) * result.value() << " +- " << result.error()
		<< " chi^2/dof=" << chi_square_dof << "\n\n";
}

//////////////

template<typename T>
T Wilkinsons_polynomial(const T& x, int n)
{
	T r = 1;
	for (int m = 1; m <= n; m++)
		r *= x - m;
	return r;
}

template <typename F, typename elem, int order>
multicomplex<elem, order> Generalized_midpoint
(
	F func,
	const multicomplex<elem, order>& A,
	const multicomplex<elem, order>& B,
	const size_t M,
	const size_t N
)
{
	multicomplex<elem, order + 2> d1;
	multicomplex<elem, order + 4> d2;
	multicomplex<elem, order + 6> d3;
	multicomplex<elem, order + 8> d4;
	multicomplex<elem, order + 10> d5;

	multicomplex<elem, order> sum = 0;

	for (size_t m = 1; m <= M; m++) {

		MX0 x = (elem(m) - 0.5) / elem(M);
		for (size_t n = 0; n <= N; n++) {

			elem a = pow(2 * M, 2 * n + 1) * Fac<elem>(2 * n + 1);

			if (n == 0) { sum += func((B - A) * x + A, (B - A) * x + A) / a; }
			else if (n == 1) {
				sh(d1, x); sum += dv(func((B - A) * d1 + A, (B - A) * x + A)) / a;
			}
			else if (n == 2) {
				sh(d2, x); sum += dv(func((B - A) * d2 + A, (B - A) * x + A)) / a;
			}

			else if (n == 3) {
				sh(d3, x); sum += dv(func((B - A) * d3 + A, (B - A) * x + A)) / a;
			}

			else if (n == 4) {
				sh(d4, x); sum += dv(func((B - A) * d4 + A, (B - A) * x + A)) / a;
			}

			else {
				sh(d5, x); sum += dv(func((B - A) * d5 + A, (B - A) * x + A)) / a;
			}

		}
	}
	return (B - A) * 2 * sum;
}

template <typename F, typename elem>
elem Generalized_midpoint
(
	F func,
	const elem& A,
	const elem& B,
	const elem& A2,
	const elem& B2,
	const size_t M,
	const size_t N
)
{
	MX2 d1;
	MX4 d2;
	MX6 d3;
	MX8 d4;
	MX10 d5;

	elem sum = 0;

	for (size_t m = 1; m <= M; m++) {

		MX0 x = (elem(m) - 0.5) / elem(M);
		for (size_t n = 0; n <= N; n++) {

			elem a = pow(2 * M, 2 * n + 1) * Fac<elem>(2 * n + 1);

			if (n == 0) { sum +=   func((B - A) *  x + A, (B2 - A2) * x + A2).real / a; }
			else if (n == 1) {
				sh(d1, x); sum += dv(func((B - A) * d1 + A, (B2 - A2) * x + A2)).real / a;
			}
			else if (n == 2) {
				sh(d2, x); sum += dv(func((B - A) * d2 + A, (B2 - A2) * x + A2)).real / a;
			}

			else if (n == 3) {
				sh(d3, x); sum += dv(func((B - A) * d3 + A, (B2 - A2) * x + A2)).real / a;
			}

			else if (n == 4) {
				sh(d4, x); sum += dv(func((B - A) * d4 + A, (B2 - A2) * x + A2)).real / a;
			}

			else {
				sh(d5, x); sum += dv(func((B - A) * d5 + A, (B2 - A) * x + A)).real / a;
			}

		}
	}
	return (B - A) * 2 * sum;
}