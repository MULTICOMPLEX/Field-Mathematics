
#ifndef __MXWS_HPP__
#define __MXWS_HPP__

#ifndef __x86_64
#define __x86_64 1
#endif
#include "../pcg_random.hpp"

#include <random>
#include <numbers>
#include <iostream>
#include "../ziggurat.hpp"

 
template <typename RN>
	requires
std::same_as<RN, uint32_t> ||
std::same_as<RN, uint64_t>
class mxws
{

private:

	std::random_device r;
	pcg_extras::seed_seq_from<std::random_device> seed_source;

public:

	uint64_t x, w, x1, x2, w1, w2;
	std::mt19937 MT;  //alternative
	pcg32 PCG32; //alternative

	typedef RN result_type;

	void seed()
	{
		init();
		x1 = x2 = 1;
	}

	void seed(uint64_t x)
	{
		w = x;
		x = 1;
		w1 = w;
		w2 = w1 + 1;
		x1 = x2 = 1;
	}

	template <typename T>
		requires	std::same_as<T, std::seed_seq>
	mxws(const T& seq)
	{
		if (seq.size() == 2)
		{
			std::vector<uint32_t> seeds(seq.size());
			seq.param(seeds.rbegin());
			w = (uint64_t(seeds[1]) << 32) | seeds[0];
			x = 1;

			w1 = w;
			w2 = w1 + 1;
			x1 = x2 = 1;
		}

		else init();
	}

	mxws()
	{
		init();
	}

	template <typename T>
		requires	std::same_as<T, uint64_t>
	mxws(const T& seed)
	{
		init();
	}

	void init()
	{
		w = (uint64_t(r()) << 32) | r();
		x = 1;
		w1 = w;
		w2 = w1 + 1;
		x1 = x2 = 1;

		MT.seed(r());
		PCG32.seed(seed_source);
		sw = r();
		s1w = r();
		s2w = r();
	}

	template <typename T>
		requires	std::same_as<T, uint64_t>
	void init(const T& seed)
	{
		w = seed;
		x = 1;

		MT.seed(r());
		PCG32.seed(seed_source);
		sw = r();
		s1w = r();
		s2w = r();
	}

	virtual ~mxws() = default;

	static constexpr RN min() { return std::numeric_limits<RN>::min(); }
	static constexpr RN max() { return std::numeric_limits<RN>::max(); }

	inline RN operator()()
		requires
	std::same_as<RN, uint32_t>
	{
		w += x = std::rotr(x *= w, 32);
		return RN(x);
	}

	inline RN operator()()
		requires
	std::same_as<RN, uint64_t>
	{
		x1 *= w1;
		x1 = std::rotr(x1, 32);
		w1 += x1;

		x2 *= w2;
		x2 = std::rotr(x2, 32);
		w2 += x2;

		return (x1 << 32) | uint32_t(x2);
	}

	template <typename T>
		requires std::floating_point<T>&&
	std::same_as<RN, uint64_t>
		inline T operator()(const T& f)
	{
		return (rng() >> 11) / T(9007199254740992) * f;
	}

	template <typename T>
		requires std::floating_point<T>&&
	std::same_as<RN, uint32_t>
		inline T operator()(const T& f)
	{
		return rng() / T(4294967296) * f;
	}

	template <typename T>
		requires std::floating_point<T>
	inline T operator()(const T& min, const T& max)
	{
		return rng(1.0) * (max - min) + min;
	}

	template <typename T, typename U>
		requires std::integral<T>&& std::floating_point<U>
	inline U operator()(const T& min, const U& max)
	{
		return rng(1.0) * (max - min) + min;
	}

	template <typename T>
		requires std::integral<T>
	inline T operator()(const T& max)
	{
		return rng() % (max + 1);
	}

	template <typename T>
		requires std::integral<T>
	inline T operator()(const T& min, const T& max)
	{
		return min + (rng() % (max - min + 1));
	}

	template <typename T>
	inline T rng(const T& x)
	{
		return (*this)(x);
	}

	inline RN rng()
	{
		return (*this)();
	}

	template <typename T>
		requires std::floating_point<T>
	T round_to_half(T in)
	{
		in *= 2; in = round(in); return in /= 2;
	}

	template <typename T>
		requires std::floating_point<T>
	inline T to_float(const T& in)
	{
		return in / T(4294967296);
	}

	template <typename T>
		requires std::integral<T>
	inline uint32_t to_int(const T& n)
	{
		return uint32_t(n >> 32);
	}

	template <typename T>
		requires std::floating_point<T>
	T inline normalRandom(const T& mean, const T& sigma)
	{
		// return a normally distributed random value
		T v1 = rng(1.0);
		T v2 = rng(1.0);
		return std::cos(2 * std::numbers::pi * v2) *
			std::sqrt(-2 * std::log(v1)) * sigma + mean;
	}


	cxx::ziggurat_normal_distribution<double> normalRandomZ;


	template <typename T, typename L>
		requires std::floating_point<T>&&
	std::integral<L>
		T error_function_1(const T& x, const L& iterations)
	{
		const auto f = [](const auto& t) {return exp(-pow(t, 2)); };

		T	totalSum = 0;

		T lowBound = 0, upBound = x;

		for (auto i = 0; i < iterations; i++)
			totalSum += f(rng(x));

		T estimate = (upBound - lowBound) * totalSum / iterations;

		estimate *= 2 / sqrt(std::numbers::pi);

		//std::cout << "error function(" << x << ") = " << estimate << std::endl;
		//std::cout << " std::function(" << x << ") = " << std::erf(x) << std::endl;

		return estimate;
	}

	template <typename T, typename L>
		requires std::floating_point<T>&&
	std::integral<L>
		T error_function_2(const T& x, const L& iterations)
	{
		T erf;
		L erft = 0;

		mxws <uint32_t>RNG;

		cxx::ziggurat_normal_distribution<T> normal(0, 1. / std::numbers::sqrt2);

		auto k = abs(x);

		for (L i = 0; i < iterations; i++) {

			erf = normal(RNG);

			if ((erf >= -k) && (erf <= k))
				erft++;
		}

		//std::cout << "error_function_mc(" << x << ") = " << erft / T(iterations) << std::endl;
		//std::cout << "    std::function(" << x << ") = " << std::erf(x) << std::endl;

		auto ret = erft / T(iterations);
		if (x < 0)ret = -ret;
		return ret;
	}

	template <typename T>
		requires std::floating_point<T>
	T erf_inv(T x) {

		if (x < -1 || x > 1) {
			return NAN;
		}
		else if (x == 1.0) {
			return INFINITY;
		}
		else if (x == -1.0) {
			return -INFINITY;
		}

		const long double LN2 = 6.931471805599453094172321214581e-1L;

		const long double A0 = 1.1975323115670912564578e0L;
		const long double A1 = 4.7072688112383978012285e1L;
		const long double A2 = 6.9706266534389598238465e2L;
		const long double A3 = 4.8548868893843886794648e3L;
		const long double A4 = 1.6235862515167575384252e4L;
		const long double A5 = 2.3782041382114385731252e4L;
		const long double A6 = 1.1819493347062294404278e4L;
		const long double A7 = 8.8709406962545514830200e2L;

		const long double B0 = 1.0000000000000000000e0L;
		const long double B1 = 4.2313330701600911252e1L;
		const long double B2 = 6.8718700749205790830e2L;
		const long double B3 = 5.3941960214247511077e3L;
		const long double B4 = 2.1213794301586595867e4L;
		const long double B5 = 3.9307895800092710610e4L;
		const long double B6 = 2.8729085735721942674e4L;
		const long double B7 = 5.2264952788528545610e3L;

		const long double C0 = 1.42343711074968357734e0L;
		const long double C1 = 4.63033784615654529590e0L;
		const long double C2 = 5.76949722146069140550e0L;
		const long double C3 = 3.64784832476320460504e0L;
		const long double C4 = 1.27045825245236838258e0L;
		const long double C5 = 2.41780725177450611770e-1L;
		const long double C6 = 2.27238449892691845833e-2L;
		const long double C7 = 7.74545014278341407640e-4L;

		const long double D0 = 1.4142135623730950488016887e0L;
		const long double D1 = 2.9036514445419946173133295e0L;
		const long double D2 = 2.3707661626024532365971225e0L;
		const long double D3 = 9.7547832001787427186894837e-1L;
		const long double D4 = 2.0945065210512749128288442e-1L;
		const long double D5 = 2.1494160384252876777097297e-2L;
		const long double D6 = 7.7441459065157709165577218e-4L;
		const long double D7 = 1.4859850019840355905497876e-9L;

		const long double E0 = 6.65790464350110377720e0L;
		const long double E1 = 5.46378491116411436990e0L;
		const long double E2 = 1.78482653991729133580e0L;
		const long double E3 = 2.96560571828504891230e-1L;
		const long double E4 = 2.65321895265761230930e-2L;
		const long double E5 = 1.24266094738807843860e-3L;
		const long double E6 = 2.71155556874348757815e-5L;
		const long double E7 = 2.01033439929228813265e-7L;

		const long double F0 = 1.414213562373095048801689e0L;
		const long double F1 = 8.482908416595164588112026e-1L;
		const long double F2 = 1.936480946950659106176712e-1L;
		const long double F3 = 2.103693768272068968719679e-2L;
		const long double F4 = 1.112800997078859844711555e-3L;
		const long double F5 = 2.611088405080593625138020e-5L;
		const long double F6 = 2.010321207683943062279931e-7L;
		const long double F7 = 2.891024605872965461538222e-15L;

		long double abs_x = fabsl(x);

		if (abs_x <= 0.85L) {
			long double r = 0.180625L - 0.25L * x * x;
			long double num = (((((((A7 * r + A6) * r + A5) * r + A4) * r + A3) * r + A2) * r + A1) * r + A0);
			long double den = (((((((B7 * r + B6) * r + B5) * r + B4) * r + B3) * r + B2) * r + B1) * r + B0);
			return x * num / den;
		}

		long double r = sqrtl(LN2 - logl(1.0L - abs_x));

		long double num, den;
		if (r <= 5.0L) {
			r = r - 1.6L;
			num = (((((((C7 * r + C6) * r + C5) * r + C4) * r + C3) * r + C2) * r + C1) * r + C0);
			den = (((((((D7 * r + D6) * r + D5) * r + D4) * r + D3) * r + D2) * r + D1) * r + D0);
		}
		else {
			r = r - 5.0L;
			num = (((((((E7 * r + E6) * r + E5) * r + E4) * r + E3) * r + E2) * r + E1) * r + E0);
			den = (((((((F7 * r + F6) * r + F5) * r + F4) * r + F3) * r + F2) * r + F1) * r + F0);
		}

		return copysignl(num / den, x);
	}

	template<typename T>
		requires std::floating_point<T>
	T normalCDF(const T x)
	{
		return std::erfc(-x / std::numbers::sqrt2) / 2.;
	}

	template <typename T, typename L>
		requires std::floating_point<T>&&
	std::integral<L>
		T normalCDF(const T x, const L n)
	{
		return erfc_mc(-x / std::numbers::sqrt2, n) / 2;
	}

	template <typename T, typename L>
		requires std::floating_point<T>&&
	std::integral<L>
		T erfc(const T x, const L n)
	{
		return 1. - error_function_mc2(x, n);
	}

	template<typename T>
		requires std::floating_point<T>
	T inline probit(const T& p)
	{
		return std::numbers::sqrt2 * erf_inv(2 * p - 1);
	}

	template <typename R, typename I, typename L>
		requires
	std::same_as<R, double>&&
		std::integral<I>&&
		std::same_as<L, std::uint64_t>
		std::tuple<R, I> Probability_Wave(const I& Board_SIZE,
			std::vector<L>& cycle, const L& TRIALS) {

		I Board_size;
		I rn_range;

		Board_size = I(round(std::log(Board_SIZE * 6) * std::sqrt(std::numbers::pi)));
		rn_range = I(round(Board_SIZE / std::sqrt(log2(Board_SIZE))));

		L random_walk = {};

		for (L i = 0; i < TRIALS; i++, random_walk = 0)
		{
			for (I j = 0; j < Board_size; j++)
				random_walk += rng();

			cycle[std::modulus()(random_walk * rn_range >> 32, Board_SIZE)]++;
			
		}

		return std::make_tuple(rn_range, Board_size);
	}

	template <typename I>
		requires
	std::integral<I>
		double Wave_Distribution(const I& board_SIZE)
	{
		typedef double T;

		//const I Board_size = Board_SIZE;
		//const T rn_range = std::sqrt(Board_size) + std::log(Board_size / 4));

		const I board_size = I(round(log(board_SIZE * 6) * sqrt(std::numbers::pi)));
		const T rn_range = board_SIZE / sqrt(log2(board_SIZE));

		T random_walk = 0;

		for (auto j = 0; j < board_size; j++)
			random_walk += rng(1.0);

		return fmod(random_walk * rn_range, board_SIZE);
	}

	template <typename T, typename L>
		requires std::floating_point<T>&&
	std::same_as<L, std::uint64_t>
		T sqrt(T z = 2, L throws = 10000000000)
	{
		L tel = 0, i = 0;
		T r;

		if (z < 1) {
			while (i < throws)
			{
				r = rng(1.0 / z);
				r *= r;
				if (r < z)tel++;
				i++;
			}
			return (1.0 / z) * T(tel) / throws;
		}

		else {
			while (i < throws)
			{
				r = rng(z);
				r *= r;
				if (r < z)tel++;
				i++;
			}
			return z * T(tel) / throws;
		}
	}

	template <typename T, typename L>
		requires std::floating_point<T>&&
	std::same_as<L, std::uint64_t>
		T exp(T x = 0.9, L n_samples = 10000000000)
	{
		if (x == 0) return 1;

		T h = 0;
		T xi = 1;
		L tot = 0;

		if (x < -1 || x > 1)
		{
			x = std::modf(x, &xi);
			//The integer part is stored in the object pointed by intpart, 
			//and the fractional part is returned by the function.
			xi = std::pow(std::numbers::e, xi);
		}

		if (x == 0)return xi;

		if (x > 0)
		{
			for (auto i = 0; i < n_samples; i++)
			{
				while (h < 1)
				{
					h += rng(1.0 / x);
					tot++;
				}
				h = 0;
			}
			return T(tot) / n_samples * xi;
		}

		else {

			x = std::abs(x);
			for (auto i = 0; i < n_samples; i++)
			{
				while (h < x)
				{
					h += rng(1.0);
					tot++;
				}
				h = 0;
			}

			return n_samples / T(tot) * xi;
		}
	}

	template <typename T, typename L>
		requires std::floating_point<T>&&
	std::same_as<L, std::uint64_t>
		T log(T x = 0.9, L n_samples = 10000000000)
	{
		T x_0 = 1;
		T e;
		for (int t = 0; t < 17; t++) {
			e = exp(x_0, n_samples);
			x_0 -= 2 * (e - x) / (e + x);
		}
		return x_0;
	}

	template <typename T, typename L>
		requires std::floating_point<T>&&
	std::same_as<L, std::uint64_t>
		T cosh(T x = 0.9, L n_samples = 10000000000)
	{
		return (exp(x, n_samples) + exp(-x, n_samples)) / 2;
	}

	template <typename T, typename L>
		requires std::floating_point<T>&&
	std::same_as<L, std::uint64_t>
		T sinh(T x = 0.9, L n_samples = 10000000000)
	{
		return (exp(x, n_samples) - exp(-x, n_samples)) / 2;
	}

	template <typename L> requires
		std::same_as<L, std::uint64_t>
		double pi(const L INTERVAL)
	{
		double rand_x, rand_y, origin_dist, pi;
		L circle_points = 0, square_points = 0;

		// Total Random numbers generated = possible x
	// values * possible y values
		for (auto i = 0; i < (INTERVAL * INTERVAL); i++) {

			// Randomly generated x and y values
			rand_x = rng(1.0);
			rand_y = rng(1.0);

			// Distance between (x, y) from the origin
			origin_dist = rand_x * rand_x + rand_y * rand_y;

			// Checking if (x, y) lies inside the define
			// circle with R=1
			if (origin_dist <= 1)
				circle_points++;

			// Total number of points generated
			square_points++;

			// estimated pi after this iteration
			pi = double(4 * circle_points) / square_points;

			// For visual understanding (Optional)
			//std::cout << rand_x << " " << rand_y << " "
			//	<< circle_points << " " << square_points
			//	<< " - " << pi << std::endl
			//	<< std::endl;

		}

		// Final Estimated Value
		std::cout << std::endl << "Final Estimation of Pi = " << std::setprecision(12) << pi << std::endl;

		return pi;
	}

	//https://arxiv.org/abs/1704.00358
	std::uint64_t xw = 0, ww = 0, sw = 0xb5ad4eceda1ce2a9;
	inline std::uint32_t msws32() {
		xw *= xw; 
		xw += (ww += sw); 
		xw = (xw >> 32) | (xw << 32);
		return std::uint32_t(xw);
	}

	std::uint64_t x1w = 0, w1w = 0, s1w = 0xb5ad4eceda1ce2a9;
	std::uint64_t x2w = 0, w2w = 0, s2w = 0x278c5a4d8419fe6b;
	inline std::uint64_t msws64() {
		std::uint64_t xx;
		x1w *= x1w; xx = x1w += (w1w += s1w); x1 = (x1w >> 32) | (x1w << 32);
		x2w *= x2w; x2w += (w2w += s2w); x2w = (x2w >> 32) | (x2w << 32);
		return xx ^ x2w;
	}

};

#endif //__MXWS_HPP__ 