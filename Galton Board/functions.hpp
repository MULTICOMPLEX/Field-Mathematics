
#ifndef __FUNCTIONS_HPP__
#define __FUNCTIONS_HPP__
#include <future>
#include <numbers>
#include <span>
#include "MULTICOMPLEX.hpp"
//typedef std::complex<double> Complex;
using Complex = std::complex<double>;
double min_distance = std::numeric_limits<double>::max();

//#include "coroutine.hpp"
#include "matplotlib.hpp"
#include "number_theory.hpp"
#include <map>
#include <unordered_set>
#include "constants.hpp"
#include "vector_operators.hpp"
#include "fft.hpp"
#include "fftw3.h"

template <typename T, typename I>
	requires std::floating_point<T>&&
std::integral<I>
I FP_digits(const T& var, const I& digits)
{
	T p;
	std::modf(var, &p);
	auto dec = std::pow(10, digits - 1);
	T value = var * dec + .5;
	return I(value);
}

plot_matplotlib plot;

auto Galton_Classic = []<typename L, typename K>
	requires std::same_as<L, uint64_t>
(
	const L& balls,
	std::vector<K>& galton_arr,
	double stddev,
	double mean,
	bool RandomWWalk)
{

	mxws <uint32_t>RNG;
	double random_walk = 0;

	const auto Board_SIZE = galton_arr.size();

	uint32_t k;

	cxx::ziggurat_normal_distribution<double> normalz(mean, (Board_SIZE / 12) * stddev);

	for (L i = 0; i < balls; i++, random_walk = 0) {

		if (RandomWWalk) {
			for (auto j = 0; j < Board_SIZE; j++)
				random_walk += RNG(1.);

			k = uint32_t((random_walk - mean) / sqrt(12. / Board_SIZE) * stddev + mean);
			//The 1D board
			if (k < Board_SIZE) galton_arr[k]++;
		}

		else {

			k = uint32_t(normalz(RNG));

			//The 1D board
			if (k < Board_SIZE) galton_arr[k]++;
		}
	}
};

template <typename A, typename I, typename B>
	requires std::integral<I>&&
std::same_as<A, uint64_t>&&
std::same_as<B, bool>
std::vector<A> Galton(
	const A& trials,
	const I& Board_SIZE,
	const I& N_cycles,
	std::vector<A>& galton_arr,
	B probability_wave, A Seed, B Enable_Seed)
{

	mxws <uint32_t>RNG;
	if (Enable_Seed)
		RNG.seed(Seed);
	//std::cout << RNG.x << " " << RNG.w << std::endl;

	std::vector<A> vec;

	if (probability_wave)
		vec = RNG.Probability_Wave(Board_SIZE, galton_arr, trials);

	else {
		vec = { 0, Board_SIZE, Board_SIZE, Board_SIZE };
		Galton_Classic(trials, galton_arr, 1.0, Board_SIZE / 2.0, false);
	}

	return vec;
}

std::vector<std::string> ones{ "","one", "two", "three", "four", "five", "six", "seven", "eight", "nine" };
std::vector<std::string> teens{ "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen","sixteen", "seventeen", "eighteen", "nineteen" };
std::vector<std::string> tens{ "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };

template <typename T>
std::string nameForNumber
(
	const T number)
{
	if (number < 10) {
		return ones[number];
	}
	else if (number < 20) {
		return teens[number - 10];
	}
	else if (number < 100) {
		return tens[number / 10] + ((number % 10 != 0) ? " " + nameForNumber(number % 10) : "");
	}
	else if (number < 1000) {
		return nameForNumber(number / 100) + " hundred" + ((number % 100 != 0) ? " " + nameForNumber(number % 100) : "");
	}
	else if (number < 1000000) {
		return nameForNumber(number / 1000) + " thousand" + ((number % 1000 != 0) ? " " + nameForNumber(number % 1000) : "");
	}
	else if (number < 1000000000) {
		return nameForNumber(number / 1000000) + " million" + ((number % 1000000 != 0) ? " " + nameForNumber(number % 1000000) : "");
	}
	else if (number < 1000000000000) {
		return nameForNumber(number / 1000000000) + " billion" + ((number % 1000000000 != 0) ? " " + nameForNumber(number % 1000000000) : "");
	}
	return "error";
}

template <typename T>
std::vector<T> plot_doDFTr(const std::vector<T>& v, bool distribution_twiddlefactors)
{
	auto ds = v;
	std::vector<T> X, Y, k2;

	Read_DFTCoeff(ds);

	auto k = doDFTr(v, false, false, ds, distribution_twiddlefactors);

	for (auto i = 0; i < k.size(); i++)
		k2.push_back(sqrt(k[i].real * k[i].real + k[i].imag * k[i].imag) / (k.size() / 2.));

	for (auto i = 0; i < k.size() / 2; i++) {
		Y.push_back(20 * std::log10(sqrt(k[i].real * k[i].real + k[i].imag * k[i].imag) / (k.size() / 2.) + .001));
		X.push_back(i);
	}
	std::string str = "Fourier transform with probability wave twiddle factors, ";
	str += nameForNumber(50000000000ull);
	str += " Trials";
	plot.plot_somedata(X, Y, "", str, "green");

	return k2;
}

template <typename T>
void null_offset_vector(std::vector<T>& v)
{
	T mean = 0;
	for (auto& d : v)
		mean += d;

	mean /= v.size();

	for (auto& d : v)
		d -= mean;
}

template <typename T>
void normalize_vector
(
	std::vector<T>& v,
	const double a,
	const double b)
{
	auto k = std::ranges::minmax_element(v);
	auto min = *k.min;
	auto max = *k.max;

	auto normalize = [&](auto& n) {n = a + (n - min) * (b - a) / (max - min); };

	std::ranges::for_each(v, normalize);
}

template <typename T>
T ac_amplite_vector
(
	std::vector<T>& v)
{
	auto k = std::ranges::minmax_element(v);
	return *k.max - *k.min;
}

template <typename T>
void normalize_vector
(
	std::span<T>& v,
	const double a,
	const double b)
{
	auto k = std::ranges::minmax_element(v);
	auto min = *k.min;
	auto max = *k.max;

	auto normalize = [&](auto& n) {n = a + (n - min) * (b - a) / (max - min); };

	std::ranges::for_each(v, normalize);
}

template <typename T>
void Sine_Wave_function
(
	std::vector<T>& Y,
	auto freq
) {

	auto k = 1. / (Y.size() / 360.);
	const T rads = k * std::numbers::pi / 180;

	for (auto i = 0; i < Y.size(); i++)
		Y[i] = std::sin(freq * i * rads);
}

template <typename T>
void Cosine_Wave_function
(
	std::vector<T>& Y,
	auto freq
) {

	auto k = 1. / (Y.size() / 360.);
	const T rads = k * std::numbers::pi / 180;

	for (auto i = 0; i < Y.size(); i++)
		Y[i] = std::cos(freq * i * rads);
}

template <typename T>
void Hann_function
(
	std::vector<T>& Y
) {

	for (auto i = 0; i < Y.size(); i++)
	{
		double multiplier = 0.5 * (1 - std::cos(2 * std::numbers::pi * i / (Y.size() - 1)));
		Y[i] *= multiplier;
	}
}

int MakeWavFromVector
(
	std::string file_name,
	int sample_rate_arg,
	std::vector<short>& sound
);

const char line[] = "    +------+------+------+------+------+------+------+------+------+------+------+------+------+----->";

template <typename T, typename T2>
void cout_galton
(
	const T nboxes,
	const std::vector<T2>& galton_arr
) {
	unsigned i, j;

	auto max = double(*std::ranges::max_element(galton_arr));

	max /= sizeof(line) - 7;

	std::cout << line << std::endl;

	/* print out the results */
	for (i = 0; i < nboxes; i++) {

		std::cout << std::setw(3) << i << " |";
		for (j = 0; j < ceil(galton_arr[i] / float(max)); j++) {
			/* the '#' indicates the balls */
			std::cout << "#";
		}
		std::cout << " " << galton_arr[i] << std::endl << line << std::endl;
	}
}

template <typename T>
bool findParity(T x)
{
	T y = x ^ (x >> 1);
	y ^= std::rotr(y, 2);
	y ^= std::rotr(y, 4);
	y ^= std::rotr(y, 8);
	y ^= std::rotr(x, 32);
	y ^= std::size_t(y) >> 32;

	// Rightmost bit of y holds the parity value
	// if (y&1) is 1 then parity is odd else even
	if (y & 1)
		return 1;
	return 0;
}

template <typename T>
double avarage_vector(std::vector<T>& v)
{
	double mean = 0;
	for (auto& d : v)
		mean += d;

	return mean /= v.size();
}

inline double to_int(double d)
{
	d += 6755399441055744.0;
	return reinterpret_cast<int&>(d);
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << a_value;
	return out.str();
}

template <typename T>
T inline fast_mod(const T& input, const T& ceil)
{
	// apply the modulo operator only when needed
	// (i.e. when the input is greater than the ceiling)
	return input >= ceil ? input % ceil : input;
	// NB: the assumption here is that the numbers are positive
}

std::string utf8_encode(std::u8string const& s)
{
	return (const char*)(s.c_str());
}

std::ostream& operator <<
(
	std::ostream& o,
	std::u8string& s
	)
{
	o << (const char*)(s.c_str());
	return o;
}

#ifdef _WIN32

#include <Windows.h>

class UTF8CodePage {
public:
	UTF8CodePage() : m_old_code_page(GetConsoleOutputCP()) {
		SetConsoleOutputCP(CP_UTF8);
	}
	~UTF8CodePage() { SetConsoleOutputCP(m_old_code_page); }

private:
	UINT m_old_code_page;
};

UTF8CodePage use_utf8;

#endif

struct Color {
	union {
		struct { unsigned char b, g, r, a; };
		unsigned char bytes[4];
		unsigned int ref;
	};
	Color(unsigned int ref = 0) : ref(ref) {}
};

class Surface {
	int width, height;
	std::vector<Color> pixels;
public:
	void Fill(Color color) { std::fill(pixels.begin(), pixels.end(), color); }
	void HLine(int left, int y, int len, Color color) {
		if (y < 0 || y >= height || left >= width) { return; }
		if (left < 0) { len += left; left = 0; }
		if (left + len > width) { len = width - left; }
		int offset = y * width + left;
		std::fill(pixels.begin() + offset, pixels.begin() + offset + len, color);
	}
	void RectFill(int x, int y, int w, int h, Color color) {
		for (int i = 0; i < h; ++i) { HLine(x, y + i, w, color); }
	}
	Surface(int width, int height) :
		width(width),
		height(height),
		pixels(width* height, Color())
	{}
	template <typename I>       Color& operator () (const I& x, const I& y) { return pixels[y * width + x]; }
	template <typename I> const Color& operator () (const I& x, const I& y) const { return pixels[y * width + x]; }

	class Writer {
		std::ofstream ofs;
	public:
		Writer(const char* filename) : ofs(filename, std::ios_base::out | std::ios_base::binary) {}
		void operator () (const void* pbuf, int size) { ofs.write(static_cast<const char*>(pbuf), size); }
		template <typename T> void operator () (const T& obj) { operator () (&obj, sizeof(obj)); }
	};

	struct BIH {
		unsigned int   sz;
		int            width, height;
		unsigned short planes;
		short          bits;
		unsigned int   compress, szimage;
		int            xppm, yppm;
		unsigned int   clrused, clrimp;
	};

	void Save(const char* filename) const {
		Writer w(filename);;
		w("BM", 2);
		BIH bih = { sizeof(bih) };
		bih.width = width;
		bih.height = -height;
		bih.planes = 1;
		bih.bits = 32;
		const unsigned int headersize = sizeof(bih) + 14;
		const int szbuf = int(sizeof(Color) * pixels.size());
		const unsigned int filesize = static_cast<unsigned int>(headersize + szbuf);
		w(filesize);
		const unsigned short z = 0;
		w(z);
		w(z);
		w(headersize);
		w(bih);
		w(pixels.data(), szbuf);
	}
};

void Circle_Squares_fractal(int width = 1024, int height = 1024)
{
	Surface surf(width, height);
	Color color;

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			auto x = (i * i + j * j) / 4;
			color.r = color.g = color.b = 255 - x;
			surf.RectFill(i, j, 1, 1, color);
		}
	}
	surf.Save("Circle_Squares_fractal.bmp");
}

double round_to(double value, double precision)
{
	return std::round(value / precision) * precision;
}

template <typename T>
std::size_t getIndex(const std::vector<T>& v, const T K)
{
	auto it = std::find(v.begin(), v.end(), K);
	auto index = std::distance(v.begin(), it);
	return index;
}

template <typename T>
long double ShannonEntropy(const std::vector<T>& data) {
	long double entropy = 0;
	auto elements = data.size();
	std::map<T, std::size_t> frequencies;

	std::ranges::for_each(data, [&](auto& n) {frequencies[n]++; });

	std::ranges::for_each(frequencies, [&](auto& p) {
		auto p_x = (long double)p.second / elements;
		entropy -= p_x * log2(p_x); });

	return entropy;
}

// function that returns correlation coefficient.
template <typename T>
double correlationCoefficient(std::vector<T>& X, std::vector<T>& Y)
{

	T sum_X = 0, sum_Y = 0, sum_XY = 0;
	T squareSum_X = 0, squareSum_Y = 0;
	const auto n = X.size();
	const auto n2 = Y.size();

	if (n != n2)
		exit(-1);

	for (auto i = 0; i < n; i++)
	{
		// sum of elements of array X.
		sum_X = sum_X + X[i];

		// sum of elements of array Y.
		sum_Y = sum_Y + Y[i];

		// sum of X[i] * Y[i].
		sum_XY = sum_XY + X[i] * Y[i];

		// sum of square of array elements.
		squareSum_X = squareSum_X + X[i] * X[i];
		squareSum_Y = squareSum_Y + Y[i] * Y[i];
	}

	// use formula for calculating correlation coefficient.
	double corr = (double)(n * sum_XY - sum_X * sum_Y)
		/ std::sqrt((n * squareSum_X - sum_X * sum_X)
			* (n * squareSum_Y - sum_Y * sum_Y));

	return corr;
}

template <typename T>
void PlotBoardsizeRandomNumberRange(T N_Bins)
{
	plot_matplotlib plot;

	typedef unsigned I;
	typedef double R;
	typedef bool B;

	std::vector<double> X, Y;
	for (auto i = 1.; i <= 1024; i++) {
		X.push_back(i);
		I board_SIZE = I(round(N_Bins / i));
		I board_size = I(round(log(board_SIZE * 6) * sqrt(std::numbers::pi)));
		Y.push_back(board_size);
	}
	plot.plot_somedata(X, Y, "", "Board size", "green");
	plot.set_title("Board size");
	plot.show();
	Y.clear();
	for (auto i = 1.; i <= 1024; i++) {

		I board_SIZE = I(round(N_Bins / i));
		I rn_range = I(floor(board_SIZE / sqrt(log2(board_SIZE))));
		Y.push_back(rn_range);
	}

	plot.plot_somedata(X, Y, "", "Random number range", "red");
	plot.set_title("Random number range");

	plot.show();

}

template<typename T>
std::vector<std::pair<T, std::size_t>> get_duplicate_indices(const std::vector<T>& vec) {
	auto first = vec.begin();
	auto last = vec.end();
	std::unordered_set<std::size_t> rtn;
	std::unordered_map<T, std::size_t> dup;
	for (std::size_t i = 0; first != last; ++i, ++first) {
		auto iter_pair = dup.insert(std::make_pair(*first, i));
		if (!iter_pair.second) {
			rtn.insert(iter_pair.first->second);
			rtn.insert(i);
		}
	}
	std::vector<std::pair<T, std::size_t>> v;

	for (const auto& i : rtn)
		v.push_back(std::make_pair(vec[i], i));

	std::ranges::sort(v);

	return v;
}

template <typename T>
void count_duplicates(const std::vector<T>& nums)
{
	std::map<T, int> duplicate;
	std::map<int, int> duplicate2;
	std::vector<T> nums_sorted{ nums };
	std::sort(begin(nums_sorted), end(nums_sorted));

	auto beg = begin(nums_sorted) + 1;
	for (; beg != end(nums_sorted); ++beg) {
		if (*beg == *(beg - 1)) {
			duplicate[*beg]++;
		}
	}
	for (const auto& i : duplicate) {
		duplicate2[i.second + 1]++;
		std::cout << std::setprecision(8) << std::setw(14) << i.first << " :: " << i.second + 1 << std::endl;
	}
	for (const auto& i : duplicate2) {
		std::cout << std::setw(19) << std::right << i.first << " :: " << i.second << std::endl;
	}
}

template<typename T>
void cout_ShannonEntropy(std::vector<T>& Y_buf, auto Board_SIZE, auto N_cycles) {

	auto v = get_duplicate_indices(Y_buf);
	T sum = 0;
	if (!v.empty()) {
		std::cout << std::endl << "     Value        Index   Cycle" << std::endl << std::endl;
		for (auto i = 0; i < v.size(); i++) {
			auto s = v[i].first;
			sum += s;
			std::cout << std::setprecision(8) << std::setw(14) <<
				std::right << s << " :: [" << std::modulus()(v[i].second, Board_SIZE) << "]" <<
				" :: [" << v[i].second / Board_SIZE + 1 << "]" << std::endl;
			if (i < v.size())
				if (v[i + 1ull].first != s) std::cout << std::endl;
		}
	}
	if (!v.empty())
		std::cout << "    " << sum << "  Sum ";

	std::cout << std::endl << std::endl;

	std::vector<T> se;
	for (auto k = 0ull; k < N_cycles; k++) {
		se.clear();
		for (auto i = 0ull; i < Board_SIZE; i++) {
			se.push_back(Y_buf[k * Board_SIZE + i]);
		}
		auto entropy = to_string_with_precision(ShannonEntropy(se), 8);
		std::cout << " ShannonEntropy Cycle[" << k + 1 << "]  " << entropy;
		std::cout << std::endl;
		count_duplicates(se);
		std::cout << std::endl;
	}
}

template <typename T>
auto arcsin(const T& x)
{	//-i log(sqrt(1 - x ^ 2) + i x)
	return (-MX0(0, 1) * log(MX0(sqrt(1. - x * x), 0.5))).real;
}

template <typename T>
std::vector<T> wavepacket(std::vector<T>& v, std::uint64_t N_Trials, unsigned N_cycles, bool RandomWWalk)
{
	auto Y = v;
	auto Y_buf = Y;
	auto X = Y;
	auto Y_buf2 = Y;

	int x = 0;
	X.clear();
	while (x < Y.size()) {

		X.push_back(x); x++;
	}
	Y.resize(X.size());

	Galton_Classic(N_Trials, Y, 1.0, Y.size() / 2., RandomWWalk);
	normalize_vector(Y, 0., 1.);

	plot.plot_somedata(X, Y, "", "Gaussian", "green");

	Y_buf *= -Y;

	plot.plot_somedata(X, Y_buf, "", "Wave packet real", "red");
	std::ranges::rotate(Y_buf2, Y_buf2.begin() + Y_buf2.size() / (4ull * N_cycles));

	Y_buf2 *= -Y;

	plot.plot_somedata(X, Y_buf2, "", "Wave packet imag", "blue");

	plot.show();
	return Y_buf2;
}

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
	std::vector<T> values;
	for (T value = start; value < stop; value += step)
		values.push_back(value);
	return values;
}

template<typename T>
	requires std::floating_point<T>
std::vector<double> linspace(T start_in, T end_in, std::uint64_t num_in)
{
	std::vector<double> linspaced;

	double start = start_in;
	double end = end_in;
	double num = double(num_in);

	if (num == 0) { return linspaced; }
	if (num == 1)
	{
		linspaced.push_back(start);
		return linspaced;
	}

	auto delta = (end - start) / (num - 1);

	for (auto i = 0; i < num - 1; ++i)
	{
		linspaced.push_back(start + delta * i);
	}
	linspaced.push_back(end); // I want to ensure that start and end
	// are exactly the same as the input
	return linspaced;
}

template<typename T>
	requires std::floating_point<T>
void Plot_2D_Brownian_Motion(std::vector<T>& B_t_X, std::vector<T>& B_t_Y, std::u8string title, int n)
{

	std::vector<T> x, y;
	for (auto k = 0; k < B_t_X.size(); k += n)
	{
		x.push_back(B_t_X[k]);
		y.push_back(B_t_Y[k]);
	}

	plot.run_customcommand("figure(figsize = (8, 8))");
	plot.plot_somedata(x, y, "o-", "BrownianMotion", "gray", 0.5, 2);

	std::vector<T> btx, bty;
	btx.push_back(B_t_X.front());
	bty.push_back(B_t_Y.front());
	plot.plot_somedata(btx, bty, "o-", "Start", "blue", 0.5, 8);
	btx.clear();
	bty.clear();
	btx.push_back(B_t_X.back());
	bty.push_back(B_t_Y.back());
	plot.plot_somedata(btx, bty, "o-", "End", "orange", 0.5, 8);

	plot.set_xlabel("Time");
	plot.set_ylabel("B(t)");

	plot.run_customcommand("grid(alpha = 0.4)");
	plot.run_customcommand("axis('equal')");

	plot.grid_on();
	plot.set_title(utf8_encode(title));

}

// Function to simulate Brownian motion
template<typename T, typename K>
	requires std::floating_point<T>&&
std::same_as<K, uint64_t>
void Simulate_Brownian_motion_RNGnormal(
	K num_terms, T spread, K seed, std::vector<T>& B_t_x, std::vector<T>& B_t_y) {

	const auto pi = std::numbers::pi;
	const auto pi2 = 2 / std::sqrt(pi);
	const auto pi3 = 1 / std::sqrt(2 * pi);

	// Time points
	std::vector<T> t = linspace(0., 2 * pi, num_terms);

	// Random number generation
	mxws<uint64_t> rng(seed);

	cxx::ziggurat_normal_distribution<T> normalRandomZ;

	// Generate independent standard normal variables
	std::vector<T> xi(num_terms), yi(num_terms);

	for (auto n = 0; n < num_terms; n++) {
		xi[n] = rng.normalRandom(0., 1.);
		yi[n] = rng.normalRandom(0., 1.);
		//xi[n] = normalRandomZ(rng);
		//yi[n] = normalRandomZ(rng);
		if (n > 0) {
			xi[n] /= n;
			yi[n] /= n;
		}
	}

	// Brownian motion calculation
#pragma omp parallel for
	for (auto i = 0; i < num_terms; i++) {
		B_t_x[i] = xi[0] * spread * t[i] * pi3;
		B_t_y[i] = yi[0] * spread * t[i] * pi3;
		for (auto n = 1; n < num_terms; n++) {
			auto k = std::sin(n * t[i] / 2);
			B_t_x[i] += k * xi[n];
			B_t_y[i] += k * yi[n];
		}
		B_t_x[i] *= pi2;
		B_t_y[i] *= pi2;
	}
}


// Function to simulate Brownian motion
template<typename T, typename K>
	requires std::floating_point<T>&&
std::same_as<K, uint64_t>
void Simulate_Brownian_motion_RNGuniform(
	K num_terms, T spread, K seed, std::vector<T>& B_t_x, std::vector<T>& B_t_y) {

	const auto pi = std::numbers::pi;

	// Random number generation
	mxws<uint64_t> rng(seed);

	// Generate independent standard uniform variables
	std::vector<T> xi(num_terms), yi(num_terms);

	const auto nt = 1024;

	auto v = std::sqrt(2) * pi * spread / (num_terms - 1);
	auto spread_x = rng(-v, v);
	auto spread_y = rng(-v, v);

	for (auto n = 1; n < num_terms; n++) {
		xi[n] = rng(-2. / n, 2. / n);
		yi[n] = rng(-2. / n, 2. / n);
	}

	std::array<T, nt> st;

	for (auto i = 0; i < nt; i++) {
		auto angle = i * 2 * pi / nt;
		st[i] = std::sin(angle);
	}

	auto delta = nt / (2.0 * (num_terms - 1));

	// Brownian motion calculation
#pragma omp parallel for
	for (auto i = 0; i < num_terms; i++) {
		auto j = delta * i;
		B_t_x[i] = spread_x * i;
		B_t_y[i] = spread_y * i;
		for (auto n = 1; n < num_terms; n++) {
			auto k = st[K(n * j) % nt];
			B_t_x[i] += k * xi[n];
			B_t_y[i] += k * yi[n];
		}
	}
}


// Function to simulate Brownian motion
template<typename T, typename K>
	requires std::floating_point<T>&&
std::same_as<K, uint64_t>
void Simulate_Brownian_motion_RNGuniform_no_global_storage(
	K num_terms, T spread, K seed, std::vector<T>& B_t_x, std::vector<T>& B_t_y) {

	const auto pi = std::numbers::pi;

	// Random number generation
	mxws<uint64_t> rng1, rng2;
	rng1.seed(seed);
	rng2.seed(seed + 1);

	const auto nt = 1024;

	auto v = std::sqrt(2) * pi * spread / (num_terms - 1);
	auto spread_x = rng1(-v, v);
	auto spread_y = rng2(-v, v);

	std::array<T, nt> st;

	for (auto i = 0; i < nt; i++) {
		auto angle = i * 2 * pi / nt;
		st[i] = std::sin(angle);
	}

	auto delta = nt / (2.0 * (num_terms - 1));

	// Brownian motion calculation
	for (auto i = 0; i < num_terms; i++) {
		auto j = delta * i;
		B_t_x[i] = spread_x * i;
		B_t_y[i] = spread_y * i;

		rng1.seed(seed + 2);
		rng2.seed(seed + 3);

		for (auto n = 1; n < num_terms; n++) {
			auto k = st[K(n * j) % nt];
			auto v = 2. / n;
			B_t_x[i] += k * rng1(-v, v);
			B_t_y[i] += k * rng2(-v, v);
		}
	}

}

void plot_fft(std::vector<double>& v, std::u8string title)
{
	plot.run_customcommand("figure(figsize = (10, 6))");
	plot.set_xlabel("Frequency");
	//plot.run_customcommand("axis('equal')");
	plot.run_customcommand("grid(alpha = 0.4)");
	plot.grid_on();
	plot.set_title(utf8_encode(title));
	plot.set_xlabel("dB");
	plot.set_ylabel("dB");
	plot.mlab_psd(v, v.size());
}


// Function to calculate PWM Fourier coefficient
double pwmCoefficient(int n, double dutyCycle) {
	if (n == 0) {
		return dutyCycle;
	}
	else {
		return (2.0 / (n * pi)) * sin(n * pi * dutyCycle);
	}
}


//Violet noise is essentially the derivative (or difference) of white noise
template<typename K>
std::vector<double> generateVioletNoise(K numSamples, K Seed) {
	std::vector<double> violetNoise(numSamples);
	mxws <uint64_t> gen(Seed);
	std::uniform_real_distribution<> dis(-1.0, 1.0);

	for (K i = 1; i < numSamples; ++i) {
		double whiteNoise = dis(gen);
		violetNoise[i] = whiteNoise - violetNoise[i - 1] / 100;
	}

	return violetNoise;
}

// Function to simulate Brownian motion
template<typename T, typename K>
	requires std::floating_point<T>&&
std::same_as<K, uint64_t>
void Simulate_test(
	K num_terms, T spread, K seed, std::vector<T>& B_t_x, std::vector<T>& B_t_y) {

	const auto pi = std::numbers::pi;

	// Random number generation
	mxws<uint64_t> rng(seed);

	auto v = std::sqrt(pi);

	// Generate independent standard uniform variables
	std::vector<T> xi(num_terms), yi(num_terms);

	const auto nt = 1024;

	// Time points
	std::vector<T> t = linspace(0., 2 * pi, num_terms);

	for (int i = 0; i < num_terms; i++) {
		xi[i] = rng(-std::sqrt(pi), std::sqrt(pi));
		yi[i] = rng(-std::sqrt(pi), std::sqrt(pi));
	}


	std::array<T, nt> st = {};

	bool sine_wave = 1;

	if (sine_wave) {
		for (auto i = 0; i < nt; i++) {
			auto angle = i * 2 * pi / nt;
			st[i] = std::sin(angle);
		}
	}

	else {

		auto numHarmonics = 256;
		double dutyCycle = 0.95; // Initial duty cycle (50%)

		// Generate PWM signal using Fourier series
		for (int i = 0; i < nt; ++i) {
			double angle = i * 2.0 * pi / nt;
			for (int n = 0; n <= numHarmonics; n++) {  // Include DC component (n = 0)
				st[i] += pwmCoefficient(n, dutyCycle) * cos(n * angle);
			}
			st[i] *= 4;
			st[i] -= 3;
		}
	}


#pragma omp parallel for
	for (auto i = 0; i < num_terms; i++) {
		B_t_x[i] = xi[0] * spread * t[i] * (1. / std::sqrt(2 * pi));
		B_t_y[i] = yi[0] * spread * t[i] * (1. / std::sqrt(2 * pi));
		for (auto n = 1; n < num_terms; n++) {
			//auto k = std::sin(n * t[i] / 2);
			auto p = K(fmod(n * t[i] * nt / pi / 4, nt));
			auto k = st[p];
			B_t_x[i] += k * xi[n] / n;
			B_t_y[i] += k * yi[n] / n;
		}
		B_t_x[i] *= 2 / std::sqrt(pi);
		B_t_y[i] *= 2 / std::sqrt(pi);
	}

}

template <typename T>
T myMax(T a, T b) {
	return (a > b) ? a : b;
}

// Main function to generate power-law PSD Gaussian noise
std::vector<double> powerlaw_psd_gaussian(double exponent, uint64_t samples, auto fmin = 0.0) {
	// Calculate frequencies
	std::vector<double> f(samples / 2 + 1); // Frequencies (real FFT)

	// Validate / normalize fmin
	if (0 <= fmin && fmin <= 0.5) {
		fmin = myMax(fmin, 1.0 / samples); // Low frequency cutoff
	}
	else {
		throw std::invalid_argument("fmin must be chosen between 0 and 0.5.");
	}

	// rfftfreq equivalent
	for (int i = 0; i < f.size(); ++i) {
		f[i] = double(i) / samples; // Calculate frequencies
	}

	// Build scaling factors
	std::vector<double> s_scale = f; // Initialize with frequencies
	auto ix = std::ranges::count_if(s_scale,
		[fmin](double freq) { return freq < fmin; }); // Count frequencies below fmin

	if (ix > 0 && ix < int64_t(s_scale.size())) {
		std::fill(s_scale.begin(), s_scale.begin() + ix, s_scale[ix]); // Replace below-cutoff values
	}

	// Apply exponent scaling
	for (auto& scale : s_scale) {
		scale = std::pow(scale, -exponent / 2.0);
	}

	// Calculate theoretical output standard deviation from scaling
	double sum_squares = 0;
	for (double s : s_scale) {
		sum_squares += s * s;
	}
	double sigma = 2 * std::sqrt(sum_squares) / samples;

	// Prepare random number generator
	std::random_device rd;
	mxws<uint32_t> gen(rd());
	std::uniform_real_distribution<> dis(-std::sqrt(std::numbers::pi), std::sqrt(std::numbers::pi));

	// Generate scaled random power + phase
	std::vector<double> sr(f.size()), si(f.size());
	for (size_t i = 0; i < f.size(); ++i) {
		sr[i] = dis(gen) * s_scale[i];
		si[i] = dis(gen) * s_scale[i];
	}

	// Combine power + corrected phase to Fourier components
	std::vector<std::complex<double>> s(f.size());
	for (size_t i = 0; i < f.size(); ++i) {
		s[i] = std::complex<double>(sr[i], si[i]);
	}

	std::vector<double> y(samples);
	
	fftw_plan plan = fftw_plan_dft_c2r_1d(int(samples), reinterpret_cast<fftw_complex*>(s.data()), y.data(), FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
	
	//y = inverse_fft_real(s);
	// Transform to real time series & scale to unit variance
	for (auto& value : y) 
		value /= sigma * samples;
	
	return y;
}

// Define a structure to represent a point in 2D space
struct Point {
	double x;
	double y;
};

// Function to compute the Euclidean distance between two points
double euclidean_distance(const Point& p1, const Point& p2) {
	double dx = p1.x - p2.x;
	double dy = p1.y - p2.y;
	return std::sqrt(dx * dx + dy * dy);
}

void Red_Noise() //Brownian noise, also known as Brown noise or red noise
{
	const uint64_t Nsamples = 8192;
	const auto spread = 0.0001;
	const bool enable_random_seed = true;
	uint64_t seed = 10;

	std::random_device r;
	if (enable_random_seed)
		seed = (uint64_t(r()) << 32) | r();

	std::vector<double> B_t_x(Nsamples, 0.0), B_t_y(Nsamples, 0.0);

	auto begin = std::chrono::high_resolution_clock::now();
	Simulate_Brownian_motion_RNGnormal(Nsamples, spread, seed, B_t_x, B_t_y);
	auto end = std::chrono::high_resolution_clock::now();

	auto k1 = std::ranges::minmax_element(B_t_x);
	auto Amplitude_x = *k1.max - *k1.min;
	auto k2 = std::ranges::minmax_element(B_t_y);
	auto Amplitude_y = *k2.max - *k2.min;
	std::cout << std::setprecision(3) << "RNG Normal Amplitude  P-P: " << "X{" << Amplitude_x << "}, Y{" << Amplitude_y << "}" << std::endl;

	Plot_2D_Brownian_Motion(B_t_x, B_t_y, u8"Simulated Brownian Motion, RNG Normal", 1);

	std::cout << " Duration     "
		<< std::chrono::nanoseconds(end - begin).count() / 1e9
		<< "[s]" << std::endl << std::endl << std::endl;

	plot_fft(B_t_x, u8"Power spectral density RNG Normal");


	///////////////
	begin = std::chrono::high_resolution_clock::now();
	auto N = Nsamples * 32;
	double beta = 3;
	double fmin = 0;
	auto x = powerlaw_psd_gaussian(beta, N, fmin);
	auto y = powerlaw_psd_gaussian(beta, N, fmin);
	end = std::chrono::high_resolution_clock::now();
	std::cout << " Duration     "
		<< std::chrono::nanoseconds(end - begin).count() / 1e9
		<< "[s]" << std::endl;

	plot_fft(x, u8"Power spectral density powerlaw_psd_gaussian");

	Plot_2D_Brownian_Motion(x, y, u8"Simulated Brownian Motion, Powerlaw", 32);
	Point point1 = { x.front(), y.front()};
	Point point2 = { x.back(), y.back()};
	auto distance = euclidean_distance(point1, point2);
	std::cout << utf8_encode(u8"Δ Start-End= ") << std::scientific << std::setprecision(6) << distance << std::defaultfloat << std::endl << std::endl;
	//plot_fft(x, u8"Powerlaw PSD FFT");
	///////////////


	begin = std::chrono::high_resolution_clock::now();
	//Simulate_test(Nsamples, spread, seed, B_t_x, B_t_y);
	Simulate_Brownian_motion_RNGuniform(Nsamples, spread, seed, B_t_x, B_t_y);
	//Simulate_Brownian_motion_RNGuniform_no_global_storage(Nsamples, spread, seed, B_t_x, B_t_y);
	end = std::chrono::high_resolution_clock::now();

	std::cout << " Duration     "
		<< std::chrono::nanoseconds(end - begin).count() / 1e9
		<< "[s]" << std::endl << std::endl << std::endl;

	k1 = std::ranges::minmax_element(B_t_x);
	Amplitude_x = *k1.max - *k1.min;
	k2 = std::ranges::minmax_element(B_t_y);
	Amplitude_y = *k2.max - *k2.min;
	std::cout << std::setprecision(3) << "RNG Uniform Amplitude P-P: " << "X{" << Amplitude_x << "}, Y{" << Amplitude_y << "}" << std::endl;


	Plot_2D_Brownian_Motion(B_t_x, B_t_y, u8"Simulated Brownian Motion, RNG Uniform", 1);

	plot_fft(B_t_x, u8"Power spectral density RNG Uniform");


	plot.show();

}

#endif // FUNCTIONS

