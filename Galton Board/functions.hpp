
#pragma once
#include <future>
#include <numbers>
#include <span>
#include "MULTICOMPLEX.hpp"
//#include "coroutine.hpp"
#include "matplotlib.hpp"
#include "number_theory.hpp"
#include <map>
#include <unordered_set>


plot_matplotlib plot;

auto Galton_Classic = []<typename L, typename T>
	requires std::same_as<L, uint64_t>&&
std::integral<T>
(
	const L& balls,
	const T& Board_SIZE,
	std::vector<L>& galton_arr)
{
	const bool RandomWWalk = false;

	mxws <uint32_t>RNG;
	double random_walk = 0;
	T k;

	const double mean = Board_SIZE / 2, stddev = 1;

	cxx::ziggurat_normal_distribution<double> normalz(mean, (Board_SIZE / 12) * stddev);

	for (L i = 0; i < balls; i++, random_walk = 0) {

		if (RandomWWalk) {
			for (T j = 0; j < Board_SIZE; j++)
				random_walk += RNG(1.);

			k = T((random_walk - mean) / sqrt(12. / Board_SIZE) * stddev + mean);
			//The 1D board
			if (k < Board_SIZE) galton_arr[k]++;
		}

		else {

			k = T(normalz(RNG));

			//The 1D board
			if (k < Board_SIZE) galton_arr[k]++;
		}
	}
};

template <typename R, typename A, typename I, typename B>
	requires std::integral<I>&&
std::same_as<A, uint64_t>
std::tuple<R, I> Galton(
	const A& trials,
	const I& Board_SIZE,
	const I& N_cycles,
	std::vector<A>& galton_arr,
	B probability_wave, B speedup)
{
	mxws <uint32_t>RNG;

	std::tuple<R, I> tuple;

	if (probability_wave)
		tuple = RNG.Probability_Wave<R>(Board_SIZE, galton_arr, trials, speedup);

	else {
		Galton_Classic(trials, Board_SIZE, galton_arr);
		tuple = std::make_tuple(0., Board_SIZE);
	}

	return tuple;
}

template <typename T>
T logint
(
	const T n)
{
	auto lg = 0;
	while ((T(1) << lg) < n)
		lg++;
	return lg;
}

template <typename T>
T rev
(
	T x,
	T lgn
) {
	T res = 0;
	while (lgn--) {
		res = 2 * res + (x & 1);
		x /= 2;
	}
	return res;
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
void Write_DFTCoeff(const std::vector<T>& v) {
	std::fstream file;
	file.open("DFT_coeff.txt", std::ios_base::out);

	for (const auto& content : v)
		file << std::setprecision(15) << content << std::endl;
	file.close();
}

template <typename T>
void Read_DFTCoeff(std::vector<T>& v) {
	std::fstream file;
	file.open("DFT_coeff.txt", std::ios_base::in);

	for (auto& content : v)
		file >> content;
	file.close();
}

template <typename T>
	requires std::same_as<T, MX0> ||
std::same_as<T, std::complex<double>>
std::vector<T> FFT
(
	const std::vector<T>& v)
{
	const T J(0, 1);

	auto n = v.size();
	std::vector<double> sine(n), cosine(n);

	/*
	for (auto k = 0; k < n; k++) {
		cosine[k] = cos(2 * std::numbers::pi * k / n);
		sine[k] = sin(2 * std::numbers::pi * k / n);
	}
	*/

	auto lgn = logint(n);
	assert((n & (n - 1)) == 0);
	std::vector<T> perm(n);

	for (size_t i = 0; auto & d : perm)
		d = v[rev(i++, lgn)];

	for (auto s = 1; s <= lgn; s++) {
		auto m = (1 << s);
		T wm = exp(-2 * std::numbers::pi * J / m);
		//auto index = std::uint64_t(1.0 / m * n);
		//T wm { cosine[index], sine[index] };

		for (auto k = 0; k < n; k += m) {
			T w = 1;
			for (auto j = 0; j < m / 2; j++) {
				T t = w * perm[k + j + m / 2];
				T u = perm[k + j];
				perm[k + j] = u + t;
				perm[k + j + m / 2] = u - t;
				w *= wm;
			}
		}
	}

	return perm;
}

template <typename T, typename Y>
void doDFT(std::vector<T>& in, std::vector<Y>& out)
{
	auto N = in.size();
	MX0	z(0, -2 * std::numbers::pi / N);
	MX0 W(exp(z)), Wk(1);

	for (auto& x : out)
	{
		MX0	Wkl(1);
		x = 0;

		for (auto& y : in)
		{
			x += y * Wkl;
			Wkl *= Wk;
		}
		Wk *= W;
	}
}

class DFT_Coeff {
public:
	double real, imag;
	DFT_Coeff() {
		real = 0.0;
		imag = 0.0;
	}
};

template <typename T>
	requires
std::same_as<T, double>
std::vector<DFT_Coeff> doDFTr(const std::vector<T> function, bool inverse, bool sd,
	std::vector<T>& ds, bool distribution_twiddlefactors) {

	auto N = function.size();

	std::vector<T> sine(N), cosine(N), idft(N);

	std::vector<DFT_Coeff> dft_value(N);

	for (auto n = 0; n < N; n++) {
		cosine[n] = cos(2 * std::numbers::pi * n / N);
		sine[n] = sin(2 * std::numbers::pi * n / N);
	}

	if (distribution_twiddlefactors) {
		cosine = ds;
		std::ranges::rotate(ds, ds.begin() + N / 4ull);
		sine = ds;
	}

	else {
		cosine = sine;
		std::ranges::rotate(sine, sine.begin() + N / 4ull);
	}

	for (auto k = 0; k < N; k++) {
		//std::cout << std::endl;
		dft_value[k].real = 0;
		dft_value[k].imag = 0;
		for (auto n = 0; n < N; n++) {
			//cosine[k] = cos(2 * std::numbers::pi * k * n / N);
			//sine[k] = sin(2 * std::numbers::pi * k * n / N);
			auto m = std::modulus()(k * n, N);
			dft_value[k].real += function[n] * cosine[m];
			dft_value[k].imag -= function[n] * sine[m];
			//std::cout << k << " " << function[n] * cosine[m] << std::endl;
		}
	}

	if (sd)std::cout << std::endl;
	for (auto j = 0; j < N; j++) {
		//dft_value[j].real = sqrt(dft_value[j].real * dft_value[j].real + dft_value[j].imag * dft_value[j].imag);
		//std::cout << dft_value[j].real << std::endl;
		if (std::abs(dft_value[j].imag) < 0.00000001) dft_value[j].imag = 0;
		if (std::abs(dft_value[j].real) < 0.00000001) dft_value[j].real = 0;
		//if (sd)std::cout << std::setprecision(8) << dft_value[j].real << " " << dft_value[j].imag << std::endl;
	}

	if (inverse) {
		for (auto k = 0; k < N; k++) {
			//std::cout << std::endl;
			idft[k] = 0;
			for (auto n = 0; n < N; n++) {
				auto m = std::modulus()(k * n, N);
				idft[k] += dft_value[n].real * cosine[m] - dft_value[n].imag * sine[m];
			}
			idft[k] /= N;
		}

		if (sd) {
			std::cout << std::endl;
			for (auto n = 0; n < N; n++)
				std::cout << idft[n] << " " << function[n] << std::endl;
		}
	}
	return dft_value;
}

template <typename T>
void plot_doDFTr(const std::vector<T>& v, bool distribution_twiddlefactors)
{
	auto ds = v;
	std::vector<T> X, Y;

	Read_DFTCoeff(ds);

	auto k = doDFTr(v, false, false, ds, distribution_twiddlefactors);

	for (auto i = 0; i < k.size() / 2; i++) {
		Y.push_back(20 * std::log10(sqrt(k[i].real * k[i].real + k[i].imag * k[i].imag) / (k.size() / 2.) + .001));
		X.push_back(i);
	}
	std::string str = "Fourier transform with probability wave twiddle factors, ";
	str += nameForNumber(50000000000ull);
	str += " Trials";
	plot.plot_somedata(X, Y, "", str, "green");
}


template <class T, std::size_t DFT_Length>
class SlidingDFT
{
private:
	/// Are the frequency domain values valid? (i.e. have at elast DFT_Length data
	/// points been seen?)
	bool data_valid = false;

	/// Time domain samples are stored in this circular buffer.
	std::vector<T> x;

	/// Index of the next item in the buffer to be used. Equivalently, the number
	/// of samples that have been seen so far modulo DFT_Length.
	std::size_t x_index = 0;

	/// Twiddle factors for the update algorithm
	std::vector <MX0> twiddle;

	/// Frequency domain values (unwindowed!)
	std::vector<MX0> S;

	bool Hanning_window = false;

public:

	/// Frequency domain values (windowed)
	std::vector<MX0> dft;

	virtual ~SlidingDFT() = default;

	T damping_factor = std::nexttoward((T)1, (T)0);

	/// Constructor
	SlidingDFT()
	{
		x.resize(DFT_Length);
		twiddle.resize(DFT_Length);
		S.resize(DFT_Length);
		dft.resize(DFT_Length);

		const MX0 j(0, 1);
		auto N = DFT_Length;

	// Compute the twiddle factors, and zero the x 
		for (auto k = 0; k < DFT_Length; k++) {
			T factor = 2 * std::numbers::pi * k / N;
			twiddle[k] = exp(j * factor);
		}
	}

	/// Determine whether the output data is valid
	bool is_data_valid()
	{
		return data_valid;
	}

	/// Update the calculation with a new sample
	/// Returns true if the data are valid (because enough samples have been
	/// presented), or false if the data are invalid.
	bool update(T new_x)
	{
		// Update the storage of the time domain values
		const T old_x = x[x_index];
		x[x_index] = new_x;

		// Update the DFT
		const T r = damping_factor;
		const T r_to_N = pow(r, (T)DFT_Length);
		for (auto k = 0; k < DFT_Length; k++) 
			S[k] = twiddle[k] * (r * S[k] - r_to_N * old_x + new_x);

		if (Hanning_window) {
			// Apply the Hanning window
			dft[0] = (T)0.5 * S[0] - (T)0.25 * (S[DFT_Length - 1] + S[1]);
			for (size_t k = 1; k < (DFT_Length - 1); k++) {
				dft[k] = (T)0.5 * S[k] - (T)0.25 * (S[k - 1] + S[k + 1]);
			}
			dft[DFT_Length - 1] = (T)0.5 * S[DFT_Length - 1] - (T)0.25 * (S[DFT_Length - 2] + S[0]);
		}
		else
			dft = S;

		// Increment the counter
		x_index++;
		if (x_index >= DFT_Length) {
			data_valid = true;
			x_index = 0;
		}

		// Done.
		return data_valid;
	}
};

template <typename T>
void slidingDFT_driver(
	std::vector<T>& Y,
	std::vector<MX0>& cx
)
{
	const auto N = 2048;
	SlidingDFT<T, N> dft;

	for (size_t i = 0; i < N; i++) {
		dft.update(Y[i]);
		if (dft.is_data_valid()) {
			for (size_t j = 0; j < N; j++)
				cx[j] = dft.dft[j];
		}
	}
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

	std::ranges::for_each(data, [&](auto& n) {frequencies[n] ++; });

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
std::vector<std::pair<T,std::size_t>> get_duplicate_indices(const std::vector<T>& vec) {
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
		duplicate2[i.second + 1] ++;
		std::cout << std::setprecision(8) << std::setw(14) << i.first << " :: " << i.second + 1 << std::endl;
	}
	for (const auto& i : duplicate2) {
		std::cout << std::setw(18) << std::right << i.first << " :: " << i.second << std::endl;
	}
}

template<typename T>
void cout_ShannonEntropy(std::vector<T>& Y_buf, auto Board_SIZE, auto N_cycles) {

	auto v = get_duplicate_indices(Y_buf);
	if (!v.empty()) {
		std::cout << std::endl << "  ShannonEntropy  Index   Cycle" << std::endl << std::endl;
		for (auto i = 0; i < v.size(); i++) {
			std::cout << std::setprecision(8) << std::setw(14) <<
				std::right << v[i].first << " :: [" << std::modulus()(v[i].second, Board_SIZE) << "]" <<
				" :: [" << v[i].second / Board_SIZE + 1 << "]" << std::endl;
			if (i < v.size())
				if (v[i + 1ull].first != v[i].first) std::cout << std::endl;
		}
	}
	std::cout << std::endl;

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
	return (- MX0(0, 1) * log(MX0(sqrt(1. - x * x), 0.5))).real;
}