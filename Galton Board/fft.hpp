
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

template <typename T>
	requires std::same_as<T, MX0> ||
std::same_as<T, std::complex<double>>
std::vector<T> IFFT(std::vector<T> v)
{
	for (auto& i : v)
		// conjugate the complex numbers
		i = i.conj();

	// forward fft
	v = FFT(v);

	for (auto& i : v)
		// conjugate the complex numbers again
		i = i.conj();

	// scale the numbers
	for (auto& i : v)
		i /= double(v.size());

	return v;
}

template <typename T>
std::vector<MX0> doDFT(const std::vector<T>& in)
{
	//mxws<uint32_t> rng;
	auto N = in.size();
	MX0	z(0, -2 * std::numbers::pi / N);
	MX0 W(exp(z)), Wk(1);

	std::vector<MX0> out(in.size());

	for (auto& x : out)
	{
		MX0	Wkl(1);
		x = 0;

		for (auto& y : in)
		{
			x += y * Wkl;
			//Wkl *= Wk + MX0(rng(-0.0001, 0.0001), rng(-0.0001, 0.0001));
			Wkl *= Wk;
		}
		Wk *= W;
	}
	return out;
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