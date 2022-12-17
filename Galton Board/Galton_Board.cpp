
#include "functions.hpp"

/*1D Galton Board simulation with normal and sinusoidal distribution*/

int main(int argc, char** argv)
{

	std::setlocale(LC_ALL, "en_US.utf8");

	typedef unsigned U;
	typedef double R;
	typedef bool B;

	/***************SETTINGS*****************/

	std::uint64_t N_Trials = 100000000;

	U N_cycles = 8; //== Threads  

	U N_Bins = 2048;
	//speedup
	B speedup = true;
	//Sinusoidal distribution or Normal distribution
	B probability_wave = true;
	//DC distribution
	B DC = false;
	//Enable Fourier transform
	B dft = true;
	//Enable sound to file (wav format)
	B wav = false;

	//Console output
	const B cout_gal = false;

	const B Hanning_window = false;

	//Plot Fourier transform with probability wave twiddle factors
	B doDFTr = false;

	//Enable Sliding FFT
	const B Sliding_FFT = false;

	//Write twiddle factors to disk
	B W_DFTCoeff = false;
	if ((N_Bins != 2048) || (N_cycles != 1))
		W_DFTCoeff = false;

	/***************SETTINGS*****************/

	if (wav) {
		N_Trials = 100000000;
		N_Bins = 300000;
		N_cycles = 1000;
		dft = false;
	}

	U Board_SIZE = U(round(N_Bins / R(N_cycles)));

	/* get cmd args */
	if (argc < 7) {
		std::cout <<
			"Usage: Galton_Board [N]Trials(Balls) [N]Cycles [1-0]Probability-Wave\
 [1-0]Raw [1-0]DFT [1-0]Sound-wav" <<
			std::endl << std::endl;
	}

	else {
		N_Trials = atoi(argv[1]);
		N_cycles = atoi(argv[2]);
		probability_wave = atoi(argv[3]);
		DC = atoi(argv[4]);
		dft = atoi(argv[5]);
		wav = atoi(argv[6]);
		Board_SIZE = U(round(N_Bins / R(N_cycles)));
	}

	std::u8string title = u8" Probability Wave Ψ";
	if (!probability_wave) title = u8" Galton Board";
	std::cout << title << std::endl << std::endl;

	auto N_Hthreads = std::thread::hardware_concurrency();
	std::cout << " " << N_Hthreads << " concurrent cycles (threads) are supported."
		<< std::endl << std::endl;

	if (probability_wave) {
		std::cout << " Trials         " << nameForNumber(N_Trials) << " (" << N_Trials << ")"
			<< " x " << N_cycles << std::endl;
		std::cout << " Cycles         " << N_cycles << std::endl;
		std::cout << " Board SIZE     " << Board_SIZE << "[Boxes]" << std::endl;
	}

	else {
		Board_SIZE = 2048;
		N_cycles = 1;
		std::cout << " Trials         " << nameForNumber(N_Trials) << " (" << N_Trials << ")"
			<< " x " << N_cycles << std::endl;
	}

	if (doDFTr) {
		if (Board_SIZE * N_cycles < N_Bins)
			N_cycles += 1;
	}

	//std::cout << std::endl << " Sinusoid size  " << Board_SIZE * N_cycles << std::endl;

	std::tuple<R, U> tuple;

	std::vector < std::future <decltype(tuple)>> vecOfThreads;

	//std::vector<std::vector<std::uint64_t>> galton_arr(N_cycles, std::vector<std::uint64_t>(Board_SIZE));

	std::vector<std::vector<std::vector<std::uint64_t>>>
		galton_arr(4, std::vector<std::vector<std::uint64_t>>(N_cycles, std::vector<std::uint64_t>(Board_SIZE, 0ull)));

	auto begin = std::chrono::high_resolution_clock::now();

	const U integrations = 1;
	for (U i = 0; i < integrations; i++) {
		for (U k = 0; k < N_cycles; k++)
			vecOfThreads.push_back(std::async([&, i, k] {
			return Galton<R>(N_Trials, Board_SIZE, N_cycles, galton_arr[i][k], probability_wave, speedup); }));
	}

	for (auto& th : vecOfThreads)
		tuple = th.get();

	auto end = std::chrono::high_resolution_clock::now();

	U Board_size = std::get<U>(tuple);

	if (probability_wave)
	{
		std::cout << std::endl << " RNG range      " << std::get<R>(tuple) << "[Boxes]" << std::endl;
	}

	std::cout << " Board size     " << Board_size << "[Boxes]" << std::endl;

	std::cout << std::endl << " Duration Ball  "
		<< std::chrono::nanoseconds(end - begin).count() / N_Trials
		<< "[ns]" << std::endl << std::endl;

	if (Board_SIZE <= 256 && cout_gal)
		cout_galton(Board_SIZE, galton_arr[0][0]);

	std::vector<R> X, Y, Y_buf;

	if (!probability_wave) {

		for (auto i = 0; auto & k : std::span(galton_arr[0].front())) {
			X.push_back(i++);
			Y.push_back(R(k));
		}

		plot.plot_somedata(X, Y, "", "Binomial-Normal Distribution", "blue");

		std::string str = "Number of Balls : ";
		str += nameForNumber(N_Trials);

		auto max = *std::ranges::max_element(Y);

		plot.text(0, max / 3, str, "green", 11);

		auto entropy = to_string_with_precision(ShannonEntropy(Y), 8);
		str = "Shannon entropy= ";
		str += entropy;
		plot.text(0, max / 4, str, "red", 11);

		std::cout << " ShannonEntropy " << entropy << std::endl;

		plot.set_xlabel("Boxes");
		plot.grid_on();
		plot.set_title(utf8_encode(title));
		plot.show();
	}

	else {

		for (auto& i : galton_arr.front())
			std::ranges::transform(i, std::back_inserter(Y),
				[](auto& c) {return double(c); });

		std::cout << " Avarage        " << avarage_vector(Y) / N_cycles << std::endl;
		std::cout << " AC Amplitude   " << ac_amplite_vector(Y) / N_cycles << std::endl;

		Y_buf = Y;

		if (doDFTr) {
			if (Y.size() > 2048ull) {
				Y.clear();
				for (auto x = 0; x < 2048; x++)
					Y.push_back(Y_buf[x]);
			}
		}

		if (!DC) {
			normalize_vector(Y, 1., -1.);
			null_offset_vector(Y);
		}

		if (Hanning_window)
			Hann_function(Y);

		Y_buf = Y;

		B pow2 = ((Y.size() & (Y.size() - 1)) == 0);

		if (wav)
			pow2 = false;

		std::vector<MX0> cx;

		if (((Y.size() <= 1000000) || pow2) && !DC && dft)
		{
			if (pow2) {
				if (Sliding_FFT) {
					cx.resize(Y.size());
					slidingDFT_driver(Y, cx);
				}
				else {
					std::ranges::transform(Y, std::back_inserter(cx),
						[](auto& c) {return c; });
					cx = FFT(cx);
				}
			}

			else {
				cx.resize(Y.size());
				doDFT(Y, cx);
			}

			X.clear();
			Y.clear();

			for (auto i = 0.; auto & d : std::span(cx).subspan(0, cx.size() / 2))
			{
				X.push_back(i);
				Y.push_back(20 * std::log10(std::sqrt(d.norm()) / (cx.size() / 2.) + .001));
				i++;
			}

			plot.set_ylabel("dB");
			plot.plot_somedata(X, Y, "", "Fourier Transform", "red");

			R rms = 0;
			for (auto& d : std::span(cx).subspan(0, cx.size() / 2))
				rms += std::sqrt(d.norm());

			rms *= 1. / (cx.size()) * 2;

			std::cout << " RMS	        " << rms << "[dB]" << std::endl << std::endl;

			std::string str = "RMS= ";
			str += std::to_string(rms);
			str += "    RNG range= ";

			auto text_x_offset = N_Bins / 10; //210

			auto rng_range = std::get<R>(tuple);
			str += to_string_with_precision(rng_range, 3);
			plot.text(text_x_offset, -23, str, "green", 11);

			if (Board_SIZE > Board_size) str = "Shrunken to ";
			else if (Board_SIZE < Board_size) str = "Grown to ";
			else str = "Size stayed the same= ";
			plot.text(text_x_offset, -13, str + std::to_string(Board_size), "purple", 11);

			str = "Factor= "; plot.text(text_x_offset, -18, str, "orange", 11);
			if (Board_SIZE > Board_size)
				str += to_string_with_precision(R(Board_SIZE) / Board_size, 3);
			else if (Board_SIZE < Board_size)
				str += to_string_with_precision(Board_size / R(Board_SIZE), 3);
			else
				str += to_string_with_precision(R(Board_SIZE), 3);
			plot.text(text_x_offset, -18, str, "orange", 11);

			str = "Number of cycles= ";
			str += std::to_string(N_cycles);
			str += ", Board size= ";
			plot.text(text_x_offset, -8, str + std::to_string(Y_buf.size() / N_cycles),
				"blue", 11);

			str = "Shannon entropy= ";
			auto entropy = to_string_with_precision(ShannonEntropy(Y_buf), 8);
			str += entropy;
			plot.text(text_x_offset, -28, str, "black", 11);
			
			std::cout << " ShannonEntropy Cycles[1.." << N_cycles << "] " << entropy << std::endl << std::endl;
			count_duplicates(Y_buf);

			cout_ShannonEntropy(Y_buf, Board_SIZE, N_cycles);
			
			X.clear();

			for (auto i = 0.; i < Y_buf.size() / 2.; i += 0.5)
				X.push_back(i);

			plot.line(Y.size() - R(Board_size / 2),
				R(Y.size()), -2, -2, "purple", 4, "solid");

		}

		if (!wav)
		{
			if (!dft || DC) {
				X.clear();
				for (auto i = 0; i < Y_buf.size(); i++)
					X.push_back(i);
			}

			std::string str = "Number of Trials: ";

			str += nameForNumber(N_Trials);
			str += " x ";
			str += std::to_string(N_cycles);

			plot.plot_somedata(X, Y_buf, "", str, "blue");

			if (W_DFTCoeff)
				Write_DFTCoeff(Y_buf);

			if (doDFTr) {
				plot_doDFTr(Y_buf, true);

				//std::cout << std::endl << "DFT Shannon Entropy " << std::setprecision(10) << ShannonEntropy(Y_buf);
				//count_duplicates(Y_buf);
			}

			plot.set_title(utf8_encode(title));
			if (!DC)plot.set_xlabel("Bins / 2");
			else plot.set_xlabel("Bins");
			plot.grid_on();
			plot.show();

			///////////////////
			str = "Number of Trials: ";
			str += nameForNumber(N_Trials);
			str += " x ";
			str += std::to_string(N_cycles);

			double x = -std::numbers::pi;
			X.clear();
			for (auto& i : Y_buf) {
				i *= 3;
				x += 1. / (Y_buf.size() - 1) * 2 * std::numbers::pi;
				X.push_back(x);
			}

			plot.PyRun_Simple("fig = plt.figure()");
			plot.PyRun_Simple("ax = fig.add_subplot(projection = 'polar')");

			plot.plot_polar(X, Y_buf, "", str, "red", 1.0);
			std::ranges::rotate(Y_buf, Y_buf.begin() + Y_buf.size() / (2ull * N_cycles)); //Rotate left
			plot.plot_polar(X, Y_buf, "", str, "blue", 1.0);

			title = u8" Rose Curve Φ";
			plot.set_title(utf8_encode(title));

			plot.show();
		}

		else {

			std::vector<short> sound;
			sound.reserve(Y_buf.size());

			for (auto& d : Y_buf)
				sound.push_back(short((d * 2500.)));

			MakeWavFromVector("Galton.wav", 44100, sound);

			if (X.size() <= 20480) {

				X.clear();

				for (auto i = 0.; i < Y_buf.size(); i++)
					X.push_back(i);

				plot.plot_somedata(X, Y_buf, "", "Wav", "blue");

				plot.set_title("Galton.wav");
				plot.show();
			}
		}
	}

	return 0;
}

