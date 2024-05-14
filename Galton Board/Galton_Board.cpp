
#include "functions.hpp"

/*1D Galton Board simulation with normal and sinusoidal distribution*/

int main(int argc, char** argv)
{
	std::setlocale(LC_ALL, "en_US.utf8");

	typedef unsigned U;
	typedef double R;
	typedef bool B;

	/***************SETTINGS*****************/

	std::uint64_t Ntrials = 1000000000;

	//Wave cycles or threads  
	U Ncycles = 5;
	//Number of integrations
	U N_Integrations = 1;
	//Initial number of bins
	U Nbins = 3000;
	if (Nbins < 3 * Ncycles)//minimum 3 x Ncycles
		Nbins = 3 * Ncycles;
	//Sinusoidal distribution or Normal distribution
	U Binormal_Distribution_Nbins = 2000;
	B Probability_wave = true;
	//Entropy analysis
	B Entropy = false;
	//DFT Entropy analysis
	B DFTEntropy = false;
	//DC distribution
	B DC = false;
	//Enable Fourier transform
	B DFT = true;
	//Enable sound to file (WAV format)
	B WAV = false;


	//Console output
	const B cout_gal = false;

	const B Hanning_window = false;

	//Plot Fourier transform with probability wave twiddle factors
	B doDFTr = true;
	if (Nbins != 2048)
		doDFTr = false;

	//Enable Sliding FFT
	const B Sliding_FFT = false;

	//Write twiddle factors to disk
	B W_DFTCoeff = false;
	if ((Nbins != 2048) || (Ncycles != 1))
		W_DFTCoeff = false;

	/***************SETTINGS*****************/

	if (WAV) {
		Ntrials = 100000000;
		Nbins = 300000;
		Ncycles = 1000;
		DFT = false;
	}

	U Initial_Board_size = U(round(Nbins / R(Ncycles)));


	/* get cmd args */
	if (argc < 7) {
		std::cout <<
			"Usage: Galton_Board [N]Trials(Balls) [N]Cycles [1-0]Probability-Wave\
 [1-0]Raw [1-0]DFT [1-0]Sound-wav" <<
			std::endl << std::endl;
	}

	else {
		Ntrials = atoi(argv[1]);
		Ncycles = atoi(argv[2]);
		Probability_wave = atoi(argv[3]);
		DC = atoi(argv[4]);
		DFT = atoi(argv[5]);
		WAV = atoi(argv[6]);
		Initial_Board_size = U(round(Nbins / R(Ncycles)));
	}

	std::u8string title = u8" Probability Wave Φ";
	if (!Probability_wave) title = u8" Galton Board";
	std::cout << title << std::endl << std::endl;

	auto N_Hthreads = std::thread::hardware_concurrency();
	std::cout << " " << N_Hthreads << " concurrent cycles (threads) are supported."
		<< std::endl << std::endl;

	Ntrials *= N_Integrations;

	if (Probability_wave) {
		std::cout << " Trials            " << nameForNumber(Ntrials) << " (" << Ntrials << ")"
			<< " x " << Ncycles << std::endl;
		std::cout << " Cycles            " << Ncycles << std::endl;

	}

	else {
		Initial_Board_size = Binormal_Distribution_Nbins;
		Ncycles = 1;
		std::cout << " Trials            " << nameForNumber(Ntrials) << " (" << Ntrials << ")"
			<< " x " << Ncycles << std::endl;
	}

	if (doDFTr) {
		if (Initial_Board_size * Ncycles < Nbins)
			Ncycles += 1;
	}

	Nbins = Initial_Board_size * Ncycles;
	std::cout << " NBins             " << Nbins << std::endl << std::endl;

	typedef std::uint64_t L;
	std::vector<L> vec = {};

	std::vector < std::future <decltype(vec)>> vecOfThreads;

	std::vector<std::vector<std::vector<std::uint64_t>>>
		galton_arr(N_Integrations, std::vector<std::vector<std::uint64_t>>
			(Ncycles, std::vector<std::uint64_t>(Initial_Board_size, 0ull)));

	auto begin = std::chrono::high_resolution_clock::now();

	for (U i = 0; i < N_Integrations; i++)
		for (U k = 0; k < Ncycles; k++)
			vecOfThreads.push_back(std::async([&, i, k] {
			return Galton(Ntrials / N_Integrations, Initial_Board_size, Ncycles, galton_arr[i][k], Probability_wave); }));

	for (auto& th : vecOfThreads)
		vec = th.get();

	auto end = std::chrono::high_resolution_clock::now();

	L Board_size = {};

	if (Probability_wave)
	{

		std::cout << " Inital Board size " << Initial_Board_size << "[Boxes]" << std::endl;

		Board_size = vec[1];
		std::cout << " Board size        " << Board_size << "[Boxes]" << std::endl;
		std::cout << std::endl << " RNMag             " << vec[0] << "[Boxes]" << std::endl << std::endl;

		auto Amplitude = vec[2];
		std::cout << " Amplitude         " << Amplitude << std::endl << std::endl;
		auto DC = vec.back();

		std::cout << " DC                " << DC << std::endl;
		std::cout << " DC Calculated     " << std::uint64_t(round((Ntrials / (double)Nbins) * Ncycles)) << " (Ntrials / Nbins) x Ncycles"
			<< std::endl << std::endl;
	}

	std::cout << std::endl << " Duration Ball     "
		<< std::chrono::nanoseconds(end - begin).count() / Ntrials
		<< "[ns]" << std::endl << std::endl;

	if (Initial_Board_size <= 256 && cout_gal)
		cout_galton(Initial_Board_size, galton_arr[0][0]);


	std::vector<R> X, Y, Y_buf;

	if (!Probability_wave) {

		for (auto i = 0; auto & k : std::span(galton_arr.front().front()))
			X.push_back(i++);

		Y_buf.resize(galton_arr.front().front().size() * galton_arr.front().size());

		for (auto k = 0ull; k < N_Integrations; k++) {

			Y.clear();
			for (const auto& i : galton_arr[k])
				std::ranges::transform(i, std::back_inserter(Y), [](auto& c) {return double(c); });

			Y_buf += Y;
		}

		Y_buf /= R(N_Integrations);

		Y = Y_buf;

		plot.plot_somedata(X, Y, "", "Binomial-Normal Distribution", "blue");

		std::string str = "Ntrials= ";
		str += nameForNumber(Ntrials);

		auto max = *std::ranges::max_element(Y);

		plot.text(0, max / 3, str, "green", 11);

		auto entropy = to_string_with_precision(ShannonEntropy(Y), 4);
		str = "Entropy= ";
		str += entropy;
		plot.text(0, max / 4, str, "red", 11);

		std::cout << " Entropy " << entropy << std::endl;

		plot.set_xlabel("Boxes");
		plot.set_ylabel("Frequency");
		plot.grid_on();
		plot.set_title(utf8_encode(title));
		plot.show();
	}

	else {

		Y_buf.resize(galton_arr.front().front().size() * galton_arr.front().size());

		for (auto k = 0ull; k < N_Integrations; k++) {

			Y.clear();
			for (const auto& i : galton_arr[k])
				std::ranges::transform(i, std::back_inserter(Y), [](auto& c) {return double(c); });

			Y_buf += Y;
		}

		Y_buf /= R(N_Integrations);

		std::cout << " Avarage           " << avarage_vector(Y_buf) / Ncycles << std::endl;
		std::cout << " AC Amplitude      " << ac_amplite_vector(Y_buf) / Ncycles << std::endl;

		Y = Y_buf;

		if (doDFTr)
			if (Y.size() > 2048ull)
				Y.resize(2048);

		if (!DC) {
			normalize_vector(Y, 1., -1.);
			null_offset_vector(Y);
		}

		if (Hanning_window)
			Hann_function(Y);

		Y_buf = Y;

		B pow2 = ((Y.size() & (Y.size() - 1)) == 0);

		if (WAV)
			pow2 = false;

		std::vector<MX0> cx;

		if (((Y.size() <= 1000000) || pow2) && !DC && DFT)
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

			else

				cx = doDFT(Y);

			//wavepacket(Y, Ntrials, Ncycles, false);
			//wave_packet(0, 40, 0.2);

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

			std::cout << " RMS	           " << rms << "[dB]" << std::endl << std::endl;

			auto text_x_offset = Nbins / 10; //210

			std::string str = "NCycles=";
			str += std::to_string(Ncycles);
			plot.text(text_x_offset, -8, str, "blue", 11);

			str = "Board size=";
			str += std::to_string(Y_buf.size() / Ncycles);

			if (Initial_Board_size > Board_size) str = str + ", shrunken to ";
			else if (Initial_Board_size < Board_size) str = str + ", grown to ";
			else str = str + ", size stayed the same=";
			str += to_string_with_precision(R(Board_size), 0);

			str += ", ratio=";
			if (Initial_Board_size > Board_size)
				str += to_string_with_precision(R(Initial_Board_size) / Board_size, 1);
			else if (Initial_Board_size < Board_size)
				str += to_string_with_precision(Board_size / R(Initial_Board_size), 1);
			else
				str += to_string_with_precision(R(Initial_Board_size), 1);
			plot.text(text_x_offset, -13, str, "purple", 11);

			str = "RNMag=";
			auto rng_mag = vec.front();
			str += to_string_with_precision(R(rng_mag), 0);
			plot.text(text_x_offset, -18, str, "green", 11);

			str = "Entropy=";
			auto entropy = to_string_with_precision(ShannonEntropy(Y_buf), 4);
			str += entropy;
			str += ", RMS=";
			str += to_string_with_precision(rms, 4);
			plot.text(text_x_offset, -23, str, "black", 11);

			std::cout << " Max Entropy log2(" << Nbins << ") = " <<
				to_string_with_precision(std::log2(Nbins), 6) << std::endl;
			std::cout << " Entropy Cycles[1.." << Ncycles << "] " << entropy << std::endl << std::endl;

			if (Entropy) {
				count_duplicates(Y_buf);
				cout_ShannonEntropy(Y_buf, Initial_Board_size, Ncycles);
			}

			X.clear();

			for (auto i = 0.; i < Y_buf.size() / 2.; i += 0.5)
				X.push_back(i);

			plot.line(Y.size() - R(Board_size / 2),
				R(Y.size()), -2, -2, "purple", 4, "solid");

		}

		if (!WAV)
		{
			if (!DFT || DC) {
				X.clear();
				for (auto i = 0; i < Y_buf.size(); i++)
					X.push_back(i);
			}

			std::string str = "NTrials: ";

			str += nameForNumber(Ntrials);
			str += " x ";
			str += std::to_string(Ncycles);

			plot.plot_somedata(X, Y_buf, "", str, "blue");

			if (W_DFTCoeff)
				Write_DFTCoeff(Y_buf);

			if (doDFTr) {
				auto Y = plot_doDFTr(Y_buf, true);
				if (DFTEntropy) {
					auto entropy = to_string_with_precision(ShannonEntropy(Y), 8);
					std::cout << " Max Entropy DFT log2(" << Nbins << ") = " <<
						to_string_with_precision(std::log2(Nbins), 8) << std::endl;
					std::cout << " Entropy DFT Cycles[1.." << Ncycles << "] " << entropy << std::endl;
					count_duplicates(Y);
					cout_ShannonEntropy(Y, Initial_Board_size, Ncycles);
				}
			}

			plot.set_title(utf8_encode(title));
			if (!DC)plot.set_xlabel("Bins / 2");
			else plot.set_xlabel("Bins");
			plot.grid_on();
			plot.show();

			///////////////////
			str = "NTrials: ";
			str += nameForNumber(Ntrials);
			str += " x ";
			str += std::to_string(Ncycles);

			double x = -std::numbers::pi;
			X.clear();
			for (auto& i : Y_buf) {
				i *= 3;
				x += 1. / (Y_buf.size() - 1) * 2 * std::numbers::pi;
				X.push_back(x);
			}

			plot.Py_STR("fig = plt.figure()");
			plot.Py_STR("ax = fig.add_subplot(projection = 'polar')");

			plot.plot_polar(X, Y_buf, "", str, "red", 1.0);
			std::ranges::rotate(Y_buf, Y_buf.begin() + Y_buf.size() / (2ull * Ncycles)); //Rotate left
			plot.plot_polar(X, Y_buf, "", str, "blue", 1.0);

			title = u8" Rose Curve Φ";
			plot.set_title(utf8_encode(title));

			plot.show();

			X.clear();
			Y.clear();
			
			//Peak - to - Peak Values, Ntrials = 1000000000
			Y = { 333421, 204726, 310488, 367831, };

			for (std::uint64_t i = 1; i <= Y.size(); i++)
				X.push_back((double)i);

			plot.plot_somedata_step(X, Y, "", "Peak-to-Peak Values", "blue");

			plot.set_xlabel("Frequency");
			plot.set_ylabel("Amplitude");
			plot.grid_settings(X);
			plot.grid_on();
			auto d = "Peak-to-Peak Values, Ntrials = " + nameForNumber(1000000000);
			plot.set_title(d);
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

