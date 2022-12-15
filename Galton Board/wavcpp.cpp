#include <iostream>
#include <fstream>
#include <string>
#include <vector>

struct WavSound {

public:
	// wav waveform parameters as defined by the standard
	short	num_channels{};
	int	sample_rate{};
	int	byte_rate{};
	int block_align{};
	int bits_per_sample{};

	WavSound() {};
	virtual ~WavSound() {};
};

void Set4CharsSEndFromInt
(
	const unsigned& n,
	std::vector<uint8_t>& a)
{
	a.push_back(uint8_t(n));
	a.push_back(uint8_t(n >> 8));
	a.push_back(uint8_t(n >> 16));
	a.push_back(uint8_t(n >> 24));
}

int MakeWavFromVector
(
	const std::string file_name,
	const int sample_rate_arg,
	std::vector<short int>& sound) {

	WavSound ws;

	// compute wav file parameters
	std::cout << std::endl << " Making wav file from vector" << std::endl;
	ws.sample_rate = sample_rate_arg;

	ws.bits_per_sample = 16;
	unsigned n_chars = unsigned(sound.size() * (ws.bits_per_sample / 8)); //wav file data chaunk size

	std::vector<uint8_t> raw;
	raw.reserve(44);

	// prepare header
	raw.push_back('R');
	raw.push_back('I');
	raw.push_back('F');
	raw.push_back('F');

	unsigned chunk_size = 36 + n_chars;
	Set4CharsSEndFromInt(chunk_size, raw);

	raw.push_back('W'); raw.push_back('A'); raw.push_back('V'); raw.push_back('E');
	raw.push_back('f'); raw.push_back('m'); raw.push_back('t'); raw.push_back(' ');
	
	int r = 16;
	Set4CharsSEndFromInt(r, raw);

	raw.push_back(1);
	raw.push_back(0); // PCM
	ws.num_channels = 1;
	raw.push_back(1); // number of channels
	raw.push_back(0);

	Set4CharsSEndFromInt(ws.sample_rate, raw);

	ws.bits_per_sample = 16;
	ws.byte_rate = ws.sample_rate * ws.num_channels * (ws.bits_per_sample / 8);

	std::cout << " byte_rate=" << ws.byte_rate << std::endl;
	Set4CharsSEndFromInt(ws.byte_rate, raw);

	ws.block_align = ws.num_channels * (ws.bits_per_sample / 8);
	raw.push_back((ws.block_align));
	raw.push_back((ws.block_align >> 8));
	raw.push_back((ws.bits_per_sample));
	raw.push_back((ws.bits_per_sample >> 8));

	// subchunk2
	raw.push_back('d');
	raw.push_back('a');
	raw.push_back('t');
	raw.push_back('a');

	// subchunk2 size
	Set4CharsSEndFromInt(n_chars, raw);

	// save raw data to the wav. file
	std::ofstream ofs;
	ofs.open(file_name, std::ofstream::out | std::ofstream::binary);
	if (ofs.is_open()) {
		ofs.write((char *)raw.data(), raw.size());
		ofs.write((char *)sound.data(), sound.size() * 2);
		ofs.close();
		std::cout << " Waveform data copied to wav sound" << std::endl;
		return 0;
	}
	else {
		std::cout << " Waveform data can not be copied to wav sound file" << file_name << std::endl;
		return -1;
	}

	return 0;
}