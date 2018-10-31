#pragma once

#ifdef _WIN32
	using filepath_string = std::wstring;
	#define standard_input_stream std::wcin
	#define OPEN_FILE _wfopen
	#define WIDEN(x) L ## x
#else
	using filepath_string = std::string;
	#define standard_input_stream std::cin
	#define OPEN_FILE fopen
	#define WIDEN(x) x
#endif