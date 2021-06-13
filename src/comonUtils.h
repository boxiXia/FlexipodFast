#pragma once

#include <sstream>
#include <fstream>
#include<iostream>
#include<string>
#include <chrono> // for time measurement


std::string getWorkingDir(); // returns the working directory from where the program is called
std::string getProgramDir(); // return the directory where this program is located


/* helper class for measuring time difference
usage:
	tm = Timer();
	// ....do something...
	auto duration = tm.dtMicroSeconds();
*/
class Timer {
public:
	std::chrono::steady_clock::time_point start;
	std::chrono::steady_clock::time_point end;
	/*reset the start time*/
	void reset() { start = std::chrono::steady_clock::now(); }
	Timer() {
		reset();
	}
	/*return raw duration*/
	auto duration() {
		end = std::chrono::steady_clock::now();
		return end - start;
	}
	/*return duration in seconds */
	double dtSeconds() {
		return ((double)std::chrono::duration_cast<
			std::chrono::nanoseconds>(duration()).count()) / 1e9;
	}
	/*return duration in miliseconds */
	double dtMiliSeconds() {
		return ((double)std::chrono::duration_cast<
			std::chrono::nanoseconds>(duration()).count()) / 1e6;
	}
	/*return duration in miliseconds */
	double dtMicroSeconds() {
		return ((double)std::chrono::duration_cast<
			std::chrono::nanoseconds>(duration()).count()) / 1e3;
	}
};