#include <iostream>
#include <string>
#include <ostream>
#include <iomanip>
#include <locale>
#include "mission.h"

#define STEP_MAX            3000
#define IS_TIME_BOUNDED     false
#define STOP_BY_SPEED       true
#define TIME_MAX            3000 * 60 * 1


int main(int argc, char *argv[]) {

	if (argc != 3) {
		//  std::cout << "Error! Invalid number of arguments.\n";
		return -1;
	}

	string taskfile = string(argv[1]); // xml file
	int num = 0;
	try {
		string tmp = string(argv[2]);
		num = stoi(tmp);
		if (num < 1) {
			throw std::invalid_argument("Number of agents must be positive");
		}
	}
	catch (std::invalid_argument) {
		//  std::cout << "Error! Invalid number of agents." << std::endl;
		return -1;
	}


	Mission task = Mission(taskfile, num, STEP_MAX, IS_TIME_BOUNDED, TIME_MAX, STOP_BY_SPEED);
	if (task.ReadTask()) {
		auto summary = task.StartMission();
		auto full_summary = summary.getFullSummary();
		std::vector<std::string> keys, values;
		for(auto it = full_summary.begin(); it != full_summary.end(); ++it) {
			keys.push_back(it->first);
			values.push_back(it->second);
		}

		for (auto &key : keys) {
			//  std::cout << std::right << std::setfill(' ') << std::setw(15) << key  << ' ';
		}
		//  std::cout << std::endl;
		for (auto &value : values) {
			//  std::cout << std::right << std::setfill(' ') << std::setw(15) << value  << ' ';
		}
		//  std::cout << std::endl;

#if FULL_LOG
		task.SaveLog();
#endif

	}
	else {
		//  std::cout << "Error during task execution\n";
		return -1;
	}

	return 0;
}
