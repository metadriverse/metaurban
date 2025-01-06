#include <ostream>
#include <string>
#include <map>

#ifndef ORCA_SUMMARY_H
#define ORCA_SUMMARY_H


class Summary {
	public:
		Summary() = default;

		~Summary() = default;

		std::map<std::string, std::string> getFullSummary() {
			return fields;
		}

		std::string &operator[](std::string idx) {
			return fields[idx];
		}

	private:
		std::map<std::string, std::string> fields;


};


#endif //ORCA_SUMMARY_H
