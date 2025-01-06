#ifndef ORCA_LOGGER_H
#define ORCA_LOGGER_H

#include <string>
#include <vector>

#include "summary.h"
#include "geom.h"
#include "agent.h"

class Logger {
	public:
		virtual ~Logger() {};

		virtual bool GenerateLog() = 0;

		virtual void SetSummary(Summary &res) = 0;

		virtual void
		SetResults(const std::unordered_map<int, std::vector<Point>> &stepsLog,
				   const std::unordered_map<int, std::vector<Point>> &goalsLog,
				   const std::unordered_map<int, std::pair<bool, int>> &resultsLog) = 0;

		virtual Logger *Clone() const = 0;
};


#endif //ORCA_LOGGER_H
