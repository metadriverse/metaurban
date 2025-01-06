#ifndef ORCA_XMLLOGGER_H
#define ORCA_XMLLOGGER_H

#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>

#include "tinyxml2.h"
#include "logger.h"

using namespace std;
using namespace tinyxml2;

class XMLLogger : public Logger {
	public:
		XMLLogger();

		XMLLogger(std::string fileName, std::string inpFileName);

		XMLLogger(const XMLLogger &obj);

		~XMLLogger() override;

		bool GenerateLog() override;

		void SetSummary(Summary &res) override;

		void
		SetResults(const std::unordered_map<int, std::vector<Point>> &stepsLog,
				   const std::unordered_map<int, std::vector<Point>> &goalsLog,
				   const std::unordered_map<int, std::pair<bool, int>> &resultsLog) override;

		XMLLogger *Clone() const override;

		static std::string GenerateLogFileName(std::string inpFileName, int agentsNum);

	private:
		bool CloneInputFile();

		std::string fileName;
		std::string inpFileName;
		XMLDocument *doc;
		XMLElement *root;
		XMLElement *log;

};


#endif //ORCA_XMLLOGGER_H
