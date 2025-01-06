#include <string>
#include <sstream>

#include "tinyxml2.h"
#include "reader.h"
#include "const.h"
#include "agent.h"
#import "orca_agent.h"
#import "orca_diff_drive_agent.h"
#include "thetastar.h"
#include "direct_planner.h"
#include "agent_pnr.h"
#include "agent_pnr_ecbs.h"
#include "agent_returning.h"


#ifndef ORCA_XMLREADER_H
#define ORCA_XMLREADER_H


using namespace tinyxml2;

class XMLReader : public Reader {
	public:
		XMLReader();

		XMLReader(const std::string &fileName);

		XMLReader(const XMLReader &obj);

		~XMLReader() override;

		bool ReadData() override;

		bool GetMap(Map **map) override;

		bool GetEnvironmentOptions(environment_options **envOpt) override;

		bool GetAgents(std::vector<Agent *> &agents, const int &numThreshold) override;

		XMLReader *Clone() const override;

		XMLReader &operator=(const XMLReader &obj);


	private:
		std::string fileName;

		XMLDocument *doc;
		XMLElement *root;

		std::vector<Agent *> *allAgents;
		Map *map;
		environment_options *options;
		std::vector<std::vector<int>> *grid;
		std::vector<std::vector<Point>> *obstacles;
		int plannertype;

		bool ReadMap();

		bool ReadAgents();

		bool ReadAlgorithmOptions();

};


#endif //ORCA_XMLREADER_H
