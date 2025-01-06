#include <vector>

#include "map.h"
#include "environment_options.h"
#include "agent.h"

#ifndef ORCA_READER_H
#define ORCA_READER_H


class Reader {
	public:
		virtual ~Reader() {}

		virtual bool ReadData() = 0;

		virtual bool GetMap(Map **map) = 0;

		virtual bool GetEnvironmentOptions(environment_options **envOpt) = 0;

		virtual bool GetAgents(std::vector<Agent *> &agents, const int &numThreshold) = 0;

		virtual Reader *Clone() const = 0;
};


#endif //ORCA_READER_H
