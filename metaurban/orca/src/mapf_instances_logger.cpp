#include "mapf_instances_logger.h"


MAPFInstancesLogger::MAPFInstancesLogger(std::string pathTempl) {
	fileID = 0;
	pathTemplate = pathTempl;
//	auto found = pathTemplate.find_last_of("/");
//	auto piece = "/mapf/";
//	pathTemplate.erase(found, 1);
//	pathTemplate.insert(found, piece);

}


bool MAPFInstancesLogger::SaveInstance(MAPFActorSet &agents, SubMap &map, MAPFConfig &conf) {

	XMLDocument *docMainXML, *docAgentsXML;
	std::string tmp1 = pathTemplate;
	std::string tmp2 = pathTemplate;


	size_t found = tmp1.find_last_of("/");
	std::string piece = "/MAP_" + std::to_string(fileID) + "_";
	tmp1.erase(found, 1);
	tmp1.insert(found, piece);

	found = tmp2.find_last_of("/");
	piece = "/AGENT_" + std::to_string(fileID) + "_";
	tmp2.erase(found, 1);
	tmp2.insert(found, piece);
	std::string agentSmall = tmp2;
	agentSmall.erase(0, found + 1);


	std::string mainFile = tmp1 + ".xml";
	std::string agentFile = tmp2 + "-1.xml";
	/* MAIN ROOT */
	docMainXML = new XMLDocument();
	auto rootXML = docMainXML->NewElement(CNS_TAG_ROOT);
	docMainXML->InsertFirstChild(rootXML);

	/* MAP */
	auto mapXML = docMainXML->NewElement(CNS_TAG_MAP);
	auto gridXML = docMainXML->NewElement(CNS_TAG_GRID);


	gridXML->SetAttribute(CNS_TAG_WIDTH, map.GetWidth());
	gridXML->SetAttribute(CNS_TAG_HEIGHT, map.GetHeight());

	for (size_t i = 0; i < map.GetHeight(); i++) {
		auto rowXML = docMainXML->NewElement(CNS_TAG_ROW);

		std::string rowStr = "";
		for (size_t j = 0; j < map.GetWidth(); j++) {
			if (j != 0) {
				rowStr += " ";
			}

			if (map.CellIsObstacle(i, j)) {
				rowStr += "1";
			}
			else {
				rowStr += "0";
			}
		}

		rowXML->SetText(rowStr.c_str());
		gridXML->InsertEndChild(rowXML);
	}
	mapXML->InsertEndChild(gridXML);
	rootXML->InsertFirstChild(mapXML);
	/* ALGORITHM */
	auto algorithmXML = docMainXML->NewElement(CNS_TAG_ALG);

	auto boolToString = [](bool arg) {
		return (arg) ? "true" : "false";
	};

	std::string algTagName[] =
			{
					"planner",
					"low_level",
					"with_cat",
					"with_perfect_h",
					"with_card_conf",
					"with_bypassing",
					"with_matching_h",
					"with_disjoint_splitting",
					"focal_w",
					"pp_order",
					"parallelize_paths_1",
					"parallelize_paths_2"
			};
	vector<string> algTagText =
			{
					conf.planner,
					conf.lowLevel,
					boolToString(conf.withCAT),
					boolToString(conf.withPerfectHeuristic),
					boolToString(conf.withCardinalConflicts),
					boolToString(conf.withBypassing),
					boolToString(conf.withMatchingHeuristic),
					boolToString(conf.withDisjointSplitting),
					std::to_string(conf.focalW),
					"0",
					boolToString(conf.parallelizePaths1),
					boolToString(conf.parallelizePaths2),

			};

	for (size_t tag = 0; tag < 12; tag++) {
		auto algTagXML = docMainXML->NewElement(algTagName[tag].c_str());
		algTagXML->SetText(algTagText[tag].c_str());
		algorithmXML->InsertEndChild(algTagXML);
	}

	rootXML->InsertEndChild(algorithmXML);

	/* OPTIONS */
	auto optionsXML = docMainXML->NewElement(CNS_TAG_OPT);

	std::string optTagNames[] = {

			"agents_file",
			"tasks_count",
			"maxtime",
			"single_execution"
	};

	std::string optTagText[] = {
			"",
			"1",
			"",
			"false"
	};

	optTagText[0] = agentSmall;
	optTagText[2] = std::to_string(conf.maxTime);

	for (size_t tag = 0; tag < 4; tag++) {
		auto optTagXML = docMainXML->NewElement(optTagNames[tag].c_str());
		optTagXML->SetText(optTagText[tag].c_str());
		optionsXML->InsertEndChild(optTagXML);
	}

	auto rangeXML = docMainXML->NewElement("agents_range");
	rangeXML->SetAttribute("min", agents.getActorCount());
	rangeXML->SetAttribute("max", agents.getActorCount());

	optionsXML->InsertEndChild(rangeXML);

	auto logpathXML = docMainXML->NewElement("logpath");
	auto logfilenameXML = docMainXML->NewElement("logfilename");
	optionsXML->InsertEndChild(logpathXML);
	optionsXML->InsertEndChild(logfilenameXML);
	rootXML->InsertEndChild(optionsXML);


	/* SAVE MAIN*/
	bool resMain = (docMainXML->SaveFile(mainFile.c_str()) == XMLError::XML_SUCCESS);
	docAgentsXML = new XMLDocument();
	rootXML = docAgentsXML->NewElement(CNS_TAG_ROOT);
	docAgentsXML->InsertFirstChild(rootXML);
	for (size_t agent = 0; agent < agents.getActorCount(); agent++) {
		auto agentXML = docAgentsXML->NewElement("agent");
		agentXML->SetAttribute("id", (int) agent);
		agentXML->SetAttribute("start_i", agents.getActor(agent).getStart_i());
		agentXML->SetAttribute("start_j", agents.getActor(agent).getStart_j());
		agentXML->SetAttribute("goal_i", agents.getActor(agent).getGoal_i());
		agentXML->SetAttribute("goal_j", agents.getActor(agent).getGoal_j());

		rootXML->InsertEndChild(agentXML);
	}


	bool resAgents = (docAgentsXML->SaveFile(agentFile.c_str()) == XMLError::XML_SUCCESS);

	fileID++;
	return resMain && resAgents;

}

//void MAPFInstancesLogger::AddResults(const MAPFSearchResult &result)
//{
//
//    size_t makespan = 0, timeflow = 0;
//    if(result.pathfound)
//        for (int i = 0; i < result.agentsPaths->size(); i++)
//        {
//            makespan = std::max(makespan, result.agentsPaths->at(i).size() - 1);
//            int lastMove;
//            for (lastMove = result.agentsPaths->at(i).size() - 1; lastMove > 1 && result.agentsPaths->at(i)[lastMove] == result.agentsPaths->at(i)[lastMove - 1]; --lastMove);
//            timeflow += lastMove;
//        }
//    pre_log << result.pathfound << "\t" << result.time << "\t" << makespan << "\t" << timeflow << "\t" << result.HLExpansions << "\t" <<result.HLNodes << "\n";
//
//    return ;
//}

MAPFInstancesLogger::~MAPFInstancesLogger() {
//    pre_log.close();
}

MAPFInstancesLogger::MAPFInstancesLogger(const MAPFInstancesLogger &obj) {
	this->fileID = obj.fileID;
	this->pathTemplate = obj.pathTemplate;
//    this->resPath = obj.resPath;
//    this->pre_log.close();
//    this->pre_log.open(this->resPath);
}

MAPFInstancesLogger &MAPFInstancesLogger::operator=(const MAPFInstancesLogger &obj) {
	if (this != &obj) {
		this->fileID = obj.fileID;
		this->pathTemplate = obj.pathTemplate;
//        this->resPath = obj.resPath;
//        this->pre_log.close();
//        this->pre_log.open(this->resPath);
	}
	return *this;
}


size_t MAPFInstancesLogger::GetFileID() const {
	return fileID;
}