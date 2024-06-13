#include "xml_logger.h"

XMLLogger::XMLLogger(std::string fileName, std::string inpFileName) {
	this->fileName = fileName;
	this->inpFileName = inpFileName;

	this->doc = new XMLDocument();
	root = doc->NewElement(CNS_TAG_ROOT);
	doc->InsertFirstChild(root);
	log = doc->NewElement(CNS_TAG_LOG);
}


XMLLogger::XMLLogger(const XMLLogger &obj) {
	fileName = obj.fileName;
	doc = obj.doc;
	root = obj.root;
	log = obj.log;
	inpFileName = obj.inpFileName;
}


XMLLogger::~XMLLogger() {
	doc = nullptr;
	root = nullptr;
	log = nullptr;
}


bool XMLLogger::GenerateLog() {
	if (doc == nullptr || !CloneInputFile()) {
		return false;
	}
	root->InsertEndChild(log);
	return (doc->SaveFile(fileName.c_str()) == XMLError::XML_SUCCESS);
}


void XMLLogger::SetSummary(Summary &res) {
	if (doc != nullptr) {
		XMLElement *tmpsum = doc->NewElement(CNS_TAG_SUM);
		tmpsum->SetAttribute(CNS_TAG_ATTR_SR, res[CNS_SUM_SUCCESS_RATE].c_str());
		tmpsum->SetAttribute(CNS_TAG_ATTR_RUNTIME, res[CNS_SUM_RUN_TIME].c_str());
		tmpsum->SetAttribute(CNS_TAG_ATTR_FLOWTIME, res[CNS_SUM_FLOW_TIME].c_str());
		tmpsum->SetAttribute(CNS_TAG_ATTR_MAKESPAN, res[CNS_SUM_MAKESPAN].c_str());
		tmpsum->SetAttribute(CNS_TAG_ATTR_COL_AGNT, res[CNS_SUM_COLLISIONS].c_str());
		tmpsum->SetAttribute(CNS_TAG_ATTR_COL_OBST, res[CNS_SUM_COLLISIONS_OBS].c_str());
		log->InsertFirstChild(tmpsum);
	}
}


void XMLLogger::SetResults(const std::unordered_map<int, std::vector<Point>> &stepsLog,
						   const std::unordered_map<int, std::vector<Point>> &goalsLog,
						   const std::unordered_map<int, std::pair<bool, int>> &resultsLog) {
	
	if (doc != nullptr) {
		XMLElement *tmpagent, *tmppath, *tmpstep;
		int j = 0;

		for (auto &agentPath: stepsLog) {
			j = 0;
			tmpagent = doc->NewElement(CNS_TAG_AGENT);
			tmpagent->SetAttribute(CNS_TAG_ATTR_ID, agentPath.first);
			tmppath = doc->NewElement(CNS_TAG_PATH);
			tmppath->SetAttribute(CNS_TAG_ATTR_PATHFOUND, resultsLog.at(agentPath.first).first);
			tmppath->SetAttribute(CNS_TAG_ATTR_STEPS, resultsLog.at(agentPath.first).second);

			for (int i = 0; i < agentPath.second.size(); i++) {
				Point step = agentPath.second[i];
				tmpstep = doc->NewElement(CNS_TAG_STEP);
				tmpstep->SetAttribute(CNS_TAG_ATTR_NUM, j);
				tmpstep->SetAttribute(CNS_TAG_ATTR_X, step.X());
				tmpstep->SetAttribute(CNS_TAG_ATTR_Y, step.Y());

				tmpstep->SetAttribute("next.xr", goalsLog.at(agentPath.first)[i].X());
				tmpstep->SetAttribute("next.yr", goalsLog.at(agentPath.first)[i].Y());

				j++;
				tmppath->InsertEndChild(tmpstep);
			}

			tmpagent->InsertEndChild(tmppath);
			log->InsertEndChild(tmpagent);
		}
	}
}


std::string XMLLogger::GenerateLogFileName(std::string inpFileName, int agentsNum) {
	std::string str;
	str.append(inpFileName);
	size_t found = str.find_last_of(".");
	std::string piece = "_" + std::to_string(agentsNum) + "_log";
	if (found != std::string::npos)
		str.insert(found, piece);
	else {
		str.append(piece);
		str.append(".xml");
	}

	return str;
}

bool XMLLogger::CloneInputFile() {
	XMLDocument file;
	if (file.LoadFile(inpFileName.c_str()) == XMLError::XML_SUCCESS) {
		XMLNode *prev = nullptr;
		for (XMLNode *node = file.RootElement()->FirstChild(); node; node = node->NextSibling()) {
			XMLNode *clone = node->DeepClone(doc);
			if (!prev)
				doc->RootElement()->InsertFirstChild(clone);
			else
				doc->RootElement()->InsertAfterChild(prev, clone);
			prev = clone;
		}
		return true;
	}
	return false;
}

XMLLogger::XMLLogger() {
	fileName = "";
	inpFileName = "";
	doc = nullptr;
	root = nullptr;
	log = nullptr;
}

XMLLogger *XMLLogger::Clone() const {
	return new XMLLogger(*this);
}




