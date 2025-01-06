// #include "xml_reader.h"


// XMLReader::XMLReader(const std::string &fileName) {
// 	this->fileName = fileName;
// 	this->doc = new XMLDocument();
// 	this->allAgents = new std::vector<Agent *>();

// 	this->map = nullptr;
// 	this->options = nullptr;
// 	this->grid = nullptr;
// 	this->obstacles = nullptr;
// 	this->root = nullptr;
// 	this->plannertype = 0;
// }


// XMLReader::XMLReader() {
// 	this->fileName = "";
// 	this->doc = nullptr;
// 	this->allAgents = nullptr;

// 	this->map = nullptr;
// 	this->options = nullptr;
// 	this->grid = nullptr;
// 	this->obstacles = nullptr;
// 	this->root = nullptr;
// 	this->plannertype = 0;
// }


// XMLReader::XMLReader(const XMLReader &obj) {

// 	fileName = obj.fileName;
// 	if (doc != nullptr) {
// 		delete doc;
// 		doc = nullptr;
// 	}


// 	if (obj.doc != nullptr) {
// 		doc = new XMLDocument();
// 		obj.doc->DeepCopy(doc);
// 		root = doc->FirstChildElement(CNS_TAG_ROOT);

// 	}


// 	plannertype = obj.plannertype;
// 	allAgents = (obj.allAgents == nullptr) ? nullptr : new std::vector<Agent *>(*(obj.allAgents));
// 	map = (obj.map == nullptr) ? nullptr : new Map(*obj.map);
// 	options = (obj.options == nullptr) ? nullptr : new environment_options(*obj.options);
// 	grid = (obj.grid == nullptr) ? nullptr : new std::vector<std::vector<int>>(*obj.grid);
// 	obstacles = (obj.obstacles == nullptr) ? nullptr : new std::vector<std::vector<Point>>(*obj.obstacles);
// }


// XMLReader::~XMLReader() {

// 	if (allAgents != nullptr) {
// 		delete allAgents;
// 		allAgents = nullptr;
// 	}

// 	if (options != nullptr) {
// 		delete options;
// 		options = nullptr;
// 	}

// 	if (map != nullptr) {
// 		delete map;
// 		map = nullptr;
// 	}
// 	if (grid != nullptr) {
// 		delete grid;
// 		grid = nullptr;
// 	}
// 	if (obstacles != nullptr) {
// 		delete obstacles;
// 		obstacles = nullptr;
// 	}

// 	if (doc != nullptr) {
// 		delete doc;
// 		doc = nullptr;
// 		root = nullptr;
// 	}

// }


// bool XMLReader::GetMap(Map **map) {
// 	if (this->map != nullptr) {
// 		*map = this->map;
// 		this->map = nullptr;
// 		return true;
// 	}

// 	return false;
// }


// bool XMLReader::GetEnvironmentOptions(environment_options **envOpt) {
// 	if (this->options != nullptr) {
// 		*envOpt = this->options;
// 		this->options = nullptr;
// 		return true;
// 	}
// 	return false;
// }


// bool XMLReader::GetAgents(std::vector<Agent *> &agents, const int &numThreshold) {

// 	if (allAgents == nullptr || allAgents->size() >= numThreshold) {
// 		agents.clear();
// 		for (int i = 0; i < numThreshold; i++) {
// 			agents.push_back((*allAgents)[i]);
// 		}
// 		for (int i = numThreshold; i < allAgents->size(); i++) {
// 			delete (*allAgents)[i];
// 		}
// 		allAgents->clear();
// 		delete allAgents;
// 		allAgents = nullptr;

// 		//  std::cout << "Agent Obtained" << " Successfully" << "\n";
// 		return true;
// 	}
// 	else {
// 		agents.clear();
// 		for (int i = 0; i < allAgents->size(); i++) {
// 			agents.push_back((*allAgents)[i]);
// 		}
// 		for (int i = allAgents->size(); i < allAgents->size(); i++) {
// 			delete (*allAgents)[i];
// 		}
// 		allAgents->clear();
// 		delete allAgents;
// 		allAgents = nullptr;

// 		//  std::cout << "Agent Obtained" << " Successfully" << "\n";
// 		return true;
// 	}
// }


// bool XMLReader::ReadData() {

// 	if (doc == nullptr || allAgents == nullptr) {
// 		return false;
// 	}

// 	/* Reading task from XML-file */

// 	if (doc->LoadFile(fileName.c_str()) != XMLError::XML_SUCCESS) {
// 		//  std::cout << "Error opening input XML file\n";
// 		return false;
// 	}

// 	root = doc->FirstChildElement(CNS_TAG_ROOT);
// 	if (!root) {
// 		//  std::cout << CNS_TAG_ROOT << " element not found in XML file\n";
// 		return false;
// 	}

// 	// bool agents_valid = ReadAgents();
// 	// //  std::cout << agents_valid << " Agents\n";

// 	return ReadMap() && ReadAlgorithmOptions() && ReadAgents();
// }

// XMLReader &XMLReader::operator=(const XMLReader &obj) {
// 	if (this != &obj) {
// 		fileName = obj.fileName;
// 		if (doc != nullptr) {
// 			delete doc;
// 		}

// 		if (obj.doc != nullptr) {
// 			doc = new XMLDocument();
// 			obj.doc->DeepCopy(doc);
// 			root = root = doc->FirstChildElement(CNS_TAG_ROOT);
// 		}
// 		else {
// 			doc = nullptr;
// 			root = nullptr;
// 		}

// 		if (allAgents != nullptr) {
// 			delete allAgents;
// 		}
// 		allAgents = (obj.allAgents == nullptr) ? nullptr : new std::vector<Agent *>(*(obj.allAgents));

// 		if (map != nullptr) {
// 			delete map;
// 		}
// 		map = (obj.map == nullptr) ? nullptr : new Map(*obj.map);

// 		if (options != nullptr) {
// 			delete options;
// 		}
// 		options = (obj.options == nullptr) ? nullptr : new environment_options(*obj.options);

// 		if (grid != nullptr) {
// 			delete grid;
// 		}
// 		grid = (obj.grid == nullptr) ? nullptr : new std::vector<std::vector<int>>(*obj.grid);

// 		if (obstacles != nullptr) {
// 			delete obstacles;
// 		}
// 		obstacles = (obj.obstacles == nullptr) ? nullptr : new std::vector<std::vector<Point>>(*obj.obstacles);

// 		plannertype = obj.plannertype;
// 	}
// 	return *this;
// }

// XMLReader *XMLReader::Clone() const {
// 	return new XMLReader(*this);
// }

// bool XMLReader::ReadMap() {
// 	XMLElement *tmpElement;
// 	// Чтение информации о карте и препятствиях / Reading information about map and obstacles //

// 	XMLElement *mapTag = 0, *element = 0, *mapnode, *obsts = 0, *obstElem = 0;

// 	std::string value;
// 	std::stringstream stream;

// 	bool hasGridMem = false, hasGrid = false, hasHeight = false, hasWidth = false, hasCellSize = false;
// 	int height = -1, width = -1, rowIter = 0, gridI = 0, gridJ = 0, obstNum = -1, count = 0;
// 	float cellSize;

// 	// Чтение карты / Map reading //

// 	mapTag = root->FirstChildElement(CNS_TAG_MAP);
// 	if (!mapTag) {
// 		//  std::cout << CNS_TAG_MAP << " element not found in XML file\n";
// 		return false;
// 	}

// 	for (mapnode = mapTag->FirstChildElement(); mapnode; mapnode = mapnode->NextSiblingElement()) {
// 		element = mapnode->ToElement();
// 		value = mapnode->Value();
// 		std::transform(value.begin(), value.end(), value.begin(), ::tolower);

// 		stream.str("");
// 		stream.clear();

// 		if (value != CNS_TAG_GRID) {
// 			stream << element->GetText();
// 		}

// 		if (!hasGridMem && hasHeight && hasWidth) {
// 			grid = new std::vector<std::vector<int>>(height);
// 			for (int i = 0; i < height; ++i)
// 				(*grid)[i] = std::vector<int>(width);
// 			hasGridMem = true;
// 		}

// 		if (value == CNS_TAG_HEIGHT) {
// 			if (hasHeight) {
// #if FULL_OUTPUT
// 				//  std::cout << "Warning! Duplicate '" << CNS_TAG_HEIGHT << "' encountered." << std::endl;
// 				//  std::cout << "Only first value of '" << CNS_TAG_HEIGHT << "' =" << height << "will be used."
// 						  << std::endl;
// #endif
// 			}
// 			else {
// 				if (!((stream >> height) && (height > 0))) {
// #if FULL_OUTPUT
// 					//  std::cout << "Warning! Invalid value of '" << CNS_TAG_HEIGHT
// 							  << "' tag encountered (or could not convert to integer)." << std::endl;
// 					//  std::cout << "Value of '" << CNS_TAG_HEIGHT << "' tag should be an integer >=0" << std::endl;
// 					//  std::cout << "Continue reading XML and hope correct value of '" << CNS_TAG_HEIGHT
// 							  << "' tag will be encountered later..." << std::endl;
// #endif
// 				}
// 				else {
// 					hasHeight = true;
// 				}
// 			}
// 		}
// 		else if (value == CNS_TAG_WIDTH) {
// 			if (hasWidth) {
// #if FULL_OUTPUT
// 				//  std::cout << "Warning! Duplicate '" << CNS_TAG_WIDTH << "' encountered." << std::endl;
// 				//  std::cout << "Only first value of '" << CNS_TAG_WIDTH << "' =" << width << "will be used." << std::endl;
// #endif
// 			}
// 			else {
// 				if (!((stream >> width) && (width > 0))) {
// #if FULL_OUTPUT
// 					//  std::cout << "Warning! Invalid value of '" << CNS_TAG_WIDTH
// 							  << "' tag encountered (or could not convert to integer)." << std::endl;
// 					//  std::cout << "Value of '" << CNS_TAG_WIDTH << "' tag should be an integer AND >0" << std::endl;
// 					//  std::cout << "Continue reading XML and hope correct value of '" << CNS_TAG_WIDTH
// 							  << "' tag will be encountered later..." << std::endl;
// #endif
// 				}
// 				else {
// 					hasWidth = true;
// 				}
// 			}
// 		}
// 		else if (value == CNS_TAG_CELLSIZE) {
// 			if (hasCellSize) {
// #if FULL_OUTPUT
// 				//  std::cout << "Warning! Duplicate '" << CNS_TAG_CELLSIZE << "' encountered." << std::endl;
// 				//  std::cout << "Only first value of '" << CNS_TAG_CELLSIZE << "' =" << cellSize << "will be used."
// 						  << std::endl;
// #endif
// 			}
// 			else {
// 				if (!((stream >> cellSize) && (cellSize > 0))) {
// #if FULL_OUTPUT
// 					//  std::cout << "Warning! Invalid value of '" << CNS_TAG_CELLSIZE
// 							  << "' tag encountered (or could not convert to double)." << std::endl;
// 					//  std::cout << "Value of '" << CNS_TAG_CELLSIZE
// 							  << "' tag should be double AND >0. By default it is defined to '1'" << std::endl;
// 					//  std::cout << "Continue reading XML and hope correct value of '" << CNS_TAG_CELLSIZE
// 							  << "' tag will be encountered later..." << std::endl;
// #endif
// 				}
// 				else
// 					hasCellSize = true;
// 			}
// 		}
// 		else if (value == CNS_TAG_GRID) {
// 			hasGrid = true;
// 			if (!(hasHeight && hasWidth)) {
// 				//  std::cout << "Error! No '" << CNS_TAG_WIDTH << "' tag or '" << CNS_TAG_HEIGHT << "' tag before '"
// 						  << CNS_TAG_GRID << "'tag encountered!" << std::endl;
// 				return false;
// 			}
// 			element = mapnode->FirstChildElement();
// 			while (gridI < height) {
// 				if (!element) {
// 					//  std::cout << "Error! Not enough '" << CNS_TAG_ROW << "' tags inside '" << CNS_TAG_GRID << "' tag."
// 							  << std::endl;
// 					//  std::cout << "Number of '" << CNS_TAG_ROW
// 							  << "' tags should be equal (or greater) than the value of '" << CNS_TAG_HEIGHT
// 							  << "' tag which is " << height << std::endl;
// 					return false;
// 				}
// 				std::string str = element->GetText();
// 				std::vector<std::string> elems;
// 				std::stringstream ss(str);
// 				std::string item;
// 				while (std::getline(ss, item, ' ')) {
// 					elems.push_back(item);
// 				}
// 				rowIter = gridJ = 0;
// 				int val;
// 				if (elems.size() > 0) {
// 					for (gridJ = 0; gridJ < width; ++gridJ) {
// 						if (gridJ == elems.size()) {
// 							break;
// 						}
// 						stream.str("");
// 						stream.clear();
// 						stream << elems[gridJ];
// 						stream >> val;
// 						(*grid)[gridI][gridJ] = val;
// 					}
// 				}
// 				if (gridJ != width) {
// 					//  std::cout << "Invalid value on " << CNS_TAG_GRID << " in the " << gridI + 1 << " " << CNS_TAG_ROW
// 							  << std::endl;
// 					return false;
// 				}
// 				++gridI;

// 				element = element->NextSiblingElement();
// 			}
// 		}
// 	}

// 	if (!hasGrid) {
// 		//  std::cout << CNS_TAG_GRID << " element not found in XML file\n";
// 		return false;
// 	}

// 	// Чтение препятствий / Obstacles reading //
// 	obsts = root->FirstChildElement(CNS_TAG_OBSTS);
// 	if (!obsts) {
// 		//  std::cout << CNS_TAG_OBSTS << " element not found in XML file\n";
// 		return false;
// 	}

// 	if (obsts->QueryIntAttribute(CNS_TAG_ATTR_NUM, &obstNum) != XMLError::XML_SUCCESS) {
// 		//  std::cout << CNS_TAG_ATTR_NUM << " element not found at " << CNS_TAG_OBSTS << " tag in XML file\n";
// 		return false;
// 	}

// 	obstacles = new std::vector<std::vector<Point>>(obstNum);
// 	float oX, oY;
// 	obstElem = obsts->FirstChildElement(CNS_TAG_OBST);
// 	for (count = 0; obstElem && count < obstNum; obstElem = obstElem->NextSiblingElement(CNS_TAG_OBST), count++) {
// 		for (tmpElement = obstElem->FirstChildElement(
// 				CNS_TAG_VERTEX); tmpElement; tmpElement = tmpElement->NextSiblingElement(CNS_TAG_VERTEX)) {
// 			if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_X, &oX) != XMLError::XML_SUCCESS) {
// 				//  std::cout << CNS_TAG_ATTR_X << " element not found at " << CNS_TAG_OBST << " #" << count + 1
// 						  << " tag in XML file\n";
// 				return false;
// 			}
// 			if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_Y, &oY) != XMLError::XML_SUCCESS) {
// 				//  std::cout << CNS_TAG_ATTR_Y << " element not found at " << CNS_TAG_OBST << " #" << count + 1
// 						  << " tag in XML file\n";
// 				return false;
// 			}
// 			(*obstacles)[count].push_back(Point(oX, oY));
// 		}

// 	}

// 	if (count != obstNum) {
// 		//  std::cout << "The number of obstacles does not match the stated in " << CNS_TAG_ATTR_NUM << "attribute of "
// 				  << CNS_TAG_OBSTS << " tag\n";
// 		return false;
// 	}

// 	map = new Map(cellSize, *grid, *obstacles);
// 	//  std::cout << "Map Phase Done" << " Successfully" << "\n";
// 	return true;
// }


// // TODO Refactor tag reading as loop and refactor constants
// bool XMLReader::ReadAlgorithmOptions() {
// 	XMLElement *tmpElement;
// 	bool tagFlag;

// 	/* Reading options of the environment and algorithm */
// 	options = new environment_options(CN_DEFAULT_METRIC_TYPE, CN_DEFAULT_BREAKINGTIES, CN_DEFAULT_ALLOWSQUEEZE,
// 									  CN_DEFAULT_CUTCORNERS,
// 									  CN_DEFAULT_HWEIGHT, CN_DEFAULT_TIME_STEP, CN_DEFAULT_DELTA,
// 									  CN_DEFAULT_MAPF_TRIGGER, CN_DEFAULT_MAPFNUM);

// 	XMLElement *alg = root->FirstChildElement(CNS_TAG_ALG);
// 	if (!alg) {
// 		//  std::cout << CNS_TAG_ALG << " element not found in XML file\n";
// 		return false;
// 	}

// 	/* Planner type */
// 	plannertype = CN_DEFAULT_ST;
// 	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_ST)) && tmpElement->GetText();
// 	if (!tagFlag) {
// #if FULL_OUTPUT
// 		//  std::cout << CNS_TAG_ST << " element not found in XML file. It was defined to " << CNS_DEFAULT_ST << "\n";
// #endif
// 	}
// 	else {
// 		auto tmpst = std::string(tmpElement->GetText());
// 		if (tmpst == CNS_SP_ST_THETA) {
// 			plannertype = CN_SP_ST_THETA;
// 		}
// 		else if (tmpst == CNS_SP_ST_DIR) {
// 			plannertype = CN_SP_ST_DIR;
// 		}
// 		else {
// 			//  std::cout << CNS_TAG_ST << " element are incorrect. It was defined to " << CNS_DEFAULT_ST << "\n";
// 		}
// 	}

// 	/* BREAKINGTIES */
// 	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_BT)) &&
// 			  (tmpElement->QueryBoolText(&(options->breakingties)) == XMLError::XML_SUCCESS);
// 	if (!tagFlag) {
// #if FULL_OUTPUT
// 		//  std::cout << CNS_TAG_BT << " element not found in XML file. It was defined to " << CN_DEFAULT_BREAKINGTIES
// 				  << "\n";
// #endif
// 	}

// 	/* ALLOWSQUEEZE */
// 	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_AS)) &&
// 			  (tmpElement->QueryBoolText(&(options->allowsqueeze)) == XMLError::XML_SUCCESS);
// 	if (!tagFlag) {
// #if FULL_OUTPUT
// 		//  std::cout << CNS_TAG_AS << " element not found in XML file. It was defined to " << CN_DEFAULT_ALLOWSQUEEZE
// 				  << "\n";
// #endif
// 	}

// 	/* CUTCORNERS */
// 	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_CC)) &&
// 			  (tmpElement->QueryBoolText(&(options->cutcorners)) == XMLError::XML_SUCCESS);
// 	if (!tagFlag) {
// #if FULL_OUTPUT
// 		//  std::cout << CNS_TAG_CC << " element not found in XML file. It was defined to " << CN_DEFAULT_CUTCORNERS
// 				  << "\n";
// #endif
// 	}

// 	/* HWEIGHT */
// 	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_HW)) &&
// 			  (tmpElement->QueryFloatText(&(options->hweight)) == XMLError::XML_SUCCESS);
// 	if (!tagFlag) {
// #if FULL_OUTPUT
// 		//  std::cout << CNS_TAG_HW << " element not found in XML file. It was defined to " << CN_DEFAULT_HWEIGHT << "\n";
// #endif
// 	}

// 	/* TIME_STEP */
// 	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_TS)) &&
// 			  (tmpElement->QueryFloatText(&(options->timestep)) == XMLError::XML_SUCCESS);
// 	if (!tagFlag) {
// #if FULL_OUTPUT
// 		//  std::cout << CNS_TAG_TS << " element not found in XML file. It was defined to " << CN_DEFAULT_TIME_STEP << "\n";
// #endif
// 	}

// 	/* DELTA */
// 	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_DEL)) &&
// 			  (tmpElement->QueryFloatText(&(options->delta)) == XMLError::XML_SUCCESS);
// 	if (!tagFlag) {
// #if FULL_OUTPUT
// 		//  std::cout << CNS_TAG_DEL << " element not found in XML file. It was defined to " << CN_DEFAULT_DELTA << "\n";
// #endif
// 	}

// 	/* MAPF_TRIGGER */
// 	std::string MAPFTrigStr = CNS_DEFAULT_MAPF_TRIGGER;
// 	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_TR)) && tmpElement->GetText();
// 	if (!tagFlag) {
// #if FULL_OUTPUT
// 		//  std::cout << CNS_TAG_TR << " element not found in XML file. It was defined to " << CNS_DEFAULT_MAPF_TRIGGER
// 				  << "\n";
// #endif
// 	}
// 	else {
// 		MAPFTrigStr = std::string(tmpElement->GetText());
// 		if (MAPFTrigStr == CNS_AT_ST_COMMONPOINT) {
// 			options->trigger = COMMON_POINT;
// 		}
// 		else if (MAPFTrigStr == CNS_AT_ST_SPEEDBUFFER) {
// 			options->trigger = SPEED_BUFFER;
// 		}
// 		else {
// 			//  std::cout << CNS_TAG_TR << " element are incorrect. It was defined to " << CNS_DEFAULT_MAPF_TRIGGER << "\n";
// 		}
// 	}

// 	/* MAPFNUM */
// 	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_MN)) &&
// 			  (tmpElement->QueryIntText(&(options->MAPFNum)) == XMLError::XML_SUCCESS);
// 	if (!tagFlag && (options->MAPFNum >= 0)) {
// #if FULL_OUTPUT
// 		//  std::cout << CNS_TAG_MN << " element not found in XML file. It was defined to " << CN_DEFAULT_MAPFNUM << "\n";
// #endif
// 		options->MAPFNum = CN_DEFAULT_MAPFNUM;
// 	}

// 	//  std::cout << "Algorithm Phase Done" << " Successfully" << "\n";
// 	return true;
// }

// bool XMLReader::ReadAgents() {
// 	XMLElement *tmpElement;
// 	int count;
// 	// Чтение информации о агентах// Reading agents information //

// 	AgentParam defaultParam = AgentParam();
// 	int agentsNumber;
// 	const char *agType = CNS_DEFAULT_AGENT_TYPE;
// 	const char *agTrigCStr = CNS_DEFAULT_MAPF_TRIGGER;

// 	XMLElement *agents = root->FirstChildElement(CNS_TAG_AGENTS);
// 	if (!agents) {
// 		//  std::cout << CNS_TAG_AGENTS << " element not found in XML file\n";
// 		return false;
// 	}

// 	/* Agent number */
// 	if (agents->QueryIntAttribute(CNS_TAG_ATTR_NUM, &agentsNumber) != XMLError::XML_SUCCESS) {
// 		//  std::cout << CNS_TAG_ATTR_NUM << " element not found at " << CNS_TAG_AGENTS << " tag in XML file\n";
// 		return false;
// 	}

// 	/* Agents type */
// 	if (agents->QueryStringAttribute(CNS_TAG_ATTR_TYPE, &agType) != XMLError::XML_SUCCESS) {
// #if FULL_OUTPUT
// 		//  std::cout << CNS_TAG_ATTR_TYPE << " element not found at " << CNS_TAG_AGENTS
// 				  << " tag in XML file. It was defined to " << CNS_DEFAULT_AGENT_TYPE << "\n";
// #endif
// 	}
// 	std::string agTypeStr = std::string(agType);

// 	/* Default parameters */
// 	tmpElement = agents->FirstChildElement(CNS_TAG_DEF_PARAMS);
// 	if (!tmpElement) {
// #if FULL_OUTPUT
// 		//  std::cout << CNS_TAG_DEF_PARAMS << " element not found in XML file. The following default values was defined\n";
// 		//  std::cout << "Default " << CNS_TAG_ATTR_SIZE << " = " << CN_DEFAULT_SIZE << "\n";
// 		//  std::cout << "Default " << CNS_TAG_ATTR_MAXSPEED << " = " << CN_DEFAULT_MAX_SPEED << "\n";
// 		//  std::cout << "Default " << CNS_TAG_ATTR_AGENTSMAXNUM << " = " << CN_DEFAULT_AGENTS_MAX_NUM << "\n";
// 		//  std::cout << "Default " << CNS_TAG_ATTR_TIMEBOUNDARY << " = " << CN_DEFAULT_TIME_BOUNDARY << "\n";
// 		//  std::cout << "Default " << CNS_TAG_ATTR_SIGHTRADIUS << " = " << CN_DEFAULT_RADIUS_OF_SIGHT << "\n";
// 		//  std::cout << "Default " << CNS_TAG_ATTR_TIMEBOUNDARYOBST << " = " << CN_DEFAULT_OBS_TIME_BOUNDARY << "\n";
// 		//  std::cout << "Default " << CNS_TAG_ATTR_REPS << " = " << CN_DEFAULT_REPS << "\n";
// #endif
// 	}
// 	else {
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_SIZE, &defaultParam.radius) != XMLError::XML_SUCCESS) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_SIZE << " element not found at " << CNS_TAG_DEF_PARAMS
// 					  << " tag. It was defined to " << CN_DEFAULT_SIZE << "\n";
// #endif
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_MAXSPEED, &defaultParam.maxSpeed) != XMLError::XML_SUCCESS) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_MAXSPEED << " element not found at " << CNS_TAG_DEF_PARAMS
// 					  << " tag. It was defined to " << CN_DEFAULT_MAX_SPEED << "\n";
// #endif
// 		}
// 		if (tmpElement->QueryIntAttribute(CNS_TAG_ATTR_AGENTSMAXNUM, &defaultParam.agentsMaxNum) !=
// 			XMLError::XML_SUCCESS) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_AGENTSMAXNUM << " element not found at " << CNS_TAG_DEF_PARAMS
// 					  << " tag. It was defined to " << CN_DEFAULT_AGENTS_MAX_NUM << "\n";
// #endif
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_TIMEBOUNDARY, &defaultParam.timeBoundary) !=
// 			XMLError::XML_SUCCESS) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_TIMEBOUNDARY << " element not found at " << CNS_TAG_DEF_PARAMS
// 					  << " tag. It was defined to " << CN_DEFAULT_TIME_BOUNDARY << "\n";
// #endif
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_SIGHTRADIUS, &defaultParam.sightRadius) !=
// 			XMLError::XML_SUCCESS) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_SIGHTRADIUS << " element not found at " << CNS_TAG_DEF_PARAMS
// 					  << " tag. It was defined to " << CN_DEFAULT_RADIUS_OF_SIGHT << "\n";
// #endif
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_TIMEBOUNDARYOBST, &defaultParam.timeBoundaryObst) !=
// 			XMLError::XML_SUCCESS) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_TIMEBOUNDARYOBST << " element not found at " << CNS_TAG_DEF_PARAMS
// 					  << " tag. It was defined to " << CN_DEFAULT_OBS_TIME_BOUNDARY << "\n";
// #endif
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_REPS, &defaultParam.rEps) != XMLError::XML_SUCCESS) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_REPS << " element not found at " << CNS_TAG_DEF_PARAMS
// 					  << " tag. It was defined to " << CN_DEFAULT_REPS << "\n";
// #endif
// 		}
// 	}
// 	tmpElement = agents->FirstChildElement(CNS_TAG_AGENT);

// 	int id;
// 	float stx = 0, sty = 0, gx = 0, gy = 0, stt = CN_DEFAULT_START_THETA;

// 	/* Each individual agent */
// 	for (count = 0; tmpElement; tmpElement = tmpElement->NextSiblingElement(CNS_TAG_AGENT), count++) {
// 		AgentParam param = AgentParam(defaultParam);
// 		stt = CN_DEFAULT_START_THETA;
// 		bool correct = true;

// 		/* Agent's ID */
// 		if (tmpElement->QueryIntAttribute(CNS_TAG_ATTR_ID, &id) != XMLError::XML_SUCCESS) {
// 			//  std::cout << CNS_TAG_ATTR_ID << " element not found in XML file at agent " << count << "\n";
// 			return false;
// 		}
// 		for (auto agent: *allAgents) {
// 			if (id == agent->GetID()) {
// 				//  std::cout << "There is an agent with same ID (" << id << ")\n";
// 				correct = false;
// 			}
// 		}

// 		/* Agent's params */
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_STX, &stx) != XMLError::XML_SUCCESS) {
// 			//  std::cout << CNS_TAG_ATTR_STX << " element not found in XML file at agent" << count << "\n";
// 			return false;
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_STY, &sty) != XMLError::XML_SUCCESS) {
// 			//  std::cout << CNS_TAG_ATTR_STY << " element not found in XML file at agent " << id << "\n";
// 			return false;
// 		}
// 		if (agTypeStr == CNS_AT_ST_ORCADD and
// 			tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_STT, &stt) != XMLError::XML_SUCCESS) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_STT << " element not found in XML file at agent " << id << " It was defined to "
// 					  << CN_DEFAULT_START_THETA << "\n";
// #endif
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_GX, &gx) != XMLError::XML_SUCCESS) {
// 			//  std::cout << CNS_TAG_ATTR_GX << " element not found in XML file at agent " << id << "\n";
// 			return false;
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_GY, &gy) != XMLError::XML_SUCCESS) {
// 			//  std::cout << CNS_TAG_ATTR_GY << " element not found in XML file at agent " << id << "\n";
// 			return false;
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_SIZE, &param.radius) != XMLError::XML_SUCCESS ||
// 			param.radius <= 0) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_SIZE << " element not found in XML file (or it is incorrect) at agent " << id
// 					  << "\n";
// #endif
// 			param.radius = defaultParam.radius;
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_MAXSPEED, &param.maxSpeed) != XMLError::XML_SUCCESS ||
// 			param.maxSpeed <= 0) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_MAXSPEED << " element not found in XML file (or it is incorrect) at agent " << id
// 					  << "\n";
// #endif
// 			param.maxSpeed = defaultParam.maxSpeed;
// 		}
// 		if (tmpElement->QueryIntAttribute(CNS_TAG_ATTR_AGENTSMAXNUM, &param.agentsMaxNum) != XMLError::XML_SUCCESS ||
// 			param.agentsMaxNum <= 0) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_AGENTSMAXNUM << " element not found in XML file (or it is incorrect) at agent "
// 					  << id << "\n";
// #endif
// 			param.agentsMaxNum = defaultParam.agentsMaxNum;
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_TIMEBOUNDARY, &param.timeBoundary) != XMLError::XML_SUCCESS ||
// 			param.timeBoundary <= 0) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_TIMEBOUNDARY << " element not found in XML file (or it is incorrect) at agent "
// 					  << id << "\n";
// #endif
// 			param.timeBoundary = defaultParam.timeBoundary;
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_SIGHTRADIUS, &param.sightRadius) != XMLError::XML_SUCCESS ||
// 			param.sightRadius <= 0) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_SIGHTRADIUS << " element not found in XML file (or it is incorrect) at agent "
// 					  << id << "\n";
// #endif
// 			param.sightRadius = defaultParam.sightRadius;
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_TIMEBOUNDARYOBST, &param.timeBoundaryObst) !=
// 			XMLError::XML_SUCCESS || param.timeBoundaryObst <= 0) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_TIMEBOUNDARYOBST
// 					  << " element not found in XML file (or it is incorrect) at agent " << id << "\n";
// #endif
// 			param.timeBoundaryObst = defaultParam.timeBoundaryObst;
// 		}
// 		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_REPS, &param.rEps) != XMLError::XML_SUCCESS ||
// 			param.rEps < 0) {
// #if FULL_OUTPUT
// 			//  std::cout << CNS_TAG_ATTR_REPS << " element not found in XML file (or it is incorrect) at agent " << id
// 					  << "\n";
// #endif
// 			param.rEps = defaultParam.rEps;
// 		}
// 		/* Checking params */
// 		// cout<<param.radius<<std::endl;
// 		if (stx <= param.radius || sty <= param.radius ||
// 			gx <= param.radius || gy <= param.radius ||
// 			stx >= (map->GetWidth() * map->GetCellSize()) - param.radius ||
// 			gx >= (map->GetWidth() * map->GetCellSize()) - param.radius ||
// 			sty >= (map->GetHeight() * map->GetCellSize()) - param.radius ||
// 			gy >= (map->GetHeight() * map->GetCellSize()) - param.radius) {
// 			//  std::cout << "Start or goal position of agent " << id << " is out of map or too close to boundaries\n";
// 			correct = false;
// 		}

// 		for (auto agent: *allAgents) {
// 			float sqRadiusSum = static_cast<float>(std::pow((agent->GetRadius() + param.radius), 2.0));
// 			float sqDist = (Point(stx, sty) - agent->GetPosition()).SquaredEuclideanNorm();
// 			if (sqDist <= sqRadiusSum) {
// 				//  std::cout << "Position of agent " << id << " is too close to another agent "<<agent->GetID()<<"\n";
// 				correct = false;
// 				break;
// 			}
// 		}
// 		Node tmpStNode = map->GetClosestNode(Point(stx, sty));
// 		Node tmpGlNode = map->GetClosestNode(Point(gx, gy));
// 		LineOfSight positionChecker(param.radius / map->GetCellSize());
// 		if (!positionChecker.checkTraversability(tmpStNode.i, tmpStNode.j, *map)) {
// 			//  std::cout << "Start position of agent " << id << " is too close to some obstacle " << tmpStNode.i<< " " <<tmpStNode.j <<" " << Point(stx, sty).ToString() <<"\n";
// 			correct = false;
// 		}
// 		if (!positionChecker.checkTraversability(tmpGlNode.i, tmpGlNode.j, *map)){
// 			//  std::cout << "Goal position of agent " << id << " is too close to some obstacle\n";
// 			correct = false;
// 		}

// 		// if (!positionChecker.checkTraversability(tmpStNode.i, tmpStNode.j, *map) ||
// 		// 	!positionChecker.checkTraversability(tmpGlNode.i, tmpGlNode.j, *map)) {
// 		// 	//  std::cout << "Position of agent " << id << " is too close to some obstacle\n";
// 		// 	correct = false;
// 		// }

// 		if (!correct) {
// #if FULL_OUTPUT
// 			//  std::cout << "Agent " << id << " was skipped\n\n";
// #endif
// 			continue;
// 		}
// #if FULL_OUTPUT
// 		//  std::cout << "Agent " << id << " was added at position " << Point(stx, sty).ToString() << "\n";
// #endif

// 		/* Creating of an agent */
// 		Agent *a;
// 		if (agTypeStr == CNS_AT_ST_ORCA) {
// 			a = new orca_agent(id, Point(stx, sty), Point(gx, gy), *map, *options, param);
// 		}
// 		else if (agTypeStr == CNS_AT_ST_ORCADD) {
// 			a = new ORCADDAgent(id, Point(stx, sty), Point(gx, gy), *map, *options, param,
// 								2 * (param.radius + param.rEps), 2 * (param.radius), stt);
// 		}
// 		else if (agTypeStr == CNS_AT_ST_ORCAPAR) {
// 			a = new agent_pnr(id, Point(stx, sty), Point(gx, gy), *map, *options, param);
// 		}
// 		else if (agTypeStr == CNS_AT_ST_ORCAPARECBS) {
// 			a = new ORCAAgentWithPARAndECBS(id, Point(stx, sty), Point(gx, gy), *map, *options, param);
// 		}
// 		else if (agTypeStr == CNS_AT_ST_ORCARETURN) {
// 			a = new ORCAAgentWithReturning(id, Point(stx, sty), Point(gx, gy), *map, *options, param);
// 		}
// 		else {
// 			a = new orca_agent(id, Point(stx, sty), Point(gx, gy), *map, *options, param);
// 		}

// 		/* Set agent's planner */
// 		// TODO Move plannertype from algorithm options to agent's parameters
// 		switch (plannertype) {
// 			case CN_SP_ST_THETA: {
// 				a->SetPlanner(ThetaStar(*map, *options, Point(stx, sty), Point(gx, gy), param.radius + param.rEps));
// 				break;
// 			}
// 			case CN_SP_ST_DIR: {
// 				a->SetPlanner(DirectPlanner(*map, *options, Point(stx, sty), Point(gx, gy), param.radius + param.rEps));
// 				break;
// 			}
// 			default: {
// 				a->SetPlanner(ThetaStar(*map, *options, Point(stx, sty), Point(gx, gy), param.radius + param.rEps));
// 				break;
// 			}
// 		}
// 		allAgents->push_back(a);
// 	}

// 	//  std::cout << "Agent Phase Done" << " Successfully" << "\n";
// 	return true;
// }

#include "xml_reader.h"


XMLReader::XMLReader(const std::string &fileName) {
	this->fileName = fileName;
	this->doc = new XMLDocument();
	this->allAgents = new std::vector<Agent *>();

	this->map = nullptr;
	this->options = nullptr;
	this->grid = nullptr;
	this->obstacles = nullptr;
	this->root = nullptr;
	this->plannertype = 0;
}


XMLReader::XMLReader() {
	this->fileName = "";
	this->doc = nullptr;
	this->allAgents = nullptr;

	this->map = nullptr;
	this->options = nullptr;
	this->grid = nullptr;
	this->obstacles = nullptr;
	this->root = nullptr;
	this->plannertype = 0;
}


XMLReader::XMLReader(const XMLReader &obj) {

	fileName = obj.fileName;
	if (doc != nullptr) {
		delete doc;
		doc = nullptr;
	}


	if (obj.doc != nullptr) {
		doc = new XMLDocument();
		obj.doc->DeepCopy(doc);
		root = doc->FirstChildElement(CNS_TAG_ROOT);

	}


	plannertype = obj.plannertype;
	allAgents = (obj.allAgents == nullptr) ? nullptr : new std::vector<Agent *>(*(obj.allAgents));
	map = (obj.map == nullptr) ? nullptr : new Map(*obj.map);
	options = (obj.options == nullptr) ? nullptr : new environment_options(*obj.options);
	grid = (obj.grid == nullptr) ? nullptr : new std::vector<std::vector<int>>(*obj.grid);
	obstacles = (obj.obstacles == nullptr) ? nullptr : new std::vector<std::vector<Point>>(*obj.obstacles);
}


XMLReader::~XMLReader() {

	if (allAgents != nullptr) {
		delete allAgents;
		allAgents = nullptr;
	}

	if (options != nullptr) {
		delete options;
		options = nullptr;
	}

	if (map != nullptr) {
		delete map;
		map = nullptr;
	}
	if (grid != nullptr) {
		delete grid;
		grid = nullptr;
	}
	if (obstacles != nullptr) {
		delete obstacles;
		obstacles = nullptr;
	}

	if (doc != nullptr) {
		delete doc;
		doc = nullptr;
		root = nullptr;
	}

}


bool XMLReader::GetMap(Map **map) {
	if (this->map != nullptr) {
		*map = this->map;
		this->map = nullptr;
		return true;
	}

	return false;
}


bool XMLReader::GetEnvironmentOptions(environment_options **envOpt) {
	if (this->options != nullptr) {
		*envOpt = this->options;
		this->options = nullptr;
		return true;
	}
	return false;
}


bool XMLReader::GetAgents(std::vector<Agent *> &agents, const int &numThreshold) {

	if (allAgents == nullptr || allAgents->size() >= numThreshold) {
		agents.clear();
		for (int i = 0; i < numThreshold; i++) {
			agents.push_back((*allAgents)[i]);
		}
		for (int i = numThreshold; i < allAgents->size(); i++) {
			delete (*allAgents)[i];
		}
		allAgents->clear();
		delete allAgents;
		allAgents = nullptr;

		return true;
	}
	return false;
}


bool XMLReader::ReadData() {

	if (doc == nullptr || allAgents == nullptr) {
		return false;
	}

	/* Reading task from XML-file */

	if (doc->LoadFile(fileName.c_str()) != XMLError::XML_SUCCESS) {
		//  std::cout << "Error opening input XML file\n";
		return false;
	}

	root = doc->FirstChildElement(CNS_TAG_ROOT);
	if (!root) {
		//  std::cout << CNS_TAG_ROOT << " element not found in XML file\n";
		return false;
	}

	return ReadMap() && ReadAlgorithmOptions() && ReadAgents();
}

XMLReader &XMLReader::operator=(const XMLReader &obj) {
	if (this != &obj) {
		fileName = obj.fileName;
		if (doc != nullptr) {
			delete doc;
		}

		if (obj.doc != nullptr) {
			doc = new XMLDocument();
			obj.doc->DeepCopy(doc);
			root = root = doc->FirstChildElement(CNS_TAG_ROOT);
		}
		else {
			doc = nullptr;
			root = nullptr;
		}

		if (allAgents != nullptr) {
			delete allAgents;
		}
		allAgents = (obj.allAgents == nullptr) ? nullptr : new std::vector<Agent *>(*(obj.allAgents));

		if (map != nullptr) {
			delete map;
		}
		map = (obj.map == nullptr) ? nullptr : new Map(*obj.map);

		if (options != nullptr) {
			delete options;
		}
		options = (obj.options == nullptr) ? nullptr : new environment_options(*obj.options);

		if (grid != nullptr) {
			delete grid;
		}
		grid = (obj.grid == nullptr) ? nullptr : new std::vector<std::vector<int>>(*obj.grid);

		if (obstacles != nullptr) {
			delete obstacles;
		}
		obstacles = (obj.obstacles == nullptr) ? nullptr : new std::vector<std::vector<Point>>(*obj.obstacles);

		plannertype = obj.plannertype;
	}
	return *this;
}

XMLReader *XMLReader::Clone() const {
	return new XMLReader(*this);
}

bool XMLReader::ReadMap() {
	XMLElement *tmpElement;
	// Чтение информации о карте и препятствиях / Reading information about map and obstacles //

	XMLElement *mapTag = 0, *element = 0, *mapnode, *obsts = 0, *obstElem = 0;

	std::string value;
	std::stringstream stream;

	bool hasGridMem = false, hasGrid = false, hasHeight = false, hasWidth = false, hasCellSize = false;
	int height = -1, width = -1, rowIter = 0, gridI = 0, gridJ = 0, obstNum = -1, count = 0;
	float cellSize;

	// Чтение карты / Map reading //

	mapTag = root->FirstChildElement(CNS_TAG_MAP);
	if (!mapTag) {
		//  std::cout << CNS_TAG_MAP << " element not found in XML file\n";
		return false;
	}

	for (mapnode = mapTag->FirstChildElement(); mapnode; mapnode = mapnode->NextSiblingElement()) {
		element = mapnode->ToElement();
		value = mapnode->Value();
		std::transform(value.begin(), value.end(), value.begin(), ::tolower);

		stream.str("");
		stream.clear();

		if (value != CNS_TAG_GRID) {
			stream << element->GetText();
		}

		if (!hasGridMem && hasHeight && hasWidth) {
			grid = new std::vector<std::vector<int>>(height);
			for (int i = 0; i < height; ++i)
				(*grid)[i] = std::vector<int>(width);
			hasGridMem = true;
		}

		if (value == CNS_TAG_HEIGHT) {
			if (hasHeight) {
#if FULL_OUTPUT
				//  std::cout << "Warning! Duplicate '" << CNS_TAG_HEIGHT << "' encountered." << std::endl;
				//  std::cout << "Only first value of '" << CNS_TAG_HEIGHT << "' =" << height << "will be used."
						//   << std::endl;
#endif
			}
			else {
				if (!((stream >> height) && (height > 0))) {
#if FULL_OUTPUT
					//  std::cout << "Warning! Invalid value of '" << CNS_TAG_HEIGHT
							//   << "' tag encountered (or could not convert to integer)." << std::endl;
					//  std::cout << "Value of '" << CNS_TAG_HEIGHT << "' tag should be an integer >=0" << std::endl;
					//  std::cout << "Continue reading XML and hope correct value of '" << CNS_TAG_HEIGHT
							//   << "' tag will be encountered later..." << std::endl;
#endif
				}
				else {
					hasHeight = true;
				}
			}
		}
		else if (value == CNS_TAG_WIDTH) {
			if (hasWidth) {
#if FULL_OUTPUT
				//  std::cout << "Warning! Duplicate '" << CNS_TAG_WIDTH << "' encountered." << std::endl;
				//  std::cout << "Only first value of '" << CNS_TAG_WIDTH << "' =" << width << "will be used." << std::endl;
#endif
			}
			else {
				if (!((stream >> width) && (width > 0))) {
#if FULL_OUTPUT
					//  std::cout << "Warning! Invalid value of '" << CNS_TAG_WIDTH
							//   << "' tag encountered (or could not convert to integer)." << std::endl;
					//  std::cout << "Value of '" << CNS_TAG_WIDTH << "' tag should be an integer AND >0" << std::endl;
					//  std::cout << "Continue reading XML and hope correct value of '" << CNS_TAG_WIDTH
							//   << "' tag will be encountered later..." << std::endl;
#endif
				}
				else {
					hasWidth = true;
				}
			}
		}
		else if (value == CNS_TAG_CELLSIZE) {
			if (hasCellSize) {
#if FULL_OUTPUT
				//  std::cout << "Warning! Duplicate '" << CNS_TAG_CELLSIZE << "' encountered." << std::endl;
				//  std::cout << "Only first value of '" << CNS_TAG_CELLSIZE << "' =" << cellSize << "will be used."
						//   << std::endl;
#endif
			}
			else {
				if (!((stream >> cellSize) && (cellSize > 0))) {
#if FULL_OUTPUT
					//  std::cout << "Warning! Invalid value of '" << CNS_TAG_CELLSIZE
							//   << "' tag encountered (or could not convert to double)." << std::endl;
					//  std::cout << "Value of '" << CNS_TAG_CELLSIZE
							//   << "' tag should be double AND >0. By default it is defined to '1'" << std::endl;
					//  std::cout << "Continue reading XML and hope correct value of '" << CNS_TAG_CELLSIZE
							//   << "' tag will be encountered later..." << std::endl;
#endif
				}
				else
					hasCellSize = true;
			}
		}
		else if (value == CNS_TAG_GRID) {
			hasGrid = true;
			if (!(hasHeight && hasWidth)) {
				//  std::cout << "Error! No '" << CNS_TAG_WIDTH << "' tag or '" << CNS_TAG_HEIGHT << "' tag before '"
						//   << CNS_TAG_GRID << "'tag encountered!" << std::endl;
				return false;
			}
			element = mapnode->FirstChildElement();
			while (gridI < height) {
				if (!element) {
					//  std::cout << "Error! Not enough '" << CNS_TAG_ROW << "' tags inside '" << CNS_TAG_GRID << "' tag."
							//   << std::endl;
					//  std::cout << "Number of '" << CNS_TAG_ROW
							//   << "' tags should be equal (or greater) than the value of '" << CNS_TAG_HEIGHT
							//   << "' tag which is " << height << std::endl;
					return false;
				}
				std::string str = element->GetText();
				std::vector<std::string> elems;
				std::stringstream ss(str);
				std::string item;
				while (std::getline(ss, item, ' ')) {
					elems.push_back(item);
				}
				rowIter = gridJ = 0;
				int val;
				if (elems.size() > 0) {
					for (gridJ = 0; gridJ < width; ++gridJ) {
						if (gridJ == elems.size()) {
							break;
						}
						stream.str("");
						stream.clear();
						stream << elems[gridJ];
						stream >> val;
						(*grid)[gridI][gridJ] = val;
					}
				}
				if (gridJ != width) {
					//  std::cout << "Invalid value on " << CNS_TAG_GRID << " in the " << gridI + 1 << " " << CNS_TAG_ROW
							//   << std::endl;
					return false;
				}
				++gridI;

				element = element->NextSiblingElement();
			}
		}
	}

	if (!hasGrid) {
		//  std::cout << CNS_TAG_GRID << " element not found in XML file\n";
		return false;
	}

	// Чтение препятствий / Obstacles reading //
	obsts = root->FirstChildElement(CNS_TAG_OBSTS);
	if (!obsts) {
		//  std::cout << CNS_TAG_OBSTS << " element not found in XML file\n";
		return false;
	}

	if (obsts->QueryIntAttribute(CNS_TAG_ATTR_NUM, &obstNum) != XMLError::XML_SUCCESS) {
		//  std::cout << CNS_TAG_ATTR_NUM << " element not found at " << CNS_TAG_OBSTS << " tag in XML file\n";
		return false;
	}

	obstacles = new std::vector<std::vector<Point>>(obstNum);
	float oX, oY;
	obstElem = obsts->FirstChildElement(CNS_TAG_OBST);
	for (count = 0; obstElem && count < obstNum; obstElem = obstElem->NextSiblingElement(CNS_TAG_OBST), count++) {
		for (tmpElement = obstElem->FirstChildElement(
				CNS_TAG_VERTEX); tmpElement; tmpElement = tmpElement->NextSiblingElement(CNS_TAG_VERTEX)) {
			if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_X, &oX) != XMLError::XML_SUCCESS) {
				//  std::cout << CNS_TAG_ATTR_X << " element not found at " << CNS_TAG_OBST << " #" << count + 1
						//   << " tag in XML file\n";
				return false;
			}
			if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_Y, &oY) != XMLError::XML_SUCCESS) {
				//  std::cout << CNS_TAG_ATTR_Y << " element not found at " << CNS_TAG_OBST << " #" << count + 1
						//   << " tag in XML file\n";
				return false;
			}
			(*obstacles)[count].push_back(Point(oX, oY));
		}

	}

	if (count != obstNum) {
		//  std::cout << "The number of obstacles does not match the stated in " << CNS_TAG_ATTR_NUM << "attribute of "
				//   << CNS_TAG_OBSTS << " tag\n";
		return false;
	}

	map = new Map(cellSize, *grid, *obstacles);
	return true;
}


// TODO Refactor tag reading as loop and refactor constants
bool XMLReader::ReadAlgorithmOptions() {
	XMLElement *tmpElement;
	bool tagFlag;

	/* Reading options of the environment and algorithm */
	options = new environment_options(CN_DEFAULT_METRIC_TYPE, CN_DEFAULT_BREAKINGTIES, CN_DEFAULT_ALLOWSQUEEZE,
									  CN_DEFAULT_CUTCORNERS,
									  CN_DEFAULT_HWEIGHT, CN_DEFAULT_TIME_STEP, CN_DEFAULT_DELTA,
									  CN_DEFAULT_MAPF_TRIGGER, CN_DEFAULT_MAPFNUM);

	XMLElement *alg = root->FirstChildElement(CNS_TAG_ALG);
	if (!alg) {
		//  std::cout << CNS_TAG_ALG << " element not found in XML file\n";
		return false;
	}

	/* Planner type */
	plannertype = CN_DEFAULT_ST;
	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_ST)) && tmpElement->GetText();
	if (!tagFlag) {
#if FULL_OUTPUT
		//  std::cout << CNS_TAG_ST << " element not found in XML file. It was defined to " << CNS_DEFAULT_ST << "\n";
#endif
	}
	else {
		auto tmpst = std::string(tmpElement->GetText());
		if (tmpst == CNS_SP_ST_THETA) {
			plannertype = CN_SP_ST_THETA;
		}
		else if (tmpst == CNS_SP_ST_DIR) {
			plannertype = CN_SP_ST_DIR;
		}
		else {
			//  std::cout << CNS_TAG_ST << " element are incorrect. It was defined to " << CNS_DEFAULT_ST << "\n";
		}
	}

	/* BREAKINGTIES */
	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_BT)) &&
			  (tmpElement->QueryBoolText(&(options->breakingties)) == XMLError::XML_SUCCESS);
	if (!tagFlag) {
#if FULL_OUTPUT
		//  std::cout << CNS_TAG_BT << " element not found in XML file. It was defined to " << CN_DEFAULT_BREAKINGTIES
				//   << "\n";
#endif
	}

	/* ALLOWSQUEEZE */
	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_AS)) &&
			  (tmpElement->QueryBoolText(&(options->allowsqueeze)) == XMLError::XML_SUCCESS);
	if (!tagFlag) {
#if FULL_OUTPUT
		//  std::cout << CNS_TAG_AS << " element not found in XML file. It was defined to " << CN_DEFAULT_ALLOWSQUEEZE
				//   << "\n";
#endif
	}

	/* CUTCORNERS */
	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_CC)) &&
			  (tmpElement->QueryBoolText(&(options->cutcorners)) == XMLError::XML_SUCCESS);
	if (!tagFlag) {
#if FULL_OUTPUT
		//  std::cout << CNS_TAG_CC << " element not found in XML file. It was defined to " << CN_DEFAULT_CUTCORNERS
				//   << "\n";
#endif
	}

	/* HWEIGHT */
	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_HW)) &&
			  (tmpElement->QueryFloatText(&(options->hweight)) == XMLError::XML_SUCCESS);
	if (!tagFlag) {
#if FULL_OUTPUT
		//  std::cout << CNS_TAG_HW << " element not found in XML file. It was defined to " << CN_DEFAULT_HWEIGHT << "\n";
#endif
	}

	/* TIME_STEP */
	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_TS)) &&
			  (tmpElement->QueryFloatText(&(options->timestep)) == XMLError::XML_SUCCESS);
	if (!tagFlag) {
#if FULL_OUTPUT
		//  std::cout << CNS_TAG_TS << " element not found in XML file. It was defined to " << CN_DEFAULT_TIME_STEP << "\n";
#endif
	}

	/* DELTA */
	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_DEL)) &&
			  (tmpElement->QueryFloatText(&(options->delta)) == XMLError::XML_SUCCESS);
	if (!tagFlag) {
#if FULL_OUTPUT
		//  std::cout << CNS_TAG_DEL << " element not found in XML file. It was defined to " << CN_DEFAULT_DELTA << "\n";
#endif
	}

	/* MAPF_TRIGGER */
	std::string MAPFTrigStr = CNS_DEFAULT_MAPF_TRIGGER;
	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_TR)) && tmpElement->GetText();
	if (!tagFlag) {
#if FULL_OUTPUT
		//  std::cout << CNS_TAG_TR << " element not found in XML file. It was defined to " << CNS_DEFAULT_MAPF_TRIGGER
				//   << "\n";
#endif
	}
	else {
		MAPFTrigStr = std::string(tmpElement->GetText());
		if (MAPFTrigStr == CNS_AT_ST_COMMONPOINT) {
			options->trigger = COMMON_POINT;
		}
		else if (MAPFTrigStr == CNS_AT_ST_SPEEDBUFFER) {
			options->trigger = SPEED_BUFFER;
		}
		else {
			//  std::cout << CNS_TAG_TR << " element are incorrect. It was defined to " << CNS_DEFAULT_MAPF_TRIGGER << "\n";
		}
	}

	/* MAPFNUM */
	tagFlag = (tmpElement = alg->FirstChildElement(CNS_TAG_MN)) &&
			  (tmpElement->QueryIntText(&(options->MAPFNum)) == XMLError::XML_SUCCESS);
	if (!tagFlag && (options->MAPFNum >= 0)) {
#if FULL_OUTPUT
		//  std::cout << CNS_TAG_MN << " element not found in XML file. It was defined to " << CN_DEFAULT_MAPFNUM << "\n";
#endif
		options->MAPFNum = CN_DEFAULT_MAPFNUM;
	}

	return true;
}

bool XMLReader::ReadAgents() {
	XMLElement *tmpElement;
	int count;
	// Чтение информации о агентах// Reading agents information //

	AgentParam defaultParam = AgentParam();
	int agentsNumber;
	const char *agType = CNS_DEFAULT_AGENT_TYPE;
	const char *agTrigCStr = CNS_DEFAULT_MAPF_TRIGGER;

	XMLElement *agents = root->FirstChildElement(CNS_TAG_AGENTS);
	if (!agents) {
		//  std::cout << CNS_TAG_AGENTS << " element not found in XML file\n";
		return false;
	}

	/* Agent number */
	if (agents->QueryIntAttribute(CNS_TAG_ATTR_NUM, &agentsNumber) != XMLError::XML_SUCCESS) {
		//  std::cout << CNS_TAG_ATTR_NUM << " element not found at " << CNS_TAG_AGENTS << " tag in XML file\n";
		return false;
	}

	/* Agents type */
	if (agents->QueryStringAttribute(CNS_TAG_ATTR_TYPE, &agType) != XMLError::XML_SUCCESS) {
#if FULL_OUTPUT
		//  std::cout << CNS_TAG_ATTR_TYPE << " element not found at " << CNS_TAG_AGENTS
				//   << " tag in XML file. It was defined to " << CNS_DEFAULT_AGENT_TYPE << "\n";
#endif
	}
	std::string agTypeStr = std::string(agType);

	/* Default parameters */
	tmpElement = agents->FirstChildElement(CNS_TAG_DEF_PARAMS);
	if (!tmpElement) {
#if FULL_OUTPUT
		//  std::cout << CNS_TAG_DEF_PARAMS << " element not found in XML file. The following default values was defined\n";
		//  std::cout << "Default " << CNS_TAG_ATTR_SIZE << " = " << CN_DEFAULT_SIZE << "\n";
		//  std::cout << "Default " << CNS_TAG_ATTR_MAXSPEED << " = " << CN_DEFAULT_MAX_SPEED << "\n";
		//  std::cout << "Default " << CNS_TAG_ATTR_AGENTSMAXNUM << " = " << CN_DEFAULT_AGENTS_MAX_NUM << "\n";
		//  std::cout << "Default " << CNS_TAG_ATTR_TIMEBOUNDARY << " = " << CN_DEFAULT_TIME_BOUNDARY << "\n";
		//  std::cout << "Default " << CNS_TAG_ATTR_SIGHTRADIUS << " = " << CN_DEFAULT_RADIUS_OF_SIGHT << "\n";
		//  std::cout << "Default " << CNS_TAG_ATTR_TIMEBOUNDARYOBST << " = " << CN_DEFAULT_OBS_TIME_BOUNDARY << "\n";
		//  std::cout << "Default " << CNS_TAG_ATTR_REPS << " = " << CN_DEFAULT_REPS << "\n";
#endif
	}
	else {
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_SIZE, &defaultParam.radius) != XMLError::XML_SUCCESS) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_SIZE << " element not found at " << CNS_TAG_DEF_PARAMS
					//   << " tag. It was defined to " << CN_DEFAULT_SIZE << "\n";
#endif
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_MAXSPEED, &defaultParam.maxSpeed) != XMLError::XML_SUCCESS) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_MAXSPEED << " element not found at " << CNS_TAG_DEF_PARAMS
					//   << " tag. It was defined to " << CN_DEFAULT_MAX_SPEED << "\n";
#endif
		}
		if (tmpElement->QueryIntAttribute(CNS_TAG_ATTR_AGENTSMAXNUM, &defaultParam.agentsMaxNum) !=
			XMLError::XML_SUCCESS) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_AGENTSMAXNUM << " element not found at " << CNS_TAG_DEF_PARAMS
					//   << " tag. It was defined to " << CN_DEFAULT_AGENTS_MAX_NUM << "\n";
#endif
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_TIMEBOUNDARY, &defaultParam.timeBoundary) !=
			XMLError::XML_SUCCESS) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_TIMEBOUNDARY << " element not found at " << CNS_TAG_DEF_PARAMS
					//   << " tag. It was defined to " << CN_DEFAULT_TIME_BOUNDARY << "\n";
#endif
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_SIGHTRADIUS, &defaultParam.sightRadius) !=
			XMLError::XML_SUCCESS) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_SIGHTRADIUS << " element not found at " << CNS_TAG_DEF_PARAMS
					//   << " tag. It was defined to " << CN_DEFAULT_RADIUS_OF_SIGHT << "\n";
#endif
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_TIMEBOUNDARYOBST, &defaultParam.timeBoundaryObst) !=
			XMLError::XML_SUCCESS) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_TIMEBOUNDARYOBST << " element not found at " << CNS_TAG_DEF_PARAMS
					//   << " tag. It was defined to " << CN_DEFAULT_OBS_TIME_BOUNDARY << "\n";
#endif
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_REPS, &defaultParam.rEps) != XMLError::XML_SUCCESS) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_REPS << " element not found at " << CNS_TAG_DEF_PARAMS
					//   << " tag. It was defined to " << CN_DEFAULT_REPS << "\n";
#endif
		}
	}
	tmpElement = agents->FirstChildElement(CNS_TAG_AGENT);

	int id;
	float stx = 0, sty = 0, gx = 0, gy = 0, stt = CN_DEFAULT_START_THETA;

	/* Each individual agent */
	for (count = 0; tmpElement; tmpElement = tmpElement->NextSiblingElement(CNS_TAG_AGENT), count++) {
		AgentParam param = AgentParam(defaultParam);
		stt = CN_DEFAULT_START_THETA;
		bool correct = true;

		/* Agent's ID */
		if (tmpElement->QueryIntAttribute(CNS_TAG_ATTR_ID, &id) != XMLError::XML_SUCCESS) {
			//  std::cout << CNS_TAG_ATTR_ID << " element not found in XML file at agent " << count << "\n";
			return false;
		}
		for (auto agent: *allAgents) {
			if (id == agent->GetID()) {
				//  std::cout << "There is an agent with same ID (" << id << ")\n";
				correct = false;
			}
		}

		/* Agent's params */
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_STX, &stx) != XMLError::XML_SUCCESS) {
			//  std::cout << CNS_TAG_ATTR_STX << " element not found in XML file at agent" << count << "\n";
			return false;
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_STY, &sty) != XMLError::XML_SUCCESS) {
			//  std::cout << CNS_TAG_ATTR_STY << " element not found in XML file at agent " << id << "\n";
			return false;
		}
		if (agTypeStr == CNS_AT_ST_ORCADD and
			tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_STT, &stt) != XMLError::XML_SUCCESS) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_STT << " element not found in XML file at agent " << id << " It was defined to "
					//   << CN_DEFAULT_START_THETA << "\n";
#endif
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_GX, &gx) != XMLError::XML_SUCCESS) {
			//  std::cout << CNS_TAG_ATTR_GX << " element not found in XML file at agent " << id << "\n";
			return false;
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_GY, &gy) != XMLError::XML_SUCCESS) {
			//  std::cout << CNS_TAG_ATTR_GY << " element not found in XML file at agent " << id << "\n";
			return false;
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_SIZE, &param.radius) != XMLError::XML_SUCCESS ||
			param.radius <= 0) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_SIZE << " element not found in XML file (or it is incorrect) at agent " << id
					//   << "\n";
#endif
			param.radius = defaultParam.radius;
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_MAXSPEED, &param.maxSpeed) != XMLError::XML_SUCCESS ||
			param.maxSpeed <= 0) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_MAXSPEED << " element not found in XML file (or it is incorrect) at agent " << id
					//   << "\n";
#endif
			param.maxSpeed = defaultParam.maxSpeed;
		}
		if (tmpElement->QueryIntAttribute(CNS_TAG_ATTR_AGENTSMAXNUM, &param.agentsMaxNum) != XMLError::XML_SUCCESS ||
			param.agentsMaxNum <= 0) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_AGENTSMAXNUM << " element not found in XML file (or it is incorrect) at agent "
					//   << id << "\n";
#endif
			param.agentsMaxNum = defaultParam.agentsMaxNum;
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_TIMEBOUNDARY, &param.timeBoundary) != XMLError::XML_SUCCESS ||
			param.timeBoundary <= 0) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_TIMEBOUNDARY << " element not found in XML file (or it is incorrect) at agent "
					//   << id << "\n";
#endif
			param.timeBoundary = defaultParam.timeBoundary;
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_SIGHTRADIUS, &param.sightRadius) != XMLError::XML_SUCCESS ||
			param.sightRadius <= 0) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_SIGHTRADIUS << " element not found in XML file (or it is incorrect) at agent "
					//   << id << "\n";
#endif
			param.sightRadius = defaultParam.sightRadius;
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_TIMEBOUNDARYOBST, &param.timeBoundaryObst) !=
			XMLError::XML_SUCCESS || param.timeBoundaryObst <= 0) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_TIMEBOUNDARYOBST
					//   << " element not found in XML file (or it is incorrect) at agent " << id << "\n";
#endif
			param.timeBoundaryObst = defaultParam.timeBoundaryObst;
		}
		if (tmpElement->QueryFloatAttribute(CNS_TAG_ATTR_REPS, &param.rEps) != XMLError::XML_SUCCESS ||
			param.rEps < 0) {
#if FULL_OUTPUT
			//  std::cout << CNS_TAG_ATTR_REPS << " element not found in XML file (or it is incorrect) at agent " << id
					//   << "\n";
#endif
			param.rEps = defaultParam.rEps;
		}
		/* Checking params */
		// cout<<param.radius<<std::endl;
		if (stx <= param.radius || sty <= param.radius ||
			gx <= param.radius || gy <= param.radius ||
			stx >= (map->GetWidth() * map->GetCellSize()) - param.radius ||
			gx >= (map->GetWidth() * map->GetCellSize()) - param.radius ||
			sty >= (map->GetHeight() * map->GetCellSize()) - param.radius ||
			gy >= (map->GetHeight() * map->GetCellSize()) - param.radius) {
			//  std::cout << "Start or goal position of agent " << id << " is out of map or too close to boundaries\n";
			correct = false;
		}

		for (auto agent: *allAgents) {
			float sqRadiusSum = static_cast<float>(std::pow((agent->GetRadius() + param.radius), 2.0));
			float sqDist = (Point(stx, sty) - agent->GetPosition()).SquaredEuclideanNorm();
			if (sqDist <= sqRadiusSum) {
				//  std::cout << "Position of agent " << id << " is too close to another agent "<<agent->GetID()<<"\n";
				correct = false;
				break;
			}
		}
		Node tmpStNode = map->GetClosestNode(Point(stx, sty));
		Node tmpGlNode = map->GetClosestNode(Point(gx, gy));
		LineOfSight positionChecker(param.radius / map->GetCellSize());
		if (!positionChecker.checkTraversability(tmpStNode.i, tmpStNode.j, *map)) {
			//  std::cout << "Start position of agent " << id << " is too close to some obstacle " << tmpStNode.i<< " " <<tmpStNode.j <<" " << Point(stx, sty).ToString() <<"\n";
			correct = false;
		}
		if (!positionChecker.checkTraversability(tmpGlNode.i, tmpGlNode.j, *map)){
			//  std::cout << "Goal position of agent " << id << " is too close to some obstacle\n";
			correct = false;
		}

		// if (!positionChecker.checkTraversability(tmpStNode.i, tmpStNode.j, *map) ||
		// 	!positionChecker.checkTraversability(tmpGlNode.i, tmpGlNode.j, *map)) {
		// 	//  std::cout << "Position of agent " << id << " is too close to some obstacle\n";
		// 	correct = false;
		// }

		if (!correct) {
#if FULL_OUTPUT
			//  std::cout << "Agent " << id << " was skipped\n\n";
#endif
			continue;
		}
#if FULL_OUTPUT
		//  std::cout << "Agent " << id << " was added at position " << Point(stx, sty).ToString() << "\n";
#endif

		/* Creating of an agent */
		Agent *a;
		if (agTypeStr == CNS_AT_ST_ORCA) {
			a = new orca_agent(id, Point(stx, sty), Point(gx, gy), *map, *options, param);
		}
		else if (agTypeStr == CNS_AT_ST_ORCADD) {
			a = new ORCADDAgent(id, Point(stx, sty), Point(gx, gy), *map, *options, param,
								2 * (param.radius + param.rEps), 2 * (param.radius), stt);
		}
		else if (agTypeStr == CNS_AT_ST_ORCAPAR) {
			a = new agent_pnr(id, Point(stx, sty), Point(gx, gy), *map, *options, param);
		}
		else if (agTypeStr == CNS_AT_ST_ORCAPARECBS) {
			a = new ORCAAgentWithPARAndECBS(id, Point(stx, sty), Point(gx, gy), *map, *options, param);
		}
		else if (agTypeStr == CNS_AT_ST_ORCARETURN) {
			a = new ORCAAgentWithReturning(id, Point(stx, sty), Point(gx, gy), *map, *options, param);
		}
		else {
			a = new orca_agent(id, Point(stx, sty), Point(gx, gy), *map, *options, param);
		}

		/* Set agent's planner */
		// TODO Move plannertype from algorithm options to agent's parameters
		switch (plannertype) {
			case CN_SP_ST_THETA: {
				a->SetPlanner(ThetaStar(*map, *options, Point(stx, sty), Point(gx, gy), param.radius + param.rEps));
				break;
			}
			case CN_SP_ST_DIR: {
				a->SetPlanner(DirectPlanner(*map, *options, Point(stx, sty), Point(gx, gy), param.radius + param.rEps));
				break;
			}
			default: {
				a->SetPlanner(ThetaStar(*map, *options, Point(stx, sty), Point(gx, gy), param.radius + param.rEps));
				break;
			}
		}
		allAgents->push_back(a);
	}
	return true;
}
