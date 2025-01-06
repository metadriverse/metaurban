#include "agent.h"


Agent::Agent() {
	id = -1;
	start = Point();
	goal = Point();
	planner = nullptr;
	options = nullptr;
	map = nullptr;
	Neighbours = std::vector<std::pair<float, Agent *>>();
	NeighboursObst = std::vector<std::pair<float, ObstacleSegment>>();
	ORCALines = std::vector<Line>();
	position = Point();
	prefV = Point();
	newV = Point();
	currV = Point();
	param = AgentParam();
	invTimeBoundaryObst = 0;
	invTimeBoundary = 0;
	collisions = 0;
	collisionsObst = 0;
	maxSqObstDist = 0;
	speedSaveBuffer = std::list<float>(SPEED_BUFF_SIZE, 1.0f);
	meanSavedSpeed = 1.0f;
}


Agent::Agent(const int &id, const Point &start, const Point &goal, const Map &map, const environment_options &options,
			 AgentParam param) {
	this->id = id;
	this->start = start;
	this->goal = goal;
	this->map = &map;
	this->options = &options;
	this->param = param;

	Neighbours = std::vector<std::pair<float, Agent *>>();
	Neighbours.reserve((unsigned int) param.agentsMaxNum);
	NeighboursObst = std::vector<std::pair<float, ObstacleSegment>>();
	ORCALines = std::vector<Line>();
	ORCALines.reserve((unsigned int) param.agentsMaxNum);
	position = start;
	prefV = Point();
	newV = Point();
	currV = Point();
	invTimeBoundaryObst = 1 / param.timeBoundaryObst;
	invTimeBoundary = 1 / param.timeBoundary;
	collisions = 0;
	collisionsObst = 0;
	float maxObstDist = param.maxSpeed;
	maxSqObstDist = maxObstDist * maxObstDist;
	//maxSqObstDist = std::pow((param.sightRadius + param.radius), 2.0f);
	// maxSqObstDist = std::pow((param.maxSpeed * param.timeBoundaryObst + param.radius), 2.0f);
	speedSaveBuffer = std::list<float>(SPEED_BUFF_SIZE, param.maxSpeed);
	meanSavedSpeed = 1.0f;

}


Agent::Agent(const Agent &obj) {
	this->id = obj.id;
	this->start = obj.start;
	this->goal = obj.goal;
	this->map = obj.map;
	this->options = obj.options;
	this->param = obj.param;

	if (obj.planner != nullptr) {
		this->planner = obj.planner->Clone();
	}

	Neighbours = obj.Neighbours;
	NeighboursObst = obj.NeighboursObst;
	ORCALines = obj.ORCALines;
	position = obj.position;
	prefV = obj.prefV;
	newV = obj.newV;
	currV = obj.currV;
	invTimeBoundaryObst = obj.invTimeBoundaryObst;
	invTimeBoundary = obj.invTimeBoundary;
	collisions = obj.collisions;
	collisionsObst = obj.collisionsObst;
	maxSqObstDist = obj.maxSqObstDist;
	speedSaveBuffer = obj.speedSaveBuffer;
	meanSavedSpeed = obj.meanSavedSpeed;
}


Agent::~Agent() {
	if (planner != nullptr) {
		delete planner;
		planner = nullptr;
	}
	options = nullptr;
	map = nullptr;
	Neighbours.clear();
}


bool Agent::isFinished() {
	return ((this->position - this->goal).EuclideanNorm() < options->delta);
}


bool Agent::operator==(const Agent &another) const {
	return this->id == another.id;
}


bool Agent::operator!=(const Agent &another) const {
	return this->id != another.id;
}


void Agent::AddNeighbour(Agent &neighbour, float distSq) {
	float sightSq = param.sightRadius * param.sightRadius;

	if (distSq >= sightSq) {
		return;
	}
	int i = 0;
	auto tmpit = Neighbours.begin();
	while (tmpit != Neighbours.end() && i < param.agentsMaxNum && Neighbours[i].first < distSq) {
		i++;
		tmpit++;
	}
	if (i < param.agentsMaxNum) {
		Neighbours.insert(tmpit, std::pair<float, Agent *>(distSq, &neighbour));
	}

}


void Agent::SetPosition(const Point &pos) {
	position = pos;
}


Point Agent::GetPosition() const {
	return position;
}


void Agent::UpdateNeighbourObst() {
	NeighboursObst.clear();
	std::vector<std::vector<ObstacleSegment>> tmpObstacles = map->GetObstacles();
	float distSq = 0;

	for (int i = 0; i < tmpObstacles.size(); i++) {
		for (int j = 0; j < tmpObstacles[i].size(); j++) {

			distSq = Utils::SqPointSegDistance(static_cast<Point>(tmpObstacles[i][j].left),
											   static_cast<Point>(tmpObstacles[i][j].right), position);
			if (distSq < maxSqObstDist) {
				NeighboursObst.push_back({distSq, tmpObstacles[i][j]});
			}
		}
	}

	std::sort(NeighboursObst.begin(), NeighboursObst.end(), Utils::Less<ObstacleSegment>);
}


Point Agent::GetVelocity() const {
	return currV;
}


std::pair<unsigned int, unsigned int> Agent::GetCollision() const {
	return {collisions, collisionsObst};
}


bool Agent::InitPath() {
	return planner->CreateGlobalPath();
}


int Agent::GetID() const {
	return id;
}


float Agent::GetRadius() const {
	return param.radius;
}


Agent &Agent::operator=(const Agent &obj) {

	if (this != &obj) {
		start = obj.start;
		goal = obj.goal;
		id = obj.id;
		position = obj.position;
		prefV = obj.prefV;
		newV = obj.newV;
		currV = obj.currV;
		param = obj.param;
		invTimeBoundaryObst = obj.invTimeBoundaryObst;
		invTimeBoundary = obj.invTimeBoundary;
		maxSqObstDist = obj.maxSqObstDist;
		collisions = obj.collisions;
		collisionsObst = obj.collisionsObst;
		ORCALines = obj.ORCALines;
		NeighboursObst = obj.NeighboursObst;
		Neighbours = obj.Neighbours;
		options = obj.options;
		map = obj.map;
		speedSaveBuffer = obj.speedSaveBuffer;
		meanSavedSpeed = obj.meanSavedSpeed;
		if (planner != nullptr) {
			delete planner;
			planner = nullptr;
		}
		planner = (obj.planner == nullptr) ? nullptr : obj.planner->Clone();
	}
	return *this;
}


Point Agent::GetNext() const {
	return nextForLog;
}


bool Agent::CommonPointMAPFTrigger(float distToTargetPoint) {
	return (Neighbours.size() >= options->MAPFNum) && (distToTargetPoint < param.sightRadius);
}


//bool Agent::MeanSpeedMAPFTrigger()
//{
//    float mean = 0.0f;
//
//    if(options->MAPFNum > Neighbours.size()) return false;
//
//    for(size_t nCount = 0; nCount < options->MAPFNum; nCount++)
//    {
//        mean += Neighbours[nCount].second->GetVelocity().EuclideanNorm();
//    }
//    mean /= static_cast<float>(options->MAPFNum);
//
//    return (mean < SMALL_SPEED);
//}


bool Agent::NeighbourGroupMeanSpeedMAPFTrigger() {
	size_t nNum = (options->MAPFNum > Neighbours.size()) ? Neighbours.size() : options->MAPFNum;
	if (!nNum) return false;

	// Kahan summation algorithm
	float sum = 0.0f;
	float c = 0.0f;
	float y, t;

	for (size_t nCount = 0; nCount < nNum; nCount++) {
		y = Neighbours[nCount].second->meanSavedSpeed - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}

	float mean = sum / nNum;
	return (mean < SMALL_SPEED);
}


bool Agent::SingleNeighbourMeanSpeedMAPFTrigger() {
	if (!Neighbours.size()) return false;

	if (this->meanSavedSpeed < SMALL_SPEED) {
		for (size_t nCount = 0; nCount < Neighbours.size(); nCount++) {
			if (Neighbours[nCount].second->meanSavedSpeed < SMALL_SPEED) return true;
		}
	}

	return false;
}