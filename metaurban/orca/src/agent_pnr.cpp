#include "agent_pnr.h"


agent_pnr::agent_pnr() : Agent() {
	fakeRadius = 0;
	PARAgents = std::set<agent_pnr *>();
	inPARMode = false;
	moveToPARPos = false;
	PARExec = false;
	PARMap = SubMap();

	conf.parallelizePaths1 = true;
	conf.parallelizePaths2 = false;
	conf.maxTime = CN_DEFAULT_MAPF_MAXTIME;
	conf.withFocalSearch = true;
	conf.withCAT = false;
	conf.withPerfectHeuristic = false;
	conf.withCardinalConflicts = false;
	conf.withBypassing = false;
	conf.withMatchingHeuristic = false;
	conf.storeConflicts = true;
	conf.withDisjointSplitting = false;
	conf.focalW = 1;
	conf.planner = "push_and_rotate";
	conf.lowLevel = "astar";

	PARVis = false;
	PARExec = false;
	PARStart = Point(-1, -1);
	PARGoal = Point(-1, -1);
	buffPar = std::vector<Point>();
	PARUnion = false;
	notPARVis = false;
	currPARPos = -1;
	PARActorId = -1;
	waitForStart = false;
	waitForFinish = false;

	initCount = 0;
	updCount = 0;
	uniCount = 0;
	timeMAPF = 0.0;
	successCount = 0;
	unsuccessCount = 0;
	flowtimeCount = 0;


}


agent_pnr::agent_pnr(const int &id, const Point &start, const Point &goal, const Map &map,
					 const environment_options &options, AgentParam param) : Agent(id, start, goal, map,
																								 options, param) {
	fakeRadius = param.rEps + param.radius;
	PARAgents = std::set<agent_pnr *>();
	inPARMode = false;
	moveToPARPos = false;
	PARVis = false;
	PARExec = false;
	PARMap = SubMap();

	conf.parallelizePaths1 = true;
	conf.parallelizePaths2 = false;
	conf.maxTime = CN_DEFAULT_MAPF_MAXTIME;
	conf.withFocalSearch = true;
	conf.withCAT = false;
	conf.withPerfectHeuristic = false;
	conf.withCardinalConflicts = false;
	conf.withBypassing = false;
	conf.withMatchingHeuristic = false;
	conf.storeConflicts = true;
	conf.withDisjointSplitting = false;
	conf.focalW = 1;
	conf.planner = "push_and_rotate";
	conf.lowLevel = "astar";

	PARStart = Point(-1, -1);
	PARGoal = Point(-1, -1);
	PARsearch = Astar<>(false);
	PARSolver = PushAndRotate(&PARsearch);
	buffPar = std::vector<Point>();
	PARUnion = false;
	notPARVis = false;
	currPARPos = -1;
	PARActorId = -1;
	waitForStart = false;
	waitForFinish = false;

	initCount = 0;
	updCount = 0;
	uniCount = 0;
	timeMAPF = 0.0;
	successCount = 0;
	unsuccessCount = 0;
	flowtimeCount = 0;
}


agent_pnr::agent_pnr(const agent_pnr &obj) : Agent(obj) {
	fakeRadius = obj.fakeRadius;
	PARAgents = obj.PARAgents;
	inPARMode = obj.inPARMode;
	moveToPARPos = obj.moveToPARPos;
	PARMap = obj.PARMap;
	PARSet = obj.PARSet;
	conf = obj.conf;
	PARVis = obj.PARVis;
	PARExec = obj.PARExec;
	PARStart = obj.PARStart;
	PARGoal = obj.PARGoal;
	PARsearch = obj.PARsearch;
	PARSolver = obj.PARSolver;
	buffPar = obj.buffPar;
	PARUnion = obj.PARUnion;
	notPARVis = obj.notPARVis;
	currPARPos = obj.currPARPos;
	PARActorId = obj.PARActorId;
	waitForStart = obj.waitForStart;
	waitForFinish = obj.waitForFinish;

	initCount = obj.initCount;
	updCount = obj.updCount;
	uniCount = obj.uniCount;
	timeMAPF = obj.timeMAPF;

	successCount = obj.successCount;
	unsuccessCount = obj.unsuccessCount;
	flowtimeCount = obj.flowtimeCount;

}


agent_pnr::~agent_pnr() = default;


agent_pnr &agent_pnr::operator=(const agent_pnr &obj) {

	if (this != &obj) {
		Agent::operator=(obj);
		fakeRadius = obj.fakeRadius;
		PARAgents = obj.PARAgents;
		inPARMode = obj.inPARMode;
		moveToPARPos = obj.moveToPARPos;
		PARMap = obj.PARMap;
		PARSet = obj.PARSet;
		conf = obj.conf;
		PARVis = obj.PARVis;
		PARExec = obj.PARExec;
		PARStart = obj.PARStart;
		PARGoal = obj.PARGoal;
		PARsearch = obj.PARsearch;
		PARSolver = obj.PARSolver;
		buffPar = obj.buffPar;
		PARUnion = obj.PARUnion;
		notPARVis = obj.notPARVis;
		currPARPos = obj.currPARPos;
		PARActorId = obj.PARActorId;
		waitForStart = obj.waitForStart;
		waitForFinish = obj.waitForFinish;
		initCount = obj.initCount;
		updCount = obj.updCount;
		uniCount = obj.uniCount;
		timeMAPF = obj.timeMAPF;
		successCount = obj.successCount;
		unsuccessCount = obj.unsuccessCount;
		flowtimeCount = obj.flowtimeCount;
	}
	return *this;
}


void agent_pnr::ComputeNewVelocity() {
	if (PARExec) {
		// Collision count
		unsigned long minMaxNum = (param.agentsMaxNum < Neighbours.size()) ? param.agentsMaxNum : Neighbours.size();

		for (unsigned long i = 0; i < minMaxNum; i++) {
			auto curragent = dynamic_cast<agent_pnr *>(Neighbours[i].second);
			if ((curragent->position - position).SquaredEuclideanNorm() <
				(param.radius + curragent->param.radius) * (param.radius + curragent->param.radius)) {
				collisions++;
			}
		}

		for (int i = 0; i < NeighboursObst.size(); i++) {
			Vertex *left = &(NeighboursObst[i].second.left);
			Vertex *right = &(NeighboursObst[i].second.right);

			Vector lRelativePosition = *left - position;
			Vector rRelativePosition = *right - position;

			float lSqDist = lRelativePosition.SquaredEuclideanNorm();
			float rSqDist = rRelativePosition.SquaredEuclideanNorm();

			float sqTrueRadius = param.radius * param.radius;

			Vector obstacleVector = *right - *left;
			float s = -lRelativePosition.ScalarProduct(obstacleVector) / obstacleVector.SquaredEuclideanNorm();
			float lineSqDist = (-lRelativePosition - obstacleVector * s).SquaredEuclideanNorm();

			if ((s < 0.0f && lSqDist < sqTrueRadius) || (s > 1.0f && rSqDist < sqTrueRadius) ||
				(s >= 0.0f && s < 1.0f && lineSqDist < sqTrueRadius)) {
				collisionsObst++;
			}
		}


		if (currPARPos < (*PARres.agentsPaths)[PARActorId].size()) {
			auto currGoal = PARMap.GetPoint((*PARres.agentsPaths)[PARActorId][currPARPos]);
			float dist = (currGoal - position).EuclideanNorm();
			newV = (currGoal - position);
			if (dist > 1) {
				newV = newV / dist;
			}
			else if (dist < 0.01) {
				newV = Point();
			}
		}
		else {
			newV = Point();
		}
		Neighbours.clear();
		return;
	}


	ORCALines.clear();


	// Получение ORCA-линий препятсвий
	for (int i = 0; i < NeighboursObst.size(); i++) {
		Line line;

		Vertex *left = &(NeighboursObst[i].second.left);
		Vertex *right = &(NeighboursObst[i].second.right);

		Vector lRelativePosition = *left - position;
		Vector rRelativePosition = *right - position;


		bool alreadyCovered = false;

		for (int j = 0; j < ORCALines.size(); j++) {
			if ((lRelativePosition * invTimeBoundaryObst - ORCALines[j].liesOn).Det(ORCALines[j].dir) -
				invTimeBoundaryObst * fakeRadius >= -CN_EPS &&
				(rRelativePosition * invTimeBoundaryObst - ORCALines[j].liesOn).Det(ORCALines[j].dir) -
				invTimeBoundaryObst * fakeRadius >= -CN_EPS) {
				alreadyCovered = true;
				break;
			}
		}
		if (alreadyCovered)
			continue;

		float lSqDist = lRelativePosition.SquaredEuclideanNorm();
		float rSqDist = rRelativePosition.SquaredEuclideanNorm();

		float sqFakeRadius = fakeRadius * fakeRadius;
		float sqTrueRadius = param.radius * param.radius;

		Vector obstacleVector = *right - *left;
		float s = -lRelativePosition.ScalarProduct(obstacleVector) / obstacleVector.SquaredEuclideanNorm();
		float lineSqDist = (-lRelativePosition - obstacleVector * s).SquaredEuclideanNorm();

		if ((s < 0.0f && lSqDist < sqTrueRadius) || (s > 1.0f && rSqDist < sqTrueRadius) ||
			(s >= 0.0f && s < 1.0f && lineSqDist < sqTrueRadius)) {
			collisionsObst++;
		}


		if (s < 0.0f && lSqDist < sqFakeRadius) {

			if (left->IsConvex()) {
				line.liesOn = Point();
				line.dir = Point(-lRelativePosition.Y(), lRelativePosition.X()) /
						   sqrt(lSqDist); // Построение единичного вектора, нормального к относительному положению
				ORCALines.push_back(line);
			}

			continue;
		}
		else if (s > 1.0f && rSqDist < sqFakeRadius) {

			if (right->IsConvex() && rRelativePosition.Det(NeighboursObst[i].second.next->dir) >= 0.0f) {
				line.liesOn = Point();
				line.dir = Point(-rRelativePosition.Y(), rRelativePosition.X()) / sqrt(rSqDist);
				ORCALines.push_back(line);
			}

			continue;
		}
		else if (s >= 0.0f && s < 1.0f && lineSqDist < sqFakeRadius) {
			line.liesOn = Point();
			line.dir = -(NeighboursObst[i].second.dir);
			ORCALines.push_back(line);
			continue;
		}


		Vector lLegDirection, rLegDirection;

		if (s < 0.0f && lineSqDist <= sqFakeRadius) {

			if (!left->IsConvex()) {
				continue;
			}

			right = left;

			float leg1 = sqrt(lSqDist - sqFakeRadius);

			lLegDirection = Point(lRelativePosition.X() * leg1 - lRelativePosition.Y() * fakeRadius,
								  lRelativePosition.X() * fakeRadius + lRelativePosition.Y() * leg1) / lSqDist;
			rLegDirection = Point(lRelativePosition.X() * leg1 + lRelativePosition.Y() * fakeRadius,
								  -lRelativePosition.X() * fakeRadius + lRelativePosition.Y() * leg1) / lSqDist;
		}
		else if (s > 1.0f && lineSqDist <= sqFakeRadius) {

			if (!right->IsConvex()) {
				continue;
			}

			left = right;

			float leg2 = std::sqrt(rSqDist - sqFakeRadius);
			lLegDirection = Point(rRelativePosition.X() * leg2 - rRelativePosition.Y() * fakeRadius,
								  rRelativePosition.X() * fakeRadius + rRelativePosition.Y() * leg2) / rSqDist;
			rLegDirection = Point(rRelativePosition.X() * leg2 + rRelativePosition.Y() * fakeRadius,
								  -rRelativePosition.X() * fakeRadius + rRelativePosition.Y() * leg2) / rSqDist;
		}
		else {
			if (left->IsConvex()) {
				float leg1 = std::sqrt(lSqDist - sqFakeRadius);
				lLegDirection = Point(lRelativePosition.X() * leg1 - lRelativePosition.Y() * fakeRadius,
									  lRelativePosition.X() * fakeRadius + lRelativePosition.Y() * leg1) / lSqDist;
			}
			else {
				lLegDirection = -NeighboursObst[i].second.dir;
			}

			if (right->IsConvex()) {
				float leg2 = std::sqrt(rSqDist - sqFakeRadius);
				rLegDirection = Point(rRelativePosition.X() * leg2 + rRelativePosition.Y() * fakeRadius,
									  -rRelativePosition.X() * fakeRadius + rRelativePosition.Y() * leg2) / rSqDist;
			}
			else {
				rLegDirection = NeighboursObst[i].second.dir;
			}
		}

		ObstacleSegment *leftNeighbor = NeighboursObst[i].second.prev;

		bool isLLegForeign = false, isRLegForeign = false;

		if (left->IsConvex() && lLegDirection.Det(-leftNeighbor->dir) >= 0.0f) {
			lLegDirection = -leftNeighbor->dir;
			isLLegForeign = true;
		}

		if (right->IsConvex() && rLegDirection.Det(NeighboursObst[i].second.next->dir) <= 0.0f) {
			rLegDirection = NeighboursObst[i].second.next->dir;
			isRLegForeign = true;
		}

		Point leftCutoff = (*left - position) * invTimeBoundaryObst;
		Point rightCutoff = (*right - position) * invTimeBoundaryObst;
		Vector cutoffVec = rightCutoff - leftCutoff;

		const float t = (right == left ? 0.5f : ((currV - leftCutoff).ScalarProduct(cutoffVec)) /
												cutoffVec.SquaredEuclideanNorm());
		const float tLeft = ((currV - leftCutoff).ScalarProduct(lLegDirection));
		const float tRight = ((currV - rightCutoff).ScalarProduct(rLegDirection));

		if ((t < 0.0f && tLeft < 0.0f) || (left == right && tLeft < 0.0f && tRight < 0.0f)) {
			Vector unitW = (currV - leftCutoff) / (currV - leftCutoff).EuclideanNorm();

			line.dir = Vector(unitW.Y(), -unitW.X());
			line.liesOn = leftCutoff + unitW * fakeRadius * invTimeBoundaryObst;
			ORCALines.push_back(line);
			continue;
		}
		else if (t > 1.0f && tRight < 0.0f) {
			Vector unitW = (currV - rightCutoff) / (currV - rightCutoff).EuclideanNorm();

			line.dir = Vector(unitW.Y(), -unitW.X());
			line.liesOn = rightCutoff + unitW * fakeRadius * invTimeBoundaryObst;
			ORCALines.push_back(line);
			continue;
		}

		float cutoffSqDist = ((t < 0.0f || t > 1.0f || right == left) ? std::numeric_limits<float>::infinity() : (
				currV - (leftCutoff + cutoffVec * t)).SquaredEuclideanNorm());
		float lLegSqDist = ((tLeft < 0.0f) ? std::numeric_limits<float>::infinity() : (currV - (leftCutoff +
																								lLegDirection *
																								tLeft)).SquaredEuclideanNorm());
		float rLegSqDist = ((tRight < 0.0f) ? std::numeric_limits<float>::infinity() : (currV - (rightCutoff +
																								 rLegDirection *
																								 tRight)).SquaredEuclideanNorm());

		if (cutoffSqDist <= lLegSqDist && cutoffSqDist <= rLegSqDist) {
			line.dir = -NeighboursObst[i].second.dir;
			line.liesOn = leftCutoff + Point(-line.dir.Y(), line.dir.X()) * fakeRadius * invTimeBoundaryObst;
			ORCALines.push_back(line);
			continue;
		}
		else if (lLegSqDist <= rLegSqDist) {
			if (isLLegForeign) {
				continue;
			}

			line.dir = lLegDirection;
			line.liesOn = leftCutoff + Point(-line.dir.Y(), line.dir.X()) * fakeRadius * invTimeBoundaryObst;;
			ORCALines.push_back(line);
			continue;
		}
		else {
			if (isRLegForeign) {
				continue;
			}

			line.dir = -rLegDirection;
			line.liesOn = rightCutoff + Point(-line.dir.Y(), line.dir.X()) * fakeRadius * invTimeBoundaryObst;
			ORCALines.push_back(line);
			continue;
		}
	}


	size_t numObstLines = ORCALines.size();

	//Получение ORCA-линий агентов
	//std::sort(Neighbours.begin(),Neighbours.end(), Compare);

	Line currline;
	agent_pnr *curragent;
	Vector u, w;

	unsigned long minMaxNum = (param.agentsMaxNum < Neighbours.size()) ? param.agentsMaxNum : Neighbours.size();

	for (unsigned long i = 0; i < minMaxNum; i++) {
		auto Neighbour = Neighbours[i];
		curragent = dynamic_cast<agent_pnr *>(Neighbour.second);
		auto circlecenter = curragent->position - this->position; //(P_b - P_a)
		auto relvelocity = this->currV - curragent->currV; //(V_a - V_b)

		float radiussum = fakeRadius + curragent->fakeRadius; //(R_a + R_b)
		float radiussum2 = radiussum * radiussum;
		float distSq = circlecenter.SquaredEuclideanNorm();
		float trueSqRadSum = (param.radius + curragent->param.radius) * (param.radius + curragent->param.radius);

		if (distSq < trueSqRadSum) {
			collisions++;
		}

		if (distSq >= radiussum2) {
			w = relvelocity - (circlecenter *
							   invTimeBoundary); //w -- вектор на плоскости скоростей от центра малой окружности (основания VO) до скорости другого агента относительно этого
			float sqwlength = w.SquaredEuclideanNorm();
			float wproj = w.ScalarProduct(circlecenter);

			// если эти условия выполняются, то вектор w отложенный из центра окружности-основания VO будет своим концом ближе к
			// этой самой окружности, а значит и ближайшая точка на границе VO -- это какая-то точка на этой окружнрости

			if (wproj < 0.0f && (wproj * wproj) > sqwlength * radiussum2) {
				const float wlength = std::sqrt(sqwlength);
				const Vector nw = w / wlength;
				currline.dir = Vector(nw.Y(), -nw.X());
				u = nw * (radiussum * invTimeBoundary - wlength);
			}
			else {
				//иначе проекция на стороны VO
				//длина проекции вектора относительных положений на сторону VO

				float leg = std::sqrt(distSq - radiussum2);

				if (circlecenter.Det(w) > 0.0f) //если точка ближе к левой стороне VO
				{
					currline.dir = Vector(circlecenter.X() * leg - circlecenter.Y() * radiussum,
										  circlecenter.X() * radiussum + circlecenter.Y() * leg) / distSq;
				}
				else //если точка ближе к правой стороне VO
				{
					currline.dir = -(Vector(circlecenter.X() * leg + circlecenter.Y() * radiussum,
											-circlecenter.X() * radiussum + circlecenter.Y() * leg) / distSq);
				}

				float rvproj = relvelocity.ScalarProduct(currline.dir);

				u = currline.dir * rvproj - relvelocity;
			}
		}
		else {
			const float invTimeStep = 1.0f / options->timestep;

			Vector w = relvelocity - circlecenter * invTimeStep;
			float wlength = w.EuclideanNorm();
			Vector wn = w / wlength;
			currline.dir = Vector(wn.Y(), -wn.X());
			u = wn * (radiussum * invTimeStep - wlength);
		}

		currline.liesOn = this->currV + u * 0.5f;
		ORCALines.push_back(currline);
		Neighbours.pop_back();
	}

	auto lineFail = Utils::linearProgram2(ORCALines, param.maxSpeed, this->prefV, false, this->newV);
	if (lineFail < this->ORCALines.size()) {
		Utils::linearProgram3(ORCALines, numObstLines, lineFail, param.maxSpeed, this->newV);
	}

	Neighbours.clear();
}


void agent_pnr::ApplyNewVelocity() {
	PARVis = false;
	notPARVis = false;
	PARUnion = false;
	currV = newV;
	waitForStart = false;
	waitForFinish = false;

	speedSaveBuffer.pop_front();
	if (inPARMode) {
		if (PARExec) {
			flowtimeCount++;
		}
		speedSaveBuffer.push_back(param.maxSpeed);
	}
	else {
		speedSaveBuffer.push_back(currV.EuclideanNorm());
	}


	// Kahan summation algorithm
	float sum = 0.0f;
	float c = 0.0f;
	float y, t;
	for (auto speed: speedSaveBuffer) {
		y = speed - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}
	meanSavedSpeed = sum / speedSaveBuffer.size();
}


bool agent_pnr::UpdatePrefVelocity() {

	Point next;
	if (inPARMode) {
		if (PARUnion) {
			UnitePAR();
			prefV = Point();
			nextForLog = position;
			return true;
		}

		if (notPARVis) {
			UpdatePAR();
			prefV = Point();
			nextForLog = position;
			return true;
		}

		if (PARExec) {
			nextForLog = PARGoal;
			bool allOnPos = true;
			bool allFin = true;

			for (auto &ag: PARAgents) {
				int agPARId = ag->PARActorId;
				if (agPARId == -1 && !ag->inPARMode) {
					continue;
				}

				int agPARPos = (currPARPos < (*(PARres.agentsPaths))[agPARId].size()) ? currPARPos : int(
						(*(PARres.agentsPaths))[agPARId].size() - 1);
				Point agCurrGoal = PARMap.GetPoint((*(PARres.agentsPaths))[agPARId][agPARPos]);

				if (!((ag->position - agCurrGoal).EuclideanNorm() < options->delta) || !ag->PARExec) {
					allOnPos = false;
				}

				if (!((ag->position - ag->PARGoal).EuclideanNorm() < options->delta) && ag->inPARMode) {
					allFin = false;
				}
			}

			if (allFin) {
//                //  std::cout<<"Fin\n";
				waitForFinish = true;
				inPARMode = false;
				PARExec = false;
				moveToPARPos = false;
				currPARPos = -1;
				PARActorId = -1;
				PARStart = Point(-1, -1);
				PARGoal = Point(-1, -1);
				prefV = Point();
				nextForLog = position;
				PARres.Clear();
				PARAgents.clear();
				PARSet.clear();
				buffPar.clear();
				return true;
			}
			else if (allOnPos) {
				currPARPos++;
				return true;
			}
			else {
				prefV = Point();
				nextForLog = PARGoal;
				return true;
			}


		}

		if (moveToPARPos) {
			if (waitForStart) {
				waitForStart = false;
				PARExec = false;
				nextForLog = PARStart;
				prefV = Point();
				return true;
			}
			nextForLog = PARStart;
			float dist = (position - PARStart).EuclideanNorm();
			if (dist < options->delta) {

				prefV = Point();
				PARExec = true;

				for (auto &a: PARAgents) {
					float distOtherAgent = (a->position - a->PARStart).EuclideanNorm();
					if (!(distOtherAgent < options->delta)) {
						PARExec = false;
						break;
					}
				}

				if (PARExec) {
					nextForLog = PARGoal;
					moveToPARPos = false;
					currPARPos = 1;

					return true;
				}
				else {
					PARExec = false;
					return true;
				}
			}
			else {
				nextForLog = PARStart;
				Vector goalVector = PARStart - position;
				float dist = goalVector.EuclideanNorm();

				if (dist > 1) {
					goalVector = (goalVector / dist);
				}
				prefV = goalVector;
				return true;
			}
		}
	}
	else {
		if (PARVis) {
			prefV = Point();
			return true;
		}
		if (planner->GetNext(position, next)) {
			nextForLog = next;
			Vector goalVector = next - position;
			float dist = goalVector.EuclideanNorm();
			if ((options->trigger == MAPFTriggers::COMMON_POINT && CommonPointMAPFTrigger(dist)) ||
				(options->trigger == MAPFTriggers::SPEED_BUFFER && SingleNeighbourMeanSpeedMAPFTrigger())) {
				PreparePARExecution();
				prefV = Point();

				return true;
			}
			else {
				if (next == goal && dist < options->delta) {
					prefV = Point();
					return true;
				}

				if (dist > CN_EPS) {
					goalVector = (goalVector / dist) * param.maxSpeed;
				}

				prefV = goalVector;
				return true;
			}
		}
		nextForLog = position;
		prefV = Point();
		return false;
	}
}


bool agent_pnr::operator==(const agent_pnr &another) const {
	return this->id == another.id;
}


bool agent_pnr::operator!=(const agent_pnr &another) const {
	return this->id != another.id;
}


agent_pnr *agent_pnr::Clone() const {
	return new agent_pnr(*this);
}


void agent_pnr::AddNeighbour(Agent &neighbour, float distSq) {
	float sightSq = param.sightRadius * param.sightRadius;

	if (!(distSq < sightSq)) {
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

		auto tmpAgentPAR = dynamic_cast<agent_pnr *>(&neighbour);
		if (tmpAgentPAR->inPARMode) {
			PARVis = true;

			if (inPARMode) {
				if (!tmpAgentPAR->PARUnion && PARAgents.find(tmpAgentPAR) == PARAgents.end()) {
					if (distSq < (param.radius + tmpAgentPAR->param.radius + 0.25) *
								 (param.radius + tmpAgentPAR->param.radius + 0.25)) {
						PARUnion = true;
					}
				}
			}
		}
		else if (inPARMode) {
			notPARVis = true;
		}
	}
}


std::set<agent_pnr *> agent_pnr::GetAgentsForCentralizedPlanning() {
	return PARAgents;
}


void agent_pnr::SetAgentsForCentralizedPlanning(std::set<agent_pnr *> agents) {
	PARAgents.clear();
	for (auto &a: agents) {
		PARAgents.insert(a);
	}
}


void agent_pnr::PreparePARExecution() {
	PARAgents.clear();
	inPARMode = true;
	moveToPARPos = true;
	PARExec = false;
	initCount++;

	PARAgents.insert(this);
	for (auto &el: Neighbours) {
		agent_pnr *neighbour = dynamic_cast<agent_pnr *>(el.second);
		if (!neighbour->inPARMode && !neighbour->waitForFinish) {
			PARAgents.insert(neighbour);
			neighbour->inPARMode = true;
			neighbour->moveToPARPos = true;
			neighbour->PARExec = false;
			neighbour->prefV = Point();

			for (auto &el2: neighbour->GetNeighbours()) {
				auto neighbourOfNeighbour = dynamic_cast<agent_pnr *>(el2.second);
				if (!neighbourOfNeighbour->inPARMode && !neighbourOfNeighbour->waitForFinish) {
					neighbourOfNeighbour->inPARMode = true;
					neighbourOfNeighbour->moveToPARPos = true;
					neighbourOfNeighbour->PARExec = false;
					neighbourOfNeighbour->prefV = Point();
					PARAgents.insert(neighbourOfNeighbour);
				}
			}
		}
	}

	for (auto &ag: PARAgents) {
		if (ag != this) {
			ag->SetAgentsForCentralizedPlanning(PARAgents);
		}
		if (!ag->ComputePAREnv()) {
			for (auto &ag1: PARAgents) {
				while (!ag1->buffPar.empty()) {
					ag1->planner->AddPointToPath(ag1->buffPar.back());
					ag1->buffPar.pop_back();
				}

				ag1->PARStart = Point(-1, -1);
				ag1->PARGoal = Point(-1, -1);
				ag1->inPARMode = false;
				ag1->moveToPARPos = false;
				ag1->PARExec = false;
				ag1->currPARPos = -1;
				ag1->PARActorId = -1;
			}
			return;
		}
	}

	auto startpnt = std::chrono::high_resolution_clock::now();
	for (auto &ag: PARAgents) {
		if (!ag->ComputePAR()) {
			auto endpnt = std::chrono::high_resolution_clock::now();
			size_t res = std::chrono::duration_cast<std::chrono::milliseconds>(endpnt - startpnt).count();
			timeMAPF += static_cast<float>(res);
			unsuccessCount++;
			for (auto &ag1: PARAgents) {
				while (!ag1->buffPar.empty()) {
					ag1->planner->AddPointToPath(ag1->buffPar.back());
					ag1->buffPar.pop_back();
				}

				ag1->PARStart = Point(-1, -1);
				ag1->PARGoal = Point(-1, -1);
				ag1->inPARMode = false;
				ag1->moveToPARPos = false;
				ag1->PARExec = false;
				ag1->currPARPos = -1;
				ag1->PARActorId = -1;
			}
			return;
		}
	}
	auto endpnt = std::chrono::high_resolution_clock::now();
	size_t res = std::chrono::duration_cast<std::chrono::milliseconds>(endpnt - startpnt).count();
	timeMAPF += static_cast<float>(res);
	successCount++;
#if MAPF_LOG
	// Save here
	PARLog->SaveInstance(PARSet, PARMap, conf);
#endif

	for (auto &ag: PARAgents) {
		ag->inPARMode = true;
		ag->moveToPARPos = true;
		ag->PARVis = false;
		ag->PARExec = false;
		ag->currPARPos = 0;
		ag->waitForStart = true;
	}
}


bool agent_pnr::UnitePAR() {
	PARUnion = false;
	uniCount++;
	for (auto &ag: PARAgents) {
		ag->PARStart = Point(-1, -1);
		ag->PARGoal = Point(-1, -1);
		ag->PARres.Clear();
		ag->currPARPos = -1;
		ag->PARActorId = -1;
		ag->PARUnion = false;
	}


	for (auto &el: Neighbours) {
		agent_pnr *neighbour = dynamic_cast<agent_pnr *>(el.second);
		if (neighbour->inPARMode && !neighbour->waitForFinish && PARAgents.find(neighbour) == PARAgents.end()) {
			for (auto &nn: neighbour->PARAgents) {
				while (!nn->buffPar.empty()) {
					nn->planner->AddPointToPath(nn->buffPar.back());
					nn->buffPar.pop_back();
				}

				nn->PARStart = Point(-1, -1);
				nn->PARGoal = Point(-1, -1);
				nn->currPARPos = -1;
				nn->PARActorId = -1;
				nn->PARUnion = false;
				nn->PARres.Clear();
			}

			PARAgents.insert(neighbour->PARAgents.begin(), neighbour->PARAgents.end());
		}
	}


	for (auto &ag: PARAgents) {
		if (ag != this) {
			ag->SetAgentsForCentralizedPlanning(PARAgents);
		}
		if (!ag->ComputePAREnv()) {
			for (auto &ag1: PARAgents) {
				while (!ag1->buffPar.empty()) {
					ag1->planner->AddPointToPath(ag1->buffPar.back());
					ag1->buffPar.pop_back();
				}

				ag1->PARStart = Point(-1, -1);
				ag1->PARGoal = Point(-1, -1);
				ag1->inPARMode = false;
				ag1->moveToPARPos = false;
				ag1->PARExec = false;
				ag1->currPARPos = -1;
				ag1->PARActorId = -1;
			}
			return false;
		}
	}

	auto startpnt = std::chrono::high_resolution_clock::now();
	for (auto &ag: PARAgents) {
		if (!ag->ComputePAR()) {
			auto endpnt = std::chrono::high_resolution_clock::now();
			size_t res = std::chrono::duration_cast<std::chrono::milliseconds>(endpnt - startpnt).count();
			timeMAPF += static_cast<float>(res);
			unsuccessCount++;
			for (auto &ag1: PARAgents) {
				while (!ag1->buffPar.empty()) {
					ag1->planner->AddPointToPath(ag1->buffPar.back());
					ag1->buffPar.pop_back();
				}

				ag1->PARStart = Point(-1, -1);
				ag1->PARGoal = Point(-1, -1);
				ag1->inPARMode = false;
				ag1->moveToPARPos = false;
				ag1->PARExec = false;
				ag1->currPARPos = -1;
				ag1->PARActorId = -1;
			}
			return false;
		}
	}
	auto endpnt = std::chrono::high_resolution_clock::now();
	size_t res = std::chrono::duration_cast<std::chrono::milliseconds>(endpnt - startpnt).count();
	timeMAPF += static_cast<float>(res);
	successCount++;

	for (auto &ag: PARAgents) {
		ag->inPARMode = true;
		ag->moveToPARPos = true;
		ag->PARVis = false;
		ag->PARExec = false;
		ag->currPARPos = 0;
		ag->waitForStart = true;
	}


	return true;

}


bool agent_pnr::UpdatePAR() {
	updCount++;
	for (auto &ag: PARAgents) {
		ag->PARStart = Point(-1, -1);
		ag->PARGoal = Point(-1, -1);
		ag->PARres.Clear();
		ag->currPARPos = -1;
		ag->PARActorId = -1;
	}

	std::set<agent_pnr *> tmpAgents;
	for (auto &ag: PARAgents) {

		auto N = ag->GetNeighbours();
		for (auto &el: N) {
			agent_pnr *neighbour = dynamic_cast<agent_pnr *>(el.second);
			if (!neighbour->inPARMode && !neighbour->waitForFinish) {
				neighbour->PARStart = Point(-1, -1);
				neighbour->PARGoal = Point(-1, -1);
				neighbour->PARres.Clear();
				neighbour->currPARPos = -1;
				neighbour->PARActorId = -1;
				tmpAgents.insert(neighbour);
			}
		}
		ag->notPARVis = false;
	}

	PARAgents.insert(tmpAgents.begin(), tmpAgents.end());

	for (auto &ag: PARAgents) {
		if (ag != this) {
			ag->SetAgentsForCentralizedPlanning(PARAgents);
		}

		if (!ag->ComputePAREnv()) {
			for (auto &ag1: PARAgents) {
				while (!ag1->buffPar.empty()) {
					ag1->planner->AddPointToPath(ag1->buffPar.back());
					ag1->buffPar.pop_back();
				}

				ag1->PARStart = Point(-1, -1);
				ag1->PARGoal = Point(-1, -1);
				ag1->inPARMode = false;
				ag1->moveToPARPos = false;
				ag1->PARExec = false;
				ag1->currPARPos = -1;
				ag1->PARActorId = -1;
			}
			return false;
		}
	}

	auto startpnt = std::chrono::high_resolution_clock::now();
	for (auto &ag: PARAgents) {
		if (!ag->ComputePAR()) {
			auto endpnt = std::chrono::high_resolution_clock::now();
			size_t res = std::chrono::duration_cast<std::chrono::milliseconds>(endpnt - startpnt).count();
			timeMAPF += static_cast<float>(res);
			unsuccessCount++;
			for (auto &ag1: PARAgents) {
				while (!ag1->buffPar.empty()) {
					ag1->planner->AddPointToPath(ag1->buffPar.back());
					ag1->buffPar.pop_back();
				}

				ag1->PARStart = Point(-1, -1);
				ag1->PARGoal = Point(-1, -1);
				ag1->inPARMode = false;
				ag1->moveToPARPos = false;
				ag1->PARExec = false;
				ag1->currPARPos = -1;
				ag1->PARActorId = -1;
			}
			return false;
		}
	}
	auto endpnt = std::chrono::high_resolution_clock::now();
	size_t res = std::chrono::duration_cast<std::chrono::milliseconds>(endpnt - startpnt).count();
	timeMAPF += static_cast<float>(res);
	successCount++;


	for (auto &ag: PARAgents) {
		ag->inPARMode = true;
		ag->moveToPARPos = true;
		ag->PARVis = false;
		ag->PARExec = false;
		ag->currPARPos = 0;
		ag->waitForStart = true;
	}

	return true;
}


bool agent_pnr::ComputePAREnv() {

	float minX = static_cast<float>(map->GetWidth() * map->GetCellSize()), minY = static_cast<float>(map->GetHeight() *
																									 map->GetCellSize()), maxX = 0.0f, maxY = 0.0f;

	for (auto &ag: PARAgents) {
		if ((ag->GetPosition().X() - ag->param.sightRadius) < minX) {
			minX = (ag->GetPosition().X() - ag->param.sightRadius);
		}
		if ((ag->GetPosition().Y() - ag->param.sightRadius) < minY) {
			minY = (ag->GetPosition().Y() - ag->param.sightRadius);
		}

		if ((ag->GetPosition().X() + ag->param.sightRadius) > maxX) {
			maxX = (ag->GetPosition().X() + ag->param.sightRadius);
		}
		if ((ag->GetPosition().Y() + ag->param.sightRadius) > maxY) {
			maxY = (ag->GetPosition().Y() + ag->param.sightRadius);
		}
	}

	Node nodeLeftTop = map->GetClosestNode(Point(minX, maxY));
	Node nodeRightBottom = map->GetClosestNode(Point(maxX, minY));
	int width = nodeRightBottom.j - nodeLeftTop.j + 1;
	int height = nodeRightBottom.i - nodeLeftTop.i + 1;
	PARMap = SubMap(map, nodeLeftTop, {height, width}, 1);

	// Starts and goals
	std::unordered_map<int, Node> starts;
	std::unordered_map<int, Node> goals;
	std::unordered_map<int, Node> finalGoals;

	PARSet = MAPFActorSet();
	PARSet.clear();
	Point tmpGoalPoint;
	for (auto &ag: PARAgents) {
		Node tmpStart, tmpGoal;

		if (ag->PARStart.X() < 0) {
			tmpStart = PARMap.FindCloseToPointAvailableNode(ag->GetPosition(), starts);

			if (tmpStart.i < 0) {
				for (auto &ag1: PARAgents) {
					while (!ag1->buffPar.empty()) {
						ag1->planner->AddPointToPath(ag1->buffPar.back());
						ag1->buffPar.pop_back();
					}
					ag1->PARStart = Point(-1, -1);
					ag1->PARGoal = Point(-1, -1);
					ag1->inPARMode = false;
					ag1->moveToPARPos = false;
					ag1->PARVis = false;
					ag1->PARExec = false;
				}
				return false;
			}
			ag->PARStart = PARMap.GetPoint(tmpStart);
		}
		else {
			tmpStart = PARMap.GetClosestNode(ag->PARStart);
		}

		if (ag->PARGoal.X() < 0) {
			ag->buffPar.clear();
			tmpGoalPoint = ag->GetGoalPointForMAPF(PARMap);
			tmpGoal = PARMap.FindAccessibleNodeForGoal(tmpStart, tmpGoalPoint, goals, finalGoals);

			if (tmpGoal.i < 0) {
				for (auto &ag1: PARAgents) {
					while (!ag1->buffPar.empty()) {
						ag1->planner->AddPointToPath(ag1->buffPar.back());
						ag1->buffPar.pop_back();
					}
					ag1->PARStart = Point(-1, -1);
					ag1->PARGoal = Point(-1, -1);
					ag1->inPARMode = false;
					ag1->moveToPARPos = false;
					ag1->PARVis = false;
					ag1->PARExec = false;
				}

				return false;
			}
			ag->PARGoal = PARMap.GetPoint(tmpGoal);
		}
		else {
			tmpGoal = PARMap.GetClosestNode(ag->PARGoal);
		}


		starts.insert({tmpStart.i * PARMap.GetWidth() + tmpStart.j, tmpStart});
		if ((tmpGoalPoint - ag->goal).EuclideanNorm() < options->delta) {
//            if(finalGoals.find(tmpGoal.i * PARMap.GetWidth() + tmpGoal.j) != finalGoals.end())
//            {
//                //  std::cout << "AAAAAA 3\n";
//            }
			finalGoals.insert({tmpGoal.i * PARMap.GetWidth() + tmpGoal.j, tmpGoal});
		}
		else {
//            if(goals.find(tmpGoal.i * PARMap.GetWidth() + tmpGoal.j) != goals.end())
//            {
//                //  std::cout << "AAAAAA 3\n";
//            }
			goals.insert({tmpGoal.i * PARMap.GetWidth() + tmpGoal.j, tmpGoal});
		}

		PARSet.addActor(tmpStart.i, tmpStart.j, tmpGoal.i, tmpGoal.j);
		if (ag == this) {
			PARActorId = PARSet.getActorId(tmpStart.i, tmpStart.j);
			currPARPos = 0;
		}
	}
	return true;
}


bool agent_pnr::ComputePAR() {
	PARsearch = Astar<>(false);
	PARSolver = PushAndRotate(&PARsearch);
	PARres = PARSolver.startSearch(PARMap, conf, PARSet);
	return PARres.pathfound;
}


std::vector<std::pair<float, Agent *>> &agent_pnr::GetNeighbours() {
	return Neighbours;
}


Point agent_pnr::GetGoalPointForMAPF(SubMap Area) {
	Point nextPoint = planner->PullOutNext();
	buffPar.push_back(nextPoint);

	while (Area.PointBelongsToArea(nextPoint) && !(nextPoint == this->goal)) {
		nextPoint = planner->PullOutNext();
		buffPar.push_back(nextPoint);
	}
	return nextPoint;
}


unordered_map<std::string, float> agent_pnr::GetMAPFStatistics() const {
	unordered_map<std::string, float> stat;
	stat[CNS_MAPF_COMMON_TIME] = timeMAPF;
	stat[CNS_MAPF_INIT_COUNT] = static_cast<float>(initCount);
	stat[CNS_MAPF_UNITE_COUNT] = static_cast<float>(uniCount);
	stat[CNS_MAPF_UPDATE_COUNT] = static_cast<float>(updCount);
	stat[CNS_MAPF_FLOWTIME] = static_cast<float>(flowtimeCount);
	stat[CNS_MAPF_SUCCESS_COUNT] = static_cast<float>(successCount);
	stat[CNS_MAPF_UNSUCCESS_COUNT] = static_cast<float>(unsuccessCount);

	return stat;
}

#if MAPF_LOG
void ORCAAgentWithPAR::SetMAPFInstanceLoggerRef(MAPFInstancesLogger *log)
{
	PARLog = log;
}
#endif