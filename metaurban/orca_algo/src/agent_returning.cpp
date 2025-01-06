#include "agent_returning.h"


ORCAAgentWithReturning::ORCAAgentWithReturning() : Agent() {
	fakeRadius = 0;
	gen = std::mt19937(0);
}


ORCAAgentWithReturning::ORCAAgentWithReturning(const int &id, const Point &start, const Point &goal, const Map &map,
											   const environment_options &options, AgentParam param) : Agent(id, start,
																											 goal, map,
																											 options,
																											 param) {
	fakeRadius = param.rEps + param.radius;
	gen = std::mt19937(id);
}


ORCAAgentWithReturning::ORCAAgentWithReturning(const ORCAAgentWithReturning &obj) : Agent(obj) {
	fakeRadius = obj.fakeRadius;
	gen = obj.gen;
}


ORCAAgentWithReturning::~ORCAAgentWithReturning() = default;


ORCAAgentWithReturning &ORCAAgentWithReturning::operator=(const ORCAAgentWithReturning &obj) {

	if (this != &obj) {
		Agent::operator=(obj);
		fakeRadius = obj.fakeRadius;
		gen = obj.gen;
	}
	return *this;
}


ORCAAgentWithReturning *ORCAAgentWithReturning::Clone() const {
	return new ORCAAgentWithReturning(*this);


}


bool ORCAAgentWithReturning::operator==(const ORCAAgentWithReturning &another) const {
	return this->id == another.id;
}


bool ORCAAgentWithReturning::operator!=(const ORCAAgentWithReturning &another) const {
	return this->id != another.id;
}


void ORCAAgentWithReturning::ComputeNewVelocity() {
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
	ORCAAgentWithReturning *curragent;
	Vector u, w;

	unsigned long minMaxNum = (param.agentsMaxNum < Neighbours.size()) ? param.agentsMaxNum : Neighbours.size();

	for (unsigned long i = 0; i < minMaxNum; i++) {
		auto Neighbour = Neighbours[i];
		curragent = dynamic_cast<ORCAAgentWithReturning *>(Neighbour.second);
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


void ORCAAgentWithReturning::ApplyNewVelocity() {

	currV = newV;

	// TODO delay for returning
	speedSaveBuffer.pop_front();
	speedSaveBuffer.push_back(currV.EuclideanNorm());

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


bool ORCAAgentWithReturning::UpdatePrefVelocity() {
	Point next;

	if (planner->GetNext(position, next)) {
		nextForLog = next;
		Vector goalVector = next - position;
		float dist = goalVector.EuclideanNorm();

		if ((options->trigger == MAPFTriggers::COMMON_POINT && CommonPointMAPFTrigger(dist)) ||
			(options->trigger == MAPFTriggers::SPEED_BUFFER && SingleNeighbourMeanSpeedMAPFTrigger())) {
			SendRequestsForReturning();
		}

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

	nextForLog = position;
	prefV = Point();
	return false;

}


void ORCAAgentWithReturning::AddNeighbour(Agent &neighbour, float distSq) {
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
	}
}


std::vector<std::pair<float, Agent *>> &ORCAAgentWithReturning::GetNeighbours() {
	return Neighbours;
}


void ORCAAgentWithReturning::SendRequestsForReturning() {
	std::set<ORCAAgentWithReturning *> closeAgents;
	//closeAgents.insert(this);
	auto buffSize = speedSaveBuffer.size();
	speedSaveBuffer = std::list<float>(buffSize, param.maxSpeed);

	for (auto &el: Neighbours) {
		auto neighbour = dynamic_cast<ORCAAgentWithReturning *>(el.second);
		closeAgents.insert(neighbour);
		for (auto &el2: neighbour->GetNeighbours()) {
			auto neighbourOfNeighbour = dynamic_cast<ORCAAgentWithReturning *>(el2.second);
			closeAgents.insert(neighbourOfNeighbour);
		}
	}

//    std::uniform_int_distribution<int> uid(0, closeAgents.size() - 1);
//    auto a = uid(gen);
//    int i = 0;
	for (auto &el: closeAgents) {
		auto buffSize = speedSaveBuffer.size();
		el->speedSaveBuffer = std::list<float>(buffSize, param.maxSpeed);
//        auto a = uid(gen);
//        //  std::cout << a << "\n";
//        if(i != )
//        {
		el->ReturnTosPastPoint();
//        }
//        i++;
	}
}


void ORCAAgentWithReturning::ReturnTosPastPoint() {
	planner->AddPointToPath(planner->GetPastPoint());
}
