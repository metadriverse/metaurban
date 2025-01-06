#include "orca_diff_drive_agent.h"


ORCADDAgent::ORCADDAgent() : Agent() {
	leftV = 0;
	rightV = 0;
	effectivePosition = Point();
	effectiveVelocity = Point();
	D = 0.0f;
	sin0 = 0.0f;
	cos0 = 0.0f;
	tet = 0.0f;
	wheelTrack = 0.0f;
	effectiveRadius = 0.0f;
}


ORCADDAgent::ORCADDAgent(const int &id, const Point &start, const Point &goal, const Map &map,
						 const environment_options &options,
						 AgentParam param, float effR, float wheelTrack, float theta) : Agent(id, start, goal, map,
																							  options, param) {
	this->effectiveRadius = effR;
	this->wheelTrack = wheelTrack;
	this->D = effR - param.radius - param.rEps;
	leftV = 0;
	rightV = 0;
	effectivePosition = Point();
	effectiveVelocity = Point();

	tet = theta;
	sin0 = sin(tet);
	cos0 = cos(tet);
	effectivePosition = Point(position.X() + D * cos0, position.Y() + D * sin0);
}


ORCADDAgent::ORCADDAgent(const ORCADDAgent &obj) : Agent(obj) {
	leftV = obj.leftV;
	rightV = obj.rightV;
	effectivePosition = obj.effectivePosition;
	effectiveVelocity = obj.effectiveVelocity;
	D = obj.D;
	sin0 = obj.sin0;
	cos0 = obj.cos0;
	tet = obj.tet;
	effectiveRadius = obj.effectiveRadius;
	wheelTrack = obj.wheelTrack;

}

ORCADDAgent::~ORCADDAgent() = default;


ORCADDAgent &ORCADDAgent::operator=(const ORCADDAgent &obj) {
	if (this != &obj) {
		Agent::operator=(obj);
		leftV = obj.leftV;
		rightV = obj.rightV;
		effectivePosition = obj.effectivePosition;
		effectiveVelocity = obj.effectiveVelocity;
		D = obj.D;
		sin0 = obj.sin0;
		cos0 = obj.cos0;
		tet = obj.tet;
		effectiveRadius = obj.effectiveRadius;
		wheelTrack = obj.wheelTrack;
	}

	return *this;
}


bool ORCADDAgent::operator==(const ORCADDAgent &another) const {
	return this->id == another.id;
}


bool ORCADDAgent::operator!=(const ORCADDAgent &another) const {
	return this->id != another.id;
}


ORCADDAgent *ORCADDAgent::Clone() const {
	return new ORCADDAgent(*this);
}


void ORCADDAgent::ComputeNewVelocity() {
	ORCALines.clear();

	// Получение ORCA-линий препятсвий
	for (int i = 0; i < NeighboursObst.size(); i++) {
		Line line;

		Vertex *left = &(NeighboursObst[i].second.left);
		Vertex *right = &(NeighboursObst[i].second.right);

		Vector lRelativePosition = *left - effectivePosition;
		Vector rRelativePosition = *right - effectivePosition;

		Vector lTrueRelativePosition = *left - position;
		Vector rTrueRelativePosition = *right - position;


		bool alreadyCovered = false;

		for (int j = 0; j < ORCALines.size(); j++) {
			if ((lRelativePosition * invTimeBoundaryObst - ORCALines[j].liesOn).Det(ORCALines[j].dir) -
				invTimeBoundaryObst * effectiveRadius >= -CN_EPS &&
				(rRelativePosition * invTimeBoundaryObst - ORCALines[j].liesOn).Det(ORCALines[j].dir) -
				invTimeBoundaryObst * effectiveRadius >= -CN_EPS) {
				alreadyCovered = true;
				break;
			}
		}
		if (alreadyCovered)
			continue;

		float lSqDist = lRelativePosition.SquaredEuclideanNorm();
		float rSqDist = rRelativePosition.SquaredEuclideanNorm();

		float lTrueSqDist = lTrueRelativePosition.SquaredEuclideanNorm();
		float rTrueSqDist = rTrueRelativePosition.SquaredEuclideanNorm();

		float sqEfRadius = effectiveRadius * effectiveRadius;

		float sqRadius = param.radius * param.radius;

		Vector obstacleVector = *right - *left;
		float s = -lRelativePosition.ScalarProduct(obstacleVector) / obstacleVector.SquaredEuclideanNorm();
		float sTrue = -lTrueRelativePosition.ScalarProduct(obstacleVector) / obstacleVector.SquaredEuclideanNorm();

		float lineSqDist = (-lRelativePosition - obstacleVector * s).SquaredEuclideanNorm();
		float lineSqDistTrue = (-lTrueRelativePosition - obstacleVector * sTrue).SquaredEuclideanNorm();

		if ((sTrue < 0.0f && lTrueSqDist < sqRadius) || (sTrue > 1.0f && rTrueSqDist < sqRadius) ||
			(sTrue >= 0.0f && sTrue < 1.0f && lineSqDistTrue < sqRadius)) {
			collisionsObst++;
		}

		if (s < 0.0f && lSqDist < sqEfRadius) {
			if (left->IsConvex()) {
				line.liesOn = Point();
				line.dir = Point(-lRelativePosition.Y(), lRelativePosition.X()) /
						   sqrt(lSqDist); // Построение единичного вектора, нормального к относительному положению
				ORCALines.push_back(line);
			}

			continue;
		}
		else if (s > 1.0f && rSqDist < sqEfRadius) {
			if (right->IsConvex() && rRelativePosition.Det(NeighboursObst[i].second.next->dir) >= 0.0f) {
				line.liesOn = Point();
				line.dir = Point(-rRelativePosition.Y(), rRelativePosition.X()) / sqrt(rSqDist);
				ORCALines.push_back(line);

			}

			continue;
		}
		else if (s >= 0.0f && s < 1.0f && lineSqDist < sqEfRadius) {
			line.liesOn = Point();
			line.dir = -(NeighboursObst[i].second.dir);
			ORCALines.push_back(line);
			continue;
		}


		Vector lLegDirection, rLegDirection;

		if (s < 0.0f && lineSqDist <= sqEfRadius) {

			if (!left->IsConvex()) {
				continue;
			}

			right = left;

			float leg1 = sqrt(lSqDist - sqEfRadius);

			lLegDirection = Point(lRelativePosition.X() * leg1 - lRelativePosition.Y() * effectiveRadius,
								  lRelativePosition.X() * effectiveRadius + lRelativePosition.Y() * leg1) / lSqDist;
			rLegDirection = Point(lRelativePosition.X() * leg1 + lRelativePosition.Y() * effectiveRadius,
								  -lRelativePosition.X() * effectiveRadius + lRelativePosition.Y() * leg1) / lSqDist;
		}
		else if (s > 1.0f && lineSqDist <= sqEfRadius) {

			if (!right->IsConvex()) {
				continue;
			}

			left = right;

			float leg2 = std::sqrt(rSqDist - sqEfRadius);
			lLegDirection = Point(rRelativePosition.X() * leg2 - rRelativePosition.Y() * effectiveRadius,
								  rRelativePosition.X() * effectiveRadius + rRelativePosition.Y() * leg2) / rSqDist;
			rLegDirection = Point(rRelativePosition.X() * leg2 + rRelativePosition.Y() * effectiveRadius,
								  -rRelativePosition.X() * effectiveRadius + rRelativePosition.Y() * leg2) / rSqDist;
		}
		else {
			if (left->IsConvex()) {
				float leg1 = std::sqrt(lSqDist - sqEfRadius);
				lLegDirection = Point(lRelativePosition.X() * leg1 - lRelativePosition.Y() * effectiveRadius,
									  lRelativePosition.X() * effectiveRadius + lRelativePosition.Y() * leg1) / lSqDist;
			}
			else {
				lLegDirection = -NeighboursObst[i].second.dir;
			}

			if (right->IsConvex()) {
				float leg2 = std::sqrt(rSqDist - sqEfRadius);
				rLegDirection = Point(rRelativePosition.X() * leg2 + rRelativePosition.Y() * effectiveRadius,
									  -rRelativePosition.X() * effectiveRadius + rRelativePosition.Y() * leg2) /
								rSqDist;
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

		Point leftCutoff = (*left - effectivePosition) * invTimeBoundaryObst;
		Point rightCutoff = (*right - effectivePosition) * invTimeBoundaryObst;
		Vector cutoffVec = rightCutoff - leftCutoff;

		const float t = (right == left ? 0.5f : ((effectiveVelocity - leftCutoff).ScalarProduct(cutoffVec)) /
												cutoffVec.SquaredEuclideanNorm());
		const float tLeft = ((effectiveVelocity - leftCutoff).ScalarProduct(lLegDirection));
		const float tRight = ((effectiveVelocity - rightCutoff).ScalarProduct(rLegDirection));

		if ((t < 0.0f && tLeft < 0.0f) || (left == right && tLeft < 0.0f && tRight < 0.0f)) {
			Vector unitW = (effectiveVelocity - leftCutoff) / (effectiveVelocity - leftCutoff).EuclideanNorm();

			line.dir = Vector(unitW.Y(), -unitW.X());
			line.liesOn = leftCutoff + unitW * effectiveRadius * invTimeBoundaryObst;
			ORCALines.push_back(line);
			continue;
		}
		else if (t > 1.0f && tRight < 0.0f) {
			Vector unitW = (effectiveVelocity - rightCutoff) / (effectiveVelocity - rightCutoff).EuclideanNorm();

			line.dir = Vector(unitW.Y(), -unitW.X());
			line.liesOn = rightCutoff + unitW * effectiveRadius * invTimeBoundaryObst;
			ORCALines.push_back(line);
			continue;
		}

		float cutoffSqDist = ((t < 0.0f || t > 1.0f || right == left) ? std::numeric_limits<float>::infinity() : (
				effectiveVelocity - (leftCutoff + cutoffVec * t)).SquaredEuclideanNorm());
		float lLegSqDist = ((tLeft < 0.0f) ? std::numeric_limits<float>::infinity() : (effectiveVelocity - (leftCutoff +
																											lLegDirection *
																											tLeft)).SquaredEuclideanNorm());
		float rLegSqDist = ((tRight < 0.0f) ? std::numeric_limits<float>::infinity() : (effectiveVelocity -
																						(rightCutoff + rLegDirection *
																									   tRight)).SquaredEuclideanNorm());

		if (cutoffSqDist <= lLegSqDist && cutoffSqDist <= rLegSqDist) {
			line.dir = -NeighboursObst[i].second.dir;
			line.liesOn = leftCutoff + Point(-line.dir.Y(), line.dir.X()) * effectiveRadius * invTimeBoundaryObst;
			ORCALines.push_back(line);
			continue;
		}
		else if (lLegSqDist <= rLegSqDist) {
			if (isLLegForeign) {
				continue;
			}

			line.dir = lLegDirection;
			line.liesOn = leftCutoff + Point(-line.dir.Y(), line.dir.X()) * effectiveRadius * invTimeBoundaryObst;;
			ORCALines.push_back(line);
			continue;
		}
		else {
			if (isRLegForeign) {
				continue;
			}

			line.dir = -rLegDirection;
			line.liesOn = rightCutoff + Point(-line.dir.Y(), line.dir.X()) * effectiveRadius * invTimeBoundaryObst;
			ORCALines.push_back(line);
			continue;
		}
	}


	size_t numObstLines = ORCALines.size();

	//Получение ORCA-линий агентов

	Line currline;
	ORCADDAgent *curragent;
	Vector u, w;

	unsigned long minMaxNum = (param.agentsMaxNum < Neighbours.size()) ? param.agentsMaxNum : Neighbours.size();

	for (unsigned long i = 0; i < minMaxNum; i++) {
		auto Neighbour = Neighbours[i];
		curragent = dynamic_cast<ORCADDAgent *>(Neighbour.second);
		auto circlecenter = curragent->effectivePosition - this->effectivePosition; //(P_b - P_a)
		auto relvelocity = this->effectiveVelocity - curragent->effectiveVelocity; //(V_a - V_b)

		float radiussum = effectiveRadius + curragent->effectiveRadius; //(R_a + R_b)
		float radiussum2 = radiussum * radiussum;
		float distSq = circlecenter.SquaredEuclideanNorm();

		float trueSqRadSum = (param.radius + curragent->param.radius) * (param.radius + curragent->param.radius);
		if ((position - curragent->position).SquaredEuclideanNorm() < trueSqRadSum) {
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

		currline.liesOn = this->effectiveVelocity + u * 0.5f;
		ORCALines.push_back(currline);
		Neighbours.pop_back();
	}

	auto lineFail = Utils::linearProgram2(ORCALines, param.maxSpeed, this->prefV, false, this->newV);
	if (lineFail < this->ORCALines.size()) {
		Utils::linearProgram3(ORCALines, numObstLines, lineFail, param.maxSpeed, this->newV);
	}

	Neighbours.clear();
}


void ORCADDAgent::ApplyNewVelocity() {
	ComputeWheelsSpeed();
	effectiveVelocity = newV;
}


bool ORCADDAgent::UpdatePrefVelocity() {
	Point next;
	if (planner->GetNext(effectivePosition, next)) {
		Vector goalVector = next - effectivePosition;

		float dist = goalVector.EuclideanNorm();
		if (next == goal) {
			Vector true_goal_vector = next - position;
			float true_dist = true_goal_vector.EuclideanNorm();

			if (true_dist < options->delta) {
				prefV = Point();
				nextForLog = next;
				return true;
			}

			if (dist < options->delta) {
				goalVector = true_goal_vector;
				dist = true_dist;
			}
		}

		if (dist > CN_EPS) {
			goalVector = (goalVector / dist) * param.maxSpeed;
		}
		prefV = goalVector;

//		std::random_device rd{};
//		std::mt19937 gen{rd()};
//		std::normal_distribution<> d{0.0, 0.3};
//		float rand_x = d(gen);
//		float rand_y = d(gen);
//		prefV = prefV + Point(rand_x, rand_y);
		nextForLog = next;
		return true;
	}
	nextForLog = Point();
	prefV = Point();
	return false;
}


void ORCADDAgent::ComputeWheelsSpeed() {
	// TODO : make it faster
	// TODO : make comments

	const float invL = 1 / wheelTrack;
	float A1 = cos0 * 0.5f + D * sin0 * invL;
	float A2 = sin0 * 0.5f - D * cos0 * invL;
	float B1 = cos0 * 0.5f - D * sin0 * invL;
	float B2 = sin0 * 0.5f + D * cos0 * invL;

	float divB = B1 * (1 / B2);

	float invden = 1 / (A1 - A2 * divB);
	float l = (newV.X() - newV.Y() * divB) * invden;
	float r = (newV.Y() - A2 * l) * (1 / B2);

	if (isnan(l) or isnan(r)) // TODO refactoring
	{
		tet += 0.0001;
		sin0 = sin(tet);
		cos0 = cos(tet);


		const float invL = 1 / wheelTrack;
		float A1 = cos0 * 0.5f + D * sin0 * invL;
		float A2 = sin0 * 0.5f - D * cos0 * invL;
		float B1 = cos0 * 0.5f - D * sin0 * invL;
		float B2 = sin0 * 0.5f + D * cos0 * invL;

		float divB = B1 * (1 / B2);

		float invden = 1 / (A1 - A2 * divB);
		float l = (newV.X() - newV.Y() * divB) * invden;
		float r = (newV.Y() - A2 * l) * (1 / B2);

	}

	this->leftV = (abs(l) <= param.maxSpeed) ? l : copysign(param.maxSpeed, l);
	this->rightV = (abs(r) <= param.maxSpeed) ? r : copysign(param.maxSpeed, r);

	float avg = (leftV + rightV) * 0.5f;

	currV = Point(avg * cos0, avg * sin0);
	tet = tet + ((rightV - leftV) * invL) * options->timestep;


	sin0 = sin(tet);
	cos0 = cos(tet);


}


void ORCADDAgent::SetPosition(const Point &pos) {
	Agent::SetPosition(pos);
	effectivePosition = Point(pos.X() + D * cos0, pos.Y() + D * sin0);
}

bool ORCADDAgent::isFinished() {
	return ((this->effectivePosition - this->goal).EuclideanNorm() < options->delta);
}