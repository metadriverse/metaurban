#include "geom.h"

int Node::convolution(int width, int height, bool withTime) const {
	int res = withTime ? width * height * g : 0;
	return res + i * width + j;
}

Point::Point() {
	x = 0.0;
	y = 0.0;
}


Point::Point(float x, float y) {
	this->x = x;
	this->y = y;
}


float Point::X() const {
	return x;
}


float Point::Y() const {
	return y;
}


Point::Point(const Point &obj) {
	this->x = obj.x;
	this->y = obj.y;
}


//std::pair<float, float> Point::GetPair()
//{
//    return {x, y};
//}
//
//std::string Point::ToString() const
//{
//    return "x: " + std::to_string(x) + "\ty: "
//           + std::to_string(y) + "\n";
//
//}

bool Vertex::IsConvex() const {
	return convex;
}


void Vertex::SetConvex(bool cvx) {
	convex = cvx;
}

using namespace Utils;

float Utils::SqPointSegDistance(Point L1, Point L2, Point P) {
	auto v = L2 - L1;
	auto w = P - L1;
	float c1, c2;
	if ((c1 = w.ScalarProduct(v)) <= 0.0f) {
		return (P - L1).SquaredEuclideanNorm();
	}
	if ((c2 = v.ScalarProduct(v)) <= c1) {
		return (P - L2).SquaredEuclideanNorm();
	}

	float b = c1 / c2;
	Point Pb = L1 + v * b;
	return (P - Pb).SquaredEuclideanNorm();
}


bool Utils::linearProgram1(const std::vector<Line> &lines, unsigned long curr, float radius, const Vector &optVelocity,
						   bool directionOpt, Vector &result) {
	float dotProduct = lines[curr].liesOn.ScalarProduct(lines[curr].dir);
	float discriminant = dotProduct * dotProduct + radius * radius - lines[curr].liesOn.SquaredEuclideanNorm();

	if (discriminant < 0.0f) {
		// Максимальная скорость не позволяет удовлетворить это условие
		return false;
	}

	float sqrtDiscriminant = std::sqrt(discriminant);
	float tLeft = -dotProduct - sqrtDiscriminant;
	float tRight = -dotProduct + sqrtDiscriminant;

	for (int i = 0; i < curr; ++i) {

		const float denominator = lines[curr].dir.Det(lines[i].dir);
		const float numerator = lines[i].dir.Det(lines[curr].liesOn - lines[i].liesOn);

		if (std::fabs(denominator) <= CN_EPS) {
			// Текущая и сравниваемая линии параллельны
			if (numerator < 0.0f) {
				return false;
			}
			else {
				continue;
			}
		}

		const float t = numerator / denominator;

		if (denominator >= 0.0f) {
			/* Line i bounds line lineNo on the right. */
			tRight = std::min(tRight, t);
		}
		else {
			/* Line i bounds line lineNo on the left. */
			tLeft = std::max(tLeft, t);
		}

		if (tLeft > tRight) {
			return false;
		}
	}

	if (directionOpt) {
		/* Optimize direction. */
		if (optVelocity.ScalarProduct(lines[curr].dir) > 0.0f) {
			/* Take right extreme. */
			result = lines[curr].liesOn + lines[curr].dir * tRight;
		}
		else {
			/* Take left extreme. */
			result = lines[curr].liesOn + lines[curr].dir * tLeft;
		}
	}
	else {
		/* Optimize closest point. */
		const float t = lines[curr].dir.ScalarProduct(optVelocity - lines[curr].liesOn);

		if (t < tLeft) {
			result = lines[curr].liesOn + lines[curr].dir * tLeft;
		}
		else if (t > tRight) {
			result = lines[curr].liesOn + lines[curr].dir * tRight;
		}
		else {
			result = lines[curr].liesOn + lines[curr].dir * t;
		}
	}

	return true;
}


unsigned long int
Utils::linearProgram2(const std::vector<Line> &lines, float radius, const Vector &optVelocity, bool directionOpt,
					  Vector &result) {
	if (directionOpt) {
		result = optVelocity * radius;
	}
	else if (optVelocity.SquaredEuclideanNorm() > radius * radius) {
		result = (optVelocity / optVelocity.EuclideanNorm()) * radius;
	}
	else {
		result = optVelocity;
	}

	for (unsigned long int i = 0; i < lines.size(); ++i) {

		if (lines[i].dir.Det(lines[i].liesOn - result) > 0.0f) {
			const Vector tempResult = result;

			if (!linearProgram1(lines, i, radius, optVelocity, directionOpt, result)) {
				result = tempResult;
				return i;
			}
		}
	}

	return lines.size();
}


void Utils::linearProgram3(const std::vector<Line> &lines, size_t numObstLines, size_t beginLine, float radius,
						   Vector &result) {
	float distance = 0.0f;

	for (size_t i = beginLine; i < lines.size(); ++i) {
		if (lines[i].dir.Det(lines[i].liesOn - result) > distance) {
			/* Result does not satisfy constraint of line i. */
			std::vector<Line> projLines(lines.begin(), lines.begin() + static_cast<ptrdiff_t>(numObstLines));

			for (size_t j = numObstLines; j < i; ++j) {
				Line line;

				float determinant = lines[i].dir.Det(lines[j].dir);

				if (std::fabs(determinant) <= CN_EPS) {
					/* Line i and line j are parallel. */
					if (lines[i].dir.ScalarProduct(lines[j].dir) > 0.0f) {
						/* Line i and line j point in the same direction. */
						continue;
					}
					else {
						/* Line i and line j point in opposite direction. */
						line.liesOn = (lines[i].liesOn + lines[j].liesOn) * 0.5f;
					}
				}
				else {
					line.liesOn = lines[i].liesOn +
								  lines[i].dir * (lines[j].dir.Det(lines[i].liesOn - lines[j].liesOn) / determinant);
				}


				line.dir = (lines[j].dir - lines[i].dir);
				line.dir = line.dir / line.dir.EuclideanNorm();
				projLines.push_back(line);
			}

			const Vector tempResult = result;

			if (linearProgram2(projLines, radius, Vector(-lines[i].dir.Y(), lines[i].dir.X()), true, result) <
				projLines.size()) {
				result = tempResult;
			}

			distance = lines[i].dir.Det(lines[i].liesOn - result);
		}
	}
}

bool Node::operator<(const Node &other) const {
	return std::tuple<int, int, int, int>(F, -g, i, j) <
		   std::tuple<int, int, int, int>(other.F, -other.g, other.i, other.j);
}


