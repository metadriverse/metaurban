#include <cmath>
#include <vector>
#include <string>
#include <tuple>
#include <cstddef>

#include "const.h"

#ifndef ORCA_GEOM_H
#define ORCA_GEOM_H


class Node {
	public:
		int i, j;
		double F, g, H;
		Node *parent;

		int subgraph = -1;
		int depth;
		int conflictsCount;

		Node(int i = 0, int j = 0, Node *p = nullptr, double g = 0, double h = 0, int d = 0, int conf = 0)
				: i(i), j(j), parent(p), g(g), H(h), F(g + h), depth(d), conflictsCount(conf) {}

		bool operator!=(const Node &other) const;

		bool operator<(const Node &other) const;


		bool operator==(const Node &another) const;

		int convolution(int width, int height, bool withTime = false) const;
};


struct NodeHash {
	size_t operator()(const Node &node) const {
		return (node.i + node.j) * (node.i + node.j + 1) + node.j;
	}
};

struct NodePairHash {
	size_t operator()(const std::pair<Node, Node> &pair) const {
		NodeHash nodeHash;
		size_t hash1 = nodeHash(pair.first), hash2 = nodeHash(pair.second);
		return (hash1 + hash2) * (hash1 + hash2 + 1) + hash2;
	}
};


class Point {
	public:
		Point();

		Point(float x, float y);

		Point(const Point &obj);

		virtual ~Point() = default;

		float X() const;

		float Y() const;

		std::pair<float, float> GetPair();

		float ScalarProduct(const Point &another) const;

		float EuclideanNorm() const;

		float SquaredEuclideanNorm() const;

		float Det(Point another) const;

		std::string ToString() const {
			std::string res = std::to_string(x) + " " + std::to_string(y);
			return res;
		}

		Point operator-(const Point &another) const;

		Point operator+(const Point &another) const;

		bool operator==(const Point &another) const;

		Point operator*(float k) const;

		Point operator/(float k) const;

		Point operator-() const;

		virtual Point &operator=(const Point &obj);


	private:
		float x;
		float y;
};


class Line {
	public:
		Vector dir;
		Point liesOn;
};


class Vertex : public Point {
	public:
		Vertex() : Point() {}

		Vertex(float x, float y, bool cvx = false) : Point(x, y), convex(cvx) {}

		Vertex(const Point &obj, bool cvx = false) : Point(obj), convex(cvx) {}

		Vertex(const Vertex &obj) : Point(obj), convex(obj.convex) {}

		~Vertex() override = default;

		bool IsConvex() const;

		void SetConvex(bool cvx);

		Vertex &operator=(const Vertex &obj);

	private:
		bool convex;
};


class ObstacleSegment {
	public:
		ObstacleSegment() = default;

		ObstacleSegment(const ObstacleSegment &obj) : left(obj.left), right(obj.right), id(obj.id), next(obj.next),
													  prev(obj.prev), dir(obj.dir) {}

		ObstacleSegment(int id, const Vertex &left, const Vertex &right) : left(left), right(right), id(id) {
			dir = right - left;
			dir = dir / dir.EuclideanNorm();
		}

		~ObstacleSegment() = default;

		bool operator==(const ObstacleSegment &another) const;

		ObstacleSegment &operator=(const ObstacleSegment &obj);

		int id;
		Vertex left;
		Vertex right;
		Vector dir;
		ObstacleSegment *next;
		ObstacleSegment *prev;

};

struct ActorMove {
	int di, dj;
	int id;

	ActorMove(int i, int j, int id) : di(i), dj(j), id(id) {}
};

namespace Utils {
	float SqPointSegDistance(Point L1, Point L2, Point P);

	bool linearProgram1(const std::vector<Line> &lines, unsigned long curr, float radius, const Vector &optVelocity,
						bool directionOpt, Vector &result);

	unsigned long int linearProgram2(const std::vector<Line> &lines, float radius, const Vector &optVelocity,
									 bool directionOpt, Vector &result);

	void linearProgram3(const std::vector<Line> &lines, size_t numObstLines, size_t beginLine,
						float radius, Vector &result);

	template<typename T>
	bool More(std::pair<float, T> a, std::pair<float, T> b) {
		return (a.first > b.first);
	}

	template<typename T>
	bool Less(std::pair<float, T> a, std::pair<float, T> b) {
		return (a.first < b.first);
	}

	struct NodeHash {
		size_t operator()(const Node &node) const {
			return (node.i + node.j) * (node.i + node.j + 1) + node.j;
		}
	};

	struct NodePairHash {
		size_t operator()(const std::pair<Node, Node> &pair) const {
			NodeHash nodeHash;
			size_t hash1 = nodeHash(pair.first), hash2 = nodeHash(pair.second);
			return (hash1 + hash2) * (hash1 + hash2 + 1) + hash2;
		}
	};
};


namespace std {
	template<>
	struct hash<Node> {
		size_t operator()(const Node &node) const {
			return (node.i + node.j) * (node.i + node.j + 1) + node.j;
		}
	};

	template<>
	struct hash<std::pair<Node, Node>> {
		size_t operator()(const std::pair<Node, Node> &pair) const {
			hash<Node> nodeHash;
			size_t hash1 = nodeHash(pair.first), hash2 = nodeHash(pair.second);
			return (hash1 + hash2) * (hash1 + hash2 + 1) + hash2;
		}
	};


}


/*********************************************************
 *                Methods Implementations                *
 *********************************************************/


inline bool Node::operator==(const Node &another) const {
	return i == another.i && j == another.j;
}

inline bool Node::operator!=(const Node &other) const {
	return i != other.i || j != other.j;
}


inline bool ObstacleSegment::operator==(const ObstacleSegment &another) const {
	return (this->id == another.id);
}

inline ObstacleSegment &ObstacleSegment::operator=(const ObstacleSegment &obj) {

	id = obj.id;
	left = obj.left;
	right = obj.right;
	dir = obj.dir;
	next = obj.next;
	prev = obj.prev;
	return *this;
}

inline float Point::ScalarProduct(const Point &another) const {
	return this->x * another.x + this->y * another.y;
}


inline Point Point::operator-(const Point &another) const {
	return {this->x - another.x, this->y - another.y};
}


inline Point Point::operator+(const Point &another) const {
	return {this->x + another.x, this->y + another.y};
}


inline Point Point::operator*(float k) const {
	return {this->x * k, this->y * k};
}


inline Point Point::operator/(float k) const {
	const float invK = 1.0f / k;
	return {this->x * invK, this->y * invK};
}


inline float Point::SquaredEuclideanNorm() const {
	return this->ScalarProduct(*this);
}


inline float Point::EuclideanNorm() const {
	return std::sqrt(this->ScalarProduct(*this));
}


inline float Point::Det(Point another) const {
	return (this->x * another.y - this->y * another.x);
}


inline bool Point::operator==(const Point &another) const {
	return (this->x == another.x) && (this->y == another.y);
}


inline Point Point::operator-() const {
	return Point(-this->x, -this->y);
}


inline Point &Point::operator=(const Point &obj) {
	x = obj.x;
	y = obj.y;
	return *this;
}


inline Vertex &Vertex::operator=(const Vertex &obj) {
	Point::operator=(obj);
	convex = obj.convex;
	return *this;
}


#endif //ORCA_GEOM_H
