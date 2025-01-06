#include "sub_map.h"


SubMap::SubMap() {
	cellSize = 0.0;
	size = {0, 0};
	origin = {0, 0};
	divKoef = 0;
	fullMap = nullptr;
	emptyCells = 0;
}


SubMap::SubMap(const Map *map, Node mapOrigin, std::pair<int, int> mapSize, int divK) {
	this->cellSize = map->GetCellSize() / divK;
	origin = mapOrigin;
	size = {mapSize.first * divK, mapSize.second * divK};
	divKoef = divK;
	fullMap = map;
	emptyCells = 0;
	for (int i = 0; i < size.first; i++) {
		for (int j = 0; j < size.second; j++) {
			if (CellOnGrid(i, j) && CellIsTraversable(i, j)) {
				emptyCells++;
			}
		}
	}
	emptyCells *= (divK * divK);
}


SubMap::SubMap(const SubMap &obj) {
	this->cellSize = obj.cellSize;
	origin = obj.origin;
	size = obj.size;
	divKoef = obj.divKoef;
	fullMap = obj.fullMap;
	emptyCells = obj.emptyCells;
}


SubMap::~SubMap() {
	fullMap = nullptr;
}


bool SubMap::CellIsObstacle(int i, int j) const {
	int iM = origin.i + i / divKoef;
	int jM = origin.j + j / divKoef;
	return CellOnGrid(i, j) && fullMap->CellIsObstacle(iM, jM);
}


bool SubMap::CellIsTraversable(int i, int j) const {
	int iM = origin.i + i / divKoef;
	int jM = origin.j + j / divKoef;
	return CellOnGrid(i, j) && fullMap->CellIsTraversable(iM, jM);
}


bool SubMap::CellOnGrid(int i, int j) const {
	return (i < size.first && i >= 0 && j < size.second && j >= 0);
}


int SubMap::GetHeight() const {
	return size.first;
}


int SubMap::GetWidth() const {
	return size.second;
}


float SubMap::GetCellSize() const {
	return cellSize;
}


Node SubMap::GetClosestNode(const Point &point) const {
	Node res;
	res.i = fullMap->GetHeight() * divKoef - 1 - (int) (point.Y() / cellSize);
	res.j = (int) (point.X() / cellSize);

	res.i = res.i - origin.i * divKoef;
	res.j = res.j - origin.j * divKoef;

	if (res.i < 0) {
		res.i = 0;
	}
	if (res.i > size.first - 1) {
		res.i = size.first - 1;
	}
	if (res.j < 0) {
		res.j = 0;
	}
	if (res.j > size.second - 1) {
		res.j = size.second - 1;
	}

	return res;
}


Point SubMap::GetPoint(const Node &node) const {
	Point originPoint = fullMap->GetPoint(origin) + Point(-fullMap->GetCellSize() / 2, fullMap->GetCellSize() / 2);

	return {(originPoint.X() + node.j * cellSize + cellSize / 2),
			(originPoint.Y() - (node.i * cellSize + cellSize / 2))};
}


SubMap &SubMap::operator=(const SubMap &obj) {
	if (this != &obj) {
		this->cellSize = obj.cellSize;
		origin = obj.origin;
		size = obj.size;
		divKoef = obj.divKoef;
		fullMap = obj.fullMap;
		emptyCells = obj.emptyCells;
	}
	return *this;
}


bool SubMap::CellIsTraversable(int i, int j, const std::unordered_set<Node, NodeHash> &occupiedNodes) const {
	return CellIsTraversable(i, j) && occupiedNodes.find(Node(i, j)) == occupiedNodes.end();
}


int SubMap::GetCellDegree(int i, int j) const {
	int degree = 0;
	for (int di = -1; di <= 1; ++di) {
		for (int dj = -1; dj <= 1; ++dj) {
			if ((di == 0) ^ (dj == 0)) {
				if (CellOnGrid(i + di, j + dj) && !CellIsObstacle(i + di, j + dj)) {
					++degree;
				}
			}
		}
	}
	return degree;
}


int SubMap::GetEmptyCellCount() const {
	return emptyCells;
}


Node SubMap::FindUnoccupiedNode(Node start, std::unordered_map<int, Node> occupied) {
	if (occupied.find(start.i * GetWidth() + start.j) == occupied.end() && CellIsTraversable(start.i, start.j)) {
		return start;
	}
	std::list<Node> open = std::list<Node>();
	std::unordered_map<int, Node> close = std::unordered_map<int, Node>();

	open.push_back(start);

	while (!open.empty()) {

		Node curr = open.front();

		close.insert({curr.i * GetWidth() + curr.j, curr});
		open.pop_front();

		if (CellIsTraversable(curr.i, curr.j) && occupied.find(curr.i * GetWidth() + curr.j) == occupied.end()) {
			return curr;
		}

		std::vector<Node> successors = {Node(curr.i + 1, curr.j), Node(curr.i - 1, curr.j), Node(curr.i, curr.j + 1),
										Node(curr.i, curr.j - 1)};

		for (auto &s: successors) {
			if (CellOnGrid(s.i, s.j)) {
				if (CellIsObstacle(curr.i, curr.j) || CellIsTraversable(s.i, s.j)) {
					if (close.find(s.i * GetWidth() + s.j) == close.end() &&
						find(open.begin(), open.end(), s) == open.end()) {
						open.push_back(s);
					}

				}

			}
		}

	}
	return Node(-1, -1);
}


Node SubMap::FindAccessibleNodeForGoal(Node start, Point goal, std::unordered_map<int, Node> occupied,
									   std::unordered_map<int, Node> undesirable) {
	std::list<Node> open = std::list<Node>();
	std::unordered_map<int, Node> close = std::unordered_map<int, Node>();
	auto popMin = [&open]() -> Node {
		auto bestIt = open.begin();
		for (auto node = open.begin(); node != open.end(); node++) {
			if (node->g < bestIt->g) {
				bestIt = node;
			}
		}
		Node best = *bestIt;
		open.erase(bestIt);
		return best;
	};

	auto addNode = [&open](Node item) {
		for (auto node = open.begin(); node != open.end(); node++) {
			if (*node == item) {
				if (node->g > item.g) {
					node->H = item.H;
					node->g = item.g;
					node->F = item.F;
					node->parent = item.parent;
				}
				return;
			}
		}
		open.push_back(item);
		return;
	};
	Node currNode;

	if (occupied.find(start.i * GetWidth() + start.j) != occupied.end() ||
		undesirable.find(start.i * GetWidth() + start.j) != undesirable.end()) {
		auto tmpOcupied = occupied;
		tmpOcupied.insert(undesirable.begin(), undesirable.end());
		currNode = FindUnoccupiedNode(start, tmpOcupied);
		if (currNode.i < 0) {
			return currNode;
		}
	}
	else {
		currNode.i = start.i;
		currNode.j = start.j;
	}

	currNode.g = 0;
	currNode.H = (GetPoint(currNode) - goal).EuclideanNorm();
	currNode.F = currNode.H;
	currNode.parent = nullptr;


	addNode(currNode);
	Node bestNode = currNode;

	while (open.size()) {
		currNode = popMin();
		close.insert({currNode.i * GetWidth() + currNode.j, currNode});

		if ((currNode.F < bestNode.F || (abs(currNode.F - bestNode.F) < CN_EPS && currNode.H < bestNode.H)) &&
			undesirable.find(currNode.i * GetWidth() + currNode.j) == undesirable.end()) {
			bestNode = currNode;
		}

		if (currNode.H < CN_EPS && undesirable.find(currNode.i * GetWidth() + currNode.j) == undesirable.end()) {
			currNode.parent = nullptr;
			currNode.F = 0.0;
			currNode.H = 0.0;
			currNode.g = 0.0;


//            if(occupied.find(currNode.i * GetWidth() + currNode.j) != occupied.end())
//            {
//                //  std::cout << "AAAAAAAAAA\n";
//            }
			return currNode;
		}

		std::list<Node> successors = {Node(currNode.i + 1, currNode.j), Node(currNode.i - 1, currNode.j),
									  Node(currNode.i, currNode.j + 1), Node(currNode.i, currNode.j - 1)};
		auto it = successors.begin();
		auto parent = &(close.find(currNode.i * GetWidth() + currNode.j)->second);
		while (it != successors.end()) {
			if (CellOnGrid(it->i, it->j) && CellIsTraversable(it->i, it->j) &&
				occupied.find(it->i * GetWidth() + it->j) == occupied.end()) {
				if (close.find(it->i * GetWidth() + it->j) == close.end()) {
					it->parent = parent;
					it->H = (GetPoint(currNode) - goal).EuclideanNorm();
					it->F = it->g + it->H;
//                    //  std::cout << it->g << "\n";
					addNode(*it);
				}
			}
			it++;
		}
	}
	bestNode.parent = nullptr;
	bestNode.F = 0.0;
	bestNode.H = 0.0;
	bestNode.g = 0.0;

//    if(occupied.find(bestNode.i * GetWidth() + bestNode.j) != occupied.end())
//    {
//        //  std::cout << "AAAAAAAAAA 2\n";
//    }
	return bestNode;
}


Node SubMap::FindCloseToPointAvailableNode(Point pos, std::unordered_map<int, Node> occupied) {
	auto start = this->GetClosestNode(pos);


	if (occupied.find(start.i * GetWidth() + start.j) == occupied.end() && CellIsTraversable(start.i, start.j)) {
		return start;
	}

	start.H = (this->GetPoint(start) - pos).EuclideanNorm();

	auto cmp = [](Node left, Node right) { return (left.H > right.H); };

	std::priority_queue<Node, std::vector<Node>, decltype(cmp)> open(cmp);

	std::unordered_map<int, Node> openDupl = std::unordered_map<int, Node>();

	std::unordered_map<int, Node> close = std::unordered_map<int, Node>();

	open.push(start);
	openDupl.insert({start.i * GetWidth() + start.j, start});

	while (!open.empty()) {
		Node curr = open.top();
		close.insert({curr.i * GetWidth() + curr.j, curr});
		open.pop();
		openDupl.erase(curr.i * GetWidth() + curr.j);

		if (CellIsTraversable(curr.i, curr.j) && occupied.find(curr.i * GetWidth() + curr.j) == occupied.end()) {
			return curr;
		}

		std::vector<Node> successors = {Node(curr.i + 1, curr.j), Node(curr.i - 1, curr.j), Node(curr.i, curr.j + 1),
										Node(curr.i, curr.j - 1)};


//            while(!open.empty())
//            {
//                //  std::cout << open.top().H << " ";
//                open.pop();
//            }
//            //  std::cout << '\n';


		for (auto &s: successors) {
			if (CellOnGrid(s.i, s.j)) {
				if (CellIsObstacle(curr.i, curr.j) || CellIsTraversable(s.i, s.j)) {
					if (close.find(s.i * GetWidth() + s.j) == close.end() &&
						openDupl.find(s.i * GetWidth() + s.j) == openDupl.end()) {
						s.H = (this->GetPoint(s) - pos).SquaredEuclideanNorm();
						open.push(s);
						openDupl.insert({s.i * GetWidth() + s.j, s});
					}
				}

			}
		}

	}
	return Node(-1, -1);
}


bool SubMap::PointBelongsToArea(const Point &point) const {
	Node res;
	res.i = static_cast<int>(fullMap->GetHeight()) * divKoef - 1 - (int) (point.Y() / cellSize);
	res.j = (int) (point.X() / cellSize);

	res.i = res.i - origin.i * divKoef;
	res.j = res.j - origin.j * divKoef;

	if (res.i < 0 || res.i > size.first - 1 || res.j < 0 || res.j > size.second - 1) {
		return false;
	}
	return true;
}
