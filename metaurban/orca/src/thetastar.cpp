#include "thetastar.h"

ThetaStar::ThetaStar(const Map &map, const environment_options &options, const Point &start, const Point &goal,
					 const float &radius)
		: PathPlanner(map, options, start, goal, radius) {
	currPath = std::list<Point>();
	close = std::unordered_map<int, Node>();
	open = std::vector<std::list<Node>>();
	openSize = 0;
	glPathCreated = false;
	visChecker = LineOfSight(this->radius / map.GetCellSize());
	past = glStart;
}


ThetaStar::ThetaStar(const ThetaStar &obj) : PathPlanner(obj) {
	currPath = obj.currPath;
	close = obj.close;
	open = obj.open;
	openSize = obj.openSize;
	glPathCreated = obj.glPathCreated;
	visChecker = obj.visChecker;
	past = obj.past;

}


ThetaStar::~ThetaStar() {}


bool ThetaStar::GetNext(const Point &curr, Point &next) {
	if (glPathCreated) {
		if (currPath.size() == 0) {
			currPath.push_back(glGoal);
		}

		if (currPath.size() > 1) {
			float sqDistToCurr = (currPath.front() - curr).SquaredEuclideanNorm();
			float sqDelta = options->delta * options->delta; // TODO Separate option
			if (sqDistToCurr < sqDelta) {

				if (!(currPath.front() == this->glGoal)) {
					past = currPath.front();
				}
				currPath.pop_front();
				next = currPath.front();
				return true;
			}
		}

		Node currNode = map->GetClosestNode(curr);

		Node nextNode = map->GetClosestNode(currPath.front());
		if (std::next(currPath.begin()) != currPath.end()) {
			Node nextNextNode = map->GetClosestNode(*std::next(currPath.begin()));
			if (currNode == nextNode || visChecker.checkLine(currNode.i, currNode.j, nextNode.i, nextNode.j, *map)) {
				next = currPath.front();
				if (currNode == nextNextNode ||
					visChecker.checkLine(currNode.i, currNode.j, nextNextNode.i, nextNextNode.j, *map)) {
					currPath.pop_front();
					next = currPath.front();
				}
				return true;
			}
		}
		else {
			if (currNode == nextNode || visChecker.checkLine(currNode.i, currNode.j, nextNode.i, nextNode.j, *map)) {
				next = currPath.front();
				return true;
			}
		}


//        Node currNode = map->GetClosestNode(curr), nextNode = map->GetClosestNode(currPath.front());
//        if(currNode == nextNode || visChecker.checkLine(currNode.i, currNode.j, nextNode.i, nextNode.j, *map))
//        {
//            next = currPath.front();
//            return true;
//        }

		Point last = currPath.front();
		currPath.pop_front();

//        bool isLastAccessible = SearchPath(map->GetClosestNode(curr), map->GetClosestNode(last));
//        if(isLastAccessible)
//        {
//            next = currPath.front();
//            return true;
//        }
		currPath.clear();
		bool isGoalAccessible = SearchPath(map->GetClosestNode(curr), map->GetClosestNode(glGoal));
		if (isGoalAccessible) {
			next = currPath.front();
			return true;
		}


	}
	return false;
}


bool ThetaStar::SearchPath(const Node &start, const Node &goal) {
	if (glPathCreated) {
		close.clear();
		for (auto &l: open) {
			l.clear();
		}
		openSize = 0;
	}
	else {
		open.resize(map->GetHeight());
	}

	Node curNode;
	curNode.i = start.i;
	curNode.j = start.j;
	curNode.g = 0;
	curNode.H = ComputeHFromCellToCell(curNode.i, curNode.j, goal.i, goal.j);
	curNode.F = options->hweight * curNode.H;
	curNode.parent = nullptr;
	AddOpen(curNode);
	int closeSize = 0;
	bool pathfound = false;
	while (!StopCriterion()) {
		curNode = FindMin();
		close.insert({curNode.i * map->GetWidth() + curNode.j, curNode});
		closeSize++;
		open[curNode.i].pop_front();
		openSize--;
		if (curNode == goal) {
			pathfound = true;
			break;
		}
		std::list<Node> successors = FindSuccessors(curNode);
		std::list<Node>::iterator it = successors.begin();
		auto parent = &(close.find(curNode.i * map->GetWidth() + curNode.j)->second);
		while (it != successors.end()) {
			it->parent = parent;
			it->H = ComputeHFromCellToCell(it->i, it->j, goal.i, goal.j);
			*it = ResetParent(*it, *it->parent);
			it->F = it->g + options->hweight * it->H;
			AddOpen(*it);
			it++;
		}
	}

	if (pathfound) {
		MakePrimaryPath(curNode);
	}
	return pathfound;
}


bool ThetaStar::StopCriterion() const {
	if (!openSize) {
		return true;
	}
	return false;
}


void ThetaStar::AddOpen(Node newNode) {
	std::list<Node>::iterator iter, pos;

	if (open[newNode.i].size() == 0) {
		open[newNode.i].push_back(newNode);
		openSize++;
		return;
	}

	pos = open[newNode.i].end();
	bool posFound = false;
	for (iter = open[newNode.i].begin(); iter != open[newNode.i].end(); ++iter) {
		if (!posFound && iter->F >= newNode.F) {
			if (iter->F == newNode.F) {
				if ((options->breakingties == CN_SP_BT_GMAX && newNode.g >= iter->g)
					|| (options->breakingties == CN_SP_BT_GMIN && newNode.g <= iter->g)) {
					pos = iter;
					posFound = true;
				}
			}
			else {
				pos = iter;
				posFound = true;
			}
		}
		if (iter->j == newNode.j) {
			if (newNode.F >= iter->F)
				return;
			else {
				if (pos == iter) {
					iter->F = newNode.F;
					iter->g = newNode.g;
					iter->parent = newNode.parent;
					return;
				}
				open[newNode.i].erase(iter);
				openSize--;
				break;
			}
		}
	}
	openSize++;
	open[newNode.i].insert(pos, newNode);
}


Node ThetaStar::FindMin() {
	Node min;
	min.F = std::numeric_limits<double>::infinity();
	for (int i = 0; i < open.size(); i++) {
		if (!open[i].empty() && open[i].begin()->F <= min.F) {
			if (open[i].begin()->F == min.F) {
				if ((options->breakingties == CN_SP_BT_GMAX && open[i].begin()->g >= min.g)
					|| (options->breakingties == CN_SP_BT_GMIN && open[i].begin()->g <= min.g)) {
					min = *open[i].begin();
				}
			}
			else {
				min = *open[i].begin();
			}
		}
	}
	return min;
}


float ThetaStar::ComputeHFromCellToCell(int i1, int j1, int i2, int j2) const {
	switch (options->metrictype) {
		case CN_SP_MT_EUCL:
			return static_cast<float>(sqrt((i2 - i1) * (i2 - i1) + (j2 - j1) * (j2 - j1)));
		case CN_SP_MT_DIAG:
			return static_cast<float>(abs(abs(i2 - i1) - abs(j2 - j1)) +
									  sqrt(2) * (std::min(abs(i2 - i1), abs(j2 - j1))));
		case CN_SP_MT_MANH:
			return (abs(i2 - i1) + abs(j2 - j1));
		case CN_SP_MT_CHEB:
			return std::max(abs(i2 - i1), abs(j2 - j1));
		default:
			return 0;
	}
}


Node ThetaStar::ResetParent(Node current, Node parent) {

	if (parent.parent == nullptr)
		return current;
	if (current == *parent.parent)
		return current;

	if (visChecker.checkLine(parent.parent->i, parent.parent->j, current.i, current.j, *map)) {
		current.g = parent.parent->g + Distance(parent.parent->i, parent.parent->j, current.i, current.j);
		current.parent = parent.parent;
		return current;
	}
	return current;
}


float ThetaStar::Distance(int i1, int j1, int i2, int j2) const {
	return static_cast<float>(sqrt(pow(i1 - i2, 2.0f) + pow(j1 - j2, 2.0f)));
}


void ThetaStar::MakePrimaryPath(Node curNode) {


	Node current = curNode;
	while (current.parent) {
		currPath.push_front(map->GetPoint(current));
		current = *current.parent;
	}
	////  std::cout<<map->GetPoint(current).ToString();
}


std::list<Node> ThetaStar::FindSuccessors(Node curNode) {
	Node newNode;
	std::list<Node> successors;
	for (int i = -1; i <= +1; i++)
		for (int j = -1; j <= +1; j++)
			if ((i != 0 || j != 0) && map->CellOnGrid(curNode.i + i, curNode.j + j) &&
				(visChecker.checkTraversability(curNode.i + i, curNode.j + j, *map))) {
				if (i != 0 && j != 0) {
					if (!options->cutcorners) {
						if (map->CellIsObstacle(curNode.i, curNode.j + j) ||
							map->CellIsObstacle(curNode.i + i, curNode.j))
							continue;
					}
					else if (!options->allowsqueeze) {
						if (map->CellIsObstacle(curNode.i, curNode.j + j) &&
							map->CellIsObstacle(curNode.i + i, curNode.j))
							continue;
					}
				}
				if (close.find((curNode.i + i) * map->GetWidth() + curNode.j + j) == close.end()) {
					newNode.i = curNode.i + i;
					newNode.j = curNode.j + j;
					if (i == 0 || j == 0)
						newNode.g = curNode.g + 1;
					else
						newNode.g = curNode.g + sqrt(2);
					successors.push_front(newNode);
				}
			}
	return successors;
}


bool ThetaStar::CreateGlobalPath() {
	if (!glPathCreated) {
		Node start = map->GetClosestNode(glStart);
		Node goal = map->GetClosestNode(glGoal);

		glPathCreated = SearchPath(start, goal);
		//currPath.push_back(glGoal);
	}
	return glPathCreated;
}


ThetaStar &ThetaStar::operator=(const ThetaStar &obj) {
	if (this != &obj) {
		PathPlanner::operator=(obj);
		currPath = obj.currPath;
		close = obj.close;
		open = obj.open;
		openSize = obj.openSize;
		glPathCreated = obj.glPathCreated;
		visChecker = obj.visChecker;
		past = obj.past;
	}
	return *this;
}

ThetaStar *ThetaStar::Clone() const {
	return new ThetaStar(*this);
}

void ThetaStar::AddPointToPath(Point p) {
	currPath.push_front(p);
}

Point ThetaStar::PullOutNext() {
	if (currPath.size() > 0) {
		Point res = currPath.front();
		if (currPath.size() > 1) {
			currPath.pop_front();
		}
		return res;
	}

	return Point();
}

Point ThetaStar::GetPastPoint() {
	return past;
}
