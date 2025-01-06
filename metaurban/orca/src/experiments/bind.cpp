#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <string>
#include <ostream>
#include <iomanip>
#include <locale>
#include "mission.h"

#define STEP_MAX            1200
#define IS_TIME_BOUNDED     false
#define STOP_BY_SPEED       true
#define TIME_MAX            1200 * 60 * 1


namespace py = pybind11;

unordered_map<int, trajectory_dict> demo(string taskfile, int num)
{
	unordered_map<int, trajectory_dict> resultdict; 
	// //  std::cout << taskfile << std::endl;
	// //  std::cout << num << std::endl;

	Mission task = Mission(taskfile, num, STEP_MAX, IS_TIME_BOUNDED, TIME_MAX, STOP_BY_SPEED);
	if (task.ReadTask()) {
		auto summary = task.StartMission();
		auto full_summary = summary.getFullSummary();
		std::vector<std::string> keys, values;
		for(auto it = full_summary.begin(); it != full_summary.end(); ++it) {
			keys.push_back(it->first);
			values.push_back(it->second);
		}

		// for (auto &key : keys) {
		// 	//  std::cout << std::right << std::setfill(' ') << std::setw(15) << key  << ' ';
		// }
		// //  std::cout << std::endl;
		// for (auto &value : values) {
		// 	//  std::cout << std::right << std::setfill(' ') << std::setw(15) << value  << ' ';
		// }
		// //  std::cout << std::endl;
	
	
// #if FULL_LOG
		// //  std::cout << "Saving log\n";
		// task.SaveLog();
		std::unordered_map<int, trajectory_dict> resultdict = task.SaveLog_dict();
		// for (const auto& entry: resultdict){
		// 	int key = entry.first;
		// 	cout<< "key: " << key<<endl;
		// 	const auto& value = entry.second;
		// 	cout<<" xr: ";
		// 	for (float val : value.xr){ cout<<val<< " ";}
		// 	cout<<endl;
			
		// 	cout<<" yr: ";
		// 	for (float val : value.yr){ cout<<val<< " ";}
		// 	cout<<endl;

		// 	cout<<" nextxr: ";
		// 	for (float val : value.nextxr){ cout<<val<< " ";}
		// 	cout<<endl;

		// 	cout<<" nextyr: ";
		// 	for (float val : value.nextyr){ cout<<val<< " ";}
		// 	cout<<endl;

		// 	cout<<"foundpath: "<<value.foundpath<<endl;
		// 	cout<<"total_step: "<<value.total_step<<endl;
		// }
		return resultdict;
// #endif


	// parse xml

	}
	else {
		//  std::cout << "Error during task execution\n";
		// return -1;
		return resultdict;
	}
	return resultdict;
	
	// return 0;
}



PYBIND11_MODULE(bind, m)
{
    // 可选，说明这个模块的作用
    m.doc() = "pybind11 test plugin";
    //def("提供给python调用的方法名"， &实际操作的函数， "函数功能说明"， 默认参数). 其中函数功能说明为可选
    m.def("demo", &demo, "A function which multiplies two numbers", py::arg("taskfile")="tmp", py::arg("num")=7);
	py::class_<trajectory_dict>(m, "trajectory_dict")
		.def(py::init<>())
		.def_readwrite("xr", &trajectory_dict::xr)
		.def_readwrite("yr", &trajectory_dict::yr)
		.def_readwrite("nextxr", &trajectory_dict::nextxr)
		.def_readwrite("nextyr", &trajectory_dict::nextyr)
		.def_readwrite("total_step", &trajectory_dict::total_step)
		.def_readwrite("foundpath", &trajectory_dict::foundpath)
		;

	// py::class_<Mission>(m, "Mission")
    //     .def(py::init<std::string, unsigned int, unsigned int, bool, size_t, bool>())
	// 	;		

}
