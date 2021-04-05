#include<iostream>
#include <sstream>
#include<vector>
#include <omp.h>
#include <chrono> // for time measurement

//#include <iomanip>
//#include <nlohmann/json.hpp>
//using json = nlohmann::json;


#include<testjsonHeader.h>

//#ifndef __CUDACC__
//#include "simdjson.h"
//#endif

int main(void) {


//#ifndef __CUDACC__
//    simdjson::ondemand::parser parser;
//    simdjson::padded_string json = simdjson::padded_string::load("../twitter.json");
//    simdjson::ondemand::document tweets = parser.iterate(json);
//    std::cout << uint64_t(tweets["search_metadata"]["count"]) << " results." << std::endl;
//#endif

//int i;
//	int numthreads = 8;
//#pragma omp parallel for default(none) num_threads(numthreads) private(i)
//	for (i = 0; i < 100; i++)
//	{
//		int tid = omp_get_thread_num();
//		printf("omp thread %d\n", tid);
//	}

    //parseJson();

	// for time measurement
	auto start = std::chrono::steady_clock::now();


    testClass tk;

    tk.test();

    
    //parseJson();

    Model bot(getProgramDir() + "\\..\\src\\flexipod_12dof.msgpack"); //defined in sim.h

    bot.colors;
    auto end = std::chrono::steady_clock::now();
    printf("main():Elapsed time:%d ms \n",
        (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());


}