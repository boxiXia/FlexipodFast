#include "sim.h"


Model::Model(const std::string& file_path, bool versbose) {
	// get the msgpack robot model
	// Deserialize the serialized data
	std::ifstream ifs(file_path, std::ifstream::in | std::ifstream::binary);
	std::stringstream buffer;
	buffer << ifs.rdbuf();
	msgpack::unpacked upd;//unpacked data
	msgpack::unpack(upd, buffer.str().data(), buffer.str().size());
	//    std::cout << upd.get() << std::endl;
	*this = (upd.get().as<Model>());
	if (versbose) {
		printf("Loaded %s\n", file_path.c_str());
		printf("radius_poisson=%.3e [m] \n", radius_poisson);
		printf("#vertices=%d, #edges=%d, #triangles = %d, #joints = %d\n",
			(int)vertices.size(), (int)edges.size(), (int)triangles.size(), (int)joints.size());

	}
}