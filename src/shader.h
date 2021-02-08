/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, “Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,” ICRA 2020, May 2020.
*/


#ifndef shader_hpp
#define shader_hpp



#include <GL/glew.h>// Include GLEW
#include <GLFW/glfw3.h>// Include GLFW
#include <glm/glm.hpp>// Include GLM

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

GLuint LoadShaders(const char* vertex_file_path, const char* fragment_file_path);

/* directional light*/
class DirectionLight { 
public:
	glm::vec3 direction = glm::vec3(0.f, 0.f, -1.f);// light direction, pointing from light
	glm::vec3 color = glm::vec3(1.f, 1.f, 1.f); // light color
	float ambient = 0.4f; // ambient intensity
	float diffuse = 0.6f; // diffuse intensity
	float specular = 1.f; // specular intensity
	
	/*associate with the shader given shade id and the varibales's base name*/
	void setName(GLuint& shaderID, const std::string& name) {
		id_direction = glGetUniformLocation(shaderID, (name + ".direction").c_str());
		id_color = glGetUniformLocation(shaderID, (name + ".color").c_str());
		id_ambient = glGetUniformLocation(shaderID, (name + ".ambient").c_str());
		id_diffuse = glGetUniformLocation(shaderID, (name + ".diffuse").c_str());
		id_specular = glGetUniformLocation(shaderID, (name + ".specular").c_str());
	}
	/*should be called after setName*/
	void setValues() {
		glUniform3fv(id_direction, 1, &direction[0]);
		glUniform3fv(id_color, 1, &color[0]);
		glUniform1f(id_ambient, ambient);
		glUniform1f(id_diffuse, diffuse);
		glUniform1f(id_specular, specular);
	}
	/*set values to shader given shade id and the varibales's base name*/
	void set(GLuint& shaderID, const std::string& name) {
		setName(shaderID, name);
		setValues();
	}
private:
	GLuint id_direction;
	GLuint id_color;
	GLuint id_ambient;
	GLuint id_diffuse;
	GLuint id_specular;
};


#endif /* shader_hpp */
