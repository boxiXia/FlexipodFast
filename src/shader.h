/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, “Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,” ICRA 2020, May 2020.
*/


#ifndef shader_hpp
#define shader_hpp

#include <GL/glew.h>// Include GLEW
#include <GLFW/glfw3.h>// Include GLFW
#include <glm/glm.hpp>// Include GLM
#include <glm/gtx/norm.hpp>
#include<glm/gtx/rotate_vector.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>

// vertex gl buffer: x,y,z,r,g,b,nx,ny,nz
struct VERTEX_DATA {
    glm::vec3 pos; // 0: vertex position
    glm::vec3 color; // 1: vertex color
    glm::vec3 normal; //3: vertex normal
};

// spherical linear interpolation
glm::vec3 slerp(glm::vec3 p0, glm::vec3 p1, float t);


// renderQuad() renders a 1x1 XY quad in NDC
class NdcQuad {
public:
    unsigned int quadVAO = 0;
    unsigned int quadVBO;
    NdcQuad() {
        float quadVertices[] = {
            // positions        // texture Coords
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };
        // setup plane VAO
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    void render() {
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
    }
};

/* directional light*/
class DirectionLight { 
public:
	glm::vec3 direction = glm::vec3(0.f, 0.f, 1.f);// light direction, pointing towards light
	glm::vec3 color = glm::vec3(1.f, 1.f, 1.f); // light color
	float ambient = 0.4f; // ambient intensity
	float diffuse = 0.6f; // diffuse intensity
	float specular = 0.8f; // specular intensity
	
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



/* camera */
class Camera {
public:
    // for projection matrix 
    glm::vec3 pos;// camera position
    glm::vec3 dir;//camera look at direction (front)
    glm::vec3 up;// camera up vector
    
    float yaw = 0;  // rotation angle of the vector from target to camera about camera_up vector
    float h_offset = 1.f; // distance b/w target and camera in plane normal to camera_up vector 
    float up_offset = 1.f; // distance b/w target and camera in camera_up direction

    Camera() {};
    Camera(
        glm::vec3 camera_position,
        glm::vec3 target_position,
        glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f)) {
        pos = camera_position;
        dir = glm::normalize(target_position - camera_position);
        up = camera_up;
    }
    // returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 getViewMatrix()
    {   // ref: https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml
        return glm::lookAt(pos/*camera position*/,pos + dir/*look at position*/,up);
    }

    void follow(const glm::vec3& target, const float interp_factor);

    /*process keyboard*/
    void processKeyboard(int key,float vel=0.05) {
        if (key == GLFW_KEY_W) { h_offset -= vel; }//camera moves closer
        else if (key == GLFW_KEY_S) { h_offset += vel; }//camera moves away
        else if (key == GLFW_KEY_A) { yaw -= vel; } //camera moves left
        else if (key == GLFW_KEY_D) { yaw += vel; }//camera moves right
        else if (key == GLFW_KEY_Q) { up_offset -= vel; } // camera moves down
        else if (key == GLFW_KEY_E) { up_offset += vel; }// camera moves up
    }

private:
    glm::vec3 getHorizontalDirection();
};

/* 
for loading shaders and setup uniforms
ref: https://github.com/JoeyDeVries/LearnOpenGL/blob/master/includes/learnopengl/shader.h
*/
class Shader
{
public:
    unsigned int ID;
    // constructor generates the shader on the fly
    // ------------------------------------------------------------------------
    Shader() {};
    Shader(const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath = "")
    {
        // 1. retrieve the vertex/fragment source code from filePath
        std::string vertexCode;
        std::string fragmentCode;
        std::string geometryCode;
        std::ifstream vShaderFile;
        std::ifstream fShaderFile;
        std::ifstream gShaderFile;
        // ensure ifstream objects can throw exceptions:
        vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        try
        {
            // open files
            vShaderFile.open(vertexPath);
            fShaderFile.open(fragmentPath);
            std::stringstream vShaderStream, fShaderStream;
            // read file's buffer contents into streams
            vShaderStream << vShaderFile.rdbuf();
            fShaderStream << fShaderFile.rdbuf();
            // close file handlers
            vShaderFile.close();
            fShaderFile.close();
            // convert stream into string
            vertexCode = vShaderStream.str();
            fragmentCode = fShaderStream.str();
            // if geometry shader path is present, also load a geometry shader
            if (!geometryPath.empty())
            {
                gShaderFile.open(geometryPath);
                std::stringstream gShaderStream;
                gShaderStream << gShaderFile.rdbuf();
                gShaderFile.close();
                geometryCode = gShaderStream.str();
            }
        }
        catch (std::ifstream::failure& e)
        {
            
            std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ:"<<e.what() << std::endl;
        }
        const char* vShaderCode = vertexCode.c_str();
        const char* fShaderCode = fragmentCode.c_str();
        // 2. compile shaders
        unsigned int vertex, fragment;
        // vertex shader
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, NULL);
        glCompileShader(vertex);
        checkCompileErrors(vertex, "VERTEX");
        // fragment Shader
        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fShaderCode, NULL);
        glCompileShader(fragment);
        checkCompileErrors(fragment, "FRAGMENT");
        // if geometry shader is given, compile geometry shader
        unsigned int geometry;
        if (!geometryPath.empty())
        {
            const char* gShaderCode = geometryCode.c_str();
            geometry = glCreateShader(GL_GEOMETRY_SHADER);
            glShaderSource(geometry, 1, &gShaderCode, NULL);
            glCompileShader(geometry);
            checkCompileErrors(geometry, "GEOMETRY");
        }
        // shader Program
        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        if (!geometryPath.empty())
            glAttachShader(ID, geometry);
        glLinkProgram(ID);
        checkCompileErrors(ID, "PROGRAM");
        // delete the shaders as they're linked into our program now and no longer necessery
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        if (!geometryPath.empty())
            glDeleteShader(geometry);

    }
    // activate the shader
    // ------------------------------------------------------------------------
    void use()
    {
        glUseProgram(ID);
    }
    // utility uniform functions
    // ------------------------------------------------------------------------
    void setBool(const std::string& name, bool value) const
    {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
    }
    // ------------------------------------------------------------------------
    void setInt(const std::string& name, int value) const
    {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
    }
    // ------------------------------------------------------------------------
    void setFloat(const std::string& name, float value) const
    {
        glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
    }
    // ------------------------------------------------------------------------
    void setVec2(const std::string& name, const glm::vec2& value) const
    {
        glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
    }
    void setVec2(const std::string& name, float x, float y) const
    {
        glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y);
    }
    // ------------------------------------------------------------------------
    void setVec3(const std::string& name, const glm::vec3& value) const
    {
        glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
    }
    void setVec3(const std::string& name, float x, float y, float z) const
    {
        glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z);
    }
    // ------------------------------------------------------------------------
    void setVec4(const std::string& name, const glm::vec4& value) const
    {
        glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
    }
    void setVec4(const std::string& name, float x, float y, float z, float w)
    {
        glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
    }
    // ------------------------------------------------------------------------
    void setMat2(const std::string& name, const glm::mat2& mat) const
    {
        glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }
    // ------------------------------------------------------------------------
    void setMat3(const std::string& name, const glm::mat3& mat) const
    {
        glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }
    // ------------------------------------------------------------------------
    void setMat4(const std::string& name, const glm::mat4& mat) const
    {
        glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }

private:
    // utility function for checking shader compilation/linking errors.
    // ------------------------------------------------------------------------
    void checkCompileErrors(GLuint shader, std::string type)
    {
        GLint success;
        GLchar infoLog[1024];
        if (type != "PROGRAM")
        {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success)
            {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        else
        {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success)
            {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
    }
};


#endif /* shader_hpp */
