#version 460 core
// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition;// at model space
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in vec3 vertexNormal;

// Output data ; will be interpolated for each fragment.
out vec3 fragColor;  // fragment color
out vec3 fragPos;    // fragment position
out vec3 fragNormal; // fragment normal
out vec4 fragPosLightSpace; //fragment position in light space

// Values that stay constant for the whole mesh.
uniform mat4 MVP;       // model view projection matrix
uniform mat4 lightSpaceMatrix; // light space matrix

void main() {
    // REF: https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/2.lighting/2.2.basic_lighting_specular/2.2.basic_lighting.vs
    // https://learnopengl.com/Lighting/Basic-Lighting
    fragPos = vertexPosition;// assume M=identity
    fragColor = vertexColor;
    fragNormal = vertexNormal;

    // Output position of the vertex, in clip space : MVP * position
    gl_Position = MVP * vec4(vertexPosition, 1);

    fragPosLightSpace = lightSpaceMatrix * vec4(fragPos,1.0);
}