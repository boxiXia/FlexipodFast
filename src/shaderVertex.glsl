#version 460 core
// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in vec3 vertexNormal;

// Output data ; will be interpolated for each fragment.
out vec3 fragmentColor;
// out vec3 fragPos;// fragment position

// Values that stay constant for the whole mesh.
uniform mat4 MVP;
uniform vec3 viewPos;//view position
uniform vec3 lightDir;//light direction,pointing towards light
uniform vec3 lightColor;//light color

void main() {
    // Output position of the vertex, in clip space : MVP * position
    gl_Position = MVP * vec4(vertexPosition_modelspace, 1);
    // The color of each vertex will be interpolated
    // to produce the color of each fragment
    if(vertexNormal.x!=0 && vertexNormal.y!=0 && vertexNormal.z!=0){
        vec3 normal = normalize(vertexNormal);
        // diffuse
        float diffuse = max(dot(normal,lightDir),0.0);
        // ambient
        float ambient = 0.5;
        // specular
        vec3 viewDir = normalize(viewPos - vertexPosition_modelspace);
        vec3 reflectDir = reflect(-lightDir, normal); 
        float specular = pow(max(dot(viewDir, reflectDir), 0.0), 32);

        // float ratio = clamp(diffuse+ambient+specular,0,1);
        float ratio = diffuse+ambient+specular;

        fragmentColor = lightColor*vertexColor*ratio;
    }else{
        fragmentColor = lightColor*vertexColor;
    }
}