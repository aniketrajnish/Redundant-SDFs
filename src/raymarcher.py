import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import numpy as np

class Raymarcher:
    def __init__(self):
        pass
        # self.vertex_shader = """
        # #version 330
        # in vec2 position;
        # void main()
        # {
        #     gl_Position = vec4(position, 0.0, 1.0);
        # }
        # """

        # with open('src/sdfs.glsl', 'r') as f:
        #     sdf_functions = f.read()

        # self.fragment_shader = f"""
        # #version 330
        # out vec4 fragColor;
        # uniform vec2 resolution;
        # uniform float time;
        # uniform int selectedShape;

        # {sdf_functions}

        # float sceneSDF(vec3 p) {{
        #     float d = 1e10;
        #     switch (selectedShape) {{
        #         case 0: d = sdSphere(p, 1.0); break;
        #         case 1: d = sdBox(p, vec3(0.8)); break;
        #         case 2: d = sdTorus(p, vec2(0.8, 0.2)); break;
        #         case 3: d = sdCylinder(p, vec3(0.0, 0.0, 0.5)); break;
        #         case 4: d = sdCone(p, vec2(0.8, 1.0), 1.5); break;
        #     }}
        #     return d;
        # }}

        # vec3 estimateNormal(vec3 p) {{
        #     float eps = 0.001;
        #     return normalize(vec3(
        #         sceneSDF(p + vec3(eps, 0.0, 0.0)) - sceneSDF(p - vec3(eps, 0.0, 0.0)),
        #         sceneSDF(p + vec3(0.0, eps, 0.0)) - sceneSDF(p - vec3(0.0, eps, 0.0)),
        #         sceneSDF(p + vec3(0.0, 0.0, eps)) - sceneSDF(p - vec3(0.0, 0.0, eps))
        #     ));
        # }}

        # vec3 rayMarch(vec3 ro, vec3 rd) {{
        #     float t = 0.0;
        #     for(int i = 0; i < 100; i++) {{
        #         vec3 p = ro + rd * t;
        #         float d = sceneSDF(p);
        #         if(d < 0.001) {{
        #             vec3 n = estimateNormal(p);
        #             return 0.5 + 0.5 * n;
        #         }}
        #         t += d;
        #         if(t > 100.0) break;
        #     }}
        #     return vec3(0.0);
        # }}

        # void main() {{
        #     vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / resolution.y;
        #     vec3 ro = vec3(0.0, 0.0, -3.0);
        #     vec3 rd = normalize(vec3(uv, 1.0));
            
        #     vec3 color = rayMarch(ro, rd);
            
        #     fragColor = vec4(color, 1.0);
        # }}
        # """

        # self.shader = shaders.compileProgram(
        #     shaders.compileShader(self.vertex_shader, gl.GL_VERTEX_SHADER),
        #     shaders.compileShader(self.fragment_shader, gl.GL_FRAGMENT_SHADER)
        # )

        # self.vao = gl.glGenVertexArrays(1)
        # gl.glBindVertexArray(self.vao)

        # self.vbo = gl.glGenBuffers(1)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        # vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        # gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        # position = gl.glGetAttribLocation(self.shader, "position")
        # gl.glEnableVertexAttribArray(position)
        # gl.glVertexAttribPointer(position, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # self.resolution_loc = gl.glGetUniformLocation(self.shader, "resolution")
        # self.time_loc = gl.glGetUniformLocation(self.shader, "time")
        # self.selected_shape_loc = gl.glGetUniformLocation(self.shader, "selectedShape")

    def render(self, width, height, time, selected_shape):
        pass
        # gl.glUseProgram(self.shader)
        # gl.glUniform2f(self.resolution_loc, width, height)
        # gl.glUniform1f(self.time_loc, time)
        # gl.glUniform1i(self.selected_shape_loc, selected_shape)

        # gl.glBindVertexArray(self.vao)
        # gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)